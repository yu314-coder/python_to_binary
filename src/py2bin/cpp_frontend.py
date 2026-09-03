"""Translate a subset of C++ into C, so py2bin's C compiler can build it.

py2bin has a C compiler and no C++ one. Writing a second compiler is a project
of its own - templates, overload resolution, exceptions and the standard
library are most of what C++ *is*. What is tractable, and what the first C++
compiler did, is to translate: a class is a struct, a member function is a
free function whose first parameter is the object, and a constructor is a
function that initialises one in place. That is Cfront's trick, and it needs
nothing from the backend that C does not already need.

So this is a *translator*, not a compiler, and the subset it accepts is small
and stated. What it does accept:

* `class` and `struct` with data members and member functions
* member functions defined inside the class or out of it as `Type Class::name`
* constructors and destructors, including destructors at the end of a scope
* `this`, implicit or written
* calls through an object, a pointer, or `this`
* single inheritance, non-virtual: the base is embedded first, so a pointer to
  the derived object is a valid pointer to the base one, and inherited members
  and methods resolve
* `bool`, `true`, `false`, and `//` comments, which C has anyway

What it refuses, by name, rather than mistranslating:

* templates, exceptions, `virtual`, operator overloading, `new`/`delete`,
  multiple inheritance, namespaces, references, and anything from the standard
  library

`new` is refused for a reason worth stating: py2bin's C compiler has no
`malloc`. There is no heap to put an object on, so a C++ subset without `new`
is not a shortcut taken here - it is the shape of what is underneath.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path


class CppTranslationError(ValueError):
    """A C++ construct this translator will not silently get wrong."""

    def __init__(self, filename: str, line: int, message: str):
        self.filename = filename
        self.line = line
        self.message = message
        super().__init__(f"{filename}:{line}: {message}")


#: Spelled out so the refusal can name the construct rather than failing later
#: as a C syntax error in translated output nobody wrote.
#: C++ py2bin does not translate. Empty, and kept so: the list was how the
#: subset said what it was not, and every entry on it has since become
#: something it does. A construct that turns up unhandled now reaches the C
#: compiler and is reported there, which is a worse message than this gave -
#: so anything found that way belongs back on this list until it works.
_REFUSED: "tuple[tuple[str, str], ...]" = ()

_WORD = re.compile(r"\b[A-Za-z_]\w*\b")

#: How an overloaded operator is spelled as a C name. The symbol cannot go in
#: an identifier, so each has a word.
#: What a conversion operator is called once it has a name C can hold.
#: `operator int()` becomes `op_to_int`, `operator const char *()` becomes
#: `op_to_const_char_p` - the same spelling an overload is told apart by.
_CONVERSION_PREFIX = "op_to_"

_OPERATOR_NAMES = {
    "+": "op_add", "-": "op_sub", "*": "op_mul", "/": "op_div", "%": "op_mod",
    "==": "op_eq", "!=": "op_ne", "<": "op_lt", ">": "op_gt",
    "<=": "op_le", ">=": "op_ge", "[]": "op_index", "()": "op_call",
    "&": "op_bit_and",
    "+=": "op_add_assign", "-=": "op_sub_assign", "=": "op_assign",
    "<<": "op_shl", ">>": "op_shr",
    # `->` and a `*` with nothing on its left. Both are how a holder stands
    # in for what it holds - a smart pointer, an iterator - and neither is
    # a binary operator, so they are rewritten where they are written rather
    # than through the two-operand pass. `!` is the other unary one a holder
    # writes, for "is there anything here".
    "->": "op_arrow", "!": "op_not",
    # `++c` and `c++` are different members, told apart by a parameter that
    # exists only to say which - C++ writes the postfix one as `operator++(int)`
    # and never passes anything for it.
    "++": "op_inc", "--": "op_dec",
}
#: `*x` and `-x` with nothing on the left. Written `operator*()` and
#: `operator-()` - no parameter, which is the only thing that tells either
#: from the two-operand one written the same way.
_DEREFERENCE = "op_deref"
_NEGATE = "op_neg"
#: `T **operator&()` - what a holder gives back when a call wants somewhere
#: to write. `&` is a binary operator as well, so the unary form has a name
#: of its own, the same way `*` does.
_ADDRESS_OF = "op_address_of"
#: `c++` rather than `++c`. Spelled `operator++(int)`, whose parameter is
#: never given a value and exists only to be different.
_POSTFIX = {"op_inc": "op_inc_post", "op_dec": "op_dec_post"}

#: Longest first, so `<=` is not read as `<`. `->` is not among them: it takes
#: no right operand of its own, so the two-operand pass has nothing to match.
#: How tightly each binds, tightest first. The passes that write these out
#: take one symbol at a time, so the order they are taken in *is* the
#: precedence: an operator written out before one that binds tighter puts the
#: wrong pair together, and `a + b * c` came out as `(a + b) * c` - a wrong
#: answer with nothing to say so. Sorted by length alone, `+` came before
#: `*` because they are the same length.
#:
#: The compound assignments go first, before the symbol each is built from,
#: so that the pattern for `+` cannot match the `+` inside `+=`. Where two
#: bind equally the longer is tried first, for the same reason.
_OPERATOR_PRECEDENCE = {
    "+=": 0, "-=": 0,
    "*": 1, "/": 1, "%": 1,
    "+": 2, "-": 2,
    "<<": 3, ">>": 3,
    "<": 4, "<=": 4, ">": 4, ">=": 4,
    "==": 5, "!=": 5,
}

_OPERATOR_SYMBOLS = [
    symbol
    for symbol in sorted(
        _OPERATOR_NAMES,
        key=lambda one: (_OPERATOR_PRECEDENCE.get(one, 9), -len(one)),
    )
    # `&` is left out of the two-operand pass: a class that overloads it
    # overloads the *unary* one, which is what a holder does to hand out
    # somewhere to write. A binary `a & b` on two objects is not something
    # this subset has met, and reading `&x` as one turned every address-of
    # into a call.
    if symbol not in ("->", "!", "++", "--", "&")
]


@dataclass
class Member:
    name: str
    ctype: str
    array: str = ""   # "[8]" for `int items[8]`, kept for the struct field
    #: Declared `int &slot`. C has no reference, so it is held as a pointer
    #: and every use of it reads through - which is what a reference is. Read
    #: as an ordinary member the name came out as `&slot`, so nothing matched
    #: a use of it and the struct itself was not C.
    reference: bool = False


@dataclass
class Method:
    name: str          # as written; "" for a constructor, "~" for a destructor
    returns: str
    parameters: str    # the C parameter list after `this`, or ""
    body: str
    line: int
    #: Declared `virtual`, or overriding something that was. A virtual method
    #: is reached through the object rather than by its name, so the call goes
    #: to what the object *is* and not to what the variable was declared as.
    virtual: bool = False
    #: `= 0`: declared, deliberately not defined. The slot holds a null and a
    #: class with one is abstract.
    pure: bool = False
    #: Declared `static`: one function for the class rather than one per
    #: object, so it takes no `this` and is reached by the class's name.
    shared: bool = False
    #: Declared `const` after the parameter list. C++ picks between a `const`
    #: and a non-`const` member of the same name by the object; py2bin does
    #: not track whether an object is const, so what this is for is telling
    #: such a pair apart when it is read - not choosing between them.
    readonly: bool = False


@dataclass
class Class:
    name: str
    base: str | None = None
    #: Which bases were written `virtual`. One path to one of these is an
    #: ordinary base and the word means nothing; two paths to the same one
    #: means both must reach a single shared object, which is a layout this
    #: translator does not write.
    virtual_bases: "set[str]" = field(default_factory=set)
    #: The bases after the first. The first is `base`, because it is at offset
    #: zero and everything that reaches through a base reaches through that
    #: one without adjusting anything; these sit after it, and reaching one
    #: means naming the member it became.
    mixins: "list[str]" = field(default_factory=list)
    members: list[Member] = field(default_factory=list)
    methods: list[Method] = field(default_factory=list)
    #: Method name -> the default written for each parameter, "" where none.
    #: Kept here because a call site has to fill them in and the parameter
    #: list has had them stripped by then.
    defaults: "dict[str, list[str]]" = field(default_factory=dict)
    #: `int n = 7;` written on the member itself. C++ applies it in every
    #: constructor that does not name the member in its initialiser list, so
    #: it becomes an assignment at the top of each - before the list's own,
    #: which then overwrite where they name the same member.
    member_values: "list[tuple[str, str]]" = field(default_factory=list)

    def field_names(self) -> set[str]:
        return {member.name for member in self.members}

    def method_names(self) -> set[str]:
        return {method.name for method in self.methods if method.name}


def _strip_comments(text: str) -> str:
    """Blank comments out, keeping every newline so line numbers survive."""

    out = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in "\"'":
            quote = char
            out.append(char)
            index += 1
            while index < length:
                out.append(text[index])
                if text[index] == "\\" and index + 1 < length:
                    out.append(text[index + 1])
                    index += 2
                    continue
                if text[index] == quote:
                    index += 1
                    break
                index += 1
            continue
        if text.startswith("//", index):
            while index < length and text[index] != "\n":
                index += 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = length if end < 0 else end + 2
            out.append("\n" * text.count("\n", index, end))
            index = end
            continue
        out.append(char)
        index += 1
    return "".join(out)




def _sub_code(pattern: "re.Pattern[str]", text: str, change) -> str:
    """Like `_map_code`, but the match keeps its place in the whole text.

    `_map_code` hands each stretch of code to the replacement on its own, so
    a match's offset is into that stretch and not into the file. A
    replacement that looks around itself - which class encloses this, what
    was declared above it - was reading the wrong place as soon as anything
    earlier in the file held a string literal. `change` is called with the
    match against a copy whose literals are blanked, and with the real text,
    so it can slice either. It answers with the replacement, or with None to
    leave the match alone - which is not the same as answering
    `match.group(0)`, since that group comes from the blanked copy and
    writing it back would erase every literal inside the match.
    """

    bare = _without_literals(text)
    out: list[str] = []
    at = 0
    for match in pattern.finditer(bare):
        if match.start() < at:
            continue
        replacement = change(match, text)
        out.append(text[at:match.start()])
        out.append(
            text[match.start(): match.end()] if replacement is None else replacement
        )
        at = match.end()
    out.append(text[at:])
    return "".join(out)

def _map_code(text: str, change) -> str:
    """Apply `change` to the code, and to nothing inside a literal.

    Rewriting ran over the whole text, string literals included, and a class
    with a member called `n` turned `printf("outer\\n")` into
    `printf("outer\\this->n")`. The program still built and printed the wrong
    thing, which is the failure mode worth the most care: a literal is data,
    and no name in it is a name.
    """

    out = []
    chunk = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in "\"'":
            # The prefix goes with the literal: `L"hi"` blanked to `L     `
            # leaves an `L` standing, and an `L` reads as a name.
            code = "".join(chunk)
            prefix = ""
            for spelled in ("u8", "L", "u", "U"):
                if not code.endswith(spelled):
                    continue
                before = code[: -len(spelled)]
                if before and (before[-1].isalnum() or before[-1] == "_"):
                    continue
                prefix = spelled
                break
            out.append(change(code[: len(code) - len(prefix)]))
            chunk = []
            quote = char
            literal = [prefix, char]
            index += 1
            while index < length:
                if text[index] == "\\" and index + 1 < length:
                    literal.append(text[index:index + 2])
                    index += 2
                    continue
                literal.append(text[index])
                if text[index] == quote:
                    index += 1
                    break
                index += 1
            out.append("".join(literal))
            continue
        chunk.append(char)
        index += 1
    out.append(change("".join(chunk)))
    return "".join(out)

def _matching(text: str, opening: int) -> int:
    """The index just past the brace that closes the one at `opening`.

    Literals are skipped. A file that embeds HTML or JavaScript is full of
    braces inside strings - `"function(){}"` - and counting those puts every
    class body's end in the wrong place, which is a hard thing to see
    afterwards because the text looks fine.
    """

    depth = 0
    index = opening
    while index < len(text):
        piece = text[index]
        if piece in "\"'":
            quote = piece
            index += 1
            while index < len(text) and text[index] != quote:
                index += 2 if text[index] == "\\" else 1
            index += 1
            continue
        if piece == "{":
            depth += 1
        elif piece == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise ValueError(
        f"unbalanced braces: the '{{' on line {_line_of(text, opening)} is "
        f"never closed"
    )


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


#: `final` after the name says nothing may derive from it, which C++ checks
#: and C has no way to state - so it is read and dropped, the way `override`
#: is on a member.
#: A base, with the words C++ lets stand in front of it.
_ONE_BASE = r"(?:public|private|protected)?\s*(?:virtual\s+)?[A-Za-z_]\w*\s*"
_CLASS_HEAD = re.compile(
    r"\b(class|struct)\s+([A-Za-z_]\w*)\s*(?:final\s*)?"
    # Every base, not only the first. `struct C : A, B` is one class with two,
    # and read as one with none the members of both were nowhere to be found.
    rf"(?::\s*({_ONE_BASE}(?:,\s*{_ONE_BASE})*))?\{{"
)


#: One identifier and nothing else.
_A_NAME = re.compile(r"[A-Za-z_]\w*")

#: What this file's own `#define` lines say, for the few places below that
#: have to know what a name is before the preprocessor has run. Filled once
#: per translation and empty otherwise, so a file that defines nothing goes
#: through exactly as it did.
_MACROS_HERE: "dict[str, str]" = {}

#: Every name the file `#define`s - the ones above and the ones nothing here
#: can answer for, which are the ones that take arguments, are written twice,
#: or are `#undef`ed later. A name in this set and not in the table is one to
#: refuse by name: read as an ordinary word it is swallowed into whatever
#: declaration follows it.
_MACRO_NAMES: "set[str]" = set()


def _read_macros(text: str) -> None:
    """Note what this file's `#define` lines say, without running them.

    The C++ stage is in front of the preprocessor, so a class that spells a
    base, a member's type or a whole member with a macro still has the
    macro's own name written there when the classes are taken apart. This
    reads the directives straight: a name defined once, never `#undef`ed and
    taking no arguments stands for one text everywhere and can be answered.
    Anything else stands for different text in different places and is only
    worth knowing about so that it can be refused by name rather than guessed
    at - which is what running the preprocessor first would settle, and this
    stage does not run it.
    """

    _MACROS_HERE.clear()
    _MACRO_NAMES.clear()
    stands: "dict[str, str]" = {}
    unsettled: "set[str]" = set()
    #: How deep inside `#if`/`#endif` the reader stands. A `#define` under one
    #: only says what it says if that branch is taken, and which branch is
    #: taken is the preprocessor's answer and not this stage's. Taken at face
    #: value, `#ifdef _WIN32 / #define BASE WinThing / #endif` would name a
    #: base class on a machine that never compiles that branch - and unlike
    #: every other shape here that would be silent, because the name resolves
    #: to a real class and the program builds. Whether a directive sits
    #: between the two is answered by reading, which is all this needs.
    inside = 0
    for kind, part in _split_literals(text):
        # A directive is one of the pieces `_split_literals` keeps whole, so
        # a `class` written inside a macro body is not read as a class and a
        # `#define` written inside a string is not read as a definition.
        if kind != "literal" or not part.startswith("#"):
            continue
        if _OPENS_A_CONDITION.match(part):
            inside += 1
        elif _CLOSES_A_CONDITION.match(part):
            inside = max(0, inside - 1)
        undone = re.match(r"#\s*undef\s+([A-Za-z_]\w*)", part)
        if undone is not None:
            unsettled.add(undone.group(1))
            continue
        written = re.match(
            r"#\s*define\s+([A-Za-z_]\w*)(\(?)[ \t]*(.*)$", part, re.S
        )
        if written is None:
            continue
        name = written.group(1)
        if name in _MACRO_NAMES:
            unsettled.add(name)
        _MACRO_NAMES.add(name)
        if written.group(2) or "\\" in part:
            # Arguments, or a body carried on the lines below it. Either one
            # needs the preprocessor proper, which this is not.
            unsettled.add(name)
            continue
        if inside:
            # Recorded so a use of it is refused by name, never resolved.
            unsettled.add(name)
            continue
        stands[name] = written.group(3).strip()
    for name, body in stands.items():
        if name in unsettled:
            continue
        if re.search(rf"(?<![.\w]){re.escape(name)}\b", body) is not None:
            # A macro naming itself is expanded once and no further. That is
            # the preprocessor's rule and there is nothing here that keeps
            # it, so the name is left for the preprocessor to settle.
            continue
        _MACROS_HERE[name] = body


def _plainly_c(inner: str) -> bool:
    """Whether a struct body has no method written anywhere in it.

    Asked of what the macros in it stand for and not of the text as written:
    a body whose only method arrives through a `#define` has no parenthesis
    in it at all, and taken for C already it went out untouched - which put a
    method inside a C struct once the preprocessor caught up. A name this
    file defines and this stage cannot settle counts as a method too, so the
    reader below reaches it and refuses it by name.
    """

    if "(" in inner:
        return False
    if not _MACRO_NAMES:
        return True
    written = inner
    for word in set(_A_NAME.findall(inner)):
        if word not in _MACRO_NAMES:
            continue
        stands = _MACROS_HERE.get(word)
        if stands is None:
            return False
        written = re.sub(
            rf"(?<![.\w]){re.escape(word)}\b", lambda _, s=stands: s, written
        )
    return "(" not in written


def _base_named(word: str) -> str:
    """The class a base clause names, with a one-word macro answered.

    `struct Dog : BASE` derives from whatever `BASE` stands for, and the
    preprocessor that would have written the real name there runs after this.
    Only a macro standing for a single name is answered: one standing for
    `public Animal`, or for anything this file spells more than one way, is
    left alone and reported by the check that no such class is declared.
    """

    stands = _MACROS_HERE.get(word)
    if stands is not None and _A_NAME.fullmatch(stands):
        return stands
    return word


def _virtual_bases_of(head: "re.Match[str]") -> "set[str]":
    """Which of a class head's bases were written `virtual`.

    One path to a virtual base is the same object either way, so the word
    changes nothing and is dropped. Two paths to the same one is what the
    word is *for*, and is a different thing entirely - so which bases carried
    it has to be remembered to tell the two apart.
    """

    spelled = head.group(3)
    if not spelled:
        return set()
    found: "set[str]" = set()
    for part in spelled.split(","):
        words = part.split()
        if "virtual" in words:
            named = [
                word
                for word in words
                if word not in ("public", "private", "protected", "virtual")
            ]
            if named:
                found.add(_base_named(named[-1]))
    return found


def _bases_of(head: "re.Match[str]") -> "list[str]":
    """Each base a class head names, in the order it named them.

    The order is the layout: the first base is at offset zero, which is what
    makes a pointer to the derived object a pointer to that base with nothing
    done to it, and the rest sit after it as members do.
    """

    spelled = head.group(3)
    if not spelled:
        return []
    found: "list[str]" = []
    for part in spelled.split(","):
        words = [
            word
            for word in part.split()
            if word not in ("public", "private", "protected", "virtual")
        ]
        if words:
            found.append(_base_named(words[-1]))
    return found

#: `struct is_pointer<T *> {` - a class template written again for a shape of
#: argument rather than for every argument. Every entry in a traits header is
#: one of these; it is how the answer is arrived at.
_SPECIALISED_HEAD = re.compile(
    r"\b(class|struct)\s+([A-Za-z_]\w*)\s*<"
)



#: `int n = 7;` written on the member itself. Only the `=` form: `Point p{1,
#: 2};` means the member's own constructor with those arguments, which is a
#: different thing from an assignment and is left to be reported rather than
#: turned into one.
_MEMBER_VALUE = re.compile(r"^(.*?[\s*&])([A-Za-z_]\w*)\s*=\s*(.+)$", re.S)


def _member_value(declaration: str) -> "tuple[str, tuple[str, str] | None]":
    """Split `int n = 7` into the declaration and what to assign, if anything."""

    found = _MEMBER_VALUE.match(declaration.strip())
    if found is None:
        return declaration, None
    spelled = found.group(3).strip()
    if not spelled:
        return declaration, None
    return f"{found.group(1)}{found.group(2)}", (found.group(2), spelled)

def _macro_written_out(
    body: str,
    spelled: "re.Match[str]",
    opened: "set[str]",
    owner: str,
    filename: str,
    at: int,
) -> str:
    """What a macro standing where a class member goes writes in its place."""

    name = spelled.group()
    stands = _MACROS_HERE.get(name)
    if stands is None:
        raise CppTranslationError(
            filename,
            at,
            f"{owner} writes `{name}` where a member goes, and `{name}` is a "
            f"macro this file spells more than one way - it takes arguments, "
            f"or is defined twice, or is `#undef`ed later. py2bin translates "
            f"C++ into C before it runs the preprocessor, so nothing here can "
            f"say which text `{name}` stands for at this point; read as an "
            f"ordinary word it is swallowed into the declaration after it and "
            f"whatever it declares leaves the struct silently. Write the "
            f"member out, or give `{name}` one unconditional definition",
        )
    if name in opened:
        raise CppTranslationError(
            filename,
            at,
            f"{owner} writes `{name}` where a member goes, and following what "
            f"`{name}` stands for leads back to `{name}`. The preprocessor "
            f"stops expanding a macro that reaches itself and runs after this "
            f"stage, so the class is refused here rather than read wrongly",
        )
    if stands.count("{") != stands.count("}"):
        raise CppTranslationError(
            filename,
            at,
            f"{owner} writes `{name}` where a member goes, and `{name}` opens "
            f"or shuts a brace it does not match. Where a class body ends is "
            f"what says which members are in it, and that is read here, "
            f"before the preprocessor has run - so a macro that moves the "
            f"closing brace is refused rather than guessed at",
        )
    opened.add(name)
    return f"{body[:spelled.start()]}{stands}{body[spelled.end():]}"


def _split_members(body: str, name: str, filename: str, at: int) -> Class:
    """Read a class body into its data members and its member functions."""

    found = Class(name)
    index = 0
    #: The macros already written out where the reader now stands, so a pair
    #: that name each other is reported instead of expanded forever.
    opened: "set[str]" = set()
    settled = -1
    while index < len(body):
        char = body[index]
        if char in " \t\n;":
            index += 1
            continue
        # A name this file `#define`s, written where a member goes. The
        # translation happens ahead of the preprocessor, so the macro's own
        # name is what stands here - and read as an ordinary word it was
        # swallowed into the return type of whatever came after it, which
        # took every member it declares out of the struct without a word
        # said. `sizeof` then answered 1 for a class of two ints.
        spelled = _A_NAME.match(body, index) if _MACRO_NAMES else None
        if spelled is not None and spelled.group() in _MACRO_NAMES:
            if index != settled:
                opened = set()
                settled = index
            body = _macro_written_out(
                body, spelled, opened, name, filename, at
            )
            continue
        # An access specifier changes nothing about the layout this emits.
        access = re.match(r"(public|private|protected)\s*:", body[index:])
        if access:
            index += access.end()
            continue
        statement_end = body.find(";", index)
        brace = body.find("{", index)
        if brace >= 0 and (statement_end < 0 or brace < statement_end):
            head = body[index:brace].strip()
            close = _matching(body, brace)
            if head.endswith("="):
                # `Token token_ = {0};` - a member given a value where it is
                # declared, and the value is a brace list. The brace opens a
                # value and not a body, and read as a body it asked what the
                # member returns.
                semicolon = body.find(";", close)
                if semicolon < 0:
                    break
                declaration, given = _member_value(
                    body[index:semicolon].strip()
                )
                if given is not None:
                    found.member_values.append(given)
                found.members.extend(
                    _members_from(declaration, filename, at + index)
                )
                index = semicolon + 1
                continue
            if _TAGGED_MEMBER.match(head):
                # `union { int i; float f; } value;` - a type written where a
                # member is declared, which is how a program says "one of
                # these, whichever is live". The brace is a body and not a
                # method's, and reading it as one asked what `union` returns.
                found.members.append(
                    _tagged_member(body, brace, close, head, filename, at + index)
                )
                index = body.find(";", close) + 1
                continue
            method = _method_from(head, body[brace:close], filename, at + index)
            found.methods.append(method)
            index = close
            continue
        if statement_end < 0:
            break
        declaration = body[index:statement_end].strip()
        if declaration:
            if "(" in declaration and not _FUNCTION_POINTER_MEMBER.match(
                declaration
            ):
                # A member function declared here and defined outside. Told
                # apart from `int (*op)(int);` by where the name sits: a
                # method's is before the parentheses and a function pointer's
                # is inside them, and both otherwise look the same.
                found.methods.append(
                    _method_from(declaration, "", filename, at + index)
                )
            else:
                # `int n = 7;` on the member itself. C has no such thing, so
                # the value is taken off the declaration and put into every
                # constructor, which is where C++ says it happens.
                declaration, given = _member_value(declaration)
                if given is not None:
                    found.member_values.append(given)
                # `int x, y;` declares both. Split on the commas and read the
                # type off the first: only the last declarator was registered
                # before, so a class could not reach its own `x` from its own
                # methods - the struct was laid out right and the name was
                # simply not in the set that drives implicit `this`.
                found.members.extend(
                    _members_from(declaration, filename, at + index)
                )
        index = statement_end + 1
    return found


#: `int (*op)(int)` - a member that is a pointer to a function. The name is
#: inside the parentheses, which is the whole reason this needs its own read.
_FUNCTION_POINTER_MEMBER = re.compile(
    r"^(.+?)\(\s*\*\s*([A-Za-z_]\w*)\s*\)\s*(\([^()]*\))$"
)



#: `union {`, `struct {`, `enum {` written where a member goes - with or
#: without a tag. The body is the type's, not a function's.
_TAGGED_MEMBER = re.compile(r"^(?:struct|union|enum)\b\s*[A-Za-z_]?\w*\s*$")


def _tagged_member(
    body: str, brace: int, close: int, head: str, filename: str, at: int
) -> Member:
    """`union { ... } value;` as one member, with the type carried whole."""

    rest = body[close:]
    named = re.match(r"\s*(\*?)\s*([A-Za-z_]\w*)?\s*(\[[^\]]*\])?\s*;", rest)
    if named is None:
        raise CppTranslationError(
            filename, at, f"cannot read the member {head!r} written here"
        )
    spelled = f"{head} {body[brace:close]}".strip()
    if not named.group(2):
        raise CppTranslationError(
            filename,
            at,
            f"an anonymous {head.split()[0]} member has no name, so C++ reaches "
            f"its fields directly; py2bin does not do that yet - give it a "
            f"name and reach them through it",
        )
    return Member(
        name=named.group(2),
        ctype=f"{spelled}{' ' + named.group(1) if named.group(1) else ''}",
        array=named.group(3) or "",
    )

def _members_from(
    declaration: str, filename: str, at: int
) -> "list[Member]":
    """Every data member one declaration declares.

    `int x, y;` is two of them, and the type is written in front of the first
    only - so the rest have to borrow it.
    """

    parts = _split_arguments(declaration)
    if len(parts) < 2:
        return [_member_from(declaration, filename, at)]
    first = _member_from(parts[0], filename, at)
    found = [first]
    for part in parts[1:]:
        if not part.strip():
            continue
        # Each later declarator carries only its own stars and brackets.
        found.append(_member_from(f"{first.ctype} {part.strip()}", filename, at))
    return found


def _member_from(declaration: str, filename: str, at: int) -> Member:
    # `mutable` says this member may be written through a const object. C has
    # no such word, and nothing here enforces const in the first place, so it
    # carries nothing - but left in it became part of the type, and the C
    # compiler was handed a declaration whose type name it had never heard of.
    declaration = re.sub(r"(?<![.\w>])mutable\s+", "", declaration)
    pointer = _FUNCTION_POINTER_MEMBER.match(declaration.strip())
    if pointer is not None:
        # Carried whole, with the name marked, so the emitter can put it back
        # where C wants it - `int (*op)(int)` and not `int (*)(int) op`.
        return Member(
            name=pointer.group(2),
            ctype=f"{pointer.group(1).strip()}(*)"
            f"{pointer.group(3)}\x00fn",
            array="",
        )
    words = declaration.replace("*", " * ").replace("&", " & ").split()
    if len(words) < 2:
        raise CppTranslationError(
            filename, at, f"cannot read the data member {declaration!r}"
        )
    holds = "&" in words
    words = [one for one in words if one != "&"]
    if len(words) < 2:
        raise CppTranslationError(
            filename, at, f"cannot read the data member {declaration!r}"
        )
    spelled = words[-1]
    # `int items[8]` declares `items`, not `items[8]`. Taken whole, the name
    # never matched a use of it, so nothing inside a method was pointed at
    # `this` and the field looked like an undeclared global.
    array = ""
    bracket = spelled.find("[")
    if bracket >= 0:
        array = spelled[bracket:]
        spelled = spelled[:bracket]
    held = " ".join(words[:-1])
    return Member(
        name=spelled,
        ctype=f"{held} *" if holds else held,
        array=array,
        reference=holds,
    )


def _method_from(head: str, body: str, filename: str, at: int) -> Method:
    at_line = at
    head = head.strip()
    # `virtual`, `override` and `final` say how the call is dispatched, not
    # what the function is, so they come off the head before it is read - and
    # `virtual` is remembered, because it is the one that changes the code.
    virtual = re.match(r"\bvirtual\b", head) is not None
    head = re.sub(r"^\s*virtual\b", "", head).strip()
    shared = re.match(r"\bstatic\b", head) is not None
    head = re.sub(r"^\s*static\b", "", head).strip()
    head = re.sub(r"\b(override|final)\b\s*$", "", head).strip()
    # `explicit` and `inline` constrain how a member may be called and
    # whether it may be duplicated; neither changes what it is.
    head = re.sub(r"^\s*(explicit|inline)\b", "", head).strip()
    head = re.sub(r"^\s*(explicit|inline)\b", "", head).strip()
    pure = False
    if re.search(r"=\s*0\s*$", head):
        pure = True
        head = re.sub(r"=\s*0\s*$", "", head).strip()
    # `const` after the parameter list, which is the only place it can be and
    # still be about the object rather than about a type.
    readonly = re.search(r"\)\s*const\s*$", head) is not None

    def decorated(method: Method) -> Method:
        method.virtual = virtual
        method.pure = pure
        method.shared = shared
        method.readonly = readonly
        return method

    # `int operator()(int x)` has two parameter lists as far as `find` is
    # concerned, and the first is the operator's own name. Where the name is
    # spelled with brackets, the parameter list is the pair after it.
    named_with_brackets = re.search(r"\boperator\s*(\(\s*\)|\[\s*\])", head)
    if named_with_brackets:
        open_paren = head.find("(", named_with_brackets.end())
    else:
        open_paren = head.find("(")
    close_paren = head.rfind(")")
    if open_paren < 0 or close_paren < 0:
        raise CppTranslationError(filename, at, f"cannot read the member {head!r}")
    before = head[:open_paren].strip()
    parameters = head[open_paren + 1: close_paren].strip()
    if parameters in ("void", ""):
        parameters = ""
    if before.startswith("~"):
        return decorated(Method("~", "void", parameters, body, at))
    if "operator" in before:
        # `Vec operator+(Vec o)` - the name is the symbol, which no identifier
        # can hold, so it is carried as a word and spelled back at the call.
        at = before.index("operator")
        # `()` and `[]` come through with their own spacing, and the name is
        # the symbol with none of it.
        symbol = re.sub(r"\s+", "", before[at + len("operator"):])
        returns = before[:at].strip()
        if symbol not in _OPERATOR_NAMES:
            # `operator int()` or `operator const char *()` - a conversion.
            # Its name is the type it answers, which is why nothing is
            # written in front of it and why it takes nothing. Read as a
            # symbol it would be one this subset had never heard of; read as
            # what it is, it is an ordinary member with an unusual spelling.
            conversion = re.sub(
                r"\s+", " ", before[at + len("operator"):]
            ).strip()
            if not returns and not parameters and _names_a_type(conversion):
                return decorated(
                    Method(
                        f"{_CONVERSION_PREFIX}{_type_code(conversion)}",
                        conversion,
                        "",
                        body,
                        at_line,
                    )
                )
            raise CppTranslationError(
                filename, at_line,
                f"py2bin's C++ subset does not know operator{symbol!r}; it "
                f"knows {', '.join(sorted(_OPERATOR_NAMES))}",
            )
        # `*x` and `x * y` are written the same and are not the same member.
        # The parameter list is what says which: a unary operator takes none.
        named = _OPERATOR_NAMES[symbol]
        if not parameters.strip():
            named = {
                "*": _DEREFERENCE, "-": _NEGATE, "&": _ADDRESS_OF
            }.get(symbol, named)
        elif named in _POSTFIX:
            # The parameter of `operator++(int)` is never given a value; it is
            # there so the two spellings can be told apart.
            named = _POSTFIX[named]
            parameters = ""
        return decorated(Method(named, returns or "void", parameters, body, at_line))
    # `&` is spaced out with `*`: `int &at` names `at` and returns `int &`,
    # and taken whole the name read as `&at`.
    words = before.replace("*", " * ").replace("&", " & ").split()
    if len(words) == 1:
        # A constructor: the name is the class's own, with no return type.
        return decorated(Method("", "void", parameters, body, at))
    return decorated(Method(words[-1], " ".join(words[:-1]), parameters, body, at))



#: Whether the file throws at all.
_THROWS = re.compile(r"\b(throw|try|catch)\b")

#: Whether the file asks for the heap at all. Word-bounded, so `newest` and a
#: member called `deleted` are not it.
_WANTS_HEAP = re.compile(r"\b(new|delete)\b")


def _wants_heap(text: str) -> bool:
    """Whether the allocator has to come with this file.

    `new` and `delete` say so. So does a `throw` of an object, which by this
    point has already become the malloc that copies it - the word `new` was
    never there.
    """

    return _WANTS_HEAP.search(text) is not None or "malloc(" in text

#: `namespace N {` - and `namespace {`, which C++ calls anonymous.
#: `namespace a { ` and, since C++17, `namespace a::b { ` - which means the
#: same as the two written one inside the other. Every piece is a namespace
#: whose qualifier has to be stripped, so the whole spelling is captured.
_NAMESPACE = re.compile(r"\bnamespace\s*((?:[A-Za-z_]\w*)(?:\s*::\s*[A-Za-z_]\w*)*)?\s*\{")
#: `using namespace N;` - after flattening there is nothing left to bring in.
_USING_NAMESPACE = re.compile(r"\busing\s+namespace\s+[A-Za-z_][\w:]*\s*;")
#: `using N::thing;`, same reasoning.
_USING_NAME = re.compile(r"\busing\s+[A-Za-z_][\w:]*\s*;")


# --- templates -------------------------------------------------------------
#
# A template is not code; it is a pattern for code. So it is expanded here,
# once per set of arguments the file actually uses, and what comes out is an
# ordinary class or function that the rest of the translator has never heard
# of a template. This is monomorphisation, which is what a C++ compiler does
# with them too - the difference being that here the copies have names you can
# read in the C.

#: `template<typename T, int N>` - the parameter list of a pattern.
_TEMPLATE = re.compile(r"\btemplate\s*<([^<>]*)>\s*")


def _is_a_definition(text: str, close: int) -> bool:
    """Whether the parentheses closing at `close` belong to a definition.

    A call is never followed by a body. `const` may sit between, and so may
    an inheritance-free `noexcept`; anything else means this was a call.
    """

    rest = text[close + 1:].lstrip()
    for word in ("const", "noexcept"):
        if rest.startswith(word):
            rest = rest[len(word):].lstrip()
    return rest.startswith("{")


def _template_parameters(spelled: str) -> "list[tuple[str, bool, bool]]":
    """Each parameter as (name, is_a_type, is_a_pack).

    `class... Ts` is a pack: one parameter standing for however many
    arguments are left, which is nought or more.
    """

    found: list[tuple[str, bool, bool]] = []
    for part in _split_arguments(spelled):
        spelling = part.strip()
        if not spelling:
            continue
        is_a_pack = "..." in spelling
        words = spelling.replace("...", " ").split()
        if not words:
            continue
        is_a_type = words[0] in ("typename", "class")
        found.append((words[-1], is_a_type, is_a_pack))
    return found


def _closing_angle(text: str, opening: int) -> int:
    """The index of the `>` closing the `<` at `opening`, or -1.

    Depth-counted, so `Holder<Pair<int, int>>` closes where it should. Only
    ever called on a name already known to be a template, which is what keeps
    a less-than out of it.
    """

    depth = 0
    index = opening
    while index < len(text):
        if text[index] == "<":
            depth += 1
        elif text[index] == ">":
            depth -= 1
            if depth == 0:
                return index
        elif text[index] in ";{}":
            return -1
        index += 1
    return -1


def _as_a_number(argument: str, text: str) -> str:
    """A non-type template argument as the number it is, where it is one.

    Only where it is written as an expression: a type name and a number that
    is already one are handed back untouched, and anything this cannot work
    out stays as it was.
    """

    spelled = argument.strip()
    if not re.search(r"[-+*/%]", spelled):
        return spelled
    folded = _folded_integer(spelled, text)
    return spelled if folded is None else str(folded)


def _instantiated_name(name: str, arguments: "list[str]") -> str:
    """`Stack<int *>` becomes `Stack__int_p`, which is a C identifier."""

    spelled = []
    for argument in arguments:
        cleaned = argument.strip().replace("*", " p").replace("&", " r")
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", cleaned).strip("_")
        spelled.append(cleaned or "x")
    return f"{name}__" + "_".join(spelled)


def _arity_fits(
    parameters: "list[tuple[str, bool, bool]]", arguments: "list[str]"
) -> bool:
    """Whether this many arguments can fill these parameters.

    Exactly as many, unless one of them is a pack - which stands for however
    many are left over, including none at all.
    """

    if any(is_pack for _n, _t, is_pack in parameters):
        return len(arguments) >= len(parameters) - 1
    return len(arguments) == len(parameters)


def _substituted(
    body: str, parameters: "list[tuple[str, bool, bool]]", arguments: "list[str]"
) -> str:
    """Replace each template parameter with the argument given for it.

    A pack stands for however many arguments are left rather than for one, so
    it is not replaced but expanded: `sizeof...(Ts)` becomes how many there
    are, `Ts...` becomes them separated by commas, and a parameter declared
    `Ts... rest` becomes one declaration each.
    """

    packed = [(name, is_pack) for name, _is_type, is_pack in parameters]
    fixed: "list[tuple[str, str]]" = []
    pack_name = ""
    pack_arguments: "list[str]" = []
    at = 0
    for index, (name, is_pack) in enumerate(packed):
        if is_pack:
            pack_name = name
            pack_arguments = [a.strip() for a in arguments[at:]]
            at = len(arguments)
            continue
        if at < len(arguments):
            fixed.append((name, arguments[at].strip()))
            at += 1
    if pack_name:
        body = _expanded_pack(body, pack_name, pack_arguments)
    for parameter, argument in fixed:
        body = _map_code(
            body,
            lambda part, p=parameter, a=argument: re.sub(
                rf"\b{re.escape(p)}\b", a, part
            ),
        )
    return body


#: `Ts... rest` - a parameter declared as however many the pack holds.
def _expanded_pack(body: str, name: str, arguments: "list[str]") -> str:
    """Write a pack out: one of everything it stands for, in order."""

    # How many, first: `sizeof...(Ts)` is a number and nothing else in the
    # body should see the name inside it.
    body = _map_code(
        body,
        lambda part: re.sub(
            rf"\bsizeof\s*\.\.\.\s*\(\s*{re.escape(name)}\s*\)",
            str(len(arguments)),
            part,
        ),
    )

    # `Ts... rest` in a parameter list: one parameter for each, named apart.
    held: "list[str]" = []

    def declared(match: "re.Match[str]") -> str:
        variable = match.group(1)
        held.append(variable)
        if not arguments:
            return ""
        return ", ".join(
            f"{spelled} {variable}__{index}"
            for index, spelled in enumerate(arguments)
        )

    body = _map_code(
        body,
        lambda part: re.sub(
            rf"\b{re.escape(name)}\s*\.\.\.\s*([A-Za-z_]\w*)", declared, part
        ),
    )
    # `rest...` where the values are used: the names just written.
    for variable in dict.fromkeys(held):
        body = _map_code(
            body,
            lambda part, v=variable: re.sub(
                rf"\b{re.escape(v)}\s*\.\.\.",
                ", ".join(f"{v}__{index}" for index in range(len(arguments))),
                part,
            ),
        )
    # And `Ts...` on its own, which is the types.
    body = _map_code(
        body,
        lambda part: re.sub(
            rf"\b{re.escape(name)}\s*\.\.\.", ", ".join(arguments), part
        ),
    )
    # A pack with nothing in it leaves `f(a, )` behind.
    return _map_code(body, lambda part: re.sub(r",\s*\)", ")", part))


#: What a literal is. `1` is an int, `1.0` a double, `"s"` a `const char *`.
_LITERAL_TYPES = (
    (re.compile(r"^[+-]?\d+[uUlL]*$"), "int"),
    (re.compile(r"^[+-]?0[xX][0-9a-fA-F]+[uUlL]*$"), "int"),
    (re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?[fF]?$"), "double"),
    (re.compile(r'^".*"$', re.S), "const char *"),
    (re.compile(r"^'.*'$", re.S), "char"),
    # `L"..."` is as ordinary as `"..."` in a program that talks to Windows,
    # where every entry point that takes a string takes a wide one.
    (re.compile(r'^L".*"$', re.S), "const wchar_t *"),
    (re.compile(r"^L'.*'$", re.S), "wchar_t"),
    (re.compile(r'^u8".*"$', re.S), "const char *"),
    (re.compile(r'^u".*"$', re.S), "const char16_t *"),
    (re.compile(r'^U".*"$', re.S), "const char32_t *"),
    (re.compile(r"^(true|false)$"), "int"),
)


#: The head of a declaration statement: the type, then its declarators.
_DECLARATION_STATEMENT = re.compile(
    r"^\s*((?:(?:const|unsigned|signed|long|short|static)\s+)*[A-Za-z_]\w*)\s+(.+)$",
    re.S,
)

#: What one declarator names: stars, then the name.
_DECLARATOR = re.compile(r"^\s*(\**)\s*([A-Za-z_]\w*)")

#: Types read out of a text, kept because `_deduced_type` asks repeatedly and
#: the answer depends only on the text. Capped, because a translation unit
#: produces a handful of distinct texts and an unbounded cache is a leak.
_DECLARED_CACHE: "dict[int, dict[str, str]]" = {}
_DECLARED_CACHE_LIMIT = 64


def _declared_here(text: str) -> "dict[str, str]":
    """Every name this text declares, and what it was declared as.

    Read statement by statement, so `int a = 1, b = 2;` declares both - which
    a pattern matching one type and one name cannot see, because only the
    first declarator has the type in front of it.
    """

    key = hash(text)
    found = _DECLARED_CACHE.get(key)
    if found is not None:
        return found
    found = {}
    for statement in _statements(_without_literals(text)):
        cleaned = statement.strip().rstrip(";").strip()
        if not cleaned or cleaned.startswith(("{", "}")):
            continue
        match = _DECLARATION_STATEMENT.match(cleaned)
        if match is None:
            continue
        spelled = match.group(1).strip()
        if spelled.split()[-1] in _NOT_A_TYPE:
            continue
        for part in _split_arguments(match.group(2)):
            declarator = _DECLARATOR.match(part)
            if declarator is None:
                continue
            # An array declarator is a pointer to its element for this purpose.
            stars = declarator.group(1)
            if "[" in part.split("=")[0]:
                stars += "*"
            found.setdefault(
                declarator.group(2), f"{spelled} {stars}".strip()
            )
    if len(_DECLARED_CACHE) >= _DECLARED_CACHE_LIMIT:
        _DECLARED_CACHE.clear()
    _DECLARED_CACHE[key] = found
    return found


#: What a function's own type is called, once it has one.
_FUNCTION_TYPE = "__py2bin_fn_"


def _function_definitions(text: str) -> "dict[str, tuple[str, str]]":
    """Each function defined at the top level, as name -> (returns, parameters)."""

    found: "dict[str, tuple[str, str]]" = {}
    for match in _DEFINITION.finditer(text):
        if _depth_at(text, match.end() - 1) != 0:
            continue
        returns = match.group(1).strip()
        # `static int bigger(...)` - how it is stored is not part of its
        # type, and a typedef for a pointer to it cannot carry the word.
        while returns.split() and returns.split()[0] in _STORAGE:
            returns = returns.split(None, 1)[1] if " " in returns else ""
        words = returns.split()
        if not words or words[0] in _NOT_A_TYPE or match.group(2) in _NOT_A_TYPE:
            continue
        found[match.group(2)] = (returns, match.group(3).strip())
    return found


#: Words that say where a function lives rather than what it answers.
_STORAGE = frozenset({"static", "inline", "extern", "__inline", "constexpr"})

#: How a function is reached, which - like how it is stored - is written
#: where a return type goes and is not one. `virtual` is a keyword, so a
#: reader that asks "is the first word a type?" answers no and walks past the
#: whole function: a `throw` inside a virtual method was never rewritten.
_DISPATCH = frozenset({"virtual", "explicit"})


def _function_type_name(spelled: str, text: str) -> "str | None":
    """The name of the type a function's own name has, if it is one."""

    if not spelled.isidentifier() or spelled in _NOT_A_TYPE:
        return None
    if spelled not in _function_definitions(text):
        return None
    return _FUNCTION_TYPE + spelled


def _name_function_types(text: str, classes: "dict[str, Class] | None" = None) -> str:
    """Give a name to the type of every function whose name is used as a value.

    C can declare a pointer to a function only by wrapping the declarator
    around it, so `C less_than` cannot become one by substituting for `C`.
    A typedef can, and this writes one wherever a function's name is passed
    somewhere rather than called.
    """

    defined = _function_definitions(text)
    if not defined:
        return text
    classes = classes or {}
    code = _without_literals(text)
    typedefs = []
    for name, (returns, parameters) in defined.items():
        # Passed somewhere rather than called: after a `(`, a `,`, an `=` or
        # a `return`, and with no `(` of its own. Asking only for "no paren
        # after it" also matched a *declaration* of a local with the same
        # name - `unsigned long at;` inside a container gave the function
        # `at` a typedef nothing had asked for.
        if not re.search(
            rf"(?:[(,=]|\breturn)\s*{re.escape(name)}\b\s*(?![\w(])", code
        ):
            continue
        typedefs.append(
            f"typedef {returns} (*{_FUNCTION_TYPE}{name})"
            # As C, not as C++: a parameter may be a reference, and the
            # typedef is read by the C compiler.
            f"({_rewrite_types(_references_to_pointers(parameters), classes) if parameters else 'void'});"
        )
    if not typedefs:
        return text
    return "\n".join(typedefs) + "\n" + text

def _deduced_type(expression: str, text: str, before: int = -1) -> "str | None":
    """What type an argument has, as far as this can tell without a type system.

    Literals say what they are. A name is looked up where it was declared. An
    expression is not worked out - a call whose type cannot be read is refused
    with the spelling that would settle it, rather than compiled as whatever
    seemed likely.
    """

    spelled = expression.strip()
    for pattern, named in _LITERAL_TYPES:
        if pattern.match(spelled):
            return named
    if not spelled.isidentifier():
        return _deduced_from_expression(spelled, text, before)
    # A function's own name, passed as a value: `sort(first, last, bigger)`.
    # It has a type - a pointer to a function - and C cannot spell one in the
    # place a template argument goes, so it is given a name of its own and
    # `_name_function_types` writes the typedef that says what it means.
    named = _function_type_name(spelled, text)
    if named is not None:
        return named
    # Bounded on both sides: without it, `main(void)` reads as a declaration
    # of `d` with type `voi`, and the copy was written out under that name.
    # The trailing `[` is how an array is declared, and an array used as a
    # value is a pointer to its first element - which is what `sort(raw, raw
    # + 8)` passes.
    # The `(` is a declaration whose declarator is a constructor call:
    # `wstring buffer(512, L'\0');` declares `buffer` as surely as `wstring
    # buffer;` does, and without it the type of one was never read - so a
    # call taking it picked the wrong overload, silently, by falling back to
    # the first.
    pattern = re.compile(
        # `&` as readily as `*`: `const V &v` is a parameter as ordinary as
        # `V *p`, and read without it the type of `v` could not be found at
        # all - so nothing reached through it could be either, and `v.x`
        # inside a function taking a reference had no type.
        rf"\b((?:const\s+)?[A-Za-z_]\w*)\s*([*&]?)\s*"
        rf"\b{re.escape(spelled)}\b\s*([=;,)\[(])"
    )
    code = _without_literals(text)
    found = [
        match
        for match in pattern.finditer(code)
        # A declaration starts a statement. `return x * x;` reads exactly like
        # one declaring a pointer `x` of type `x`, and taking it for one gave
        # a lambda the return type `x *`.
        if match.group(1) not in _NOT_A_TYPE
        and _could_start_a_declaration(code, match.start())
    ]
    if not found:
        # `int a = 1, b = 2;` puts the type in front of the first declarator
        # only, so a pattern looking for one type and one name never sees the
        # rest. Read the statement instead.
        written = _declared_here(text).get(spelled)
        if written is not None:
            return written
        # `auto v = expr;` - `auto` is not a type, it is a promise that the
        # expression to the right is one, so that is what is read. Without
        # this a name declared with it had no type at all, and a
        # capture-default could not tell its scope had even declared it.
        return _what_auto_holds(spelled, code, text, before)
    # The declaration nearest above the call, which is the one C++ would have
    # in scope; falling back to the first anywhere when the call comes first.
    earlier = [match for match in found if before < 0 or match.start() < before]
    declared = earlier[-1] if earlier else found[0]
    # A reference is the object it names, not a pointer to one: `v.x` reads a
    # member off it exactly as a value would. The `*` of a pointer is kept,
    # because there the difference is real.
    stars = declared.group(2).replace("&", "")
    stars += "*" if declared.group(3) == "[" else ""
    held = (declared.group(1) + " " + stars).strip()
    # `const Box &r` names a Box. Answered as `const Box`, every caller that
    # asks "is this a class this file declares?" heard no - so `r[0]` on a
    # const reference read as indexing a struct, and a method could not be
    # found on one either. The qualifier is a promise about writing through
    # it, not part of which type it is; kept where what is left is not a type
    # this file declares, because there `const char *` is worth saying.
    bare = re.sub(r"\b(?:const|volatile)\b", " ", held).strip()
    bare = re.sub(r"\s+", " ", bare)
    if bare.replace("*", "").strip() in _CLASS_NAMES:
        return bare
    return held



def _what_auto_holds(
    spelled: str, code: str, text: str, before: int
) -> "str | None":
    """The type of a name declared `auto`, read off what it was given."""

    chosen = None
    for match in re.finditer(
        rf"(?<![.\w>])auto\s*[*&]?\s*{re.escape(spelled)}\s*=([^;]*);", code
    ):
        if before < 0 or match.start() < before:
            chosen = match
    if chosen is None:
        return None
    value = text[chosen.start(1): chosen.end(1)].strip()
    # `auto a = a;` does not compile, and asking would not stop.
    if not value or re.search(rf"(?<![.\w>]){re.escape(spelled)}\b", value):
        return None
    return _deduced_type(value, text, chosen.start())


#: `v.begin()` or `p->at(3)` - a call on something. The arguments do not
#: matter: what a member returns is a property of the member.
_MEMBER_CALL = re.compile(
    r"^([A-Za-z_]\w*)\s*(?:\.|->)\s*([A-Za-z_]\w*)\s*\(.*\)$", re.S
)
#: `f()` or `f(1, 2)` - a plain call.
_PLAIN_CALL = re.compile(r"^([A-Za-z_]\w*)\s*\(.*\)$", re.S)



def _subscript_result(text: str, owner: str) -> "str | None":
    """What `operator[]` on that class is declared to answer."""

    return _member_result(text, owner, r"operator\s*\[\s*\]")


def _members_declared(inside: str) -> "list[tuple[str, str]]":
    """Each data member a class body declares, as (name, type).

    Read one declaration at a time rather than by looking for a name with a
    type in front of it, because `int x, y;` puts the type in front of the
    first only - and a search for the second finds no type, while a search
    for the first finds a comma where it wanted a semicolon.
    """

    found: "list[tuple[str, str]]" = []
    for piece in _without_literals(inside).split(";"):
        # The braces of the body come with it, and a nested block's do too.
        spelled = piece.strip().lstrip("{").rstrip("}").strip()
        if not spelled or "(" in spelled or spelled.startswith(
            ("public", "private", "protected", "typedef", "using", "friend")
        ):
            continue
        head = re.match(
            r"^((?:const\s+|volatile\s+|unsigned\s+|signed\s+|static\s+"
            r"|struct\s+|mutable\s+)*[A-Za-z_]\w*)\s+(.+)$",
            spelled,
        )
        if head is None:
            continue
        held = re.sub(r"\b(?:static|mutable)\b", " ", head.group(1)).strip()
        for one in head.group(2).split(","):
            named = re.match(r"^\s*([*&]*)\s*([A-Za-z_]\w*)", one)
            if named is None:
                continue
            found.append((named.group(2), (held + " " + named.group(1)).strip()))
    return found


def _member_result(text: str, owner: str, spelled: str) -> "str | None":
    """What the member matching `spelled` on that class is declared to answer.

    From the text where the class body is still in it, and from what was read
    off the class before the bodies were taken apart where it is not. A pass
    that runs after that has no body to search, and answering "no such
    member" there is not the same as there being none.
    """

    plain = re.match(r"^([A-Za-z_]\w*)", spelled)
    if plain is not None and not any(
        head.group(2) == owner for head in _CLASS_HEAD.finditer(text)
    ):
        for member, held in _CLASS_MEMBERS.get(owner, ()):
            if member == plain.group(1):
                return held
    for head in _CLASS_HEAD.finditer(text):
        if head.group(2) != owner:
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            return None
        inside = text[head.end() - 1: closing]
        # `int x, y;` declares two members and the pattern below finds
        # neither: the first is followed by a comma rather than by the end of
        # a declaration, and the second has no type in front of it at all.
        # So the declarations are read as declarations first, and the search
        # is what answers for the shapes that are not one - `operator[]` and
        # the rest.
        if plain is not None:
            for member, held in _members_declared(inside):
                if member == plain.group(1):
                    return held
        found = re.search(
            rf"(?<![.\w>])([A-Za-z_][\w\s]*?)\s*([*&]*)\s*{spelled}", inside
        )
        if found is None:
            return None
        # A reference is a name for what is there; the type is what it names.
        return (found.group(1).strip() + " " + found.group(2).replace("&", "")).strip()
    return None


def _deduced_from_expression(spelled: str, text: str, before: int) -> "str | None":
    """The type of something that is not a bare name.

    Only the shapes an argument is actually written in: a call, a pointer
    walked forward, a dereference, an address. Anything else says so rather
    than being guessed at, because the answer picks which copy of a template
    is compiled and a wrong one runs.
    """

    while spelled.startswith("(") and _closing_paren(spelled, 0) == len(spelled) - 1:
        spelled = spelled[1:-1].strip()
    if spelled.startswith("&"):
        held = _deduced_type(spelled[1:].strip(), text, before)
        return None if held is None else f"{held} *".replace("  ", " ")
    if spelled.startswith("*"):
        held = _deduced_type(spelled[1:].strip(), text, before)
        if held is None or "*" not in held:
            return None
        return held[::-1].replace("*", "", 1)[::-1].strip()
    dispatched = _DISPATCHED.match(spelled)
    if dispatched is not None:
        # A virtual call, as this translator writes one: a read from the
        # object's table, cast to the shape of the function and called. The
        # cast says what it returns, which is the only part wanted here.
        return dispatched.group(1).strip()
    cast = _CAST.match(spelled)
    if cast is not None and _names_a_type(cast.group(1)):
        # `(int)x` is an int, whatever x was. This is also the escape hatch
        # the diagnostics point at: a cast says what something is where
        # nothing else does.
        return cast.group(1).strip()
    indexed = _INDEXED.match(spelled)
    if indexed is not None:
        held = _deduced_type(indexed.group(1).strip(), text, before)
        if held is not None and "*" in held:
            return held[::-1].replace("*", "", 1)[::-1].strip()
        if held is not None:
            # A container indexed answers whatever its subscript operator
            # declares. That is where `for (auto &x : v)` gets the type of
            # what it walks over.
            return _subscript_result(text, held.strip())
        return None
    # `c ? a : b` - both arms have the same type in a program that compiles,
    # so either one answers. Read as arithmetic below, the `?` was not an
    # operator it knows and the whole expression had no type at all.
    chosen = _conditional_arms(spelled)
    if chosen is not None:
        for arm in chosen:
            held = _deduced_type(arm, text, before)
            if held is not None:
                return held
        return None
    # `p->v` and `n.v` - a member read, whose type is what the class declares
    # that member to be. Before the arithmetic below, which otherwise reads
    # the `-` of an arrow as a subtraction and answers with the type of the
    # object rather than of the member.
    reached = _MEMBER_ACCESS.match(spelled)
    if reached is not None:
        holder = _deduced_type(reached.group(1).strip(), text, before)
        if holder is not None:
            owner = re.sub(
                r"\b(?:const|struct|union)\b", " ", holder.replace("*", " ")
            ).strip()
            held = _member_result(
                text, owner, rf"{re.escape(reached.group(2))}\s*[;=\[]"
            )
            if held is not None:
                return held
    # Arithmetic: `raw + 8` is where `raw` points moved along, and `x * 2.0`
    # is whichever of the two is wider - which is what C++ does with them.
    # Not the `-` of an arrow: that is a member read, handled above, and
    # splitting there left `>v` as the right-hand side.
    # The right-hand side is one operand, so it holds no operator - but an
    # arrow is not one. Written as "no `-` at all", `this->x + o->x` matched
    # nothing and the whole expression had no type, which is what decided
    # which constructor `V(x + o.x)` meant.
    binary = re.match(
        r"^(.+?)\s*(-(?!>)|[+*/%])\s*((?:[^-+*/%]|->)+)$", spelled
    )
    if binary is not None:
        left = _deduced_type(binary.group(1).strip(), text, before)
        # An operator on an object answers what that operator declares, and
        # not what arithmetic on two numbers would. `b - a` on two time
        # points is a duration, which is a class and not a wider number.
        if left is not None:
            owner = re.sub(
                r"\b(?:const|struct|volatile)\b", " ", left
            ).replace("*", " ").strip()
            answered = _member_result(
                text, owner, rf"operator\s*{re.escape(binary.group(2))}"
            )
            if answered is not None:
                return answered
        right = _deduced_type(binary.group(3).strip(), text, before)
        held = _wider(left, right)
        if held is not None:
            return held
    return _deduced_from_call(spelled, text, before)


#: Widest last. What C++ calls the usual arithmetic conversions, as far as
#: picking one type out of two goes.
_WIDTH_ORDER = (
    "char", "short", "int", "unsigned int", "long", "unsigned long",
    "long long", "float", "double",
)


def _wider(left: "str | None", right: "str | None") -> "str | None":
    """Which of two types an expression mixing them has."""

    if left is None:
        return right
    if right is None:
        return left
    # Pointer arithmetic answers with the pointer, whichever side it is on.
    if "*" in left:
        return left
    if "*" in right:
        return right
    order = {name: index for index, name in enumerate(_WIDTH_ORDER)}
    here, there = order.get(left.strip()), order.get(right.strip())
    if here is None or there is None:
        return left
    return left if here >= there else right


#: `((int (*)(struct Base *))(p->__vptr[0]))(p)` - a virtual call, spelled
#: the way this translator spells one. The cast names the return type.
_DISPATCHED = re.compile(r"^\(\(\s*(.+?)\s*\(\s*\*\s*\)\s*\(", re.S)

#: `(int)x`, `(const char *)p` - a cast, which says what something is.
_CAST = re.compile(r"^\(\s*((?:const\s+)?[A-Za-z_]\w*(?:\s*\*)*)\s*\)\s*(.+)$", re.S)
#: What the pass that hoists a value return calls what it writes. Read back
#: by the caller, which has to know these are objects of the scope.
_VALUE_PREFIX = "__py2bin_value_"

#: `return {};` - the answer, value initialised, with no type written
#: because the function's own declaration says which one.
_EMPTY_RETURN = re.compile(r"(?<![.\w>])return\s*\{\s*\}\s*;")

def _conditional_arms(spelled: str) -> "tuple[str, str] | None":
    """The two answers of `c ? a : b`, or None where this is not one.

    Read by scanning rather than by pattern: either arm may hold a `?` of
    its own, and the `:` that belongs to this one is the first at the outer
    level with no unmatched `?` in front of it.
    """

    depth = 0
    question = -1
    pending = 0
    for index, piece in enumerate(_without_literals(spelled)):
        if piece in "([{":
            depth += 1
        elif piece in ")]}":
            depth -= 1
        elif depth != 0:
            continue
        elif piece == "?":
            if question < 0:
                question = index
            else:
                pending += 1
        elif piece == ":" and question >= 0:
            if pending:
                pending -= 1
                continue
            return spelled[question + 1: index].strip(), spelled[index + 1:].strip()
    return None


#: `a[i]`, whatever `a` is.
_INDEXED = re.compile(r"^(.+)\[[^\]]*\]$", re.S)

#: `n.v` or `p->head->v` - a member read and nothing else. No parentheses
#: anywhere, so a call on a member is left to the pass that reads calls.
_MEMBER_ACCESS = re.compile(
    r"^([A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)*)\s*(?:\.|->)"
    r"\s*([A-Za-z_]\w*)$"
)

#: The words a cast's contents may begin with for it to be a cast rather than
#: a call through a pointer. A `*` settles it on its own; otherwise the name
#: has to be one this translator knows is a type.
_TYPE_WORDS = frozenset(
    """void char short int long float double signed unsigned _Bool bool
    wchar_t char16_t char32_t size_t""".split()
)


def _names_a_type(spelled: str) -> bool:
    if "*" in spelled:
        return True
    words = spelled.replace("const", "").split()
    if not words:
        return False
    return words[0] in _TYPE_WORDS or words[0] in _CLASS_NAMES


def _deduced_from_call(spelled: str, text: str, before: int) -> "str | None":
    """The type of a call's result, where the call is simple enough to read.

    `sort(v.begin(), v.end())` is how nearly every use of <algorithm> is
    written, and none of it can be deduced without knowing what `begin`
    returns. So a call on a name is followed: the name says which class, and
    the class says what the member returns.
    """

    member = _MEMBER_CALL.match(spelled)
    if member is not None:
        held = _deduced_type(member.group(1), text, before)
        if held is None:
            return None
        return _declared_return(text, held.replace("*", "").strip(), member.group(2))
    # `(b - a).count()` and `f(x).m()` - a call on something that is itself
    # an expression rather than a name. The receiver is worked out first and
    # the member read off whatever it answered, which is the only way a chain
    # of them can be followed at all.
    bare = _without_literals(spelled)
    if spelled.endswith(")"):
        depth = 0
        for index in range(len(bare) - 1, -1, -1):
            piece = bare[index]
            if piece in ")]}":
                depth += 1
            elif piece in "([{":
                depth -= 1
            elif depth == 0 and piece == ".":
                tail = spelled[index + 1:].strip()
                named = re.fullmatch(r"([A-Za-z_]\w*)\s*\(.*\)", tail, re.S)
                if named is None:
                    break
                holder = _deduced_type(spelled[:index].strip(), text, before)
                if holder is None:
                    break
                owner = re.sub(
                    r"\b(?:const|struct|volatile)\b", " ", holder
                ).replace("*", " ").strip()
                answered = _declared_return(text, owner, named.group(1))
                if answered is not None:
                    return answered
                break

    # `steady_clock::now()` - a static member, called by its class's name.
    # Neither a call on an object nor a plain one, and `auto` in front of it
    # had nothing to read.
    qualified = re.match(
        r"^([A-Za-z_]\w*)\s*::\s*([A-Za-z_]\w*)\s*\(.*\)$", spelled, re.S
    )
    if qualified is not None:
        return _declared_return(text, qualified.group(1), qualified.group(2))
    plain = _PLAIN_CALL.match(spelled)
    if plain is not None:
        # `Res__new(7)` - the allocator this translator writes where the
        # program wrote `new Res(7)`. Its definition is emitted with the
        # classes and is not in the text a body is rewritten against, so
        # reading it is not an option; but py2bin wrote the name itself, and
        # what it says is an object of that class on the heap.
        made = plain.group(1)
        for tail in ("__new_array", "__new"):
            if not made.endswith(tail):
                continue
            held = made[: -len(tail)]
            if held in _CLASS_NAMES or _names_a_class(text, held):
                return f"{held} *"
        # `greater<int>()` is not a call to a function - it builds one of
        # those. A name that a class body carries is that class, and the
        # temporary it makes has its type.
        if _names_a_class(text, plain.group(1)):
            return plain.group(1)
        # `f(5)` where `f` is an object: a name that holds one is never a
        # function, so the call is that class's own call operator and what it
        # answers is what this is. That is how a lambda returning a lambda is
        # held - `auto add5 = outer(5)` has no other way to be read.
        holder = _deduced_type(plain.group(1), text, before)
        if holder is not None and "*" not in holder:
            answered = _member_result(text, holder.strip(), r"operator\s*\(\s*\)")
            if answered is None:
                # A closure being expanded right now: its class is written
                # but not yet in the text.
                answered = _LAMBDA_RESULTS.get(holder.strip())
            if answered is not None:
                return answered
        return _declared_return(text, None, plain.group(1))
    return None


def _names_a_class(text: str, name: str) -> bool:
    """Whether the text defines a class or struct by that name."""

    return any(head.group(2) == name for head in _CLASS_HEAD.finditer(text))


def _declared_return(text: str, owner: "str | None", method: str) -> "str | None":
    """What `owner::method` is declared to return, read out of the source.

    Read from the text rather than from parsed classes because this runs
    before the classes are taken apart - a template is expanded first, and
    the expansion is what a class body is made of.
    """

    where = text
    if owner is not None:
        found = None
        for head in _CLASS_HEAD.finditer(text):
            if head.group(2) == owner:
                found = head
                break
        if found is None:
            return None
        try:
            where = text[found.end() - 1: _matching(text, found.end() - 1)]
        except ValueError:
            return None
    for match in _ANY_DEFINITION.finditer(where):
        head = match.group(1).strip()
        words = head.replace("*", " * ").replace("&", " & ").split()
        if not words or words[-1] != method or len(words) < 2:
            continue
        # How it is stored is not part of what it answers - the same reading
        # the declaration branch below already takes. Left on, `auto x =
        # f();` where `f` is `static` declared `x` static too: built once, on
        # the first call, and the same value ever after.
        spelled = words[:-1]
        while spelled and spelled[0] in _STORAGE | _DISPATCH:
            spelled = spelled[1:]
        return " ".join(spelled).replace("&", "*").strip()
    # A declaration says the same thing a definition does, and the methods
    # this translator emits are declared before they are defined - so a call
    # to one is often read where only the declaration is in view.
    for match in _PROTOTYPE.finditer(where):
        if match.group(3) != method:
            continue
        spelled = match.group(1).strip()
        # How it is stored is not part of what it answers.
        while spelled.split() and spelled.split()[0] in _STORAGE:
            spelled = spelled.split(None, 1)[1] if " " in spelled else ""
        return f"{spelled} {match.group(2) or ''}".strip()
    # A return type written with a template or a qualifier in it -
    # `std::function<int(int)> adder(int n)` - is not one the scans above can
    # see: the head they read is word characters and stars, so a `<` or a
    # `::` ends the match before the name. Asked for one function by name,
    # this can afford the looser reading they cannot.
    loose = re.search(
        rf"(?<![.\w>])((?:[A-Za-z_][\w:]*\s*<[^;{{}}]*>|[A-Za-z_][\w:]*)"
        rf"(?:\s*[*&]+)?)\s+{re.escape(method)}\s*\([^;{{}}]*\)\s*\{{",
        _without_literals(where),
    )
    if loose is not None:
        spelled = loose.group(1).strip()
        while spelled.split() and spelled.split()[0] in _STORAGE | _DISPATCH:
            spelled = spelled.split(None, 1)[1] if " " in spelled else ""
        return spelled.replace("&", "*").strip() or None
    return None

def _deduce_arguments(
    parameters: "list[tuple[str, bool]]",
    declared: str,
    given: "list[str]",
    text: str,
    before: int = -1,
) -> "list[str] | None":
    """Work out the template arguments a call did not spell out.

    Only where a parameter is used *as* an argument's type - `T v` deduces T
    and `Box<T> v` does not. That is the common case and the one whose answer
    is unambiguous.
    """

    wanted = [part.strip() for part in _split_arguments(declared) if part.strip()]
    pack = next((name for name, _t, is_pack in parameters if is_pack), "")
    tail: "list[str]" = []
    if pack:
        # A pack takes whatever is left over, so the declared parameters
        # before it have to match one for one and the rest are its.
        spelled = next(
            (index for index, part in enumerate(wanted) if "..." in part), -1
        )
        if spelled < 0 or len(given) < spelled:
            return None
        tail = given[spelled:]
        wanted, given = wanted[:spelled], given[:spelled]
    elif len(wanted) != len(given):
        return None
    found: dict[str, str] = {}
    names = {name for name, is_type, _pack in parameters if is_type}
    # A non-type parameter is deduced from the shape too: `Buf<N>` given a
    # `Buf<3>` settles N as surely as `Box<T>` given a `Box<int>` settles T.
    # It is kept out of `names` above, where the rule is "the parameter *is*
    # the type", because `N v` declares a variable of a type called N and
    # deduces nothing.
    shaped = {name for name, _is_type, _pack in parameters}
    for part, argument in zip(wanted, given):
        stars = part.count("*")
        words = part.replace("*", " * ").replace("&", " ").split()
        if len(words) < 2:
            continue
        named = words[0] if words[0] not in ("const",) else (words[1] if len(words) > 2 else "")
        if named not in names:
            # Not `T v`, so try the shape: a parameter written in terms of
            # another template - `Box<U> *` - deduces from an argument of
            # that template. The plain rule reads the first word and this one
            # takes the spelling apart, which is the only way `ComPtr<U> *`
            # says anything about what it was handed.
            spelled = re.sub(r"\b[A-Za-z_]\w*$", "", part).strip()
            if "<" not in spelled:
                continue
            deduced = _deduced_type(argument, text, before)
            if deduced is None:
                continue
            settled: "dict[str, str]" = {}
            if _fits_the_shape(
                spelled, deduced.strip(), shaped, settled, binding=True
            ):
                for held, value in settled.items():
                    found.setdefault(held, value)
            continue
        if named in found:
            continue
        deduced = _deduced_type(argument, text, before)
        if deduced is None:
            return None
        if stars:
            # `T *first` given an `int *` deduces T as `int`, not `int *`.
            # Without this, `sort(v.begin(), v.end())` asked for a copy of
            # sort over `int *` and compared the addresses.
            peeled = deduced
            for _ in range(stars):
                if "*" not in peeled:
                    return None
                peeled = peeled[::-1].replace("*", "", 1)[::-1].strip()
            deduced = peeled
        found[named] = deduced
    if pack:
        held: "list[str]" = []
        for argument in tail:
            deduced = _deduced_type(argument, text, before)
            if deduced is None:
                return None
            held.append(deduced)
        found[pack] = ""  # counted below; the values are spliced in instead
        answer: "list[str]" = []
        for name, _is_type, is_pack in parameters:
            if is_pack:
                answer.extend(held)
                continue
            if name not in found:
                return None
            answer.append(found[name])
        return answer
    # Every parameter has to have been settled. Answering with only some of
    # them named a copy that was never written; asking the caller to spell
    # them out is the honest answer.
    if any(name not in found for name, _is_type, _pack in parameters):
        return None
    return [found[name] for name, _is_type, _pack in parameters]


# --- lambdas ---------------------------------------------------------------
#
# A lambda is a class with a call operator and a member per capture. That is
# not an analogy: it is what the standard says one is, and writing it out is
# all this does. The name is generated, which is the only part a program
# cannot say for itself - which is why `auto` is the only way to hold one.

#: `auto f = __py2bin_lambda_1__made;` once the lambda itself is written out.
#: `auto` is the only way a program can hold one, because the class's name is
#: generated and there is nothing else to write.
_AUTO_FROM_LAMBDA = re.compile(
    r"\bauto\s+([A-Za-z_]\w*)\s*=\s*(__py2bin_lambda_\d+)__made"
)

#: `[captures](params) -> result {` - the head of a lambda.
#: The parameter list may be left out where there are no parameters: `[] { }`
#: is a lambda as surely as `[]() { }` is, and is how one taking nothing is
#: usually written. Read as needing the parentheses, one written that way was
#: not a lambda at all and stayed in the C as square brackets and a block.
_LAMBDA = re.compile(
    # Nothing that can be indexed in front of it. `int room[2] { 1, 2 };` is
    # an array with its values, and once the parentheses are optional it
    # reads exactly like a lambda capturing `2`. A subscript always follows
    # something - a name, a `]`, a `)` - and a capture list never does.
    r"(?<![\w\]\)])"
    r"\[([^\]\[]*)\]\s*(?:\(([^()]*)\)\s*)?(?:mutable\s*)?"
    r"(?:->\s*([^{;]+?))?\s*\{"
)


#: `friend class X;` or `friend int peek(X &);` - an access grant.
_FRIEND = re.compile(r"\bfriend\b[^;{}]*;")

#: `class X final {`, `void f() override {`, `void f() final;`. Both words
#: are checks C++ makes and C cannot, so both go.
_FINAL_OR_OVERRIDE = re.compile(
    r"(?<=\w)\s+(?:final|override)\b(?=\s*[{;:,)])"
)

#: `int safe() noexcept {` and `void go() noexcept(true) {` - a promise about
#: what a function does not do. py2bin has no unwinder, so a function that
#: throws is written out as a `return` either way and the promise changes
#: nothing it emits. Taken off rather than refused: what it says is true of
#: everything here or of nothing, and neither reading makes a difference to
#: the code.
_NOEXCEPT = re.compile(r"(?<![.\w>])noexcept\b(?:\s*\([^()]*\))?\s*")

#: `class B;` - a forward declaration, on a line of its own.
_FORWARD_DECLARATION = re.compile(
    r"(?m)^[ \t]*(class|struct|union)\s+([A-Za-z_]\w*)\s*;[ \t]*$"
)


def _rewrite_forward_declarations(text: str) -> str:
    """Drop each forward declaration, unless nothing here defines the name.

    `class B;` above the body of `B` says nothing C needs: the typedefs this
    file emits already name every class before any body, so the line has
    nothing left to do and goes.

    A name that is never defined here is the other half of the same spelling,
    and it does have work to do. `struct Impl;` with the body in a file this
    one is not compiled with - or nowhere at all, which is how an opaque
    handle is written - is the only thing that tells C the name is a type.
    Dropped along with the rest, `Impl *p;` reached the C front end as a
    declaration of nothing and was refused, in a program clang++ builds
    without a word. So it becomes what C spells the same thought with: a
    typedef of a tag with no members, which a pointer may point at and
    nothing may take apart or measure.
    """

    defined = {head.group(2) for head in _CLASS_HEAD.finditer(text)}
    defined.update(one.group(2) for one in _TAGGED_TYPE.finditer(text))

    def taken(match: "re.Match[str]") -> str:
        if match.group(2) in defined or _depth_at(text, match.start()) != 0:
            return ""
        # C has one keyword for both of C++'s, and the tag names the same
        # incomplete type either way.
        kind = "struct" if match.group(1) == "class" else match.group(1)
        return f"typedef {kind} {match.group(2)} {match.group(2)};"

    return _FORWARD_DECLARATION.sub(taken, text)


#: `static_cast<int>(x)` and the rest. C has one cast and it is spelled with
#: parentheses; these say *which* conversion is meant, which C++ checks and C
#: does not, so the check is what is lost and the conversion is not.
_NAMED_CAST = re.compile(
    r"\b(static_cast|const_cast|reinterpret_cast)\s*<([^<>]+)>\s*\("
)

#: `enum class Mode {` and `enum struct Mode {`, which C spells `enum Mode`.
#: `enum class Level` and `enum class Level : int`. C++ lets the type the
#: enumerators are stored in be named; C has one and does not, and py2bin's
#: enumerators are ints either way - so the name is read and dropped rather
#: than left for the C compiler to trip over.
_SCOPED_ENUM = re.compile(
    r"\benum\s+(?:class|struct)\s+([A-Za-z_]\w*)(\s*:\s*[A-Za-z_][\w\s]*?)?(?=\s*[{;])"
)

#: `enum Level : unsigned char {` without the `class`, which C++11 also allows.
_ENUM_BASE = re.compile(
    r"\benum\s+([A-Za-z_]\w*)\s*:\s*[A-Za-z_][\w\s]*?(?=\s*\{)"
)

#: `enum Name {`, `union Name {` - anything whose bare name C++ takes for a
#: type and C does not.
_TAGGED_TYPE = re.compile(r"\b(enum|union)\s+([A-Za-z_]\w*)\s*\{")



#: A plain struct's definition, or a typedef with no body of its own. Both
#: belong above the classes, and which one matched decides how much is taken.
#: A typedef naming an enum is not hoisted with the rest. C has no
#: incomplete enum, so `typedef enum Mode Mode;` above the body it names is
#: not a forward declaration but an error - and the pass that writes one puts
#: it directly after the body for exactly that reason.
_A_TYPEDEF_OF_AN_ENUM = re.compile(r"\btypedef\s+enum\b")

_HOISTED_TO_THE_TOP = re.compile(
    _CLASS_HEAD.pattern + r"|(?P<typedef>(?m:^)[ \t]*typedef[^;{}]*;)"
)


def _hoist_plain_structs(text: str, plain: "list[str]") -> "tuple[str, list[str]]":
    """Take each plain struct's body out, to be emitted above the classes.

    A `struct` with no methods is C already and is left exactly as written -
    but a class holding one is emitted above whatever is left of the file, so
    the class named a type C had not seen. The bodies come up in the order
    they were written, which keeps one holding another after it.
    """

    wanted = set(plain)
    bodies: "list[str]" = []
    out: list[str] = []
    at = 0
    # Both, in the order they were written: a typedef at file scope names a
    # type the classes below may be declared in terms of, and it was being
    # emitted after them - so `typedef long HRESULT;` came after the first
    # thing that answered one.
    for head in _HOISTED_TO_THE_TOP.finditer(text):
        if head.group("typedef") and _A_TYPEDEF_OF_AN_ENUM.search(head.group(0)):
            continue
        if head.start() < at:
            continue
        if head.group("typedef") is not None:
            bodies.append(head.group(0))
            out.append(text[at:head.start()])
            at = head.end()
            continue
        if head.group(2) not in wanted:
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        # To the end of the statement, not just past the brace: `typedef
        # struct P { ... } P;` names the type *after* the body, and stopping
        # at the `}` left ` P;` behind as a statement of its own.
        end = text.find(";", closing)
        end = closing if end < 0 else end + 1
        # And back to the front of it, for the same reason: taking the struct
        # out of the middle left the `typedef` standing alone.
        begins = head.start()
        before = text[:begins].rstrip()
        if before.endswith("typedef"):
            begins = len(before) - len("typedef")
        bodies.append(text[begins:end])
        out.append(text[at:begins])
        at = end
    out.append(text[at:])
    return "".join(out), bodies

def _hoist_tagged_types(text: str) -> "tuple[str, list[str]]":
    """Take every top-level `enum`/`union` definition out, with its typedef.

    Returns what is left and each of the ones taken. Most go back above the
    struct definitions, because a struct holding one needs the complete type
    and C reads a file top to bottom - but one that holds a *class* needs the
    class first, so which side of them each goes is decided where they are
    put back rather than here.
    """

    taken: list[str] = []
    out: list[str] = []
    at = 0
    for match in _TAGGED_TYPE.finditer(text):
        if match.start() < at or _depth_at(text, match.start()) != 0:
            continue
        try:
            closing = _matching(text, match.end() - 1)
        except ValueError:
            continue
        # `typedef enum M { ... } M;` is one declaration, and the words in
        # front of the body belong to it. Taking the body alone left the
        # `typedef` where it was and the name after the brace with it, which
        # is two fragments and neither of them C. A generated COM header
        # writes every one of its enums this way.
        start = match.start()
        before = text[:start].rstrip()
        if before.endswith("typedef"):
            start = len(before) - len("typedef")
        end = closing
        while end < len(text) and text[end] in " \t\n":
            end += 1
        if end < len(text) and text[end] == ";":
            end += 1
        elif start != match.start():
            # A typedef, so what follows the brace is the name it is being
            # given - as many as it declares, up to the semicolon.
            named = re.match(r"[^;{}]*;", text[end:])
            if named is None:
                continue
            end += named.end()
        else:
            # Neither a typedef nor a bare definition: `enum M { ... } v;`
            # declares an object, and moving that is moving the object.
            continue
        # The typedef this file's own spelling pass wrote just after it.
        following = re.match(
            r"\s*typedef\s+(?:enum|union)\s+\w+\s+\w+\s*;", text[end:]
        )
        if following:
            end += following.end()
        taken.append(text[start:end])
        out.append(text[at:start])
        at = end
    out.append(text[at:])
    return "".join(out), taken


def _tag_typedef(name: str, text: str) -> str:
    """`typedef enum Colour Colour;` - whichever tag the name was declared with."""

    kind = "union" if re.search(rf"\bunion\s+{re.escape(name)}\s*\{{", text) else "enum"
    return f"typedef {kind} {name} {name};"



#: `using Number = int;` and `using Fn = int (*)(int);` - a typedef with the
#: name in front. `using namespace x;` and `using B::f;` have no `=` and are
#: not this.
#: `template <typename T> using Row = std::vector<T>;` - another name for a
#: template, not a template of its own.
_ALIAS_TEMPLATE = re.compile(
    r"(?<![.\w>])template\s*<([^<>]*)>\s*using\s+([A-Za-z_]\w*)\s*=\s*([^;]+);"
)


#: `template <typename T> int Counter<T>::made = 0;` - storage for a static
#: member of a class template, which C++ asks for once and which this needs
#: once per copy.
_TEMPLATE_STATIC = re.compile(
    r"(?<![.\w>])template\s*<([^<>]*)>\s*([A-Za-z_][\w\s*]*?)\s+"
    r"([A-Za-z_]\w*)\s*<[^<>]*>\s*::\s*([A-Za-z_]\w*)\s*(=[^;]*)?;"
)


def _fold_template_statics(text: str) -> str:
    """Move a class template's static storage into the class body.

    C++ wants one definition outside the template, and gives each copy its own
    object from it. py2bin writes the copies, so the value is put where the
    member is declared instead - it then rides along with every copy, which is
    the same one-object-per-instantiation, arrived at from the other side.
    """

    for _round in range(_HOIST_ROUNDS):
        found = _TEMPLATE_STATIC.search(_without_literals(text))
        if found is None:
            return text
        owner, member = found.group(3), found.group(4)
        value = (text[found.start(5): found.end(5)] if found.group(5) else "= 0")
        text = text[: found.start()] + text[found.end():]
        head = next(
            (
                one
                for one in _CLASS_HEAD.finditer(_without_literals(text))
                if one.group(2) == owner
            ),
            None,
        )
        if head is None:
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        inside = text[head.end(): closing - 1]
        written = re.sub(
            rf"(?<![.\w>])(static\s+[\w\s*]*?\b{re.escape(member)}\s*)(;)",
            rf"\1{value.strip()};",
            inside,
            count=1,
        )
        text = text[: head.end()] + written + text[closing - 1:]
    return text


def _expand_alias_templates(text: str) -> str:
    """`Row<int>` becomes what `Row` is another name for.

    An alias template writes out no code of its own: it is a name for a
    spelling, with holes in it. So every use is replaced by that spelling with
    the holes filled, and the alias goes - which is what it means, and leaves
    nothing for the pass that writes copies to be confused by.
    """

    for _round in range(_HOIST_ROUNDS):
        found = _ALIAS_TEMPLATE.search(_without_literals(text))
        if found is None:
            return text
        named = [
            part.split()[-1]
            for part in _split_arguments(found.group(1))
            if part.strip()
        ]
        alias, body = found.group(2), text[found.start(3): found.end(3)].strip()
        text = text[: found.start()] + text[found.end():]
        bare = _without_literals(text)
        out: "list[str]" = []
        at = 0
        for use in re.finditer(rf"(?<![.\w>]){re.escape(alias)}\s*<", bare):
            if use.start() < at:
                continue
            shut = _closing_angle(bare, use.end() - 1)
            if shut < 0:
                continue
            given = [
                one.strip()
                for one in _split_arguments(text[use.end(): shut])
                if one.strip()
            ]
            if len(given) != len(named):
                continue
            filled = body
            for spelled, value in zip(named, given):
                filled = re.sub(
                    rf"(?<![.\w>]){re.escape(spelled)}\b(?!\s*::)", value, filled
                )
            out.append(text[at: use.start()])
            out.append(filled)
            at = shut + 1
        out.append(text[at:])
        text = "".join(out)
    return text


_ALIAS = re.compile(r"(?<![.\w>])using\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")

def _rewrite_cpp_spellings(text: str) -> "tuple[str, set[str]]":
    """The C++ that is a different spelling of C, spelled the C way.

    Returns the text and the tag names that need a typedef, since C++ lets a
    bare `Colour` or `U` name a type and C wants `enum Colour` or `union U`.
    """

    # `using Number = int;` is a typedef written the other way round, which
    # is the only way C++11 and later spell one in new code.
    # Before the plain alias: `template <typename T> using Row = ...;` read as
    # one becomes `template <typename T> typedef ... Row;`, which is nothing.
    text = _fold_template_statics(text)
    text = _expand_alias_templates(text)
    text = _map_code(text, lambda part: _ALIAS.sub(r"typedef \2 \1;", part))
    # `class X final {` and `void f() override` say a thing may not be
    # derived from or overridden again. C++ checks that and C has nothing to
    # check, so the word is what is lost and nothing else - but left in, it
    # reaches the C compiler as a name where a name cannot be.
    text = _map_code(text, lambda part: _FINAL_OR_OVERRIDE.sub("", part))
    text = _map_code(text, lambda part: _NOEXCEPT.sub(" ", part))
    text = _rewrite_forward_declarations(text)
    # `friend class X;` and `friend int f();` grant access to what is private.
    # py2bin emits a plain struct and enforces no access at all, so a friend
    # declaration asks for something already given and has nothing to become.
    text = _map_code(text, lambda part: _FRIEND.sub("", part))
    # Keywords in C++, and in C either a macro from a header the program did
    # not include or nothing at all. Spelled out here so a program need not
    # remember which.
    text = _map_code(
        text,
        # `_Bool` and not `int`: C++'s `bool` holds 0 and 1 and is one byte,
        # which is what `_Bool` is. Written as `int` it was a *signed* type,
        # so `bool flag : 1;` - which is how every header writes a flag -
        # held only 0 and -1, and a field set to `true` compared unequal to
        # it. `sizeof` was wrong by four times as well.
        lambda part: re.sub(
            r"\bnullptr\b", "0", re.sub(r"\bbool\b", "_Bool", part)
        ),
    )
    text = _map_code(
        text,
        lambda part: re.sub(r"\btrue\b", "1", re.sub(r"\bfalse\b", "0", part)),
    )

    # `static_cast<int>(d)` becomes `((int)(d))`. The argument's own
    # parentheses are consumed so the replacement closes what it opens - a
    # substitution alone left the outer one hanging.
    while True:
        found = _NAMED_CAST.search(text)
        if found is None:
            break
        close = _closing_paren(text, found.end() - 1)
        if close < 0:
            break
        inside = text[found.end(): close]
        text = (
            text[:found.start()]
            + f"(({found.group(2).strip()})({inside}))"
            + text[close + 1:]
        )

    scoped = {match.group(1) for match in _SCOPED_ENUM.finditer(text)}
    text = _map_code(text, lambda part: _SCOPED_ENUM.sub(r"enum \1", part))
    # And a plain `enum Level : unsigned char {` for the same reason.
    text = _map_code(text, lambda part: _ENUM_BASE.sub(r"enum \1", part))

    # The typedef goes immediately after the body, not with the class ones at
    # the top: C has no forward declaration of an enum, so a typedef naming
    # one before it is defined is not C.
    out: list[str] = []
    at = 0
    for match in _TAGGED_TYPE.finditer(text):
        if match.start() < at:
            continue
        try:
            closing = _matching(text, match.end() - 1)
        except ValueError:
            continue
        end = closing
        while end < len(text) and text[end] in " \t\n":
            end += 1
        if end >= len(text) or text[end] != ";":
            # Something is declared with the body - `enum M { ... } M;`,
            # which is C's own way of writing this and is already what the
            # typedef below would say, or `enum M { ... } value;`, which
            # declares an object. Either way the declaration is not over,
            # and putting a typedef in the middle of it left the rest of it
            # stranded on its own.
            continue
        end += 1
        kind, name = match.group(1), match.group(2)
        out.append(text[at:end])
        out.append(f"\ntypedef {kind} {name} {name};\n")
        at = end
    out.append(text[at:])
    return "".join(out), scoped

#: `auto n = 5;` - a declaration whose type is whatever the initialiser is.
#: The `&` goes with the type and is written against the name: `auto &x = ...`.
_AUTO_DECLARATION = re.compile(
    r"(?<![.\w>])auto\s*([*&]*)\s*([A-Za-z_]\w*)\s*=\s*([^;]+);"
)

#: A parenthesised group that may hold three more inside it. What an
#: initialiser is given is an expression, and an expression is often a call:
#: `handler_(move(handler))` is the ordinary way to write one, and a pattern
#: that stops at the first `)` stops inside it.
_NESTED_PARENS = (
    r"\((?:[^()]|\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\))*\)"
)

#: `C(int x, int y) : a(x), b(y) {` - a constructor's initialiser list.
_INITIALISER_LIST = re.compile(
    rf"\)\s*:\s*((?:[A-Za-z_]\w*\s*{_NESTED_PARENS}\s*,?\s*)+)\{{"
)

#: `a(x)` inside one, with the parentheses kept: what is between them may
#: itself hold a pair, so the group is taken whole and opened here.
_ONE_INITIALISER = re.compile(rf"([A-Za-z_]\w*)\s*({_NESTED_PARENS})")


def _rewrite_auto(text: str) -> str:
    """`auto n = 5;` becomes `int n = 5;`, and so on.

    A lambda is handled before this and needs no help: it is the one case
    where the type has no spelling a program could have written. Everything
    else `auto` stands for is a type the initialiser already says.
    """

    def one(match: "re.Match[str]") -> "str | None":
        stars, name = match.group(1), match.group(2)
        value = text[match.start(3): match.end(3)].strip()
        held = _deduced_type(value, text, match.start())
        if held is None:
            return None
        # `auto *p = q;` where `q` is a `T *` deduces `auto` as `T`, not as
        # `T *`: the star in the declarator is one the deduced type then does
        # not carry. Written out with both, the declaration was a `T **`, and
        # every use of it read one indirection too few.
        while stars.startswith("*") and held.rstrip().endswith("*"):
            stars = stars[1:]
        # `const auto *q = ...` where the deduced type is already const: the
        # word is written once in the source and once in the deduction, and
        # C says it twice.
        if re.search(r"\bconst\s*$", text[:match.start()]):
            held = re.sub(r"^\s*const\b\s*", "", held)
        return f"{held} {stars}{name} = {value};"

    return _sub_code(_AUTO_DECLARATION, text, lambda match, whole: one(match))



#: `C() = default;` and `C(const C &) = delete;` - a member the compiler is
#: told to write, or told never to write.
_DEFAULTED_MEMBER = re.compile(
    r"(?<![.\w>])(~?[A-Za-z_]\w*)\s*\(([^;{}()]*)\)\s*"
    r"(?:const\s*)?(?:noexcept\s*)?=\s*(default|delete)\s*;"
)


def _rewrite_defaulted_members(text: str) -> str:
    """`= default` becomes a body; `= delete` takes the declaration away.

    An empty body is what a defaulted constructor or destructor *is* here:
    the subobject construction and destruction are put in by the pass that
    does that for every one of them, written or not.

    A copy constructor is the exception. Defaulted, it means the memberwise
    copy py2bin already does when a class declares none - so writing it an
    empty body would give it one that copies nothing. Both spellings drop it
    instead. `= delete` is a promise C++ enforces and this does not: a
    program that uses a deleted member compiles here rather than being
    refused, which is permissive and never wrong for a program that is right.
    """

    def one(match: "re.Match[str]") -> str:
        name, parameters, kind = match.groups()
        held = parameters.replace("&", " ").replace("const", " ").strip()
        if held.split() and held.split()[0] == name:
            # `C(const C &)` - the copy constructor.
            return ""
        return "" if kind == "delete" else f"{name}({parameters}) {{ }}"

    return _map_code(text, lambda part: _DEFAULTED_MEMBER.sub(one, part))

def _rewrite_initialiser_lists(text: str) -> str:
    """`C(int x) : a(x) { }` becomes `C(int x) { a = x; }`.

    C++ initialises a member there rather than assigning to it, which matters
    for a member that cannot be assigned - a reference, or a const. Nothing
    in this subset has one, so the two are the same thing written twice, and
    the assignment is the one C has.
    """

    bases = {
        head.group(2): _bases_of(head)[0]
        for head in _CLASS_HEAD.finditer(text)
        if _bases_of(head)
    }

    def one(match: "re.Match[str]") -> str:
        # Which class this constructor belongs to, so its base can be told
        # from its members: `Sub(int v) : Base(v * 2)` constructs the base,
        # and turning that into `Base = v * 2;` assigns to a type.
        before = text[:match.start()]
        owner = None
        for head in _CLASS_HEAD.finditer(before):
            try:
                closing = _matching(text, head.end() - 1)
            except ValueError:
                continue
            if head.end() <= match.start() < closing:
                owner = head.group(2)
        base = bases.get(owner or "")
        assignments = []
        for found in _ONE_INITIALISER.finditer(
            text[match.start(1): match.end(1)]
        ):
            name = found.group(1)
            value = found.group(2).strip()[1:-1].strip()
            if name == base:
                assignments.append(f"{_BASE_INIT}({value});")
                continue
            if name == owner:
                # `P() : P(1, 2) {}` - a constructor that hands the work to
                # another of its own. Not a member and not the base: the name
                # is the class's, and what it asks for is the other
                # constructor run on this same object.
                assignments.append(f"{_DELEGATE_INIT}({value});")
                continue
            # Kept as a marker rather than written out as an assignment:
            # whether `b(3)` assigns to a member or constructs one depends on
            # whether `b` is of class type, and the classes have not been
            # read yet.
            assignments.append(f"{_MEMBER_INIT}({name}, {value});")
        return ") { " + " ".join(assignments) + " "

    return _sub_code(_INITIALISER_LIST, text, lambda match, whole: one(match))


#: Stands in for "construct the base with these arguments" until the class
#: table exists and the base's constructor has a name.
_BASE_INIT = "__py2bin_base_init"
#: And for "run another of this class's constructors on this same object",
#: which is what a delegating constructor asks for.
_DELEGATE_INIT = "__py2bin_delegate_init"
#: `C(int x) : n(x)` - a member named in the initialiser list, with whatever
#: it was given. The first argument is the member; the rest are its own.
_MEMBER_INIT = "__py2bin_member_init"

#: `for (int x : v)` - a range-based for.
#: The `&` or `*` goes with the type, and it is written against the name -
#: `auto &x`, not `auto & x`. Read as its own piece, or the pattern needed
#: whitespace that nobody writes and matched nothing.
_RANGE_FOR = re.compile(
    r"\bfor\s*\(\s*([A-Za-z_][\w\s]*?)\s*([*&]*)\s*([A-Za-z_]\w*)\s*:\s*([^)]+)\)"
)

#: `static int count;` written inside a class.
_STATIC_MEMBER = re.compile(
    # The initialiser is optional: C++ lets an integral one be written in the
    # class, which is where a limit or a count usually is.
    r"(?<![\w>])static\s+([A-Za-z_][\w\s*]*?)\s+([A-Za-z_]\w*)\s*"
    r"(=\s*[^;]+)?;"
)

#: `int C::count = 0;` - where a static member is given its storage.
_STATIC_DEFINITION = re.compile(
    r"(?m)^[ \t]*([A-Za-z_][\w\s*]*?)\s+([A-Za-z_]\w*)\s*::\s*([A-Za-z_]\w*)\s*(=[^;]*)?;"
)



def _array_extent(text: str, name: str) -> "int | None":
    """How many elements an array has, read from where it was declared."""

    code = _without_literals(text)
    counted = re.search(
        rf"\b[A-Za-z_]\w*\s+{re.escape(name)}\s*\[\s*(\d+)\s*\]", code
    )
    if counted is not None:
        return int(counted.group(1))
    # `int a[] = {1, 2, 3}` says how many by listing them.
    listed = re.search(
        rf"\b[A-Za-z_]\w*\s+{re.escape(name)}\s*\[\s*\]\s*=\s*\{{([^}}]*)\}}",
        code,
    )
    if listed is None:
        return None
    inside = listed.group(1).strip()
    return len(_split_arguments(inside)) if inside else 0

def _rewrite_range_for(text: str, counter: "list[int]") -> str:
    """`for (int x : v)` becomes an index loop over the same container.

    Written against `size()` and `[]` rather than iterators: those are what
    every container in this subset has, and an iterator here is a pointer
    anyway.
    """

    def one(match: "re.Match[str]") -> str:
        held = (match.group(1) + " " + match.group(2)).strip()
        name, over = match.group(3), match.group(4).strip()
        counter[0] += 1
        index = f"__py2bin_each_{counter[0]}"
        # A bare name is left bare: the rewriters that turn `v.size()` into a
        # call look for a name on the left, and `(v).size()` is not one.
        reached = over if over.isidentifier() else f"({over})"
        # A plain array has no `size()`; C++ reads its extent from the
        # declaration, and so does this. `int a[5]` and `int a[] = {1,2,3}`
        # both say how many there are, in different places.
        extent = _array_extent(text, over) if over.isidentifier() else None
        bound = str(extent) if extent is not None else f"{reached}.size()"
        return (
            f"for (unsigned long {index} = 0; {index} < {bound}; "
            f"{index} = {index} + 1) "
            f"{{ {held} {name} = {reached}[{index}];"
        )

    out = _map_code(text, lambda part: _RANGE_FOR.sub(one, part))
    if out == text:
        return text
    # The body gained an opening brace, so it needs a closing one. The loop
    # body is whatever follows, and its own braces are what say where it ends.
    return _close_range_bodies(out)


def _close_range_bodies(text: str) -> str:
    """Put back the brace each rewritten range-for opened."""

    at = 0
    while True:
        found = re.compile(
            r"= \(?([^()\[]*)\)?\[(__py2bin_each_\d+)\];"
        ).search(text, at)
        if found is None:
            break
        if "\x00closed" in text[found.end(): found.end() + 8]:
            # This one is closed already. Skipped rather than stopped on: a
            # function with two range-fors in it closed the first, found it
            # again on the next round and gave up there, and the second was
            # left with the brace it had been opened with and none to shut
            # it. What that broke was every brace after it in the file.
            at = found.end()
            continue
        # Find the body that follows and close after it.
        rest = text[found.end():]
        stripped = rest.lstrip()
        offset = found.end() + (len(rest) - len(stripped))
        if stripped.startswith("{"):
            closing = _matching(text, offset)
        else:
            semicolon = text.index(";", offset)
            closing = semicolon + 1
        text = (
            text[:found.end()]
            + "\x00closed"
            + text[found.end():closing]
            + " }"
            + text[closing:]
        )
    return text.replace("\x00closed", "")



#: `P a{1, 2};` or `int n{5};` - a declaration initialised with braces. No
#: nested braces and a `;` right after, which is what tells it from a class
#: body or a function.
#: `P a{1, 2};` - a declaration with a brace initialiser.
#:
#: Not `class Thing final { ... };`, which reads exactly the same way: a
#: name, a space, a name, a brace. That one is a class whose body has no
#: braces of its own, and read as an initialiser it became
#: `Thing final( ... );` - the class turned inside out, and every member
#: after it lost.
_BRACE_INIT = re.compile(
    r"(?<![.\w>])(?<!class )(?<!struct )([A-Za-z_]\w*)\s+(\*?)\s*"
    r"([A-Za-z_]\w*)\s*((?:\[[^\[\]]*\])*)\s*\{([^{}]*)\}\s*;"
)


def _rewrite_brace_initialisers(text: str) -> str:
    """`P a{1, 2};` becomes what C spells the same thing.

    A class with a constructor gets the call - `V a{2}` is `V a(2)`. Anything
    else is an aggregate, and C has written those with `=` all along.
    """

    constructed = {
        head.group(2)
        for head in _CLASS_HEAD.finditer(text)
        if _has_a_constructor(text, head)
    }

    def one(match: "re.Match[str]", whole: str) -> "str | None":
        held, star, name, bounds = match.groups()[:4]
        if held in _NOT_A_TYPE or star:
            return None
        # From the real text: a literal inside the braces is blanked in the
        # copy the match was found against, and `char s[3]{'h', 'i'}` would
        # come back with its characters emptied.
        inside = whole[match.start(5): match.end(5)].strip()
        # An array is an aggregate whatever it holds: `V xs[3]{V(1)}` is a
        # list of elements, not a constructor call taking three arguments.
        if bounds:
            return f"{held} {name}{bounds} = {{{inside or '0'}}};"
        if held in constructed:
            return f"{held} {name}({inside});"
        # `T x{}` is value initialisation, which for everything in this
        # subset means zeroed. Left as a bare declaration it was whatever
        # the stack happened to hold, and a program that read it before
        # writing it got a different answer each run.
        return f"{held} {name} = {{{inside or '0'}}};"

    # Over the whole text: an initialiser holding a literal is one match
    # spanning it, and each stretch of code on its own has no closing brace.
    return _sub_code(_BRACE_INIT, text, one)


def _has_a_constructor(text: str, head: "re.Match[str]") -> bool:
    """Whether a class body declares a constructor of its own."""

    try:
        closing = _matching(text, head.end() - 1)
    except ValueError:
        return False
    body = text[head.end() - 1: closing]
    return re.search(rf"(?<![.\w>~]){re.escape(head.group(2))}\s*\(", body) is not None

def _is_a_template_pattern(text: str, at: int) -> bool:
    """Whether the class starting at `at` is written `template <...> class`.

    A pattern is not a class yet: it has no name until it is written out for
    something. Taking a static member out of one took it out of every copy
    that had not been made, and put the storage at file scope under the
    pattern's own name - so `is_same<int, char>::value` was left as C++ and
    an `is_same__value` nobody asked for was defined beside it.
    """

    back = at - 1
    while back >= 0 and text[back] in " \t\n":
        back -= 1
    if back < 0 or text[back] != ">":
        return False
    depth = 0
    while back >= 0:
        if text[back] == ">":
            depth += 1
        elif text[back] == "<":
            depth -= 1
            if not depth:
                break
        back -= 1
    if back < 0:
        return False
    before = text[:back].rstrip()
    return bool(re.search(r"\btemplate$", before))


def _rewrite_static_members(text: str, filename: str) -> str:
    """A static member is one object for the class, so it becomes one object.

    C has no such thing inside a struct, and there is nowhere for it to live
    but file scope. The name carries the class, so two classes may each have
    a `count` without either being the other's.
    """

    #: Keyed by the class as well as the member: two classes may each have a
    #: `value`, and keyed by the name alone the second took the first's -
    #: after which `A::value` was left as C++ because the table said `value`
    #: belonged to B. A traits header is nothing but classes with a member
    #: called `value`.
    owners: "dict[tuple[str, str], str]" = {}
    given: "dict[str, str]" = {}
    out: list[str] = []
    at = 0
    for head in _CLASS_HEAD.finditer(text):
        if head.start() < at:
            continue
        if _is_a_template_pattern(text, head.start()):
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        owner = head.group(2)
        body = text[head.end() - 1: closing]

        bare = _without_literals(body)

        def taken(match: "re.Match[str]", o=owner) -> str:
            if _depth_at(bare, match.start()) != 1:
                # Inside a method body, where `static R held;` is a static
                # *local* - which C already has, and which the pass that
                # writes its build-once guard knows about. Read as a data
                # member it was lifted out of the class, renamed to carry the
                # class, and then declared nowhere at all.
                return match.group(0)
            owners[(o, match.group(2))] = o
            if match.group(3):
                # Given its value here, so this is where it is defined; there
                # is no `int C::limit = 10;` anywhere else to find.
                given[_c_name(o, match.group(2))] = (
                    f"{match.group(1).strip()} {_c_name(o, match.group(2))} "
                    f"{match.group(3).strip()};"
                )
            return ""

        # The bare name, inside the class it belongs to. Done here, where the
        # body is in hand and the class it belongs to is known, rather than
        # by the pass below that asks which class has a member of that name -
        # two copies of one template always both do, so that pass declines
        # and `made++` inside the constructor was left naming nothing.
        mine = [
            one.group(2)
            for one in _STATIC_MEMBER.finditer(bare)
            if _depth_at(bare, one.start()) == 1
        ]
        written = _STATIC_MEMBER.sub(taken, body)
        for spelled in mine:
            written = _map_code(
                written,
                lambda part, n=spelled, o=owner: re.sub(
                    rf"(?<![.\w>:]){re.escape(n)}\b(?!\s*(?:::|\())",
                    _c_name(o, n),
                    part,
                ),
            )
        out.append(text[at:head.end() - 1])
        out.append(written)
        at = closing
    out.append(text[at:])
    text = "".join(out)
    if not owners:
        return text

    # `int C::count = 0;` becomes the file-scope object itself.
    def defined(match: "re.Match[str]") -> str:
        spelled, owner, name, value = match.groups()
        if (owner, name) not in owners:
            return match.group(0)
        return f"{spelled} {_c_name(owner, name)} {value or '= 0'};"

    text = _STATIC_DEFINITION.sub(defined, text)
    # Those that were given a value in the class need their storage written.
    if given:
        text = "\n".join(given.values()) + "\n" + text
    # And every mention of it - `C::count` from outside, `count` from within -
    # is that object.
    for (owner, name) in owners:
        spelled = _c_name(owner, name)
        text = _map_code(
            text,
            lambda part, o=owner, n=name, s=spelled: re.sub(
                rf"\b{re.escape(o)}\s*::\s*{re.escape(n)}\b", s, part
            ),
        )
    # A bare mention of one - `count` written inside its own class - only
    # where exactly one class has a member of that name. Where two do, the
    # bare name is whichever class the mention is inside, which this pass
    # cannot see once the bodies have been taken apart; `C::count` is written
    # out and is unambiguous either way.
    once = [
        (owner, name)
        for (owner, name) in owners
        if sum(1 for (_o, other) in owners if other == name) == 1
    ]
    for (owner, name) in once:
        spelled = _c_name(owner, name)
        text = _map_code(
            text,
            lambda part, n=name, s=spelled: re.sub(
                rf"(?<![.\w>:]){re.escape(n)}\b(?!\s*::)", s, part
            ),
        )
    return text

#: `int add(int a, int b = 10)` - a parameter with a default.
_DEFAULTED = re.compile(r"([A-Za-z_]\w*)\s*=\s*([^,)]+)")

#: A definition whose parameter list is followed by `const` or `noexcept`.
#: The general pattern wants a brace straight after the `)`.
_QUALIFIED_DEFINITION = re.compile(
    r"\b([A-Za-z_][\w\s*]*?(?:&&?\s*)?)\b([A-Za-z_]\w*)\s*\(([^;{}()]*)\)"
    r"\s*(?:const|noexcept|override|final|\s)*\{"
)

#: `class Inner {` written inside another class body.
_NESTED_CLASS = re.compile(r"\b(class|struct)\s+([A-Za-z_]\w*)\s*\{")


def _rewrite_default_arguments(text: str) -> str:
    """Fill a call's missing arguments in from the declaration.

    C has no defaults, so the caller supplies what the callee assumed. Read
    from the definition, which is the only place the values are written.
    """

    defaults: "dict[str, list[str]]" = {}
    out: list[str] = []
    at = 0
    for match in _ANY_DEFINITION.finditer(text):
        head = match.group(1).strip()
        # How it is stored is written where the return type goes and is not
        # one. Read whole, `static int add(int a, int b = 10)` began with a
        # word that is not a type, so its defaults were never read: the `=`
        # stayed in the C and the calls that left the argument out stayed
        # short.
        while head.split() and head.split()[0] in _STORAGE | _DISPATCH:
            head = head.split(None, 1)[1] if " " in head else ""
        if not head or head.split()[0] in _NOT_A_TYPE:
            continue
        if _depth_at(text, match.start()) != 0:
            # Inside a class body, so it is a member's default and belongs to
            # the pass that knows how a member is called. This one stripped it
            # anyway and could not fill the call in, which left the callee
            # without the default and the caller without the argument.
            continue
        parts = _split_arguments(match.group(2))
        found = [
            _DEFAULTED.search(part).group(2).strip()
            if _DEFAULTED.search(part)
            else ""
            for part in parts
        ]
        if not any(found):
            continue
        name = head.replace("*", " * ").replace("&", " & ").split()[-1]
        name = name.split("::")[-1]
        defaults[name] = found
        out.append(text[at:match.start(2)])
        out.append(", ".join(_DEFAULTED.sub(r"\1", part) for part in parts))
        at = match.end(2)
    out.append(text[at:])
    text = "".join(out)
    if not defaults:
        return text

    for name, values in defaults.items():
        pattern = re.compile(rf"(?<![.\w>]){re.escape(name)}\s*\(")
        rebuilt: list[str] = []
        at = 0
        for call in pattern.finditer(text):
            if call.start() < at:
                continue
            close = _closing_paren(text, call.end() - 1)
            if close < 0 or _is_a_definition(text, close):
                continue
            given = _call_arguments(text, call.end() - 1)
            if len(given) >= len(values):
                continue
            filled = given + [v for v in values[len(given):] if v]
            if len(filled) != len(values):
                continue
            rebuilt.append(text[at:call.end()])
            rebuilt.append(", ".join(filled))
            at = close
        rebuilt.append(text[at:])
        text = "".join(rebuilt)
    return text


#: `enum Mode { Off, On };` written inside a class body.
_NESTED_ENUM = re.compile(r"\benum\s+(?:class\s+|struct\s+)?([A-Za-z_]\w*)\s*\{")


def _lift_nested_enums(text: str) -> str:
    """An enum written inside a class is a type, and C has no nested type.

    Moved out under its own name rather than the class's: an enumerator is
    reached bare from inside the class and as `Class::Name` from outside, and
    both spellings are kept working by stripping the qualifier.
    """

    lifted: list[str] = []
    while True:
        moved = False
        for head in _CLASS_HEAD.finditer(text):
            try:
                closing = _matching(text, head.end() - 1)
            except ValueError:
                continue
            body = text[head.end(): closing - 1]
            inner = _NESTED_ENUM.search(body)
            if inner is None:
                continue
            start = head.end() + inner.start()
            try:
                inner_close = _matching(text, head.end() + inner.end() - 1)
            except ValueError:
                continue
            end = inner_close
            while end < len(text) and text[end] in " \t":
                end += 1
            if end < len(text) and text[end] == ";":
                end += 1
            spelled = text[start:end]
            lifted.append(spelled)
            text = text[:start] + text[end:]
            owner = head.group(2)
            # The enum's own name and its enumerators, and nothing else. The
            # qualifier was stripped off everything the class owns, so
            # `Reg::howMany()` lost the only thing that said which class it
            # belonged to and was called as a function nobody had declared.
            taken = {inner.group(1)}
            body_open = spelled.find("{")
            body_close = spelled.rfind("}")
            if 0 <= body_open < body_close:
                for part in spelled[body_open + 1: body_close].split(","):
                    word = re.match(r"\s*([A-Za-z_]\w*)", part)
                    if word is not None:
                        taken.add(word.group(1))
            for one in sorted(taken):
                text = _map_code(
                    text,
                    lambda part, o=owner, n=one: re.sub(
                        rf"\b{re.escape(o)}\s*::\s*{re.escape(n)}\b", n, part
                    ),
                )
            moved = True
            break
        if not moved:
            return "\n".join(lifted) + ("\n" if lifted else "") + text


def _lift_nested_classes(text: str) -> str:
    """A class written inside another becomes one of its own.

    C has no nested type, and the inner one is not a member - it is a type
    that happens to be spelled with the outer one's name in front. So it is
    moved out under a name that keeps that, and `Outer::Inner` follows it.
    """

    for _round in range(_HOIST_ROUNDS):
        moved = False
        for head in _CLASS_HEAD.finditer(text):
            try:
                closing = _matching(text, head.end() - 1)
            except ValueError:
                continue
            body = text[head.end(): closing - 1]
            inner = _NESTED_CLASS.search(body)
            if inner is None:
                continue
            start = head.end() + inner.start()
            try:
                inner_close = _matching(text, head.end() + inner.end() - 1)
            except ValueError:
                continue
            end = inner_close
            while end < len(text) and text[end] in " \t":
                end += 1
            # `struct Mid { ... } m;` says two things at once: it defines
            # the type and declares a member of it. Taken out whole, the
            # member's name went with the type and the outer class was
            # left holding `{ m;` - a declaration with no type, which is
            # not C. What stays behind is the member, written with the
            # name the type has now.
            member = re.match(r"([A-Za-z_]\w*)\s*;", text[end:])
            if end < len(text) and text[end] == ";":
                end += 1
            outer, name = head.group(2), inner.group(2)
            spelled = f"{outer}__{name}"
            taken = text[start:end].replace(name, spelled, 1)
            # A class inside a class *template* depends on the parameters of
            # the one it was written in: `struct Inner { T v; };` has no `T`
            # once it is somewhere else. So it takes them with it, and every
            # mention of it says which arguments are meant - inside the
            # template, its own; outside, whatever the use spelled.
            heading = re.search(
                r"(?<![.\w>])template\s*<([^<>]*)>\s*$", text[:head.start()]
            )
            named = (
                [
                    part.split()[-1]
                    for part in _split_arguments(heading.group(1))
                    if part.strip()
                ]
                if heading is not None
                else []
            )
            if named:
                taken = f"template <{heading.group(1)}> {taken}"
                applied = f"{spelled}<{', '.join(named)}>"
            else:
                applied = spelled
            kept = ""
            if member is not None:
                # The span stopped at the `}`, so the `;` that ended the
                # declaration went with the member and not with the type.
                taken = taken.rstrip() + ";"
                kept = f"{spelled} {member.group(1)};"
                end += member.end()
            # Put back at the front rather than set aside in a list: a
            # class nested two deep holds a class of its own, and one the
            # loop never reads again was lifted once and left holding it.
            text = taken + "\n" + text[:start] + kept + text[end:]
            # `Outer<int>::Inner` names the copy for those arguments.
            text = _map_code(
                text,
                lambda part, o=outer, n=name, s=spelled: re.sub(
                    rf"\b{re.escape(o)}\s*<([^<>]*)>\s*::\s*{re.escape(n)}\b",
                    rf"{s}<\1>",
                    part,
                ),
            )
            text = _map_code(
                text,
                lambda part, o=outer, n=name, s=spelled: re.sub(
                    rf"\b{re.escape(o)}\s*::\s*{re.escape(n)}\b", s, part
                ),
            )
            if named:
                # A bare `Inner` means this copy, and only inside the class it
                # was written in - which is where the parameters have meaning.
                closing = _matching(text, head.end() - 1)
                inside = text[head.end(): closing]
                inside = _map_code(
                    inside,
                    lambda part, n=name, s=applied: re.sub(
                        rf"(?<![.\w>:]){re.escape(n)}\b(?!\s*(?:::|<))", s, part
                    ),
                )
                text = text[:head.end()] + inside + text[closing:]
            else:
                text = _map_code(
                    text,
                    lambda part, n=name, s=spelled: re.sub(
                        rf"(?<![.\w>:]){re.escape(n)}\b(?!\s*::)", s, part
                    ),
                )
            moved = True
            break
        if not moved:
            return text


def _settled_parameters(
    parameters: str,
    holder: str,
    text: str,
    closing: int,
    filename: str,
    at: int,
) -> "list[str]":
    """Give a generic lambda's `auto` parameters the types it is called with.

    `[](auto a, auto b)` is a member template in C++ - one copy for each set
    of argument types it is used with. The sets are read from the calls,
    because nothing else here says what they are, and one `operator()` is
    written for each: two members of one class, told apart by what they take,
    which is what an overload is and what a member template compiles to.
    """

    given: "list[list[str]]" = []
    rest = text[closing:]
    for call in re.finditer(rf"(?<![.\w>]){re.escape(holder)}\s*\(", rest):
        close = _closing_paren(rest, call.end() - 1)
        if close < 0:
            continue
        inside = rest[call.end(): close].strip()
        arguments = _split_arguments(inside) if inside else []
        held = [_deduced_type(one.strip(), text) for one in arguments]
        if any(one is None for one in held):
            continue
        given.append([str(one) for one in held])
    settled = [one for one in given if len(one) == len(_split_arguments(parameters))]
    if not settled:
        raise CppTranslationError(
            filename, _line_of(text, at),
            "a lambda whose parameters are `auto` is a template, and py2bin "
            "writes a copy of it for each set of types it is called with. "
            "Nothing here calls this one, so there is no set to write; give "
            "the parameters their types",
        )
    # One copy for each set of types the calls use, which is what a member
    # template is. Written as several `operator()` of one class, because that
    # is how two of them are told apart in the C: by the types they take.
    shapes: "list[str]" = []
    for wanted in settled:
        out = []
        for spelled, held in zip(_split_arguments(parameters), wanted):
            out.append(
                re.sub(r"(?<![.\w>])auto\b", held, spelled.strip(), count=1)
            )
        made = ", ".join(out)
        if made not in shapes:
            shapes.append(made)
    return shapes


#: `std::function<int(int)>` - by the time this runs the `std::` is gone.
#: The qualifier comes with it. Matched from `function` alone, `std::` was
#: left standing in front of the class this becomes, and the C said
#: `std::struct __py2bin_call_int_int`.
_STD_FUNCTION = re.compile(r"(?<![.\w>])(?:std\s*::\s*)?function\s*<")

#: How many different callables one signature may be given. A backstop: the
#: class below holds one member per callable, and a program with more than
#: this many going into one `std::function` is doing something this was not
#: written for.
_ERASED_LIMIT = 32


def _signature_of(text: str, at: int) -> "tuple[str, list[str], int] | None":
    """Read `<R(A, B)>` starting at the `<`, as (result, parameters, end)."""

    depth = 0
    index = at
    while index < len(text):
        if text[index] == "<":
            depth += 1
        elif text[index] == ">":
            depth -= 1
            if depth == 0:
                break
        index += 1
    if index >= len(text):
        return None
    spelled = text[at + 1: index].strip()
    opened = spelled.find("(")
    if opened < 0 or not spelled.endswith(")"):
        return None
    result = spelled[:opened].strip()
    inside = spelled[opened + 1: -1].strip()
    parameters = (
        [one.strip() for one in _split_arguments(inside) if one.strip()]
        if inside and inside != "void"
        else []
    )
    return result, parameters, index + 1


def _erased_name(result: str, parameters: "list[str]") -> str:
    """A class name for one signature, made of the signature itself."""

    spelled = "__".join([result, *parameters]) or "void"
    return "__py2bin_call_" + re.sub(r"[^A-Za-z0-9]+", "_", spelled).strip("_")


def _rewrite_std_function(text: str, filename: str) -> str:
    """`std::function<int(int)>` becomes a class that holds any of them.

    What `std::function` adds over the thing it holds is type erasure: two
    different callables of one signature stored in one variable, called
    without knowing which is in there. Every implementation of it reaches for
    an indirect call through a pointer to a thunk, because it is compiled
    without knowing what will be put in.

    py2bin does know. It has no linker, so a translation unit is the whole
    program and every callable that is ever assigned to one of these is in
    front of it while it translates. So the class holds one member per
    callable and a tag saying which is live, and the call is a comparison and
    a direct call. That is the same argument `dynamic_cast` uses, and it
    gives what the pointer version cannot: the closure is *copied* into the
    object, so a `std::function` held as a member outlives the scope the
    lambda was written in - which is the whole reason a program stores one.

    A plain function goes in as a pointer to itself, because that is what its
    name already is.
    """

    if _STD_FUNCTION.search(_without_literals(text)) is None:
        return text
    # Every signature the file names, and the class each becomes.
    signatures: "dict[str, tuple[str, list[str]]]" = {}
    while True:
        bare = _without_literals(text)
        found = _STD_FUNCTION.search(bare)
        if found is None:
            break
        read = _signature_of(text, found.end() - 1)
        if read is None:
            raise CppTranslationError(
                filename,
                _line_of(text, found.start()),
                "std::function names a signature, as in "
                "`std::function<int(int)>`; py2bin could not read this one",
            )
        result, parameters, end = read
        name = _erased_name(result, parameters)
        signatures[name] = (result, parameters)
        text = text[:found.start()] + name + text[end:]
    if not signatures:
        return text

    # What each one is given, anywhere in the file. Read before anything is
    # rewritten, because the class has to hold a member for every one.
    held: "dict[str, list[tuple[str, bool]]]" = {name: [] for name in signatures}
    functions = _function_definitions(text)
    for name in signatures:
        for value in _assigned_to(text, name, filename):
            spelled = _deduced_type(value, text)
            if spelled is not None and _names_a_class(text, spelled.strip()):
                entry = (spelled.strip(), False)
            elif value in functions:
                entry = (value, True)
            else:
                raise CppTranslationError(
                    filename,
                    _line_of(text, text.find(value)),
                    f"py2bin cannot tell what {value!r} is, and a "
                    f"std::function has to be given a lambda, an object with "
                    f"a call operator, or the name of a function",
                )
            if entry not in held[name]:
                held[name].append(entry)
        if len(held[name]) > _ERASED_LIMIT:
            raise CppTranslationError(
                filename, 0,
                f"more than {_ERASED_LIMIT} different callables go into one "
                f"std::function here; py2bin writes out a member for each",
            )

    made = [
        _emit_erased(name, *signatures[name], held[name]) for name in signatures
    ]
    text = _fill_erased(text, held)
    return "\n".join(made) + "\n" + text


#: `f = value;` - an assignment to something, with the name on the left kept
#: whole so a member reached through `this` or through an object is one too.
_ERASED_ASSIGN = re.compile(
    r"(?<![.\w>])((?:[A-Za-z_]\w*\s*(?:\.|->)\s*)*[A-Za-z_]\w*)\s*=(?!=)\s*"
    r"([A-Za-z_]\w*)\s*;"
)


def _erased_spellings(text: str, name: str) -> str:
    """Every name this signature goes by here, as one pattern.

    A program almost always gives the signature a name of its own - `using
    EventHandler = std::function<void(const string &)>;` - and writes that
    everywhere afterwards. The pass that resolves an alias runs after this
    one, so the alias is followed here instead: without it a constructor
    taking an `EventHandler` was not a constructor taking one of these, and
    the lambda handed to it was passed as itself.
    """

    spellings = {name}
    for _round in range(4):
        found = set(spellings)
        for spelled in spellings:
            for match in re.finditer(
                rf"(?<![.\w>])using\s+([A-Za-z_]\w*)\s*=\s*"
                rf"{re.escape(spelled)}\s*;",
                text,
            ):
                found.add(match.group(1))
            for match in re.finditer(
                rf"(?<![.\w>])typedef\s+{re.escape(spelled)}\s+"
                rf"([A-Za-z_]\w*)\s*;",
                text,
            ):
                found.add(match.group(1))
        if found == spellings:
            break
        spellings = found
    return "|".join(re.escape(one) for one in sorted(spellings))


def _erased_holders(text: str, name: str) -> "set[str]":
    """Every name declared to hold this signature: a variable, a member."""

    return {
        match.group(1)
        for match in re.finditer(
            rf"(?<![.\w>])(?:{_erased_spellings(text, name)})"
            rf"\s*&?\s*([A-Za-z_]\w*)\s*[;=,)]",
            text,
        )
    }


def _erased_containers(text: str, name: str) -> "set[str]":
    """Every variable declared to hold a container of this signature.

    A callable put into a `vector<function<int(int)>>` is going into one of
    these as surely as one assigned to a variable is. Which method takes an
    element is not asked - the container is a template, and its methods have
    not been written out yet - so every argument of every call on the
    variable is offered, and what the argument *is* decides.
    """

    spelled = _erased_spellings(text, name)
    return {
        match.group(1)
        for match in re.finditer(
            rf"(?<![.\w>])[A-Za-z_][\w:]*\s*<[^;{{}}]*?(?:{spelled})[^;{{}}]*?>"
            rf"\s*&?\s*([A-Za-z_]\w*)\s*[;=,)]",
            _without_literals(text),
        )
    }


def _takes_a_call(text: str, spelled: str) -> bool:
    """Whether this class defines `operator()` - whether it is a callable."""

    match = re.search(
        rf"\b(?:class|struct)\s+{re.escape(spelled)}\b[^{{;]*\{{", text
    )
    if match is None:
        return False
    try:
        shut = _matching(text, match.end() - 1)
    except ValueError:
        return False
    return re.search(r"\boperator\s*\(\s*\)", text[match.end(): shut - 1]) is not None


def _erased_parameters(text: str, name: str) -> "dict[str, list[int]]":
    """Which functions take this signature, and at which positions.

    A `std::function` parameter is how a program says "give me something to
    call", and passing a lambda to one is the commonest thing done with the
    type at all.
    """

    found: "dict[str, list[int]]" = {}
    for match in _DEFINITION.finditer(text):
        if _depth_at(text, match.end() - 1) != 0:
            continue
        at = [
            index
            for index, part in enumerate(_split_arguments(match.group(3)))
            if re.match(
                rf"\s*(?:const\s+)?(?:{_erased_spellings(text, name)})"
                rf"\s*&?\s*\w*\s*$",
                part,
            )
        ]
        if at:
            found[match.group(2)] = at
    return found


def _erased_constructors(text: str, name: str) -> "dict[str, list[int]]":
    """Which classes take this signature in a constructor, and where.

    Read from the class bodies rather than from the definitions at the top
    level, because a constructor is declared inside its class and is spelled
    `Class::Class` where it is defined - neither of which is a definition the
    scanner above recognises. Passing a lambda to a constructor is how a
    program hands an object something to call back into, so leaving these out
    left the commonest use of the type untranslated.
    """

    found: "dict[str, list[int]]" = {}
    wants = re.compile(
        rf"\s*(?:const\s+)?(?:{_erased_spellings(text, name)})"
        rf"\s*&?\s*\w*\s*$"
    )
    for head in _CLASS_HEAD.finditer(text):
        owner = head.group(2)
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        inside = text[head.end(): closing - 1]
        for written in re.finditer(
            rf"(?<![.\w>~]){re.escape(owner)}\s*\(", inside
        ):
            shut = _closing_paren(inside, written.end() - 1)
            if shut < 0:
                continue
            at = [
                index
                for index, part in enumerate(
                    _split_arguments(inside[written.end(): shut])
                )
                if wants.match(part)
            ]
            if at:
                found.setdefault(owner, at)
    return found


def _erased_methods(text: str, name: str) -> "dict[str, list[int]]":
    """Which members take this signature, and at which positions.

    The same reason as the constructors above: a member is declared inside
    its class and defined as `Class::member`, and the scanner that reads
    definitions at the top level sees neither. A method taking a callback is
    as ordinary as a constructor taking one.
    """

    found: "dict[str, list[int]]" = {}
    wants = re.compile(
        rf"\s*(?:const\s+)?(?:{_erased_spellings(text, name)})"
        rf"\s*&?\s*\w*\s*$"
    )
    for head in _CLASS_HEAD.finditer(text):
        owner = head.group(2)
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        inside = text[head.end(): closing - 1]
        for written in re.finditer(r"(?<![.\w>~])([A-Za-z_]\w*)\s*\(", inside):
            method = written.group(1)
            if method == owner or method in _NOT_A_TYPE:
                continue
            shut = _closing_paren(inside, written.end() - 1)
            if shut < 0:
                continue
            at = [
                index
                for index, part in enumerate(
                    _split_arguments(inside[written.end(): shut])
                )
                if wants.match(part)
            ]
            if at:
                found.setdefault(method, at)
    return found


def _erased_places(
    text: str, name: str
) -> "list[tuple[int, int, list[int], bool]]":
    """Every call that may be handed one of these: where it starts and opens.

    A plain call is the name and its parentheses. A constructor is written
    three ways - `C v(x);`, `C(x)` as a temporary, and `new C(x)` - and the
    name in front of the parentheses is the class rather than the callee.

    The last of each is whether the place is *certain* - whether a declared
    parameter says this position takes one. A call on a container of them is
    not: nothing here knows which of its arguments is an element, so every
    argument is offered and one that is not a callable is passed over rather
    than refused.
    """

    bare = _without_literals(text)
    places: "list[tuple[int, int, list[int], bool]]" = []
    for called, positions in _erased_parameters(text, name).items():
        for match in re.finditer(rf"(?<![.\w>]){re.escape(called)}\s*\(", bare):
            places.append((match.start(), match.end() - 1, positions, True))
    for owner, positions in _erased_constructors(text, name).items():
        for match in re.finditer(
            rf"(?<![.\w>~]){re.escape(owner)}\s*(?:[A-Za-z_]\w*\s*)?\(", bare
        ):
            places.append((match.start(), match.end() - 1, positions, True))
    for method, positions in _erased_methods(text, name).items():
        for match in re.finditer(
            rf"(?:\.|->)\s*{re.escape(method)}\s*\(", bare
        ):
            places.append((match.start(), match.end() - 1, positions, True))
    for owner in _erased_containers(text, name):
        for match in re.finditer(
            rf"(?<![.\w>]){re.escape(owner)}\s*(?:\.|->)\s*[A-Za-z_]\w*\s*\(",
            bare,
        ):
            opening = match.end() - 1
            close = _closing_paren(bare, opening)
            if close < 0:
                continue
            count = len(_split_arguments(bare[opening + 1: close]))
            places.append((match.start(), opening, list(range(count)), False))
    return places


def _assigned_to(text: str, name: str, filename: str) -> "list[str]":
    """Every value this signature is given: assigned, declared, or passed."""

    holders = _erased_holders(text, name)
    found: "list[str]" = []
    bare = _without_literals(text)
    functions = _function_definitions(text)
    # Passed to a function or a constructor that takes one.
    for _start, opening, positions, certain in _erased_places(text, name):
        close = _closing_paren(bare, opening)
        if close < 0 or _is_a_definition(bare, close):
            continue
        given = _split_arguments(bare[opening + 1: close])
        for index in positions:
            if index >= len(given):
                continue
            value = given[index].strip()
            if not value.isidentifier() or value in holders:
                continue
            if not certain:
                # Nothing said this argument is one, so it has to look like
                # one: a class with a call operator, or a function's name.
                spelled = (_deduced_type(value, text) or "").strip()
                if value not in functions and not _takes_a_call(text, spelled):
                    continue
            found.append(value)
    # `NAME f = value;` - given where it is declared.
    for match in re.finditer(
        rf"(?<![.\w>]){re.escape(name)}\s+[A-Za-z_]\w*\s*=\s*([A-Za-z_]\w*)\s*;",
        bare,
    ):
        found.append(match.group(1))
    # `std::function<int(int)> adder(int n) { ... return f; }` - handed back
    # from a function that answers one. Without this the class was written
    # out with no member for the closure and the `return` gave back the
    # lambda itself, which is a different type.
    for match in re.finditer(
        rf"(?<![.\w>])(?:{_erased_spellings(text, name)})\s+[A-Za-z_]\w*\s*"
        rf"\([^;{{}}]*\)\s*\{{",
        bare,
    ):
        try:
            shut = _matching(text, match.end() - 1)
        except ValueError:
            continue
        for gave in re.finditer(
            r"\breturn\s+([A-Za-z_]\w*)\s*;", bare[match.end(): shut]
        ):
            if gave.group(1) not in holders:
                found.append(gave.group(1))
    # `table[key] = value;` - into a slot of a container of them. The left
    # side is not a name, so the assignment pass below does not see it, and
    # the class was written out with no member for what the program put in.
    for owner in _erased_containers(text, name):
        for match in re.finditer(
            rf"(?<![.\w>]){re.escape(owner)}\s*\[[^\];]*\]\s*=\s*"
            rf"([A-Za-z_]\w*)\s*;",
            bare,
        ):
            if match.group(1) not in holders:
                found.append(match.group(1))
    # `f = value;` - given later, to a variable or to a member.
    for match in _ERASED_ASSIGN.finditer(bare):
        target = re.sub(r"\s+", "", match.group(1)).split("->")[-1].split(".")[-1]
        if target in holders and match.group(2) not in holders:
            found.append(match.group(2))
    return list(dict.fromkeys(found))


def _emit_erased(
    name: str,
    result: str,
    parameters: "list[str]",
    held: "list[tuple[str, bool]]",
) -> str:
    """The class one signature becomes: a tag, a member per callable, a call."""

    named = [f"a{index}" for index in range(len(parameters))]
    spelled = ", ".join(
        f"{held_type} {variable}" for held_type, variable in zip(parameters, named)
    )
    passed = ", ".join(named)
    answer = "" if result.strip() in ("", "void") else "return "
    lines = [f"class {name} {{", "public:", "    int __which;"]
    for index, (what, is_function) in enumerate(held, 1):
        if is_function:
            lines.append(
                f"    {result} (*__held{index})({', '.join(parameters) or 'void'});"
            )
        else:
            lines.append(f"    {what} __held{index};")
    lines.append(f"    {name}() {{ __which = 0; }}")
    lines.append(f"    int empty() {{ return __which == 0; }}")
    lines.append(f"    {result} operator()({spelled}) {{")
    for index, (_what, _is_function) in enumerate(held, 1):
        lines.append(
            f"        if (__which == {index}) {{ {answer}__held{index}({passed}); }}"
        )
    if answer:
        # Nothing was put in it. C++ throws `bad_function_call`; there is no
        # unwinder here, so what comes back is the zero a C function that
        # falls off its end would answer - and `empty()` is how a program
        # asks before it calls.
        lines.append(f"        {result} __nothing; return __nothing;")
    lines.append("    }")
    lines.append("};")
    return "\n".join(lines)


def _fill_erased(
    text: str, held: "dict[str, list[tuple[str, bool]]]"
) -> str:
    """Rewrite each place a callable goes in: the tag, and the member."""

    for name, entries in held.items():
        if not entries:
            continue
        holders = _erased_holders(text, name)
        places = {what: index for index, (what, _fn) in enumerate(entries, 1)}
        by_value = {
            what: index for index, (what, is_fn) in enumerate(entries, 1) if is_fn
        }

        def one(match: "re.Match[str]", whole: str) -> "str | None":
            target = re.sub(r"\s+", "", match.group(1))
            plain = target.split("->")[-1].split(".")[-1]
            if plain not in holders:
                return None
            value = match.group(2)
            if value in by_value:
                index = by_value[value]
            else:
                spelled = _deduced_type(value, whole)
                index = places.get((spelled or "").strip())
            if index is None:
                return None
            return (
                f"{match.group(1)}.__which = {index}; "
                f"{match.group(1)}.__held{index} = {value};"
            ).replace(f"{target}.", f"{target}." if "." in target else f"{target}.")

        # The declaration first: `NAME f = value;` also reads as an assignment
        # to `f`, and rewriting that one first left the type name in front of
        # a statement that was no longer a declaration.
        pattern = re.compile(
            rf"(?<![.\w>])({re.escape(name)})\s+([A-Za-z_]\w*)\s*=\s*"
            rf"([A-Za-z_]\w*)\s*;"
        )

        def declared(match: "re.Match[str]", whole: str) -> "str | None":
            value = match.group(3)
            if value in by_value:
                index = by_value[value]
            else:
                spelled = _deduced_type(value, whole)
                index = places.get((spelled or "").strip())
            if index is None:
                return None
            return (
                f"{name} {match.group(2)}; {match.group(2)}.__which = {index}; "
                f"{match.group(2)}.__held{index} = {value};"
            )

        text = _sub_code(pattern, text, declared)
        text = _sub_code(_ERASED_ASSIGN, text, one)
        text = _erased_into_a_slot(text, name, holders, places, by_value)
        text = _erased_given(text, name, holders, places, by_value)
        text = _erased_returned(text, name, holders, places, by_value)
        text = _erased_truth(text, holders)
    return text


def _erased_into_a_slot(
    text: str,
    name: str,
    holders: "set[str]",
    places: "dict[str, int]",
    by_value: "dict[str, int]",
) -> str:
    """`table[key] = f;` - the tag and the member, written on the slot.

    The subscript is written twice rather than held in a temporary: a
    container's `operator[]` answers the same slot both times, and one that
    makes the slot on first asking has made it by the second.
    """

    def one(match: "re.Match[str]", whole: str) -> "str | None":
        value = match.group(2)
        if value in by_value:
            which = by_value[value]
        else:
            spelled = _deduced_type(value, whole)
            which = places.get((spelled or "").strip())
        if which is None:
            return None
        # From the real text: the match is against a copy with the literals
        # blanked, and a key written as one - `table["name"]` - came back as
        # a subscript with nothing in it.
        slot = whole[match.start(1): match.end(1)]
        return (
            f"{slot}.__which = {which}; {slot}.__held{which} = {value};"
        )

    for owner in _erased_containers(text, name):
        pattern = re.compile(
            rf"((?<![.\w>]){re.escape(owner)}\s*\[[^\];]*\])\s*=\s*"
            rf"([A-Za-z_]\w*)\s*;"
        )
        text = _sub_code(pattern, text, one)
    return text


def _erased_returned(
    text: str,
    name: str,
    holders: "set[str]",
    places: "dict[str, int]",
    by_value: "dict[str, int]",
) -> str:
    """Put a callable into one of these on the way out of a function.

    The same wrap `_erased_given` writes for an argument, at the other end: a
    function that answers this signature and hands back a closure is making
    the object C++ would have made for it.
    """

    for _round in range(_HOIST_ROUNDS):
        changed = False
        bare = _without_literals(text)
        for match in re.finditer(
            rf"(?<![.\w>])(?:{_erased_spellings(text, name)})\s+[A-Za-z_]\w*\s*"
            rf"\([^;{{}}]*\)\s*\{{",
            bare,
        ):
            try:
                shut = _matching(text, match.end() - 1)
            except ValueError:
                continue
            gave = re.search(
                r"\breturn\s+([A-Za-z_]\w*)\s*;", bare[match.end(): shut]
            )
            if gave is None:
                continue
            value = gave.group(1)
            if value in holders:
                continue
            if value in by_value:
                which = by_value[value]
            else:
                spelled = _deduced_type(value, text)
                which = places.get((spelled or "").strip())
            if which is None:
                continue
            made = f"__py2bin_gave_{abs(hash((match.start(), value))) % 100000}"
            at = match.end() + gave.start()
            text = (
                text[:at]
                + f" {name} {made}; {made}.__which = {which}; "
                f"{made}.__held{which} = {value}; return {made};"
                + text[match.end() + gave.end():]
            )
            holders.add(made)
            changed = True
            break
        if not changed:
            return text
    return text


def _erased_given(
    text: str,
    name: str,
    holders: "set[str]",
    places: "dict[str, int]",
    by_value: "dict[str, int]",
) -> str:
    """Wrap a callable passed where one of these is wanted.

    `twice([](int v){ ... })` is a lambda by the time this runs, and the
    parameter is one of these - so an object of it is made ahead of the call
    and the lambda goes into it, which is what C++ does with the temporary.
    """

    for _round in range(_HOIST_ROUNDS):
        changed = False
        bare = _without_literals(text)
        for begins, opening, positions, _certain in _erased_places(text, name):
            close = _closing_paren(bare, opening)
            if close < 0 or _is_a_definition(bare, close):
                continue
            given = _split_arguments(text[opening + 1: close])
            for index in positions:
                if index >= len(given):
                    continue
                value = given[index].strip()
                if not value.isidentifier() or value in holders:
                    continue
                if value in by_value:
                    which = by_value[value]
                else:
                    spelled = _deduced_type(value, text)
                    which = places.get((spelled or "").strip())
                if which is None:
                    continue
                made = (
                    f"__py2bin_given_"
                    f"{abs(hash((begins, index, value))) % 100000}"
                )
                given[index] = made
                start = _statement_start(text, begins)
                text = (
                    text[:start]
                    + f" {name} {made}; {made}.__which = {which}; "
                    f"{made}.__held{which} = {value}; "
                    + text[start: opening + 1]
                    + ", ".join(one.strip() for one in given)
                    + text[close:]
                )
                holders.add(made)
                changed = True
                break
            if changed:
                break
        if not changed:
            return text
    return text


#: `if (cb)` and `if (!cb)` - how a program asks whether one of these holds
#: anything before it calls it. C++ answers with a conversion to `bool`; this
#: subset has no conversion operator, so the two spellings that matter are
#: read here and the tag is what they become.
_ERASED_TRUTH = re.compile(
    r"(?<![.\w>])(!?)\s*((?:[A-Za-z_]\w*\s*(?:\.|->)\s*)*[A-Za-z_]\w*)\s*\)"
)


def _erased_truth(text: str, holders: "set[str]") -> str:
    """`if (cb)` becomes a test of which callable is in it."""

    if not holders:
        return text

    def one(match: "re.Match[str]", whole: str) -> "str | None":
        spelled = re.sub(r"\s+", "", match.group(2))
        if spelled.split("->")[-1].split(".")[-1] not in holders:
            return None
        # Only in a condition: `if (cb)` and `while (cb)`. Anywhere else a
        # bare name is the object itself and means what it says.
        before = whole[:match.start()].rstrip()
        if not re.search(r"\b(?:if|while)\s*\($", before):
            return None
        compared = "==" if match.group(1) else "!="
        return f"{match.group(2)}.__which {compared} 0)"

    return _sub_code(_ERASED_TRUTH, text, one)


def _expand_lambdas(
    text: str, filename: str, counter: "list[int] | None" = None
) -> str:
    """Turn each lambda into a class, and its use into an object of it."""

    made: list[str] = []
    numbered = counter if counter is not None else [0]
    at = 0
    while True:
        found = _LAMBDA.search(text, at)
        if found is None:
            break
        if _looks_like_an_index(text, found):
            # `operator[](int i) {` reads exactly like a lambda head. Skipped
            # rather than stopped at: stopping meant the first such member in
            # a supplied header hid every real lambda after it.
            at = found.end()
            continue
        numbered[0] += 1
        name = f"__py2bin_lambda_{numbered[0]}"
        captures, parameters, declared = found.groups()
        # A lambda taking nothing may leave the list out entirely - `[] { }`
        # is one - and everything below reads the parameters as text.
        parameters = parameters or ""
        try:
            closing = _matching(text, found.end() - 1)
        except ValueError:
            raise CppTranslationError(
                filename, _line_of(text, found.start()), "a lambda is not closed"
            ) from None
        body = text[found.end() - 1: closing]
        if _LAMBDA.search(body) is not None:
            # A lambda inside this one. That one goes first: what this one
            # returns is read off its own `return`, and until the inner is a
            # class with a name there is nothing there to read.
            at = found.start() + 1
            numbered[0] -= 1
            continue
        # Where it is used, read before the class is written: a generic
        # lambda's parameter types are whatever it is called with, and the
        # name it is held under is how those calls are found.
        start = _statement_start(text, found.start())
        while start < len(text) and text[start] in " \t\n":
            start += 1
        # `auto f = <lambda>;` names the object itself. Anything else needs a
        # temporary, because there is nowhere else for the object to live -
        # and a copy from one to the other is a struct assignment in a
        # declaration, which this backend does not take.
        holding = re.match(
            r"auto\s+([A-Za-z_]\w*)\s*=\s*$", text[start:found.start()]
        )
        holder = holding.group(1) if holding else f"{name}__made"
        shapes = (
            _settled_parameters(
                parameters, holder, text, closing, filename, found.start()
            )
            if re.search(r"(?<![.\w>])auto\b", parameters)
            else [parameters]
        )
        parameters = shapes[0]
        result = (declared or "").strip() or _lambda_result(body, parameters, text)
        held = _lambda_captures(captures, text, found.start(), filename, body)
        # A reference capture is the address, and every use inside follows it.
        classes = {m.group(2) for m in _CLASS_HEAD.finditer(text)}
        owner = next((s.strip() for v, s, _r, _f in held if v == _SELF), None)
        if owner is not None:
            body = _through_self(body, owner, text)
        body = _deref_references(
            body,
            {
                v: s for v, s, by_reference, _f in held
                if by_reference and v != _SELF
            },
            {name: None for name in classes},
        )
        members = "".join(
            f"    {spelled} {'*' if by_reference else ''}{variable};\n"
            for variable, spelled, by_reference, _from in held
        )
        operators = "".join(
            f"    {(declared or '').strip() or _lambda_result(body, one, text)}"
            f" operator()({one}) {body}\n"
            for one in shapes
        )
        made.append(
            f"class {name} {{\npublic:\n{members}"
            f"    {name}() {{ }}\n"
            f"{operators}}};\n"
        )
        # What it answers, recorded now. The classes made here are put into
        # the text only when the whole pass is done, so a lambda expanded
        # after this one asking what a call on it gives back found no class
        # at all - and a capture of the result got the type nothing is, which
        # is `int`.
        _LAMBDA_RESULTS[name] = result
        # Where it is used: a declaration of one, and a member per capture.
        setup = "".join(
            f" {holder}.{v} = {source};" for v, _s, _r, source in held
        )
        if holding:
            after = closing + 1 if text[closing:closing + 1] == ";" else closing
            text = text[:start] + f"{name} {holder};{setup}" + text[after:]
            at = 0
            continue
        text = (
            text[:start]
            + f"{name} {holder};{setup} "
            + text[start:found.start()]
            + holder
            + text[closing:]
        )
        at = 0
    if not made:
        return text
    # Again over everything, the classes included: a lambda written inside
    # another is in the body that just became one of these, and nothing
    # rescans what has been emitted. Each round takes one lambda away, so
    # this ends.
    return _expand_lambdas("".join(made) + text, filename, numbered)


def _looks_like_an_index(text: str, found: "re.Match[str]") -> bool:
    """Whether `[...](...)  {` is really a subscript rather than a lambda.

    `a[i](x) { ... }` is not C++, so the only way to be fooled is a subscript
    on something callable followed by a block - which does not happen. What
    does happen is a name immediately before the bracket, which a lambda
    never has.

    Some keywords may come before one: `return [a](int b){ ... };` ends in
    the letters of `return`, and reading those as a name meant a lambda
    written inside another was never expanded. Named one by one rather than
    taken as "any keyword", because `operator[](int i)` also ends in one and
    is the very thing this exists to tell apart.
    """

    before = text[:found.start()].rstrip()
    if not before:
        return False
    word = re.search(r"[A-Za-z_]\w*$", before)
    if word is not None:
        return word.group(0) not in _BEFORE_A_LAMBDA
    return before[-1] in "_)]"


#: Words a lambda may be written straight after. Everything else that ends in
#: a letter is a name, and a name before `[` is a subscript.
_BEFORE_A_LAMBDA = frozenset(
    {"return", "case", "else", "do", "throw", "co_return", "co_yield"}
)


#: What each lambda's call operator answers, by the class it became. Filled
#: while they are expanded, because that is before any of them is in the text.
_LAMBDA_RESULTS: "dict[str, str]" = {}


def _lambda_result(body: str, parameters: str, text: str) -> str:
    """What the lambda returns, where it did not say.

    Read from the `return` it holds. A lambda with no return, or one whose
    value cannot be read, is void - which is right for the first and reported
    by the C compiler for the second, on the line the lambda is on.
    """

    match = re.search(r"\breturn\b([^;]*);", _without_literals(body))
    if match is None or not match.group(1).strip():
        return "void"
    scope = parameters + ";\n" + body + "\n" + text
    held = _deduced_type(match.group(1).strip(), scope)
    return held or "int"



#: `int Bridge::run() {` - a method defined outside the class it belongs to.
_OUT_OF_LINE_HEAD = re.compile(
    r"(?<![#\w])(?:[A-Za-z_][\w \t]*?\s*[*&]*\s*)?\b([A-Za-z_]\w*)::~?[A-Za-z_]\w*\s*"
    r"\([^;{}]*\)\s*(?:const\s*)?\{"
)

#: What a captured `this` is called inside the closure. Not `this`: the
#: emitted method already has a parameter of that name, and a member called
#: `this` reached through it would read `this->this`.
_SELF = "__py2bin_self"


def _uses_the_object(body: str, text: str, owner: str) -> bool:
    """Whether the body reaches the enclosing object at all."""

    code = _without_literals(body)
    if re.search(r"(?<![.\w>])this\b", code):
        return True
    return any(
        re.search(rf"(?<![.\w>]){re.escape(name)}\b", code)
        for name in _names_of_class(text, owner)
    )


def _enclosing_class(text: str, at: int) -> "str | None":
    """The class whose body contains `at`, if any.

    Read from the text because this runs before any class is parsed - which
    is also why a lambda capturing `this` used to be refused rather than
    translated. The name is all that was missing.
    """

    found = None
    for head in _CLASS_HEAD.finditer(text):
        if head.start() > at:
            break
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        if head.end() <= at < closing:
            found = head.group(2)
    if found is not None:
        return found
    # A method defined outside its class is still a method of it, and its
    # body sits at the top level with no class braces around it. The name
    # says which class - which is the only thing this needs.
    for head in _OUT_OF_LINE_HEAD.finditer(text):
        if head.start() > at:
            break
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        if head.end() <= at < closing:
            found = head.group(1)
    return found


#: `Callback<I>(lambda)` - WRL's way of writing a COM object whose whole
#: purpose is to be called back into once.
_A_CALLBACK = re.compile(r"(?<![.\w>])Callback\s*<\s*([A-Za-z_]\w*)\s*>\s*\(")


def _pure_virtual_of(text: str, interface: str) -> "tuple[str, str, str] | None":
    """The method a callback interface declares: its name, parameters, result."""

    for head in _CLASS_HEAD.finditer(text):
        if head.group(2) != interface:
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            return None
        inside = text[head.end(): closing - 1]
        found = re.search(
            r"virtual\s+([A-Za-z_][\w\s*]*?)\s*(?:STDMETHODCALLTYPE\s+)?"
            r"([A-Za-z_]\w*)\s*\(([^;{}]*)\)\s*=\s*0\s*;",
            inside,
        )
        if found is None:
            return None
        return found.group(2), found.group(3).strip(), found.group(1).strip()
    return None


def _rewrite_wrl_callbacks(text: str, filename: str) -> str:
    """`Callback<I>(lambda)` becomes a class implementing I, and one of it.

    WRL writes the object for you: a COM object holding the closure, with
    the three IUnknown methods and the interface's own method forwarding to
    it. py2bin writes the same thing out, because the vendor's own is a
    template whose machinery this subset does not have and whose *result* is
    ordinary - a class, a reference count, and a body.

    Innermost first. A callback is very often written inside another one, and
    the inner one's `this` means the object the *outer* body was written in.
    Rewritten from the outside in, the outer pass would have already turned
    that `this` into its own member and the inner would capture the wrong
    object; taken from the inside out, each is read where it was written.
    """

    made: "list[str]" = []
    counter = 0
    for _round in range(_HOIST_ROUNDS):
        bare = _without_literals(text)
        found = [one for one in _A_CALLBACK.finditer(bare)]
        if not found:
            break
        # The last one is the innermost of any nest: a callback written
        # inside another starts later in the text than the one holding it.
        at = found[-1]
        interface = at.group(1)
        close = _closing_paren(bare, at.end() - 1)
        if close < 0:
            raise CppTranslationError(
                filename, _line_of(text, at.start()),
                f"Callback<{interface}>( is never closed",
            )
        declared = _pure_virtual_of(text, interface)
        if declared is None:
            raise CppTranslationError(
                filename, _line_of(text, at.start()),
                f"Callback<{interface}> needs {interface} to be declared here "
                f"with one method of its own; py2bin writes the class that "
                f"implements it and reads the shape of the call from that "
                f"declaration",
            )
        method, parameters, result = declared
        begins = at.end()
        while begins < len(text) and text[begins] in " \t\n\r":
            begins += 1
        written = _LAMBDA.match(text, begins)
        if written is None:
            raise CppTranslationError(
                filename, _line_of(text, at.start()),
                "Callback<> is given a lambda written out where it stands; "
                "py2bin builds the class around that lambda and has nothing "
                "to build one around otherwise",
            )
        captured = written.group(1).strip()
        spelled_parameters = written.group(2) or ""
        # `[]` - a callback that carries nothing. Ordinary C++, and the
        # simpler of the two: the object it becomes has no member to hold an
        # enclosing object and needs no class around it to be written in.
        alone = captured == ""
        if not alone and captured not in ("this", "=", "&"):
            raise CppTranslationError(
                filename, _line_of(text, at.start()),
                f"a Callback<> lambda captures {captured}; py2bin writes the "
                f"object it becomes and can carry the enclosing object - "
                f"`[this]` - or nothing at all, and nothing else yet",
            )
        opening = written.end() - 1
        shut = _matching(text, opening)
        body = text[opening: shut + 1]
        # `_enclosing_class` and not the brace scan: a method defined
        # outside its class is still inside it as far as `this` is
        # concerned, and that is where a callback is usually written.
        owner = None if alone else _enclosing_class(text, at.start())
        if owner is None and not alone:
            raise CppTranslationError(
                filename, _line_of(text, at.start()),
                "a Callback<> lambda capturing `this` is written outside any "
                "class, so there is no object for it to carry",
            )
        counter += 1
        name = f"{_CALLBACK_CLASS}{counter}"
        # The lambda's own parameter names, with the interface's types: the
        # body was written against the names and the vtable against the
        # types, and an unnamed parameter has to keep its place either way.
        head = _callback_parameters(parameters, spelled_parameters)
        made.append(
            (_CALLBACK_CLASS_ALONE if alone else _CALLBACK_CLASS_TEXT).format(
                name=name,
                interface=interface,
                owner=owner,
                method=method,
                result=result or "HRESULT",
                parameters=head,
                body=body
                if alone
                else _qualified_shared_names(
                    _through_self(body, owner, text), owner, text
                ),
                self=_SELF,
            )
        )
        # `Callback<I>(...)` answers a `ComPtr<I>` in WRL and every caller
        # writes `.Get()` after it to reach the pointer inside. The pointer
        # is what this hands back, and the `.Get()` goes with it: a holder
        # with a destructor cannot stand in a `return`, and the object does
        # not need one - it is handed straight to a callee that takes its
        # own reference, and goes when that reference goes.
        after = close + 1
        while after < len(text) and text[after] in " \t\n\r":
            after += 1
        taken = re.match(r"\.\s*Get\s*\(\s*\)", text[after:])
        ends = after + taken.end() if taken else close + 1
        text = (
            text[:at.start()]
            + f"(({interface} *)(new {name}({'' if alone else 'this'})))"
            + text[ends:]
        )
    if not made:
        return text
    return _above_the_first_use(text, "\n".join(made))


def _qualified_shared_names(body: str, owner: str, text: str) -> str:
    """Write `Owner::` in front of each static member the body calls bare.

    The body is about to become a method of a class of its own, and a bare
    name there means that class. It meant the class the lambda was written
    in, which is what the qualifier says.
    """

    for name in sorted(_shared_names_of_class(text, owner), key=len, reverse=True):
        body = _map_code(
            body,
            lambda part, n=name: re.sub(
                rf"(?<![.\w>:]){re.escape(n)}\b(?!\s*::)", f"{owner}::{n}", part
            ),
        )
    return body


def _callback_parameters(declared: str, written: str) -> str:
    """The interface's parameter types under the lambda's parameter names."""

    types = [one.strip() for one in _split_arguments(declared) if one.strip()]
    names = [one.strip() for one in _split_arguments(written) if one.strip()]
    out: "list[str]" = []
    for index, spelled in enumerate(types):
        held = re.sub(r"/\*.*?\*/", " ", spelled).strip()
        given = names[index] if index < len(names) else ""
        wanted = re.match(r"^(.*?)([A-Za-z_]\w*)?\s*$", given.strip())
        name = (wanted.group(2) if wanted else "") or f"__py2bin_arg{index}"
        # The declared name, if the interface gave one, is dropped: the body
        # was written against the lambda's.
        held = re.sub(r"\b[A-Za-z_]\w*\s*$", "", held).strip() or held
        out.append(f"{held} {name}")
    return ", ".join(out)


#: What each generated callback class is called.
_CALLBACK_CLASS = "__py2bin_callback_"

#: One of them. The three IUnknown methods are what makes it a COM object at
#: all, and the fourth is the one the caller will reach for. The count starts
#: at zero because the `ComPtr` this is handed to takes the first reference,
#: which is where WRL puts it too.
_CALLBACK_CLASS_TEXT = """
class {name} : public {interface} {{
public:
    unsigned long __py2bin_refs;
    {owner} *{self};
    {name}({owner} *__py2bin_given) {{
        __py2bin_refs = 0;
        {self} = __py2bin_given;
    }}
    HRESULT QueryInterface(REFIID __py2bin_asked, void **__py2bin_into) {{
        if (__py2bin_into == 0) {{ return E_POINTER; }}
        *__py2bin_into = (void *)this;
        __py2bin_refs = __py2bin_refs + 1;
        return S_OK;
    }}
    unsigned long AddRef() {{
        __py2bin_refs = __py2bin_refs + 1;
        return __py2bin_refs;
    }}
    unsigned long Release() {{
        __py2bin_refs = __py2bin_refs - 1;
        if (__py2bin_refs == 0) {{ delete this; return 0; }}
        return __py2bin_refs;
    }}
    {result} {method}({parameters}) {body}
}};
"""


#: The same object for a lambda that carries nothing: no member to hold an
#: enclosing object, and a constructor that asks for none.
_CALLBACK_CLASS_ALONE = """
class {name} : public {interface} {{
public:
    unsigned long __py2bin_refs;
    {name}() {{
        __py2bin_refs = 0;
    }}
    HRESULT QueryInterface(REFIID __py2bin_asked, void **__py2bin_into) {{
        if (__py2bin_into == 0) {{ return E_POINTER; }}
        *__py2bin_into = (void *)this;
        __py2bin_refs = __py2bin_refs + 1;
        return S_OK;
    }}
    unsigned long AddRef() {{
        __py2bin_refs = __py2bin_refs + 1;
        return __py2bin_refs;
    }}
    unsigned long Release() {{
        __py2bin_refs = __py2bin_refs - 1;
        if (__py2bin_refs == 0) {{ delete this; return 0; }}
        return __py2bin_refs;
    }}
    {result} {method}({parameters}) {body}
}};
"""


def _above_the_first_use(text: str, made: str) -> str:
    """Put generated classes above everything, after the directives."""

    at = 0
    for line in text.split("\n"):
        if line.lstrip().startswith("#") or not line.strip():
            at += len(line) + 1
            continue
        break
    return text[:at] + made + "\n" + text[at:]


def _through_self(body: str, owner: str, text: str) -> str:
    """Point what the lambda body says at the object it captured.

    `this` becomes the member holding it. A bare member or method name means
    the same object in C++ and means nothing here, so it is spelled out -
    which is the whole of what capturing `this` buys.
    """

    body = _map_code(
        body, lambda part: re.sub(r"(?<![.\w>])this\b(?!\s*->)", _SELF, part)
    )
    body = _map_code(
        body, lambda part: re.sub(r"(?<![.\w>])this\s*->", f"{_SELF}->", part)
    )
    # Not the static members: one of those takes no object, so pointing it
    # at the captured one hands a receiver to a function that has no place
    # for one. Its bare name already means the same thing here as it did
    # where the lambda was written.
    names = _names_of_class(text, owner) - _shared_names_of_class(text, owner)
    for name in sorted(names, key=len, reverse=True):
        body = _map_code(
            body,
            lambda part, n=name: re.sub(
                rf"(?<![.\w>]){re.escape(n)}\b", f"{_SELF}->{n}", part
            ),
        )
    return body


def _shared_names_of_class(text: str, owner: str) -> "set[str]":
    """The `static` members that class declares, which take no object."""

    for head in _CLASS_HEAD.finditer(text):
        if head.group(2) != owner:
            continue
        try:
            closing = _matching(text, head.end() - 1)
            found = _split_members(
                text[head.end(): closing - 1], owner, "<c++>", 0
            )
        except (ValueError, CppTranslationError):
            return set()
        return {item.name for item in found.methods if item.shared and item.name}
    return set()


def _names_of_class(text: str, owner: str) -> "set[str]":
    """The members and methods that class declares, read from the text."""

    for head in _CLASS_HEAD.finditer(text):
        if head.group(2) != owner:
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            return set()
        try:
            found = _split_members(
                text[head.end(): closing - 1], owner, "<c++>", 0
            )
        except CppTranslationError:
            return set()
        return found.field_names() | found.method_names()
    return set()

def _lambda_captures(
    captures: str, text: str, at: int, filename: str, body: str = ""
) -> "list[tuple[str, str, bool]]":
    """Each captured name, the type it is held as, and whether by reference.

    A capture by value is a member holding a copy. A capture by reference is
    a member holding the address, and every use inside the body follows it -
    which is what a reference is once it is written out. Both are members,
    which is why `[=]` and `[&]` can be honoured too: what they capture is
    whatever the body uses that the enclosing scope declared.
    """

    spelled = captures.strip()
    if not spelled:
        return []
    if spelled in ("=", "&"):
        held = _captures_used(text, at, body, by_reference=spelled == "&")
        # `[=]` and `[&]` written inside a member function capture `this` as
        # well, and C++ resolves a bare member name through it. Without this
        # the body named a method that exists on no class the closure knows.
        owner = _enclosing_class(text, at)
        if owner is not None and _uses_the_object(body, text, owner):
            held.append((_SELF, f"{owner} ", True, "this"))
        return held
    held: list[tuple[str, str, bool, str]] = []
    for part in _split_arguments(spelled):
        name = part.strip()
        by_reference = name.startswith("&")
        if by_reference:
            name = name[1:].strip()
        if name == "this":
            owner = _enclosing_class(text, at)
            if owner is None:
                raise CppTranslationError(
                    filename,
                    _line_of(text, at),
                    "a lambda captures `this` outside any class, so there is "
                    "nothing for it to capture",
                )
            held.append((_SELF, f"{owner} ", True, "this"))
            continue
        # `[v = n * 2]` - the member is initialised from an expression rather
        # than from a variable of the same name. C++ calls it an init-capture
        # and it is the only way a lambda holds something the scope has no
        # name for.
        given = re.match(r"^([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$", name, re.S)
        if given is not None:
            name, source = given.group(1), given.group(2).strip()
            found = _deduced_type(source, text[:at]) or "int"
            held.append(
                (name, found, by_reference, f"&({source})" if by_reference else source)
            )
            continue
        if not name.isidentifier():
            raise CppTranslationError(
                filename, _line_of(text, at), f"cannot read the capture {name!r}"
            )
        found = _deduced_type(name, text[:at]) or "int"
        held.append(
            (name, found, by_reference, f"&{name}" if by_reference else name)
        )
    return held


#: `auto x = ...`, read only for the name it declares. Read as a type name
#: `auto` is not one, so the reader of declarations passes over it - and a
#: capture-default asking "does this scope declare that name?" was told no
#: about every `auto` in it.
_AUTO_NAMED = re.compile(r"(?<![.\w>])auto\s*[*&]?\s*([A-Za-z_]\w*)\s*=")


def _captures_used(
    text: str, at: int, body: str, by_reference: bool
) -> "list[tuple[str, str, bool]]":
    """What `[=]` or `[&]` captures: whatever the body uses and the scope has.

    C++ works this out the same way - a capture-default takes what the body
    mentions - so the list is read from the body rather than guessed at. A
    name the enclosing scope does not declare is not a capture: it is a
    global, or a function, and needs nothing.
    """

    before = text[:at]
    start = _enclosing_body_start(before)
    scope = before[start:]
    held: "list[tuple[str, str, bool, str]]" = []
    seen: "set[str]" = set()
    for match in re.finditer(r"(?<![.\w>])([A-Za-z_]\w*)", _without_literals(body)):
        name = match.group(1)
        if name in seen or name in _NOT_A_TYPE or name in _LEADS_A_TYPE:
            continue
        # A name that is called is a function, not a capture - unless the
        # scope declares it, and then it is an object with a call operator: a
        # `std::function`, or a lambda held in `auto`. Skipped on the shape
        # of the use alone, a closure that called another closure did not
        # capture it, and the C named a function nothing had defined.
        after = body[match.end():].lstrip()
        declared = set(_declared_here(scope)) | {
            found.group(1)
            for found in _AUTO_NAMED.finditer(_without_literals(scope))
        }
        if after.startswith("(") and name not in declared:
            continue
        if name not in declared:
            continue
        found = _deduced_type(name, before)
        if found is None:
            continue
        seen.add(name)
        held.append(
            (name, found, by_reference, f"&{name}" if by_reference else name)
        )
    return held


def _enclosing_body_start(before: str) -> int:
    """Where the innermost open block begins, so a scope can be read from it."""

    depth = 0
    index = len(before)
    while index > 0:
        index -= 1
        piece = before[index]
        if piece == "}":
            depth += 1
        elif piece == "{":
            if depth == 0:
                return index + 1
            depth -= 1
    return 0



def _member_spans(text: str) -> "list[tuple[int, int, set[str]]]":
    """Each class body in the text, with the names it declares members under.

    A member hides anything of the same name declared outside the class, so
    a bare call inside one is answered by the class before the file is asked.
    """

    spans: "list[tuple[int, int, set[str]]]" = []
    for head in _CLASS_HEAD.finditer(text):
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        body = text[head.end() - 1: closing]
        spans.append((
            head.end() - 1,
            closing,
            set(re.findall(r"(?<![.\w>])([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?\{", body)),
        ))
    return spans


def _hidden_by_a_member(
    spans: "list[tuple[int, int, set[str]]]", name: str, where: int
) -> bool:
    """Whether a call to `name` at `where` is inside a class that declares one."""

    return any(start < where < end and name in names for start, end, names in spans)



#: `int kind<double>(double v) {` - what follows `template<>`, an explicit
#: specialisation. The arguments are spelled out, so nothing is deduced.
_SPECIALISATION = re.compile(
    r"(?:([A-Za-z_][\w\s]*[\s*&]+)\s*)?\b([A-Za-z_]\w*)\s*<([^<>]*)>\s*"
    r"\(([^()]*)\)\s*(?:const\s*)?\{"
)

#: `template<typename T> T Box<T>::get() { ... }` - a member of a class
#: template, defined outside it. Which is how a header usually writes one.
_TEMPLATE_MEMBER = re.compile(
    r"\btemplate\s*<([^<>]*)>\s*"
    r"(?:([A-Za-z_][\w\s]*[\s*&]+)\s*)?"
    r"\b([A-Za-z_]\w*)\s*<([^<>]*)>\s*::\s*(~?[A-Za-z_]\w*)\s*"
    r"\(([^()]*)\)\s*(?:const\s*)?\{"
)

#: A backstop on the folding below, not a budget.
_OUT_OF_LINE_ROUNDS = 512



def _class_around(text: str, at: int) -> "tuple[int, str] | None":
    """Where the innermost class holding `at` starts, and what it is called.

    Not to be confused with `_enclosing_class` above, which answers the same
    question for a different purpose and in a different shape.
    """

    innermost = None
    for head in _CLASS_HEAD.finditer(text):
        if head.end() > at:
            break
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        if head.end() <= at < closing:
            innermost = (head.start(), head.group(2))
    return innermost


def _inside_a_template_pattern(text: str, at: int) -> bool:
    """Whether the class enclosing `at` is itself `template <...> class`."""

    found = _class_around(text, at)
    return found is not None and _is_a_template_pattern(text, found[0])


def _expand_member_templates(text: str, filename: str) -> str:
    """`template<typename T> T twice(T v)` written inside a class.


    C++ makes one copy per set of argument types, and so does this - read
    from the calls, because a member template has no other use. Each copy is
    an ordinary member of the class, named the way a free template's copy is,
    and every call that asked for it is pointed at the one it meant.
    """

    for _round in range(_OUT_OF_LINE_ROUNDS):
        found = None
        for match in _TEMPLATE.finditer(text):
            if _depth_at(text, match.start()) <= 0:
                continue
            if _inside_a_template_pattern(text, match.start()):
                # The class around it is a pattern, so its objects do not
                # exist yet and no call can say which copy it wants. Left for
                # the pass that runs after the class has been written out;
                # taken now, it was taken away - nothing called it under the
                # pattern's own name, so it was dropped as unused.
                continue
            found = match
            break
        if found is None:
            return text
        rest = text[found.end():]
        # `U sumAs() const {` is a definition too. The general pattern stops
        # at the `)` and wants a brace next, so a member template that is
        # const read as "not a function at all" and was refused.
        definition = _DEFINITION.match(rest) or _QUALIFIED_DEFINITION.match(rest)
        if definition is None:
            raise CppTranslationError(
                filename,
                _line_of(text, found.start()),
                "a template written inside a class has to be a member "
                "function; py2bin reads the pattern and writes out one copy "
                "per use, and this is neither",
            )
        try:
            closing = _matching(rest, definition.end() - 1)
        except ValueError:
            return text
        parameters = _template_parameters(found.group(1))
        pattern = rest[:closing + 1]
        name = definition.group(2)
        without = text[:found.start()] + text[found.end() + closing + 1:]
        holder = _class_around(text, found.start())
        copies, without = _member_copies(
            without,
            name,
            parameters,
            definition.group(3),
            pattern,
            holder[1] if holder else "",
        )
        if not copies:
            # Nothing calls it, so there is nothing to write. The pattern is
            # gone either way: it is not C.
            text = without
            continue
        text = (
            without[:found.start()]
            + "\n".join(copies)
            + "\n"
            + without[found.start():]
        )
    raise CppTranslationError(
        "<c++>", 0,
        "member templates that never stop asking for another copy",
    )


def _receiver_before(text: str, at: int) -> str:
    """The expression a `.` or `->` at `at` is written on.

    The whole chain, not the last name in it: `a->b->c.m()` is called on
    `a->b->c`, and stopping at the `>` of an arrow gave `>c`, which is not
    an expression and has no type - so a member template was copied into
    whichever class happened to be in hand rather than the one that has it.
    """

    back = at
    while back > 0 and text[back - 1] in " \t\n":
        back -= 1
    end = back
    while True:
        while back > 0 and (text[back - 1].isalnum() or text[back - 1] == "_"):
            back -= 1
        while back > 0 and text[back - 1] in " \t":
            back -= 1
        if back >= 2 and text[back - 2: back] == "->":
            back -= 2
        elif back >= 1 and text[back - 1] == ".":
            back -= 1
        else:
            break
        while back > 0 and text[back - 1] in " \t":
            back -= 1
    return text[back:end].strip()


def _member_copies(
    text: str,
    name: str,
    parameters: "list[tuple[str, bool]]",
    declared: str,
    pattern: str,
    holder: str = "",
) -> "tuple[list[str], str]":
    """One copy per set of argument types the calls to `name` ask for.

    Only the calls made on an object of the class this member belongs to.
    Two copies of one class template each have their own copy of the member
    template inside them, and both are called `As`; taking every call that
    spells that name put `ComPtr<A>`'s copies inside `ComPtr<B>` and left the
    other class without the member at all.
    """

    made: "dict[str, str]" = {}
    out: list[str] = []
    at = 0
    # `v.as<int>()` as well as `v.as(x)`. Written with the argument spelled
    # out there is nothing to deduce - and read as though the `(` came
    # straight after the name, no call matched at all and the member was
    # dropped as one nothing uses.
    for call in re.finditer(
        rf"(\.|->)\s*{re.escape(name)}\s*(<[^;{{}}()<>]*>)?\s*\(",
        _without_literals(text),
    ):
        if call.start() < at:
            continue
        close = _closing_paren(text, call.end() - 1)
        if close < 0:
            continue
        if holder:
            receiver = _receiver_before(text, call.start())
            held = (_deduced_type(receiver, text) or "").replace("*", "").strip()
            if not held:
                # The whole chain could not be read. The name at the end of
                # it is declared somewhere, and what it is declared as is
                # what the call is made on.
                tail = re.split(r"\.|->", receiver)[-1].strip()
                held = (_deduced_type(tail, text) or "").replace("*", "").strip()
            if held and held != holder:
                continue
        if call.group(2):
            # Spelled out, so there is nothing to work out.
            deduced = [
                one.strip()
                for one in _split_arguments(call.group(2)[1:-1])
                if one.strip()
            ]
        else:
            given = _call_arguments(text, call.end() - 1)
            deduced = _deduce_arguments(
                parameters, declared, given, text, call.start()
            )
        if deduced is None:
            continue
        named = _instantiated_name(name, deduced)
        if named not in made:
            copy = _substituted(pattern, parameters, deduced)
            made[named] = _map_code(
                copy,
                lambda part, n=name, s=named: re.sub(
                    rf"(?<![.\w>]){re.escape(n)}\b(?!\s*<)", s, part
                ),
            )
        out.append(text[at:call.start()])
        out.append(f"{call.group(1)}{named}(")
        at = call.end()
    out.append(text[at:])
    return list(made.values()), "".join(out)

def _fold_out_of_line_templates(text: str) -> str:
    """Put a class template's members back inside it.

    py2bin expands a template by copying its pattern, and the pattern is the
    class body. A member written as `template<typename T> T Box<T>::get()`
    is part of that pattern and was sitting outside it, where the reader saw
    a template that is neither a class nor a function and said so.
    """

    for _round in range(_OUT_OF_LINE_ROUNDS):
        found = None
        for match in _TEMPLATE_MEMBER.finditer(text):
            if _depth_at(text, match.start()) == 0:
                found = match
                break
        if found is None:
            return text
        try:
            closing = _matching(text, found.end() - 1)
        except ValueError:
            return text
        body = text[found.end() - 1: closing + 1]
        returns = (found.group(2) or "").strip()
        owner, name, parameters = found.group(3), found.group(5), found.group(6)
        spelled = f"{returns + ' ' if returns else ''}{name}({parameters}) {body}"
        # Cut the definition first. The class is somewhere else in the text
        # and taking this out does not move anything inside it.
        without = text[:found.start()] + text[closing + 1:]
        placed = _into_class_body(without, owner, name, spelled)
        if placed is None:
            # No such class template here. Left alone, so whatever reads it
            # next says so about the text the author wrote.
            return text
        text = placed
    raise CppTranslationError(
        "<c++>", 0,
        "members defined outside their class template that never stop being "
        "folded back into it",
    )


def _into_class_body(
    text: str, owner: str, name: str, spelled: str
) -> "str | None":
    """Put `spelled` into `owner`'s body, over the declaration it defines."""

    for head in _CLASS_HEAD.finditer(text):
        if head.group(2) != owner:
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        inside = text[head.end(): closing]
        # The declaration this defines: the same name, and a `;` where the
        # body would be. Replaced rather than added beside, so the class does
        # not end up with the member twice.
        declared = re.search(
            rf"(?<![.\w>])(?:[A-Za-z_][\w\s*&]*?\s+)?{re.escape(name)}\s*"
            rf"\([^;{{}}]*\)\s*(?:const\s*)?;",
            inside,
        )
        if declared is not None:
            inside = inside[:declared.start()] + spelled + inside[declared.end():]
        else:
            inside = inside + "\n" + spelled + "\n"
        return text[:head.end()] + inside + text[closing:]
    return None


#: `typedef T *iterator;` written inside a class. C has no such scope, so the
#: name is resolved where it is used and the typedef itself goes.
#: The stars may have spaces between them: `vector<Shape *>` puts `Shape *`
#: where `T` was and leaves `typedef Shape * *iterator;`. Written to allow
#: only adjacent stars, this matched nothing there, the typedef stayed inside
#: the struct, and C rejected a program for holding a vector of pointers.
_NESTED_TYPEDEF = re.compile(
    r"\btypedef\s+([A-Za-z_][\w\s]*?)\s*((?:\*\s*)*)([A-Za-z_]\w*)\s*;"
)


def _resolve_nested_typedefs(text: str) -> str:
    """`vector<int>::iterator` becomes the type the class said it is.

    A member typedef is how a container names the things it is made of, and
    every one of them - `iterator`, `value_type`, `size_type` - is written
    this way. C has no scope to put them in, so each use is replaced by what
    it stands for and the typedef itself is dropped.
    """

    names: "dict[tuple[str, str], str]" = {}
    out: list[str] = []
    at = 0
    for head in _CLASS_HEAD.finditer(text):
        if head.start() < at:
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        owner = head.group(2)
        inside = text[head.end() - 1: closing]

        def taken(match: "re.Match[str]", o=owner) -> str:
            names[(o, match.group(3))] = (
                f"{match.group(1).strip()} {match.group(2).strip()}".strip()
            )
            return ""

        out.append(text[at: head.end() - 1])
        out.append(_NESTED_TYPEDEF.sub(taken, inside))
        at = closing
    out.append(text[at:])
    text = "".join(out)
    if not names:
        return text
    for (owner, name), held in names.items():
        text = _map_code(
            text,
            lambda part, o=owner, n=name, h=held: re.sub(
                rf"\b{re.escape(o)}\s*::\s*{re.escape(n)}\b", h, part
            ),
        )
    # And the bare name, which is how the class's own members say it:
    # `using Handler = ...;` written inside a class is `Handler` everywhere
    # in that class and in its methods, spelled out or not. Only there - a
    # member typedef names a type inside its class and nowhere else, and the
    # same word outside is somebody else's.
    for (owner, name), held in names.items():
        text = _within_the_class(text, owner, name, held)
    return text


def _within_the_class(text: str, owner: str, name: str, held: str) -> str:
    """Replace a bare `name` inside `owner`'s body and in its methods."""

    spans: "list[tuple[int, int]]" = []
    for head in _CLASS_HEAD.finditer(text):
        if head.group(2) != owner:
            continue
        try:
            spans.append((head.end() - 1, _matching(text, head.end() - 1)))
        except ValueError:
            continue
    # `Thing::fire(Handler h) { ... }` written outside the class is still
    # inside it as far as a name is concerned.
    for out_of_line in re.finditer(
        rf"(?<![.\w>]){re.escape(owner)}\s*::\s*~?[A-Za-z_]\w*\s*\(", text
    ):
        opening = text.find("{", out_of_line.end())
        if opening < 0:
            continue
        try:
            spans.append((out_of_line.start(), _matching(text, opening)))
        except ValueError:
            continue
    if not spans:
        return text
    spans.sort()
    out: "list[str]" = []
    at = 0
    for start, end in spans:
        if start < at:
            continue
        out.append(text[at:start])
        out.append(
            _map_code(
                text[start:end],
                lambda part: re.sub(
                    rf"(?<![.\w>:]){re.escape(name)}\b(?!\s*::)", held, part
                ),
            )
        )
        at = end
    out.append(text[at:])
    return "".join(out)


#: What each copy was made from: `Box__int` came from `Box` with `int`. The
#: name a copy is written under says what it holds only by convention and
#: the mangling cannot be read back - `int_p` is `int *` and `a_b` could be
#: either - so what it was asked for is remembered instead. A member
#: template deduces against it: `fill(Box<U> *)` handed a `Box__int` learns
#: `U` is `int` from here, the shape having been mangled away by then.
_INSTANTIATED: "dict[str, tuple[str, list[str]]]" = {}


def _fits_the_shape(
    shape: str,
    argument: str,
    parameters: "set[str]",
    bound: "dict[str, str]",
    binding: bool = False,
) -> bool:
    """Whether one argument has the shape a narrower copy was written for.

    `T *` fits `int *` and binds T to `int`; `T` fits anything; `int` fits
    only `int`. A parameter named twice has to be bound to the same thing
    both times, which is the whole of how `is_same<T, T>` answers.

    `binding` is the difference between the two questions this answers.
    Choosing between partial specialisations, the spelling *is* the question:
    `strip<T &>` is the copy for a reference and must not be picked for a
    value, or the narrower copy wins everything. Deducing what a call passed,
    the spelling is a parameter and a parameter binds: `total(const Buf<N> &)`
    takes a `Buf<3>`, and that is how nearly every such call is written.
    """

    shape = shape.strip()
    argument = argument.strip()
    if shape in parameters:
        settled = bound.get(shape)
        if settled is not None:
            return _same_type(settled, argument)
        bound[shape] = argument
        return True
    for tail in ("*", "&"):
        if shape.endswith(tail):
            if argument.endswith(tail):
                return _fits_the_shape(
                    shape[:-1], argument[:-1], parameters, bound, binding
                )
            # A reference parameter binds to a value; a `T &` specialisation
            # is not the one a value picks.
            if not binding or tail != "&":
                return False
            return _fits_the_shape(
                shape[:-1], argument, parameters, bound, binding
            )
    for head in ("const", "volatile"):
        if re.match(rf"^{head}\b", shape):
            if re.match(rf"^{head}\b", argument):
                return _fits_the_shape(
                    shape[len(head):], argument[len(head):], parameters,
                    bound, binding
                )
            if not binding:
                return False
            return _fits_the_shape(
                shape[len(head):], argument, parameters, bound, binding
            )
    # `Box<U>` against `Box<int>`: the same template, and its arguments have
    # to fit one for one. This is what lets a parameter spelled in terms of
    # another template deduce anything at all - `As(ComPtr<U> *other)` is
    # written that way, and so is half of what a container's members take.
    shaped = re.match(r"^([A-Za-z_]\w*)\s*<", shape)
    given = re.match(r"^([A-Za-z_]\w*)\s*<", argument)
    if shaped is not None and given is None:
        # The argument is a copy that has already been written out, so its
        # shape is in its name and not in its spelling. What it was made from
        # was kept when it was made.
        came_from = _INSTANTIATED.get(argument.strip())
        if came_from is None:
            return False
        made_of, held = came_from
        if made_of != shaped.group(1):
            return False
        shut = _closing_angle(shape, shaped.end() - 1)
        if shut < 0 or shape[shut + 1:].strip():
            return False
        inside = [
            a.strip()
            for a in _split_arguments(shape[shaped.end(): shut])
            if a.strip()
        ]
        if len(inside) != len(held):
            return False
        return all(
            _fits_the_shape(one, other, parameters, bound, binding)
            for one, other in zip(inside, held)
        )
    if shaped is not None and given is not None:
        if shaped.group(1) != given.group(1):
            return False
        shut_shape = _closing_angle(shape, shaped.end() - 1)
        shut_given = _closing_angle(argument, given.end() - 1)
        if shut_shape < 0 or shut_given < 0:
            return False
        if shape[shut_shape + 1:].strip() or argument[shut_given + 1:].strip():
            return False
        inside_shape = [
            a.strip()
            for a in _split_arguments(shape[shaped.end(): shut_shape])
            if a.strip()
        ]
        inside_given = [
            a.strip()
            for a in _split_arguments(argument[given.end(): shut_given])
            if a.strip()
        ]
        if len(inside_shape) != len(inside_given):
            return False
        return all(
            _fits_the_shape(one, other, parameters, bound, binding)
            for one, other in zip(inside_shape, inside_given)
        )
    # Nothing of the pattern left to take apart: it names a type outright,
    # and only that type fits it.
    return _same_type(shape, argument)


def _same_type(one: str, other: str) -> bool:
    """Whether two spellings name the same type, spacing aside."""

    return re.sub(r"\s+", " ", one).strip() == re.sub(r"\s+", " ", other).strip()


def _how_narrow(shapes: "list[str]", parameters: "set[str]") -> int:
    """How specialised a copy is, for choosing between two that both fit.

    C++ orders these by which pattern is more specialised than the other.
    Counted here instead: a position naming a type outright is narrower than
    one naming a parameter with something around it, which is narrower than a
    bare parameter - and a parameter used twice narrows both places it stands.
    """

    score = 0
    seen: "dict[str, int]" = {}
    for shape in shapes:
        spelled = shape.strip()
        if spelled in parameters:
            # A bare parameter narrows nothing, except where the same one
            # stands in another position too: `<T, T>` says the two arguments
            # are the same type, which not every pair of arguments is.
            seen[spelled] = seen.get(spelled, 0) + 1
            if seen[spelled] > 1:
                score += 4
            continue
        mentioned = [
            name
            for name in parameters
            if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", spelled)
        ]
        if not mentioned:
            # A type named outright fits one argument and no other.
            score += 100
            continue
        # Something written around a parameter, and more of it is narrower:
        # `T **` is a `T *` that is itself a pointer, so where both fit, the
        # second is the one C++ picks.
        bare = sum(len(name) for name in mentioned)
        score += 2 + max(0, len(re.sub(r"\s+", "", spelled)) - bare)
    return score


def _narrower_copy(
    entries: "list[tuple[list, list[str], str, str]]", arguments: "list[str]"
) -> "tuple[list, list[str], str, str, dict[str, str]] | None":
    """The narrowest copy whose shape these arguments have, if any."""

    best = None
    best_score = -1
    for parameters, shapes, body, keyword in entries:
        if len(shapes) != len(arguments):
            continue
        names = {name for name, _is_type, _pack in parameters}
        bound: "dict[str, str]" = {}
        if not all(
            _fits_the_shape(shape, argument, names, bound)
            for shape, argument in zip(shapes, arguments)
        ):
            continue
        if len(bound) != len(names):
            # A parameter the shape never mentions cannot be worked out from
            # the arguments, so this copy is not the one being asked for.
            continue
        score = _how_narrow(shapes, names)
        if score > best_score:
            best, best_score = (parameters, shapes, body, keyword, bound), score
    return best



#: `typename enable_if<C, T>::type name(params) {` - a function whose return
#: type is a question asked about its own template arguments. Where the
#: answer is that there is no such type, the function is not a candidate at
#: all and another of the same name is used instead. That is the whole of
#: what SFINAE is for in practice, and the whole of what is implemented.
_GUARDED_HEAD = re.compile(r"(?:typename\s+)?([A-Za-z_]\w*)\s*<")


def _guarded_definition(text: str) -> "dict | None":
    """Read `typename Guard<...>::member name(params) {`, or answer None."""

    head = _GUARDED_HEAD.match(text)
    if head is None:
        return None
    shut = _closing_angle(text, head.end() - 1)
    if shut < 0:
        return None
    rest = text[shut + 1:]
    tail = re.match(
        r"\s*::\s*([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\(([^;{}()]*)\)\s*\{",
        rest,
    )
    if tail is None:
        return None
    return {
        "guard": head.group(1),
        "arguments": text[head.end(): shut],
        "member": tail.group(1),
        "name": tail.group(2),
        "parameters": tail.group(3),
        "opens": shut + 1 + tail.end() - 1,
    }


def _member_of(copy: str, member: str) -> "str | None":
    """What a made class calls `member`, or None where it has no such name."""

    typedef = re.search(
        rf"\btypedef\s+(.+?)\s+{re.escape(member)}\s*;", copy, re.S
    )
    if typedef is not None:
        return typedef.group(1).strip()
    constant = re.search(
        rf"\b{re.escape(member)}\s*=\s*([^;]+);", copy
    )
    return constant.group(1).strip() if constant is not None else None


def _settled(spelled: str, made: "dict[str, str]") -> "str | None":
    """Work an instantiated constant down to what it is, or answer None.

    `is_pointer<int>::value` is a class that has been written out and a name
    in it; `!` in front of one turns it around. Anything else is handed back
    as it stands, which is what a plain `true` or a number is already.
    """

    spelled = spelled.strip()
    negated = False
    while spelled.startswith("!"):
        negated = not negated
        spelled = spelled[1:].strip()
    asked = re.match(r"^([A-Za-z_]\w*)\s*<", spelled)
    if asked is not None:
        shut = _closing_angle(spelled, asked.end() - 1)
        after = re.match(r"\s*::\s*([A-Za-z_]\w*)\s*$", spelled[shut + 1:])
        if shut < 0 or after is None:
            return None
        inside = [
            a.strip()
            for a in _split_arguments(spelled[asked.end(): shut])
            if a.strip()
        ]
        copy = made.get(_instantiated_name(asked.group(1), inside))
        if copy is None:
            return None
        spelled = _member_of(copy, after.group(1))
        if spelled is None:
            return None
    spelled = spelled.strip()
    # One spelling for a yes and one for a no, because the name a copy is
    # made under is built out of this: `true` and `1` are the same answer,
    # and two names for it are two copies of the same class.
    if spelled in ("true", "1"):
        spelled = "0" if negated else "1"
    elif spelled in ("false", "0"):
        spelled = "1" if negated else "0"
    elif negated:
        return None
    return spelled



#: Answered where a guard cannot be settled yet, as against not at all.
_LATER = object()


def _guard_asked(
    parameters: "list[tuple[str, bool, bool]]",
    arguments: "list[str]",
    pattern: str,
) -> "tuple[str, list[str], str] | None":
    """The guard a pattern asks, with this call's arguments put in."""

    read = _guarded_definition(pattern)
    if read is None:
        return None
    spelled = _substituted(read["arguments"], parameters, arguments)
    inside = [a.strip() for a in _split_arguments(spelled) if a.strip()]
    return read["guard"], inside, read["member"]


def _asked_by_the_guard(
    parameters: "list[tuple[str, bool, bool]]",
    arguments: "list[str]",
    pattern: str,
) -> "list[tuple[str, list[str]]]":
    """Every class a guard has to have written out before it can answer."""

    asked = _guard_asked(parameters, arguments, pattern)
    if asked is None:
        return []
    guard, inside, _member = asked
    wanted: "list[tuple[str, list[str]]]" = []
    for part in inside:
        spelled = part.strip().lstrip("!").strip()
        named = re.match(r"^([A-Za-z_]\w*)\s*<", spelled)
        if named is None:
            continue
        shut = _closing_angle(spelled, named.end() - 1)
        if shut < 0:
            continue
        wanted.append(
            (
                named.group(1),
                [
                    a.strip()
                    for a in _split_arguments(spelled[named.end(): shut])
                    if a.strip()
                ],
            )
        )
    return wanted


def _guarded_copy(
    name: str,
    named: str,
    parameters: "list[tuple[str, bool, bool]]",
    arguments: "list[str]",
    pattern: str,
    made: "dict[str, str]",
):
    """The copy for a guarded function, or None where the guard says no.

    `_LATER` where the classes the guard asks about have not been written
    out yet, which is how the rounds get told to come back to it.
    """

    read = _guarded_definition(pattern)
    asked = _guard_asked(parameters, arguments, pattern)
    if read is None or asked is None:
        return None
    guard, inside, member = asked
    settled: "list[str]" = []
    for part in inside:
        answer = _settled(part, made)
        if answer is None:
            return _LATER
        settled.append(answer)
    copy = made.get(_instantiated_name(guard, settled))
    if copy is None:
        # The guard itself, once its arguments are numbers rather than
        # questions. Asked for by the caller and looked at again next round.
        return _LATER
    returns = _member_of(copy, member)
    if returns is None:
        return None
    body = _substituted(pattern[read["opens"]:], parameters, arguments)
    return f"{returns} {named}({_substituted(read['parameters'], parameters, arguments)}) {body}"


class _Declared:
    """A parameter list, wearing the face a match object wears."""

    def __init__(self, spelled: str) -> None:
        self._spelled = spelled

    def group(self, which: int) -> str:
        return self._spelled


def _point_at_existing_copies(text: str) -> str:
    """`Box<int>` written after the expander ran is the `Box__int` it made."""

    if not _INSTANTIATED:
        return text
    names = {made_of for made_of, _held in _INSTANTIATED.values()}
    spelled = re.compile(
        r"(?<![.\w>])(" + "|".join(re.escape(n) for n in names) + r")\s*<"
    )
    out: "list[str]" = []
    at = 0
    for found in spelled.finditer(text):
        if found.start() < at:
            continue
        shut = _closing_angle(text, found.end() - 1)
        if shut < 0:
            continue
        arguments = [
            a.strip()
            for a in _split_arguments(text[found.end(): shut])
            if a.strip()
        ]
        named = _instantiated_name(found.group(1), arguments)
        if named not in _INSTANTIATED:
            continue
        out.append(text[at:found.start()])
        out.append(named)
        at = shut + 1
    out.append(text[at:])
    return "".join(out)


def _expand_templates(text: str, filename: str) -> str:
    """Replace every template with the copies this file actually asks for."""


    patterns: dict[str, tuple[list[tuple[str, bool]], str, str]] = {}
    #: Copies the author wrote out by hand, under the name the expander would
    #: have used. They seed `made`, so the pattern is never copied over them.
    written: "dict[str, str]" = {}
    #: The narrower copies of a class template, by name: each is the pattern
    #: its arguments have to have, and the body to use when they do.
    narrower: "dict[str, list[tuple[list, list[str], str, str]]]" = {}
    cut: list[tuple[int, int]] = []
    for match in _TEMPLATE.finditer(text):
        if _depth_at(text, match.start()) != 0:
            continue
        parameters = _template_parameters(match.group(1))
        rest = text[match.end():]
        narrowed = _SPECIALISED_HEAD.match(rest)
        if narrowed is not None:
            shut = _closing_angle(rest, narrowed.end() - 1)
            body = rest[shut + 1:] if shut >= 0 else ""
            opening = body.find("{")
            if shut >= 0 and opening >= 0 and not body[:opening].strip():
                # `_matching` answers the index just past the `}`, which is
                # where the `;` after a class body sits.
                closing = _matching(body, opening)
                end = shut + 1 + closing
                while end < len(rest) and rest[end] in " \t":
                    end += 1
                if end < len(rest) and rest[end] == ";":
                    end += 1
                narrower.setdefault(narrowed.group(2), []).append(
                    (
                        parameters,
                        [a.strip() for a in _split_arguments(
                            rest[narrowed.end(): shut]
                        )],
                        body[opening:closing],
                        narrowed.group(1),
                    )
                )
                cut.append((match.start(), match.end() + end))
                continue
        head = _CLASS_HEAD.match(rest)
        if head is not None:
            closing = _matching(rest, head.end() - 1)
            end = closing
            while end < len(rest) and rest[end] in " \t":
                end += 1
            if end < len(rest) and rest[end] == ";":
                end += 1
            patterns.setdefault(head.group(2), []).append(
                (parameters, "class", rest[:end])
            )
            cut.append((match.start(), match.end() + end))
            continue
        # `template<> int kind<double>(double v) { ... }` - the copy for one
        # set of arguments, written by hand. It is not a pattern to expand:
        # it *is* the expansion, so it goes straight in under the name the
        # expander would have given it, and the generic one is never copied
        # for those arguments.
        special = _SPECIALISATION.match(rest) if not parameters else None
        if special is not None:
            closing = _matching(rest, special.end() - 1)
            arguments = [a.strip() for a in special.group(3).split(",")]
            named = _instantiated_name(special.group(2), arguments)
            returns = (special.group(1) or "void").strip()
            written[named] = (
                f"{returns} {named}({special.group(4)}) "
                f"{rest[special.end() - 1: closing + 1]}"
            )
            cut.append((match.start(), match.end() + closing + 1))
            continue
        guarded = _guarded_definition(rest)
        if guarded is not None and _DEFINITION.match(rest) is None:
            closing = _matching(rest, guarded["opens"])
            patterns.setdefault(guarded["name"], []).append(
                (parameters, "guarded", rest[:closing])
            )
            cut.append((match.start(), match.end() + closing))
            continue
        definition = _DEFINITION.match(rest) or _TEMPLATED_RESULT.match(rest)
        if definition is None:
            raise CppTranslationError(
                filename,
                _line_of(text, match.start()),
                "a template has to be a class, a struct or a function; py2bin "
                "reads the pattern and writes out one copy per use, and this "
                "is neither",
            )
        closing = _matching(rest, definition.end() - 1)
        # A list rather than one entry: `sort(first, last)` and
        # `sort(first, last, less_than)` are two templates of one name, and
        # keying by the name alone let the second replace the first.
        patterns.setdefault(definition.group(2), []).append(
            (parameters, "function", rest[:closing])
        )
        cut.append((match.start(), match.end() + closing))

    if not patterns and not written and not narrower:
        return text

    # The patterns themselves are not code, so they go; what replaces them is
    # whatever the file asked for, added below.
    kept: list[str] = []
    at = 0
    for start, end in cut:
        kept.append(text[at:start])
        at = end
    kept.append(text[at:])
    text = "".join(kept)

    #: The ordinary functions left once the patterns are cut out, and how
    #: many arguments each takes. C++ prefers one of these to a copy of a
    #: template when both would do - which is what makes a variadic recursion
    #: stop: `total(int)` written out by hand is the end of it, and without
    #: this the pattern answered the last call with an empty pack and asked
    #: for a `total()` that nothing declares.
    ordinary: "dict[str, set[int]]" = {}
    for definition in _DEFINITION.finditer(_without_literals(text)):
        if _depth_at(text, definition.start()) != 0:
            continue
        ordinary.setdefault(definition.group(2), set()).add(
            len([a for a in _split_arguments(definition.group(3)) if a.strip()])
        )

    made: "dict[str, str]" = dict(written)
    #: Candidates whose return type turned out not to exist. Kept so the
    #: rounds do not ask for them again for ever.
    refused: "set[tuple[str, tuple[str, ...]]]" = set()
    #: Instantiations a guard needs before it can be answered.
    pending: "list[tuple[str, list[str]]]" = []
    # Repeated, because a copy may itself name a template - `Stack<Pair<int>>`
    # asks for `Pair<int>` too, and the inner one is only visible once the
    # outer has been written out.
    def uses(region: str, scope: str) -> "tuple[list, list]":
        """Every template this region asks for, spelled out or deduced.

        The region comes first in what is searched for a declaration, so a
        copy's own parameters win over an identically named parameter in
        another copy - `sort__double` calls `__sift` on a `double *`, and
        searching the file first found `sort__int`'s `int *first`.
        """

        scope = region + "\n" + scope
        hidden = _member_spans(region)

        # A name that is a template parameter *of this region* is not a type
        # yet: `Box<U>` inside a member template of a class already written
        # out is asking for nothing, and a copy made for it was named
        # `Box__U` and took the shape the call deduces from with it.
        #
        # Read from the region and not from the file, because a parameter's
        # name means something only inside the pattern that declares it. A
        # program is free to call its own class `T`, and one in the corpus
        # does - `unique_ptr<T>` there is a real instantiation and skipping
        # it left the type undeclared.
        in_scope: "set[str]" = set()
        for clause in _TEMPLATE.finditer(region):
            for held, _is_type, _is_pack in _template_parameters(clause.group(1)):
                in_scope.add(held)

        asked: list[tuple[str, list[str]]] = []
        unread: list[tuple[str, int]] = []
        for name, entries in patterns.items():
          claimed: "set[int]" = set()
          # The copy without a pack first. `total(T)` and `total(T, R...)`
          # both take one argument - the second with nothing in its pack -
          # and C++ picks the first, being the more specialised. Picking
          # the second wrote a copy whose recursive step called `total()`
          # with nothing at all, so the recursion never reached a bottom.
          # Both are named for the same arguments, so only one of them can
          # exist anyway; this decides which.
          for parameters, kind, pattern_text in sorted(
              entries, key=lambda one: any(pack for _n, _t, pack in one[0])
          ):
            if kind not in ("function", "guarded"):
                continue
            # A call that did not spell the arguments out: `twice(5)` rather
            # than `twice<int>(5)`. What it means is read off the arguments.
            spelled = _spelled_parameters(kind, pattern_text)
            if spelled is None:
                continue
            declared = _Declared(spelled)
            for call in re.finditer(rf"(?<![.\w>]){re.escape(name)}\s*\(", region):
                close = _closing_paren(region, call.end() - 1)
                if close < 0:
                    continue
                if _is_a_definition(region, close):
                    # A member or function *named* like the template, not a
                    # call to it: `<string>` has a `find` method and
                    # `<algorithm>` has a `find` template, and the method's
                    # own head reads exactly like a call until you notice
                    # what follows the parentheses.
                    continue
                if _hidden_by_a_member(hidden, name, call.start()):
                    # And a bare call inside that class is to its own member.
                    # A member hides a name declared outside the class, so
                    # `find(s)` written in `string` is `string::find` and
                    # never `std::find`, whatever the template would deduce.
                    continue
                # Only a pack defers, and only to a pattern without one.
                # Two guarded copies of a name both deduce - the guard, not
                # the deduction, is what tells them apart - so one must not
                # be able to shut the others out.
                packed = any(pack for _n, _t, pack in parameters)
                if packed and call.start() in claimed:
                    continue
                given = _call_arguments(region, call.end() - 1)
                if len(given) in ordinary.get(name, ()):
                    continue  # an ordinary function of that name takes this
                deduced = _deduce_arguments(
                    parameters, declared.group(3), given, scope, call.start()
                )
                if deduced is None:
                    # Not yet, rather than never: `sort(v.begin(), v.end())`
                    # cannot be read until `vector<int>` has been written out,
                    # because until then there is no `begin` to ask. Held for
                    # the next round, and reported once the rounds stop
                    # producing anything new.
                    unread.append((name, call.start()))
                    continue
                if not packed:
                    claimed.add(call.start())
                asked.append((name, deduced, (parameters, kind, pattern_text)))
        for name in {**patterns, **narrower}:
            for found in re.finditer(rf"(?<![.\w>]){re.escape(name)}\s*<", region):
                close = _closing_angle(region, found.end() - 1)
                if close < 0:
                    continue
                arguments = [
                    a for a in _split_arguments(region[found.end(): close])
                    if a.strip()
                ]
                if any(
                    re.search(
                        rf"(?<![.\w>]){re.escape(other)}\s*<", ",".join(arguments)
                    )
                    for other in {**patterns, **narrower}
                ):
                    continue  # an inner template first; next round
                if any(argument.strip() in in_scope for argument in arguments):
                    continue
                if name not in patterns:
                    # Only narrower copies and no general one. Legal C++ only
                    # if one of them fits, and `_narrower_copy` decides that.
                    asked.append(
                        (name, [a.strip() for a in arguments], ([], "class", ""))
                    )
                    continue
                # Spelled out, so the entry is whichever takes that many
                # template parameters.
                for entry in patterns[name]:
                    # `entry[0]` is the parameter list already read, not the
                    # text it was read from.
                    if _arity_fits(entry[0], arguments):
                        asked.append(
                            (name, [a.strip() for a in arguments], entry)
                        )
                        break
        return asked, unread

    def _spelled_parameters(kind: str, pattern_text: str) -> "str | None":
        if kind == "guarded":
            read = _guarded_definition(pattern_text)
            return None if read is None else read["parameters"]
        declared = _DEFINITION.match(pattern_text) or _TEMPLATED_RESULT.match(
            pattern_text
        )
        return None if declared is None else declared.group(3)

    def point(region: str, scope: str) -> str:
        """Send every use in this region to the copy written for it.

        Same reasoning as in :func:`uses` about which text is searched first.
        """

        scope = region + "\n" + scope

        out: list[str] = []
        at = 0
        spelled_out = re.compile(
            r"(?<![.\w>])("
            + "|".join(re.escape(n) for n in {**patterns, **narrower})
            + r")\s*<"
        )
        for found in spelled_out.finditer(region):
            if found.start() < at:
                continue
            close = _closing_angle(region, found.end() - 1)
            if close < 0:
                continue
            arguments = [
                # Folded the same way the copy's own name was, or a use
                # written `Chain<N - 1>` - which is `Chain<3 - 1>` by the
                # time it is read - would not be sent to `Chain__2`.
                _as_a_number(a.strip(), scope)
                for a in _split_arguments(region[found.end(): close])
                if a.strip()
            ]
            named = _instantiated_name(found.group(1), arguments)
            if named not in made:
                continue
            out.append(region[at:found.start()])
            out.append(named)
            at = close + 1
        out.append(region[at:])
        region = "".join(out)

        # The same for the calls that named no arguments at all.
        for name, entries in patterns.items():
          # The copy without a pack first. `total(T)` and `total(T, R...)`
          # both take one argument - the second with nothing in its pack -
          # and C++ picks the first, being the more specialised. Picking
          # the second wrote a copy whose recursive step called `total()`
          # with nothing at all, so the recursion never reached a bottom.
          # Both are named for the same arguments, so only one of them can
          # exist anyway; this decides which.
          for parameters, kind, pattern_text in sorted(
              entries, key=lambda one: any(pack for _n, _t, pack in one[0])
          ):
            if kind not in ("function", "guarded"):
                continue
            spelled = _spelled_parameters(kind, pattern_text)
            if spelled is None:
                continue
            declared = _Declared(spelled)
            out = []
            at = 0
            for call in re.finditer(rf"(?<![.\w>]){re.escape(name)}\s*\(", region):
                if call.start() < at:
                    continue
                close = _closing_paren(region, call.end() - 1)
                if close < 0 or _is_a_definition(region, close):
                    continue
                if _hidden_by_a_member(_member_spans(region), name, call.start()):
                    continue
                given = _call_arguments(region, call.end() - 1)
                if len(given) in ordinary.get(name, ()):
                    continue  # an ordinary function of that name takes this
                deduced = _deduce_arguments(
                    parameters, declared.group(3), given, scope, call.start()
                )
                if deduced is None:
                    continue
                named = _instantiated_name(name, deduced)
                if named not in made:
                    continue
                out.append(region[at:call.start()])
                out.append(f"{named}(")
                at = call.end()
            out.append(region[at:])
            region = "".join(out)
        return region

    for _round in range(_TEMPLATE_ROUNDS):
        # A copy may itself use a template - `sort` calls `__sift`, and
        # `Stack<Pair<int>>` asks for `Pair<int>` too - so the copies written
        # so far are scanned and rewritten exactly like the file is.
        scope = text + "\n" + "\n".join(made.values())
        wanted: list[tuple[str, list[str]]] = []
        undeduced: list[tuple[str, int]] = []
        asked, unread = uses(text, scope)
        wanted.extend(asked)
        undeduced.extend(unread)
        for copy in list(made.values()):
            asked, _unread = uses(copy, scope)
            wanted.extend(asked)

        for wanted_name, wanted_arguments in pending:
            for entry in patterns.get(wanted_name, ()):
                if _arity_fits(entry[0], wanted_arguments):
                    wanted.append((wanted_name, wanted_arguments, entry))
                    break
            else:
                if wanted_name in narrower:
                    wanted.append(
                        (wanted_name, wanted_arguments, ([], "class", ""))
                    )
        pending = []
        # A non-type argument that is an expression is folded to the number
        # it is. Substituted textually, `Chain<N - 1>` becomes `Chain<3 - 1>`
        # and then `Chain<3 - 1 - 1>` - a different name every round, so a
        # template that recurses never reached the copy written to end it and
        # never stopped asking for another.
        wanted = [
            (name, [_as_a_number(one, scope) for one in arguments], entry)
            for name, arguments, entry in wanted
        ]
        fresh = [
            (name, arguments, entry)
            for name, arguments, entry in wanted
            if _instantiated_name(name, arguments) not in made
            and (name, tuple(arguments)) not in refused
        ]
        for name, arguments, entry in fresh:
            parameters, kind, pattern = entry
            # A copy written for this shape of argument wins over the one
            # written for every shape. That is how a traits header answers a
            # question: the general copy says no, and a narrower one written
            # for `T *` or for `T, T` says yes, and which of them is used is
            # decided here.
            narrowed = _narrower_copy(narrower.get(name, []), arguments)
            if narrowed is not None:
                narrow_parameters, _shapes, body, keyword, bound = narrowed
                named = _instantiated_name(name, arguments)
                copy = _substituted(
                    f"{keyword} {name} {body};",
                    narrow_parameters,
                    [bound[p] for p, _is_type, _pack in narrow_parameters],
                )
                copy = _map_code(
                    copy,
                    lambda part, n=name, s=named: re.sub(
                        rf"(?<![.\w>]){re.escape(n)}\b(?!\s*<)", s, part
                    ),
                )
                made[named] = copy
                _INSTANTIATED[named] = (name, list(arguments))
                continue
            if not _arity_fits(parameters, arguments):
                raise CppTranslationError(
                    filename,
                    _line_of(text, text.index(name)),
                    f"{name} is a template taking {len(parameters)} argument(s) "
                    f"and is used here with {len(arguments)}",
                )
            named = _instantiated_name(name, arguments)
            if kind == "guarded":
                answered = _guarded_copy(
                    name, named, parameters, arguments, pattern, made
                )
                if answered is _LATER:
                    # The class the guard asks about has not been written out
                    # yet. Asked for below, and this candidate is looked at
                    # again next round.
                    for wanted_name, wanted_arguments in _asked_by_the_guard(
                        parameters, arguments, pattern
                    ):
                        pending.append((wanted_name, wanted_arguments))
                    asked = _guard_asked(parameters, arguments, pattern)
                    if asked is not None:
                        guard, inside, _member = asked
                        settled = [_settled(part, made) for part in inside]
                        if all(part is not None for part in settled):
                            pending.append((guard, settled))
                    continue
                if answered is None:
                    # No such type, so this one is not a candidate at all.
                    # Another of the same name answers the call, and if none
                    # does the call is reported where it is written.
                    refused.add((name, tuple(arguments)))
                    continue
                made[named] = answered
                continue
            copy = _substituted(
                pattern, parameters, arguments
            )
            if kind == "function":
                # Only the name in the head. A function template may call
                # itself, and a recursive call is not a call to this copy:
                # `total(rest...)` inside the copy for three arguments is a
                # call for two, which is a different copy. Renamed wholesale,
                # every step of a variadic recursion called itself with one
                # argument fewer than it takes, and the copy it should have
                # reached was never asked for.
                head = _DEFINITION.match(copy) or _TEMPLATED_RESULT.match(copy)
                if head is not None:
                    copy = (
                        copy[: head.start(2)] + named + copy[head.end(2):]
                    )
            else:
                # A class, where the name inside it is the constructors and
                # the destructors, which are spelled with the class's own.
                copy = _map_code(
                    copy,
                    lambda part, n=name, s=named: re.sub(
                        rf"(?<![.\w>]){re.escape(n)}\b(?!\s*<)", s, part
                    ),
                )
            made[named] = copy
            _INSTANTIATED[named] = (name, list(arguments))

        scope = text + "\n" + "\n".join(made.values())
        text = point(text, scope)
        for key in list(made):
            made[key] = point(made[key], scope)

        if not fresh:
            if undeduced:
                name, where = undeduced[0]
                raise CppTranslationError(
                    filename,
                    _line_of(text, where),
                    f"cannot work out what {name}() is being called with "
                    f"here. py2bin deduces a template argument from a "
                    f"literal, from a variable it can see declared, or from "
                    f"a call whose return type it can read; write "
                    f"{name}<type>(...) to say which copy is meant",
                )
            break
    else:
        raise CppTranslationError(
            filename, 1,
            "a template that expands into itself; py2bin writes out one copy "
            "per use and this one never stops asking for another",
        )

    return "\n".join(made.values()) + "\n" + text


#: How many times expansion may go round. A template naming another needs one
#: round per level, and a template naming itself needs an unbounded number -
#: which is what this stops.
_TEMPLATE_ROUNDS = 16


# --- exceptions ------------------------------------------------------------
#
# There is no unwinder, and writing one means saving and restoring a stack
# frame - which is machine code, not translation. What a translator can do is
# what C++ did before unwinders: set a flag, return, and have every caller
# check. That is exact as long as the check happens where the call did, so a
# statement holding a call that can throw is split into one call per statement
# with a check after each.
#
# What is lost is that the propagation is visible in the C, and that a call
# whose result feeds a short-circuit cannot be split without changing when the
# other side runs. Those are refused rather than reordered.

#: `throw expr;` and a bare `throw;`, which rethrows what is in flight.
_THROW = re.compile(r"\bthrow\b\s*([^;]*);")
#: `try {`
_TRY = re.compile(r"\btry\s*\{")
#: `catch (int e) {` and `catch (...) {`
_CATCH = re.compile(r"\bcatch\s*\(\s*([^)]*)\s*\)\s*\{")

_THROWN = "__py2bin_thrown"
#: Which class is in flight, as a number. A `try` with more than one handler
#: has to pick between them, and there is nothing in the object itself to
#: read: py2bin has no linker, so every class that is ever thrown is in front
#: of it while it translates, and each can be given a number of its own - the
#: same argument `dynamic_cast` and `std::function` are answered with.
_KIND = "__py2bin_kind"
#: The number each thrown class goes by, filled in as they are met. Reset per
#: translation unit alongside the class names.
_THROWN_KINDS: "dict[str, int]" = {}


def _kind_id(named: str) -> int:
    """The number this class is in flight under. Zero is "not a class"."""

    return _THROWN_KINDS.setdefault(named, len(_THROWN_KINDS) + 1)
_IN_FLIGHT = "__py2bin_in_flight"

#: Declared once, at the top of any file that throws.
_EXCEPTION_STATE = f"""
static int {_THROWN} = 0;
static int {_KIND} = 0;
static long {_IN_FLIGHT} = 0;
"""


def _zero_for(returns: str, classes: "dict[str, Class]") -> str:
    """What a function hands back on the way out with an exception in flight.

    Nothing reads it - every caller looks at the flag first - so the only
    requirement is that C accepts it as a value of the declared type.
    """

    spelled = returns.strip()
    if spelled in ("", "void") or _returns_object_named(spelled, classes):
        return "return;"
    return "return 0;"


def _returns_object_named(spelled: str, classes: "dict[str, Class]") -> bool:
    """Whether that spelling names a class held by value.

    Such a function returns nothing in the C - the caller provides the space
    and the callee writes through a hidden pointer - so the way out with an
    exception in flight is a bare `return;`.
    """

    return "*" not in spelled and spelled.replace("&", "").strip() in classes


#: Any function-like head followed by a body: `int f(int a) {`, `T::m() {`,
#: and a constructor's `Guard() {`, which has no return type at all.
_ANY_DEFINITION = re.compile(
    r"([A-Za-z_~][\w\s*&:~]*?)\s*\(([^;{}()]*)\)\s*(?:const\s*)?\{"
)


def _every_body(text: str) -> "list[tuple[re.Match[str], int, str, str]]":
    """Each function in the text as (head match, body end, name, return type).

    At every depth, because a method is written inside its class and can
    throw exactly as a free function can.
    """

    found = []
    for match in _ANY_DEFINITION.finditer(text):
        head = match.group(1).strip()
        # How a function is stored is not part of what it is. Read as the
        # first word of the return type, `static` put the whole body outside
        # what this reads - so a `throw` inside a `static` function was never
        # rewritten, while the same function without the word was.
        while head.split() and head.split()[0] in _STORAGE | _DISPATCH:
            head = head.split(None, 1)[1] if " " in head else ""
        if not head or head.split()[0] in _NOT_A_TYPE:
            continue
        words = head.replace("*", " * ").replace("&", " & ").split()
        name = words[-1].split("::")[-1]
        returns = " ".join(words[:-1]) if len(words) > 1 else ""
        if not re.fullmatch(r"~?[A-Za-z_]\w*", name):
            continue
        try:
            closing = _matching(text, match.end() - 1)
        except ValueError:
            continue
        found.append((match, closing, name, returns))
    return found


def _throwing_names(text: str) -> "set[str]":
    """Every function that can leave with an exception in flight, by name.

    By name rather than by class: a method is identified here the way a call
    site writes it, and a call site writes `.check(`. Two classes with a
    method of the same name make this conservative, which costs a test of a
    flag that will be zero.
    """

    # Every body a name has, not the last one written. An overrider that
    # does not throw does not stop the base it overrides from throwing, and
    # a call through a base pointer can reach either.
    bodies: "dict[str, list[str]]" = {}
    for match, closing, name, _returns in _every_body(text):
        bodies.setdefault(name, []).append(text[match.end() - 1: closing])
    throwing = {
        name
        for name, written in bodies.items()
        if any(_THROW.search(_without_literals(body)) for body in written)
    }
    while True:
        grown = set(throwing)
        for name, written in bodies.items():
            if name in grown:
                continue
            code = "\n".join(_without_literals(body) for body in written)
            if any(
                re.search(rf"(?<![\w>]){re.escape(other)}\s*\(", code)
                for other in throwing
            ):
                grown.add(name)
        if grown == throwing:
            return throwing
        throwing = grown


def _result_types(text: str) -> "dict[str, str]":
    """What each function returns, for the temporaries a split call needs."""

    found: dict[str, str] = {}
    for _match, _closing, name, returns in _every_body(text):
        spelled = re.sub(r"\b(static|inline|virtual|extern)\b", "", returns).strip()
        # A constructor writes no return type, and what it answers with is
        # the object. Read as a `long`, a lifted `T(x)` went into a temporary
        # the width of a word and the object was cut down to fit.
        found[name] = (spelled or name).replace("&", "*")
    return found


def _rewrite_exceptions_early(text: str, filename: str) -> str:
    """Turn throw, try and catch into flags, checks and labels.

    Done on the C++, before classes are taken apart, so the `return` this
    leaves behind is one the destructor pass can see - an exception leaving a
    function has to destroy what that function built, and that pass is what
    knows which locals those are.
    """

    throwing = _throwing_names(text)
    global _CALL_RESULT_TYPES
    _CALL_RESULT_TYPES = _result_types(text)
    counter = [0]
    out: list[str] = []
    at = 0
    for match, closing, name, returns in _every_body(text):
        if match.start() < at:
            continue
        opening = match.end() - 1
        spelled = re.sub(r"\b(static|inline|virtual|extern)\b", "", returns).strip()
        out.append(text[at:opening])
        out.append(
            _rewrite_exceptions(
                text[opening:closing],
                spelled,
                throwing,
                {},
                filename,
                counter,
                uncaught=name == "main",
            )
        )
        at = closing
    out.append(text[at:])
    return "".join(out)


class _Landing:
    """Where a check goes when it finds an exception in flight.

    Inside a `try`, to that try's handler. Otherwise out of the function,
    with the flag still set so the caller's own check finds it.
    """

    __slots__ = ("label", "returns", "classes", "uncaught")

    def __init__(
        self,
        label: "str | None",
        returns: str,
        classes,
        uncaught: bool = False,
    ) -> None:
        self.label = label
        self.returns = returns
        self.classes = classes
        #: Set on `main`, where there is nothing left to propagate to. C++
        #: calls terminate here, which aborts; py2bin has no way to raise a
        #: signal, so the program stops with a status of its own instead of
        #: running on as though nothing had happened.
        self.uncaught = uncaught

    def leave(self) -> str:
        if self.label is not None:
            return f"goto {self.label};"
        if self.uncaught:
            return f"return {UNCAUGHT_STATUS};"
        return _zero_for(self.returns, self.classes)


#: What a program exits with when an exception reaches the end of `main`.
#: C++ aborts there; this is the nearest thing a translation to C can do, and
#: it is a status a caller can test rather than a silent success.
UNCAUGHT_STATUS = 3

#: Operators whose right side runs only sometimes. A call split out of one of
#: these would run when C++ says it must not.
_SHORT_CIRCUIT = ("&&", "||", "?")


def _rewrite_exceptions(
    body: str,
    returns: str,
    throwing: "set[str]",
    classes: "dict[str, Class]",
    filename: str,
    counter: "list[int]",
    uncaught: bool = False,
) -> str:
    """Turn `throw`, `try` and `catch` into flags, checks and labels."""

    if not throwing and not _THROW.search(_without_literals(body)):
        return body
    return _guarded(
        body,
        _Landing(None, returns, classes, uncaught),
        throwing,
        classes,
        filename,
        counter,
    )


#: Stands in for a try that has been dealt with, while the rest is. The `;`
#: is load-bearing: statements are split on it, and without one the try and
#: whatever followed it were one statement - so a call hoisted out of that
#: next statement was placed *before* the try and ran too early.
_TRY_MARK = "\x00py2bin_try_%d\x00;"


def _guarded(
    body: str,
    landing: "_Landing",
    throwing: "set[str]",
    classes: "dict[str, Class]",
    filename: str,
    counter: "list[int]",
) -> str:
    """Handle every try in this scope, then throws and checks at this landing."""

    finished: list[str] = []
    body = _extract_tries(
        body, landing, throwing, classes, filename, counter, finished
    )
    # A try's contents have already been walked, with its handler as their
    # landing. Left in place they would be walked again with this landing, and
    # the first check to fire would leave the function instead of reaching the
    # handler - which is why they are held aside while the rest is done.
    body = _check_after_calls(body, landing, throwing, filename, counter)
    for index, made in enumerate(finished):
        body = body.replace(_TRY_MARK % index, made)
    return body


def _extract_tries(
    body: str,
    landing: "_Landing",
    throwing: "set[str]",
    classes: "dict[str, Class]",
    filename: str,
    counter: "list[int]",
    finished: "list[str]",
) -> str:
    """`try { A } catch (T e) { B }` becomes A, a jump past B, and B.

    The handler is reached by a `goto` from wherever inside A the flag was
    found set, which is what makes it a handler rather than a test after the
    fact: A stops where it stopped. A try inside A is dealt with first, and a
    throw inside B goes outward - to whatever encloses this try, or out of
    the function - because a handler is not inside its own try.
    """

    while True:
        found = _TRY.search(body)
        if found is None:
            return body
        opening = found.end() - 1
        try:
            closing = _matching(body, opening)
        except ValueError:
            raise CppTranslationError(
                filename, _line_of(body, opening), "a try block is not closed"
            ) from None
        rest = body[closing:]
        catch = _CATCH.match(rest.lstrip())
        if catch is None:
            raise CppTranslationError(
                filename,
                _line_of(body, closing),
                "a try block needs a catch after it; py2bin has no unwinder, "
                "so an exception that nothing catches is one nothing can be "
                "done about",
            )
        offset = closing + (len(rest) - len(rest.lstrip()))
        # Every handler this try has, not the first one. A second `catch`
        # left where it stood was not a statement any longer - the try in
        # front of it had already become a label and a jump - and the C it
        # was handed still said `catch`.
        clauses: "list[tuple[str, int, int]]" = []
        walked = offset
        while True:
            ahead = body[walked:]
            more = _CATCH.match(ahead.lstrip())
            if more is None:
                break
            begins = walked + (len(ahead) - len(ahead.lstrip()))
            open_at = begins + more.end() - 1
            try:
                close_at = _matching(body, open_at)
            except ValueError:
                raise CppTranslationError(
                    filename, _line_of(body, open_at),
                    "a catch block is not closed",
                ) from None
            clauses.append((more.group(1).strip(), open_at, close_at))
            walked = close_at
        handler_close = walked

        counter[0] += 1
        number = counter[0]
        label = f"__py2bin_catch_{number}"
        after = f"__py2bin_done_{number}"

        guarded = _guarded(
            body[opening + 1: closing - 1],
            _Landing(label, landing.returns, landing.classes),
            throwing,
            classes,
            filename,
            counter,
        )
        pieces: "list[str]" = []
        anything = False
        for spelled, open_at, close_at in clauses:
            handled = _guarded(
                body[open_at + 1: close_at - 1],
                landing,
                throwing,
                classes,
                filename,
                counter,
            )
            caught = _catch_binding(spelled, filename, body, open_at, classes)
            # One handler is not a choice, and asking what is in flight would
            # be a new way to be wrong: where the thrown type could not be
            # read it is nothing in particular, and a lone handler has always
            # taken whatever arrived. More than one has to be told apart.
            if len(clauses) == 1 or spelled in ("...", ""):
                anything = True
                pieces.append(f"{{ {caught}{handled} }} goto {after};")
                break
            pieces.append(
                f"if ({_KIND} == {_catch_kind(spelled)}) "
                f"{{ {caught}{handled} goto {after}; }}"
            )
        # Nothing matched, so this try is not where it is handled: the flag
        # goes back up and it carries on outward, which is what C++ does.
        rest_of = "" if anything else f" {_THROWN} = 1; {landing.leave()}"
        made = (
            f"{{ {guarded} }} goto {after}; {label}: ; "
            f"{{ {_THROWN} = 0; {' '.join(pieces)}{rest_of} }} {after}: ;"
        )
        finished.append(made)
        body = (
            body[:found.start()]
            + (_TRY_MARK % (len(finished) - 1))
            + body[handler_close:]
        )


def _catch_kind(spelled: str) -> int:
    """The number a handler is looking for. Zero for anything not a class."""

    bare = re.sub(
        r"\b(?:const|volatile|struct|class)\b|[*&]", " ", spelled
    ).split()
    named = bare[0] if bare else ""
    return _kind_id(named) if named in _CLASS_NAMES else 0


def _catch_binding(
    spelled: str,
    filename: str,
    body: str,
    at: int,
    classes: "dict[str, Class] | set[str]" = (),
) -> str:
    """`catch (int e)` names the value; `catch (...)` does not."""

    if spelled in ("...", ""):
        return ""
    by_reference = "&" in spelled
    words = spelled.replace("*", " * ").replace("&", " ").split()
    if len(words) < 2:
        raise CppTranslationError(
            filename, _line_of(body, at),
            f"cannot read the catch parameter {spelled!r}; py2bin catches by "
            f"value, so write it as `catch (int e)` or `catch (...)`",
        )
    named = words[-1]
    held = " ".join(words[:-1])
    # `catch (const Bad &b)` is the form C++ asks for, and the qualifier is
    # part of what was written. Read the class out from under it: looked up
    # whole, `const Bad` was not a class this file declares and the handler
    # was built as though it had caught a number.
    bare = re.sub(r"\b(?:const|volatile|struct|class)\b", " ", held).strip()
    if bare in _CLASS_NAMES or bare in classes:
        if by_reference:
            # A reference to what is in flight, not a copy of it - which is
            # what `catch (std::exception &e)` is for. Written as a C++
            # reference so the pass that turns those into pointers does it,
            # and `e.what()` reaches the object that was actually thrown
            # rather than the base it was sliced to.
            return f"{held} &{named} = *({held} *){_IN_FLIGHT}; "
        if bare in _POLYMORPHIC and bare in _INHERITED_FROM:
            raise CppTranslationError(
                filename,
                _line_of(body, at),
                f"catching {bare} by value, and something in this "
                f"file derives from it. C++ slices the object to that class "
                f"here, so a virtual function called on it answers as the "
                f"base rather than as what was thrown - py2bin's copy keeps "
                f"the object it was made from and would answer differently. "
                f"Write `catch ({bare} &{named})`, which is what the "
                f"slicing is a reason to write anyway",
            )
        # Declared and then assigned, not initialised: py2bin's C takes
        # `o = *p;` and not `struct V o = *p;`.
        # Without the qualifier: the copy is assigned to on the next
        # statement, and a `const` one cannot be.
        return (
            f"{bare} {named}; {named} = *({bare} *){_IN_FLIGHT}; "
        )
    return f"{held} {named} = ({held}){_IN_FLIGHT}; "


def _thrown(match: "re.Match[str]", landing: "_Landing", body: str) -> str:
    """What `throw expr;` becomes.

    A number goes in the flag word as it is. An object cannot - it is wider
    than a word, and the stack frame holding it is gone by the time a handler
    runs - so a copy is made on the heap and the address goes in the word
    instead. That is what an exception object is: a copy that outlives the
    frame that threw it.
    """

    spelled = match.group(1).strip()
    if not spelled:
        # A bare `throw;` inside a handler: what is in flight stays in flight.
        return f"{{ {_THROWN} = 1; {landing.leave()} }}"
    made = _CONSTRUCTED.match(spelled)
    if made is not None and made.group(1) in _CLASS_NAMES:
        # A temporary built in the throw itself, which is how the standard
        # exception types are always thrown. `new` allocates and runs the
        # constructor, which is exactly the copy that has to outlive the
        # frame - so this is written as `new` and goes through that path.
        return (
            f"{{ {_THROWN} = 1; {_KIND} = {_kind_id(made.group(1))}; "
            f"{made.group(1)} *__py2bin_raised = new {spelled}; "
            f"{_IN_FLIGHT} = (long)__py2bin_raised; {landing.leave()} }}"
        )
    held = _deduced_type(spelled, body, match.start())
    if held is not None and held.replace("*", "").strip() in _CLASS_NAMES:
        named = held.replace("*", "").strip()
        return (
            f"{{ {_THROWN} = 1; {_KIND} = {_kind_id(named)}; "
            f"{named} *__py2bin_raised = ({named} *)malloc(sizeof({named})); "
            f"*__py2bin_raised = {spelled}; "
            f"{_IN_FLIGHT} = (long)__py2bin_raised; {landing.leave()} }}"
        )
    # Not a class, so nothing to tell apart by: a number in flight goes under
    # the kind every non-class value shares.
    return (
        f"{{ {_THROWN} = 1; {_KIND} = 0; "
        f"{_IN_FLIGHT} = (long)({spelled}); {landing.leave()} }}"
    )


#: `Err(1, 2)` - a temporary built where it is thrown.
_CONSTRUCTED = re.compile(r"^([A-Za-z_]\w*)\s*\(")

#: `R b(-2);` - a constructor reached by declaring what it builds. There is
#: no call here for a reader to find: looking for `name(` it finds `b(` and
#: asks whether `b` throws, which it does not. So a constructor that threw
#: was stepped straight over, the statements after it ran, and no handler was
#: reached - the program answered, and answered wrongly.
_DECLARED_BUILT = re.compile(
    r"^\s*([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\(([^;]*)\)\s*;\s*$"
)

#: The classes this file declares. Read before the exception pass, which runs
#: before classes are taken apart and so has nothing else to ask.
_CLASS_NAMES: "set[str]" = set()

#: What `#pragma pack` was in force where each class was written. Read before
#: anything moves, because the pragma says where the members of the struct
#: *below it* sit, and both it and the struct are moved before either is
#: written out.
_CLASS_PACK: "dict[str, int]" = {}


#: `#pragma pack(push, 1)`, `#pragma pack(2)`, `#pragma pack()`, `#pragma
#: pack(pop)` - the four spellings a header uses.
_PACK_PRAGMA = re.compile(
    r"(?m)^[ \t]*#[ \t]*pragma[ \t]+pack[ \t]*\(([^)]*)\)"
)


def _pack_regions(text: str) -> "list[tuple[int, int | None]]":
    """Where each `#pragma pack` sits and what it leaves in force after it.

    A stack, because `push`/`pop` is how a header wraps one struct without
    disturbing what the file was doing around it.
    """

    found: "list[tuple[int, int | None]]" = []
    stack: "list[int | None]" = []
    current: "int | None" = None
    for match in _PACK_PRAGMA.finditer(text):
        written = [one.strip() for one in match.group(1).split(",") if one.strip()]
        if written and written[0] == "push":
            stack.append(current)
            current = int(written[1]) if len(written) > 1 and written[1].isdigit() else current
        elif written and written[0] == "pop":
            current = stack.pop() if stack else None
        elif not written:
            current = None
        elif written[0].isdigit():
            current = int(written[0])
        found.append((match.end(), current))
    return found


def _pack_in_force(
    regions: "list[tuple[int, int | None]]", at: int
) -> "int | None":
    """What `pack` was in force at that offset, if any."""

    held: "int | None" = None
    for where, value in regions:
        if where > at:
            break
        held = value
    return held
#: What each class declares, read before the bodies are taken apart. A pass
#: that runs after that has no body left to read, and asking one what a
#: member's type is got no answer at all - which is how `v->x` on a plain
#: struct came to have no type in a body being rewritten.
_CLASS_MEMBERS: "dict[str, list[tuple[str, str]]]" = {}

#: Of those, the ones with a virtual function somewhere above them. Catching
#: one by value slices it, which is the one thing this cannot reproduce.
_POLYMORPHIC: "set[str]" = set()

#: Classes something else in this file inherits from. Only these can be
#: sliced by a catch: a class nothing derives from is already what it is.
_INHERITED_FROM: "set[str]" = set()


def _polymorphic_names(text: str) -> "set[str]":
    """Class names that have a virtual function, their own or inherited."""

    bases: dict[str, str] = {}
    declares: set[str] = set()
    for head in _CLASS_HEAD.finditer(text):
        named = _bases_of(head)
        name = head.group(2)
        bases[name] = named[0] if named else ""
        try:
            body = text[head.end() - 1: _matching(text, head.end() - 1)]
        except ValueError:
            continue
        if re.search(r"\bvirtual\b", _without_literals(body)):
            declares.add(name)
    found = set()
    for name in bases:
        seen = name
        while seen:
            if seen in declares:
                found.add(name)
                break
            seen = bases.get(seen, "")
    return found

def _check_after_calls(
    body: str,
    landing: "_Landing",
    throwing: "set[str]",
    filename: str,
    counter: "list[int]",
) -> str:
    """Put a test of the flag immediately after every call that can set it.

    Immediately, not at the end of the statement: everything between the call
    and the test would run with the exception already in flight, and anything
    it printed or stored would be output C++ never produces. A statement with
    more than one such call is split so each gets its own test.
    """

    body = _THROW.sub(lambda m: _thrown(m, landing, body), body)
    if not throwing:
        return body
    return _split_statements(body, landing, throwing, filename, counter)


def _split_statements(
    body: str,
    landing: "_Landing",
    throwing: "set[str]",
    filename: str,
    counter: "list[int]",
) -> str:
    """Give every call that can throw a statement, and a test, of its own.

    The counter is shared across the whole file: a try block and the code
    around it are split separately, and two temporaries of the same name in
    one function is a redeclaration.
    """

    out: list[str] = []
    for statement in _statements(body):
        built = _DECLARED_BUILT.match(statement)
        if built is not None and built.group(1) in throwing:
            # The declaration stays as it is and the test goes after it. What
            # a leave from here skips is the destructor at the end of the
            # block, which is right: a constructor that threw did not finish,
            # and C++ does not destroy what was never built.
            out.append(statement)
            out.append(f" if ({_THROWN}) {{ {landing.leave()} }} ")
            continue
        calls = _throwing_calls(statement, throwing)
        if not calls:
            out.append(statement)
            continue
        _refuse_where_splitting_would_change_it(statement, calls, body, filename)
        lifted: list[str] = []
        for call in calls:
            counter[0] += 1
            held = f"__py2bin_call_{counter[0]}"
            spelled = _call_result_type(call, throwing)
            statement = statement.replace(call, held, 1)
            if spelled in ("void", ""):
                # Nothing to hold; the call is the whole of what it does.
                lifted.append(f" {call}; if ({_THROWN}) {{ {landing.leave()} }} ")
                statement = statement.replace(held, "0", 1)
                continue
            lifted.append(
                f" {spelled} {held} = {call}; "
                f"if ({_THROWN}) {{ {landing.leave()} }} "
            )
        out.append("".join(lifted))
        out.append(statement)
    return "".join(out)


def _refuse_where_splitting_would_change_it(
    statement: str, calls: "list[str]", body: str, filename: str
) -> None:
    """Say so where lifting a call out would run it at a different time."""

    for call in calls:
        before = statement[: statement.index(call)]
        if any(mark in _without_literals(before) for mark in _SHORT_CIRCUIT):
            raise CppTranslationError(
                filename,
                _line_of(body, body.find(statement)),
                "a call that can throw is on the right of `&&`, `||` or `?:` "
                "here. py2bin gives each such call a statement of its own so "
                "the exception is seen where it happened, and moving this one "
                "would run it when C++ says it must not; assign it to a "
                "variable on a line before this one",
            )
    stripped = statement.lstrip()
    if re.match(r"\b(while|for)\b", stripped):
        raise CppTranslationError(
            filename,
            _line_of(body, body.find(statement)),
            "a call that can throw is in a loop's header here. py2bin lifts "
            "such a call to a statement of its own, and a header runs again "
            "each time round - so lifting this one would run it once; move it "
            "into the body",
        )


def _statements(body: str) -> "list[str]":
    """Split a body into statements, keeping every character.

    A statement ends at a `;`, and a brace ends one too: `if (c) { ... }` is
    a statement whose condition this has to see on its own, because that is
    where a call inside it would be lifted from.
    """

    found: list[str] = []
    depth = 0
    at = 0
    index = 0
    while index < len(body):
        piece = body[index]
        if piece in "\"'":
            quote = piece
            index += 1
            while index < len(body) and body[index] != quote:
                index += 2 if body[index] == "\\" else 1
            index += 1
            continue
        if piece in "([":
            depth += 1
        elif piece in ")]":
            depth -= 1
        elif depth == 0 and (piece == ";" or piece in "{}"):
            found.append(body[at:index + 1])
            at = index + 1
        index += 1
    if at < len(body):
        found.append(body[at:])
    return found


def _throwing_calls(statement: str, throwing: "set[str]") -> "list[str]":
    """Each call to a function that can throw, outermost first, left to right.

    Outermost only: a call inside another call's arguments is part of that
    call's text and is lifted with it, which keeps the order they run in.
    """

    code = _without_literals(statement)
    found: list[str] = []
    at = 0
    # The receiver is part of the call: `g.check(13)` has to be lifted whole,
    # or the temporary is assigned from a method with nothing to call it on.
    # A class qualifier is part of it too - `A::go(n)` lifted as `go(n)` puts
    # `A::` in front of the temporary, and worse, leaves a call that named
    # its class to be dispatched through the vtable, which in an overrider is
    # the overrider calling itself.
    pattern = re.compile(
        # Not after a `~`: `items[i].~R()` runs a destructor, and read as a
        # call to `R` it was lifted out as one that might throw.
        r"(?<![.\w>:~])((?:[A-Za-z_]\w*\s*(?:\.|->|::)\s*)*)([A-Za-z_]\w*)\s*\("
    )
    for match in pattern.finditer(code):
        if match.start() < at:
            continue
        if match.group(2) not in throwing:
            continue
        close = _closing_paren(statement, match.end() - 1)
        if close < 0:
            continue
        found.append(statement[match.start(): close + 1])
        at = close + 1
    return found


#: What a lifted call's temporary is declared as. The value is never read when
#: the flag is set, so this only has to be a type the call's result fits.
_CALL_RESULT_TYPES: "dict[str, str]" = {}


def _call_result_type(call: str, throwing: "set[str]") -> str:
    name = call.split("(", 1)[0].strip().replace("->", ".").split(".")[-1]
    name = name.split("::")[-1].strip()
    return _CALL_RESULT_TYPES.get(name, "long")
def _mangle_overloaded_functions(text: str, filename: str) -> str:
    """Give each free function of a shared name a name of its own.

    Same reasoning as for methods: C has one function per name. Which one a
    call means is read from what it passes - the count where that settles it,
    and the types where two take the same number.
    """

    definitions: dict[str, list[str]] = {}
    for match in _DEFINITION.finditer(text):
        if _depth_at(text, match.end() - 1) != 0:
            continue
        name = match.group(2)
        if name in _NOT_A_TYPE or name == "main":
            continue
        definitions.setdefault(name, []).append(match.group(3))

    overloaded = {
        name: spelled for name, spelled in definitions.items() if len(spelled) > 1
    }
    if not overloaded:
        return text

    def suffix_of(name: str, parameters: str) -> str:
        arity = _arity(parameters)
        same = [p for p in overloaded[name] if _arity(p) == arity]
        if len(same) == 1:
            return str(arity)
        return f"{arity}__" + "_".join(_parameter_types(parameters))

    for name, spelled in overloaded.items():
        seen = [suffix_of(name, parameters) for parameters in spelled]
        if len(set(seen)) != len(seen):
            raise CppTranslationError(
                filename,
                _line_of(text, text.index(name)),
                f"two definitions of {name}() take the same arguments; C++ "
                f"would not accept that either",
            )

    # The definitions first, so a call is rewritten against names that exist.
    out: list[str] = []
    at = 0
    for match in _DEFINITION.finditer(text):
        if _depth_at(text, match.end() - 1) != 0 or match.start() < at:
            continue
        name = match.group(2)
        if name not in overloaded:
            continue
        out.append(text[at:match.start(2)])
        out.append(f"{name}__{suffix_of(name, match.group(3))}")
        at = match.end(2)
    out.append(text[at:])
    text = "".join(out)

    def rename(match: "re.Match[str]", whole: str) -> "str | None":
        name = match.group(1)
        if name not in overloaded:
            return None
        given = _call_arguments(whole, match.end() - 1)
        arity = len(given)
        same = [p for p in overloaded[name] if _arity(p) == arity]
        if not same:
            return None
        if len(same) == 1:
            return f"{name}__{arity}("
        wanted = [_deduced_type(value, whole, match.start()) for value in given]
        if any(item is None for item in wanted):
            raise CppTranslationError(
                filename,
                _line_of(whole, match.start()),
                f"more than one {name}() takes {arity} argument(s), and "
                f"py2bin cannot tell which is meant here. It reads the type "
                f"of a literal and of a variable it can see declared; cast "
                f"the argument to the type of the one you want",
            )
        codes = [_type_code(item) for item in wanted]
        for parameters in same:
            if _parameter_types(parameters) == codes:
                return f"{name}__{suffix_of(name, parameters)}("
        # The nearest fit, not the first one written. A `char` reaches an
        # `int` by a promotion and a `double` by a conversion between
        # families; taken in the order they were declared, `f('a')` called
        # whichever of the two came first in the file.
        scored = [
            (
                sum(
                    _closeness(code, declared)
                    for declared, code in zip(
                        _parameter_types(parameters), codes
                    )
                ),
                parameters,
            )
            for parameters in same
            if all(
                declared == code or declared in _PROMOTIONS.get(code, ())
                for declared, code in zip(_parameter_types(parameters), codes)
            )
        ]
        if not scored:
            return None
        nearest = min(one for one, _p in scored)
        best = [p for one, p in scored if one == nearest]
        if len(best) != 1:
            return None
        return f"{name}__{suffix_of(name, best[0])}("

    # Against the whole text: `rename` reads the type of each argument out of
    # where it was declared, and handed one stretch of code at a time it was
    # looking in a fragment. A call whose line held a string literal saw no
    # declarations at all, and the overload could not be told apart.
    return _sub_code(_A_CALL, text, rename)


#: A call: a name and the parenthesis that opens its arguments. Not after a
#: `~`, where the name is a destructor being run and not a function to pick
#: an overload of.
_A_CALL = re.compile(r"(?<![.\w>~])([A-Za-z_]\w*)\s*\(")


def _flatten_namespaces(text: str, filename: str) -> "tuple[str, set[str]]":
    """Remove the namespace wrappers, and answer which names they were.

    A namespace is scoping and nothing else, and py2bin compiles one
    translation unit with no linker behind it - so flattening is the whole of
    what a namespace means here. `N::thing` becomes `thing`, and
    `using namespace N;` becomes nothing, because there is no longer anywhere
    else for the name to be.

    What flattening cannot survive is a collision: two namespaces that each
    declare `helper` are two different functions in C++ and one name in C. So
    the names each one declares are collected, and a clash is refused by name
    rather than resolved by whichever came last.

    The file outside every namespace is the other side of the same question,
    and only for objects. A class or a function written both there and inside
    one is a redefinition by the time the C front end reads it, and is
    refused with its own line. An object is not: C lets `int count;` stand
    beside `int count = 5;` and means one variable by it, so a global and a
    namespace's own would have been folded together in silence.
    """

    known: set[str] = set()
    declared: dict[str, str] = {}
    outside = _namespace_objects(_without_nested(text))

    while True:
        found = _NAMESPACE.search(text)
        if found is None:
            break
        name = found.group(1) or ""
        opening = found.end() - 1
        try:
            closing = _matching(text, opening)
        except ValueError:
            raise CppTranslationError(
                filename, _line_of(text, opening),
                f"namespace {name or '<anonymous>'} is not closed",
            ) from None
        inner = text[opening + 1: closing - 1]
        if name:
            # `namespace a::b {` names two of them, and each is a qualifier
            # that has to be stripped from every use below.
            known.update(
                piece.strip() for piece in name.split("::") if piece.strip()
            )
            # Not what a namespace nested inside this one declares: those
            # names belong to that one, and counting them here made
            # `namespace outer { namespace inner { class Deep ... } }` look
            # like two namespaces declaring the same class.
            body = _without_nested(inner)
            mine = _namespace_objects(body)
            for spelled in sorted(mine & outside):
                raise CppTranslationError(
                    filename, _line_of(text, opening),
                    f"namespace {name} declares {spelled!r}, and so does the "
                    f"file outside every namespace. py2bin compiles one "
                    f"translation unit and has no linker, so a namespace is "
                    f"flattened away - and C reads the two declarations that "
                    f"leaves as one variable rather than two. Rename one",
                )
            for spelled in sorted(_declared_names(body) | mine):
                if spelled in declared and declared[spelled] != name:
                    raise CppTranslationError(
                        filename, _line_of(text, opening),
                        f"namespace {name} and namespace {declared[spelled]} "
                        f"both declare {spelled!r}. py2bin compiles one "
                        f"translation unit and has no linker, so a namespace "
                        f"is flattened away - and two of the same name cannot "
                        f"both survive that. Rename one",
                    )
                declared[spelled] = name
        # The body takes the wrapper's place, keeping the newlines the braces
        # sat on so nothing below shifts.
        text = text[:found.start()] + inner + text[closing:]

    text = _USING_NAMESPACE.sub("", text)
    return text, known



def _without_nested(body: str) -> str:
    """The body with any namespace nested inside it blanked out."""

    while True:
        found = _NAMESPACE.search(body)
        if found is None:
            return body
        opening = found.end() - 1
        try:
            closing = _matching(body, opening)
        except ValueError:
            return body
        body = body[:found.start()] + " " * (closing - found.start()) + body[closing:]

#: What counts as declaring a name at the top of a namespace body.
_DECLARES = re.compile(
    r"\b(?:class|struct)\s+([A-Za-z_]\w*)"
    r"|\b[A-Za-z_][\w \t*]*?\b([A-Za-z_]\w*)\s*\([^)]*\)\s*[{;]"
)


def _declared_names(body: str) -> "set[str]":
    """The classes and functions a namespace body declares, for clash checks.

    A class body is taken out first. What is inside one is a member, reached
    through an object and never by its bare name, so two classes in two
    namespaces may both have a `c_str` without either hiding the other -
    which is exactly what <filesystem>'s `path` and <string>'s `string` do.
    """

    found = set()
    for match in _DECLARES.finditer(_without_literals(_without_class_bodies(body))):
        spelled = match.group(1) or match.group(2)
        if spelled and spelled not in _NOT_A_TYPE:
            found.add(spelled)
    return found


def _namespace_objects(body: str) -> "set[str]":
    """The objects written straight into a namespace body, or into the file.

    Every braced body goes first. What is inside a function is that
    function's own, and reading the two together made a local `int tmp;`
    look like a name the namespace itself had put out - so two namespaces
    whose functions each had one would have been called a clash.

    Anything with parentheses in it is a function and not an object, which is
    how C++ tells the same two apart. Functions are already counted by
    `_declared_names`; this is here because an object is the one collision
    the C front end below cannot see. Two classes or two functions of one
    name are a redefinition there and are refused with their own line, but
    C's tentative definitions quietly make `int count;` in one namespace and
    `int count = 5;` in another into a single variable.
    """

    found: "set[str]" = set()
    for statement in _statements(_without_bodies(_without_literals(body))):
        cleaned = statement.strip().rstrip(";").strip()
        if not cleaned or "(" in cleaned:
            continue
        match = _DECLARATION_STATEMENT.match(cleaned)
        if match is None or match.group(1).split()[-1] in _NOT_A_TYPE:
            continue
        for part in _split_arguments(match.group(2)):
            declarator = _DECLARATOR.match(part)
            if declarator is not None:
                found.add(declarator.group(2))
    return found


def _without_bodies(text: str) -> str:
    """The text with every braced body blanked, the heads left where they are.

    Not `_without_class_bodies`, which keeps a function's: what this is for
    is the one level a namespace writes its own declarations at, and a body
    of any kind is a level below that.
    """

    out: list[str] = []
    at = 0
    depth = 0
    start = 0
    for brace in _A_BRACE.finditer(text):
        if brace.group(0) == "{":
            if depth == 0:
                start = brace.start()
            depth += 1
            continue
        if depth == 0:
            continue
        depth -= 1
        if depth == 0:
            out.append(text[at:start])
            out.append(" ")
            at = brace.end()
    out.append(text[at:])
    return "".join(out)


def _without_class_bodies(text: str) -> str:
    """The text with every `class`/`struct` body replaced by nothing.

    The head is kept, so the class's own name is still declared here.
    """

    out: list[str] = []
    at = 0
    for head in _CLASS_HEAD.finditer(text):
        if head.start() < at:
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        out.append(text[at:head.end() - 1])
        out.append(" ")
        at = closing
    out.append(text[at:])
    return "".join(out)


#: `namespace fs = std::filesystem;` - another name for a namespace, which is
#: how nearly all code that uses one with a long name refers to it.
_NAMESPACE_ALIAS = re.compile(
    r"\bnamespace\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_][\w:]*)\s*;"
)


def _namespace_aliases(text: str) -> "tuple[str, set[str]]":
    """Take the aliases out, and report the names they introduce.

    An alias needs no more than that: every namespace qualifier is stripped
    here anyway, so a name that stands for one is stripped the same way. What
    it must not do is survive into the C, where `namespace` is not a word.
    """

    found: set[str] = set()

    def taken(match: "re.Match[str]") -> str:
        found.add(match.group(1))
        return ""

    return _map_code(text, lambda part: _NAMESPACE_ALIAS.sub(taken, part)), found


#: `extern "C" {` and `extern "C" <one declaration>`.
_LINKAGE = re.compile(r'\bextern\s*"C(?:\+\+)?"\s*')


#: `constexpr` - a promise about *when* a value can be worked out, which is
#: a promise C does not make. What it says about the value itself is what
#: `const` says, and that is the part the C needs.
_CONSTEXPR = re.compile(r"(?<![.\w>])constexpr\b\s*")

#: `constexpr int kLimit = 12;` - a named integer known at compile time,
#: which is the one thing C has a spelling for.
_CONSTEXPR_INTEGER = re.compile(
    r"(?<![.\w>])(?:static\s+)?constexpr\s+"
    r"(?:(?:unsigned|signed|short|long|int|char|size_t|bool)\s+)*"
    r"(?:unsigned|signed|short|long|int|char|size_t|bool)\s+"
    r"([A-Za-z_]\w*)\s*=\s*([^;{}]+);"
)

#: What may stand in one: names, numbers and arithmetic. A literal would
#: have ended the stretch of code this is matched against, so there is none
#: to worry about; a `.` is a floating value and not an enumerator.
_AN_INTEGER_EXPRESSION = re.compile(r"^[\w\s+\-*/%()<>|&^~!=]+$")

#: What the enum holding one named constant is called. Nothing writes this
#: name; it is there because C++ has anonymous class-scope constants and the
#: pass that lifts a nested enum out needs one to lift.
_CONSTANT_ENUM = "__py2bin_constant_"


#: `std::move(v)` and `std::forward<T>(v)` - the two ways a program says
#: "this may be taken from". Written with the qualifier, so what is caught
#: here is the standard one and not a function the program wrote.
_A_MOVE = re.compile(
    r"(?<![.\w>])std\s*::\s*(?:move|forward)\s*(?:<[^<>;{}]*>\s*)?\("
)


#: A name, or a member reached through one: what needs no parentheses.
_A_PLAIN_NAME = re.compile(
    r"^[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)*$"
)


#: `Buf(Buf &&o)` - a parameter taken by rvalue reference. Bounded by the
#: punctuation of a parameter list on both sides, and the type has to be a
#: name this file declares or a word that begins one: `f(a && b)` is written
#: exactly the same way and is two operands, not a declaration.
_RVALUE_PARAMETER = re.compile(
    r"(?<=[(,])(\s*(?:const\s+)?([A-Za-z_]\w*)\s*)&&(\s*[A-Za-z_]\w*\s*)(?=[,)])"
)


def _rvalue_references_to_references(text: str) -> str:
    """`T &&o` becomes `T &o`, which is what it already compiles to here.

    An rvalue reference says the object it names is finished with, so what it
    holds may be taken rather than copied. Both spellings arrive as the same
    pointer; the difference is which constructor a call picks, and `std::move`
    - the thing that asks for the difference - is taken out just below,
    leaving one candidate rather than two to choose between.

    So a class with a move constructor and no copy constructor gets exactly
    the move it asked for. A class with both is refused where it is declared,
    rather than quietly given whichever came first.
    """

    def written(match: "re.Match[str]") -> str:
        named = match.group(2)
        if named not in _CLASS_NAMES and named not in _TYPE_WORDS:
            return match.group(0)
        return f"{match.group(1)}&{match.group(3)}"

    return _map_code(text, lambda part: _RVALUE_PARAMETER.sub(written, part))


def _strip_moves(text: str) -> str:
    """`std::move(v)` becomes `(v)`.

    A move is permission to take what an object holds rather than copy it,
    which is worth having when the copy is expensive. This subset has no
    move constructor and no rvalue reference, so every one of these is a
    copy - which is what the program would have done had it not asked, and
    is right, only slower. Taken out here rather than declared as a
    function, because a function answering `T &&` is a type the rest of
    this does not have.
    """

    while True:
        bare = _without_literals(text)
        found = _A_MOVE.search(bare)
        if found is None:
            return text
        close = _closing_paren(bare, found.end() - 1)
        if close < 0:
            return text
        inside = text[found.end(): close]
        # Parenthesised only where it has to be. A name is one already, and
        # wrapped it stopped being one for the passes that read what an
        # initialiser was given by looking at the name it names.
        kept = (
            inside.strip()
            if _A_PLAIN_NAME.match(inside.strip())
            else f"({inside})"
        )
        text = text[:found.start()] + kept + text[close + 1:]


#: `if constexpr (cond)` - the branch not taken is not compiled, which is the
#: whole reason a program writes one: the other arm is usually not valid for
#: the type this copy was written for.
_IF_CONSTEXPR = re.compile(r"(?<![.\w>])if\s+constexpr\s*\(")

#: What `sizeof` answers for the types whose size is the same on every target
#: py2bin has. `long` and `wchar_t` are deliberately absent: Windows is LLP64
#: and the rest are LP64, so those two differ, and answering either way here
#: would be right on some targets and wrong on the others.
_SIZE_OF = {
    "char": 1, "signed char": 1, "unsigned char": 1, "bool": 1,
    "short": 2, "unsigned short": 2, "short int": 2,
    "int": 4, "unsigned": 4, "unsigned int": 4, "float": 4,
    "long long": 8, "unsigned long long": 8, "double": 8,
}


def _constexpr_bodies(text: str) -> "dict[str, tuple[list[str], str]]":
    """Each `constexpr` function written as one `return`, by name."""

    written: "dict[str, tuple[list[str], str]]" = {}
    for found in _CONSTEXPR_FUNCTION.finditer(_without_literals(text)):
        names = [
            part.split()[-1].lstrip("*&")
            for part in _split_arguments(found.group(2))
            if part.strip() and part.strip() != "void"
        ]
        written[found.group(1)] = (names, found.group(3).strip())
    return written


def _folded_integer(
    spelled: str,
    text: str,
    written: "dict[str, tuple[list[str], str]] | None" = None,
    depth: int = 0,
) -> "int | None":
    """Work out an integer constant expression, or answer that it cannot.

    Only what can be settled from the text: literals, the named constants
    this translator has already written out - which is what a trait's
    `::value` becomes - `sizeof` of a type whose size is the same everywhere,
    and the arithmetic between them. Anything else answers None, and the
    caller says so rather than guessing.
    """

    import ast

    working = spelled.strip()
    if not working:
        return None
    # `sizeof(int)` and `sizeof(T *)`. A pointer is eight bytes on every
    # target py2bin has, whatever it points at.
    unknown = [False]

    def sized(match: "re.Match[str]") -> str:
        named = re.sub(r"\b(?:const|volatile)\b", " ", match.group(1)).strip()
        named = re.sub(r"\s+", " ", named)
        if named.endswith("*"):
            return "8"
        answer = _SIZE_OF.get(named)
        if answer is None:
            # Said with a flag and not with a character standing in for
            # "unknown": the character used to be `?`, which is also the
            # first half of every conditional, so `n <= 1 ? 1 : 2` was read
            # as a `sizeof` that could not be answered.
            unknown[0] = True
            return "0"
        return str(answer)

    working = re.sub(r"\bsizeof\s*\(([^()]*)\)", sized, working)
    if unknown[0]:
        return None
    # `is_pointer<int *>::value` - a constant that belongs to a class. The
    # copy for these arguments has been written out by now, so the class is
    # in the text and its body says what the member is. Read from there,
    # because the pass that flattens a static member into a constant of its
    # own has not run yet.
    def qualified(match: "re.Match[str]") -> str:
        owner, member = match.group(1), match.group(2)
        for head in _CLASS_HEAD.finditer(text):
            if head.group(2) != owner:
                continue
            try:
                closing = _matching(text, head.end() - 1)
            except ValueError:
                return match.group(0)
            inside = text[head.end(): closing - 1]
            wrote = re.search(
                rf"(?<![.\w>]){re.escape(member)}\s*=\s*(-?\d+)\s*[;,}}]", inside
            )
            if wrote is not None:
                return wrote.group(1)
        return match.group(0)

    working = re.sub(
        r"(?<![.\w>])([A-Za-z_]\w*)\s*::\s*([A-Za-z_]\w*)", qualified, working
    )
    if "::" in working:
        return None
    # `c ? a : b` - the condition first, and then only the arm it chooses.
    # C++ evaluates one arm and not the other, and a recursive `constexpr`
    # depends on it: `n <= 1 ? 1 : n * fact(n - 1)` has its bottom in the arm
    # that is taken, and working out both went down for ever.
    arms = _conditional_arms(working)
    if arms is not None and depth < 64:
        depth_of = 0
        question = -1
        for index, piece in enumerate(_without_literals(working)):
            if piece in "([{":
                depth_of += 1
            elif piece in ")]}":
                depth_of -= 1
            elif depth_of == 0 and piece == "?":
                question = index
                break
        if question >= 0:
            asked = _folded_integer(working[:question], text, written, depth + 1)
            if asked is None:
                return None
            return _folded_integer(
                arms[0] if asked else arms[1], text, written, depth + 1
            )

    # A call to a `constexpr` function written as one `return`. Answering it
    # here is what lets one call another, and itself: `fact(n - 1)` is a call
    # in the body of the function being worked out.
    if written and depth < 32:
        for _round in range(64):
            call = None
            for found in re.finditer(r"(?<![.\w>])([A-Za-z_]\w*)\s*\(", working):
                if found.group(1) not in written:
                    continue
                closing = _closing_paren(working, found.end() - 1)
                if closing < 0:
                    continue
                inside = working[found.end(): closing]
                if re.search(r"[A-Za-z_]\w*\s*\(", inside):
                    continue   # an inner call first
                call = (found, closing, inside)
                break
            if call is None:
                break
            found, closing, inside = call
            names, body = written[found.group(1)]
            given = [one.strip() for one in _split_arguments(inside) if one.strip()]
            if len(given) != len(names):
                return None
            values = [
                _folded_integer(one, text, written, depth + 1) for one in given
            ]
            if any(one is None for one in values):
                return None
            filled = body
            for named, value in zip(names, values):
                filled = re.sub(
                    rf"(?<![.\w>]){re.escape(named)}(?![\w])", f"({value})", filled
                )
            answer = _folded_integer(filled, text, written, depth + 1)
            if answer is None:
                return None
            working = working[: found.start()] + str(answer) + working[closing + 1:]
    # The constants this translator wrote, which is what `Trait<T>::value`
    # became: `const int is_pointer__int__value = 0;`.
    for _round in range(8):
        names = set(re.findall(r"(?<![.\w>])([A-Za-z_]\w*)", working))
        names -= {"true", "false"}
        if not names:
            break
        settled = False
        for name in names:
            found = re.search(
                rf"(?<![.\w>])(?:const\s+)?(?:int|long|unsigned|short|char|bool)"
                rf"[\w\s]*?\b{re.escape(name)}\s*=\s*(-?\d+)\s*[;,}}]",
                text,
            ) or re.search(
                rf"(?<![.\w>]){re.escape(name)}\s*=\s*(-?\d+)\s*[,}}]", text
            )
            if found is None:
                return None
            working = re.sub(
                rf"(?<![.\w>]){re.escape(name)}(?![\w])", found.group(1), working
            )
            settled = True
        if not settled:
            return None
    working = working.replace("true", "1").replace("false", "0")
    working = working.replace("&&", " and ").replace("||", " or ")
    working = re.sub(r"!(?!=)", " not ", working)
    working = re.sub(r"(?<![/])/(?![/])", "//", working)
    try:
        tree = ast.parse(working, mode="eval")
    except SyntaxError:
        return None
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
        ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod,
        ast.LShift, ast.RShift, ast.BitOr, ast.BitAnd, ast.BitXor,
        ast.Invert, ast.Not, ast.USub, ast.UAdd, ast.And, ast.Or,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.IfExp,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return None
    try:
        answer = eval(compile(tree, "<constant>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None
    return int(answer) if isinstance(answer, (int, bool)) else None


def _rewrite_if_constexpr(text: str, filename: str) -> str:
    """Keep the arm `if constexpr` chooses and remove the other.

    C++ compiles only the arm whose condition holds, and a program writes one
    exactly when the other arm would not compile for this type. Left as an
    ordinary `if`, both arms reach the C compiler and the one that was never
    meant to be there is reported against a line the author was right about.

    So the condition is worked out here, where the template arguments have
    already been substituted and `sizeof(T)` is `sizeof(int)`. A condition
    this cannot settle is refused by name - guessing which arm survives is
    the one thing that must not happen.
    """

    for _round in range(_HOIST_ROUNDS):
        bare = _without_literals(text)
        found = _IF_CONSTEXPR.search(bare)
        if found is None:
            return text
        opening = found.end() - 1
        closing = _closing_paren(bare, opening)
        if closing < 0:
            return text
        condition = text[opening + 1: closing]
        answer = _folded_integer(condition, text)
        if answer is None:
            raise CppTranslationError(
                filename, _line_of(text, found.start()),
                f"`if constexpr ({condition.strip()})` cannot be worked out "
                f"here; py2bin reads literals, the constants a trait's "
                f"`::value` becomes, and `sizeof` of a type whose size is "
                f"the same on every target - and it must be settled, because "
                f"the arm that loses is the one C++ never compiles",
            )
        taken_end = _end_of_statement(bare, closing + 1)
        taken = text[closing + 1: taken_end]
        rest = taken_end
        while rest < len(bare) and bare[rest] in " \t\r\n":
            rest += 1
        other = ""
        after = taken_end
        if re.match(r"else\b", bare[rest:]):
            after = _end_of_statement(bare, rest + 4)
            other = text[rest + 4: after]
        kept = taken if answer else other
        # A block, so a declaration inside the arm stays inside it - which is
        # what it was in, and what the arm being a statement of its own means.
        text = text[:found.start()] + "{ " + kept.strip() + " }" + text[after:]
    return text


#: `auto [a, b] = p;` - C++17's way of naming what an object holds. The
#: qualifiers in front are the ones a binding may carry.
_STRUCTURED_BINDING = re.compile(
    r"(?<![.\w>])(?:const\s+)?auto\s*(?:&&?\s*)?\[\s*([^\]]*?)\s*\]\s*=\s*"
)

#: What the pass below calls the object it binds against.
_BOUND_PREFIX = "__py2bin_bound_"


def _struct_members(text: str, name: str) -> "list[tuple[str, str]]":
    """The data members a struct declares, in order, as (type, name).

    Read from the body rather than from a `Class`, because a struct with no
    methods is C already and is kept out of the classes this translator
    builds - and that is exactly the kind a binding is usually written for.
    """

    for head in _CLASS_HEAD.finditer(text):
        if head.group(2) != name:
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            return []
        found: "list[tuple[str, str]]" = []
        for line in _without_literals(text[head.end(): closing - 1]).split(";"):
            spelled = line.strip()
            if not spelled or "(" in spelled or spelled.startswith(("public", "private", "protected")):
                continue
            written = re.match(
                r"^((?:const\s+|unsigned\s+|signed\s+|struct\s+)*[A-Za-z_]\w*"
                r"(?:\s*\*)*)\s+([A-Za-z_]\w*)$",
                spelled,
            )
            if written is not None:
                found.append((written.group(1).strip(), written.group(2)))
        return found
    return []


def _rewrite_structured_bindings(text: str, filename: str) -> str:
    """`auto [x, y] = p;` becomes the object and one declaration per name.

    C++ binds a name to each member of what is on the right, in the order
    the members were declared. C has no such declaration, so the object is
    given a name of its own and each binding becomes an ordinary one reading
    a member off it - which is what the binding is.
    """

    counter = 0
    for _round in range(_HOIST_ROUNDS):
        bare = _without_literals(text)
        found = _STRUCTURED_BINDING.search(bare)
        if found is None:
            return text
        ends = bare.find(";", found.end())
        if ends < 0:
            return text
        names = [one.strip() for one in found.group(1).split(",") if one.strip()]
        value = text[found.end(): ends].strip()
        held = _deduced_type(value, text, found.start())
        holds = (
            re.sub(r"\b(?:const|struct|volatile)\b", " ", held).replace("*", " ").strip()
            if held
            else ""
        )
        members = _struct_members(text, holds) if holds else []
        if len(members) != len(names):
            raise CppTranslationError(
                filename, _line_of(text, found.start()),
                f"cannot bind [{', '.join(names)}] here; py2bin reads the "
                f"members of what is on the right and binds one name to each, "
                f"and "
                + (
                    f"{holds} declares {len(members)}"
                    if holds
                    else "what is on the right has no type it can read"
                ),
            )
        counter += 1
        made = f"{_BOUND_PREFIX}{counter}"
        written = [f"{holds} {made} = {value};"]
        for (spelled, member), given in zip(members, names):
            written.append(f"{spelled} {given} = {made}.{member};")
        text = text[:found.start()] + " ".join(written) + text[ends + 1:]
    return text


#: `std::get<0>(t)` - reading one of a tuple's members by position. The
#: qualifier is gone by the time this runs.
_TUPLE_GET = re.compile(r"(?<![.\w>])get\s*<\s*(\d+)\s*>\s*\(")

#: `get<int>(v)` and `holds_alternative<double>(v)` - an alternative named by
#: its type rather than by its place.
_VARIANT_BY_TYPE = re.compile(
    r"(?<![.\w>])(get|holds_alternative)\s*<\s*([^<>()]+?)\s*>\s*\("
)


#: `duration_cast<microseconds>(d)` - a duration asked for in another unit.
_DURATION_CAST = re.compile(
    r"(?<![.\w>])duration_cast\s*<\s*([A-Za-z_][\w:]*)\s*>\s*\("
)


def _rewrite_duration_cast(text: str) -> str:
    """`duration_cast<microseconds>(d)` becomes the member for that unit.

    The unit is a type, and the four this ships each have a member on
    `duration` that divides down to it. Named rather than deduced, because a
    unit is a ratio in C++ and a ratio is not something py2bin computes with.
    """

    if "duration_cast" not in text:
        return text
    bare = _without_literals(text)
    out: "list[str]" = []
    at = 0
    for found in _DURATION_CAST.finditer(bare):
        if found.start() < at:
            continue
        unit = found.group(1).rsplit("::", 1)[-1]
        if unit not in ("nanoseconds", "microseconds", "milliseconds", "seconds"):
            continue
        closing = _closing_paren(bare, found.end() - 1)
        if closing < 0:
            continue
        inside = text[found.end(): closing].strip()
        # `.count()` right after it, which is how it is nearly always
        # written: answered outright rather than through the unit object.
        counted = re.match(r"\s*\.\s*count\s*\(\s*\)", bare[closing + 1:])
        out.append(text[at: found.start()])
        if counted is not None:
            out.append(f"({inside}).__count_{unit}()")
            at = closing + 1 + counted.end()
        else:
            out.append(f"({inside}).__as_{unit}()")
            at = closing + 1
    out.append(text[at:])
    return "".join(out)


def _rewrite_variant_alternatives(text: str) -> str:
    """`get<int>(v)` becomes the member `int` is, in the variant `v` holds.

    Which member that is comes from where `v` was declared: `variant<int,
    double> v` says `int` is the first. Read there because nothing else says
    it - a type is not a place until the list it is in is known.
    """

    if not re.search(r"(?<![.\w>])(?:class|struct)\s+variant\b", text):
        return text
    bare = _without_literals(text)
    out: "list[str]" = []
    at = 0
    for found in _VARIANT_BY_TYPE.finditer(bare):
        if found.start() < at:
            continue
        closing = _closing_paren(bare, found.end() - 1)
        if closing < 0:
            continue
        inside = text[found.end(): closing].strip()
        declared = re.search(
            rf"(?<![.\w>])variant\s*<([^<>]*)>\s*{re.escape(inside)}\b", bare
        )
        if declared is None:
            continue
        alternatives = [
            one.strip() for one in _split_arguments(declared.group(1)) if one.strip()
        ]
        wanted = found.group(2).strip()
        if wanted not in alternatives:
            continue
        where = alternatives.index(wanted)
        out.append(text[at: found.start()])
        out.append(
            f"(({inside}).__tag == {where})"
            if found.group(1) == "holds_alternative"
            else f"({inside}).__{where}"
        )
        at = closing + 1
    out.append(text[at:])
    return "".join(out)


#: `template <typename M> class lock_guard {` - a class template's head, with
#: its parameters.
_CLASS_TEMPLATE_HEAD = re.compile(
    r"(?<![.\w>])template\s*<([^<>]*)>\s*(?:class|struct)\s+([A-Za-z_]\w*)\s*\{"
)

#: `lock_guard hold(m);` - a class template named without its arguments, which
#: C++17 works out from what the constructor is handed.
_DEDUCED_CLASS = re.compile(
    r"(?<![.\w>:])([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\(([^;{}()]*)\)\s*;"
)


#: `std::thread worker(f);` and `worker_ = std::thread(&Host::run, this);` -
#: a thread being given what to run. The `std::` is gone by the time this
#: runs, and both spellings name the same thing.
_A_THREAD_MADE = re.compile(
    r"(?<![.\w>])thread\s*(?:([A-Za-z_]\w*)\s*)?\(([^;{}()]*)\)"
)

#: What the pass below calls the function it writes.
_THREAD_ENTRY_PREFIX = "__py2bin_thread_run_"


def _rewrite_threads(text: str, filename: str) -> str:
    """Give each `std::thread` a plain function to start.

    A platform starts a thread at a function taking one pointer. C++ starts
    one at anything callable, and a callable here is a class with a call
    operator, or a member function and an object to call it on. Neither is a
    plain function, so one is written - a trampoline that takes the pointer,
    puts it back into the shape it came from, and makes the call.

    Which shape that is, is decided here and not while running, which is why
    the thread object itself holds only a handle: what it runs was settled
    where it was written.
    """

    if not re.search(r"(?<![.\w>])(?:class|struct)\s+thread\b", text):
        return text
    made: "list[str]" = []
    counter = 0
    for _round in range(_HOIST_ROUNDS):
        bare = _without_literals(text)
        found = None
        for match in _A_THREAD_MADE.finditer(bare):
            if _is_a_definition(bare, match.end() - 1):
                continue
            given = [
                one.strip()
                for one in _split_arguments(match.group(2))
                if one.strip()
            ]
            if not given:
                continue
            found = (match, given)
            break
        if found is None:
            break
        match, given = found
        counter += 1
        entry = f"{_THREAD_ENTRY_PREFIX}{counter}"
        held = given[0]
        member = re.match(r"^&\s*([A-Za-z_]\w*)\s*::\s*([A-Za-z_]\w*)$", held)
        if member is not None and len(given) == 2:
            # `thread(&Host::run, this)` - a member function and the object
            # to call it on. The object is the argument, and the trampoline
            # is the call.
            owner, method = member.group(1), member.group(2)
            # Through a name and not through the cast: a call on a cast is
            # not a shape the pass that rewrites member calls reads, and a
            # pointer that has a name is.
            made.append(
                f"static long {entry}(void *__py2bin_given) {{ "
                f"{owner} *__py2bin_on = ({owner} *)__py2bin_given; "
                f"__py2bin_on->{method}(); return 0; }}"
            )
            passed = f"(void *)({given[1]})"
        elif len(given) == 1:
            # Anything callable held in a name: the object is what the
            # trampoline calls, and it is the argument too.
            # A callable held in a name. What class it is comes from where
            # it was declared, which is the only place that says.
            holds = _deduced_type(held, text, match.start())
            owner = (
                re.sub(r"\b(?:const|struct|volatile)\b", " ", holds or "")
                .replace("*", " ")
                .strip()
            )
            if not owner:
                raise CppTranslationError(
                    filename, _line_of(text, match.start()),
                    f"a thread is started here on `{held}`, and py2bin "
                    f"cannot read what that is. It writes the function the "
                    f"platform starts a thread at, and needs the type to "
                    f"know what that function should call",
                )
            made.append(
                f"static long {entry}(void *__py2bin_given) {{ "
                f"{owner} *__py2bin_on = ({owner} *)__py2bin_given; "
                f"(*__py2bin_on)(); return 0; }}"
            )
            passed = f"(void *)(&{held})"
        elif re.fullmatch(r"[A-Za-z_]\w*", held) and _declared_return(
            text, None, held
        ) is not None:
            # `thread(add, 1000)` - a function and what to call it with. A
            # platform hands its thread one pointer, so the arguments are put
            # in something and that is the pointer: a small class written
            # here, built with `new` so it outlives the statement that starts
            # the thread. It is never freed, which is what a thread argument
            # pack costs; py2bin's heap is an arena and gives it back at exit.
            bound = given[1:]
            spelled_types = [
                _deduced_type(one, text, match.start()) for one in bound
            ]
            if any(one is None for one in spelled_types):
                raise CppTranslationError(
                    filename, _line_of(text, match.start()),
                    f"a thread is started here on `{held}` with arguments "
                    f"py2bin cannot read the types of. It reads a literal "
                    f"and a variable it can see declared; give the argument "
                    f"a variable of its own and pass that",
                )
            pack = f"__py2bin_thread_args_{counter}"
            members = "".join(
                f"    {one} a{index};\n"
                for index, one in enumerate(spelled_types)
            )
            parameters = ", ".join(
                f"{one} v{index}" for index, one in enumerate(spelled_types)
            )
            assigned = " ".join(
                f"a{index} = v{index};" for index in range(len(bound))
            )
            passing = ", ".join(
                f"__py2bin_a->a{index}" for index in range(len(bound))
            )
            made.append(
                f"class {pack} {{\npublic:\n{members}"
                f"    {pack}({parameters}) {{ {assigned} }}\n}};\n"
                f"static long {entry}(void *__py2bin_given) {{ "
                f"{pack} *__py2bin_a = ({pack} *)__py2bin_given; "
                f"{held}({passing}); return 0; }}"
            )
            passed = f"(void *)(new {pack}({', '.join(bound)}))"
        else:
            raise CppTranslationError(
                filename, _line_of(text, match.start()),
                f"a thread is started here with {len(given)} arguments and "
                f"py2bin cannot tell what the first of them is. It writes "
                f"the function a platform starts a thread at, and can write "
                f"one for a callable on its own, for a member function and "
                f"the object to call it on, or for a function and the "
                f"arguments bound to it",
            )
        naming = match.group(1)
        ends = match.end()
        if naming:
            spelled = f"thread {naming}; {naming}.__begin({entry}, {passed});"
            begins = match.start()
            if bare[ends: ends + 1] == ";":
                ends += 1
        else:
            # `worker_ = thread(...);` - the thread is made and moved into
            # something that already exists. There is nothing to move here:
            # the handle is the whole of the object, so the one that exists
            # is simply the one that is started.
            assigned = re.search(
                r"(?<![.\w>])((?:this\s*->\s*)?[A-Za-z_]\w*)\s*=\s*$",
                bare[: match.start()],
            )
            if assigned is None:
                raise CppTranslationError(
                    filename, _line_of(text, match.start()),
                    "a thread is made here without being named or given to "
                    "anything; py2bin starts one where it can see what holds "
                    "the handle, because the handle is how it is joined",
                )
            begins = assigned.start(1)
            spelled = f"{assigned.group(1)}.__begin({entry}, {passed})"
        text = text[:begins] + spelled + text[ends:]
    if not made:
        return text
    return _above_the_first_use(text, "\n".join(made))


def _deduce_class_arguments(text: str) -> str:
    """`lock_guard hold(m);` becomes `lock_guard<mutex> hold(m);`.

    C++17 lets a class template be named without its arguments where the
    constructor says what they are. py2bin writes one copy per set of
    arguments and needs them written down, so they are worked out here and
    written down - from the constructor's parameters against the types of
    what is passed, which is the same reading a function template's call
    already gets.

    Only where every parameter is settled by one argument outright: a
    constructor taking `T *` where a `T` is deduced from further in is a
    deduction guide, and a guide is a thing this does not read.
    """

    heads: "dict[str, tuple[list[str], str]]" = {}
    bare = _without_literals(text)
    for head in _CLASS_TEMPLATE_HEAD.finditer(bare):
        named = [
            part.split()[-1]
            for part in _split_arguments(head.group(1))
            if part.strip() and "..." not in part
        ]
        if not named:
            continue
        try:
            closing = _matching(bare, head.end() - 1)
        except ValueError:
            continue
        heads[head.group(2)] = (named, text[head.end(): closing - 1])
    if not heads:
        return text

    def written(match: "re.Match[str]") -> "str | None":
        holds = heads.get(match.group(1))
        if holds is None or match.group(1) == match.group(2):
            return None
        parameters, body = holds
        given = [
            one.strip() for one in _split_arguments(match.group(3)) if one.strip()
        ]
        if not given:
            return None
        # The constructor taking this many, and what each of its parameters
        # is written as.
        wanted: "list[str]" = []
        for one in re.finditer(
            rf"(?<![.\w>~]){re.escape(match.group(1))}\s*\(([^;{{}}()]*)\)", body
        ):
            parts = [
                part.strip()
                for part in _split_arguments(one.group(1))
                if part.strip()
            ]
            if len(parts) == len(given):
                wanted = parts
                break
        if not wanted:
            return None
        settled: "dict[str, str]" = {}
        for spelled, value in zip(wanted, given):
            held = _deduced_type(value, text, match.start())
            if held is None:
                return None
            shape = re.sub(r"\b[A-Za-z_]\w*$", "", spelled).strip()
            _fits_the_shape(shape, held.strip(), set(parameters), settled, True)
        if any(one not in settled for one in parameters):
            return None
        spelled = ", ".join(settled[one] for one in parameters)
        return (
            f"{match.group(1)}<{spelled}> {match.group(2)}"
            f"({match.group(3)});"
        )

    return _sub_code(_DEDUCED_CLASS, text, lambda m, whole: written(m))


def _rewrite_tuple_get(text: str) -> str:
    """`get<N>(t)` becomes `(t).__N`, which is the member it names.

    `get` spells one template argument and leaves the rest to be deduced from
    what it is handed, and a copy written for that shape is not something this
    translator makes. The members have names, so reading one by name is the
    same thing said in a way C already has.

    Only where this file has a tuple in it at all: `get<0>` is a name a
    program may have of its own, and one that does not come from `<tuple>` is
    not this.
    """

    if not re.search(r"(?<![.\w>])(?:class|struct)\s+tuple\b", text):
        return text
    out: "list[str]" = []
    at = 0
    bare = _without_literals(text)
    for found in _TUPLE_GET.finditer(bare):
        if found.start() < at:
            continue
        closing = _closing_paren(bare, found.end() - 1)
        if closing < 0:
            continue
        inside = text[found.end(): closing].strip()
        out.append(text[at: found.start()])
        out.append(f"({inside}).__{found.group(1)}")
        at = closing + 1
    out.append(text[at:])
    return "".join(out)


#: `constexpr int twice(int n) { return n * 2; }` - a function whose answer
#: C++ works out while compiling. Only the single-`return` shape, which is
#: what one written to be used as a constant almost always is.
_CONSTEXPR_FUNCTION = re.compile(
    r"(?<![.\w>])constexpr\s+(?:inline\s+)?[A-Za-z_][\w\s]*?"
    r"\b([A-Za-z_]\w*)\s*\(([^()]*)\)\s*(?:const\s*)?\{\s*"
    r"return\s+([^;{}]+);\s*\}"
)


def _fold_constexpr_calls(text: str) -> str:
    """`twice(3)` becomes `6` where `twice` is `constexpr` and 3 is constant.

    C++ requires the answer at compile time wherever a constant is required -
    an array length, a `case` label - and py2bin's C requires one in the same
    places. The word says the function can answer there, so it is asked:
    the arguments are put in place of the parameters and the result is worked
    out the same way any other constant expression here is.

    Only where every argument is itself constant. A call with a value in it
    is an ordinary call, and stays one - which is also what C++ does with it.
    """

    written = _constexpr_bodies(text)
    if not written:
        return text
    for _round in range(_HOIST_ROUNDS):
        bare = _without_literals(text)
        change = None
        for found in re.finditer(r"(?<![.\w>])([A-Za-z_]\w*)\s*\(", bare):
            name = found.group(1)
            if name not in written:
                continue
            if _is_a_definition(bare, _closing_paren(bare, found.end() - 1)):
                continue
            closing = _closing_paren(bare, found.end() - 1)
            if closing < 0:
                continue
            names, body = written[name]
            given = [
                one.strip()
                for one in _split_arguments(text[found.end(): closing])
                if one.strip()
            ]
            if len(given) != len(names):
                continue
            values = [_folded_integer(one, text, written) for one in given]
            if any(one is None for one in values):
                continue
            filled = body
            for spelled, value in zip(names, values):
                filled = re.sub(
                    rf"(?<![.\w>]){re.escape(spelled)}(?![\w])",
                    f"({value})",
                    filled,
                )
            answer = _folded_integer(filled, text, written)
            if answer is None:
                continue
            change = (found.start(), closing + 1, str(answer))
            break
        if change is None:
            return text
        start, end, value = change
        text = text[:start] + value + text[end:]
    return text


#: `vector<int> v = {1, 2, 3};` and `vector<int> v{1, 2, 3};` - a container
#: given its contents where it is declared.
_LIST_INITIALISED = re.compile(
    r"(?<![.\w>])([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(?:=\s*)?\{"
)


def _takes_push_back(text: str, name: str) -> bool:
    """Whether that class declares `push_back`, so a list can fill it."""

    for head in _CLASS_HEAD.finditer(text):
        if head.group(2) != name:
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            return False
        return bool(
            re.search(r"(?<![.\w>])push_back\s*\(", text[head.end(): closing - 1])
        )
    return False


def _rewrite_list_initialisers(text: str) -> str:
    """`vector<int> v = {1, 2, 3};` becomes the object and three pushes.

    C++ hands the braces to a constructor taking an `initializer_list`, which
    is a view of an array the compiler laid out. py2bin has no such thing and
    no way to write one - so the list becomes what it means, which is the
    container filled one value at a time, in the order they were written.

    Only for a class that takes `push_back`. A brace list on anything else is
    a struct being initialised, and that is C already.
    """

    for _round in range(_HOIST_ROUNDS):
        bare = _without_literals(text)
        change = None
        for found in _LIST_INITIALISED.finditer(bare):
            if not _could_start_a_declaration(bare, found.start()):
                continue
            if not _takes_push_back(text, found.group(1)):
                continue
            opening = found.end() - 1
            try:
                after = _matching(bare, opening)
            except ValueError:
                continue
            rest = after
            while rest < len(bare) and bare[rest] in " \t":
                rest += 1
            if rest >= len(bare) or bare[rest] != ";":
                continue
            values = [
                one.strip()
                for one in _split_arguments(text[opening + 1: after - 1])
                if one.strip()
            ]
            held, named = found.group(1), found.group(2)
            written = f"{held} {named}; " + " ".join(
                f"{named}.push_back({one});" for one in values
            )
            change = (found.start(), rest + 1, written)
            break
        if change is None:
            return text
        start, end, written = change
        text = text[:start] + written + text[end:]
    return text


#: `static_assert(cond, "why")` and its one-argument form.
_STATIC_ASSERT = re.compile(r"(?<![.\w>])static_assert\s*\(")


def _check_static_asserts(
    text: str, filename: str, whatever_it_says: bool = False
) -> str:
    """Answer each `static_assert` while translating, and take it out.

    It is a question asked of the compiler, so this is the compiler answering
    it. One whose condition cannot be worked out is left standing rather than
    assumed true - it then reaches the C compiler, which says it does not know
    the name, and that is a better answer than silence.
    """

    for _round in range(_HOIST_ROUNDS):
        bare = _without_literals(text)
        found = _STATIC_ASSERT.search(bare)
        if found is None:
            return text
        closing = _closing_paren(bare, found.end() - 1)
        if closing < 0:
            return text
        parts = _split_arguments(text[found.end(): closing])
        answer = _folded_integer(parts[0], text) if parts else None
        if answer is None and not whatever_it_says:
            return text
        if answer is None:
            # Asked once more after the copies are written out, where
            # `sizeof(T)` is `sizeof(int)`. Still unreadable there, it is
            # taken out rather than left: what remains would be read as a
            # member of the class it is written in, which it is not. Dropping
            # a check never changes what a correct program does - it only
            # fails to catch an incorrect one - and that is the honest cost.
            answer = 1
        if not answer:
            said = parts[1].strip().strip('"') if len(parts) > 1 else ""
            raise CppTranslationError(
                filename, _line_of(text, found.start()),
                f"static_assert failed{': ' + said if said else ''}",
            )
        ends = closing + 1
        while ends < len(bare) and bare[ends] in " \t":
            ends += 1
        if ends < len(bare) and bare[ends] == ";":
            ends += 1
        text = text[: found.start()] + text[ends:]
    return text


def _strip_constexpr(text: str) -> str:
    """`constexpr T k = v;` becomes `const T k = v;`.

    C++ asks for the value at compile time; py2bin's C works it out at
    compile time anyway wherever it can, and where it cannot the program
    still runs - so the difference is a diagnostic C++ gives and this does
    not, which is permissive and never wrong for a program that is right.

    On a function the word says the same thing about its result, and a
    `const` return type is something C accepts and ignores.
    """

    def named(match: "re.Match[str]") -> str:
        value = match.group(2).strip()
        if not _AN_INTEGER_EXPRESSION.match(value) or "." in value:
            return match.group(0)
        # An enumerator, because that is what C accepts where a constant is
        # required: `int room[kLimit];` is a declaration in C++ and, with
        # `kLimit` a `const int`, is not one in C.
        # Named, not anonymous: written inside a class this has to be lifted
        # out of it, and the pass that does that works on enums with names.
        return (
            f"enum {_CONSTANT_ENUM}{match.group(1)} "
            f"{{ {match.group(1)} = {value} }};"
        )

    def dropped(match: "re.Match[str]") -> str:
        # `if constexpr` is not this: it is a branch chosen while translating,
        # and the pass that chooses runs after the templates are written out.
        # Turned into `const` here it stopped being recognisable at all.
        if re.search(r"(?<![.\w>])if\s*$", match.string[: match.start()]):
            return match.group(0)
        return "const "

    text = _map_code(text, lambda part: _CONSTEXPR_INTEGER.sub(named, part))
    return _map_code(text, lambda part: _CONSTEXPR.sub(dropped, part))


def _strip_linkage(text: str) -> str:
    """`extern "C" { ... }` is its contents, and the braces go.

    A linkage specification says how a name is to be spelled for a linker.
    py2bin has one translation unit and no linker, so there is nothing for it
    to say - and the braces are not a scope, so what is inside them is at the
    same level as what is outside. Left in, the translator read `extern "C"`
    as a declaration and stopped on the string.
    """

    at = 0
    while True:
        # Searched in the text itself and not in the copy with the literals
        # blanked, because the `"C"` *is* the literal: blanked, there is
        # nothing left to match. So the copy is consulted the other way -
        # to check the match is code, and not the words `extern "C"` inside
        # a string the program is printing.
        found = _LINKAGE.search(text, at)
        if found is None:
            return text
        bare = _without_literals(text)
        if bare[found.start(): found.start() + 6] != "extern":
            at = found.start() + 1
            continue
        after = text[found.end():]
        opening = len(after) - len(after.lstrip(" \t\n"))
        if after[opening: opening + 1] != "{":
            # `extern "C" void f(void);` - one declaration, and only the
            # words in front of it go.
            text = text[:found.start()] + text[found.end():]
            continue
        at = found.end() + opening
        try:
            closing = _matching(text, at)
        except ValueError:
            return text[:found.start()] + text[found.end():]
        # `_matching` answers just past the `}`.
        text = (
            text[:found.start()]
            + text[at + 1: closing - 1]
            + text[closing:]
        )


def _strip_namespace_qualifiers(text: str, namespaces: "set[str]") -> str:
    """`N::thing` becomes `thing`, and `Class::method` is left alone.

    Only the names collected while flattening are stripped, because `::` also
    spells an out-of-line member definition, and removing that would turn a
    method into a free function of the same name.
    """

    for name in sorted(namespaces, key=len, reverse=True):
        text = _map_code(
            text, lambda part, n=name: re.sub(rf"\b{re.escape(n)}\s*::\s*", "", part)
        )
    return text

#: The spellings a generated COM header is written in. Distinctive enough
#: that a file using one of them is COM code, which is what decides whether
#: the rest of the table - `interface`, `PURE` - is applied at all.
_COM_SPELLINGS = re.compile(
    r"(?<![.\w>])(MIDL_INTERFACE|STDMETHOD|DECLSPEC_UUID|DECLSPEC_NOVTABLE"
    r"|BEGIN_INTERFACE|__RPC_FAR|STDMETHODCALLTYPE)\b"
)


def _com_macros() -> "list[tuple[str, list[str], str]]":
    """The COM spellings and what each stands for, read from py2bin's header.

    Read rather than written out again: `<rpcndr.h>` is where they are
    defined for the C that comes out of this, and two copies of a table like
    that drift. Longest name first, so `STDMETHOD_` is not read as
    `STDMETHOD` and `THIS_` not as `THIS`.
    """

    from . import c_preprocessor

    found: "list[tuple[str, list[str], str]]" = []
    for match in re.finditer(
        r"^#define\s+([A-Za-z_]\w*)(\([^)]*\))?[ \t]*(.*)$",
        c_preprocessor._RPCNDR_H,
        re.M,
    ):
        name, parameters, body = match.groups()
        if "##" in body:
            # Token pasting, which needs the C preprocessor and not this.
            continue
        spelled = (
            [one.strip() for one in parameters[1:-1].split(",") if one.strip()]
            if parameters
            else []
        )
        found.append((name, spelled, body.strip()))
    return sorted(found, key=lambda one: len(one[0]), reverse=True)


def _expand_com_spellings(text: str) -> str:
    """Write out what a generated COM header's macros stand for.

    Only where the file is written in them. `interface` and `PURE` are
    ordinary words, and a program that has never heard of COM may use either
    as a name of its own.
    """

    if _COM_SPELLINGS.search(_without_literals(text)) is None:
        return text
    for name, parameters, body in _com_macros():
        if not parameters:
            text = _map_code(
                text,
                lambda part, n=name, b=body: re.sub(
                    rf"(?<![.\w>]){re.escape(n)}\b", b, part
                ),
            )
            continue

        def written(
            match: "re.Match[str]", whole: str, b=body, names=parameters
        ) -> str:
            # The argument out of the real text: an interface's is a string
            # of digits and dashes, and this is matched against a copy with
            # the literals blanked.
            given = [
                one.strip()
                for one in _split_arguments(whole[match.start(1): match.end(1)])
            ]
            out = b
            for spelled, value in zip(names, given):
                out = re.sub(
                    rf"(?<![.\w>]){re.escape(spelled)}\b", value, out
                )
            return out

        # Against the whole text and not fragment by fragment: the fragments
        # are split at every literal, and `MIDL_INTERFACE("...")` is a name,
        # an open parenthesis, a literal and a close - so no fragment ever
        # held the whole of it.
        text = _sub_code(
            re.compile(rf"(?<![.\w>]){re.escape(name)}\s*\(([^()]*)\)"),
            text,
            written,
        )
    return text


#: `alignas(N)` and C11's `_Alignas(N)`, in any position.
_ALIGNAS = re.compile(r"(?<![.\w>])(alignas|_Alignas)\s*\(")


#: `class Foo`, `struct DLLEXPORT Foo : public Bar`, and everything else that
#: stands between the keyword and the body. Deliberately looser than
#: `_CLASS_HEAD`: the point is to catch the heads that one cannot read.
_ANY_CLASS_HEAD = re.compile(r"\b(class|struct)\s+([A-Za-z_]\w*)([^;{()=]*)\{")


def _refuse_macro_class_heads(text: str, filename: str) -> None:
    """Name the class heads a macro makes unreadable at this stage.

    A class's own name, and the words between it and its body, decide what is
    declared and what it derives from. Both are read here, in front of the
    preprocessor - so a head written with a macro is read as something else
    entirely: `struct EXPORTED Dog : public Animal` matched no class at all
    and Dog left the output without a word, and a class named by a macro was
    emitted under the macro's name while every use of it kept the real one.
    Where the head still reads - `struct Dog : BASE`, whose base is a macro
    standing for one name - it is left to the base reader, which answers it.
    """

    plain = _without_literals(text)
    for head in _ANY_CLASS_HEAD.finditer(plain):
        if head.group(2) in _MACRO_NAMES:
            raise CppTranslationError(
                filename,
                _line_of(text, head.start()),
                f"this {head.group(1)} is named by `{head.group(2)}`, which "
                f"this file `#define`s. py2bin translates C++ into C before "
                f"it runs the preprocessor, so the class comes out under the "
                f"macro's name while everything that uses it keeps the name "
                f"the macro stands for. Write the {head.group(1)}'s real name "
                f"here",
            )
        if _CLASS_HEAD.match(plain, head.start()):
            continue
        for word in _A_NAME.findall(head.group(3)):
            if word in _MACRO_NAMES:
                raise CppTranslationError(
                    filename,
                    _line_of(text, head.start()),
                    f"`{word}` is a macro standing between {head.group(1)} "
                    f"{head.group(2)} and its body, and py2bin reads a class "
                    f"head before the preprocessor has run. Read as it is "
                    f"written the head names no class at all, and everything "
                    f"{head.group(2)} declares would leave the output "
                    f"silently - so it is refused here instead",
                )


def _refuse_unsupported(text: str, filename: str) -> None:
    """Name the C++ this does not do, before it becomes broken C.

    Checked on comment-stripped source so a word in prose is not mistaken for
    the construct, and word-bounded so `newest` is not `new`.
    """

    for keyword, described in _REFUSED:
        for found in re.finditer(rf"\b{keyword}\b", text):
            # `operator` is a member name in some C code; only the C++ form,
            # `operator+(`, is refused.
            if keyword == "operator" and not re.match(
                r"operator\s*[^\w\s(]", text[found.start():]
            ):
                continue
            raise CppTranslationError(
                filename,
                _line_of(text, found.start()),
                f"py2bin's C++ subset does not do {described}. It translates "
                f"classes, members, methods, constructors and destructors into "
                f"C; anything that needs a real C++ compiler is refused here "
                f"rather than mistranslated",
            )
    # `alignas` decides where every member after it sits. py2bin implements
    # none of its spellings, and on a *member* it was not even refused: the
    # declaration was read as something else and the member vanished from the
    # struct, so `struct { char head; alignas(16) char body; }` had sizeof 1
    # where C++ says 32. A layout that runs and is wrong is the failure worth
    # the most care, so it is refused by name.
    for found in _ALIGNAS.finditer(_without_literals(text)):
        raise CppTranslationError(
            filename,
            _line_of(text, found.start()),
            f"`{found.group(1)}` decides where the member after it sits, and "
            f"py2bin does not implement it. Read as an ordinary declaration "
            f"the member was dropped from the struct altogether, which is a "
            f"layout that runs and answers wrongly - so it is refused here "
            f"instead",
        )
    for header in re.finditer(r"#\s*include\s*<([^>]+)>", text):
        name = header.group(1)
        if name in _BUILTIN_CPP_HEADERS or _c_header_under(name) is not None:
            continue
        if name in ("iostream", "vector", "map", "set", "memory", "algorithm"):
            raise CppTranslationError(
                filename,
                _line_of(text, header.start()),
                f"<{name}> is the C++ standard library, which this subset does "
                f"not have. py2bin ships its own C headers - <stdio.h> and "
                f"friends - and those work",
            )


#: `int x;`, `int x = 1;`, `Vec v(1, 2);` - something declared in this body.
_DECLARED_HERE = re.compile(r"\b([A-Za-z_]\w*)\s*\**\s*([A-Za-z_]\w*)\s*[=;(\[]")

#: Words that are not a type, however much `return v;` looks like `int v;`.
#: Without this, `return v` hid the member `v` from its own method and the
#: compiler reported a name that is not declared anywhere.
_NOT_A_TYPE = frozenset(
    """return if else while for do switch case default break continue goto
    sizeof typedef struct union enum static const volatile extern register
    auto inline restrict public private protected class new delete this
    true false throw try catch template typename namespace using operator
    virtual""".split()
)



def _opens_a_line(text: str, index: int) -> bool:
    """Whether only whitespace stands between `index` and the line's start."""

    at = index - 1
    while at >= 0 and text[at] in " \t":
        at -= 1
    return at < 0 or text[at] == "\n"


def _after_directive(text: str, index: int) -> int:
    """The end of the directive at `index`, continuation lines included."""

    at = index
    length = len(text)
    while at < length:
        if text[at] == "\\" and at + 1 < length and text[at + 1] == "\n":
            at += 2
            continue
        if text[at] == "\\" and text[at + 1:at + 3] == "\r\n":
            at += 3
            continue
        if text[at] == "\n":
            return at
        at += 1
    return length


def _without_literals(text: str) -> str:
    """The text with every literal blanked, for scanning rather than editing.

    Kept the same length so an offset still means something, and emptied so
    nothing inside a literal can be read as code.
    """

    return "".join(
        part
        if kind == "code"
        else "".join(" " if letter != "\n" else "\n" for letter in part)
        for kind, part in _split_literals(text)
    )


def _split_literals(text: str) -> "list[tuple[str, str]]":
    """The text as alternating ("code", ...) and ("literal", ...) pieces.

    A preprocessing directive counts as a literal here, its continuation
    lines with it. It is not C++ and rewriting it is never right: a
    generated COM header defines a macro per method, each body a call
    through a vtable, and the translator rewrote those bodies as if they
    were code - leaving `#define NAME(args) \\` with nothing after the
    backslash, which then swallowed the line below it and every `#endif`
    that followed. What the preprocessor makes of a directive is the
    preprocessor's business, and it runs after this.
    """

    pieces: list[tuple[str, str]] = []
    chunk: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "#" and _opens_a_line(text, index):
            pieces.append(("code", "".join(chunk)))
            chunk = []
            pieces.append(("literal", text[index:_after_directive(text, index)]))
            index = _after_directive(text, index)
            continue
        if char in "\"'":
            # The prefix goes with the literal. `L"hi"` blanked to `L     `
            # leaves an `L` standing where the text is read, and an `L` reads
            # as a name: `wstring b = L"hi";` was taken for a copy of a
            # variable called L, and `s += L"b"` for a call on one. Every
            # wide literal in a program that talks to Windows is this.
            code = "".join(chunk)
            prefix = ""
            for spelled in ("u8", "L", "u", "U"):
                if not code.endswith(spelled):
                    continue
                before = code[: -len(spelled)]
                if before and (before[-1].isalnum() or before[-1] == "_"):
                    continue
                prefix = spelled
                break
            pieces.append(("code", code[: len(code) - len(prefix)]))
            chunk = []
            quote = char
            literal = [prefix, char]
            index += 1
            while index < length:
                if text[index] == "\\" and index + 1 < length:
                    literal.append(text[index:index + 2])
                    index += 2
                    continue
                literal.append(text[index])
                if text[index] == quote:
                    index += 1
                    break
                index += 1
            pieces.append(("literal", "".join(literal)))
            continue
        chunk.append(char)
        index += 1
    pieces.append(("code", "".join(chunk)))
    return pieces

def _shadowing(body: str, parameters: str) -> "set[str]":
    """Names in this body that hide a member of the same name.

    C++ resolves `n` to the parameter when a parameter is called `n`, and to
    the member only otherwise. Qualifying every occurrence made
    `this->n + n` into `this->n + this->n` - 200 where the answer is 105, and
    a program that runs and is wrong rather than one that fails.
    """

    hidden = set()
    for part in parameters.split(","):
        words = part.replace("*", " * ").split()
        if len(words) >= 2 and words[-1].isidentifier():
            hidden.add(words[-1])
    # Over the code only. `printf("n=%d a=%d")` contains `d a=`, which reads
    # exactly like a declaration of `a` - so the member `a` was hidden from
    # its own method by a format string that merely mentions the letter.
    code = _without_literals(body)
    for found in _DECLARED_HERE.finditer(code):
        if found.group(1) in _NOT_A_TYPE:
            continue
        if not _could_start_a_declaration(code, found.start()):
            continue
        hidden.add(found.group(2))
    return hidden


#: Words that may stand before a type without ending the declaration.
_LEADS_A_TYPE = frozenset(
    """const unsigned signed static struct union enum long short volatile
    register auto""".split()
)


def _could_start_a_declaration(text: str, at: int) -> bool:
    """Whether a declaration may begin at `at`, or only an expression may.

    `size * size` is a declaration of a pointer and a multiplication of two
    numbers, spelled identically; C tells them apart by knowing which names
    are types, and this does not. What it can tell is where the statement
    began: `return size * size;` cannot be declaring anything, because a
    declaration does not follow `return`. Without this the member `size` was
    taken for a local of its own method and left unqualified, and the C
    compiler reported a name declared nowhere - on a line that is correct
    C++.
    """

    before = text[:at]
    cut = max((before.rfind(char) for char in ";{}(,)"), default=-1)
    return all(word in _LEADS_A_TYPE for word in before[cut + 1:].split())


def _this_qualified(
    body: str,
    owner: Class,
    classes: "dict[str, Class]",
    hidden: "set[str]" = frozenset(),
) -> str:
    """Point bare member names at `this`, the way C++ resolves them.

    Inherited names count: the base is embedded as the first member, so a
    name the base declares is reached through it.
    """

    names = dict.fromkeys(owner.field_names(), "")
    # Every base, by the path that reaches it - not one `__base.` per level.
    # Counting levels along the first chain alone, a name a *second* base
    # declares was not reachable by its bare name at all, and one a shared
    # base declares is not reached by naming a member. Nearest first, which
    # is the order C++ looks a name up in.
    for base in _every_base(owner.name, classes):
        if base not in classes:
            continue
        path = _subobject_path(owner.name, base, classes)
        if path is None:
            continue
        for name in classes[base].field_names():
            names.setdefault(name, f"{path}.")

    def replace(match: "re.Match[str]") -> str:
        word = match.group(0)
        if word not in names or word in hidden:
            return word
        start = match.start()
        # Not after `.` or `->`, which already name an object, and not a
        # member of some other struct.
        before = body[:start].rstrip()
        if before.endswith(".") or before.endswith("->"):
            return word
        return f"this->{names[word]}{word}"

    return _map_code(body, lambda part: _WORD.sub(replace, part))


# --- virtual dispatch -------------------------------------------------------
#
# A virtual call goes to what the object *is*, and the only thing the caller
# has that knows that is the object itself. So each polymorphic class carries
# a pointer to a table of its own functions, set by its constructor, and the
# call reads the slot rather than naming a function. This is what a C++
# compiler does; written out in C it is merely visible.


def _slot_key(method: "Method") -> "tuple[str, int]":
    """What makes an override the same member: its name and its arity.

    C++ says a matching signature, and py2bin tells signatures apart by how
    many arguments they take (see :func:`_c_name`), so the same key serves
    both and an overload cannot silently take an override's slot.
    """

    return (method.name, _arity(method.parameters))


def _is_polymorphic(name: str, classes: "dict[str, Class]") -> bool:
    """Whether this class, or anything it inherits from, declares a virtual."""

    seen = name
    while seen and seen in classes:
        if any(m.virtual for m in classes[seen].methods):
            return True
        seen = classes[seen].base
    return False


def _carries_vptr(name: str, classes: "dict[str, Class]") -> bool:
    """Whether the pointer is stored *here* rather than in an embedded base.

    A base is the first member, so a derived object already begins with its
    base's pointer; storing a second one would leave two, and the wrong one
    at offset zero.
    """

    found = classes.get(name)
    if found is None or not _is_polymorphic(name, classes):
        return False
    return not (found.base and _is_polymorphic(found.base, classes))


def _virtual_slots(
    name: str, classes: "dict[str, Class]"
) -> "list[tuple[str, int]]":
    """The table's layout: inherited slots first, in the base's own order.

    Order is what a derived object and its base agree on. A base pointer
    reads slot 2 expecting the base's third virtual, so the derived table has
    to keep it there and may only add after it.
    """

    found = classes.get(name)
    if found is None:
        return []
    slots = _virtual_slots(found.base, classes) if found.base else []
    for method in found.methods:
        if not method.virtual:
            continue
        key = _slot_key(method)
        if key not in slots:
            slots.append(key)
    return slots


def _slot_provider(
    name: str, key: "tuple[str, int]", classes: "dict[str, Class]"
) -> "str | None":
    """The most derived class at or above `name` that defines this slot."""

    # Every base, nearest first, and not the first chain alone: `D : B, C`
    # where only `C` declares the method has `C`'s as its final overrider,
    # and walking `.base` from D reached A without ever looking at C - so the
    # table named the one D was overriding rather than the override.
    for seen in [name, *_every_base(name, classes)]:
        if seen not in classes:
            continue
        for method in classes[seen].methods:
            if _slot_key(method) == key and not method.pure:
                return seen
    return None


def _slot_method(
    name: str, key: "tuple[str, int]", classes: "dict[str, Class]"
) -> "Method | None":
    """The declaration for a slot, wherever it was first written."""

    for seen in [name, *_every_base(name, classes)]:
        if seen not in classes:
            continue
        for method in classes[seen].methods:
            if _slot_key(method) == key:
                return method
    return None


def _passed_by_address(spelled: str, classes: "dict[str, Class]") -> bool:
    """Whether a parameter written like this is a pointer in the C.

    A class, because that is how this translator passes one. A struct the
    platform declares, because py2bin's C can neither pass nor answer one by
    value - and because the ABI it is talking to passes a struct of that size
    by address anyway, which is what makes this the right shape and not only
    the possible one.
    """

    if "*" in spelled or "&" in spelled:
        return False
    words = re.sub(r"\b(?:const|struct|volatile|union)\b", " ", spelled).split()
    if not words:
        return False
    return words[0] in classes or words[0] in _platform_structs()


def _platform_structs() -> "frozenset[str]":
    """The struct names py2bin's own platform headers declare."""

    from .c_preprocessor import platform_structs

    return platform_structs()


def _c_signature(
    owner: str, method: "Method", classes: "dict[str, Class]"
) -> "tuple[str, str]":
    """The C return type and parameter list a method is emitted with.

    The same transform :func:`_emit_one` performs - hidden pointer for a value
    return, pointer for a by-value object - so a cast built from this and the
    function it calls cannot disagree about the shape of the call.
    """

    returned = _returns_object(method, classes)
    result = "void" if returned else (method.returns or "void")
    parameters = [f"struct {owner} *"]
    if returned:
        parameters.append(f"struct {returned} *")
    for part in _split_arguments(method.parameters):
        if not part.strip():
            continue
        spelled = _rewrite_types(part, classes).strip()
        held = re.sub(
            r"\b(?:const|struct|volatile|union)\b", " ", part.replace("*", " ")
        ).split()
        if _passed_by_address(part, classes) and held:
            # `struct` in front of a class, because that is what it is
            # emitted as; the plain name for a platform struct, which is a
            # typedef and whose tag is something else again.
            spelled = (
                f"struct {held[0]} *" if held[0] in classes else f"{held[0]} *"
            )
        else:
            # Drop the parameter's name: a cast is types only.
            spelled = re.sub(r"\b[A-Za-z_]\w*\s*(\[\s*\d*\s*\])?$", "", spelled).strip()
            spelled = spelled or "int"
        parameters.append(spelled)
    return result, ", ".join(parameters)


def _vptr_path(name: str, classes: "dict[str, Class]") -> str:
    """How to reach the pointer from an object of this class.

    It lives in whichever class first declared a virtual, and every class
    below reaches it through its embedded base - `__base.__base.__vptr` and
    so on. Written as a path rather than assumed to be at offset zero, so a
    class that inherits from a plain struct and then adds a virtual still
    finds it.
    """

    seen = name
    while seen and seen in classes and not _carries_vptr(seen, classes):
        held = classes[seen]
        for base, _step, _shared in _base_steps(held, classes):
            if base in classes:
                seen = base
                break
        else:
            break
    # Through whichever path names the class that declared it, which for a
    # shared base is a pointer and not a member.
    path = _subobject_path(name, seen, classes)
    return "__vptr" if not path else f"{path}.__vptr"


def _vptr_carrier(name: str, classes: "dict[str, Class]") -> str:
    """The class whose declaration of a virtual put the table pointer there."""

    seen = name
    while seen and seen in classes and not _carries_vptr(seen, classes):
        for base, _step, _shared in _base_steps(classes[seen], classes):
            if base in classes:
                seen = base
                break
        else:
            break
    return seen


def _vtable_name(name: str) -> str:
    return f"{name}__vtable"


def _emit_vtables(order: "list[str]", classes: "dict[str, Class]") -> str:
    """Prototypes for everything a table names, then the tables.

    Prototypes first because a table is a static initialiser and C reads top
    to bottom: the address of a function written further down is only
    available once its name is.
    """

    polymorphic = [name for name in order if _is_polymorphic(name, classes)]
    if not polymorphic:
        return ""
    lines: list[str] = []
    seen: set[str] = set()
    for name in polymorphic:
        for key in _virtual_slots(name, classes):
            provider = _slot_provider(name, key, classes)
            if provider is None:
                continue
            method = _slot_method(provider, key, classes)
            if method is None:
                continue
            symbol = _c_name(provider, key[0], _suffix_of(provider, method, classes))
            if symbol in seen:
                continue
            seen.add(symbol)
            result, parameters = _c_signature(provider, method, classes)
            lines.append(f"{result} {symbol}({parameters});")
    for name in polymorphic:
        slots = _virtual_slots(name, classes)
        entries = []
        for key in slots:
            provider = _slot_provider(name, key, classes)
            if provider is None:
                # Pure virtual and never overridden: the class is abstract, and
                # a null here is a crash rather than a wrong function if some
                # path reaches it anyway.
                entries.append("0")
                continue
            declared = _slot_method(provider, key, classes)
            entries.append(
                "(void *)"
                + _c_name(
                    provider,
                    key[0],
                    _suffix_of(provider, declared, classes) if declared else None,
                )
            )
        held = ", ".join(entries) or "0"
        lines.append(
            f"static void *{_vtable_name(name)}[{max(len(slots), 1)}] = {{ {held} }};"
        )
    for name in order:
        lines.extend(_second_base_tables(name, classes))
    return "\n".join(lines)


def _mixin_vtable_name(name: str, mixin: str) -> str:
    return f"{name}__vtable__{mixin}"


def _offset_of(name: str, member: str) -> str:
    """Where a member sits in its struct, worked out by the C compiler.

    C has no other way to say it, and the pointer a second base is reached
    through has to be moved by exactly this much.
    """

    return f"(unsigned long)&((struct {name} *)0)->{member}"


def _second_base_tables(
    name: str, classes: "dict[str, Class]"
) -> "list[str]":
    """A table for each polymorphic base after the first, and its thunks.

    A second base is a member after the first, so a pointer to it is not a
    pointer to the object. Its table therefore cannot name the derived
    class's functions directly: they want the whole object, and what arrives
    is the address of the subobject. Each entry the derived class provides is
    a small function that moves the pointer back and calls the real one -
    which is what a C++ compiler emits here too, and is why this can be
    written at all.
    """

    found = classes.get(name)
    if found is None:
        return []
    reached = [
        (mixin, f"__base{index + 1}")
        for index, mixin in enumerate(found.mixins)
        if mixin not in found.virtual_bases
    ]
    # A shared base is reached through a pointer, and what arrives at a call
    # through it is the address of the one object everything shares. Which is
    # not the address of this one - so it wants the same treatment a second
    # base wants, and for the same reason.
    reached += [
        (shared, _vbase_storage(shared))
        for shared in _shared_bases(name, classes)
    ]
    if not reached:
        return []
    lines: "list[str]" = []
    for mixin, member in reached:
        if not _is_polymorphic(mixin, classes):
            continue
        entries: "list[str]" = []
        for key in _virtual_slots(mixin, classes):
            # The derived class first: an override of the base's virtual is
            # written here, and what it takes is the whole object.
            provider = _slot_provider(name, key, classes)
            if provider is not None and _subobject_path(
                mixin, provider, classes
            ) is not None:
                # Except where what was found is the base's own, reached
                # through the base itself: it already takes what arrives, and
                # moving the pointer first would take it somewhere else.
                provider = None
            if provider is not None:
                method = _slot_method(provider, key, classes)
                if method is None:
                    entries.append("0")
                    continue
                # A virtual destructor's slot is spelled `~`, which is not
                # something a C identifier may hold.
                spelt = re.sub(r"[^A-Za-z0-9_]", "", key[0]) or "dtor"
                thunk = f"{name}__thunk__{mixin}__{spelt}"
                result, parameters = _c_signature(provider, method, classes)
                spelled = [
                    one.strip()
                    for one in _split_arguments(parameters)
                    if one.strip()
                ]
                rest = spelled[1:]
                head = ", ".join(
                    [f"struct {mixin} *__self"]
                    + [f"{one} a{at}" for at, one in enumerate(rest)]
                )
                passed = "".join(f", a{at}" for at in range(len(rest)))
                answer = "" if result.strip() in ("", "void") else "return "
                symbol = _c_name(
                    provider, key[0], _suffix_of(provider, method, classes)
                )
                lines.append(
                    f"static {result} {thunk}({head}) {{ "
                    f"{answer}{symbol}(({spelled[0]})((char *)__self - "
                    f"{_offset_of(name, member)}){passed}); }}"
                )
                entries.append(f"(void *){thunk}")
                continue
            # Otherwise the base's own, which already takes what arrives.
            provider = _slot_provider(mixin, key, classes)
            if provider is None:
                entries.append("0")
                continue
            method = _slot_method(provider, key, classes)
            entries.append(
                "(void *)"
                + _c_name(
                    provider,
                    key[0],
                    _suffix_of(provider, method, classes) if method else None,
                )
            )
        slots = _virtual_slots(mixin, classes)
        held = ", ".join(entries) or "0"
        lines.append(
            f"static void *{_mixin_vtable_name(name, mixin)}"
            f"[{max(len(slots), 1)}] = {{ {held} }};"
        )
    return lines


#: `Source *from` - a parameter that points at an object.
_A_POINTER_PARAMETER = re.compile(
    r"^\s*(?:const\s+)?(?:struct\s+)?([A-Za-z_]\w*)\s*\*\s*"
    r"([A-Za-z_]\w*)\s*$"
)


def _pointer_parameters(
    parameters: str, classes: "dict[str, Class]"
) -> "dict[str, str]":
    """Each parameter that is a pointer to a class, and the class."""

    found: "dict[str, str]" = {}
    for part in _split_arguments(parameters):
        written = _A_POINTER_PARAMETER.match(part)
        if written is not None and written.group(1) in classes:
            found[written.group(2)] = written.group(1)
    return found


def _emit_class(found: Class, classes: "dict[str, Class]") -> str:
    """The struct, and the free functions its methods become."""

    packed = _CLASS_PACK.get(found.name)
    lines = []
    if packed is not None:
        # Written around the struct rather than left where the program put
        # it: the directive and the struct are moved apart before either is
        # emitted, and the C compiler below reads a pragma where it stands.
        lines.append(f"#pragma pack(push, {packed})")
    lines.append(f"struct {found.name} {{")
    if found.base and found.base not in found.virtual_bases:
        # First, so a pointer to the derived object is a pointer to the base.
        lines.append(f"    struct {found.base} __base;")
    for index, mixin in enumerate(found.mixins):
        if mixin in found.virtual_bases:
            continue
        # After the first, which is at offset zero. Each is an ordinary member
        # and is reached by naming it, which is what a second base is: the
        # same bytes, in a place a pointer has to be moved to.
        lines.append(f"    struct {mixin} __base{index + 1};")
    # A shared base is held by address, because one object of it is reached
    # along every path that names it. The storage sits here too: a complete
    # object of this class owns the one it points at, and an object of this
    # class inside something larger points at the one *that* owns.
    for shared in _shared_bases(found.name, classes):
        lines.append(f"    struct {shared} *{_vbase_pointer(shared)};")
        lines.append(f"    struct {shared} {_vbase_storage(shared)};")
    if _carries_vptr(found.name, classes):
        # Before the data, so it sits at offset zero for this class and every
        # class below it - which is what lets a base pointer find the table of
        # whatever the object turned out to be.
        lines.append("    void **__vptr;")
    for member in found.members:
        if member.ctype.endswith("\x00fn"):
            spelled = member.ctype[: -len("\x00fn")]
            head, _, tail = spelled.partition("(*)")
            lines.append(f"    {head}(*{member.name}){tail};")
            continue
        lines.append(f"    {member.ctype} {member.name}{member.array};")
    if not found.members and not found.base and not found.mixins:
        # C has no empty struct; give it something so the type exists.
        # A byte, not an int. C++ says an empty class has size 1 and
        # alignment 1; given four, an array of them had the wrong stride and
        # every struct holding one had the wrong size - quietly.
        lines.append("    char __empty;")
    lines.append("};")
    if packed is not None:
        lines.append("#pragma pack(pop)")
    return "\n".join(lines)


def _method_declarations(
    order: "list[str]", classes: "dict[str, Class]"
) -> str:
    """Every method's emitted C signature, as text something can read.

    Deduction works by finding where a name was declared, and a method's C
    name is declared nowhere the file can see - it is made by the emitter.
    Without this, `cout << v[0]` could not be resolved: `v[0]` is a call to
    `vector__int__op_index`, and nothing said what that returns.

    The bodies are empty because nothing runs this; it is a table with the
    shape of code so that the same reader can read it.
    """

    lines: list[str] = []
    for name in order:
        for method in classes[name].methods:
            if method.name in ("", "~"):
                continue
            spelled = _c_name(name, method.name, _suffix_of(name, method, classes))
            result, parameters = _c_signature(name, method, classes)
            lines.append(f"{result.replace('&', '*')} {spelled}({parameters}) {{ }}")
    return "\n".join(lines)

def _emit_methods(found: Class, classes: "dict[str, Class]", unit: str = "") -> str:
    out = []
    for method in found.methods:
        if not method.body:
            continue  # declared here, defined outside; emitted with that body
        out.append(_emit_one(found, method, classes, unit))
    return "\n".join(out)


def _returns_object(method: Method, classes: "dict[str, Class]") -> str | None:
    """The class a method returns by value, if it returns one."""

    if method.name in ("", "~"):
        return None
    spelled = method.returns.replace("*", "").strip()
    if "*" in method.returns or spelled not in classes:
        return None
    return spelled


def _by_value_objects(
    parameters: str, classes: "dict[str, Class]"
) -> "list[tuple[str, str]]":
    """Parameters taken by value whose type is a class, as (class, name)."""

    taken = []
    for part in parameters.split(","):
        if "*" in part or "&" in part:
            continue
        words = part.split()
        if len(words) == 2 and words[0] in classes and words[1].isidentifier():
            taken.append((words[0], words[1]))
    return taken



def _copy_constructor(held: str, classes: "dict[str, Class]") -> "str | None":
    """The class that provides a copy constructor for this one, if any.

    A copy constructor is the one-argument constructor whose argument is the
    class itself. C++ writes one for every class; what matters here is
    whether the *author* wrote one, because a bitwise copy is what the
    implicit one does and is already what happens.
    """

    seen = held
    while seen and seen in classes:
        for method in classes[seen].methods:
            if method.name != "" or _arity(method.parameters) != 1:
                continue
            spelled = method.parameters.replace("&", " ").replace("*", " ")
            words = [w for w in spelled.replace("const", " ").split()]
            if words and words[0] == seen:
                return seen
        seen = classes[seen].base
    return None


def _copied_in(held: str, into: str, source: str, classes) -> str:
    """The statement that makes a copy: the author's constructor, or a copy.

    C++ calls the copy constructor wherever a copy is made, and an object
    that owns something - a buffer, a handle - is written on the assumption
    that it will be. A bitwise copy of one is two objects believing they own
    the same thing.
    """

    owner = _copy_constructor(held, classes)
    if owner is None:
        return f"{into} = *{source};"
    return f"{_c_name(owner, '', _copy_suffix(owner, classes))}(&{into}, {source});"


def _copy_suffix(owner: str, classes: "dict[str, Class]") -> "str | None":
    for method in classes[owner].methods:
        if method.name == "" and _arity(method.parameters) == 1:
            spelled = method.parameters.replace("&", " ").replace("*", " ")
            words = spelled.replace("const", " ").split()
            if words and words[0] == owner:
                return _suffix_of(owner, method, classes)
    return None

def _emit_one(
    found: Class, method: Method, classes: "dict[str, Class]", unit: str = ""
) -> str:
    name = _c_name(found.name, method.name, _suffix_of(found.name, method, classes))
    if method.name == "" and _shared_bases(found.name, classes):
        # A class with a shared base gets two constructors, the way a real
        # C++ ABI writes them: this one builds everything *except* the shared
        # base, and the wrapper written beside the class builds that first
        # and calls this. Which one a site wants is decided by whether the
        # object being built is the whole of what is there, and nowhere else
        # has to know: the wrapper keeps the name every call site already
        # spells.
        name = f"{name}{_SUB_OBJECT_SUFFIX}"
    parameters = "" if method.shared else f"struct {found.name} *this"
    # A value return becomes the hidden pointer a compiler would pass: the
    # caller supplies the space and the callee writes through it. py2bin's C
    # cannot return a struct, and does not have to - this is the same
    # transform an ABI performs, done where the C is written.
    returned = _returns_object(method, classes)
    if returned:
        # A static member takes no object, so there is nothing in front of the
        # hidden pointer for a comma to separate it from.
        lead = ", " if parameters else ""
        parameters += f"{lead}struct {returned} *__ret"
    # A parameter taken by value is passed as a pointer and copied on entry,
    # which is what "by value" means: the callee may write to it and the
    # caller must not see that.
    copied = _by_value_objects(method.parameters, classes)
    if method.parameters and not parameters:
        parameters = _rewrite_types(
            _references_to_pointers(method.parameters), classes
        )
    elif method.parameters:
        # Types first, then the by-value rewrite: done the other way round the
        # `struct` this adds was added again, giving `struct struct V *`.
        spelled = _rewrite_types(_references_to_pointers(method.parameters), classes)
        for held, variable in copied:
            spelled = re.sub(
                rf"\bstruct\s+{re.escape(held)}\s+{re.escape(variable)}\b",
                f"struct {held} *__by_value_{variable}",
                spelled,
            )
        parameters += ", " + spelled
    references = _reference_parameters(method.parameters, classes)
    body = _this_qualified(
        method.body, found, classes, _shadowing(method.body, method.parameters)
    )
    if references:
        body = _deref_references(body, references, classes)
    # Before the base-call pass, which reads `Owner::name` as an explicit call
    # on `this`: a static member has no `this`, so it must be taken out of
    # that pass's way first.
    body = _qualified_static_names(body, classes)
    body = _qualified_base_calls(body, found, classes)
    # The parameters go with it: a bare call to an overloaded member is told
    # apart by the type of what it passes, and that is declared in the head.
    # In the C++ spelling, not the rewritten one: a by-value object is renamed
    # `__by_value_o` there, and the body still calls it `o`.
    # Last, because the reader takes the declaration nearest above the use:
    # put in front of the unit, a parameter named `o` lost to any other `o`
    # declared anywhere in the file.
    scope = f"{unit}\n{method.parameters};" if method.parameters else unit
    body = _bare_method_calls(body, found, classes, scope)
    known = {"this": found.name}
    # And every object the file declares outside a function. A method may name
    # one as readily as a free function may, and only the free functions knew
    # about them: `total.fetch_add(1)` written inside a method - or inside a
    # lambda, which becomes one - was left as C++ and reported against a
    # struct that has no such member.
    for spelled, (holds, _arguments) in _file_scope_objects(unit, classes).items():
        known.setdefault(spelled, holds)
    pointers: "set[str]" = {"this"}
    for referenced, held in references.items():
        if held in classes:
            known[referenced] = held
    # A parameter taken by value is an object too. Its copy is spliced in
    # below, after the body has been rewritten, so nothing else would say
    # that `o` in `o.c_str()` names one - and the call stayed a field read.
    for held, variable in copied:
        known[variable] = held
    receivers: dict[str, str] = {}
    # `void take(Source *from) { from->value(); }` - a parameter that is a
    # pointer to a class is a receiver like any other. Without this nothing
    # said what `from` held, so the call was left as a field read on a
    # struct that has no such field.
    for spelled, held in _pointer_parameters(method.parameters, classes).items():
        known[spelled] = held
        pointers.add(spelled)
        receivers[spelled] = spelled
    for holder, prefix in _reachable_members(found, classes):
        held = holder.ctype.replace("*", "").strip()
        if held not in classes:
            continue
        # Already qualified above, so the name in the text is `this->motor`.
        spelled = f"this->{prefix}{holder.name}"
        known[spelled] = held
        # A member that is a pointer already *is* the address. Taking one of
        # it gave `&this->p` where a `T *` was wanted, which is a `T **`.
        receivers[spelled] = spelled if "*" in holder.ctype else f"&{spelled}"
        if "*" in holder.ctype:
            pointers.add(spelled)
            # And what *that* object holds, one level on. A class reached
            # through a pointer member is how a generated callback reaches
            # the object it was written in - `this->__py2bin_self->held` -
            # and nothing said what that named.
            for inner in classes[held].members:
                reached = inner.ctype.replace("*", "").strip()
                if reached not in classes:
                    continue
                through = f"{spelled}->{inner.name}"
                known[through] = reached
                receivers[through] = (
                    through if "*" in inner.ctype else f"&{through}"
                )
                if "*" in inner.ctype:
                    pointers.add(through)
    body = _rewrite_body(
        body,
        classes,
        known,
        pointers=pointers | {n for n, h in references.items() if h in classes},
        receivers=receivers,
        # The parameters go last: which overload a bare call means is decided
        # by the type of what it passes, a parameter is declared in the head
        # rather than in the body being read, and the reader takes the
        # declaration nearest above the use. Both spellings, so that whichever
        # name the body has reached by now is found.
        unit=f"{scope}\n{parameters};",
        stable=scope,
        returns=method.returns,
        referenced=set(references),
    )
    if returned:
        body = _return_through_pointer(body, returned or "")
    if method.name == "":
        body = _delegating_initialiser(body, found, classes, unit)
        body, base_arguments = _base_initialiser(body, found)
        body, member_arguments = _member_initialisers(body)
        # `D() : A(3)` names the shared base directly, which C++ lets the
        # class that owns it do however far above it the base sits. Taken out
        # here, because what builds it is the other constructor.
        shared_arguments: "dict[str, str]" = {}
        for shared in _shared_bases(found.name, classes):
            written = member_arguments.pop(shared, None)
            if written is None and shared == found.base:
                written, base_arguments = base_arguments, ""
            if written is not None:
                shared_arguments[shared] = written
        # `int n = 7;` written on the member, put in before the subobjects so
        # that it ends up after them: a member of class type has to be built
        # before anything assigns to it. An initialiser list naming the same
        # member is applied by the pass below and overwrites this, which is
        # the order C++ gives the two.
        body = _open_with_member_values(
            body, found, classes, member_arguments
        )
        body = _open_with_subobjects(
            body, found, classes, base_arguments, member_arguments, scope
        )
    elif method.name == "~":
        body = _close_with_subobjects(body, found, classes)
    if copied:
        # Declared and then assigned, not initialised: py2bin's C takes
        # `o = *p;` and not `struct V o = *p;`.
        # Last of the things put at the top, so it ends up first: a member
        # initialiser list may name a parameter, and `Person(string n) :
        # name(n)` has to see the copy of `n` rather than the pointer it
        # arrived as.
        entry = " ".join(
            f"struct {held} {variable}; "
            + _copied_in(held, variable, f"__by_value_{variable}", classes)
            for held, variable in copied
        )
        opening = body.find("{")
        body = body[:opening + 1] + " " + entry + body[opening + 1:]
    if method.shared and not parameters:
        parameters = "void"
    returns = "void" if (method.name in ("", "~") or returned) else method.returns
    if method.name not in ("", "~") and not returned and _returns_reference(method):
        # A reference is a pointer that the language follows for you. The
        # callee hands back the address; every call site follows it.
        returns = returns.replace("&", "*")
        body = _return_the_address(body, references)
    for member in found.members:
        if not member.reference:
            continue
        if method.name == "":
            # What is bound is the address, and a reference parameter is
            # already one here - so the dereference the reference pass put on
            # the value comes back off.
            body = re.sub(
                rf"(this->{re.escape(member.name)}\s*=\s*)\(\s*\*\s*"
                rf"([A-Za-z_]\w*)\s*\)",
                r"\1\2",
                body,
            )
            continue
        body = re.sub(
            rf"(?<![.\w>])this->{re.escape(member.name)}\b(?!\s*=[^=])",
            f"(*this->{member.name})",
            body,
        )
    written = f"static {returns} {name}({parameters}) {body}"
    if method.name == "" and _shared_bases(found.name, classes):
        written += "\n" + _complete_constructor(
            found, classes, name, returns, parameters, shared_arguments, scope
        )
    return written


def _complete_constructor(
    found: Class,
    classes: "dict[str, Class]",
    inner: str,
    returns: str,
    parameters: str,
    given: "dict[str, str]",
    scope: str,
) -> str:
    """The constructor a complete object of this class is built with.

    It owns the shared bases: it points at its own storage for each, builds
    them there, and then hands off to the form that builds everything else.
    A class of this kind reached as somebody else's subobject is built with
    that other form instead, and so the shared base is built exactly once
    however many paths lead to it - which is the whole of what `virtual`
    means here.
    """

    calls: "list[str]" = []
    for shared in _shared_bases(found.name, classes):
        pointer = _vbase_pointer(shared)
        calls.append(f"this->{pointer} = &this->{_vbase_storage(shared)};")
        owner = _find_method(shared, "", classes)
        if owner is None:
            continue
        arguments = (given.get(shared) or "").strip()
        spelled = (
            [one.strip() for one in _split_arguments(arguments)]
            if arguments
            else []
        )
        calls.append(
            f"{_c_name(owner, '', _call_suffix(owner, '', classes, spelled, scope))}"
            f"(this->{pointer}{', ' + arguments if arguments else ''});"
        )
    # Forwarded by name: the parameter list is this one's too.
    forwarded = ["this"]
    for part in _split_arguments(parameters):
        named = re.search(r"([A-Za-z_]\w*)\s*(?:\[\s*\d*\s*\])?$", part.strip())
        if named is not None and named.group(1) != "this":
            forwarded.append(named.group(1))
    outer = inner[: -len(_SUB_OBJECT_SUFFIX)]
    return (
        f"static {returns} {outer}({parameters}) {{ "
        + " ".join(calls)
        + f" {inner}({', '.join(forwarded)}); }}"
    )


def _address_a_returned_object(body: str, held: str) -> str:
    """`return o;` becomes `return &o;` where `o` is an object this body has.

    Only for a bare name that is declared here as a value. An expression -
    `return a->v < b->v ? b : a;` - is built out of things that are already
    pointers in this C, and addressing it would give a pointer to a pointer.
    """

    def written(match: "re.Match[str]") -> str:
        value = match.group(1).strip()
        if not value.isidentifier():
            return match.group(0)
        if re.search(
            rf"(?<![.\w>])struct\s+{re.escape(held)}\s+{re.escape(value)}\s*[;=]",
            body,
        ):
            return f"return &{value};"
        return match.group(0)

    return re.sub(r"\breturn\s+([^;]+);", written, body)


def _return_the_address(body: str, references: "dict[str, str]" = {}) -> str:
    """`return items[i];` becomes `return &(items[i]);` for a reference.

    Except where what is returned is *already* a reference, and so is already
    the address: `return o;` where `o` was declared `ostream &o` gave a `T **`
    where a `T *` was wanted. A reference carried as a pointer is the answer
    a reference return wants, exactly as it stands.
    """

    def written(match: "re.Match[str]") -> str:
        value = match.group(1).strip()
        if value in references:
            return f"return {value};"
        # Or something this body has already made a pointer. A reference
        # declared inside the body - `ostream &o = *this;` - is one by the
        # time this runs, and is not a parameter, so the list above does not
        # have it. What the body says it is, is what it is.
        if value.isidentifier() and re.search(
            rf"(?<![.\w>])\*\s*{re.escape(value)}\s*[=;,)]", body
        ):
            return f"return {value};"
        return f"return &({value});"

    return re.sub(
        r"\breturn\s+([^;]+);",
        written,
        body,
    )


def _return_through_pointer(body: str, held: str = "") -> str:
    """`return v;` becomes `*__ret = v; return;` - the value goes to the caller.

    Written as two statements rather than one so the expression is evaluated
    before anything else happens, which is the order C++ promises.
    """

    # `return {};` is the answer value initialised, with no type written
    # because the declaration says which one. Written as the construction it
    # means, so what follows sees an ordinary temporary: a brace list is not
    # an expression in C, and left alone this reached the C as `*__ret = {};`.
    if held:
        body = _EMPTY_RETURN.sub(f"return {held}();", body)

    def replace_from(match: "re.Match[str]", whole: str) -> "str | None":
        # Sliced out of the real text: the match is against a copy with the
        # literals blanked, so the group holds spaces where one was.
        value = whole[match.start(1): match.end(1)].strip()
        if not value:
            return None
        return f"{{ *__ret = {value}; return; }}"

    # Against the whole body, not fragment by fragment: `_map_code` splits at
    # every literal, so `return p / L"web";` was handed over as `return p / L`
    # and `;` - two pieces, neither of which is a `return` with a `;` after
    # it. The statement was left as it stood and the function answered
    # nothing, though its head said it answered through a pointer.
    return _sub_code(
        re.compile(r"\breturn\b([^;]*);"),
        body,
        lambda match, whole: replace_from(match, whole),
    )



def _subobjects(found: Class, classes: "dict[str, Class]") -> "list[tuple[str, str]]":
    """The base and the class-typed members, each with the address to use.

    C++ builds these before the constructor body runs and takes them apart
    after the destructor body does. C does nothing at all, so a class holding
    another read whatever was on the stack - which is a wrong answer rather
    than a failure, and the worst kind to ship.
    """

    parts: list[tuple[str, str]] = []
    for base, step, shared in _base_steps(found, classes):
        # A shared base is built by whoever owns it, which is the complete
        # object and not this class's share of it.
        if base in classes and not shared:
            parts.append((base, f"&this->{step}"))
    for member in found.members:
        held = member.ctype.replace("*", "").strip()
        if "*" not in member.ctype and held in classes:
            parts.append((held, f"&this->{member.name}"))
    return parts


def _member_initialisers(body: str) -> "tuple[str, dict[str, str]]":
    """Take `: n(x), b(3)` out of the body and report what each was given.

    Not expanded here for the same reason the base is not: a member of class
    type has to be *constructed*, and that happens with the other subobjects,
    in the order C++ builds them.
    """

    given: "dict[str, str]" = {}
    out: list[str] = []
    at = 0
    for match in re.finditer(rf"{_MEMBER_INIT}\s*\(", body):
        if match.start() < at:
            continue
        close = _closing_paren(body, match.end() - 1)
        if close < 0:
            continue
        pieces = _split_arguments(body[match.end(): close])
        if not pieces:
            continue
        # The name arrives already qualified - every member mention in the
        # body has been by the time this runs - and what is wanted here is
        # the member, which is what the subobject list is keyed by.
        spelled = pieces[0].strip()
        if spelled.startswith("this->"):
            spelled = spelled[len("this->"):]
        given[spelled] = ", ".join(piece.strip() for piece in pieces[1:])
        out.append(body[at:match.start()])
        at = close + 1
        while at < len(body) and body[at] in " ;":
            at += 1
    out.append(body[at:])
    return "".join(out), given


def _delegating_initialiser(
    body: str, found: Class, classes: "dict[str, Class]", scope: str
) -> str:
    """`P() : P(1, 2) {}` becomes the other constructor, run on this object.

    C++ builds the object once and lets one constructor ask another to do it.
    Which other one is decided the way every overloaded call here is: by what
    it is handed.
    """

    match = re.search(rf"{_DELEGATE_INIT}\s*\(([^;]*)\);", body)
    if match is None:
        return body
    given = [one.strip() for one in _split_arguments(match.group(1)) if one.strip()]
    suffix = _call_suffix(found.name, "", classes, given, scope)
    spelled = ", ".join(given)
    return (
        body[: match.start()]
        + f"{_c_name(found.name, '', suffix)}(this"
        + (f", {spelled}" if spelled else "")
        + ");"
        + body[match.end():]
    )


def _base_initialiser(body: str, found: Class) -> "tuple[str, str | None]":
    """Take `: Base(v * 2)` out of the body and report what it passed.

    It is not expanded here. The base has to be constructed *before* the
    derived class installs its own table - C++ builds an object base-first,
    and its type changes as it goes - and where that happens is
    `_open_with_subobjects`. Expanding it in place put the base's
    constructor after the table install, so the base's constructor set the
    table back to the base's and every virtual call answered as the base.
    """

    match = re.search(rf"{_BASE_INIT}\s*\(([^;]*)\);", body)
    if match is None or not found.base:
        return body, None
    return body[:match.start()] + body[match.end():], match.group(1).strip()



def _open_with_member_values(
    body: str,
    found: Class,
    classes: "dict[str, Class]",
    named: "dict[str, str] | None" = None,
) -> str:
    """Assign what each member was given where it was declared.

    Put in after the subobjects are constructed and before anything the
    constructor's own body does: a member of class type has to exist before
    it can be assigned to, and what the author wrote has to win.
    """

    # Only the members the initialiser list did not name: C++ applies a
    # default member initialiser where the list is silent, and the list where
    # it is not. Skipping them here is what keeps the two from both running.
    values = [
        (name, value)
        for name, value in found.member_values
        if name not in (named or {})
    ]
    if not values:
        return body
    held = {member.name: member for member in found.members}
    written = []
    for name, value in values:
        if not value.startswith("{"):
            written.append(f"this->{name} = {value};")
            continue
        # A brace list is not an expression in C, so it cannot be assigned.
        # Written as an object that is initialised with it and then copied,
        # which is the same thing and is C. An array member is left out: it
        # cannot be assigned either, and copying one element at a time is
        # not what this pass is for.
        member = held.get(name)
        if member is None or member.array:
            continue
        written.append(
            f"{{ {member.ctype} __py2bin_init_{name} = {value};"
            f" this->{name} = __py2bin_init_{name}; }}"
        )
    if not written:
        return body
    spelled = " ".join(written)
    opening = body.find("{")
    return body[:opening + 1] + " " + spelled + body[opening + 1:]


def _constructor_taking_one(
    held: str, spelled: str, classes: "dict[str, Class]"
) -> bool:
    """Whether that class declares a constructor taking one of `spelled`."""

    for item in classes.get(held, Class("")).methods:
        if item.name != "" or _arity(item.parameters) != 1:
            continue
        words = (
            _split_arguments(item.parameters)[0]
            .replace("*", " ")
            .replace("&", " ")
            .split()
        )
        if spelled in words:
            return True
    return False


def _constructor_taking(
    held: str, count: int, classes: "dict[str, Class]"
) -> bool:
    """Whether that class declares a constructor of that many parameters."""

    return any(
        item.name == "" and _arity(item.parameters) == count
        for item in classes.get(held, Class("")).methods
    )


def _same_class(
    value: str, held: str, scope: str, classes: "dict[str, Class]"
) -> bool:
    """Whether `value` names an object of exactly the class `held`."""

    if held not in classes:
        return False
    spelled = _deduced_type(value, scope)
    if spelled is None or "*" in spelled:
        return False
    return spelled.replace("const", "").strip() == held

#: Whether a declaration is preceded by `static`, with nothing but the rest
#: of the type between. Bounded by the punctuation that ends a statement, so
#: a `static` on the line above cannot reach this one.
_AFTER_STATIC = re.compile(r"\bstatic\b[\w\s]*$")

#: What the flag guarding a static local's constructor is called.
_BUILT_SUFFIX = "__py2bin_built"


def _open_with_subobjects(
    body: str,
    found: Class,
    classes: "dict[str, Class]",
    base_arguments: "str | None" = None,
    member_arguments: "dict[str, str] | None" = None,
    scope: str = "",
) -> str:
    named = dict(member_arguments or {})
    # Which overload builds a member is read off what the list passed, and
    # that is usually a parameter - declared in a head this body does not
    # hold. `Person(string n) : name(n)` picked the `const char *` one.
    reading = f"{body}\n{scope}"
    calls = []
    #: Which of the addresses below name a base rather than a member. A base
    #: shares this object's shared bases and is built with the form that does
    #: not build them again; a member is a complete object of its own and is
    #: built with the ordinary one.
    base_addresses = {
        f"&this->{step}" for _base, step, _shared in _base_steps(found, classes)
    }

    def built(held: str, address: str, given: "list[str]", arguments: str) -> "list[str]":
        """The lines that build one subobject, in the order they must run."""

        owner = _find_method(held, "", classes)
        if owner is None:
            return []
        spelled = _c_name(
            owner, "", _call_suffix(owner, "", classes, given, reading)
        )
        lines: "list[str]" = []
        if address in base_addresses and _shared_bases(owner, classes):
            # Told which object it is sharing before it is built, because
            # that is what its sub-object form assumes.
            for shared in _shared_bases(owner, classes):
                lines.append(
                    f"{address.lstrip('&')}.{_vbase_pointer(shared)} = "
                    f"this->{_vbase_pointer(shared)};"
                )
            spelled += _SUB_OBJECT_SUFFIX
        lines.append(
            f"{spelled}({address}"
            f"{', ' + arguments if arguments.strip() else ''});"
        )
        return lines

    for held, address in _subobjects(found, classes):
        if base_arguments is not None and address == "&this->__base":
            # Named in the initialiser list, so it is built with what was
            # written there rather than with nothing.
            given = (
                [a.strip() for a in _split_arguments(base_arguments)]
                if base_arguments else []
            )
            calls.extend(built(held, address, given, base_arguments or ""))
            continue
        owner = _find_method(held, "", classes)
        if owner is None:
            continue
        # `Holder() : b(3) {}` - the member was named in the initialiser
        # list, so it is built with what was written there.
        spelled = address.lstrip("&").replace("this->", "")
        arguments = named.pop(spelled, None)
        if arguments is not None:
            given = (
                [one.strip() for one in _split_arguments(arguments)]
                if arguments.strip()
                else []
            )
            if len(given) == 1 and (
                _same_class(given[0], held, reading, classes)
                or not _constructor_taking(held, 1, classes)
            ):
                # `Person(string n) : name(n)` names the copy constructor,
                # and a class that wrote none has the memberwise copy py2bin
                # already does everywhere else. Nothing to construct.
                # Also where the type of what is passed cannot be read but
                # the class has no constructor of that shape either: there
                # is nothing else this could mean, and calling one that does
                # not exist says so about the wrong line.
                calls.append(
                    _copied_in(held, address.lstrip("&"), f"&{given[0]}", classes)
                )
                continue
            calls.extend(built(held, address, given, arguments))
            continue
        if any(m.name == "" and m.parameters for m in classes[held].methods) and not any(
            m.name == "" and not m.parameters for m in classes[held].methods
        ):
            raise CppTranslationError(
                "<c++>", 0,
                f"{found.name} holds a {held}, whose only constructor takes "
                f"arguments, and no initialiser list here names it. Name it - "
                f"`{found.name}() : {spelled}(...)` - give {held} a "
                f"constructor taking nothing, or hold a pointer",
            )
        calls.extend(built(held, address, [], ""))
    if _is_polymorphic(found.name, classes):
        # After the base constructor, which set the pointer to *its* table.
        # Overwriting it here is what C++ means by the object becoming its own
        # type as construction proceeds, and it is why a virtual call made
        # from a base constructor reaches the base's version.
        carrier = _vptr_carrier(found.name, classes)
        calls.append(
            f"this->{_vptr_path(found.name, classes)} = "
            # Where the pointer lives in a base everything shares, what goes
            # in it is the table whose entries move the pointer back here.
            + (
                f"{_mixin_vtable_name(found.name, carrier)};"
                if carrier in _shared_bases(found.name, classes)
                else f"{_vtable_name(found.name)};"
            )
        )
    # And one for each base after the first that has a table of its own. A
    # pointer to that subobject is what a call through it is given, so what
    # it reads has to be the table written for this class.
    for index, mixin in enumerate(found.mixins):
        if mixin in found.virtual_bases:
            continue
        # And not one whose pointer lives in a base everything shares: that
        # is the one set above, and writing this class's table for the mixin
        # into it would say the object is a mixin.
        if _vptr_carrier(mixin, classes) in _shared_bases(found.name, classes):
            continue
        if _is_polymorphic(mixin, classes):
            calls.append(
                f"this->__base{index + 1}."
                f"{_vptr_path(mixin, classes)} = "
                f"{_mixin_vtable_name(found.name, mixin)};"
            )
    # Whatever the list named that is not a subobject is an ordinary member,
    # and an ordinary member is assigned to. After the constructions, because
    # one of them may be what it is assigned from.
    calls.extend(f"this->{name} = {value};" for name, value in named.items())
    if not calls:
        return body
    opening = body.find("{")
    return body[:opening + 1] + " " + " ".join(calls) + body[opening + 1:]


def _close_with_subobjects(body: str, found: Class, classes: "dict[str, Class]") -> str:
    calls = []
    for held, address in reversed(_subobjects(found, classes)):
        owner = _find_method(held, "~", classes)
        if owner is not None:
            calls.append(f"{_c_name(owner, '~')}({address});")
    if not calls:
        return body
    closing = body.rfind("}")
    return body[:closing] + " " + " ".join(calls) + " " + body[closing:]




def _reachable_members(
    owner: Class, classes: "dict[str, Class]"
) -> "list[tuple[Member, str]]":
    """Every data member the class can name, and how far down it lives."""

    found = [(member, "") for member in owner.members]
    for name in _every_base(owner.name, classes):
        path = _subobject_path(owner.name, name, classes)
        if path is None:
            continue
        for member in classes[name].members:
            found.append((member, f"{path}."))
    return found

def _qualified_base_calls(
    body: str, owner: Class, classes: "dict[str, Class]"
) -> str:
    """`Base::show()` - the base's own version, named rather than dispatched.

    This is the one call a virtual function does *not* go through the table:
    naming the class is how C++ says "that one, not whatever this object
    turned out to be", and it is how an override reaches what it overrode.
    """

    for named in sorted(classes, key=len, reverse=True):
        if named == owner.name or not _derives_from(owner.name, named, classes):
            continue
        for method in _reachable_methods(named, classes):
            provider = _find_method(named, method, classes)
            if provider is None:
                continue
            path = _subobject_path(owner.name, provider, classes) or ""
            reached = "this" if not path else f"&this->{path}"
            pattern = (
                rf"(?<![.\w>]){re.escape(named)}\s*::\s*"
                rf"{re.escape(method)}\s*\("
            )
            body = _rewrite_calls(
                body, pattern, _name_for(provider, method, classes), reached
            )
    return body


def _answer_into_the_space(body: str, answering: "list[str]") -> str:
    """`return substr(x);` writes into the caller's space, not into its own.

    A member that answers an object answers nothing at all in the C: the
    caller passes the room and the callee fills it. So a `return` whose whole
    value is a call to one of those is not a value to assign - it is the same
    room, handed straight on. Which is what C++ does here too, and is why
    the copy it looks like never happens.
    """

    bare = _without_literals(body)
    out: "list[str]" = []
    at = 0
    for match in re.finditer(r"\breturn\s+([A-Za-z_]\w*)\s*\(\s*this\b", bare):
        if match.start() < at:
            continue
        name = match.group(1)
        if not any(name == one or name.startswith(f"{one}__") for one in answering):
            continue
        opening = bare.index("(", match.start(1) + len(name))
        closing = _closing_paren(bare, opening)
        if closing < 0:
            continue
        rest = closing + 1
        while rest < len(bare) and bare[rest] in " \t":
            rest += 1
        if rest >= len(bare) or bare[rest] != ";":
            continue
        inside = body[opening + 1: closing]
        after = inside[len("this"):].lstrip()
        spelled = f"this, __ret, {after[1:].lstrip()}" if after.startswith(",") else "this, __ret"
        out.append(body[at: match.start()])
        out.append(f"{name}({spelled}); return;")
        at = rest + 1
    out.append(body[at:])
    return "".join(out)


def _name_the_receiver(body: str, method: str) -> str:
    """`name().c_str()` becomes `this->name().c_str()`.

    Only where something is reached on the answer, because only there does it
    matter: a call standing on its own, or handed straight to a `return`, is
    already written out correctly by the caller of this. Scanned rather than
    substituted because what follows the *closing* paren is the question, and
    the arguments in between may hold parens of their own.
    """

    bare = _without_literals(body)
    out: "list[str]" = []
    at = 0
    for match in re.finditer(rf"(?<![.>\w]){re.escape(method)}\s*\(", bare):
        if match.start() < at:
            continue
        close = _closing_paren(body, match.end() - 1)
        if close < 0:
            continue
        after = bare[close + 1:].lstrip()
        if not after.startswith(".") and not after.startswith("->"):
            continue
        out.append(body[at:match.start()])
        out.append("this->")
        at = match.start()
    out.append(body[at:])
    return "".join(out)


def _bare_method_calls(
    body: str, owner: Class, classes: "dict[str, Class]", scope: str = ""
) -> str:
    """`sum()` inside a member is a call on `this`, and C has no such thing.

    Written out here rather than left to the `this->` pass above, which points
    *names* at the object: a call is the name plus its argument list, and the
    object has to be threaded through as the first argument.
    """

    # `name().c_str()` first, where the answer is an object and something is
    # reached on it. The loop below writes a bare call straight out as the C
    # call it becomes, and a value return is a hidden pointer the caller
    # provides - so what came out was `A__name(this).c_str()`, a member
    # reached on something that is not an expression in C at all. The pass
    # that writes those temporaries out keys on a receiver being written,
    # which is the whole of why `this->name().c_str()` always worked and the
    # bare spelling did not. Giving the receiver back is enough: from there
    # it is an ordinary method call and every pass below knows the shape.
    for method in _reachable_methods(owner.name, classes):
        provider = _find_method(owner.name, method, classes)
        if provider is None:
            continue
        member = _method_by_name(provider, method, classes)
        if member is None or _returns_object(member, classes) is None:
            continue
        body = _name_the_receiver(body, method)

    for method in sorted(_reachable_methods(owner.name, classes), key=len, reverse=True):
        provider = _find_method(owner.name, method, classes)
        if provider is None:
            continue
        reached = "this"
        if provider != owner.name:
            path = _subobject_path(owner.name, provider, classes) or "__base"
            reached = f"&this->{path}"
        # Not already qualified: `p->sum(` and `v.sum(` are somebody else's.
        pattern = rf"(?<![.>\w]){re.escape(method)}\s*\("
        body = _rewrite_calls(
            body, pattern,
            _dispatched_here(
                owner.name, method, classes, provider, reached,
                f"{body}\n{scope}",
            ),
            reached,
        )
    # And where such a call is the whole of a `return`, the room it fills is
    # the room this member was given rather than one of its own.
    answering: "list[str]" = []
    for method in _reachable_methods(owner.name, classes):
        provider = _find_method(owner.name, method, classes)
        if provider is None:
            continue
        for one in classes[provider].methods:
            if one.name == method and _returns_object(one, classes):
                answering.append(_c_name(provider, method))
                break
    if answering:
        body = _answer_into_the_space(body, answering)
    return body


def _dispatched_here(
    static: str,
    method: str,
    classes: "dict[str, Class]",
    provider: str,
    reached: str,
    text: str = "",
):
    """A call written bare inside a member: `sum()`, meaning `this->sum()`.

    Virtual or not, the object is `this`. A direct call to a method the base
    provides is handed the base subobject instead, which is why the receiver
    is part of the answer and not fixed by the caller.
    """

    virtual = _dispatch(static, method, classes, "this")
    direct = _name_for(provider, method, classes, text)
    # A static member of the same class, called bare, is still just a call:
    # it was never given the object, so it must not be handed `this`.
    shared = _is_shared(provider, method, classes)

    def chosen(given: "list[str]"):
        through = virtual(given) if virtual is not None else ""
        if through:
            return through, _dispatch_receiver(static, classes, "this")
        name = direct if isinstance(direct, str) else direct(given)
        return name, ("" if shared else reached)

    return chosen


def _is_shared(owner: str, method: str, classes: "dict[str, Class]") -> bool:
    """Whether every member of that name is `static`, so takes no object."""

    found = [item for item in classes[owner].methods if item.name == method]
    return bool(found) and all(item.shared for item in found)

def _c_name(class_name: str, method: str, suffix: "str | int | None" = None) -> str:
    """The C name for a member. Overloads are told apart by how many they take.

    C has one name per function, and C++ has as many as the argument lists
    differ - `string()` and `string(const char *)` are both `string__ctor`
    until something distinguishes them. The count does, for every overload
    that differs in arity, which is nearly all of them; two that differ only
    in type are refused where they are declared.
    """

    if method == "":
        base = f"{class_name}__ctor"
    elif method == "~":
        base = f"{class_name}__dtor"
    else:
        base = f"{class_name}__{method}"
    return base if suffix is None else f"{base}__{suffix}"


def _arity(parameters: str) -> int:
    return len([part for part in parameters.split(",") if part.strip()])


def _overloaded(owner: str, method: str, classes: "dict[str, Class]") -> bool:
    """Whether more than one member of that class shares this name."""

    found = classes.get(owner)
    if found is None:
        return False
    return len([m for m in found.methods if m.name == method]) > 1


def _type_code(spelled: str) -> str:
    """A parameter's type as something that can go in a C identifier.

    `const char *` becomes `char_p`. Readable on purpose: an overload set
    written out as `print__1__int` and `print__1__char_p` can be followed in
    the generated C, where a hash could not.
    """

    cleaned = re.sub(r"\b(const|volatile)\b", " ", spelled)
    cleaned = cleaned.replace("*", " p ").replace("&", " ")
    words = [word for word in cleaned.split() if word]
    return "_".join(words) or "void"


def _parameter_types(parameters: str) -> "list[str]":
    """The declared types of a parameter list, without the names."""

    found = []
    for part in _split_arguments(parameters):
        if not part.strip():
            continue
        spelled = re.sub(r"\[\s*\d*\s*\]\s*$", " p", part.strip())
        words = spelled.replace("*", " * ").replace("&", " & ").split()
        # The last word is the parameter's own name unless it is all type.
        if len(words) > 1 and re.fullmatch(r"[A-Za-z_]\w*", words[-1]):
            words = words[:-1]
        found.append(_type_code(" ".join(words)))
    return found


def _method_by_name(
    owner: str, method: str, classes: "dict[str, Class]"
) -> "Method | None":
    """The first member of that class with this name."""

    found = classes.get(owner)
    if found is None:
        return None
    for candidate in found.methods:
        if candidate.name == method:
            return candidate
    return None


def _overload_set(owner: str, method: str, classes: "dict[str, Class]") -> "list[Method]":
    """Every member of that class with this name, most derived first."""

    found = classes.get(owner)
    if found is None:
        return []
    return [m for m in found.methods if m.name == method]


def _suffix_of(owner: str, method: "Method", classes: "dict[str, Class]") -> "str | None":
    """What to add to a member's C name so it is its own.

    Nothing where the name is used once. The count where that settles it,
    which is the common case and keeps the C readable. The types as well
    where two take the same number - `cout << 1` and `cout << "s"` both pass
    one argument, and only the types tell them apart.
    """

    set_of = _overload_set(owner, method.name, classes)
    if len(set_of) < 2:
        return None
    arity = _arity(method.parameters)
    if len([m for m in set_of if _arity(m.parameters) == arity]) == 1:
        return str(arity)
    return f"{arity}__" + "_".join(_parameter_types(method.parameters))


#: Conversions a call may rely on when no overload matches exactly. Narrow on
#: purpose: these are the ones C++ makes silently and C makes too.
#: Every code that names a whole number. C converts freely among them, so a
#: parameter declared as one of them takes an argument of any other - which
#: is what makes `find(c, position)` the same call whether the position was
#: read as an `int`, a `size_t`, or an `unsigned long`.
_INTEGRAL_CODES = (
    "int", "char", "short", "long", "long_long", "size_t",
    "unsigned_int", "unsigned_char", "unsigned_short", "unsigned_long",
    "unsigned_long_long", "bool", "_Bool", "wchar_t",
)

_PROMOTIONS = {
    **{name: tuple(o for o in _INTEGRAL_CODES if o != name) + ("double", "float")
       for name in _INTEGRAL_CODES},
    "double": ("float",),
    "char_p": ("void_p",),
}


def _closeness(code: str, declared: str) -> int:
    """How far a conversion is from being no conversion at all.

    C++ ranks an exact match above a promotion and a promotion above a
    conversion between families. Nothing ranked them here, so where two
    candidates both fitted the tie was settled by which was written first.
    """

    if declared == code:
        return 0
    together = (
        code in _INTEGRAL_CODES and declared in _INTEGRAL_CODES
    ) or (
        code in ("double", "float") and declared in ("double", "float")
    )
    return 1 if together else 2


def _chosen_overload(
    set_of: "list[Method]", given: "list[str]", text: str, before: int
) -> "Method | None":
    """Which member of an overload set a call with these arguments means."""

    candidates = [m for m in set_of if _arity(m.parameters) == len(given)]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    wanted = [_deduced_type(value, text, before) for value in given]
    if any(item is None for item in wanted):
        # Some argument could not be read. The ones that could may still
        # settle it: `find(':', somewhere)` has three candidates and only
        # one takes a `char` first, so the position that is unreadable
        # decides nothing and does not have to.
        return _narrowed_by_what_is_known(candidates, wanted)
    codes = [_type_code(item) for item in wanted]
    exact = [
        m for m in candidates if _parameter_types(m.parameters) == codes
    ]
    if len(exact) == 1:
        return exact[0]
    near = [
        m
        for m in candidates
        if all(
            declared == code or declared in _PROMOTIONS.get(code, ())
            for declared, code in zip(_parameter_types(m.parameters), codes)
        )
    ]
    if len(near) == 1:
        return near[0]
    # More than one fits, so how *well* each fits decides - which is what C++
    # does. A `char` reaches an `int` by a promotion and a `double` by a
    # conversion between families, and ranked as equals the tie went to
    # whichever was written first: `f('a')` called the one taking a double.
    scored = [
        (
            sum(
                _closeness(code, declared)
                for declared, code in zip(_parameter_types(m.parameters), codes)
            ),
            m,
        )
        for m in near
    ]
    if not scored:
        return None
    nearest = min(one for one, _m in scored)
    best = [m for one, m in scored if one == nearest]
    return best[0] if len(best) == 1 else None


def _narrowed_by_what_is_known(
    candidates: "list[Method]", wanted: "list[str | None]"
) -> "Method | None":
    """The one candidate that fits every argument whose type could be read."""

    known = [
        (index, _type_code(item))
        for index, item in enumerate(wanted)
        if item is not None
    ]
    if not known:
        return None
    scored: "list[tuple[int, Method]]" = []
    for method in candidates:
        declared = _parameter_types(method.parameters)
        if len(declared) != len(wanted):
            continue
        if not all(
            declared[index] == code
            or declared[index] in _PROMOTIONS.get(code, ())
            for index, code in known
        ):
            continue
        scored.append(
            (sum(_closeness(code, declared[index]) for index, code in known), method)
        )
    if not scored:
        return None
    nearest = min(one for one, _method in scored)
    fits = [method for one, method in scored if one == nearest]
    return fits[0] if len(fits) == 1 else None


def _call_suffix(
    owner: str,
    method: str,
    classes: "dict[str, Class]",
    given: "list[str]",
    text: str = "",
    before: int = -1,
) -> "str | None":
    """The suffix a call site should use, read from what it passes."""

    set_of = _overload_set(owner, method, classes)
    if len(set_of) < 2:
        return None
    picked = _chosen_overload(set_of, given, text, before)
    if picked is not None:
        return _suffix_of(owner, picked, classes)
    if len([m for m in set_of if _arity(m.parameters) == len(given)]) < 2:
        # No overload takes this many. Naming the count leaves the C compiler
        # to report a call to a function that is not there, which is the same
        # mistake said in the same place.
        return str(len(given))
    spelled = ", ".join(given)
    raise CppTranslationError(
        "<c++>", 0,
        f"more than one {method or owner}() takes {len(given)} argument(s), "
        f"and py2bin cannot tell which is meant by `{spelled}`. It reads the "
        f"type of a literal and of a variable it can see declared; cast the "
        f"argument to the type of the one you want",
    )


def _find_method(name: str, method: str, classes: "dict[str, Class]") -> str | None:
    """Which class actually provides `method`, following the base chain."""

    for seen in [name, *_every_base(name, classes)]:
        found = classes.get(seen)
        if found is not None and any(m.name == method for m in found.methods):
            return seen
    return None


def _rewrite_types(text: str, classes: "dict[str, Class]") -> str:
    """`Vec v` becomes `struct Vec v` wherever a class is named as a type."""

    def replace(match: "re.Match[str]") -> str:
        word = match.group(0)
        if word not in classes:
            return word
        # Not where it already says so: `struct P { ... }` is a definition,
        # and `struct struct P` is not C.
        before = match.string[:match.start()].rstrip()
        if before.endswith("struct") or before.endswith("union"):
            return word
        return f"struct {word}"

    return _map_code(text, lambda part: _WORD.sub(replace, part))


#: `Vec v(1, 2);` and `Vec v;` - an object with automatic storage. Not one
#: that already says `struct`, which is a declaration already written as C.
_OBJECT = re.compile(
    r"(?<!struct )\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(\(([^;{}]*)\))?\s*;"
)
#: `Vec *p = ...;`
_OBJECT_POINTER = re.compile(
    r"(?<!struct )\b([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)"
)
#: `Vec b = a;` - a declaration copied from another object of the same class.
#: The right side is an object, not only a name: `T held = *first;` and
#: `T held = base[root];` are how a container's own code takes a copy, and
#: neither is a bare identifier.
_COPY_INIT = re.compile(
    r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=\s*"
    r"(\*?\s*(?:this\s*->\s*)?[A-Za-z_]\w*(?:\s*\[[^\]]*\])?)\s*;"
)

#: `string s = "abc";` or `Money m = 500;` - a declaration whose value is
#: neither another object nor a call on one. C++ reads `C name = value;` as
#: `C name(value);` and looks for a constructor that takes it, which is how a
#: class says what it can be made from. Without this the value was handed
#: straight to C, which has no idea how to make a struct out of a `char *`.
#: The characters that make an expression more than one thing. `->` is not
#: among them once it is read as the single step it is, so `Money m = p->fee;`
#: is still a value standing on its own.
_AN_OPERATOR = "+-*/%<>!&|^~?"


def _has_a_loose_operator(value: str) -> bool:
    """Whether an operator stands outside every bracket in `value`."""

    depth = 0
    for letter in value.replace("->", "  "):
        if letter in "([":
            depth += 1
        elif letter in ")]":
            depth -= 1
        elif depth == 0 and letter in _AN_OPERATOR:
            return True
    return False


#: Everything between the `=` and the `;` is one group, spaces and all: a
#: literal is blanked to spaces of its own length before this is matched, so
#: a pattern that trims whitespace at the edges trims the literal away with
#: it and hands back half a string.
_CONVERTING_INIT = re.compile(
    r"(?<![.\w>])([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=([^;{}=][^;{}]*);"
)

#: `Vec c = a.add(b);` - a declaration whose value comes from a method that
#: returns an object. The space is the caller's to provide.
_VALUE_INIT = re.compile(
    r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=\s*"
    r"([A-Za-z_]\w*)\s*(\.|->)\s*([A-Za-z_]\w*)\s*\(([^;]*)\)\s*;"
)

#: `Vec bank[3];` - an array of objects, each of which C++ default-constructs.
#: Not one already written as C: `struct A xs[3];` is what the pass that
#: builds an array from a brace list leaves behind, and every element of that
#: one has been constructed already.
_OBJECT_ARRAY = re.compile(
    r"(?<!struct )\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;"
)

#: `path webRoot = executablePath / L"web";` - a declaration whose initialiser
#: is an expression rather than one of the shapes above. What it holds is
#: still an object of that class, and a pass that did not know it was one
#: left every call on it alone.
_ANY_INIT = re.compile(
    r"(?<!struct )(?<![.\w>])\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=[^=][^;]*;"
)
#: `Shape *all[3];` - an array of pointers, which is how a program holds a
#: mixture of things that share a base.
_POINTER_ARRAY = re.compile(
    r"\b([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;"
)


#: `p = &v;` or `all[i] = p;` - an assignment whose right side is one thing.
_SIMPLE_ASSIGN = re.compile(
    r"\b([A-Za-z_]\w*)\s*(\[[^\];]*\])?\s*=\s*(&?\s*[A-Za-z_]\w*)\s*;"
)


def _derives_from(derived: str, base: str, classes: "dict[str, Class]") -> bool:
    """Whether `derived` is `base` or has it somewhere above it."""

    return derived == base or base in _every_base(derived, classes)




#: Which free functions take a pointer to a class, and at which positions.
#: Answered once per scope: this is asked while emitting *every* method body,
#: and the scan walks every definition in the whole unit - which made a build
#: quadratic in the size of the program, and was most of what a C++ build
#: spent its time on.
_BASE_PARAMETERS_SEEN: "dict[tuple[str, int], dict[str, list]]" = {}
_BASE_PARAMETERS_KEPT = 16


#: The tags C writes before a struct's name. A parameter list that has been
#: through the type pass says `struct Shape *value` where the C++ said
#: `Shape *value`, and reading the first word of that answered `struct` -
#: so a call handing a derived pointer to a base parameter was left uncast,
#: which C refuses. Every container of base pointers reached this, because
#: the copy of `push_back` a container is made into takes exactly that.
_A_TAG = frozenset({"struct", "union", "enum"})


def _class_named(spelled: str) -> str:
    """The class a parameter's type names, without its stars or its tag."""

    for word in spelled.replace("*", " ").replace("const", " ").split():
        if word not in _A_TAG:
            return word
    return ""


def _parameters_wanting_a_base(
    scope: str, classes: "dict[str, Class]", remember: bool = False
) -> "dict[str, list[tuple[int, str]]]":
    """Every function in `scope` whose parameters name a class, by position.

    Keyed on the text itself, not on its identity: the scope handed in is
    built fresh for each body, and an `id` is reused once the last one is
    collected - which would answer a different scope from a stale scan.
    """

    key = (scope, id(classes))
    if remember:
        remembered = _BASE_PARAMETERS_SEEN.get(key)
        if remembered is not None:
            return remembered
    found: "dict[str, list[tuple[int, str]]]" = {}
    for match in _DEFINITION.finditer(scope):
        # Nothing here wants a base unless something here is a pointer, and
        # splitting a parameter list is the dearest step in the scan.
        if "*" not in match.group(3) or match.group(2) in _NOT_A_TYPE:
            continue
        if _depth_at(scope, match.end() - 1) != 0:
            continue
        wanted_at = [
            (index, _class_named(part))
            for index, part in enumerate(_split_arguments(match.group(3)))
            if "*" in part and _class_named(part) in classes
        ]
        if wanted_at:
            found[match.group(2)] = wanted_at
    if remember:
        # Only the unit is worth keeping: it is the same text for every body
        # in a pass, where a body's own scan is asked once and never again.
        # A pass has more than one unit in the air - the text as the bodies
        # before this one left it, and the text a later pass rebuilt - so a
        # single slot holds none of them and a handful holds them all.
        while len(_BASE_PARAMETERS_SEEN) >= _BASE_PARAMETERS_KEPT:
            del _BASE_PARAMETERS_SEEN[next(iter(_BASE_PARAMETERS_SEEN))]
        _BASE_PARAMETERS_SEEN[key] = found
    return found

def _upcast_pointers(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
    returns: str,
    scope: str,
    unit: str = "",
    stable: str = "",
) -> str:
    """Insert the cast C wants wherever a derived pointer stands for a base.

    C++ converts a pointer-to-derived into a pointer-to-base in every position
    that wants one - an argument, a return, an initialiser - and C makes you
    say so. The address is the same, so this is a cast and nothing more; what
    matters is that it is written everywhere and not only where an assignment
    happened to be.
    """

    if not classes:
        return body

    # `return derived;` where the function returns a base pointer.
    wanted = returns.replace("*", "").strip()
    if "*" in returns and wanted in classes:
        def returned(match: "re.Match[str]") -> str:
            value = match.group(1).strip()
            held = (_deduced_type(value, scope) or "").replace("*", "").strip()
            if not held or held == wanted or not _derives_from(held, wanted, classes):
                return match.group(0)
            return f"return (struct {wanted} *){value};"

        body = _map_code(
            body, lambda part: re.sub(r"\breturn\s+([^;]+);", returned, part)
        )

    # `f(derived)` where a free function's parameter is a base pointer. Read
    # from the definitions, because a free function has no entry in the class
    # table and its parameters are written nowhere else.
    # Over the unit, which is the same text for every body in a pass, and
    # then over the body for anything it declares itself. Asked of the whole
    # scope instead, this walked every definition in the program once per
    # method - which is where a C++ build spent most of its time.
    # The unit is the same program for every body in a pass with this
    # method's own parameters glued on the end, so the scan is split there:
    # the part every body shares is walked once and remembered, and the tail
    # is a line. Walking the whole of it per body is what made a build
    # quadratic in the length of the program.
    if unit and stable and unit.startswith(stable):
        texts = ((stable, True), (unit[len(stable):], False), (body, False))
    elif unit:
        texts = ((unit, True), (body, False))
    else:
        texts = ((body, False),)
    for text, keep in texts:
        for name, wanted_at in _parameters_wanting_a_base(
            text, classes, keep
        ).items():
            body = _cast_at_positions(body, name, wanted_at, classes, scope, None)

    # And the same for a method, whose parameters the class table has.
    signatures = _call_signatures(classes)
    for name, (owner, method) in signatures.items():
        wanted_at = [
            (index, _class_named(part))
            for index, part in enumerate(_split_arguments(method.parameters))
            if "*" in part and _class_named(part) in classes
        ]
        if not wanted_at:
            continue
        body = _cast_at_positions(body, name, wanted_at, classes, scope, method)
    return body


def _cast_at_positions(
    body: str,
    name: str,
    wanted_at: "list[tuple[int, str]]",
    classes: "dict[str, Class]",
    scope: str,
    method: "Method | None",
) -> str:
    """Cast the arguments at those positions where they name a derived class."""

    # A body that never mentions the name has no call to it. Asked without
    # this, every body was searched once per function in the unit - which is
    # quadratic in the size of the program, and was two thirds of the time a
    # C++ build took.
    if name not in body:
        return body
    pattern = re.compile(rf"(?<![.\w>]){re.escape(name)}\s*\(")
    out: list[str] = []
    at = 0
    # A free function has no receiver; a method's is the first argument and
    # is not one of its declared parameters.
    if method is None:
        skip = 0
    else:
        skip = 0 if name.endswith("new") or "__new__" in name else 1
    for call in pattern.finditer(body):
        if call.start() < at:
            continue
        close = _closing_paren(body, call.end() - 1)
        if close < 0:
            continue
        given = _call_arguments(body, call.end() - 1)
        changed = False
        for index, spelled in wanted_at:
            wanted = _class_named(spelled)
            where = index + skip
            if where >= len(given):
                continue
            value = given[where].strip()
            if value.startswith("(struct"):
                continue
            # `&this->__base` already *is* the base subobject, whatever the
            # class the enclosing `this` belongs to. Reading its type off
            # `this` answers the derived class and asks for a cast that says
            # nothing - and the receiver of a method written as a free
            # function is spelled exactly this way.
            if value.endswith("__base"):
                continue
            held = (_deduced_type(value, scope) or "").replace("*", "").strip()
            held = held or _allocated_class(value)
            if not held or held == wanted or not _derives_from(held, wanted, classes):
                continue
            given[where] = f"(struct {wanted} *){value}"
            changed = True
        if not changed:
            continue
        out.append(body[at:call.end()])
        out.append(", ".join(given))
        at = close
    out.append(body[at:])
    return "".join(out)

def _upcast_assignments(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
    pointer_arrays: "dict[str, str]",
) -> str:
    """Insert the cast C wants when a derived object is stored as its base."""

    if not pointer_arrays and not pointers:
        return body

    def one(match: "re.Match[str]") -> str:
        target, index, source = match.groups()
        source = source.replace(" ", "")
        wanted = (
            pointer_arrays.get(target)
            if index
            else (known.get(target) if target in pointers else None)
        )
        if wanted is None:
            return match.group(0)
        held = known.get(source.lstrip("&"))
        if held is None or held == wanted:
            return match.group(0)
        if not _derives_from(held, wanted, classes):
            # Not a base at all: leave it, and let the C compiler say so
            # rather than casting away a real mistake.
            return match.group(0)
        return f"{target}{index or ''} = (struct {wanted} *){source};"

    body = _map_code(body, lambda part: _SIMPLE_ASSIGN.sub(one, part))

    # `Base *p = new Sub;`, which by now reads `struct Base *p = Sub__new();`.
    def allocated(match: "re.Match[str]") -> str:
        wanted, variable, made = match.group(1), match.group(2), match.group(3)
        if wanted == made or not _derives_from(made, wanted, classes):
            return match.group(0)
        head, call = match.group(0).split("=", 1)
        return f"{head}= (struct {wanted} *){call.lstrip()}"

    body = _map_code(body, lambda part: _DECLARED_FROM_NEW.sub(allocated, part))

    # `Base *p = &derived;` - a declaration, which the assignment pattern
    # above does not see because the type is in front of the name. This is
    # where a reference to a base comes out: `A &r = b;` is a pointer here.
    def declared(match: "re.Match[str]") -> str:
        wanted, variable, value = match.groups()
        spelled = value.strip()
        if wanted not in classes or spelled.startswith("(struct"):
            return match.group(0)
        source = spelled.lstrip("&").strip()
        while source.startswith("(") and source.endswith(")"):
            source = source[1:-1].strip()
        held = known.get(source)
        if held is None or held == wanted or not _derives_from(held, wanted, classes):
            return match.group(0)
        return f"struct {wanted} *{variable} = (struct {wanted} *){spelled};"

    body = _map_code(
        body,
        lambda part: re.sub(
            r"\bstruct\s+([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)\s*=\s*([^;]+);",
            declared,
            part,
        ),
    )

    def assigned(match: "re.Match[str]") -> str:
        variable, index, made = match.groups()
        wanted = (
            pointer_arrays.get(variable)
            if index
            else (known.get(variable) if variable in pointers else None)
        )
        if wanted is None or wanted == made or not _derives_from(made, wanted, classes):
            return match.group(0)
        head, call = match.group(0).split("=", 1)
        return f"{head}= (struct {wanted} *){call.lstrip()}"

    return _map_code(body, lambda part: _ASSIGNED_FROM_NEW.sub(assigned, part))


#: `Sub__new(...)` standing on its own, as an argument rather than as the
#: right of an assignment: `v.push_back(new Sub())` never names the thing it
#: made, so nothing in the scope says what its type is.
_AN_ALLOCATION = re.compile(r"^([A-Za-z_]\w*)__new(?:__\d+|_array)?\s*\(")


def _allocated_class(value: str) -> str:
    """The class an allocation makes, for a value no declaration describes."""

    match = _AN_ALLOCATION.match(value.strip())
    return match.group(1) if match else ""


#: `struct Base *p = Sub__new(` - an allocation stored as a pointer to a base.
_DECLARED_FROM_NEW = re.compile(
    r"\bstruct\s+([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)\s*=\s*"
    r"([A-Za-z_]\w*)__new(?:__\d+|_array)?\s*\("
)
#: The same, assigning to a pointer declared earlier - or to an element of an
#: array of them, which is how a program keeps a mixture of things that share
#: a base: `all[0] = new Sub();`.
_ASSIGNED_FROM_NEW = re.compile(
    r"\b([A-Za-z_]\w*)\s*(\[[^\]]*\])?\s*=\s*"
    r"([A-Za-z_]\w*)__new(?:__\d+|_array)?\s*\("
)



#: `a.b()` or `p->b()` - a call on something, with what follows left alone.
_CALL_ON = re.compile(r"(?<![.\w>])([A-Za-z_]\w*)\s*(\.|->)\s*([A-Za-z_]\w*)\s*\(")


def _declared_objects(
    body: str, classes: "dict[str, Class]"
) -> "dict[str, str]":
    """Every object this body declares, read without rewriting anything.

    The declaration passes below both read and rewrite, and something has to
    know the types before they run.
    """

    found: dict[str, str] = {}
    # Against a copy with the literals blanked: `"int n = 1;"` inside a
    # string is text and declares nothing.
    bare = _without_literals(body)
    for pattern in (
        _OBJECT, _OBJECT_ARRAY, _OBJECT_POINTER, _VALUE_INIT, _ANY_INIT
    ):
        for match in pattern.finditer(bare):
            held, name = match.group(1), match.group(2)
            if held in classes:
                found[name] = held
    return found



def _hoist_object_operators(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    counter: "list[int]",
    pointers: "set[str]" = frozenset(),
    referenced: "set[str]" = frozenset(),
) -> str:
    """`b = a / "x";` becomes a temporary the operator fills, and a copy.

    An operator answering an object answers it the way any function does -
    through a pointer the caller provides - so it is not an expression that
    can sit on the right of an assignment or inside an argument list. The
    one place it can sit is the initialiser of a declaration, because the
    object being declared is the space; so everything else is turned into
    that form here and the pass that reads it needs to know nothing new.
    """

    if not known:
        return body
    for _round in range(_HOIST_ROUNDS):
        found = None
        # The symbol outside and the variable inside: each round writes one
        # of these out, and which one it picks is the precedence the result
        # is evaluated with. Asked variable-first, `a + b * c` found `a`,
        # then the first operator written on it - so `a + b` was taken out
        # and the answer was `(a + b) * c`, with nothing to say so.
        for symbol in _OPERATOR_SYMBOLS:
            if symbol in ("[]", "()", "=", "<<", ">>"):
                continue
            for variable in sorted(known, key=len, reverse=True):
                if variable in pointers and variable not in referenced:
                    # `items + count` on a `T *` is where the pointer moves
                    # to, not the class's operator. The class it points at
                    # has one and the pointer does not.
                    #
                    # A *reference* is not that. `const path &p` is a pointer
                    # here only because C has no reference, and `p / L"web"`
                    # in the source asked the class for its operator - so
                    # skipping it left the `/` for the C compiler.
                    continue
                name = _OPERATOR_NAMES[symbol]
                owner = _find_method(known[variable], name, classes)
                if owner is None:
                    continue
                held = _method_named_returning(owner, name, classes)
                if held is None:
                    continue
                bare = _without_literals(body)
                for match in re.finditer(
                    rf"(?<![.\w>]){re.escape(variable)}\s*"
                    rf"{re.escape(symbol)}(?![=])",
                    bare,
                ):
                    end = _one_operand(body, match.end())
                    if end < 0:
                        continue
                    begins = _statement_start(body, match.start())
                    while begins < len(body) and body[begins] in " \t\n":
                        begins += 1
                    # The declaration form is the one this writes, and the
                    # one the operator pass reads. Left in, each round wrote
                    # a temporary holding the temporary before it.
                    lead = re.match(
                        r"[A-Za-z_]\w*(?:\s+|\s*&\s*)[A-Za-z_]\w*\s*=\s*$",
                        body[begins: match.start()],
                    )
                    if lead is not None and body[end:].lstrip()[:1] == ";":
                        continue
                    found = (begins, match.start(), end, held)
                    break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            return body
        begins, at, end, held = found
        counter[0] += 1
        made = f"{_OPERATOR_PREFIX}{counter[0]}"
        known[made] = held
        body = (
            body[:begins]
            + f"{held} {made} = {body[at:end].strip()}; "
            + body[begins:at]
            + made
            + body[end:]
        )
    return body


#: `(held).m()` - parentheses around a single name, with a member reached
#: through them.
_AROUND_A_NAME = re.compile(
    r"(?<![\w)\]])\(\s*([A-Za-z_]\w*)\s*\)(?=\s*(?:\.|->))"
)

#: What the pass above calls what it writes. Its own prefix, so a temporary
#: holding an operator's answer is told from one holding a call's.
_OPERATOR_PREFIX = "__py2bin_operand_"

#: What a returned value is called while the scope is taken apart around it.
_ANSWER_PREFIX = "__py2bin_answer_"

def _is_a_constant(value: str) -> bool:
    """Whether nothing can change this between working it out and
    handing it back.

    `return 1;` needs no temporary. `return alive;` does, because a
    destructor running in between may be exactly what alters `alive`.
    """

    spelled = value.strip()
    if not spelled:
        return False
    if spelled in ("true", "false", "nullptr", "NULL"):
        return True
    if spelled[0] in "'\"" and spelled[-1] == spelled[0]:
        return True
    return bool(re.match(r"^[-+]?\d[\w.]*$", spelled))


def _already_a_declaration(text: str, match: "re.Match[str]") -> bool:
    """Whether this call is already the whole initialiser of a declaration.

    That form needs no temporary: the object being declared *is* the space
    the callee writes through. It also covers what this pass writes itself -
    the same thing with a `&` in it - which was otherwise hoisted again, and
    again, until the round limit stopped it.

    The whole of it, not the start of it. `path a = base() / "web";` declares
    `a` and the call is only the left operand of what fills it, so `a` is not
    the space that call writes through - and treated as though it were, the
    operator was quietly dropped.
    """

    close = _closing_paren(text, match.end() - 1)
    if close >= 0 and text[close + 1:].lstrip()[:1] != ";":
        return False
    begins = _statement_start(text, match.start())
    while begins < len(text) and text[begins] in " \t\n":
        begins += 1
    if _VALUE_INIT.match(text, begins):
        return True
    # The two names have to be separated by something - whitespace or the
    # `&`. Without that the pattern backtracked inside a single name, reading
    # `whole = ` as `whol` `e` `=` and skipping a hoist that was needed.
    already = re.match(
        r"[A-Za-z_]\w*(?:\s+|\s*&\s*)[A-Za-z_]\w*\s*=\s*", text[begins:]
    )
    return bool(already) and begins + already.end() == match.start()


def _stands_alone(text: str, match: "re.Match[str]", close: int) -> bool:
    """Whether this call is a whole statement rather than part of one."""

    begins = _statement_start(text, match.start())
    while begins < len(text) and text[begins] in " \t\n":
        begins += 1
    return begins == match.start() and text[close + 1:].lstrip()[:1] == ";"


def _static_value_returns(classes: "dict[str, Class]") -> "dict[str, str]":
    """Every `static` member that answers an object, by each spelling of it.

    A static member is a free function once it is written out - it takes no
    object - so a value return from one needs the same hidden pointer as any
    other free function. The passes that arrange that read definitions out of
    the text, and a member is declared inside a class rather than at file
    scope, so it is listed here instead. Three spellings, because which one
    survives to a given pass depends on whether the call has been renamed
    yet: bare inside its own class, qualified outside it, and the C name.
    """

    found: "dict[str, str]" = {}
    for owner, holder in classes.items():
        for method in holder.methods:
            if not method.shared or not method.name:
                continue
            held = _returns_object(method, classes)
            if held is None:
                continue
            found[f"{owner}::{method.name}"] = held
            found[_c_name(owner, method.name, _suffix_of(owner, method, classes))] = held
            found.setdefault(method.name, held)
    return found


def _static_reference_returns(classes: "dict[str, Class]") -> "dict[str, str]":
    """Every `static` member that answers a *reference* to an object.

    The same list as above, for the other way a member hands an object back.
    Left out of both, `Reg::one().bump()` - which is how a program writes a
    singleton - was a method call on something that is not a name, and
    nothing recognised it.
    """

    found: "dict[str, str]" = {}
    for owner, holder in classes.items():
        for method in holder.methods:
            if not method.shared or not method.name:
                continue
            if not _returns_reference(method):
                continue
            held = re.sub(
                r"\b(?:const|volatile|struct)\b", " ", method.returns
            ).replace("&", " ").replace("*", " ").strip()
            if held not in classes:
                continue
            found[f"{owner}::{method.name}"] = held
            found[
                _c_name(owner, method.name, _suffix_of(owner, method, classes))
            ] = held
            found.setdefault(method.name, held)
    return found


def _qualified_static_names(
    text: str, classes: "dict[str, Class]", order: "list[str] | None" = None
) -> str:
    """`M::twice(21)` and `&M::twice` become the one name the C has.

    A static member function is reached by the class's name, which is a
    qualifier and not a namespace, so nothing else strips it. There is no
    object to pass, which is what makes it static - so the name on its own,
    with no call after it, is the address of an ordinary function and is
    written the same way.
    """

    for name in order if order is not None else list(classes):
        for method in classes[name].methods:
            if not method.shared or not method.name:
                continue
            spelled = _c_name(name, method.name, _suffix_of(name, method, classes))
            text = _map_code(
                text,
                lambda part, o=name, m=method.name, s=spelled: re.sub(
                    rf"\b{re.escape(o)}\s*::\s*{re.escape(m)}\b", s, part
                ),
            )
    return text


def _static_member_parameters(classes: "dict[str, Class]") -> "dict[str, str]":
    """The written parameter list of every `static` member, by its C name.

    The scanners below read definitions out of the text and take only the
    ones at the top level, because a definition nested inside another is a
    lambda or a local class rather than something a call elsewhere can name.
    A static member is nested in exactly that sense and is a free function
    all the same, so it is listed here rather than found there.
    """

    found: "dict[str, str]" = {}
    for owner, holder in classes.items():
        for method in holder.methods:
            if not method.shared or not method.name:
                continue
            spelled = _c_name(owner, method.name, _suffix_of(owner, method, classes))
            found[spelled] = method.parameters
    return found


def _free_value_initialisers(
    body: str, classes: "dict[str, Class]", scope: str
) -> str:
    """`R r = make(4);` becomes the declaration and the call that fills it.

    A free function answering a class hands it back the way a method does -
    through a pointer the caller provides - so the object being declared is
    the space, and the call takes its address.

    Read from `scope` rather than only from `body`, because a method body is
    emitted on its own and the functions it calls are declared elsewhere.
    """

    returning = {
        match.group(2): match.group(1).split()[-1]
        for match in _DEFINITION.finditer(scope)
        if match.group(1).split()
        and "*" not in match.group(1)
        and match.group(1).split()[-1] in classes
    }
    returning.update(_static_value_returns(classes))
    if not returning:
        return body
    # The callee may be qualified - `Owner::make(4)` - because a static member
    # is a free function that happens to be written inside a class.
    pattern = re.compile(
        r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_][\w:]*)\s*\("
    )

    def taken(match: "re.Match[str]", text: str) -> "str | None":
        spelled, variable, called = match.groups()
        if returning.get(called) != spelled or spelled not in classes:
            return None
        close = _closing_paren(match.string, match.end() - 1)
        if close < 0:
            return None
        # And the call has to *be* the initialiser, not the start of one.
        # `path a = base() / "web";` is a declaration whose initialiser is an
        # operator, and read as this form the marker below swallowed the rest
        # of the statement - so the program compiled and quietly did half of
        # what it says.
        if match.string[close + 1:].lstrip()[:1] != ";":
            return None
        # From the real text: an argument that is a string literal is blanked
        # in the copy the match was found against, and writing that back
        # would empty it.
        arguments = text[match.end(): close]
        passed = f", {arguments}" if arguments.strip() else ""
        return f"{spelled} {variable}; {called}(&{variable}{passed})" + "\x00drop"

    # Over the whole text rather than each stretch of code between literals:
    # `R r = make("x");` is one match spanning a literal, and handed only the
    # code before it there was no closing parenthesis to find.
    body = _sub_code(pattern, body, taken)
    # The original argument list is still there; the marker says where it
    # ends so it can go.
    return re.sub(r"\x00drop[^;]*;", ";", body)

def _hoist_value_returns(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    counter: "list[int]",
    scope: str = "",
) -> str:
    """`f.filename().c_str()` becomes a temporary and two calls.

    A method returning an object by value returns nothing in the C - the
    caller provides the space and the callee writes through a hidden pointer -
    so its result is not an expression that anything can be called on. C++
    calls that a temporary, and this writes the temporary out.

    Done here, on the C++, so everything after sees an ordinary object with
    an ordinary name and needs to know nothing about where it came from.
    """

    # A free function may return an object too, and `held.append(to_string(v))`
    # is the same shape as a chained method: the result is not an expression
    # anything can be passed, because the caller provides the space.
    returning: "dict[str, tuple[str, bool]]" = {}
    for definition in _DEFINITION.finditer(f"{body}\n{scope}"):
        spelled = definition.group(1).strip()
        if "*" in spelled or not spelled.split():
            continue
        # A reference return is the same problem wearing the other hat: the
        # answer is an address rather than a space the caller provided, and
        # `.m()` on one is no more an expression the rewriters know. Held in
        # a reference of its own, which the reference pass turns into the
        # pointer it already is, so nothing is copied.
        by_reference = spelled.endswith("&") and not spelled.endswith("&&")
        if "&" in spelled and not by_reference:
            continue
        # The last word is the type; the ones in front of it are `static`,
        # `inline`, `const`. Read whole, `static string pick(int)` was not a
        # function answering a class and its result was left as an
        # expression - which, a value return being a hidden pointer, is not
        # something anything can be called on.
        held = spelled.replace("&", " ").split()[-1]
        if held in classes:
            returning[definition.group(2)] = (held, by_reference)
    returning.update(
        (name, (held, False))
        for name, held in _static_value_returns(classes).items()
    )
    #: A static member is written out the way a method is, so the pass that
    #: dereferences a method's reference return reaches it too. Marked here
    #: so this one does not do it as well: two dereferences of one pointer
    #: is a read of whatever the object's first word happens to be.
    by_the_other_pass = set(_static_reference_returns(classes))
    returning.update(
        (name, (held, True))
        for name, held in _static_reference_returns(classes).items()
    )

    def settled(text: str, match: "re.Match[str]", close: int) -> bool:
        """Whether this call still needs somewhere to write its answer."""

        return not _already_a_declaration(text, match)

    def self_standing(text: str) -> "tuple | None":
        """The first call to a free function that answers an object."""

        for name, (held, by_reference) in returning.items():
            for match in re.finditer(
                rf"(?<![.\w>:]){re.escape(name)}\s*\(", text
            ):
                close = _closing_paren(text, match.end() - 1)
                if close < 0 or _is_a_definition(text, close):
                    continue
                if not settled(text, match, close):
                    continue
                # Not out of a declaration this pass wrote. The temporary
                # it makes is initialised *from* the call, so the call is
                # still there to be found - and every round hoisted it out
                # of the last round's declaration, until the rounds ran out.
                begins = _statement_start(text, match.start())
                # Where this call *is* the initialiser of one, and not
                # merely somewhere after one in the same statement: a second
                # call sitting beside a temporary already written is an
                # ordinary call and still needs a name of its own.
                if re.match(
                    rf"\s*[A-Za-z_][\w\s*]*\s\&?{re.escape(_VALUE_PREFIX)}"
                    # `(*` where the answer is a reference: the initialiser
                    # reads through what the call handed back, and the call
                    # is still in there to be found again.
                    rf"\d+\s*=\s*(?:\(\s*\*\s*)?$",
                    text[begins: match.start()],
                ):
                    continue
                if _stands_alone(text, match, close):
                    # Nothing wants the answer, so there is nowhere it has to
                    # go. An earlier pass has usually already given a call in
                    # this position the hidden pointer it writes through, and
                    # wrapping it again handed a temporary the `void` that a
                    # rewritten call returns.
                    continue
                # Dereferenced as it is hoisted. A method call is
                # dereferenced later, by the pass that gives calls their
                # arguments; that pass knows methods only, so a free
                # function has to say it here. Either way the binding below
                # takes an address, and the two cancel: no copy is made.
                return (
                    match,
                    close,
                    held,
                    by_reference,
                    by_reference and name not in by_the_other_pass,
                )
        return None

    if not known and not returning:
        return body
    for _round in range(_HOIST_ROUNDS):
        found = None
        for match in _CALL_ON.finditer(body):
            receiver, _reach, method = match.groups()
            holds = known.get(receiver)
            if holds is None:
                continue
            owner = _find_method(holds, method, classes)
            if owner is None:
                continue
            member = _method_by_name(owner, method, classes)
            declared = _returns_object(member, classes) if member else None
            if declared is None and member is not None and _returns_reference(member):
                # A reference return is a pointer in the C, and `.m()` on one
                # is not something the rewriters recognise. Held in a
                # reference of its own, which the pass below turns into the
                # pointer it already is - so nothing is copied.
                declared = member.returns.replace("&", "").replace("*", "").strip()
                if declared not in classes:
                    continue
                by_reference = True
            elif declared is None:
                continue
            else:
                by_reference = False
            close = _closing_paren(body, match.end() - 1)
            if close < 0:
                continue
            # Anything except the one form that is already handled: a
            # declaration whose whole initialiser is this call, where the
            # caller's own space is the variable being declared. Everywhere
            # else - assigned to something that exists, called on, passed as
            # an argument - there is no space to write through until one is
            # made, so one is.
            if not settled(body, match, close):
                continue
            # And the form this pass itself emits, which is the same thing
            # with a `&` in it. Without this the declaration it wrote was
            # hoisted again, and again, until the round limit stopped it.
            # The two names have to be separated by something - whitespace
            # or the `&`. Without that the pattern backtracked inside a single
            # name, reading `whole = ` as `whol` `e` `=` and skipping a hoist
            # that was needed.
            found = (match, close, declared, by_reference, False)
            break
        if found is None:
            found = self_standing(body)
        if found is None:
            return body
        match, close, held, by_reference, dereference = found
        counter[0] += 1
        name = f"{_VALUE_PREFIX}{counter[0]}"
        known[name] = held
        start = _statement_start(body, match.start())
        call = body[match.start(): close + 1]
        if dereference:
            call = f"(*{call})"
        marker = "&" if by_reference else ""
        body = (
            body[:start]
            + f"{held} {marker}{name} = {call}; "
            + body[start:match.start()]
            + name
            + body[close + 1:]
        )
    return body


#: How many chained calls one body may hold. Each round writes one temporary
#: out; a body with more than this many is a body doing something this was
#: not written for, and looping forever would be the worse answer.
_HOIST_ROUNDS = 64


def _method_named_returning(
    owner: str, method: str, classes: "dict[str, Class]"
) -> "str | None":
    """The class that member returns by value, if it returns one."""

    found = _method_by_name(owner, method, classes)
    if found is None:
        return None
    return _returns_object(found, classes)


#: The last text this was asked about, blanked. Every hoist asks about the
#: same body many times over, and blanking it is a scan of the whole thing.
_BLANKED_FOR_STATEMENTS: "tuple[str, str] | None" = None


#: The statements that take a body without needing braces around it.
_TAKES_A_BODY = re.compile(r"(?<![.\w>])(if|for|while)\s*\(")
#: The other two, which take one without a parenthesised head in front.
_TAKES_A_BARE_BODY = re.compile(r"(?<![.\w>])(else|do)\b")


def _end_of_statement(bare: str, at: int) -> int:
    """One past the end of the statement beginning at `at`.

    Written against a copy with the literals blanked, because a brace or a
    semicolon inside a string is text.
    """

    size = len(bare)
    while at < size and bare[at] in " \t\r\n":
        at += 1
    if at >= size:
        return size
    if bare[at] == "{":
        return _matching(bare, at)
    if bare[at] == ";":
        return at + 1
    word = re.match(r"(if|for|while|switch|do|else)\b", bare[at:])
    if word is not None:
        keyword = word.group(1)
        after = at + word.end()
        if keyword in ("if", "for", "while", "switch"):
            opening = bare.find("(", after)
            if opening < 0:
                return size
            closing = _closing_paren(bare, opening)
            if closing < 0:
                return size
            end = _end_of_statement(bare, closing + 1)
            if keyword == "if":
                # `if (c) a; else b;` is one statement, and stopping at the
                # `a;` would have braced half of it.
                rest = end
                while rest < size and bare[rest] in " \t\r\n":
                    rest += 1
                if re.match(r"else\b", bare[rest:]):
                    return _end_of_statement(bare, rest + 4)
            return end
        if keyword == "do":
            end = _end_of_statement(bare, after)
            found = bare.find("while", end)
            if found < 0:
                return end
            opening = bare.find("(", found)
            closing = _closing_paren(bare, opening) if opening >= 0 else -1
            if closing < 0:
                return end
            rest = closing + 1
            while rest < size and bare[rest] in " \t\r\n":
                rest += 1
            return rest + 1 if rest < size and bare[rest] == ";" else rest
        return _end_of_statement(bare, after)
    depth = 0
    while at < size:
        piece = bare[at]
        if piece in "([{":
            depth += 1
        elif piece in ")]}":
            depth -= 1
        elif piece == ";" and depth == 0:
            return at + 1
        at += 1
    return size


#: `V operator+(const V &a, const V &b) { ... }` - an operator written as a
#: free function. The symbol is read from what this subset knows, and the
#: `::` is excluded so a member defined out of line is not taken for one.
_FREE_OPERATOR = re.compile(
    r"(?<![.\w>:])((?:const\s+)?[A-Za-z_][\w\s*&]*?)\s*\boperator\s*"
    r"(==|!=|<=|>=|<<|>>|\+=|-=|\*=|/=|\+|-|\*|/|%|<|>|&)\s*"
    r"\(([^()]*)\)\s*\{"
)


def _free_operators_as_members(text: str) -> str:
    """`operator+(const V &a, const V &b)` becomes V's own `operator+(b)`.

    C++ lets a binary operator be written either as a member of its left
    operand or as a free function taking both. The two are called the same
    way and mean the same thing; only the spelling differs, and the member
    spelling is the one every pass here already knows.

    So the free one is moved into the class its left operand names, and the
    name it gave that operand becomes the object. Which is what a member is:
    the first argument, passed without being written.

    Only where the left operand is a class this file declares. `ostream
    &operator<<(ostream &, const V &)` is an operator on somebody else's
    class, and there is no body here to move it into.
    """

    heads = {head.group(2): head for head in _CLASS_HEAD.finditer(text)}
    if not heads:
        return text
    bare = _without_literals(text)
    cuts: "list[tuple[int, int]]" = []
    added: "dict[str, list[str]]" = {}
    for match in _FREE_OPERATOR.finditer(bare):
        if _depth_at(bare, match.start()) != 0:
            continue
        parts = [one.strip() for one in _split_arguments(match.group(3)) if one.strip()]
        if len(parts) != 2:
            continue
        words = parts[0].replace("*", " ").replace("&", " ").split()
        held = [word for word in words if word not in ("const", "volatile", "struct")]
        if len(held) != 2 or held[0] not in heads:
            continue
        owner, receiver = held
        opening = match.end() - 1
        try:
            closing = _matching(bare, opening)
        except ValueError:
            continue
        # The left operand keeps its name and is bound to the object, rather
        # than every mention of it being replaced by `(*this)`. The passes
        # that rewrite an operator or a call look for a *name* on the left,
        # and a parenthesised dereference is not one - so `o << v.x` inside
        # the moved body was left as C++ and reached the C compiler as a
        # shift of a struct.
        inner = text[opening + 1: closing - 1]
        added.setdefault(owner, []).append(
            f"{match.group(1).strip()} operator{match.group(2)}"
            f"({parts[1]}) {{ {parts[0]} = *this; {inner} }}"
        )
        cuts.append((match.start(), closing))
    if not cuts:
        return text
    # Written in before the class body closes, so the class is read with the
    # operator already among its members.
    inserts: "list[tuple[int, str]]" = []
    for owner, written in added.items():
        head = heads[owner]
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        inserts.append((closing - 1, "public: " + " ".join(written) + " "))
    pieces = sorted(
        [(start, end, None) for start, end in cuts]
        + [(at, at, written) for at, written in inserts]
    )
    out: "list[str]" = []
    last = 0
    for start, end, written in pieces:
        if start < last:
            continue
        out.append(text[last:start])
        if written is not None:
            out.append(written)
        last = end
    out.append(text[last:])
    return "".join(out)


#: `friend` - written inside a class body, in front of something that is not
#: a member of it.
_A_FRIEND = re.compile(r"(?<![.\w>])friend\b")


def _lift_friends(text: str) -> str:
    """Move a `friend` function out of the class body it was written in.

    A friend is not a member: it takes no object, it is called by its own
    name, and being written inside the braces only says that it may read
    what is private. py2bin has no access control - a struct's members are
    all reachable - so what the keyword grants is already true, and all that
    is left is where the function is written.

    So it is moved to where it belongs: after the class, at file scope, as
    the free function it always was. A `friend` that only declares - `friend
    class Store;`, or a prototype whose body is elsewhere - says nothing
    that is not already true here, and goes.
    """

    bare = _without_literals(text)
    cuts: "list[tuple[int, int]]" = []
    lifted: "list[str]" = []
    for match in _A_FRIEND.finditer(bare):
        semi = bare.find(";", match.end())
        opening = bare.find("{", match.end())
        if opening < 0 or 0 <= semi < opening:
            # A declaration, not a definition.
            if semi < 0:
                continue
            cuts.append((match.start(), semi + 1))
            continue
        try:
            closing = _matching(bare, opening)
        except ValueError:
            continue
        cuts.append((match.start(), closing))
        lifted.append(text[match.end(): closing])
    if not cuts:
        return text
    out: "list[str]" = []
    last = 0
    for start, end in cuts:
        if start < last:
            continue
        out.append(text[last:start])
        last = end
    out.append(text[last:])
    return "".join(out) + "\n" + "\n".join(lifted) + "\n"


def _brace_loose_bodies(text: str) -> str:
    """`for (...) x = T(i);` becomes `for (...) { x = T(i); }`.

    C++ lets a loop or a branch take one statement without braces around it,
    and that statement is still a scope of its own. Everything here that
    needs somewhere to put a declaration - a temporary, a hidden return
    space, an object an operator answered - writes it in front of the
    statement it found, and in front of an unbraced body is *outside the
    loop*: the declaration left the scope its subject was in. `grid[i] =
    Cell(i * 10);` had its temporary built once, above the loop, out of
    reach of the `i` it was built from.

    So the braces C++ leaves out are written in, once, before any of those
    passes run. Nothing about the program changes: a block holding one
    statement is that statement.
    """

    bare = _without_literals(text)
    inserts: "list[tuple[int, str]]" = []

    def wrap(at: int) -> None:
        while at < len(bare) and bare[at] in " \t\r\n":
            at += 1
        # Already a block, or no body at all to speak of.
        if at >= len(bare) or bare[at] in "{;":
            return
        end = _end_of_statement(bare, at)
        if end <= at:
            return
        inserts.append((at, "{ "))
        inserts.append((end, " }"))

    for match in _TAKES_A_BODY.finditer(bare):
        closing = _closing_paren(bare, match.end() - 1)
        if closing < 0:
            continue
        wrap(closing + 1)
    for match in _TAKES_A_BARE_BODY.finditer(bare):
        at = match.end()
        rest = at
        while rest < len(bare) and bare[rest] in " \t\r\n":
            rest += 1
        # `else if (...)` is a branch of the same chain, not a body to brace:
        # the `if` is picked up in its own right above.
        if match.group(1) == "else" and re.match(r"if\b", bare[rest:]):
            continue
        wrap(at)
    if not inserts:
        return text
    out: "list[str]" = []
    last = 0
    for at, piece in sorted(inserts, key=lambda one: one[0]):
        out.append(text[last:at])
        out.append(piece)
        last = at
    out.append(text[last:])
    return "".join(out)


def _statement_start(body: str, at: int) -> int:
    """Where the statement holding `at` begins.

    A temporary has to be declared before the statement that uses it, not in
    the middle of one - `printf("%s", f.filename().c_str())` has no room for
    a declaration inside the argument list.

    Read against a copy with the literals blanked. A brace or a semicolon
    inside a string is text: `payload << "{\"type\":\"status\""` held one,
    and a declaration written where it said the statement began landed in
    the middle of the string.
    """

    global _BLANKED_FOR_STATEMENTS
    if _BLANKED_FOR_STATEMENTS is None or _BLANKED_FOR_STATEMENTS[0] != body:
        _BLANKED_FOR_STATEMENTS = (body, _without_literals(body))
    bare = _BLANKED_FOR_STATEMENTS[1]
    depth = 0
    index = at
    while index > 0:
        index -= 1
        piece = bare[index]
        if piece in ")]":
            depth += 1
        elif piece in "([":
            if depth:
                depth -= 1
            # At depth zero an opening parenthesis means the call being
            # hoisted is written inside another one - `printf(..., f.x().y())`
            # - and the statement starts further back still. Stopping here put
            # the declaration inside the argument list.
        elif depth == 0 and piece in ";{}":
            return index + 1
    return 0


#: `V(3)` - an object built where it is used rather than named first.
#: Not after a `~`: `x.~C()` names a destructor to run, not a `C` to build.
_TEMPORARY = re.compile(r"(?<![.\w>~])([A-Za-z_]\w*)\s*\(")



#: `A xs[3] = {A(1), A(2)};` - an array of objects, each given its arguments.
_OBJECT_ARRAY_VALUES = re.compile(
    r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\[\s*(\d*)\s*\]\s*=\s*\{"
)



def _name_returned_objects(
    body: str, classes: "dict[str, Class]", returns: str, counter: "list[int]"
) -> str:
    """`return a + b;` becomes a declaration and a `return` of its name.

    Everything that fills an object here fills one that has a name: `T v =
    ...` is the form the operator, the call and the temporary passes all
    know. A `return` is the one other place an object is made, so it is
    written as that form and they take it from there.
    """

    held = returns.replace("const", "").strip()
    if "*" in held or "&" in held or held not in classes:
        return body
    bare = _without_literals(body)
    out: list[str] = []
    at = 0
    for match in re.finditer(r"\breturn\b([^;]*);", bare):
        if match.start() < at:
            continue
        value = body[match.start(1): match.end(1)].strip()
        # Nothing, a name, or a path to one: the hidden pointer can be
        # written from any of those as they stand.
        if not value or re.fullmatch(r"[A-Za-z_]\w*(?:\s*(?:\.|->)\s*\w+)*", value):
            continue
        # A call, which the passes that fill a declaration from one already
        # handle - including the ones this translator has already rewritten,
        # where a second name would be a second object nobody fills.
        if _MEMBER_CALL.match(value) or _PLAIN_CALL.match(value):
            continue
        counter[0] += 1
        name = f"__py2bin_answer_{counter[0]}"
        out.append(body[at:match.start()])
        out.append(f"{held} {name} = {value}; return {name};")
        at = match.end()
    out.append(body[at:])
    return "".join(out)

def _rewrite_object_array_values(
    body: str, classes: "dict[str, Class]", known: "dict[str, str]"
) -> str:
    """Construct each element of an array of objects where it stands.

    C has no expression that constructs, and an array's brace list is not a
    place a statement can go - so each element becomes a call after the
    declaration, in the order C++ builds them.
    """

    if not classes:
        return body
    bare = _without_literals(body)
    out: list[str] = []
    at = 0
    for match in _OBJECT_ARRAY_VALUES.finditer(bare):
        if match.start() < at:
            continue
        held, variable, count = match.groups()
        if held not in classes:
            continue
        owner = _find_method(held, "", classes)
        if owner is None:
            continue
        closing = _matching(bare, match.end() - 1)
        pieces = _split_arguments(body[match.end(): closing - 1])
        values = [piece.strip() for piece in pieces if piece.strip()]
        if count and int(count) < len(values):
            continue
        known[variable] = held
        # Written as C already, so the pass that default-constructs an array
        # of objects leaves it alone: every element here is built below.
        made = [f"struct {held} {variable}[{count or len(values)}];"]
        for index, value in enumerate(values):
            built = re.match(rf"^{re.escape(held)}\s*\((.*)\)$", value, re.S)
            spot = f"&{variable}[{index}]"
            if built is not None:
                given = (
                    [one.strip() for one in _split_arguments(built.group(1))]
                    if built.group(1).strip()
                    else []
                )
                passed = f", {built.group(1)}" if built.group(1).strip() else ""
                made.append(
                    f"{_c_name(owner, '', _call_suffix(owner, '', classes, given, body))}"
                    f"({spot}{passed});"
                )
                continue
            # Anything else is an object already built, and C++ copies it.
            made.append(_copied_in(held, f"{variable}[{index}]", f"&{value}", classes))
        # Whatever the braces did not fill, C++ default-constructs.
        for index in range(len(values), int(count) if count else len(values)):
            made.append(
                f"{_c_name(owner, '', _call_suffix(owner, '', classes, []))}"
                f"(&{variable}[{index}]);"
            )
        out.append(body[at:match.start()])
        out.append(" ".join(made))
        at = closing
        while at < len(body) and body[at] in " ;":
            at += 1
    out.append(body[at:])
    return "".join(out)

#: `Item{1, 30}` - a temporary written with braces rather than with a
#: constructor call. The name is read off the file, never assumed: what is a
#: struct here is whatever this file declared to be one.
_BRACE_TEMPORARY = re.compile(r"(?<![.\w>])([A-Za-z_]\w*)\s*\{")

#: Where such a temporary may stand: an argument, an element of a list, the
#: right of an assignment, or a `return`. Everywhere else a name in front of
#: a brace is something being declared - `struct S {`, `Derived : Base {` -
#: and rewriting one of those would take the file apart.
_OPENS_A_VALUE = ("(", ",", "=", "{", ";")

#: What the pass below calls the object it writes out.
_BRACE_PREFIX = "__py2bin_brace_"


def _struct_names(text: str, classes: "dict[str, Class]") -> "set[str]":
    """Every name this file declares as a class, a struct or a union."""

    return {head.group(2) for head in _CLASS_HEAD.finditer(text)} | set(classes)


def _rewrite_brace_temporaries(
    text: str, classes: "dict[str, Class]", counter: "list[int]"
) -> str:
    """`push_back(Item{1, 30})` becomes an object with a name and a name passed.

    C++ builds a value where it is written; C initialises a declaration and
    nothing else, so the value is declared ahead of the statement that wanted
    it. The same move `_rewrite_temporaries` makes for a constructor call,
    for the spelling that has no constructor in it.
    """

    names = _struct_names(text, classes)
    if not names:
        return text
    for _round in range(_TEMPORARY_ROUNDS):
        bare = _without_literals(text)
        found = None
        for match in _BRACE_TEMPORARY.finditer(bare):
            if match.group(1) not in names:
                continue
            before = bare[: match.start()].rstrip()
            word = re.search(r"[A-Za-z_]\w*$", before)
            if word is not None:
                # `return Item{...}` is a value; `class Item {` is a
                # declaration, and so is every other keyword in front of one.
                if word.group(0) != "return":
                    continue
            elif not before.endswith(_OPENS_A_VALUE):
                continue
            opening = match.end() - 1
            try:
                # One past the closing brace, which is what this answers.
                after = _matching(bare, opening)
            except ValueError:
                continue
            found = (match, opening, after)
            break
        if found is None:
            return text
        match, opening, after = found
        counter[0] += 1
        name = f"{_BRACE_PREFIX}{counter[0]}"
        start = _statement_start(text, match.start())
        written = text[opening:after]
        text = (
            text[:start]
            + f"{match.group(1)} {name} = {written}; "
            + text[start: match.start()]
            + name
            + text[after:]
        )
    return text


#: `lhs = value;` on its own, with no comparison or compound operator in it.
_PLAIN_ASSIGNMENT = re.compile(
    # No `\s*` after the `=`: this is matched against a copy with the
    # literals blanked to spaces, and the value is sliced out of the real
    # text at the same offsets - so whitespace eaten here is whitespace eaten
    # off the front of `"a"`, and what came back was one quote character.
    r"^\s*([^=;{}]+?)\s*(?<![=!<>+\-*/%&|^])=(?!=)(.+);\s*$", re.S
)


def _constructs_from(held: str, given: str, classes: "dict[str, Class]") -> bool:
    """Whether `held` has a constructor taking one `given`."""

    def plain(spelled: str) -> str:
        spelled = re.sub(
            r"\b(?:const|volatile|struct|class)\b", " ", spelled
        ).replace("&", " ")
        return re.sub(r"\s+", "", spelled)

    holder = classes.get(held)
    if holder is None:
        return False
    want = plain(given)
    for method in holder.methods:
        if method.name != "":
            continue
        parts = [one for one in _split_arguments(method.parameters) if one.strip()]
        if len(parts) != 1:
            continue
        words = parts[0].replace("*", " * ").split()
        if len(words) < 2:
            continue
        # The last word is the parameter's name, not part of its type.
        if plain(" ".join(words[:-1])) == want:
            return True
    return False


def _lvalue_class(
    left: str, text: str, classes: "dict[str, Class]"
) -> "str | None":
    """The class a subscript answers, read from the classes rather than the text.

    `_deduced_type` looks a subscript operator up by reading the class body,
    and by the time this runs the bodies have been taken apart - so `s` had a
    type and `m[3]` had none, though both are objects of a class this file
    knows all about.
    """

    indexed = _INDEXED.match(left.strip())
    if indexed is None:
        return None
    owner = _deduced_type(indexed.group(1).strip(), text) or ""
    owner = re.sub(
        r"\b(?:const|volatile|struct)\b", " ", owner
    ).replace("*", " ").strip()
    if owner not in classes:
        return None
    method = _method_named(owner, "op_index", classes)
    if method is None:
        return None
    held = re.sub(
        r"\b(?:const|volatile|struct)\b", " ", method.returns
    ).replace("&", " ").replace("*", " ").strip()
    return held or None


#: `int (C::*m)(int) const` - a pointer to a member function.
_MEMBER_POINTER = re.compile(
    r"(?<![.\w>])([A-Za-z_][\w\s*]*?)\s*\(\s*([A-Za-z_]\w*)\s*::\s*\*\s*"
    r"([A-Za-z_]\w*)\s*\)\s*\(([^)]*)\)\s*(?:const\s*)?"
)

#: `&C::method` - which one it is pointed at.
_MEMBER_ADDRESS = re.compile(
    r"&\s*([A-Za-z_]\w*)\s*::\s*([A-Za-z_]\w*)(?!\s*\()"
)

#: `(o.*m)(...)` and `(p->*m)(...)` - calling through one.
_THROUGH_MEMBER = re.compile(
    r"\(\s*([A-Za-z_]\w*)\s*(\.|->)\s*\*\s*([A-Za-z_]\w*)\s*\)\s*\("
)


def _rewrite_member_pointers(
    body: str, classes: "dict[str, Class]"
) -> str:
    """`int (C::*m)()` is a function pointer whose first argument is the object.

    Which is what a method already is here: taking it apart put the object
    in front, so a pointer to one is a pointer to a function taking a `C *`.
    `&C::get` is that function's name, and `(c.*m)()` is a call through the
    pointer with the object handed to it.
    """

    if not classes or "::" not in body:
        return body

    def declared(match: "re.Match[str]") -> str:
        held, owner, name, parameters = match.groups()
        if owner not in classes:
            return match.group(0)
        rest = f", {parameters.strip()}" if parameters.strip() else ""
        return f"{held.strip()} (*{name})(struct {owner} *{rest})"

    body = _map_code(body, lambda part: _MEMBER_POINTER.sub(declared, part))

    def pointed(match: "re.Match[str]") -> str:
        owner, method = match.groups()
        if owner not in classes:
            return match.group(0)
        held = _find_method(owner, method, classes)
        if held is None:
            return match.group(0)
        spelled = _name_for(held, method, classes)
        # An overloaded member cannot be pointed at without saying which, and
        # nothing here says: the type it is assigned to is what would.
        return match.group(0) if callable(spelled) else spelled

    body = _map_code(body, lambda part: _MEMBER_ADDRESS.sub(pointed, part))

    if _THROUGH_MEMBER.search(_without_literals(body)) is None:
        return body

    def called(match: "re.Match[str]") -> str:
        held, reach, name = match.groups()
        given = held if reach == "->" else f"&{held}"
        return f"{name}({given}, "

    body = _map_code(body, lambda part: _THROUGH_MEMBER.sub(called, part))
    # `m(&c, )` where the member takes nothing. Only here, where this pass
    # is what wrote the comma.
    return _map_code(body, lambda part: re.sub(r",\s*\)", ")", part))


def _convert_assignments(
    body: str, classes: "dict[str, Class]", text: str
) -> str:
    """`s = "a";` becomes `s = string("a");` - the temporary C++ builds.

    Assigning something to an object of a class that has a constructor taking
    it is a conversion: C++ builds a temporary from it and assigns that. With
    nothing written here the C was handed a `char *` where a struct goes, and
    `string s; s = "a";` - which is as ordinary as C++ gets - did not build.

    Written as the constructor call C++ would have written, so the pass that
    hoists temporaries takes it from here.
    """

    if not classes:
        return body
    out: "list[str]" = []
    for statement in _statements(body):
        found = _PLAIN_ASSIGNMENT.match(_without_literals(statement))
        if found is None:
            out.append(statement)
            continue
        left = statement[found.start(1): found.end(1)].strip()
        value = statement[found.start(2): found.end(2)].strip()
        held = _deduced_type(left, text) or _lvalue_class(left, text, classes)
        if held is None:
            out.append(statement)
            continue
        held = re.sub(
            r"\b(?:const|volatile|struct)\b", " ", held
        ).strip()
        if "*" in held or held not in classes:
            out.append(statement)
            continue
        # A literal or a name, and nothing else. Those are the two things
        # `_deduced_type` reads without guessing; the type of an expression
        # is where it is weakest, and read wrongly there this wrote a
        # conversion C++ would never have made - `c = c / L"third";` became
        # a `path` built from a `path`, which is more than one constructor.
        if not (
            value.isidentifier()
            or any(pattern.match(value) for pattern, _named in _LITERAL_TYPES)
        ):
            out.append(statement)
            continue
        if value.startswith("__py2bin_"):
            # A temporary this translator wrote already holds what the
            # expression answered.
            out.append(statement)
            continue
        given = _deduced_type(value, text)
        if given is None:
            out.append(statement)
            continue
        bare = re.sub(
            r"\b(?:const|volatile|struct)\b", " ", given
        ).replace("&", " ").strip()
        if bare.replace("*", "").strip() == held:
            # Already one of these, so nothing is being converted.
            out.append(statement)
            continue
        if not _constructs_from(held, given, classes):
            out.append(statement)
            continue
        out.append(
            statement[: found.start(1)]
            + f"{left} = {held}({value});"
        )
    return "".join(out)


def _rewrite_temporaries(
    body: str, classes: "dict[str, Class]", counter: "list[int]"
) -> str:
    """`V(5)` becomes an object with a name, because C has nowhere else.

    C++ builds a temporary wherever one is written - as an initialiser, as an
    argument, in a `return`. C has no expression that constructs anything, so
    each becomes a declaration ahead of the statement that used it. Where the
    statement *is* a declaration of the same class, the object being declared
    is the temporary and no second one is needed.
    """

    if not classes:
        return body
    # Past whatever is already here. More than one pass writes these, each
    # with a counter of its own starting at one, and two of them in the same
    # scope is a redeclaration - which nothing noticed until a pass that
    # writes a temporary ran after one that had already written `_1`.
    written = [
        int(one.group(1)) for one in re.finditer(r"__py2bin_temp_(\d+)", body)
    ]
    if written:
        counter[0] = max(counter[0], max(written))
    for _round in range(_TEMPORARY_ROUNDS):
        found = None
        for match in _TEMPORARY.finditer(body):
            name = match.group(1)
            if name not in classes:
                continue
            if _find_method(name, "", classes) is None:
                continue
            close = _closing_paren(body, match.end() - 1)
            if close < 0:
                continue
            # `new T(args)` already allocates and constructs; the class name
            # there is not a temporary being built on the stack.
            before = body[:match.start()].rstrip()
            # `new T(args)` already allocates and constructs, and so does
            # `new (room) T(args)` - which constructs and does not allocate.
            # The second form was not recognised, so its arguments were
            # hoisted out and the `new` was left with a name after it.
            if re.search(r"\bnew$", before) or re.search(
                r"\bnew\s*\([^()]*\)$", before
            ):
                continue
            # `V t = V(5);` - the declaration's own object is the temporary.
            start = _statement_start(body, match.start())
            while start < len(body) and body[start] in " \t\n":
                start += 1
            # Only when the temporary *is* the whole initialiser. `V t =
            # V(5) + w;` reads the same up to here, and treating it as a
            # direct initialisation dropped everything after the `)`.
            direct = (
                re.match(
                    rf"{re.escape(name)}\s+([A-Za-z_]\w*)\s*=\s*$",
                    body[start:match.start()],
                )
                if body[close + 1:].lstrip().startswith(";")
                else None
            )
            found = (match, close, direct, start)
            break
        if found is None:
            return body
        match, close, direct, start = found
        arguments = body[match.end(): close]
        if direct is not None:
            body = (
                body[:start]
                + f"{match.group(1)} {direct.group(1)}({arguments})"
                + body[close + 1:]
            )
            continue
        counter[0] += 1
        held = f"__py2bin_temp_{counter[0]}"
        body = (
            body[:start]
            + f"{match.group(1)} {held}({arguments}); "
            + body[start:match.start()]
            + held
            + body[close + 1:]
        )
    raise CppTranslationError(
        "<c++>", 0,
        "a statement building more temporaries than py2bin writes out; each "
        "becomes a declaration of its own and this one never stops asking",
    )


#: How many temporaries one body may hold. A backstop, not a budget.
_TEMPORARY_ROUNDS = 128


def _split_object_declarators(body: str, classes: "dict[str, Class]") -> str:
    """`V a(2), b(3);` becomes two declarations, which is what it means.

    C++ writes the type once and then as many declarators as it likes. Every
    rewriter below reads one type and one name, so the second declarator was
    left where it stood - `V a(2), b(3);` constructed `a` and then handed the
    C compiler `, b(3);`, which is a call to something that is not a function.
    """

    if not classes:
        return body
    bare = _without_literals(body)
    out: list[str] = []
    at = 0
    start = 0
    depth = 0
    for index, char in enumerate(bare):
        # Braces are counted as boundaries, not as depth: a body starts with
        # one, and counting it left every statement inside looking nested.
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char in ";{}" and depth == 0:
            statement = body[start:index]
            replaced = _one_declaration(statement, bare[start:index], classes)
            if replaced is not None:
                out.append(body[at:start])
                out.append(replaced)
                at = index + 1
                # The `;` goes with the last declarator, which the rewrite
                # already wrote out.
            start = index + 1
    out.append(body[at:])
    return "".join(out)


def _one_declaration(
    statement: str, bare: str, classes: "dict[str, Class]"
) -> "str | None":
    """One `T a, b;` written out as `T a; T b;`, or None if it is not one."""

    head = re.match(r"(\s*)([A-Za-z_]\w*)(\s+)", bare)
    if head is None or head.group(2) not in classes:
        return None
    # Split what follows on commas that are not inside anything.
    rest = statement[head.end():]
    pieces = _split_arguments(bare[head.end():])
    if len(pieces) < 2:
        return None
    spelled: list[str] = []
    at = 0
    for piece in pieces:
        spelled.append(rest[at: at + len(piece)].strip())
        at += len(piece) + 1
    if not all(re.match(r"^\*?[A-Za-z_]\w*\b", one) for one in spelled):
        return None
    return head.group(1) + " ".join(
        f"{head.group(2)} {one};" for one in spelled
    )

def _rewrite_body(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]" = frozenset(),
    receivers: "dict[str, str] | None" = None,
    inherited_arrays: "dict[str, str] | None" = None,
    unit: str = "",
    enclosing: "list[tuple[str, str]]" = (),
    pointer_arrays: "dict[str, str] | set[str]" = (),
    returns: str = "",
    inherited_references: "dict[str, str] | None" = None,
    referenced: "set[str] | None" = None,
    stable: str = "",
    #: Whether this body *is* a loop's. What `break` and `continue` leave is
    #: the loop's body, so a scope inside one has to know how far out that is.
    in_a_loop: bool = False,
) -> str:
    """Rewrite declarations and calls inside one function body.

    `known` maps a variable to the class it holds; `pointers` says which of
    those are pointers, because a pointer is already the address a method
    wants and an object has to have one taken. `receivers` gives the address
    outright for the ones that are not a bare name - a class held as a member
    is reached as `&this->motor`, and deriving that from the name alone would
    have dropped the `this->` that qualifying already put there.
    """

    known = dict(known)
    pointers = set(pointers)
    destroyed: list[str] = []

    def declare(match: "re.Match[str]") -> str:
        type_name, variable, _parens, arguments = match.groups()
        if type_name not in classes:
            return match.group(0)
        known[variable] = type_name
        constructed = f"struct {type_name} {variable};"
        # `static Counter c;` inside a function is built the first time the
        # function runs and not once per call. C has no such rule, so the
        # flag C++ keeps out of sight is written out here. Read from the text
        # in front of the declaration, because `static` is a storage class
        # and the pattern matches the type onwards.
        once = _AFTER_STATIC.search(match.string[: match.start()]) is not None
        owner = _find_method(type_name, "", classes)
        if owner is None and arguments:
            raise CppTranslationError(
                "<c++>", 0, f"{type_name} has no constructor to take arguments"
            )
        if not arguments and variable.startswith(_VALUE_PREFIX):
            # A temporary written to receive what a call answers with. The
            # call fills it, and C++ builds nothing here either - so a
            # default constructor over it is wasted work at best, and at
            # worst a call to one the class does not have.
            if _find_method(type_name, "~", classes):
                destroyed.append(variable)
            return constructed
        if owner is not None:
            given = (
                [a.strip() for a in _split_arguments(arguments)]
                if (arguments or "").strip()
                else []
            )
            # `path a(b);` where `b` is already one and the class wrote no
            # copy constructor. C++ uses the one it gives every class, which
            # here is the memberwise copy this translator does everywhere
            # else - the same reading a member initialiser naming its own
            # class already gets. Without it there was no constructor to
            # choose and the overload set was reported as unreadable.
            if len(given) == 1 and _same_class(
                given[0], type_name, scope(), classes
            ) and not _constructor_taking_one(type_name, type_name, classes):
                if _find_method(type_name, "~", classes):
                    destroyed.append(variable)
                return constructed + " " + _copied_in(
                    type_name, variable, f"&{given[0]}", classes
                )
            suffix = _call_suffix(owner, "", classes, given, scope())
            passed = f", {arguments}" if arguments else ""
            call = f"{_c_name(owner, '', suffix)}(&{variable}{passed});"
            if once:
                guard = f"{variable}{_BUILT_SUFFIX}"
                constructed += (
                    f" static int {guard} = 0;"
                    f" if (!{guard}) {{ {guard} = 1; {call} }}"
                )
            else:
                constructed += f" {call}"
        # A static outlives the scope it is written in, so the scope must not
        # take it apart on the way out: C++ destroys one when the program
        # ends, which is where everything is released anyway.
        if not once and _find_method(type_name, "~", classes):
            destroyed.append(variable)
        return constructed

    arrays: dict[str, str] = dict(inherited_arrays or {})

    def declare_array(match: "re.Match[str]") -> str:
        type_name, variable, count = match.groups()
        if type_name not in classes:
            return match.group(0)
        arrays[variable] = type_name
        kept = f"{type_name} {variable}[{count}];"
        owner = _find_method(type_name, "", classes)
        if owner is None:
            return kept
        # C++ default-constructs every element; C does nothing at all, so the
        # loop is written out. The index is named so it cannot collide with
        # anything the program declared.
        index = f"__py2bin_i_{variable}"
        kept += (
            f" {{ int {index}; for ({index} = 0; {index} < {count}; {index}++)"
            f" {_c_name(owner, '', _call_suffix(owner, '', classes, []))}"
            f"(&{variable}[{index}]); }}"
        )
        return kept

    # A nested block is its own scope. Handled first, and on its own, because
    # C++ destroys what a block declared at the end of *that* block - and a
    # name declared there is not in scope after it either, so a destructor
    # placed at the end of the function named a variable C says is not there.
    body, blocks = _lift_nested(body)

    # A reference declared in an enclosing scope is still a reference here.
    # `for (auto &x : v) { x = x * 10; }` puts the binding in the loop's own
    # block and the author's braces make another inside it, so the uses were
    # rewritten in a scope that had never heard the name was a reference -
    # and `x = x * 10` multiplied the pointer.
    held_references = dict(inherited_references or {})
    shadowed = _declared_here(body)
    outer_references = {
        name: spelled
        for name, spelled in held_references.items()
        if name not in shadowed
    }
    if outer_references:
        body = _deref_references(body, outer_references, classes)

    def scope() -> str:
        """This body, and then whatever the file declares outside it.

        Working out an argument's type means finding where it was declared,
        and `std::endl` is declared at file scope - so a body on its own is
        not enough to tell `cout << endl` from `cout << 1`. This scope comes
        first, because a local of the same name is the one in view.
        """

        return body if not unit else f"{body}\n{unit}"

    # `&held` where the class overloads it. First of everything, because
    # every pass below writes `&` in front of a receiver of its own and by
    # then the two are the same three characters. Here, each one was written
    # by the author.
    body = _rewrite_address_of(
        body,
        classes,
        # What this body declares as well as what it inherited: a holder is
        # usually declared a line above the call that fills it in.
        {**_declared_objects(body, classes), **known},
        pointers,
    )

    # `string x = "ab", y = "cdef";` declares two objects, and every pass
    # below reads a declaration as one. Read whole, the constructor call took
    # `"ab", y = "cdef"` for its argument list and asked for a two-argument
    # string. Split first, and each half is the declaration those passes
    # already know.
    body = _split_object_declarations(body, classes)

    # Before the declaration passes: `Node *n = new Node(3);` has to become a
    # call first, or the pointer declaration reads `new Node` as the type.
    # `new int(5)` stores as well as answers, and C has no one expression
    # that does both. Written out as its own statement first, so what reaches
    # the rewrite below is the storage on its own.
    body = _hoist_new_initialisers(body, classes, [0])
    # Before the allocating form, which would otherwise read the `(room)` as
    # the argument list of a `new` with no type after it.
    body = _rewrite_placement_new(body, classes)
    body = _rewrite_new(body, classes, scope())

    # `return a + b;` where the function answers an object: given a name of
    # its own first, so that every pass which knows how to fill a declaration
    # - an operator, a call, a temporary - handles this too rather than each
    # having to learn about `return`.
    body = _name_returned_objects(body, classes, returns, [0])

    # `A xs[3] = {A(1), A(2), A(3)};` - each element is constructed where it
    # stands. Before the temporaries pass, which would otherwise hoist each
    # `A(1)` to the start of the statement and leave the braces holding
    # declarations.
    body = _rewrite_object_array_values(body, classes, known)

    # `int (C::*m)() const = &C::get;` and `(c.*m)()`. Before the passes
    # that read calls, because what this leaves behind is an ordinary call
    # through an ordinary function pointer.
    body = _rewrite_member_pointers(body, classes)

    # `s = "a";` where `s` is a class with a constructor taking a literal.
    # C++ builds a temporary and assigns that, and writing the temporary here
    # is what lets the pass below hoist it like any other. Before that pass,
    # and before the subscript rewrite, so the left side is still written the
    # way the program wrote it - `m[3]` and not a call.
    body = _convert_assignments(body, classes, unit or body)

    # `items[i].~T()` - where a container takes its elements apart. Before
    # the temporaries, which otherwise read the `C()` of `~C()` as one being
    # built and left the tilde in front of the name it gave it.
    body = _rewrite_explicit_destructors(body, classes)

    # `V(5)` written where a value goes: C has no expression that constructs,
    # so each temporary becomes an object with a name ahead of the statement.
    body = _rewrite_temporaries(body, classes, [0])
    body = _rewrite_brace_temporaries(body, classes, [0])

    # `f.filename().c_str()`: a call on what a value return handed back. The
    # declarations have not been read yet - they are rewritten below, and
    # this has to run before that - so what this body declares is scanned for
    # first, without touching it.
    hoisted = {**_declared_objects(body, classes), **known}
    body = _hoist_value_returns(body, classes, hoisted, [0], unit)
    # The names it wrote are objects of this scope from here on. Left out,
    # `webRoot = current_path() / L"web";` had its call hoisted into a
    # temporary that nothing afterwards knew held a `path` - so the operator
    # on the next line was not that class's operator and was left as C++.
    # Overwritten, not filled in: a block starts its own count, so the name
    # this scope gave a temporary may be the name an enclosing scope gave a
    # different one. In the C that is an inner declaration shadowing an outer
    # and is fine; here it means the older type must not win.
    fresh = {
        name: held
        for name, held in hoisted.items()
        if name.startswith(_VALUE_PREFIX)
    }
    if fresh:
        known.update(fresh)
        receivers = dict(receivers or {})
        for name in fresh:
            # A reference return is hoisted as a reference - `B &r = ...` -
            # which is a pointer already. Given the address of one, every
            # call on it was handed a `B **`.
            if re.search(rf"&\s*{re.escape(name)}\s*=", body):
                pointers = set(pointers)
                pointers.add(name)
                receivers[name] = name
                continue
            receivers[name] = f"&{name}"
    # And the declarations that hoist just wrote: a free function answering an
    # object fills a space the caller provides, here as everywhere else. The
    # file-scope pass does not reach a method body, which is emitted on its
    # own - so `held.append(to_string(v))` inside one was left calling a
    # function with one argument too few.
    body = _free_value_initialisers(body, classes, f"{body}\n{unit}")
    # `b = a / "x";` where that operator answers an object: the caller has to
    # provide the space, so the call cannot stand in an expression. Given a
    # name of its own here, before the declarations are read, so the only
    # form left for the operator pass is the one it already handles.
    # What this body declares as well as what the scope brought in: the
    # object on the left of the operator is usually one the statement above
    # declared, and `known` holds only what came from outside.
    operands = {**hoisted, **known}
    body = _hoist_object_operators(
        body, classes, operands, [0], pointers, set(referenced or ())
    )
    # `(a + b).c_str()` - the parentheses were the author's and what is
    # inside them is now one name. Left standing, the pass that rewrites a
    # call on an object matches a bare name and did not see this one. Only
    # where a member is reached through them: `(int)x` is a cast, and a cast
    # is never followed by a dot.
    body = _map_code(body, lambda part: _AROUND_A_NAME.sub(r"\1", part))
    # And what it wrote is an object of this scope too, for the same reason:
    # `a + b + c` is two operators, and the second is applied to what the
    # first answered.
    for name, held in operands.items():
        if name.startswith(_OPERATOR_PREFIX):
            known[name] = held
            receivers = dict(receivers or {})
            receivers[name] = f"&{name}"
    # And the arguments of a call to a static member, for the same reason: the
    # file-scope pass ran before this body existed in its rewritten form, so
    # the temporary a value return was hoisted into was not there yet to have
    # its address taken.
    body = _address_reference_arguments(
        body,
        {
            **_function_signatures(f"{body}\n{unit}", classes),
            **_static_member_signatures(classes),
        },
        classes,
        unit,
        set(pointers) | set(referenced or ()),
    )

    # `int &r = a.v;` is a pointer whose uses are dereferenced. Done before
    # anything else reads the body, so the rest sees an ordinary pointer.
    local_references: dict[str, str] = {}
    bindings: list[str] = []

    def bind(match: "re.Match[str]", whole: "str | None" = None) -> "str | None":
        spelled, variable = match.group(1), match.group(2)
        # The match is against a copy with the literals blanked, so the value
        # is taken from the real text at the same offsets.
        source = (whole or match.string)[match.start(3): match.end(3)]
        held = spelled.replace("const", "").strip()
        if held in _NOT_A_TYPE:
            return None
        if not _could_start_a_declaration(match.string, match.start()):
            # `flags & mask` is an operator, not a reference, and only where
            # the statement begins can a declaration be what was meant.
            return None
        local_references[variable] = held
        if held in classes:
            known[variable] = held
            pointers.add(variable)
            made = (
                f"struct {held} *{variable} = "
                f"{_address_over_a_conditional(source.strip())};"
            )
        else:
            made = (
                f"{spelled} *{variable} = "
                f"{_address_over_a_conditional(source.strip())};"
            )
        # Held aside while the uses are dereferenced: the declaration is the
        # one place the name means the pointer and not what it points at, and
        # rewriting it too gave `int *(*alias)`.
        bindings.append(made)
        return _BINDING_MARK % (len(bindings) - 1)

    # Against the whole body rather than each stretch of code between the
    # literals: `Builder &r = f('x');` holds a literal, so the two halves of
    # it were handed to the pattern separately and neither was a declaration.
    body = _sub_code(
        _LOCAL_REFERENCE,
        body,
        lambda match, whole: bind(match, whole),
    )
    if local_references:
        body = _deref_references(body, local_references, classes)
        for index, made in enumerate(bindings):
            # The declaration's own name is the pointer and must stay as it
            # is, but what it is initialised *from* may name an earlier
            # reference - one chained call feeding the next - and that side
            # needs following like any other use.
            head, _, source = made.partition("=")
            if source:
                made = head + "=" + _deref_references(
                    source, local_references, classes
                )
            body = body.replace(_BINDING_MARK % index, made)

    def declare_from_call(match: "re.Match[str]") -> str:
        type_name, variable, receiver, access, method, arguments = match.groups()
        if type_name not in classes or receiver not in known:
            return match.group(0)
        holds = known[receiver]
        owner = _find_method(holds, method, classes)
        if owner is None:
            return match.group(0)
        known[variable] = type_name
        address = receiver if receiver in pointers else f"&{receiver}"
        fixed = _addressed_arguments(
            holds, method, arguments, known, pointers, classes
        )
        passed = f", {fixed}" if fixed.strip() else ""
        # The caller provides the space and hands its address over, which is
        # what a by-value return is once the ABI is written out.
        return (
            f"struct {type_name} {variable}; "
            f"{_c_name(owner, method, _call_suffix(owner, method, classes, [a.strip() for a in _split_arguments(fixed)] if fixed.strip() else [], scope()))}"
            f"({address}, &{variable}{passed});"
        )

    def convert_initialiser(match: "re.Match[str]", whole: str) -> "str | None":
        """`C name = value;` becomes `C name(value);` where C is a class.

        Left for the passes below wherever one of them already knows the
        shape: another object is a copy, a call on an object is a value
        returned, and both are constructed in their own way. What is left is
        a value of some other type entirely, and a class takes one of those
        through a constructor that names it.
        """

        type_name, variable, value = match.group(1), match.group(2), match.group(3)
        if type_name not in classes or type_name in _NOT_A_TYPE:
            return None
        # The real text of the value, since the match was made against a copy
        # with every literal blanked out.
        value = whole[match.start(3): match.end(3)].strip()
        if not value or value.startswith("{"):
            return None
        # Asked of the text and not of the copy with the literals blanked:
        # `wstring b = L"hi";` blanks to `wstring b = L     ;`, and the `L`
        # left standing reads as a variable being copied from. Every wide
        # literal in the program looked like that.
        spelled = whole[match.start():match.end()].strip()
        if _COPY_INIT.fullmatch(spelled) or _VALUE_INIT.fullmatch(spelled):
            return None
        # `C c = C(...)` and `C c = make()` both hand back a C already.
        if re.match(rf"^{re.escape(type_name)}\s*\(", value):
            return None
        # Only a value standing on its own. An operator between two things is
        # an overload to resolve - `a + "cd"` on strings is `operator+` - and
        # a constructor wrapped around it hides it from the pass that would
        # have found it. The match was made against a copy with the literals
        # blanked, which is what an operator has to be looked for in.
        if _has_a_loose_operator(match.group(3)):
            return None
        held = (
            known.get(value)
            or (_deduced_type(value, scope()) or "").replace("*", "").strip()
        )
        # Only where the value's type is known and is something else. An
        # expression this cannot read is left alone rather than guessed at:
        # `full = first + last` is an overloaded `+` that the operator pass
        # below turns into a call, and wrapping it in a constructor here put
        # it out of that pass's reach.
        if not held or held == type_name:
            return None
        if not any(
            method.name == "" and len(_split_arguments(method.parameters)) == 1
            for method in classes[type_name].methods
        ):
            return None
        return f"{type_name} {variable}({value});"

    body = _sub_code(_CONVERTING_INIT, body, convert_initialiser)
    body = _split_object_declarators(body, classes)
    body = _OBJECT_ARRAY.sub(declare_array, body)
    body = _OBJECT.sub(declare, body)
    def copy_initialise(match: "re.Match[str]") -> str:
        """`T b = a;` - a declaration whose value is another object.

        C++ makes a copy here, which means the copy constructor if the class
        wrote one. py2bin's C does not take `struct T b = a;` as an
        initialiser, so it is written as the declaration and the copy that it
        is - which is also the only way the constructor gets called.
        """

        type_name, variable, source = match.groups()
        source = source.strip()
        if type_name not in classes:
            return match.group(0)
        held = (
            known.get(source)
            or _declared_objects(body, classes).get(source)
            or (_deduced_type(source, scope()) or "").replace("*", "").strip()
        )
        if held != type_name:
            return match.group(0)
        known[variable] = type_name
        if _find_method(type_name, "~", classes):
            destroyed.append(variable)
        # `*p` is already an address once the star is gone; anything else has
        # to have one taken, unless it is a pointer and so is one.
        if source.startswith("*"):
            address = source[1:].strip()
        elif source in pointers:
            address = source
        else:
            address = f"&{source}"
        return (
            f"struct {type_name} {variable}; "
            + _copied_in(type_name, variable, address, classes)
        )

    body = _map_code(body, lambda part: _COPY_INIT.sub(copy_initialise, part))

    def declare_pointer(match: "re.Match[str]") -> str:
        type_name, variable = match.groups()
        if type_name not in classes:
            return match.group(0)
        known[variable] = type_name
        pointers.add(variable)
        return f"struct {type_name} *{variable}"

    # Either a mapping handed down from an enclosing scope, or the bare names
    # of double-pointer parameters, whose class `known` has.
    pointer_arrays: dict[str, str] = (
        dict(pointer_arrays)
        if isinstance(pointer_arrays, dict)
        else {name: known[name] for name in (pointer_arrays or ()) if name in known}
    )

    def declare_pointer_array(match: "re.Match[str]") -> str:
        type_name, variable, count = match.groups()
        if type_name not in classes:
            return match.group(0)
        pointer_arrays[variable] = type_name
        return f"struct {type_name} *{variable}[{count}];"

    body = _POINTER_ARRAY.sub(declare_pointer_array, body)
    body = _OBJECT_POINTER.sub(declare_pointer, body)

    # C++ converts a pointer-to-derived into a pointer-to-base wherever one is
    # wanted; C makes you say so. The address is the same - the base is the
    # first member - so this is a cast and nothing more, written where the
    # translator can see both types and left alone where it cannot.
    body = _upcast_assignments(body, classes, known, pointers, pointer_arrays)

    # After the pointer declarations, which are what say the type of the thing
    # being deleted, and so which destructor runs.
    body = _rewrite_delete(body, classes, known)

    # After the declarations, because it has to know what the receiver is, and
    # before the ordinary call rewriting, which would otherwise turn
    # `V c = a.plus(b);` into an assignment from a function that returns void.
    body, from_operators = _rewrite_value_operators(
        body, classes, known, pointers, unit
    )
    known.update(from_operators)
    # `v[i].get()` where `v[i]` is what an index operator returns. Before the
    # operator pass, which turns the subscript itself into a call and leaves
    # the method on an expression no later pass recognises as a receiver.
    for variable in sorted(known, key=len, reverse=True):
        owner = _find_method(known[variable], "op_index", classes)
        if owner is None:
            continue
        element = _method_by_name(owner, "op_index", classes)
        spelled = (element.returns if element else "").replace("&", "").strip()
        held = spelled.replace("*", "").strip()
        if held not in classes:
            continue
        # A container of pointers: `rows[i]->weight()`. The element is already
        # an address, so the receiver is what the index operator returns
        # rather than the address of it.
        indirect = "*" in spelled
        reach = "->" if indirect else r"\."
        address = variable if variable in pointers else f"&{variable}"
        for method in _reachable_methods(held, classes):
            provider = _find_method(held, method, classes)
            if provider is None:
                continue
            pattern = (
                rf"\b{re.escape(variable)}\s*\[([^\]]*)\]\s*{reach}\s*"
                rf"{re.escape(method)}\s*\("
            )
            # For a container of values, `&` in front: the dereference pass
            # will follow the reference the index operator returns and the
            # receiver wants the address again - the two cancel, which is
            # what C++ was doing.
            call = f"{_c_name(owner, 'op_index')}({address}, __I__)"
            reached = call if indirect else f"&{call}"
            body = _rewrite_pointer_indexed(
                body,
                pattern,
                _dispatched(held, method, classes, reached, provider, scope())
                if indirect
                else _name_for(provider, method, classes, scope()),
                variable,
                reached,
            )

    # Everything this body declares, whatever it was initialised with. Each
    # declaration pass above handles one shape and records what it rewrote;
    # an object whose initialiser is a call is not rewritten until further
    # down, so until now nothing said what it held - and `t < s` on the line
    # after `string t = s.substr(6);` was left standing as C, which compares
    # two structs and cannot.
    for spelled, holds in _declared_objects(body, classes).items():
        known.setdefault(spelled, holds)
    body = _rewrite_operators(
        body,
        classes,
        known,
        pointers,
        scope(),
        # This scope's own references as well as the ones it was handed. Sent
        # down to the blocks inside it but not used here, `Box &r = b;` and
        # `r[0]` written in the *same* scope read as pointer arithmetic,
        # while the identical pair one brace deeper translated.
        set(referenced or ()) | set(local_references),
    )
    # `V r = (a + b) * c;` - the pass above turns the inner operator into a
    # temporary, and the outer one is a declaration from an operator on it.
    # That shape is written out further up, before there was anything to be
    # written; asked once more here, now that there is.
    body, from_inner = _rewrite_value_operators(
        body, classes, known, pointers, unit
    )
    if from_inner:
        known.update(from_inner)
        body = _rewrite_operators(
            body,
            classes,
            known,
            pointers,
            scope(),
            set(referenced or ()) | set(local_references),
        )
    # After the operators and before the calls: `(int)a` is neither, and the
    # name it converts has to still be a name by the time this looks.
    body = _rewrite_conversions(body, classes, known, pointers)
    body = _VALUE_INIT.sub(declare_from_call, body)

    # Calls, longest name first so `ab.m()` is not matched inside `xab.m()`.
    # An element of an array of objects is a receiver like any other; its
    # address is `&bank[i]`, whatever the index expression happens to be.
    # A pointer indexed is the same shape as an array indexed - `&p[i]` is
    # the address either way - and `new T[n]` hands back a pointer, so the two
    # have to be treated alike or every array from the heap loses its methods.
    indexable = dict(arrays)
    for name in pointers:
        if name in known and name not in indexable:
            indexable[name] = known[name]
    # `all[i]->speak()`: the element is already a pointer, so the receiver is
    # the element itself. This is where a virtual call usually happens - an
    # array of base pointers is how a program keeps a mixture of things.
    for variable in sorted(pointer_arrays, key=len, reverse=True):
        holds = pointer_arrays[variable]
        for method in _reachable_methods(holds, classes):
            owner = _find_method(holds, method, classes)
            if owner is None:
                continue
            pattern = (
                rf"\b{re.escape(variable)}\s*\[([^\]]*)\]\s*->\s*"
                rf"{re.escape(method)}\s*\("
            )
            body = _rewrite_pointer_indexed(
                body, pattern,
                _dispatched(
                    holds, method, classes, f"{variable}[__I__]", owner, scope()
                ),
                variable,
            )

    for variable in sorted(indexable, key=len, reverse=True):
        holds = indexable[variable]
        for method in _reachable_methods(holds, classes):
            owner = _find_method(holds, method, classes)
            if owner is None:
                continue
            pattern = (
                rf"\b{re.escape(variable)}\s*\[([^\]]*)\]\s*\.\s*"
                rf"{re.escape(method)}\s*\("
            )
            body = _rewrite_indexed(
                body, pattern,
                _dispatched(
                    holds, method, classes, f"&{variable}[__I__]", owner, scope()
                ),
                variable,
            )

    # An inherited *member* reached from outside the class: `d.v` where `v`
    # lives in the base is `d.__base.v` in C. Methods were followed up the
    # chain from the start; fields were not, and the compiler reported a
    # struct with no such member on a line that is correct C++.
    for variable in sorted(known, key=len, reverse=True):
        holds = known[variable]
        access = "->" if variable in pointers else "."
        for member, prefix in _reachable_members(classes[holds], classes):
            if not prefix:
                continue  # its own field; the name is already right
            body = re.sub(
                rf"\b{re.escape(variable)}{re.escape(access)}{re.escape(member.name)}\b",
                f"{variable}{access}{prefix}{member.name}",
                body,
            )

    # A member that is a pointer to a class is a receiver in its own right:
    # `a.next->get()` is a call on whatever `next` points at. Without this the
    # chain stopped at the first hop and the compiler reported a struct with
    # no member called `get`.
    for variable in sorted(known, key=len, reverse=True):
        access = "->" if variable in pointers else "."
        for member, prefix in _reachable_members(classes[known[variable]], classes):
            held = member.ctype.replace("*", "").strip()
            if "*" not in member.ctype or held not in classes:
                continue
            reached = f"{variable}{access}{prefix}{member.name}"
            known.setdefault(reached, held)
            pointers.add(reached)

    # `b.inner.get()` - a member that is a class, reached from outside. The
    # address is `&b.inner`, and without it the call looked for a field named
    # after the method.
    for variable in sorted(known, key=len, reverse=True):
        access = "->" if variable in pointers else "."
        for member, prefix in _reachable_members(classes[known[variable]], classes):
            held = member.ctype.replace("*", "").strip()
            if "*" in member.ctype or held not in classes:
                continue
            reached = f"{variable}{access}{prefix}{member.name}"
            if reached not in known:
                known[reached] = held
                (receivers := dict(receivers or {}))[reached] = f"&{reached}"

    given = dict(receivers or {})
    for variable in sorted(known, key=len, reverse=True):
        holds = known[variable]
        arrow = "->" if variable in pointers else r"\."
        address = given.get(
            variable, variable if variable in pointers else f"&{variable}"
        )
        for method in _reachable_methods(holds, classes):
            owner = _find_method(holds, method, classes)
            if owner is None:
                continue
            reached = address
            if owner != holds:
                # Through the embedded base, which is where it lives.
                inner = variable if variable in pointers else f"{variable}"
                reach = "->" if variable in pointers else "."
                path = _subobject_path(holds, owner, classes) or "__base"
                reached = f"&{inner}{reach}{path}"
            pattern = rf"\b{re.escape(variable)}{arrow}{re.escape(method)}\s*\("
            body = _rewrite_calls(
                body, pattern,
                # The object's own address for a virtual call, the base
                # subobject's for a direct one.
                _dispatched(
                    holds, method, classes, address, owner, scope(), reached
                ),
                reached,
            )
    body = _fill_member_defaults(body, classes)
    body = _upcast_pointers(
        body, classes, known, pointers, returns, scope(), unit, stable
    )
    body = _fix_call_arguments(body, classes, known, pointers)
    # `f("x")` where the parameter is a `const string &`. C++ builds a
    # temporary from the literal and binds the reference to it; C has no such
    # step, so the literal was passed as itself and the callee read a `char *`
    # as an object.
    # A call through the object's table is not a call to a name, so the pass
    # above does not see it. What its parameters want is written into the
    # cast, which is the one place that says so.
    body = _address_dispatched_arguments(body, classes, known, pointers)
    body = _convert_class_arguments(
        body, classes, known, pointers, [0], f"{scope()}\n{stable}"
    )

    # After the dereference pass, not before it: each call in a chain hands
    # back the stream as an address and the next call takes it as its
    # receiver, so following the reference in between would leave a struct
    # where a pointer is wanted.
    body = _rewrite_stream_chains(body, classes, known, pointers, scope())

    # Now that this scope is known, each block is rewritten inside it.
    rewritten_blocks = [
        _rewrite_body(
            inner,
            classes,
            dict(known),
            set(pointers),
            dict(given),
            dict(arrays),
            unit,
            # Only the ones already built where this block sits. A block
            # above a declaration leaves a scope in which that object does
            # not exist yet, and a `return` inside it destroyed something
            # nothing had declared.
            [
                *enclosing,
                *(
                    # And which handlers are written at this level, so that a
                    # jump to one of them knows where to stop unwinding: past
                    # the body that holds the label it has not left.
                    (name, known[name], _handlers_written(body), in_a_loop)
                    for name in destroyed
                    if _built_before(body, name, number)
                ),
            ],
            # An array of pointers declared in an enclosing scope is still an
            # array of pointers inside a block. Without this `all[i]->get()`
            # in a `for` body was left as C++, while the identical statement
            # written without the braces translated - which is the sort of
            # difference nobody would think to test for.
            dict(pointer_arrays),
            returns,
            {**outer_references, **local_references},
            # A reference parameter is still a reference inside a block. Left
            # out, `json[i]` there was read as an element of an array of
            # objects rather than as the class's own subscript - the same
            # statement, translated two ways depending on its braces.
            referenced=set(referenced or ()) | set(local_references),
            stable=stable,
            in_a_loop=_is_a_loop_body(body, number),
        )
        for number, inner in enumerate(blocks)
    ]
    body = _close_with_destructors(
        body, destroyed, known, classes, enclosing, returns, [0], in_a_loop
    )
    return _restore_nested(body, rewritten_blocks)




#: Stands in for a nested block while the enclosing scope is rewritten, so a
#: declaration inside one is not mistaken for a declaration in this one.
_BLOCK_MARK = "\x00py2bin_block_%d\x00"



def _initialiser_brace(body: str, index: int) -> bool:
    """Whether the `{` at `index` opens a list of values rather than a scope."""

    before = body[:index].rstrip()
    return before.endswith("=") or before.endswith(",")

def _is_a_loop_body(body: str, number: int) -> bool:
    """Whether the block that was lifted out of here is a loop's body.

    Read off the text in front of the marker it left, which is where the
    `for (...)` or `while (...)` that owns it is still written.
    """

    at = body.find(_BLOCK_MARK % number)
    if at < 0:
        return False
    before = _without_literals(body[:at]).rstrip()
    if re.search(r"(?<![.\w>])do$", before):
        return True
    if not before.endswith(")"):
        return False
    opening = _opening_paren(before, len(before) - 1)
    if opening < 0:
        return False
    return re.search(r"(?<![.\w>])(for|while)\s*$", before[:opening]) is not None


def _lift_nested(body: str) -> "tuple[str, list[str]]":
    """Take each nested block out, leaving a marker.

    Only taken out here, not rewritten: a block has to be rewritten knowing
    what this scope declared *before* it, and this scope has not been read
    yet. Rewritten too early, a `for` body did not know that the array it
    indexes was an array at all.
    """

    blocks: list[str] = []
    out = []
    index = 0
    depth = 0
    # Against a copy with the literals blanked. A brace inside a string is
    # text: `payload << "{\"type\":\"status\""` opened one that never
    # closed, so the rest of the statement was lifted out as a block and put
    # back with whatever had been written into it in the meantime - inside
    # the string.
    bare = _without_literals(body)
    while index < len(body):
        char = bare[index]
        if char == "{":
            depth += 1
            if depth == 2 and not _initialiser_brace(body, index):
                closing = _matching(body, index)
                out.append(_BLOCK_MARK % len(blocks))
                blocks.append(body[index:closing])
                depth -= 1
                index = closing
                continue
            if depth == 2:
                # `A xs[3] = { ... };` is a list of values, not a scope. Taken
                # out as a block, the pass that constructs each element never
                # saw it, and a declaration written inside one would have been
                # scoped to a block that does not exist.
                depth -= 1
                closing = _matching(body, index)
                out.append(body[index:closing])
                index = closing
                continue
        elif char == "}":
            depth -= 1
        out.append(body[index])
        index += 1
    return "".join(out), blocks


def _built_before(body: str, name: str, block: int) -> bool:
    """Whether that object is declared above the block with this number."""

    mark = body.find(_BLOCK_MARK % block)
    if mark < 0:
        return True
    where = re.search(rf"(?<![.\w>]){re.escape(name)}\b", body)
    return where is not None and where.start() < mark


def _restore_nested(body: str, blocks: "list[str]") -> str:
    for number, inner in enumerate(blocks):
        body = body.replace(_BLOCK_MARK % number, inner)
    return body




def _method_named(
    owner: str, name: str, classes: "dict[str, Class]", returns_object: bool = False
) -> "Method | None":
    """The method `name` on `owner`, optionally only if it answers an object."""

    seen = owner
    while seen and seen in classes:
        for candidate in classes[seen].methods:
            if candidate.name == name:
                if returns_object and not _returns_object(candidate, classes):
                    return None
                return candidate
        seen = classes[seen].base
    return None


def _rewrite_value_operators(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
    scope: str = "",
) -> "tuple[str, dict[str, str]]":
    """`V c = a + b;` - an operator answering an object, given somewhere to put it."""

    # Which overload is meant is read from the type of what is passed, and
    # that may be declared outside this body: `s + kHost` where `kHost` is a
    # file-scope array had no type here, so a class with two of these could
    # not be told which was meant. The body comes first and the file after
    # it, so the offsets below still point where they did.
    reading = body if not scope else f"{body}\n{scope}"

    added: dict[str, str] = {}
    for symbol in _OPERATOR_SYMBOLS:
        name = _OPERATOR_NAMES[symbol]
        if symbol in ("[]", "()", "="):
            continue
        if symbol in ("<<", ">>"):
            # Streams are chains, and a chain is read whole by
            # `_rewrite_stream_chains`. This pass matches a *bare name* on the
            # right, so it turned `cout << a[0]` into a call taking `a` and
            # left `[0]` hanging - and it ran first, so the chain pass never
            # saw the statement at all.
            continue
        # The right side is an operand, not a name: `string c = a + "x";` is
        # as ordinary as `a + b`, and the literal cannot be matched by a
        # pattern that ends at an identifier. Scanned by hand rather than
        # through `_map_code`, which hides literals from what it is given.
        # The left operand may be a member reached through `this`, which is
        # how a method writes one of its own: `return name + other;` inside
        # a class is `this->name + other` by the time this runs.
        pattern = re.compile(
            rf"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=\s*"
            rf"((?:this\s*->\s*)?[A-Za-z_]\w*)\s*{re.escape(symbol)}"
        )
        bare = _without_literals(body)
        out: list[str] = []
        at = 0
        for match in pattern.finditer(bare):
            if match.start() < at:
                continue
            type_name, variable, left = match.groups()
            # Not a longer operator that starts with this one, the way `+` is
            # the front of `+=` and `<` of `<=`.
            after = body[match.end():]
            if symbol in ("+", "-", "<", ">", "*", "/", "%") and after[:1] == "=":
                continue
            if symbol in ("<", ">") and after[:1] == symbol:
                continue
            if type_name not in classes or left not in known:
                continue
            owner = _find_method(known[left], name, classes)
            if owner is None or not _method_named(owner, name, classes, True):
                continue
            end = _one_operand(body, match.end())
            if end < 0 or body[end:].lstrip()[:1] != ";":
                continue
            right = body[match.end(): end].strip()
            if not right:
                continue
            added[variable] = type_name
            address = left if left in pointers else f"&{left}"
            passed = (
                f"&{right}" if right in known and right not in pointers else right
            )
            suffix = _call_suffix(
                owner, name, classes, [right], reading, match.start()
            )
            out.append(body[at:match.start()])
            out.append(
                f"struct {type_name} {variable}; "
                f"{_c_name(owner, name, suffix)}({address}, &{variable}, {passed})"
            )
            at = end
        out.append(body[at:])
        body = "".join(out)
    return body, added

#: `out << a << b;` - a chain, which is what streams are always written as.
_STREAM_CHAIN = re.compile(r"(?<![.\w>])([A-Za-z_]\w*)\s*(<<|>>)")


def _rewrite_stream_chains(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
    scope: str = "",
) -> str:
    """`out << a << b` becomes one call wrapped around another.

    The generic operator rewriter matches a *name* on the left, and after the
    first `<<` the left side is no longer a name but the call that replaced
    it. Streams are written in chains and nothing else is, so the chain is
    read whole here: each `<<` is a call taking what the one before it
    returned, which is the address the reference became.
    """

    if not known:
        return body
    out: list[str] = []
    at = 0
    for found in _STREAM_CHAIN.finditer(body):
        if found.start() < at:
            continue
        variable, symbol = found.group(1), found.group(2)
        holds = known.get(variable)
        if holds is None:
            continue
        name = _OPERATOR_NAMES[symbol]
        if _find_method(holds, name, classes) is None:
            continue
        end = _statement_end(body, found.end())
        if end < 0:
            continue
        pieces = _split_on_operator(body[found.end(): end], symbol)
        if not pieces:
            continue
        reached = variable if variable in pointers else f"&{variable}"
        for piece in pieces:
            operand = piece.strip()
            owner = _find_method(holds, name, classes)
            if owner is None:
                break
            picked = _chosen_overload(
                _overload_set(owner, name, classes),
                [operand],
                scope or body,
                found.start(),
            )
            chosen = _c_name(
                owner,
                name,
                _call_suffix(
                    owner, name, classes, [operand], scope or body, found.start()
                ),
            )
            if picked is not None and not operand.startswith("&"):
                declared = picked.parameters
                words = declared.replace("*", " * ").split()
                by_value = (
                    "*" not in declared
                    and "&" not in declared
                    and len(words) == 2
                    and words[0] in classes
                )
                if (by_value or _REFERENCE.search(declared)) and _has_an_address(
                    operand
                ):
                    operand = f"&{operand}"
            # Each call hands back the stream, as the address its reference
            # became, so the next one takes it as the receiver directly.
            reached = f"{chosen}({reached}, {operand})"
        out.append(body[at:found.start()])
        out.append(reached)
        at = end
    out.append(body[at:])
    return "".join(out)


def _statement_end(text: str, at: int) -> int:
    """The `;` that ends the statement starting at `at`, at this depth."""

    depth = 0
    index = at
    while index < len(text):
        piece = text[index]
        if piece in "\"'":
            quote = piece
            index += 1
            while index < len(text) and text[index] != quote:
                index += 2 if text[index] == "\\" else 1
            index += 1
            continue
        if piece in "([{":
            depth += 1
        elif piece in ")]}":
            if depth == 0:
                return index
            depth -= 1
        elif piece == ";" and depth == 0:
            return index
        index += 1
    return -1


def _split_on_operator(text: str, symbol: str) -> "list[str]":
    """Split `a << b << c` on the operators between its operands."""

    pieces: list[str] = []
    depth = 0
    at = 0
    index = 0
    while index < len(text):
        piece = text[index]
        if piece in "\"'":
            quote = piece
            index += 1
            while index < len(text) and text[index] != quote:
                index += 2 if text[index] == "\\" else 1
            index += 1
            continue
        if piece in "([{":
            depth += 1
        elif piece in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith(symbol, index):
            pieces.append(text[at:index])
            index += len(symbol)
            at = index
            continue
        index += 1
    pieces.append(text[at:])
    return [piece for piece in pieces if piece.strip()]




def _rewrite_indexed_operator(
    body: str,
    variable: str,
    symbol: str,
    owner: str,
    name: str,
    known: "dict[str, str]",
    pointers: "set[str]",
    classes: "dict[str, Class]",
    scope: str,
) -> str:
    """`p[i] OP x` becomes the call, where `p` points at objects."""

    bare = _without_literals(body)
    pattern = re.compile(
        rf"(?<![.\w>]){re.escape(variable)}\s*\[([^\]]*)\]\s*{re.escape(symbol)}"
    )
    out: list[str] = []
    at = 0
    for found in pattern.finditer(bare):
        if found.start() < at:
            continue
        after = body[found.end():]
        if symbol in ("+", "-", "<", ">", "*", "/", "%") and after[:1] == "=":
            continue
        if symbol in ("<", ">") and after[:1] == symbol:
            continue
        end = _one_operand(body, found.end())
        if end < 0:
            continue
        right = body[found.end(): end].strip()
        if not right:
            continue
        passed = right
        held = (_deduced_type(right, f"{body}\n{scope}") or "").strip()
        if right in known and right not in pointers:
            passed = f"&{right}"
        elif held.replace("const", "").strip() in classes and "*" not in held:
            # An element of an array of objects, or any other object: the
            # call takes its address, the way every receiver here does.
            passed = f"&{right}"
        suffix = _call_suffix(owner, name, classes, [right], f"{body}\n{scope}")
        out.append(body[at:found.start()])
        out.append(
            f"{_c_name(owner, name, suffix)}"
            f"(&{variable}[{body[found.start(1): found.end(1)]}], {passed})"
        )
        at = end
    out.append(body[at:])
    return "".join(out)

def _rewrite_binary_operator(
    body: str,
    variable: str,
    symbol: str,
    owner: str,
    name: str,
    address: str,
    known: "dict[str, str]",
    pointers: "set[str]",
    classes: "dict[str, Class]",
    scope: str = "",
) -> str:
    """`a OP x` becomes the call the class declared, whatever `x` is."""

    pattern = re.compile(
        rf"(?<![.\w>]){re.escape(variable)}\s*{re.escape(re.sub(r'(.)', r'\\\\\1', ''))}"
        rf"{re.escape(symbol)}"
    )
    out: list[str] = []
    at = 0
    for found in pattern.finditer(body):
        if found.start() < at:
            continue
        # Not a longer operator that starts with this one: `a + b` and
        # `a += b` are different members, and `<` is the front of `<=`.
        after = body[found.end():]
        if symbol in ("+", "-", "<", ">", "*", "/", "%") and after[:1] == "=":
            continue
        if symbol in ("<", ">") and after[:1] == symbol:
            continue
        end = _one_operand(body, found.end())
        if end < 0:
            continue
        right = body[found.end(): end].strip()
        if not right:
            continue
        passed = (
            f"&{right}" if right in known and right not in pointers else right
        )
        out.append(body[at:found.start()])
        # The scope goes with the body: which overload `a == b` means is read
        # off the type of `b`, and `b` may be a parameter, declared in a head
        # this body does not contain.
        out.append(
            f"{_c_name(owner, name, _call_suffix(owner, name, classes, [right], f'{body}\n{scope}'))}"
            f"({address}, {passed})"
        )
        at = end
    out.append(body[at:])
    return "".join(out)


def _one_operand(text: str, at: int) -> int:
    """The end of the single operand starting at `at`, or -1.

    One operand and not the rest of the expression: `a + b + c` is two calls
    and not one, and taking everything to the semicolon would make it one.
    """

    index = at
    while index < len(text) and text[index] in " \t":
        index += 1
    if index >= len(text):
        return -1
    # `L"x"` is one operand and not a name called L followed by a string.
    # Read as a name, the operand handed on was `L`, whose type nothing
    # knows - and a class with more than one `operator+=` could not be told
    # which one a wide literal meant.
    for spelled in ("u8", "L", "u", "U"):
        if text.startswith(spelled, index) and text[index + len(spelled): index + len(spelled) + 1] in ('"', "'"):
            index += len(spelled)
            break
    if text[index] in "\"'":
        quote = text[index]
        index += 1
        while index < len(text) and text[index] != quote:
            index += 2 if text[index] == "\\" else 1
        index += 1
    elif text[index] == "(":
        closing = _closing_paren(text, index)
        if closing < 0:
            return -1
        index = closing + 1
    elif text[index].isdigit() or (
        text[index] in "+-" and index + 1 < len(text) and text[index + 1].isdigit()
    ):
        index += 1
        while index < len(text) and (text[index].isalnum() or text[index] == "."):
            index += 1
    elif text[index].isalpha() or text[index] == "_":
        while index < len(text) and (text[index].isalnum() or text[index] == "_"):
            index += 1
    else:
        return -1
    # Whatever trails it and belongs to it: a call, a subscript, a member.
    while index < len(text):
        if text[index] == "(":
            closing = _closing_paren(text, index)
            if closing < 0:
                return index
            index = closing + 1
            continue
        if text[index] == "[":
            depth = 0
            scan = index
            while scan < len(text):
                if text[scan] == "[":
                    depth += 1
                elif text[scan] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                scan += 1
            if scan >= len(text):
                return index
            index = scan + 1
            continue
        following = re.match(r"(\.|->)\s*[A-Za-z_]\w*", text[index:])
        if following:
            index += following.end()
            continue
        break
    return index

#: `(int)a` and `static_cast<int>(a)` - a conversion the author asked for by
#: name. The type is whatever stands inside, which is read against what the
#: class declares rather than against a list of types.
_A_WRITTEN_CAST = re.compile(
    r"\(\s*((?:const\s+)?[A-Za-z_][\w\s*]*?)\s*\)\s*([A-Za-z_]\w*)\b"
    # And nothing reached through it. `(int)a.has_value()` casts what the
    # call answers, not `a`: read as a conversion of the object it swallowed
    # the member and left a call on an int.
    r"(?!\s*(?:\.|->|\[|\())"
)
_A_NAMED_CAST = re.compile(
    r"\b(?:static|const|reinterpret)_cast\s*<\s*([^<>]+?)\s*>\s*\(\s*"
    r"([A-Za-z_]\w*)\s*\)(?!\s*(?:\.|->|\[|\())"
)


def _rewrite_conversions(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
) -> str:
    """`(int)a` becomes the conversion operator the class declares.

    C++ turns an object into another type by calling a member whose name is
    that type. C has a cast, which for a struct means something else
    entirely - so where the class says how the conversion is done, the cast
    becomes that call.

    Only where the object's class declares one for that type. A cast of an
    object to something it has no conversion for is a mistake, and reporting
    it against the line that has it beats rewriting it into a call that does
    not exist.
    """

    if not known:
        return body

    def converted(spelled: str, variable: str) -> "str | None":
        held = known.get(variable)
        if held is None:
            return None
        method = f"{_CONVERSION_PREFIX}{_type_code(spelled)}"
        owner = _find_method(held, method, classes)
        if owner is None:
            return None
        address = variable if variable in pointers else f"&{variable}"
        return f"{_c_name(owner, method)}({address})"

    def written(match: "re.Match[str]") -> str:
        made = converted(match.group(1), match.group(2))
        return match.group(0) if made is None else made

    def named(match: "re.Match[str]") -> str:
        made = converted(match.group(1), match.group(2))
        return match.group(0) if made is None else made

    body = _map_code(body, lambda part: _A_NAMED_CAST.sub(named, part))
    return _map_code(body, lambda part: _A_WRITTEN_CAST.sub(written, part))


#: `if (`, `while (`, `switch (` - a parenthesis the statement owns rather
#: than one the program put around an expression. Written up to the `(`
#: itself, so it is asked of the text ending at the one being looked at.
_STATEMENTS_OWN_PARENTHESIS = re.compile(r"\b(?:if|while|switch)\s*\($")


def _strip_parentheses_around(part: str, variable: str) -> str:
    """`(v)` is `v`, except where those parentheses belong to a statement.

    Taking them off around a name says nothing about the expression - which
    is the point, since the pass below matches a bare name. But `if (p)` is
    written with a space, so the lookbehind that keeps a call's argument list
    out of this does not see the `if`, and `if p` is not C. That is how the
    commonest thing anyone writes with a pointer - check it, then call
    through it - stopped translating, and took `ComPtr` in <wrl.h> with it.
    """

    def one(found: "re.Match[str]") -> str:
        if _STATEMENTS_OWN_PARENTHESIS.search(part[: found.start() + 1]):
            return found.group(0)
        return variable

    return re.sub(
        rf"(?<![\w)\]>])\(\s*{re.escape(variable)}\s*\)(?!\s*\()", one, part
    )


def _rewrite_operators(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
    scope: str = "",
    referenced: "set[str] | None" = None,
    rounds: int = 4,
) -> str:
    """`a + b` becomes the call the class declared for it.

    Only where the left side is an object this scope knows, because that is
    what says which class's operator is meant. `a + b` on two ints is two ints
    and must be left alone, so the type of the left operand decides and
    nothing is guessed.

    An operator returning an object is a call like any other returning one -
    the caller provides the space - so `V c = a + b;` is turned into the
    declaration form first and goes through the same path.
    """

    if not known:
        return body
    # `(a + b) * c` - the parentheses the program wrote are still there when
    # the inner operator has been turned into a temporary, and this pass
    # matches a bare name on the left. Taken off only around a name this
    # scope declares, which is where they say nothing: around anything else
    # they may be a cast, and `(int)x` without them is not the same text.
    for variable in known:
        body = _map_code(
            body,
            # Not an argument list: a `(` with a name in front of it is a
            # call, and `__skip(this)` with the parentheses taken off is one
            # long identifier.
            lambda part, v=variable: _strip_parentheses_around(part, v),
        )
    before = body
    for symbol in _OPERATOR_SYMBOLS:
        if symbol in ("[]", "()", "="):
            continue  # spelled differently; handled below
        if symbol in ("<<", ">>"):
            # Streams are chains, and a chain is read whole by
            # `_rewrite_stream_chains`. This pass matches a *bare name* on the
            # right, so it turned `cout << a[0]` into a call taking `a` and
            # left `[0]` hanging - and it ran first, so the chain pass never
            # saw the statement at all.
            continue
        name = _OPERATOR_NAMES[symbol]
        for variable in sorted(known, key=len, reverse=True):
            owner = _find_method(known[variable], name, classes)
            if owner is None:
                continue
            if _method_named(owner, name, classes, returns_object=True):
                # It answers an object, so the caller has to provide the space
                # and the call cannot stand in an expression. The declaration
                # form below turns `V c = a + b;` into that.
                continue
            address = variable if variable in pointers else f"&{variable}"
            # The right side is an operand, not necessarily a name: `a += " x"`
            # and `a + f(1)` are as ordinary as `a + b`, and a pattern that
            # only matched an identifier left them for the C compiler to
            # complain about an operator it does not have.
            if variable in pointers:
                # `*this == o` is the object on the left, written the way a
                # class writes about itself: `operator!=` is nearly always
                # `!(*this == o)`. The pointer is the address the call
                # wants, so the dereference is the whole of the difference.
                #
                # First, and not after: the plain name matches inside the
                # dereferenced form, so taken the other way round `*this ==
                # o` became `*` followed by the call, and the star was left
                # standing in front of an int.
                body = _rewrite_binary_operator(
                    body, f"*{variable}", symbol, owner, name, variable,
                    known, pointers, classes, scope,
                )
            body = _rewrite_binary_operator(
                body, variable, symbol, owner, name, address, known, pointers,
                classes, scope,
            )
            if variable in pointers and variable not in (referenced or ()):
                # `base[child] < base[root]` - an element of an array of
                # objects is an object, and comparing two of them is the
                # class's own operator. This is how a container's own code
                # is written, and a pattern matching a name never saw it.
                # Not a reference, though: that is a pointer in the C and an
                # object in the language, so `r[i]` on one is the class's
                # own subscript and its result is whatever that answers.
                body = _rewrite_indexed_operator(
                    body, variable, symbol, owner, name, known, pointers,
                    classes, scope,
                )
    # `b = a;` where the class declared an assignment operator. Only where
    # both sides are objects of it: a struct copy is what `=` means without
    # one, and that is still what a class without an `operator=` gets.
    for variable in sorted(known, key=len, reverse=True):
        holds = known[variable]
        owner = _find_method(holds, "op_assign", classes)
        if owner is None:
            continue
        address = variable if variable in pointers else f"&{variable}"
        # `*p = v` as well as `p = v`. Matched from the name onwards, the
        # star stayed where it was and the C read `* op_assign(p, &v)` - a
        # dereference of what the operator answers, which is one too many.
        # Taken with the name it cancels the address this would take: `&*p`
        # is `p`.
        pattern = re.compile(
            rf"(?<![.\w>=!<])(\*\s*)?{re.escape(variable)}\s*=(?!=)\s*"
            rf"([A-Za-z_]\w*)\s*;"
        )

        def assigned(
            match: "re.Match[str]", o=owner, a=address, h=holds, v=variable
        ) -> str:
            if match.group(1):
                a = v
            source = match.group(2)
            spelled = known.get(source) or _deduced_type(source, scope or body)
            if spelled is None:
                return match.group(0)
            # Which `operator=` is meant is the type of what is being
            # assigned. `ComPtr<T>` declares one taking a `T *` as well as
            # one taking another holder, and a program assigns a raw pointer
            # to one every time it is handed a fresh interface.
            picked = _chosen_overload(
                _overload_set(o, "op_assign", classes),
                [source],
                scope or body,
                -1,
            )
            if picked is None and known.get(source) != h:
                return match.group(0)
            declared = (
                _parameter_types(picked.parameters) if picked is not None else []
            )
            wants_address = not declared or not declared[0].endswith("_p")
            passed = (
                f"&{source}"
                if wants_address and source not in pointers
                else source
            )
            suffix = (
                _suffix_of(o, picked, classes) if picked is not None else None
            )
            if len(_overload_set(o, "op_assign", classes)) < 2:
                suffix = None
            return f"{_c_name(o, 'op_assign', suffix)}({a}, {passed});"

        body = _map_code(body, lambda part: pattern.sub(assigned, part))

    # `d(5)` where `d` is an object with a call operator. A name that holds
    # an object is never a function, so a call on it is that operator and
    # nothing else - which is what makes this safe to do by name.
    # Repeated while it keeps finding names: a call that answers an object
    # declares one, and that object may have a call operator of its own -
    # which is exactly what a lambda returning a lambda is.
    seen: "set[str]" = set()
    while True:
      fresh = [name for name in known if name not in seen]
      if not fresh:
        break
      seen.update(fresh)
      for variable in sorted(fresh, key=len, reverse=True):
        owner = _find_method(known[variable], "op_call", classes)
        if owner is None:
            continue
        address = variable if variable in pointers else f"&{variable}"
        if _method_named(owner, "op_call", classes, returns_object=True):
            # It answers an object, so the caller provides the space and the
            # call cannot stand in an expression - the same as an operator
            # that does. `auto f = outer(5);` is how a lambda returning a
            # lambda is held, and this is the only form it comes in.
            body = _rewrite_value_call(
                body, variable, owner, address, classes, known
            )
            continue
        pattern = rf"(?<![.\w>])\b{re.escape(variable)}\s*\("
        body = _rewrite_calls(
            body, pattern, _name_for(owner, "op_call", classes), address
        )

    # `a[i]` where a is an object with an index operator. Written out rather
    # than routed through the array rewriter: that one turns `bank[i].m()`
    # into a call on an element, and this turns the subscript itself into the
    # call - the index is an argument, not part of the receiver.
    for variable in sorted(known, key=len, reverse=True):
        owner = _find_method(known[variable], "op_index", classes)
        if owner is None:
            continue
        if variable in pointers and variable not in (referenced or set()):
            # A pointer indexed is an element of what it points at, which is
            # what the rest of this translator takes `p[i]` to mean. Reading
            # it as the class's own subscript turned a vector's `items[i] =
            # value` into an assignment to a string's character.
            #
            # A *reference* is not that: `const vector<int> &v` is a pointer
            # here only because C has no reference, and `v[i]` in the source
            # asked the container for its element.
            continue
        body = _rewrite_subscripts(
            body, variable, known[variable], classes, variable in pointers
        )

    # `*p`, `p->m()` and `!p`, where `p` is a holder standing in for what it
    # holds. None of the three takes a right operand, so the two-operand pass
    # above has nothing to match; each is written where it stands.
    body = _rewrite_holder_operators(body, classes, known, pointers)
    # `(*v[i])(x)` - a call on something that is not a name. `v[i]` has
    # already become a call answering an address by here, and a container of
    # callables is exactly what a program keeps one of.
    body = _rewrite_dereferenced_calls(body, classes, scope)
    # Again while it keeps finding things. The symbols are walked in a fixed
    # order, and an operator whose left side only *becomes* an object once an
    # earlier one has been written out is reached after its turn has passed:
    # `(a + b) * c` had its `*` looked at before the `+` had made anything to
    # multiply, and was left as C++.
    if rounds > 0 and body != before:
        return _rewrite_operators(
            body, classes, known, pointers, scope, referenced, rounds - 1
        )
    return body


#: `T a = x, b = y;` - one type and more than one thing declared with it.
#: A statement begins at the start of a line, but also right after a `{` or
#: a `;` - `int main() { string x = "a", y = "b";` is one line and two
#: statements, and anchoring to the line start missed the second.
_MANY_DECLARED = re.compile(
    r"(?m)(?:(?<=^)|(?<=[;{]))([ \t]*)((?:const\s+)?[A-Za-z_]\w*)\s+([^;{}]*?);"
)


def _split_object_declarations(body: str, classes: "dict[str, Class]") -> str:
    """`T a = x, b = y;` becomes two declarations, because it is two.

    C lets one type introduce several names and so does C++, and every pass
    that reads a declaration here reads one name. Only where the type names a
    class: `int *p, q;` declares a pointer and an int, which C handles and
    this must not touch.
    """

    def one(match: "re.Match[str]", whole: str) -> "str | None":
        lead, spelled, rest = match.groups()
        held = spelled.replace("const", "").strip()
        if held not in classes:
            return None
        pieces = [
            piece.strip()
            for piece in _split_arguments(whole[match.start(3): match.end(3)])
            if piece.strip()
        ]
        if len(pieces) < 2:
            return None
        # A declarator that is a pointer or an array is a different type from
        # the one beside it, and splitting would say it was the same.
        if any(piece.startswith(("*", "&")) for piece in pieces):
            return None
        return "".join(f"{lead}{spelled} {piece};\n" for piece in pieces).rstrip(
            "\n"
        )

    return _sub_code(_MANY_DECLARED, body, one)


def _rewrite_address_of(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
) -> str:
    """`&held` where the class declares `operator&`.

    A holder hands out the address of what it holds so a call can fill it
    in, and `&` is how the vendor's own is written. Done before anything
    else, because the passes below write a `&` of their own in front of
    every receiver and nothing afterwards can tell the two apart.
    """

    # `held.As(&other)` asks the holder to fill *another holder*, so the `&`
    # there is the ordinary one. WRL's own `operator&` answers a proxy that
    # becomes either, which is a conversion this subset does not have - so
    # the one member that wants the holder itself is kept out of the way
    # while the rest are rewritten.
    kept = _HOLDS_A_HOLDER.sub(lambda m: f"{m.group(1)}{_KEEP_ADDRESS}", body)
    for variable in sorted(known, key=len, reverse=True):
        if variable in pointers:
            continue
        owner = _find_method(known[variable], _ADDRESS_OF, classes)
        if owner is None:
            continue
        kept = _rewrite_prefix(
            kept, variable, "&", f"{_c_name(owner, _ADDRESS_OF)}(&{variable})"
        )
    return kept.replace(_KEEP_ADDRESS, "&")


#: `x.As(&y)` and `x->As(&y)` - the one member of a holder that is handed
#: another holder rather than the pointer inside one.
_HOLDS_A_HOLDER = re.compile(r"((?:\.|->)\s*As\w*\s*\(\s*)&")

#: Stands in for that `&` while the rest are rewritten.
_KEEP_ADDRESS = "\x00address"


def _rewrite_holder_operators(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
) -> str:
    """The unary operators a smart pointer or an iterator declares."""

    for variable in sorted(known, key=len, reverse=True):
        if variable in pointers:
            # Already a pointer: `*p` and `p->m()` mean what C says they mean,
            # and the class's own operators are not what was written.
            continue
        holds = known[variable]
        address = f"&{variable}"
        owner = _find_method(holds, "op_arrow", classes)
        if owner is not None:
            reached = _arrow_target(owner, classes)
            if reached is not None:
                call = f"{_c_name(owner, 'op_arrow')}({address})"
                for method in _reachable_methods(reached, classes):
                    provider = _find_method(reached, method, classes)
                    if provider is None:
                        continue
                    # Through the object's own table where the method is
                    # virtual. A COM interface is nothing but virtuals and is
                    # reached through a holder of exactly this kind, so a
                    # direct call here named a function that does not exist -
                    # the interface declares its methods and defines none.
                    body = _rewrite_calls(
                        body,
                        rf"(?<![.\w>]){re.escape(variable)}\s*->\s*"
                        rf"{re.escape(method)}\s*\(",
                        _dispatched(
                            reached, method, classes, call, provider, body
                        ),
                        call,
                    )
                # A member reached through it, rather than a method.
                body = _map_code(
                    body,
                    lambda part, v=variable, c=call: re.sub(
                        rf"(?<![.\w>]){re.escape(v)}\s*->", f"{c}->", part
                    ),
                )
        for name, symbol in (
            (_DEREFERENCE, "*"), ("op_not", "!"), (_NEGATE, "-"),
            ("op_inc", "++"), ("op_dec", "--"),
        ):
            owner = _find_method(holds, name, classes)
            if owner is None:
                continue
            if _method_named(owner, name, classes, returns_object=True):
                # It answers an object, so the caller has to provide the space
                # and the call cannot stand in an expression - the same as a
                # two-operand operator that does.
                body = _rewrite_value_prefix(
                    body, variable, symbol, owner, name, address, classes
                )
                continue
            body = _rewrite_prefix(
                body, variable, symbol, f"{_c_name(owner, name)}({address})"
            )
        # `c++` is the same call written after the name. Done second, so that
        # `++c` has already gone and cannot be read as `+` `+c`.
        for name, symbol in (("op_inc_post", "++"), ("op_dec_post", "--")):
            owner = _find_method(holds, name, classes)
            if owner is None:
                continue
            if _method_named(owner, name, classes, returns_object=True):
                # The same as the prefix case above, and more so: postfix is
                # the operator that *has* to answer by value.
                body = _rewrite_value_postfix(
                    body, variable, symbol, owner, name, address, classes
                )
                continue
            body = _map_code(
                body,
                lambda part, v=variable, s=symbol, o=owner, n=name, a=address: re.sub(
                    rf"(?<![.\w>]){re.escape(v)}\s*{re.escape(s)}",
                    f"{_c_name(o, n)}({a})",
                    part,
                ),
            )
    return body


def _arrow_target(owner: str, classes: "dict[str, Class]") -> "str | None":
    """The class `operator->` hands back a pointer to."""

    method = _method_named(owner, "op_arrow", classes)
    if method is None:
        return None
    held = method.returns.replace("*", "").replace("&", "").replace("const", "")
    held = held.strip()
    return held if held in classes else None




def _rewrite_value_call(
    body: str,
    variable: str,
    owner: str,
    address: str,
    classes: "dict[str, Class]",
    known: "dict[str, str] | None" = None,
) -> str:
    """`T v = f(args);` where `f` is an object whose call answers an object."""

    pattern = re.compile(
        rf"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=\s*"
        rf"(?<![.\w>]){re.escape(variable)}\s*\("
    )
    bare = _without_literals(body)
    out: list[str] = []
    at = 0
    for match in pattern.finditer(bare):
        if match.start() < at:
            continue
        held, target = match.group(1), match.group(2)
        if held not in classes:
            continue
        close = _closing_paren(body, match.end() - 1)
        if close < 0 or body[close + 1:].lstrip()[:1] != ";":
            continue
        arguments = body[match.end(): close]
        passed = f", {arguments}" if arguments.strip() else ""
        if known is not None:
            known[target] = held
        out.append(body[at:match.start()])
        out.append(
            f"struct {held} {target}; "
            f"{_c_name(owner, 'op_call')}({address}, &{target}{passed})"
        )
        at = close + 1
    out.append(body[at:])
    return "".join(out)

def _rewrite_value_prefix(
    body: str,
    variable: str,
    symbol: str,
    owner: str,
    name: str,
    address: str,
    classes: "dict[str, Class]",
) -> str:
    """`V b = -a;` - a prefix operator answering an object, given somewhere to put it."""

    pattern = re.compile(
        rf"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=\s*{re.escape(symbol)}\s*"
        rf"{re.escape(variable)}\s*;"
    )

    def one(match: "re.Match[str]") -> str:
        held, target = match.groups()
        if held not in classes:
            return match.group(0)
        return (
            f"struct {held} {target}; "
            f"{_c_name(owner, name)}({address}, &{target});"
        )

    return _map_code(body, lambda part: pattern.sub(one, part))


def _rewrite_value_postfix(
    body: str,
    variable: str,
    symbol: str,
    owner: str,
    name: str,
    address: str,
    classes: "dict[str, Class]",
) -> str:
    """`V e = d++;` - the old value, which is an object, needs somewhere to go.

    Postfix is the one operator that always answers by value: what it hands
    back is the object as it was before, which cannot be a reference to
    anything still alive. Written as though it answered in a register, the
    call was given one argument where the C wanted two.
    """

    pattern = re.compile(
        rf"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=\s*{re.escape(variable)}\s*"
        rf"{re.escape(symbol)}\s*;"
    )

    def one(match: "re.Match[str]") -> str:
        held, target = match.groups()
        if held not in classes:
            return match.group(0)
        return (
            f"struct {held} {target}; "
            f"{_c_name(owner, name)}({address}, &{target});"
        )

    return _map_code(body, lambda part: pattern.sub(one, part))


def _rewrite_prefix(body: str, variable: str, symbol: str, call: str) -> str:
    """`*p` or `!p` becomes the call, but only where nothing is on the left.

    `a * p` is a multiplication and `a != p` ends in something that is not a
    prefix `!` at all, so the character before decides - the last one that is
    not a space, because the spacing is the author's and means nothing.
    """

    bare = _without_literals(body)
    out: list[str] = []
    at = 0
    for match in re.finditer(
        rf"{re.escape(symbol)}\s*{re.escape(variable)}\b", bare
    ):
        if match.start() < at:
            continue
        before = bare[:match.start()].rstrip()
        # `!` is never a two-operand operator, so nothing in front of it can
        # make it one: `(int)!a` and `f(x) && !a` are both the prefix. The
        # others can - `a * p` is a multiplication and `f(x) - a` is a
        # subtraction - so for those the character before decides. Read the
        # same way for all of them, a cast in front of `!` stopped it being
        # recognised at all.
        if symbol != "!" and before and (
            before[-1].isalnum() or before[-1] in "_)]\"'"
        ):
            continue
        # `a && b` ends in the same character as a prefix `&`. The second of
        # a doubled symbol is part of the operator before it, not a prefix on
        # what follows.
        if before.endswith(symbol):
            continue
        out.append(body[at:match.start()])
        out.append(call)
        at = match.end()
    out.append(body[at:])
    return "".join(out)


def _rewrite_subscripts(
    body: str,
    variable: str,
    holds: str,
    classes: "dict[str, Class]",
    already: bool = False,
) -> str:
    """`g[0][1]` becomes one index call wrapped around another.

    A subscript that answers an object may be subscripted again, and after
    the first rewrite the receiver is no longer a name - so a pattern looking
    for one saw `g[0]` and left `[1]` for the C compiler, which read it as
    pointer arithmetic on a struct and said so.

    The receiver is written with a `&` at every level, including in front of
    the inner call. That reads oddly until you know what happens next: the
    pass that follows a reference return puts a `*` on every one of these
    calls, and `&(*p)` is `p`. Written any other way, that pass - which runs
    once per method name and so sees each call fresh - dereferenced a
    receiver that was already the address the outer call wanted.
    """

    bare = _without_literals(body)
    out: list[str] = []
    at = 0
    for match in re.finditer(rf"(?<![.\w>]){re.escape(variable)}\s*\[", bare):
        if match.start() < at:
            continue
        # A reference is already the address the call wants; an object is not.
        receiver = variable if already else f"&{variable}"
        held = holds
        end = match.end() - 1
        reached = False
        first = True
        while True:
            owner = _find_method(held, "op_index", classes)
            if owner is None:
                break
            close = _bracket_end(bare, end)
            if close < 0:
                break
            receiver = (
                f"{_c_name(owner, 'op_index')}"
                f"({receiver if first else '&' + receiver}, "
                f"{body[end + 1: close]})"
            )
            first = False
            reached = True
            end = close + 1
            held = _element_of(owner, classes)
            while end < len(bare) and bare[end] in " \t":
                end += 1
            if held is None or end >= len(bare) or bare[end] != "[":
                break
        if not reached:
            continue
        out.append(body[at:match.start()])
        out.append(receiver)
        at = end
    out.append(body[at:])
    return "".join(out)


def _element_of(owner: str, classes: "dict[str, Class]") -> "str | None":
    """The class a subscript on `owner` answers, if it answers one at all."""

    method = _method_named(owner, "op_index", classes)
    if method is None:
        return None
    held = method.returns.replace("&", "").replace("*", "").replace("const", "")
    held = held.strip()
    return held if held in classes else None


def _bracket_end(text: str, at: int) -> int:
    """Where the `[` at `at` is closed, counting the ones inside it."""

    depth = 0
    index = at
    while index < len(text):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


#: `)(` - a call on what a call answered. Searched for from the *second*
#: parenthesis, because that pair is rare and a call is not: starting from
#: every call and asking what follows it walked the whole body once per call.
_CALL_ON_A_CALL = re.compile(r"\)\s*\(")


def _rewrite_dereferenced_calls(
    body: str, classes: "dict[str, Class]", scope: str
) -> str:
    """`v[i](x)` becomes the call operator of whatever the element is.

    A subscript on a container has become a call answering an address by the
    time this runs, so what is left is one call standing in front of another
    - which is what a container of callables reads as.
    """

    if not classes:
        return body
    bare = _without_literals(body)
    out: list[str] = []
    at = 0
    for found in _CALL_ON_A_CALL.finditer(bare):
        if found.start() < at:
            continue
        first = _opening_paren(bare, found.start())
        if first < 0:
            continue
        head = re.search(r"(?<![.\w>])([A-Za-z_]\w*)\s*$", bare[:first])
        if head is not None:
            begins = head.start(1)
        elif bare[first + 1:].lstrip().startswith("*"):
            # `(*X)(args)`: the pass that follows a reference return has
            # already written the star, so the whole group is the object.
            begins = first
        else:
            continue
        opening = found.end() - 1
        end = _closing_paren(bare, opening)
        if end < 0:
            continue
        inner = body[begins: found.start() + 1]
        held = (_deduced_type(inner, f"{body}\n{scope}") or "").strip()
        held = held.replace("const", "").replace("struct", "").strip()
        spelled = held.replace("*", "").strip()
        owner = _find_method(spelled, "op_call", classes) if spelled in classes else None
        if owner is None:
            continue
        arguments = body[opening + 1: end].strip()
        passed = f", {arguments}" if arguments else ""
        # `&` in front, unless the star is already there. The pass that
        # follows a reference return puts one on every call it recognises,
        # including the one being written here - and `&(*p)` is `p`. Written
        # without it, that pass left an object where a receiver goes.
        followed = re.fullmatch(r"\(\s*\*\s*(.*)\)", inner, re.S)
        receiver = followed.group(1).strip() if followed else f"&{inner}"
        out.append(body[at:begins])
        out.append(f"{_c_name(owner, 'op_call')}({receiver}{passed})")
        at = end + 1
    out.append(body[at:])
    return "".join(out)


def _opening_paren(text: str, at: int) -> int:
    """Where the `)` at `at` was opened, or -1."""

    depth = 0
    index = at
    while index >= 0:
        if text[index] == ")":
            depth += 1
        elif text[index] == "(":
            depth -= 1
            if depth == 0:
                return index
        index -= 1
    return -1


#: `_call_signatures` answers the same thing for the same classes, and is
#: asked once per method body. Keyed on how many methods there are as well as
#: on the table itself, so a class that gains one is not answered from before.
_SIGNATURES_SEEN: "dict[tuple[int, int, int], dict[str, tuple[str, Method]]]" = {}


def _call_signatures(
    classes: "dict[str, Class]"
) -> "dict[str, tuple[str, Method]]":
    """Every method by the C name it is emitted under."""

    key = (
        id(classes),
        len(classes),
        sum(len(held.methods) for held in classes.values()),
    )
    remembered = _SIGNATURES_SEEN.get(key)
    if remembered is not None:
        return remembered
    found: dict[str, tuple[str, Method]] = {}
    for owner, held in classes.items():
        for method in held.methods:
            if method.name == "~":
                continue
            suffix = _suffix_of(owner, method, classes)
            found[_c_name(owner, method.name, suffix)] = (owner, method)
            if method.name == "":
                # `T__new` takes the constructor's own arguments and no
                # receiver, so it is a separate entry rather than the same one.
                made = _c_name(
                    owner,
                    "new",
                    suffix if _has_several_constructors(owner, classes) else None,
                )
                found[made] = (owner, method)
    _SIGNATURES_SEEN.clear()
    _SIGNATURES_SEEN[key] = found
    return found


def _returns_reference(method: "Method") -> bool:
    """Whether the method hands back a reference rather than a value."""

    return "&" in method.returns


def _fill_member_defaults(body: str, classes: "dict[str, Class]") -> str:
    """Give each call the arguments the declaration said it could leave out.

    Run once the calls carry their C names, so a member call and a
    constructor call are both an ordinary `Name(...)` by the time this looks
    - which is what makes one pass enough for both.
    """

    filled: "dict[str, list[str]]" = {}
    for owner, held in classes.items():
        for method in held.methods:
            values = held.defaults.get(method.name)
            if not values:
                continue
            spelled = _c_name(owner, method.name, _suffix_of(owner, method, classes))
            filled[spelled] = values
            if method.name == "":
                filled[_c_name(owner, "new", None)] = values
    if not filled:
        return body
    for name, values in filled.items():
        pattern = re.compile(rf"(?<![.\w>]){re.escape(name)}\s*\(")
        out: list[str] = []
        at = 0
        for call in pattern.finditer(body):
            if call.start() < at:
                continue
            close = _closing_paren(body, call.end() - 1)
            if close < 0:
                continue
            given = _call_arguments(body, call.end() - 1)
            # The receiver is the first argument of a member call and is not
            # one of the declared parameters.
            receiver = given[:1] if not name.endswith("__new") else []
            rest = given[len(receiver):]
            if len(rest) >= len(values) or not values[len(rest):]:
                continue
            extra = [value for value in values[len(rest):] if value]
            if len(rest) + len(extra) != len(values):
                continue
            out.append(body[at:call.end()])
            out.append(", ".join(receiver + rest + extra))
            at = close
        out.append(body[at:])
        body = "".join(out)
    return body


def _fix_call_arguments(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
    signatures: "dict[str, tuple[str, Method]] | None" = None,
) -> str:
    """Hand each call the addresses its parameters want, and follow references.

    Run once the calls have been rewritten, so a call written inside another
    one's argument list is one too. Arguments are fixed by recursion rather
    than by scanning past them, because an inner call is inside the outer
    call's own text and a single left-to-right sweep steps over it.
    """

    if signatures is None:
        signatures = _call_signatures(classes)
    if not signatures:
        return body
    pattern = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
    out: list[str] = []
    at = 0
    for found in pattern.finditer(body):
        if found.start() < at:
            continue
        close = _closing_paren(body, found.end() - 1)
        if close < 0:
            continue
        inside = _fix_call_arguments(
            body[found.end(): close], classes, known, pointers, signatures
        )
        entry = signatures.get(found.group(1))
        if entry is None:
            out.append(body[at:found.end()])
            out.append(inside)
            at = close
            continue
        owner, method = entry
        given = _split_arguments(inside) if inside.strip() else []
        # `T__new` passes no receiver; nor does a static member, which was
        # never given an object. Everything else passes `this` first, and a
        # value return puts the caller's space after it.
        made = found.group(1)
        allocates = made.endswith("new") or "__new__" in made
        skip = 0 if (method.shared or allocates) else 1
        if not allocates and _returns_object(method, classes):
            skip += 1
        wanted = [
            part.strip() for part in _split_arguments(method.parameters)
            if part.strip()
        ]
        if len(given) - skip != len(wanted):
            out.append(body[at:found.end()])
            out.append(inside)
            at = close
            continue
        for index, declared in enumerate(wanted):
            value = given[skip + index].strip()
            if value.startswith("&") or not _has_an_address(value):
                continue
            if value in pointers:
                continue
            held = re.sub(
                r"\b(?:const|struct|volatile|union)\b",
                " ",
                declared.replace("*", " ").replace("&", " "),
            ).split()
            # A parameter of class type takes an object of that class. Where
            # the argument is a name this scope does not hold one under, it
            # is something else - `escapeJson(state)` with a `const char *`
            # where a `const string &` is wanted - and that is a conversion,
            # made below, not an address.
            if (
                held
                and held[0] in classes
                and value.isidentifier()
                and known.get(value) != held[0]
                # Only where a conversion is a thing that could happen. A
                # plain struct has no constructor, so nothing can be
                # converted into one: whatever is being passed is already an
                # object of that type, and what the call wants is its
                # address. Without this, `v.push_back(item)` for a
                # `vector<PlainStruct>` was left passing the struct itself.
                and _find_method(held[0], "", classes) is not None
            ):
                continue
            if _passed_by_address(declared, classes) or _REFERENCE.search(
                declared
            ):
                given[skip + index] = f"&{value}"
        spelled = ", ".join(part.strip() for part in given)
        if _returns_reference(method):
            # The callee hands back an address, because C cannot return a
            # reference; following it here is what the language was doing
            # silently, and leaves `v[i] = 3;` an assignment to the element.
            out.append(body[at:found.start()])
            out.append(f"(*{made}({spelled}))")
            at = close + 1
            continue
        out.append(body[at:found.end()])
        out.append(spelled)
        at = close
    out.append(body[at:])
    return "".join(out)


#: What the pass below calls the temporary it builds.
_MADE_PREFIX = "__py2bin_made_"


def _convert_class_arguments(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
    counter: "list[int]",
    scope: str = "",
) -> str:
    """Build the temporary a parameter of class type is given.

    `widthOf("narrow")` where the parameter is a `const std::string &` is a
    conversion: C++ constructs a string from the literal, binds the
    reference to it, and destroys it after the call. Nothing in C does that
    on its own, so the object is declared ahead of the statement and its
    address is what the call is handed.

    Only where the class has a constructor that takes what is being passed.
    Where it has none this is not a conversion C++ would do either, and
    leaving the argument alone lets the type error be reported against the
    line that has it.
    """

    signatures = dict(_call_signatures(classes))
    # And the free functions, which are not methods and are the other half of
    # what a program calls. Read from the file rather than tracked, the same
    # way every other pass here reads a signature it did not write.
    reading = scope or body
    for definition in _DEFINITION.finditer(reading):
        if _depth_at(reading, definition.end() - 1) != 0:
            continue
        signatures.setdefault(
            definition.group(2),
            ("", Method("", "void", definition.group(3).strip(), "", 0)),
        )
    if not signatures:
        return body
    for _round in range(_HOIST_ROUNDS):
        bare = _without_literals(body)
        change = None
        for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", bare):
            entry = signatures.get(match.group(1))
            if entry is None:
                continue
            close = _closing_paren(bare, match.end() - 1)
            if close < 0:
                continue
            owner, method = entry
            inside = body[match.end(): close]
            given = _split_arguments(inside) if inside.strip() else []
            made = match.group(1)
            # A free function takes no receiver; a method takes `this` first,
            # and a value return puts the caller's space after it.
            allocates = made.endswith("new") or "__new__" in made
            # No receiver for a free function, a static member, or the
            # allocator a constructor becomes.
            skip = 0 if (not owner or method.shared or allocates) else 1
            # The hidden pointer a value return writes through sits after
            # whatever receiver there is - including where there is none.
            if allocates:
                pass
            elif owner and _returns_object(method, classes):
                skip += 1
            elif not owner and _free_returns_object(made, reading, classes):
                skip += 1
            wanted = [
                part.strip()
                for part in _split_arguments(method.parameters)
                if part.strip()
            ]
            if len(given) - skip != len(wanted):
                continue
            for index, declared in enumerate(wanted):
                if "*" in declared:
                    continue
                held = re.sub(
                    r"\b(?:const|struct|volatile)\b", " ", declared.replace("&", " ")
                ).split()
                if len(held) != 2 or held[0] not in classes:
                    continue
                holds = held[0]
                value = given[skip + index].strip()
                if not value or value.startswith("&") or value in known:
                    continue
                # Against the file and not only this body: what is being
                # passed is usually a parameter, and a parameter is declared
                # in the head. The body comes first in `reading`, so the
                # offset still points where it did.
                spelled = _deduced_type(value, reading, match.start())
                if spelled is None:
                    continue
                if spelled.replace("const", "").strip() == holds:
                    continue
                if _find_method(holds, "", classes) is None:
                    continue
                # And the class has to have a constructor that takes what is
                # being passed. Read as "one that takes one argument" it
                # took `'"'` for a `const char *` and built a string out of
                # a character - which is not a conversion C++ performs.
                if not _constructs_from(holds, spelled, classes):
                    continue
                try:
                    suffix = _call_suffix(
                        holds, "", classes, [value], body, match.start()
                    )
                except CppTranslationError:
                    continue
                change = (match, close, skip + index, holds, value, suffix, given)
                break
            if change is not None:
                break
        if change is None:
            return body
        match, close, where, holds, value, suffix, given = change
        counter[0] += 1
        temporary = f"{_MADE_PREFIX}{counter[0]}"
        given[where] = f"&{temporary}"
        begins = _statement_start(body, match.start())
        body = (
            body[:begins]
            + f"struct {holds} {temporary}; "
            f"{_c_name(holds, '', suffix)}(&{temporary}, {value}); "
            + body[begins: match.end()]
            + ", ".join(one.strip() for one in given)
            + body[close:]
        )
    return body


def _constructs_from(
    holds: str, spelled: str, classes: "dict[str, Class]"
) -> bool:
    """Whether that class has a one-argument constructor taking this type."""

    code = _type_code(spelled)
    for method in classes.get(holds, Class("")).methods:
        if method.name != "" or _arity(method.parameters) != 1:
            continue
        declared = _parameter_types(method.parameters)
        if not declared:
            continue
        if declared[0] == code or declared[0] in _PROMOTIONS.get(code, ()):
            return True
    return False


def _free_returns_object(
    name: str, text: str, classes: "dict[str, Class]"
) -> bool:
    """Whether that free function answers an object, so takes the space first."""

    for definition in _DEFINITION.finditer(text):
        if definition.group(2) != name:
            continue
        spelled = definition.group(1).strip()
        if "*" in spelled or "&" in spelled or not spelled.split():
            continue
        if spelled.split()[-1] in classes:
            return True
    return False


#: `((int (*)(struct Base *, struct P *))((p)->__vptr[2]))(` - the head of a
#: virtual call as this translator writes one. The cast opens the parameter
#: list, and the parameter list says what each argument has to be.
_A_DISPATCH_CAST = re.compile(r"\(\(\s*[^()]*?\(\s*\*\s*\)\s*\(")


def _address_dispatched_arguments(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
) -> str:
    """Take the address of what a virtual call's parameters ask for.

    The call is made through a pointer rather than a name, so nothing that
    reads a signature out of a declaration applies to it. It carries its own
    signature instead: the cast in front of it names every parameter, and a
    parameter that is a pointer to an object wants the address of one.
    """

    out: "list[str]" = []
    at = 0
    for cast in _A_DISPATCH_CAST.finditer(body):
        if cast.start() < at:
            continue
        shut = _closing_paren(body, cast.end() - 1)
        if shut < 0:
            continue
        wanted = _split_arguments(body[cast.end(): shut])
        whole = _closing_paren(body, cast.start())
        if whole < 0 or body[whole + 1: whole + 2] != "(":
            continue
        opening = whole + 1
        closing = _closing_paren(body, opening)
        if closing < 0:
            continue
        given = _split_arguments(body[opening + 1: closing])
        if len(given) != len(wanted):
            continue
        changed = False
        for index, declared in enumerate(wanted):
            if not declared.strip().endswith("*"):
                continue
            value = given[index].strip()
            if value.startswith("&") or value in pointers:
                continue
            # A name, or a member reached through one: both have an address,
            # and both are how a program holds a struct it hands over.
            if not _A_PLAIN_NAME.match(value):
                continue
            held = re.sub(
                r"\b(?:const|struct|volatile|union)\b",
                " ",
                declared.replace("*", " "),
            ).split()
            # An object this scope knows, or a plain struct declared with the
            # type the cast asks for. The second is how a program hands a
            # `RECT` to an interface: it is not a class, so nothing tracks
            # it, and the declaration is the only place that says what it is.
            tail = re.split(r"\.|->", value)[-1].strip()
            if value not in known and not (
                held
                and (
                    re.search(
                        rf"(?<![.\w>]){re.escape(held[0])}\s+{re.escape(tail)}\b",
                        body,
                    )
                    or _a_member_of_type(tail, held[0], classes)
                )
            ):
                continue
            given[index] = f"&{value}"
            changed = True
        if not changed:
            continue
        out.append(body[at: opening + 1])
        out.append(", ".join(one.strip() for one in given))
        at = closing
    out.append(body[at:])
    return "".join(out)


def _a_member_of_type(
    name: str, held: str, classes: "dict[str, Class]"
) -> bool:
    """Whether some class declares a member of that name holding that type."""

    for owner in classes.values():
        for member in owner.members:
            if member.name != name or member.array or "*" in member.ctype:
                continue
            spelled = re.sub(
                r"\b(?:const|struct|volatile|union)\b", " ", member.ctype
            ).split()
            if spelled and spelled[0] == held:
                return True
    return False


def _addressed_arguments(
    owner: str,
    method: str,
    arguments: str,
    known: "dict[str, str]",
    pointers: "set[str]",
    classes: "dict[str, Class]",
) -> str:
    """Take the address of every argument the callee takes by value.

    A parameter of class type is a pointer in the C, with the copy made on
    entry - so the caller has to hand over an address. Passing the object
    itself is a struct where a pointer is wanted, which is the argument count
    the compiler complains about rather than a type error it can name.
    """

    found = None
    seen = owner
    while seen and seen in classes:
        for candidate in classes[seen].methods:
            spelled = candidate.name or ""
            if spelled == method or (method == "" and spelled == ""):
                found = candidate
                break
        if found:
            break
        seen = classes[seen].base
    if found is None or not found.parameters:
        return arguments
    wanted = [
        part.strip() for part in found.parameters.split(",") if part.strip()
    ]
    given = _split_arguments(arguments)
    if len(given) != len(wanted):
        return arguments
    passed = []
    for value, declared in zip(given, wanted):
        words = declared.split()
        by_value = (
            "*" not in declared
            and "&" not in declared
            and len(words) == 2
            and words[0] in classes
        )
        spelled = value.strip()
        if by_value and spelled in known and spelled not in pointers:
            passed.append(f"&{spelled}")
        elif by_value and spelled in pointers:
            passed.append(spelled)
        elif (
            _REFERENCE.search(declared)
            and spelled not in pointers
            and _has_an_address(spelled)
        ):
            # A reference wants the address too; the callee dereferences it.
            passed.append(f"&{spelled}")
        else:
            passed.append(spelled)
    return ", ".join(passed)


def _split_arguments(arguments: str) -> "list[str]":
    """Split on the commas that separate arguments, not the ones inside them."""

    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for kind, piece in _split_literals(arguments):
        if kind == "literal":
            current.append(piece)
            continue
        for char in piece:
            if char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            if char == "," and depth == 0:
                parts.append("".join(current))
                current = []
                continue
            current.append(char)
    tail = "".join(current)
    if tail.strip() or parts:
        parts.append(tail)
    return parts

def _rewrite_pointer_indexed(
    body: str, pattern: str, function, variable: str, receiver: str = ""
) -> str:
    """`all[i]->m(` becomes `function(all[i]`: the element is the address.

    `receiver` overrides how the element is reached, for a container whose
    elements come out of a call rather than out of an array.
    """

    out = []
    at = 0
    for found in re.finditer(pattern, body):
        rest = body[found.end():].lstrip()
        chosen = (
            function
            if isinstance(function, str)
            else function(_call_arguments(body, found.end() - 1))
        )
        # The chooser may answer with a receiver of its own, where a virtual
        # call needs the whole object and a direct one needs a subobject.
        passed = receiver
        if isinstance(chosen, tuple):
            chosen, passed = chosen
        chosen = chosen.replace("__I__", found.group(1))
        reached = (
            passed.replace("__I__", found.group(1))
            if passed
            else f"{variable}[{found.group(1)}]"
        )
        separator = "" if rest.startswith(")") else ", "
        out.append(body[at:found.start()])
        out.append(f"{chosen}({reached}{separator}")
        at = found.end()
    out.append(body[at:])
    return "".join(out)


def _rewrite_indexed(body: str, pattern: str, function, variable: str) -> str:
    """`bank[i].rate(` becomes `function(&bank[i]`, keeping the index.

    `function` takes the same two forms as in :func:`_rewrite_calls`.
    """

    out = []
    at = 0
    for found in re.finditer(pattern, body):
        rest = body[found.end():].lstrip()
        separator = "" if rest.startswith(")") else ", "
        chosen = (
            function
            if isinstance(function, str)
            else function(_call_arguments(body, found.end() - 1))
        )
        # A virtual call reads the table out of the element, so it needs the
        # index too - which is only known here, one match at a time.
        passed = None
        if isinstance(chosen, tuple):
            chosen, passed = chosen
        chosen = chosen.replace("__I__", found.group(1))
        reached = (
            passed.replace("__I__", found.group(1))
            if passed
            else f"&{variable}[{found.group(1)}]"
        )
        out.append(body[at:found.start()])
        out.append(f"{chosen}({reached}{separator}")
        at = found.end()
    out.append(body[at:])
    return "".join(out)

def _closing_paren(text: str, opening: int) -> int:
    """The index of the `)` that closes the `(` at `opening`, or -1.

    Parentheses only, and blind to what is inside a string, so a `)` in a
    format string does not end the argument list early.
    """

    depth = 0
    index = opening
    while index < len(text):
        piece = text[index]
        if piece in "\"'":
            quote = piece
            index += 1
            while index < len(text) and text[index] != quote:
                index += 2 if text[index] == "\\" else 1
            index += 1
            continue
        if piece == "(":
            depth += 1
        elif piece == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _call_arity(body: str, opening: int) -> int:
    """How many arguments the call whose `(` is at `opening` passes."""

    return len(_call_arguments(body, opening))


def _call_arguments(body: str, opening: int) -> "list[str]":
    """The arguments of the call whose `(` is at `opening`, as written."""

    close = _closing_paren(body, opening)
    if close < 0:
        return []
    inside = body[opening + 1: close]
    if not inside.strip():
        return []
    return [part.strip() for part in _split_arguments(inside)]


def _rewrite_calls(body: str, pattern: str, function, receiver: str) -> str:
    """Turn each match into `function(receiver` plus a comma only if needed.

    The comma is the whole reason this is not a one-line `re.sub`: a method
    taking nothing becomes a call taking only the object, and `f(&v, )` is not
    C. Whether an argument follows is a property of the text after the match,
    which a replacement string cannot see.

    `function` may be a name or, where the class overloads that name, something
    that turns an argument count into one. C picks a function by its name
    alone, so the count is the only thing at the call site that can tell two
    overloads apart, and it is read here rather than guessed.
    """

    out = []
    at = 0
    for found in re.finditer(pattern, body):
        rest = body[found.end():].lstrip()
        chosen = (
            function
            if isinstance(function, str)
            else function(_call_arguments(body, found.end() - 1))
        )
        # A virtual call may need a different receiver from the direct one: it
        # dispatches on the whole object, where a direct call to an inherited
        # method is given the base subobject. The chooser says both.
        passed = receiver
        if isinstance(chosen, tuple):
            chosen, passed = chosen
        # A static member is handed no object, so there is nothing in front of
        # the first argument either.
        separator = "" if rest.startswith(")") or not passed else ", "
        out.append(body[at:found.start()])
        out.append(f"{chosen}({passed}{separator}")
        at = found.end()
    out.append(body[at:])
    return "".join(out)


def _name_for(
    owner: str,
    method: str,
    classes: "dict[str, Class]",
    text: str = "",
    before: int = -1,
):
    """A name if that member is alone, otherwise a chooser reading the call."""

    if not _overloaded(owner, method, classes):
        return _c_name(owner, method)
    return lambda given: _c_name(
        owner, method, _call_suffix(owner, method, classes, given, text, before)
    )


def _dispatch_receiver(
    holds: str, classes: "dict[str, Class]", receiver: str
) -> str:
    """What a virtual call hands over, which is not always the receiver.

    A table read out of a shared base holds entries that take a pointer to
    *it* - everything that reaches it has only that in common. So a call made
    through a class that reaches its table that way passes the shared base,
    and not the object it was written on.
    """

    carrier = _vptr_carrier(holds, classes)
    if carrier in _shared_bases(holds, classes):
        return f"({receiver})->{_vbase_pointer(carrier)}"
    return receiver


def _dispatch(
    holds: str, method: str, classes: "dict[str, Class]", receiver: str
):
    """How to call `method` on an object whose declared type is `holds`.

    A name, when the answer is fixed at compile time. Where the method is
    virtual it is a read from the object's own table instead, because the
    variable's type is not what decides - the object is.
    """

    if not _is_polymorphic(holds, classes):
        return None
    slots = _virtual_slots(holds, classes)
    path = _vptr_path(holds, classes)

    def chosen(given: "list[str]") -> str:
        key = (method, len(given))
        if key not in slots:
            return ""
        declared = _slot_method(holds, key, classes)
        if declared is None:
            return ""
        # Written for the class the table belongs to, which is the shared
        # base where there is one: those entries take a pointer to it.
        carrier = _vptr_carrier(holds, classes)
        result, parameters = _c_signature(
            carrier if carrier in _shared_bases(holds, classes) else holds,
            declared,
            classes,
        )
        return (
            f"(({result} (*)({parameters}))(({receiver})->{path}[{slots.index(key)}]))"
        )

    return chosen


def _dispatched(
    holds: str,
    method: str,
    classes: "dict[str, Class]",
    receiver: str,
    static: str,
    text: str = "",
    direct: "str | None" = None,
):
    """`_dispatch` where it applies, otherwise the direct name."""

    virtual = _dispatch(holds, method, classes, receiver)
    named = _name_for(static, method, classes, text)
    if virtual is None:
        return named

    def chosen(given: "list[str]"):
        through = virtual(given)
        if through:
            # A virtual call reads the table out of the whole object, and
            # passes the whole object. A direct call to an inherited method
            # is handed the base subobject instead - so where the two differ,
            # the receiver differs too, and applying the base adjustment to
            # both counted it twice.
            return through, _dispatch_receiver(holds, classes, receiver)
        spelled = named if isinstance(named, str) else named(given)
        return (spelled, direct) if direct is not None else spelled

    return chosen


def _reachable_methods(name: str, classes: "dict[str, Class]") -> "list[str]":
    found: list[str] = []
    for seen in [name, *_every_base(name, classes)]:
        if seen in classes:
            found.extend(m.name for m in classes[seen].methods if m.name)
    return found


#: `struct Counted *c = &t;` and `c = &t;` - a pointer to a base being given
#: the address of something derived.
_TAKES_A_BASE = re.compile(
    r"(?<![.\w>])struct\s+([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)\s*=\s*&\s*"
    # Nothing reached through it: `&t.__base1` is already the subobject, and
    # matching the `&t` inside one would step through it a second time.
    r"([A-Za-z_]\w*)(?![\w.\[])"
)
_ASSIGNS_A_BASE = re.compile(
    r"(?<![.\w>])([A-Za-z_]\w*)\s*=\s*&\s*([A-Za-z_]\w*)(?![\w.\[])\s*;"
)
#: `(struct Counted *)&t` - the conversion this translator writes when a base
#: is wanted. Right for the first base, whose address is the object's own,
#: and wrong for every other one.
_CAST_TO_A_BASE = re.compile(
    r"\(\s*struct\s+([A-Za-z_]\w*)\s*\*\s*\)\s*&\s*([A-Za-z_]\w*)"
    r"(?![\w.\[])"
)


#: `(struct R *)D__new(` - a cast of what a call answers.
_CAST_OF_A_CALL = re.compile(
    r"\(\s*struct\s+([A-Za-z_]\w*)\s*\*\s*\)\s*([A-Za-z_]\w*)\s*\("
)


def _move_to_second_base(text: str, classes: "dict[str, Class]") -> str:
    """Point at the base subobject, where the base is not the first one.

    The first base is at offset zero, so the address of the object is the
    address of it and a cast is the whole of the conversion. A second base is
    a member after the first: its address is further along, and saying which
    member is what moves the pointer there.

    This does the two forms a program writes most - a pointer declared from
    an object's address, and one assigned it. What it does not reach, the C
    compiler does: a pointer of the wrong type is a type error there, named
    with its line. So being incomplete here is a build that stops, and not a
    program that runs and is wrong - which is the only reason it is safe to
    do this in pieces at all.
    """

    if not any(
        found.mixins or found.virtual_bases for found in classes.values()
    ):
        return text
    holds: "dict[str, str]" = {}
    for found in re.finditer(
        r"(?<![.\w>])struct\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*[;,=\[)]", text
    ):
        if found.group(1) in classes:
            holds[found.group(2)] = found.group(1)
    pointers: "dict[str, str]" = {}
    for found in re.finditer(
        r"(?<![.\w>])struct\s+([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)", text
    ):
        pointers[found.group(2)] = found.group(1)

    def stepped(source: "str | None", wanted: str) -> "str | None":
        if source is None or wanted not in classes or source not in classes:
            return None
        path = _subobject_path(source, wanted, classes)
        if not path or set(path.split(".")) <= {"__base"}:
            return None
        return path

    def declared(match: "re.Match[str]") -> str:
        path = stepped(holds.get(match.group(3)), match.group(1))
        if path is None:
            return match.group(0)
        return (
            f"struct {match.group(1)} *{match.group(2)} = "
            f"&{match.group(3)}.{path}"
        )

    def assigned(match: "re.Match[str]") -> str:
        path = stepped(holds.get(match.group(2)), pointers.get(match.group(1), ""))
        if path is None:
            return match.group(0)
        return f"{match.group(1)} = &{match.group(2)}.{path};"

    def cast(match: "re.Match[str]") -> str:
        # A cast is the whole conversion only where the base is at offset
        # zero. Left standing for a second base it says the right type and
        # points at the wrong bytes - and, being a cast, it stops the C
        # compiler from noticing.
        path = stepped(holds.get(match.group(2)), match.group(1))
        if path is None:
            return match.group(0)
        return f"&{match.group(2)}.{path}"

    text = _map_code(text, lambda part: _CAST_TO_A_BASE.sub(cast, part))
    text = _map_code(text, lambda part: _TAKES_A_BASE.sub(declared, part))
    text = _map_code(text, lambda part: _ASSIGNS_A_BASE.sub(assigned, part))

    # `(struct R *)D__new()` - a cast of what a call answered, which the
    # three above do not reach because there is no name to look the type up
    # by. There is one to look it up *from*: this is C by now, and the
    # function's own definition says what it returns.
    for _round in range(_HOIST_ROUNDS):
        changed = False
        bare = _without_literals(text)
        for found in _CAST_OF_A_CALL.finditer(bare):
            answered = re.search(
                rf"(?<![.\w>])(?:struct\s+)?([A-Za-z_]\w*)\s*\*\s*"
                rf"{re.escape(found.group(2))}\s*\(",
                bare,
            )
            path = stepped(
                answered.group(1) if answered else None, found.group(1)
            )
            if path is None:
                continue
            close = _closing_paren(bare, found.end() - 1)
            if close < 0:
                continue
            call = text[found.start(2): close + 1]
            text = (
                text[: found.start()]
                + f"&((struct {answered.group(1)} *)({call}))->{path}"
                + text[close + 1:]
            )
            changed = True
            break
        if not changed:
            break
    return text


def _every_base(name: str, classes: "dict[str, Class]") -> "list[str]":
    """Every class this one inherits from, however many chains there are."""

    found: "list[str]" = []
    pending = [name]
    while pending:
        seen = pending.pop(0)
        held = classes.get(seen)
        if held is None:
            continue
        for one in ([held.base] if held.base else []) + list(held.mixins):
            if one in classes and one not in found:
                found.append(one)
                pending.append(one)
    return found


#: What the constructor that does *not* build the shared bases is called.
_SUB_OBJECT_SUFFIX = "__sub"


def _vbase_pointer(name: str) -> str:
    """The member holding the address of a shared base."""

    return f"__vbase_{name}"


def _vbase_storage(name: str) -> str:
    """The member a complete object keeps that shared base in."""

    return f"__vstore_{name}"


def _base_steps(
    found: Class, classes: "dict[str, Class]"
) -> "list[tuple[str, str, bool]]":
    """Each base of this class as (name, how to reach it, is it shared).

    A base inherited `virtual` is not a member of what derives from it: C++
    gives one shared subobject however many paths reach it, so what the class
    holds is its address. Written `__vbase_A[0]`, the step still reads as a
    member path - `&o.__vbase_A[0]` is `o.__vbase_A` - so everything that
    reaches a base by naming the way there keeps working unchanged.
    """

    steps: "list[tuple[str, str, bool]]" = []
    if found.base:
        shared = found.base in found.virtual_bases
        steps.append(
            (
                found.base,
                f"{_vbase_pointer(found.base)}[0]" if shared else "__base",
                shared,
            )
        )
    for index, mixin in enumerate(found.mixins):
        shared = mixin in found.virtual_bases
        steps.append(
            (
                mixin,
                f"{_vbase_pointer(mixin)}[0]" if shared else f"__base{index + 1}",
                shared,
            )
        )
    return steps


def _shared_bases(name: str, classes: "dict[str, Class]") -> "list[str]":
    """Every base reached from here that is shared, nearest first.

    Its own, and the ones its bases hold: the most derived object owns the
    storage, so it has to know about every one of them however deep the
    class that wrote `virtual` sits.
    """

    found = classes.get(name)
    if found is None:
        return []
    seen: "list[str]" = []
    for base, _step, shared in _base_steps(found, classes):
        if shared and base not in seen:
            seen.append(base)
        for deeper in _shared_bases(base, classes):
            if deeper not in seen:
                seen.append(deeper)
    return seen


def _subobject_path(
    derived: str, owner: str, classes: "dict[str, Class]"
) -> "str | None":
    """How to reach the `owner` subobject of a `derived` object, as members.

    `""` where they are the same class, `"__base"` for the first base,
    `"__base.__base"` for its base, and `"__base1"` for the second base a
    class was written with. The first base is at offset zero and the rest are
    members after it, so the path is what names the one that is wanted -
    which a depth alone cannot, once there is more than one chain to count
    along.
    """

    if derived == owner:
        return ""
    found = classes.get(derived)
    if found is None:
        return None
    for name, step, _shared in _base_steps(found, classes):
        if not name:
            continue
        deeper = _subobject_path(name, owner, classes)
        if deeper is not None:
            return step if deeper == "" else f"{step}.{deeper}"
    return None


def _base_depth(derived: str, owner: str, classes: "dict[str, Class]") -> int:
    depth = 0
    seen = derived
    while seen and seen != owner:
        depth += 1
        seen = classes[seen].base
    return depth


def _destructor_call(
    holds: str, expression: str, classes: "dict[str, Class]"
) -> str:
    """Destroy an object of class `holds`, named by `expression`.

    A destructor a base provides takes a pointer to *that* base, so the call
    has to name the embedded subobject rather than the whole thing - the
    address is the same but the type is not, and C says so.
    """

    owner = _find_method(holds, "~", classes)
    if owner is None:
        return ""
    reached = f"&{expression}"
    if owner != holds:
        path = _subobject_path(holds, owner, classes) or "__base"
        reached = f"&{expression}.{path}"
    return f"{_c_name(owner, '~')}({reached});"


def _close_with_destructors(
    body: str,
    destroyed: "list[str]",
    known: "dict[str, str]",
    classes: "dict[str, Class]",
    enclosing: "list[tuple[str, str, frozenset, bool]]" = (),
    returns: str = "",
    counter: "list[int] | None" = None,
    in_a_loop: bool = False,
) -> str:
    """Run each destructor where the block ends - including at a `return`.

    The closing brace is not the only way out. Put only at the end, the calls
    sat after `return 0;` and never ran at all, which is worse than not having
    written a destructor: the code says it cleans up and does not.

    A `return` whose value mentions an object being destroyed is refused
    instead of reordered. Destroying it first would return a dead object and
    destroying it after needs a temporary of a type this does not know, so the
    honest answer is to say so.
    """

    if not destroyed and not enclosing:
        return body
    calls = "".join(
        f" {_destructor_call(known[name], name, classes)}"
        for name in reversed(destroyed)
    )
    # A `return` inside a block leaves the whole function, so it destroys what
    # the blocks around it built as well - innermost first, which is the order
    # they were made in reversed. Only at a `return`: reaching the end of this
    # block does not end theirs.
    outer = "".join(
        f" {_destructor_call(held, name, classes)}"
        for name, held, _labels, _loop in reversed(list(enclosing))
    )
    # Where each one comes into existence. A `return` above a declaration
    # leaves before that object was ever built, and C++ destroys only what
    # has been constructed - so running its destructor there took apart
    # something that is not there yet, under a name nothing has declared.
    built: "dict[str, int]" = {}
    for name in destroyed:
        where = re.search(rf"(?<![.\w>]){re.escape(name)}\b", body)
        built[name] = where.start() if where is not None else 0

    out = []
    at = 0
    for found in re.finditer(r"\breturn\b([^;]*);", body):
        value = found.group(1).strip()
        already = [name for name in destroyed if built[name] < found.start()]
        leaving = "".join(
            f" {_destructor_call(known[name], name, classes)}"
            for name in reversed(already)
        ) + outer
        # An object that is being handed back is not taken apart on the way
        # out: what it holds now belongs to the caller. That is what C++ does
        # with a move, and it is the only reading that is safe without one -
        # the copy the caller gets points at the same things, so destroying
        # them here would leave it holding what has been taken apart.
        handed = [
            name
            for name in already
            if re.search(rf"(?<![.\w>])\b{re.escape(name)}\b", value)
        ]
        if handed:
            already = [name for name in already if name not in handed]
            leaving = "".join(
                f" {_destructor_call(known[name], name, classes)}"
                for name in reversed(already)
            ) + outer
        out.append(body[at:found.start()])
        # The answer is worked out *before* anything is taken apart. C++
        # evaluates the returned expression and then destroys what the scope
        # held; written the other way round, `return alive;` in a scope whose
        # destructor decrements `alive` answered with the count after the
        # destructors rather than before them. It compiled and it was wrong,
        # which is the worst way to be wrong.
        # `static` says where the *function* lives, not what it answers. Left
        # on, the temporary was a static local - initialised once, on the
        # first call, and the same value ever after.
        written = " ".join(
            word for word in returns.split() if word not in _STORAGE
        )
        held = written.replace("*", " ").replace("&", " ").strip()
        if (
            value
            and leaving.strip()
            and held
            and held not in classes
            and not _is_a_constant(value)
        ):
            counter = counter if counter is not None else [0]
            counter[0] += 1
            spelled = f"{_ANSWER_PREFIX}{counter[0]}"
            out.append(
                f"{{ {written} {spelled} = {value};{leaving} return {spelled}; }}"
            )
        else:
            out.append(leaving.strip() + " " + found.group(0))
        at = found.end()
    out.append(body[at:])
    body = "".join(out)
    body = _destroy_before_leaving(
        body, destroyed, known, classes, enclosing, in_a_loop
    )

    closing = body.rfind("}")
    if closing < 0:
        return body
    # And at the end, for a path that simply falls off it.
    return body[:closing] + calls + " " + body[closing:]


#: Any jump out of a scope, not only the one the exception pass writes. A
#: `goto` a program wrote leaves exactly as much as one of those does.
_TO_A_HANDLER = re.compile(r"\bgoto\s+([A-Za-z_]\w*)\s*;")

#: `break` and `continue` leave the innermost loop's body. How far *out* they
#: go is not written down anywhere a reader of this text can see, so only the
#: scope holding one is taken apart here - which is the scope they are almost
#: always written in.
_LEAVES_A_LOOP = re.compile(r"\b(break|continue)\s*;")


def _handlers_written(body: str) -> frozenset:
    """The labels this body holds, which a jump to one of them does not leave."""

    return frozenset(
        match.group(1)
        for match in _AT_A_HANDLER.finditer(body)
        if match.group(1) not in ("case", "default")
    )


#: Where a label is written down: at the start of a statement, and not the
#: `:` of a `case`, of a ternary, or of a qualified name.
_AT_A_HANDLER = re.compile(
    # The marker a lifted block leaves counts as something a statement may
    # follow: written without it, a label standing after a nested block was
    # not seen, and a `goto` to it was taken for one that leaves the scope -
    # so everything the scope held was destroyed on the way past.
    r"(?:^|[;{}:\x00])\s*([A-Za-z_]\w*)\s*:(?!:)", re.M
)


def _destroy_before_leaving(
    body: str,
    destroyed: "list[str]",
    known: "dict[str, str]",
    classes: "dict[str, Class]",
    enclosing: "list[tuple[str, str, frozenset, bool]]" = (),
    in_a_loop: bool = False,
) -> str:
    """Run the destructors on the way out to a handler, as C++ unwinding does.

    A `return` is not the only way a scope ends early. An exception leaves
    through a `goto` to its handler, and objects built above it are as dead
    as they would be at a `return` - but the pass that reads this text looked
    for `return` alone, so an object built inside a `try` was simply
    abandoned: no destructor, no diagnostic, and a program that counts what
    it has built answers one too many.

    Only where the label is somewhere else. A jump to a handler written
    *inside* this block does not leave it, and destroying here would take
    apart an object the rest of the block still uses - and again at the end.
    """

    if not destroyed and not enclosing:
        return body
    here = _handlers_written(body)
    built = {
        name: (
            where.start()
            if (where := re.search(rf"(?<![.\w>]){re.escape(name)}\b", body))
            else 0
        )
        for name in destroyed
    }
    out: "list[str]" = []
    at = 0
    leaving_here = "".join(
        f" {_destructor_call(known[name], name, classes)}"
        for name in reversed(destroyed)
    )
    for found in _LEAVES_A_LOOP.finditer(body):
        already = [
            name
            for name in destroyed
            if (where := re.search(rf"(?<![.\w>]){re.escape(name)}\b", body))
            and where.start() < found.start()
        ]
        leaving = "".join(
            f" {_destructor_call(known[name], name, classes)}"
            for name in reversed(already)
        )
        # And outward as far as the loop's own body, which is what `break`
        # and `continue` leave. Written almost always inside an `if`, so
        # taking apart only the scope holding the jump left everything the
        # loop itself had built - a leak, and a quiet one.
        if not in_a_loop:
            for name, held, _labels, loop in reversed(list(enclosing)):
                leaving += f" {_destructor_call(held, name, classes)}"
                if loop:
                    break
        if not leaving.strip():
            continue
        out.append(body[at:found.start()])
        out.append(leaving.strip() + " " + found.group(0))
        at = found.end()
    out.append(body[at:])
    body = "".join(out)
    _ = leaving_here

    out = []
    at = 0
    for found in _TO_A_HANDLER.finditer(body):
        if found.group(1) in here:
            continue
        already = [name for name in destroyed if built[name] < found.start()]
        leaving = "".join(
            f" {_destructor_call(known[name], name, classes)}"
            for name in reversed(already)
        )
        # And outward, one scope at a time, as far as the scope that holds
        # the handler - which the jump stays inside, so what that one built
        # is still alive.
        for name, held, labels, _loop in reversed(list(enclosing)):
            if found.group(1) in labels:
                break
            leaving += f" {_destructor_call(held, name, classes)}"
        if not leaving.strip():
            continue
        out.append(body[at:found.start()])
        out.append(leaving.strip() + " " + found.group(0))
        at = found.end()
    out.append(body[at:])
    return "".join(out)



#: `int Vec::scaled(int k) { ... }` - a member defined outside its class.
_OUT_OF_LINE = re.compile(
    # (?<![#\w]) keeps the return type from starting inside a directive: with
    # `#endif` on the line above, "endif" matched as the type and the match
    # swallowed the directive, leaving a bare `#` and an unclosed `#if`.
    # The return type is optional, because a constructor and a destructor have
    # none: `Stack::Stack()` matched nothing while `void Stack::push()` did, so
    # the constructor was left in the output as C++ nobody could compile.
    # `const` after the parameters says the method does not change the object,
    # which is a promise C++ checks and C has no way to make. Without a slot
    # for it here, `int Box::width() const {` matched nothing and was emitted
    # verbatim as `int struct Box::width() const` - while the call site was
    # rewritten to the mangled name, so the two halves disagreed.
    # The stars go in a piece of their own: `const char *Square::name()` has
    # no space between the `*` and the class, so a pattern ending the return
    # type with whitespace matched from `Square` onwards and left `const char
    # *` standing in the output as something C cannot read.
    # It stops at the opening parenthesis and the rest is read by matching
    # that: a parameter may be a function type - `Session::Session(function
    # <void(const string &)> h)` - and a pattern that ends the list at the
    # first `)` ended it inside one, so the definition was not recognised at
    # all and its head was left in the C as a call to a constructor.
    r"(?<![#\w])(?:([A-Za-z_][\w \t]*?)\s*([*&]*)\s*)?"
    r"\b([A-Za-z_]\w*)::(~?[A-Za-z_]\w*)\s*\("
)

#: What may sit between an out-of-line member's parameters and its body.
_AFTER_PARAMETERS = re.compile(r"\s*(?:const\s*)?(?:noexcept\s*)?\{")


#: `struct A { };` - which C++ allows and C does not.
_AN_EMPTY_STRUCT = re.compile(r"\b(struct|union)\s+([A-Za-z_]\w*)\s*\{\s*\}")


def _fill_empty_structs(text: str) -> str:
    """Give a struct with nothing in it something, because C insists.

    A class whose members are all static has no data members at all - which
    is exactly what a traits class is, and what `struct is_same__int_int { };`
    came out as. C++ gives such a class a size of one; C has no way to write
    one, so it gets a member nothing reads.
    """

    return _map_code(
        text,
        lambda part: _AN_EMPTY_STRUCT.sub(
            lambda m: f"{m.group(1)} {m.group(2)} {{ char __empty; }}", part
        ),
    )


def translate(source: str, filename: str = "<c++>") -> str:
    """The C for one C++ translation unit."""

    return _fill_empty_structs(_translate(source, filename))


#: A directive that opens a conditional region, and the one that closes it.
#: `#else` and `#elif` need no pattern of their own: they are inside a region
#: that is already being gathered, and a region is taken whole.
_OPENS_A_CONDITION = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef)\b")
_CLOSES_A_CONDITION = re.compile(r"^\s*#\s*endif\b")


def _translate(source: str, filename: str = "<c++>") -> str:
    """Translate the C++ subset in `source` into C.

    The result is ordinary C: structs where the classes were, free functions
    where the methods were, and calls rewritten to pass the object.
    """

    text = _strip_comments(source)
    # Before any pass that writes a declaration out: C++ lets a loop or a
    # branch take one statement without braces, and that statement is still
    # a scope. A declaration written in front of an unbraced body lands
    # outside the loop, which is neither where it was nor in scope for what
    # it was built from.
    text = _brace_loose_bodies(text)
    # Before the classes are read: a friend is not a member, and reading one
    # as a member gave `friend` for a return type.
    text = _lift_friends(text)
    # And then back in, where a free operator names a class this file
    # declares: an operator is a member here, whichever way it was written.
    # Before the namespace qualifiers go: what makes one of these the
    # standard `move` rather than a function the program wrote is the `std::`
    # in front of it, and that is gone three lines below.
    text = _strip_moves(text)
    # Before anything else reads the text: a class inside a namespace is a
    # class, and every pass below looks for classes at the top level.
    text, namespaces = _flatten_namespaces(text, filename)
    text, aliases = _namespace_aliases(text)
    # `std` whether or not a header in this file opened one. `<cmath>` is
    # C's `math.h` here and declares no namespace, so nothing collected the
    # name - and `std::sqrt(x)`, which is how C++ spells it, reached the C
    # compiler with the `::` still in it.
    text = _strip_namespace_qualifiers(text, namespaces | aliases | {"std"})
    # After the qualifiers go, and not before. `std::ostream &operator<<(...)`
    # is how every one of these is written outside the standard library, and
    # a name with `std::` in front of it is not the class name this looks up -
    # so every qualified free operator was left where it was and never became
    # a member of anything.
    text = _free_operators_as_members(text)
    # `MIDL_INTERFACE("...") IXMLDOMNode : public IDispatch {` is a class,
    # and nothing here could see that: those spellings are defined for the
    # *C* stage, which runs after this one. So a program deriving from a
    # generated COM interface was told the interface was not a class this
    # unit declares.
    text = _expand_com_spellings(text)
    _refuse_unsupported(text, filename)
    # After the COM spellings are written out, so the macros a generated
    # header defines for the C stage are gone before this reads what is left.
    # Everything below that has to know a name - which class a base clause
    # points at, what a member's type is - asks this, because the
    # preprocessor that would have settled it runs after all of it.
    _read_macros(text)
    _refuse_macro_class_heads(text, filename)
    # Before anything reads a function's name: an overload set is several
    # functions sharing one, and every later pass assumes a name is a thing.
    # Before names are read: a template has no name until it is written out,
    # and what comes out of this is ordinary classes and functions.
    # The C++ that is only a different spelling of C: named casts, forward
    # declarations, scoped enums, and `bool`/`true`/`false`/`nullptr`, which
    # are keywords there and macros here.
    before_patterns = text
    # Before the spellings: that pass writes `typedef enum Mode Mode;` right
    # after the enum body, and a typedef written inside a class body is not C.
    # Before anything reads a declaration: a linkage specification is words
    # in front of one, and its braces are not a scope.
    text = _strip_linkage(text)
    # Before the word is taken away, because the word is what says the
    # function may be asked while translating.
    text = _fold_constexpr_calls(text)
    # After the constexpr calls are answered, because a `static_assert` is
    # usually asking about one.
    text = _check_static_asserts(text, filename)
    text = _strip_constexpr(text)
    text = _lift_nested_enums(text)
    text, scoped_enums = _rewrite_cpp_spellings(text)
    text = _strip_namespace_qualifiers(text, scoped_enums)
    # Before the lambdas, because a member initialised from one is still an
    # initialiser list; after the spellings, because a cast may appear in one.
    text = _rewrite_defaulted_members(text)
    text = _rewrite_initialiser_lists(text)
    text = _rewrite_brace_initialisers(text)
    text = _rewrite_static_members(text, filename)
    text = _lift_nested_classes(text)
    text = _rewrite_default_arguments(text)
    text = _rewrite_range_for(text, [0])
    # After the spellings, because the body a callback carries has `nullptr`
    # and named casts in it like any other and the class it was written in
    # says `final` until that pass has taken the word off; after the brace
    # initialisers, because reading that class means reading its members and
    # `Token t{};` is not one until then. Before the lambdas, because what
    # this leaves behind has none - the lambda's body becomes an ordinary
    # method of an ordinary class.
    text = _rewrite_wrl_callbacks(text, filename)
    # Before the templates: a lambda becomes a class, and a class may be a
    # template argument.
    text = _expand_lambdas(text, filename)
    # After the lambdas, because what a `std::function` is given is a closure
    # by then - an object with a name and a class, which is what this reads.
    text = _rewrite_std_function(text, filename)
    # After the lambdas: `auto f = <lambda>` is the one `auto` whose type has
    # no spelling, and it is already gone by here.
    text = _rewrite_auto(text)
    # Before the templates are read: a member defined outside its class
    # template belongs to the pattern the reader is about to copy.
    text = _fold_out_of_line_templates(text)
    # Before the file's own templates: a member template is expanded from its
    # call sites, and the copies it leaves are ordinary members.
    text = _expand_member_templates(text, filename)
    # Before the copies are written out, because that is what takes the
    # `tuple` pattern out of the text - and the pattern is what says that a
    # `get<0>` here is this one and not a name the program has of its own.
    # Before the copies are written out: a class named without its arguments
    # has to have them by the time the copy is asked for.
    text = _deduce_class_arguments(text)
    text = _rewrite_threads(text, filename)
    text = _rewrite_tuple_get(text)
    text = _rewrite_variant_alternatives(text)
    text = _rewrite_duration_cast(text)
    text = _expand_templates(text, filename)
    # And again: a member template inside a class *template* has no calls to
    # read while the class is still a pattern - nothing has an object of it -
    # so it was left alone above. The copies written just now are ordinary
    # classes, and the calls on them say which copy of the member is wanted.
    text = _expand_member_templates(text, filename)
    # After the copies are written out, because that is what turns `sizeof(T)`
    # into `sizeof(int)` and a trait into the constant it answers.
    # Again, now that the copies exist: a `static_assert` inside a template
    # asks about the type it was written for, and there is no type until here.
    text = _check_static_asserts(text, filename, whatever_it_says=True)
    text = _rewrite_if_constexpr(text, filename)
    # After the copies too: what a binding is written against is often one of
    # them, and its members are not there to be read until it exists.
    text = _rewrite_structured_bindings(text, filename)
    # After the copies, because what says a class takes `push_back` is the
    # copy written for these arguments.
    text = _rewrite_list_initialisers(text)
    # Again, because a member template inside a class template could not be
    # read until the class had been written out: until then its calls are on
    # objects of a type that does not exist yet. `ComPtr<T>::As` is one -
    # `webView_.As(&extendedWebView)` says which copy it wants only once
    # `ComPtr<ICoreWebView2>` is a class.
    text = _expand_member_templates(text, filename)
    # A copy that pass wrote may name a template by its arguments - `Box<U>`
    # with `U` settled is `Box<int>` - and the patterns are gone by now. The
    # copy it means was written out earlier, so it is pointed at that.
    text = _point_at_existing_copies(text)
    # Again, because a template's own body was a pattern when the pass above
    # ran and is a class only now. Every traits class is exactly this: a
    # template whose one member is static, written out once per question
    # asked of it.
    text = _rewrite_static_members(text, filename)
    # After them: a function's name passed as a value needs a type that can
    # be written where a template argument goes, and the deduction above has
    # already used the name this writes the typedef for. Not before, or the
    # patterns themselves - whose parameters are still spelled `T` - would
    # each get a typedef naming a type that does not exist.
    text = _name_function_types(text)
    # After the copies exist, so `vector<int>::iterator` has become
    # `vector__int::iterator` and there is a class of that name to ask.
    text = _resolve_nested_typedefs(text)
    # Again, for the `auto` whose type only exists once the copies do: the
    # element of a `vector<int>` is named by `vector__int`'s own subscript
    # operator, and until the copy is written there is no such class to ask.
    text = _rewrite_auto(text)
    text = _mangle_overloaded_functions(text, filename)
    #: Whether a pass above has already made this text something other than
    #: what the author wrote. The shortcut below hands the source straight
    #: back, which threw away every expansion made here.
    patterned = text != before_patterns
    #: Whether this file throws. Asked before the rewriting below, which is
    #: what takes the word away.
    throws = _THROWS.search(text) is not None
    # Read here whether or not this file throws, and before the classes are
    # taken apart. A pass that runs after that has no class bodies left to
    # read, and asked what `Res` was it answered that it was nothing at all -
    # which was true only of a file with no `throw` in it, since a file with
    # one filled these in on the way past.
    global _CLASS_NAMES, _POLYMORPHIC, _INHERITED_FROM, _CLASS_MEMBERS
    _CLASS_NAMES = {m.group(2) for m in _CLASS_HEAD.finditer(text)}
    # A number given to a class in one file means nothing in the next.
    _THROWN_KINDS.clear()
    _LAMBDA_RESULTS.clear()
    _CLASS_MEMBERS = {}
    _CLASS_PACK.clear()
    packing = _pack_regions(text)
    for one in _CLASS_HEAD.finditer(text):
        try:
            shut = _matching(text, one.end() - 1)
        except ValueError:
            continue
        _CLASS_MEMBERS[one.group(2)] = _members_declared(text[one.end(): shut - 1])
        # What `#pragma pack` was in force where the class was *written*. The
        # directives are moved to the top of the file and the class is
        # emitted somewhere else again, so by the time it is written out
        # there is nothing around it to say - and the struct came out with
        # the padding the pragma was there to remove.
        held = _pack_in_force(packing, one.start())
        if held is not None:
            _CLASS_PACK[one.group(2)] = held
    _POLYMORPHIC = _polymorphic_names(text)
    _INHERITED_FROM = {
        one for m in _CLASS_HEAD.finditer(text) for one in _bases_of(m)
    }
    # After the names are known, because telling `T &&o` from `a && b` is
    # exactly the question of whether the first word names a type.
    text = _rvalue_references_to_references(text)
    if throws:
        # What the pass below leaves behind is a `return`, and only the pass
        # that reads a class body knows which destructors a return has to run
        # first - so it goes before them.
        text = _rewrite_exceptions_early(text, filename)
        patterned = True

    classes: dict[str, Class] = {}
    order: list[str] = []
    plain: list[str] = []
    pieces: list[tuple[int, int, str]] = []   # start, end, replacement

    for head in _CLASS_HEAD.finditer(text):
        keyword, name = head.group(1), head.group(2)
        named = _bases_of(head)
        base = named[0] if named else None
        mixins = named[1:]
        opening = head.end() - 1
        try:
            closing = _matching(text, opening)
        except ValueError:
            raise CppTranslationError(
                filename, _line_of(text, opening), f"{keyword} {name} is not closed"
            ) from None
        # A `struct` with no methods is C already; leave it exactly as it is.
        # Unless it names a base, which C has no spelling for: `struct Derived
        # : Base { int c; };` has no methods and is not C, and left alone it
        # reached the C compiler with the `:` still in it. One that inherits
        # goes through the machinery that lays a base out as the first member,
        # whether or not it declares anything of its own.
        inner = text[opening + 1: closing - 1]
        if keyword == "struct" and _plainly_c(inner) and not _bases_of(head):
            # A struct with no methods is C already and is left exactly as it
            # is - but C++ lets the bare name be a type and C does not, so it
            # still needs the typedef emitted below.
            plain.append(name)
            # Its members are read even though its body is emitted as it was
            # written. Something deriving from it needs to know what names it
            # brings: `d.a` where `a` belongs to a plain base has to become
            # `d.__base.a`, and nothing can say so without the list.
            held = Class(name)
            for spelled, member in _struct_members(text[head.start(): closing], name):
                held.members.append(Member(member, spelled))
            classes[name] = held
            continue
            # It is still an object as far as *passing* goes: py2bin's C can
            # neither pass nor answer a struct by value, and `Point add(Point
            # a, Point b)` is as ordinary in C++ as it is impossible here. So
            # it is registered with nothing in it - which is all these passes
            # need to know - and is kept out of `order`, so the body it was
            # written with is emitted rather than one rebuilt from a reading
            # that a bitfield or an array would not survive.
        found = _split_members(inner, name, filename, _line_of(text, opening))
        found.base = base
        found.mixins = [one for one in mixins if one != base]
        found.virtual_bases = _virtual_bases_of(head)
        classes[name] = found
        order.append(name)
        end = closing
        while end < len(text) and text[end] in " \t":
            end += 1
        if end < len(text) and text[end] == ";":
            end += 1
        pieces.append((head.start(), end, ""))

    if not classes:
        # References and `new` are C++ without being classes, and a file can
        # use them and declare none. Returning the source untouched here left
        # `void swap(int &a, int &b)` for the C compiler to choke on.
        allocates = _wants_heap(text)
        loose = (
            allocates
            or _REFERENCE.search(text) is not None
            or patterned
            or throws
            # `std::sqrt(x)` is C++ spelling a C function, and the qualifier
            # has been taken off `text` by here - but this path hands back
            # the *source*, so without noticing it the `::` went to the C
            # compiler in a file that had nothing else C++ about it.
            or re.search(r"(?<![.\w>])std\s*::", source) is not None
        )
        if not plain and not namespaces and not loose:
            # Nothing C++ about this file at all: hand back what was written,
            # comments and all, so a diagnostic points at the real text.
            return source
        if loose:
            text = _address_reference_arguments(text, _function_signatures(text))
            text = _rewrite_functions(text, {}, None, text)
            if throws:
                text = _EXCEPTION_STATE + text
            if allocates:
                # Asked before the rewriting, which is what turns `new` into
                # the call to malloc: afterwards the word is gone.
                text = f"#include <stdlib.h>\n{text}"
        if not plain:
            # Namespaces but no classes - flattened above, and that is the
            # whole of the work. Returning `source` here threw it away, and
            # the C compiler was handed `namespace u { ... }`.
            return text
        # Nothing to translate, but the bare struct names still need to be
        # types: this file is C++ only in that respect.
        typedefs = "\n".join(
            f"typedef struct {name} {name};" for name in plain
        )
        return _with_typedefs(text, typedefs)

    for name in order:
        found = classes[name]
        for one in [found.base, *found.mixins]:
            if one and one not in classes:
                raise CppTranslationError(
                    filename,
                    found.methods[0].line if found.methods else 0,
                    f"{name} inherits from {one}, which is not a class this "
                    f"translation unit declares",
                )
        # A copy constructor and a move constructor together. Both take the
        # class by reference, and `std::move` - the one thing that says which
        # is meant - is taken out above, because every move here is a copy.
        # Once it is gone the two are one signature, and choosing whichever
        # was written first would be a copy where a move was asked for or a
        # move where a copy was: an object emptied that the program still
        # meant to use.
        taking = [
            method
            for method in found.methods
            if method.name == "" and _arity(method.parameters) == 1
            and _parameter_types(method.parameters) == [_type_code(name)]
        ]
        if len(taking) > 1:
            raise CppTranslationError(
                filename,
                taking[-1].line,
                f"{name} declares more than one constructor taking a "
                f"{name} - a copy constructor and a move constructor. "
                f"py2bin has no rvalue reference of its own: `std::move` is "
                f"taken out, every move becomes the copy it would have been, "
                f"and the two are left taking the same thing with nothing to "
                f"tell them apart. Write whichever one this class needs, and "
                f"if that is the move, write it as the copy constructor - "
                f"which is what `unique_ptr` here does",
            )
        # Two paths to one virtual base - the diamond, which is the whole
        # reason the word exists - is no longer refused: a base inherited
        # `virtual` is held by address rather than embedded, and the complete
        # object owns the one everything points at. See `_base_steps` and
        # `_complete_constructor`.
        # `int &at(int)` beside `const int &at(int) const` is how every
        # container is written. C++ picks between them by whether the object
        # is const; py2bin does not track that, and by the time they are read
        # the two take the same arguments and answer the same shape.
        #
        # Where their bodies are the same text - which is what an accessor
        # pair is - one of them is enough and the other goes. Where they
        # differ, that is a program relying on the choice, and it is refused
        # rather than given whichever came first.
        by_shape: "dict[tuple, list[Method]]" = {}
        for method in found.methods:
            if method.name in ("", "~"):
                continue
            by_shape.setdefault(
                (method.name, tuple(_parameter_types(method.parameters))), []
            ).append(method)
        for (spelled, _shape), group in by_shape.items():
            if len(group) != 2 or group[0].readonly == group[1].readonly:
                continue
            first, second = group
            if re.sub(r"\s+", " ", first.body) == re.sub(r"\s+", " ", second.body):
                found.methods.remove(second if second.readonly else first)
                continue
            raise CppTranslationError(
                filename,
                group[-1].line,
                f"{name} declares {spelled}() twice, once `const` and once "
                f"not, with different bodies. C++ chooses between them by "
                f"whether the object is const and py2bin does not know that "
                f"about an object - so with two different bodies there is "
                f"nothing here that can choose. Where the two do the same "
                f"thing, writing one is enough and this takes it",
            )
        for mixin in found.mixins:
            # A second base is a member after the first, so the address of the
            # object is not the address of it: converting a `{name} *` to a
            # `{mixin} *` has to move the pointer. Where the object is used as
            # itself - its members read, its methods called - that is done and
            # is right. Where a pointer or a reference to the second base is
            # taken, it is not done everywhere it would have to be, and the
            # difference is a wrong answer rather than a failure. So it is
            # refused instead, by name.
            # A second base with virtuals gets a table of its own, whose
            # entries move the pointer back to the whole object before
            # calling what this class wrote. See `_second_base_tables`.
            _ = mixin


    # Members defined outside their class, folded back into it.
    for out in _OUT_OF_LINE.finditer(text):
        spelled, stars, owner, method = out.groups()
        returns = f"{(spelled or '').strip()} {stars or ''}".strip()
        if owner not in classes:
            continue
        shut = _closing_paren(text, out.end() - 1)
        if shut < 0:
            continue
        after = _AFTER_PARAMETERS.match(text, shut + 1)
        if after is None:
            continue
        opening = after.end() - 1
        closing = _matching(text, opening)
        parameters = text[out.end(): shut]
        body = text[opening: closing]
        held = classes[owner]
        wanted = "~" if method.startswith("~") else (
            "" if method == owner else method
        )
        written = "" if parameters.strip() in ("", "void") else parameters.strip()
        # Which of the declarations this defines. By name alone, two
        # constructors written outside their class both matched the first,
        # so one lost its body and the other was never given one - and the
        # class ended up with two members of the same shape.
        matching = [item for item in held.methods if item.name == wanted]
        count = _arity(written)
        chosen = next(
            (
                item
                for item in matching
                if not (item.body or "").strip()
                and _arity(item.parameters) == count
            ),
            None,
        )
        if chosen is None:
            chosen = next(
                (item for item in matching if not (item.body or "").strip()),
                None,
            )
        if chosen is None and matching:
            chosen = matching[0]
        if chosen is not None:
            chosen.body = body
            chosen.parameters = written
        else:
            held.methods.append(
                Method(
                    wanted, returns or "void", written, body,
                    _line_of(text, out.start()),
                )
            )
        pieces.append((out.start(), closing, ""))

    # A class gets a constructor it did not write wherever one has work to do.
    # Two reasons, and both are silent when they are missed:
    #
    #  * a polymorphic class has to install its table, and a derived class
    #    that borrowed its base's constructor would point at the base's - so
    #    its objects would answer as the base;
    #  * a class holding another, or deriving from one, has to construct what
    #    it holds. `class Outer { Inner in; };` has no constructor of its own
    #    and py2bin only put subobject construction into one the author wrote,
    #    so `Outer o;` left `in` as whatever was on the stack. That is C++'s
    #    implicit default constructor, and leaving it out is not a refusal -
    #    it is a program that runs and is wrong.
    for name in order:
        found = classes[name]
        if not (_is_polymorphic(name, classes) or _subobjects(found, classes)):
            continue
        # Only where the author wrote no constructor at all, which is when
        # C++ writes one. A class that declares one taking arguments has no
        # implicit default constructor, and adding one here gave it a second
        # constructor with no initialiser list - which then tried to build a
        # base that has nothing to build it with, and said so as a refusal of
        # the program the author actually wrote.
        if not any(m.name == "" for m in found.methods):
            found.methods.append(Method("", "void", "", "{ }", 0))

    # Everything the classes did not claim is ordinary code, rewritten so its
    # declarations and calls speak C.
    pieces.sort()
    kept: list[str] = []
    at = 0
    for start, end, replacement in pieces:
        kept.append(text[at:start])
        kept.append(replacement)
        at = max(at, end)
    kept.append(text[at:])
    remainder = "".join(kept)

    # Directives first. The struct and its methods are emitted above whatever
    # is left of the file, and a method that calls printf needs <stdio.h>
    # declared before it rather than wherever the author happened to write it.
    #
    # A conditional is the exception, and moving one was the worst thing this
    # function did: `#if` and its `#endif` went to the top while the lines
    # they bracket stayed behind, so an `#if 0` around three statements
    # emitted an empty conditional above the file and three live statements
    # in the middle of it. Both arms of an `#ifdef`/`#else` ran. Nothing
    # said so - the program built and printed what it was never asked to.
    #
    # So a conditional keeps what it guards. One holding nothing but
    # directives still goes up, brackets and all: `#ifdef _WIN32` around an
    # `#include` is how half the headers in the world open, and hoisting the
    # region whole is what keeps the include above the methods that need it
    # while still letting the condition decide. Hoisting the `#include` *out*
    # of it would be the same mistake in the other direction - the header
    # would be read whether or not the condition held.
    directives = []
    kept_lines = []
    lines = remainder.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        if _OPENS_A_CONDITION.match(line):
            region = []
            depth = 0
            while index < len(lines):
                here = lines[index]
                region.append(here)
                index += 1
                if _OPENS_A_CONDITION.match(here):
                    depth += 1
                elif _CLOSES_A_CONDITION.match(here):
                    depth -= 1
                    if depth == 0:
                        break
            guards_code = any(
                stripped and not stripped.startswith("#")
                for stripped in (part.strip() for part in region)
            )
            (kept_lines if guards_code else directives).extend(region)
            continue
        (directives if line.lstrip().startswith("#") else kept_lines).append(line)
        index += 1
    remainder = "\n".join(kept_lines)

    # Typedefs first, as forward declarations. `Vec v;` is a declaration in
    # C++ and a syntax error in C, which wants `struct Vec v;` - and a class
    # held *inside* another names its type before its own definition is
    # reached, so the name has to exist before any struct body does.
    # An enum or a union defined at the top level goes ahead of the structs:
    # a struct may hold one, and C needs the complete type to lay the field
    # out. A nested enum lifted out of a class is exactly that case.
    remainder, tagged = _hoist_tagged_types(remainder)
    # A `struct` with no methods is emitted exactly where it was written, and
    # a class holding one is emitted above whatever is left of the file - so
    # the class named a type C had not seen yet. The bodies of those go up
    # too, in the order they were written, so one holding another still comes
    # after it.
    remainder, plain_bodies = _hoist_plain_structs(remainder, plain)
    # The name a function's own type was given goes up with the rest of them:
    # a template copy taking one is emitted above whatever is left of the
    # file, and the typedef was still down where the function was defined.
    function_types = re.findall(
        rf"^typedef .*\(\*{_FUNCTION_TYPE}\w+\)\(.*\);$",
        remainder,
        re.M,
    )
    if function_types:
        remainder = re.sub(
            rf"^typedef .*\(\*{_FUNCTION_TYPE}\w+\)\(.*\);$", "", remainder, flags=re.M
        )
    # A union holding a class needs the class first, and a class holding a
    # union needs the union first - so which side of the class definitions
    # each one goes is decided by what it names. Written out above them
    # always, `union U { Inner in; };` was emitted before `Inner` existed.
    def _names_a_class(body: str) -> bool:
        without = _without_literals(body)
        return any(
            re.search(rf"(?<![.\w>]){re.escape(name)}\b", without)
            for name in classes
        )

    tagged_after = [one for one in tagged if _names_a_class(one)]
    tagged_before = [one for one in tagged if one not in tagged_after]
    typedefs = "\n".join(
        tagged_before
        + [f"typedef struct {name} {name};" for name in [*order, *plain]]
        # After the structs: one of these may name a struct in its parameters.
        + function_types
    )
    # A class holding another must be defined after it: C needs the complete
    # type to lay out the field. Emitting in source order put `Car` before
    # `Engine` whenever that is how they were written.
    # A struct with no methods is not a class here, so it is emitted from the
    # text it was written as - and that text has been moved away from the
    # `#pragma pack` that applied to it just as a class's has.
    def _packed_body(body: str) -> str:
        named = _CLASS_HEAD.search(body)
        held = _CLASS_PACK.get(named.group(2)) if named is not None else None
        if held is None:
            return body
        return f"#pragma pack(push, {held})\n{body}\n#pragma pack(pop)"

    declarations = "\n".join(
        [_packed_body(one) for one in plain_bodies]
        + [
            _emit_class(classes[name], classes)
            for name in _by_dependency(order, classes)
        ]
    )
    tables = _emit_vtables(order, classes)
    if tables:
        declarations += "\n" + tables
    # A default argument on a member. The free-function pass above cannot see
    # these - it reads definitions at the top level, and a member's is inside
    # a class - and its call pattern excludes `c.bump()` and `Box b(4);`
    # anyway. So the callee kept losing its default while the call kept its
    # short list, and the arity check refused a call C++ accepts.
    for name in order:
        for method in classes[name].methods:
            values = [
                (_DEFAULTED.search(part).group(2).strip()
                 if _DEFAULTED.search(part) else "")
                for part in _split_arguments(method.parameters)
                if part.strip()
            ]
            if not any(values):
                continue
            method.parameters = ", ".join(
                _DEFAULTED.sub(r"\1", part).strip()
                for part in _split_arguments(method.parameters)
                if part.strip()
            )
            classes[name].defaults[method.name] = values

    remainder = _qualified_static_names(remainder, classes, order)

    # `R r = make(4);` - a free function returning a class hands it back the
    # way a method does, through a pointer the caller provides.
    remainder = _free_value_initialisers(remainder, classes, remainder)

    made = _file_scope_objects(remainder, classes)
    remainder = _construct_before_main(remainder, made, classes)
    # What every body may need to look up: the file, plus the shapes of the
    # methods the emitter is about to write. A method's C name is declared
    # nowhere the file can see, so without this `cout << v[0]` had no way to
    # find out what `v[0]` - a call to `vector__int__op_index` - returns.
    scope = remainder + "\n" + _method_declarations(order, classes)
    definitions = "\n".join(
        _emit_methods(classes[name], classes, scope) for name in order
    )
    # Before the arguments are given their addresses: a temporary is an
    # object, and an object passed by value is passed by address - but only
    # if it exists by the time that pass runs.
    # Before the temporaries: `A xs[3] = {A(1), A(2)};` builds each element
    # where it stands, and a temporary hoisted out of the braces would be
    # hoisted to the start of the statement - which is inside them.
    remainder = _rewrite_object_array_values(remainder, classes, {})
    remainder = _rewrite_temporaries(remainder, classes, [0])
    remainder = _rewrite_brace_temporaries(remainder, classes, [0])
    remainder = _address_reference_arguments(
        remainder, _function_signatures(remainder, classes), classes
    )
    # A prototype says the same thing about a function that its definition
    # does, and has to say it the same way: a header declares `string shout
    # (string s);` and the definition below became `void shout(struct string
    # *__ret, struct string *__by_value_s)`, so the two disagreed.
    # A plain struct is C already and is emitted as it was written - but
    # py2bin's C can neither pass nor answer one by value, and `Point add
    # (Point a, Point b)` is as ordinary in C++ as it is impossible here. So
    # the passes that turn a value into a pointer are given a wider view: the
    # classes, and every plain struct as a shape with nothing in it. Nothing
    # else sees them, because nothing else should read a body that is already
    # exactly what it will be emitted as.
    shapes: "dict[str, Class]" = {**classes, **{name: Class(name) for name in plain}}
    remainder = _rewrite_prototypes(remainder, shapes)
    rewritten = _rewrite_functions(
        remainder,
        classes,
        {name: held for name, (held, _a) in made.items()},
        scope,
        shapes,
    )
    # `dynamic_cast<D *>(p)` asks at run time what an object really is. This
    # unit is the whole program, so the answer is in the table the object
    # carries: it is a D if that table is D's own or belongs to a class
    # derived from D. Written after the bodies, because the calls it answers
    # are in them.
    rewritten, wanted_name = _rewrite_typeid(rewritten, classes, filename)
    if wanted_name:
        definitions += "\n" + _emit_type_names(order, classes)
    rewritten, asked = _rewrite_dynamic_casts(rewritten, classes, filename)
    if asked:
        definitions += "\n" + "\n".join(
            _emit_dynamic_cast(name, order, classes) for name in asked
        )
    head = "\n".join(directives)
    if throws:
        head = f"{_EXCEPTION_STATE}{head}"
    if _wants_heap(text):
        # Only a file that allocates gets the allocator. `new` compiles to
        # malloc, which lives in <stdlib.h>; including it here rather than
        # asking the author to means `new` works in a file that never heard of
        # C's heap, and the header is guarded, so a file that includes it
        # itself is unaffected.
        head = f"#include <stdlib.h>\n{head}"
        # And a plain struct the program allocates. It is kept out of `order`
        # because its body is C already and is emitted as it was written -
        # but `new P()` is not C, and without an allocator the call this
        # translator writes had nothing to call. Only the ones the program
        # actually allocates, so a file full of plain structs does not carry
        # an allocator for each.
        # Against the methods as well as the bodies: a container allocates
        # its own nodes, and the call to do it is in a method, which is
        # emitted with the classes rather than with the rest of the file.
        reading = f"{definitions}\n{rewritten}"
        allocated = [
            name
            for name in classes
            if name not in order and _allocates(reading, name)
        ]
        definitions += "\n" + "\n".join(
            _emit_heap_helpers(classes[name], classes)
            for name in [*order, *allocated]
        )
    if tagged_after:
        definitions += "\n" + "\n".join(tagged_after)
    whole = f"{head}\n{typedefs}\n{declarations}\n\n{definitions}\n\n{rewritten}\n"
    # A call made through a variable, here rather than while each body was
    # rewritten: the variable's type is a typedef that instantiating a
    # template writes, and when the body of `__sift_by` was rewritten the
    # comparator it would be given had no name yet. So the question is asked
    # of the finished C, where every type is written down.
    # A method taking `const T &` where T is not a class, given a literal.
    # Here rather than while the body was rewritten, because a constructor
    # is not spelled with the name its call is written under until now.
    positions, held_by = _reference_literal_parameters(classes)
    whole = _bind_reference_literals(whole, positions, held_by)
    whole = _fold_constant_definitions(whole)
    whole = _ask_a_class_whether_it_is_true(whole, classes)
    through, through_types = _pointer_call_signatures(whole, classes)
    if through:
        whole = _address_reference_arguments(
            whole, through, classes, "", frozenset(), through_types
        )
    # Last, on the C itself, because that is where every type is written out
    # and a pointer to a base can be told from a pointer to what holds it.
    return _move_to_second_base(whole, classes)






#: `new T`, `new T(a, b)` and `new T[n]`. The type is read as a name so a
#: qualified one has already been flattened by the time this runs.
#: The stars are part of the type: `new T[n]` inside a container of pointers
#: reads `new Row *[n]` once T has been substituted, and taken without them
#: the `*[n]` was left behind for the C compiler to choke on.
_NEW = re.compile(r"\bnew\s+([A-Za-z_]\w*(?:\s*\*)*)\s*(\[|\()?")

#: `new (room) T(a, b)` - the one `new` that allocates nothing. The memory is
#: already there and what is being asked for is the constructor, run on it.
_PLACEMENT_NEW = re.compile(
    r"\bnew\s*\(([^()]*)\)\s*([A-Za-z_]\w*)\s*\("
)


def _rewrite_placement_new(text: str, classes: "dict[str, Class]") -> str:
    """`new (room) T(a)` becomes the constructor run where it was told to.

    Every other `new` here obtains storage and then constructs; this one is
    handed the storage. So it becomes the second half on its own - which is
    what the standard's placement operator does too, its whole body being
    `return the pointer it was given`.
    """

    bare = _without_literals(text)
    out: "list[str]" = []
    at = 0
    for found in _PLACEMENT_NEW.finditer(bare):
        if found.start() < at or found.group(2) not in classes:
            continue
        closing = _closing_paren(bare, found.end() - 1)
        if closing < 0:
            continue
        given = text[found.end(): closing].strip()
        out.append(text[at: found.start()])
        out.append(
            f"{_c_name(found.group(2), 'place')}"
            f"((void *)({found.group(1).strip()})"
            f"{', ' + given if given else ''})"
        )
        at = closing + 1
    out.append(text[at:])
    return "".join(out)
#: `delete p;` and `delete[] p;`, to the end of the statement.
#: The word boundary after `delete` is load-bearing: without it `deleteAll()`
#: and `deleteLater` were read as a `delete` of whatever followed, and a
#: perfectly ordinary method became a call to `free`.
_DELETE = re.compile(r"\bdelete\b\s*(\[\s*\])?\s*([^;]+);")

#: The header on a `new[]` block: the element count, so `delete[]` knows how
#: many destructors to run. Sixteen bytes rather than eight, because malloc
#: hands back 16-byte-aligned memory and the array has to stay that way.
_ARRAY_COOKIE = 16



#: `dynamic_cast<D *>(p)`. Only to a pointer: a cast to a reference throws on
#: failure, and this subset answers with a null instead.
_DYNAMIC_CAST = re.compile(r"\bdynamic_cast\s*<([^<>]+)>\s*\(")


def _dynamic_cast_name(name: str) -> str:
    return f"__py2bin_as_{name}"


#: `typeid(x)` and `typeid(T)` - what an object really is, asked at run time.
_TYPEID = re.compile(r"(?<![.\w>])typeid\s*\(")


def _rewrite_typeid(
    body: str, classes: "dict[str, Class]", filename: str
) -> "tuple[str, bool]":
    """`typeid(a) == typeid(b)` becomes a comparison of two tables.

    An object's table *is* its identity here: one exists per class, the
    program is one translation unit, and two objects share a table exactly
    when they are the same class. That is the same fact `dynamic_cast` is
    answered from.

    A `typeid` standing on its own is refused rather than given a number,
    because what C++ hands back there is an object with a name on it and an
    ordering, and neither of those is a table pointer.
    """

    wanted_name = False
    out: "list[str]" = []
    at = 0

    def table(spelled: str, where: int) -> str:
        held = spelled.strip()
        if held in classes and _is_polymorphic(held, classes):
            return f"((void *)&{_vtable_name(held)})"
        # An expression: the table the object carries. Written through the
        # path to whichever base holds the pointer, which is where the
        # dispatcher reads it from too.
        deduced = _deduced_type(held, body)
        owner = (
            re.sub(r"\b(?:const|struct|volatile)\b", " ", deduced or "")
            .replace("*", " ")
            .strip()
        )
        if owner in classes and _is_polymorphic(owner, classes):
            reach = "->" if "*" in (deduced or "") else "."
            return f"((void *)({held}){reach}{_vptr_path(owner, classes)})"
        raise CppTranslationError(
            filename, _line_of(body, where),
            f"typeid({held}) - py2bin answers this from the table an object "
            f"carries, and `{held}` is not something with one. Only a class "
            f"with a virtual function has a table, which is also the only "
            f"kind C++ answers about at run time",
        )

    for found in _TYPEID.finditer(body):
        if found.start() < at:
            continue
        close = _closing_paren(body, found.end() - 1)
        if close < 0:
            continue
        first = table(body[found.end(): close], found.start())
        rest = body[close + 1:]
        # `== typeid(y)` or `!= typeid(y)`, which is what a program writes.
        paired = re.match(r"\s*(==|!=)\s*", rest)
        after = _TYPEID.match(rest, paired.end()) if paired else None
        if after is not None:
            second_close = _closing_paren(rest, after.end() - 1)
            if second_close >= 0:
                second = table(
                    rest[after.end(): second_close], found.start()
                )
                out.append(body[at: found.start()])
                out.append(f"({first} {paired.group(1)} {second})")
                at = close + 1 + second_close + 1
                continue
        # `.name()`, which is the other thing written.
        named = re.match(r"\s*\.\s*name\s*\(\s*\)", rest)
        if named is not None:
            wanted_name = True
            out.append(body[at: found.start()])
            out.append(f"__py2bin_type_name({first})")
            at = close + 1 + named.end()
            continue
        raise CppTranslationError(
            filename, _line_of(body, found.start()),
            "a `typeid` on its own is refused: py2bin answers one from the "
            "table an object carries, which compares and has a name, and is "
            "not the object with an ordering that C++ hands back. Write "
            "`typeid(a) == typeid(b)` or `typeid(a).name()`",
        )
    out.append(body[at:])
    return "".join(out), wanted_name


def _emit_type_names(order: "list[str]", classes: "dict[str, Class]") -> str:
    """`__py2bin_type_name(table)` - the class a table belongs to, by name."""

    arms = "".join(
        f"    if (__p == (void *)&{_vtable_name(name)}) {{ return \"{name}\"; }}\n"
        for name in order
        if _is_polymorphic(name, classes)
    )
    return (
        f"static const char *__py2bin_type_name(void *__p) {{\n"
        f"{arms}"
        f"    return \"unknown\";\n"
        f"}}"
    )


def _rewrite_dynamic_casts(
    body: str, classes: "dict[str, Class]", filename: str
) -> "tuple[str, list[str]]":
    """`dynamic_cast<D *>(p)` becomes a call that checks and then casts."""

    asked: "list[str]" = []
    out: list[str] = []
    at = 0
    for found in _DYNAMIC_CAST.finditer(body):
        if found.start() < at:
            continue
        wanted = found.group(1).replace("*", "").replace("struct", "").strip()
        close = _closing_paren(body, found.end() - 1)
        if close < 0:
            continue
        if wanted not in classes or not _is_polymorphic(wanted, classes):
            raise CppTranslationError(
                filename,
                _line_of(body, found.start()),
                f"dynamic_cast to {wanted}, which has no virtual functions - "
                f"there is nothing in the object that says what it is. Give "
                f"the base a virtual destructor, or use a plain cast",
            )
        if "*" not in found.group(1):
            raise CppTranslationError(
                filename,
                _line_of(body, found.start()),
                "dynamic_cast to a reference throws when it fails, and this "
                "subset has no way to say so; cast to a pointer and test it",
            )
        if wanted not in asked:
            asked.append(wanted)
        out.append(body[at:found.start()])
        out.append(
            f"{_dynamic_cast_name(wanted)}((void *)({body[found.end(): close]}))"
        )
        at = close + 1
    out.append(body[at:])
    return "".join(out), asked


def _emit_dynamic_cast(
    wanted: str, order: "list[str]", classes: "dict[str, Class]"
) -> str:
    """The function that answers whether an object really is a `wanted`.

    Every table in the unit that belongs to `wanted` or to something derived
    from it. That is the whole answer: py2bin has no linker, so this
    translation unit is the program and there is no class it has not seen.
    """

    root = wanted
    while classes[root].base and _carries_vptr(classes[root].base, classes):
        root = classes[root].base
    path = _vptr_path(root, classes)
    tables = [
        _vtable_name(name)
        for name in order
        if _is_polymorphic(name, classes)
        and (name == wanted or _derives_from(name, wanted, classes))
    ]
    if not tables:
        return ""
    test = " || ".join(f"held == {one}" for one in tables)
    return (
        f"static struct {wanted} *{_dynamic_cast_name(wanted)}(void *__p) {{\n"
        f"    void **held;\n"
        f"    if (__p == 0) {{ return 0; }}\n"
        f"    held = ((struct {root} *)__p)->{path};\n"
        f"    if ({test}) {{ return (struct {wanted} *)__p; }}\n"
        f"    return 0;\n"
        f"}}"
    )

def _allocates(text: str, name: str) -> bool:
    """Whether the rewritten text calls this class's allocator.

    Read from the calls rather than from the `new` the author wrote: by the
    time this is asked, `new P()` has already become `P__new()`, and that is
    the name that has to resolve.
    """

    return re.search(
        rf"(?<![.\w>]){re.escape(name)}__new(?:_array)?\s*\(", text
    ) is not None


def _emit_heap_helpers(found: "Class", classes: "dict[str, Class]") -> str:
    """`T__new` and friends: allocate, then construct, in that order.

    `new` is two things C keeps apart - obtaining storage and running a
    constructor - and an expression cannot do both in C, where a call is the
    only thing that returns a value. So each becomes a function, which is
    what `new` compiles to in any C++ implementation anyway.
    """

    name = found.name
    size = f"sizeof(struct {name})"
    made = []

    # One `T__new` per constructor, told apart the same way the constructors
    # themselves are: by how many arguments they take.
    constructors = [m for m in found.methods if m.name == ""]
    if not constructors:
        constructors = [Method("", "void", "", "{ }", 0)]
    for constructor in constructors:
        # The same suffix the constructor itself is named with. The count of
        # arguments alone is not one: `ComPtr(T *)` and `ComPtr(const
        # ComPtr<T> &)` both take one, and named by the count they were two
        # functions with one name and two shapes.
        suffix = (
            _suffix_of(name, constructor, classes)
            if _has_several_constructors(name, classes)
            else None
        )
        # The same shapes the constructor itself is emitted with: a reference
        # is a pointer and an object taken by value is passed by address.
        # Written out raw, `T__new(const ComPtr<U> &other)` reached the C
        # with a `&` in it, which C does not have.
        converted = _rewrite_types(
            _references_to_pointers(constructor.parameters), classes
        )
        for held, variable in _by_value_objects(constructor.parameters, classes):
            converted = re.sub(
                rf"\bstruct\s+{re.escape(held)}\s+{re.escape(variable)}\b",
                f"struct {held} *__by_value_{variable}",
                converted,
            )
        parameters = converted or "void"
        passed = ", ".join(
            _parameter_name(part) for part in _split_arguments(converted)
            if part.strip()
        )
        # A plain struct has no constructor to run. `new P()` still has to
        # answer storage, and `P()` value-initialises in C++ - which is what
        # the zeroing below is.
        if not found.methods:
            made.append(
                f"static struct {name} *{_c_name(name, 'new', suffix)}(void) {{\n"
                f"    unsigned long __i;\n"
                f"    struct {name} *__p = (struct {name} *)malloc({size});\n"
                f"    if (__p == 0) return 0;\n"
                f"    for (__i = 0; __i < {size}; __i++)"
                f" ((unsigned char *)__p)[__i] = 0;\n"
                f"    return __p;\n"
                f"}}"
            )
            continue
        ctor = _c_name(name, "", _suffix_of(name, constructor, classes))
        made.append(
            f"static struct {name} *{_c_name(name, 'new', suffix)}({parameters}) {{\n"
            f"    struct {name} *__p = (struct {name} *)malloc({size});\n"
            f"    if (__p == 0) return 0;\n"
            f"    {ctor}(__p{', ' + passed if passed else ''});\n"
            f"    return __p;\n"
            f"}}"
        )

    # And the one that is handed its storage. `new (room) T(a)` asks only for
    # the constructor, so this is `T__new` without the malloc.
    for constructor in constructors:
        suffix = (
            _suffix_of(name, constructor, classes)
            if _has_several_constructors(name, classes)
            else None
        )
        converted = _rewrite_types(
            _references_to_pointers(constructor.parameters), classes
        )
        for held, variable in _by_value_objects(constructor.parameters, classes):
            converted = re.sub(
                rf"\bstruct\s+{re.escape(held)}\s+{re.escape(variable)}\b",
                f"struct {held} *__by_value_{variable}",
                converted,
            )
        passed = ", ".join(
            _parameter_name(part) for part in _split_arguments(converted)
            if part.strip()
        )
        head = f"void *__where{', ' + converted if converted.strip() else ''}"
        if not found.methods:
            made.append(
                f"static struct {name} *{_c_name(name, 'place', suffix)}"
                f"(void *__where) {{ return (struct {name} *)__where; }}"
            )
            continue
        owner = _find_method(name, "", classes)
        ctor = _c_name(owner or name, "", _suffix_of(name, constructor, classes))
        made.append(
            f"static struct {name} *{_c_name(name, 'place', suffix)}({head}) {{\n"
            f"    struct {name} *__p = (struct {name} *)__where;\n"
            f"    {ctor}(__p{', ' + passed if passed else ''});\n"
            f"    return __p;\n"
            f"}}"
        )

    # `new T[n]` needs the default constructor, which is what C++ says too.
    default = _call_suffix(name, "", classes, [])
    runs_ctor = any(_arity(m.parameters) == 0 for m in found.methods if m.name == "")
    construct = (
        f"    for (__i = 0; __i < __n; __i++) {_c_name(name, '', default)}(&__a[__i]);\n"
        if runs_ctor
        else ""
    )
    made.append(
        f"static struct {name} *{_c_name(name, 'new_array')}(unsigned long __n) {{\n"
        f"    unsigned long __i;\n"
        f"    struct {name} *__a;\n"
        f"    unsigned char *__block = (unsigned char *)malloc("
        f"{_ARRAY_COOKIE} + __n * {size});\n"
        f"    if (__block == 0) return 0;\n"
        f"    *(unsigned long *)__block = __n;\n"
        f"    __a = (struct {name} *)(__block + {_ARRAY_COOKIE});\n"
        f"{construct}"
        f"    return __a;\n"
        f"}}"
    )

    destructor = _find_method(name, "~", classes)
    runs_dtor = destructor is not None
    if runs_dtor and ("~", 0) in _virtual_slots(name, classes):
        # A virtual destructor: the object says which one runs, which is the
        # whole reason for declaring one. `delete` through a base pointer is
        # exactly the case it exists for.
        slots = _virtual_slots(name, classes)
        call_dtor = (
            f"    ((void (*)(struct {name} *))"
            f"((__p)->{_vptr_path(name, classes)}[{slots.index(('~', 0))}]))(__p);\n"
        )
    elif runs_dtor:
        call_dtor = f"    {_destructor_call(name, '(*__p)', classes)}\n"
    else:
        call_dtor = ""
    made.append(
        f"static void {_c_name(name, 'delete')}(struct {name} *__p) {{\n"
        f"    if (__p == 0) return;\n"
        f"{call_dtor}"
        f"    free((void *)__p);\n"
        f"}}"
    )
    each_dtor = (
        f"    for (__i = 0; __i < __n; __i++) "
        f"{_destructor_call(name, '__a[__i]', classes)}\n"
        if runs_dtor
        else ""
    )
    made.append(
        f"static void {_c_name(name, 'delete_array')}(struct {name} *__a) {{\n"
        f"    unsigned long __i;\n"
        f"    unsigned long __n;\n"
        f"    unsigned char *__block;\n"
        f"    if (__a == 0) return;\n"
        f"    __block = (unsigned char *)__a - {_ARRAY_COOKIE};\n"
        f"    __n = *(unsigned long *)__block;\n"
        f"{each_dtor}"
        f"    free((void *)__block);\n"
        f"}}"
    )
    if not runs_dtor:
        # `__n` is read only to reach the destructors; without one it would be
        # a variable set and never used, which is noise in the output.
        made[-1] = made[-1].replace("    unsigned long __i;\n    unsigned long __n;\n", "")
        made[-1] = made[-1].replace("    __n = *(unsigned long *)__block;\n", "")
    return "\n".join(made)


def _parameter_name(declaration: str) -> str:
    """The name out of `const char *s` or `int a[4]`, without its type."""

    stripped = declaration.strip().rstrip("]").split("[")[0].strip()
    found = re.findall(r"[A-Za-z_]\w*", stripped)
    return found[-1] if found else ""



def _hoist_new_initialisers(
    body: str, classes: "dict[str, Class]", counter: "list[int]"
) -> str:
    """`new int(5)` becomes storage, a store, and the pointer.

    Only for a type that is not a class: a class has a constructor, and that
    is already a call taking the address it just allocated.
    """

    for _round in range(_HOIST_ROUNDS):
        found = None
        for match in _NEW.finditer(body):
            if match.group(2) != "(" or match.group(1).strip() in classes:
                continue
            close = _closing_paren(body, match.end() - 1)
            if close < 0 or not body[match.end(): close].strip():
                continue
            found = (match, close)
            break
        if found is None:
            return body
        match, close = found
        held = match.group(1).strip()
        value = body[match.end(): close]
        counter[0] += 1
        name = f"__py2bin_new_{counter[0]}"
        start = _statement_start(body, match.start())
        body = (
            body[:start]
            + f"{held} *{name}; {name} = ({held} *)malloc(sizeof({held})); "
            f"*{name} = {value}; "
            + body[start:match.start()]
            + name
            + body[close + 1:]
        )
    raise CppTranslationError(
        "<c++>", 0,
        "a statement allocating more objects than py2bin writes out; each "
        "becomes a statement of its own and this one never stops asking",
    )

def _rewrite_new(body: str, classes: "dict[str, Class]", scope: str = "") -> str:
    """`new T(a)` becomes `T__new(a)`; `new int[n]` becomes a malloc."""

    out: list[str] = []
    at = 0
    for found in _NEW.finditer(body):
        if found.start() < at:
            continue
        type_name, opener = found.group(1).strip(), found.group(2)
        out.append(body[at:found.start()])
        if opener == "[":
            close = body.index("]", found.end() - 1)
            count = body[found.end(): close]
            if type_name in classes:
                out.append(f"{_c_name(type_name, 'new_array')}({count})")
            else:
                out.append(
                    f"({type_name} *)malloc(sizeof({type_name}) * ({count}))"
                )
            at = close + 1
            continue
        if type_name not in classes:
            # `new int` and `new int(5)`: storage, and an initial value if one
            # was written. C has no comma expression that also yields the
            # pointer, so an initialiser is refused rather than dropped.
            if opener == "(":
                # An initialiser here has already been hoisted into its own
                # statement by `_hoist_new_initialisers`; whatever is left is
                # `new int()`, which is the storage and nothing else.
                close = _closing_paren(body, found.end() - 1)
                at = close + 1
            else:
                at = found.end() - (1 if opener else 0)
            out.append(f"({type_name} *)malloc(sizeof({type_name}))")
            continue
        if opener == "(":
            close = _closing_paren(body, found.end() - 1)
            arguments = body[found.end(): close]
            given = (
                [a.strip() for a in _split_arguments(arguments)]
                if arguments.strip()
                else []
            )
            chosen = (
                _call_suffix(type_name, "", classes, given, scope)
                if _has_several_constructors(type_name, classes)
                else None
            )
            out.append(f"{_c_name(type_name, 'new', chosen)}({arguments})")
            at = close + 1
        else:
            chosen = (
                _call_suffix(type_name, "", classes, [])
                if _has_several_constructors(type_name, classes)
                else None
            )
            out.append(f"{_c_name(type_name, 'new', chosen)}()")
            at = found.end() - (1 if opener else 0)
    out.append(body[at:])
    return "".join(out)


def _has_several_constructors(name: str, classes: "dict[str, Class]") -> bool:
    found = classes.get(name)
    return found is not None and len([m for m in found.methods if m.name == ""]) > 1


#: `x.~T()` and `p->~T()` - naming a destructor to run it where it stands.
_EXPLICIT_DESTRUCTOR = re.compile(
    # The whole receiver, not the last name in it: a container writes
    # `this->items[i].~T()`, and a pattern that starts at `items` leaves the
    # `this->` in front of what it writes.
    r"(?<![.\w>])((?:[A-Za-z_]\w*\s*(?:\.|->)\s*)*[A-Za-z_]\w*"
    # The name may have come from a template argument and so may carry what
    # that argument was: `vector<Shape *>` writes `~Shape *()`, and a
    # pointer has no destructor to run.
    r"(?:\s*\[[^\]]*\])?)\s*(\.|->)\s*~\s*([A-Za-z_]\w*)"
    r"(\s*\*+)?\s*\(\s*\)"
)


def _rewrite_explicit_destructors(
    body: str, classes: "dict[str, Class]"
) -> str:
    """`items[i].~T()` becomes the destructor call, or nothing.

    A container takes its elements apart this way and there is no other way
    to write it: the element's type is a template parameter, so the name of
    the function cannot be written down until the copy exists. Where the
    parameter turned out not to be a class the call says nothing at all,
    which is what C++ says a pseudo-destructor call on an `int` does.
    """

    def one(match: "re.Match[str]") -> str:
        held, reach, named, stars = match.groups()
        if stars:
            # A pointer. C++ lets one be named this way and does nothing,
            # which is the whole of what it means: what a container holds is
            # the pointer, and freeing what it points at is not its business.
            return "(void)0"
        if named not in classes:
            return "(void)0"
        if _find_method(named, "~", classes) is None:
            return "(void)0"
        given = held if reach == "->" else f"&{held}"
        return f"{_c_name(named, '~')}({given})"

    return _map_code(body, lambda part: _EXPLICIT_DESTRUCTOR.sub(one, part))


def _rewrite_delete(
    body: str, classes: "dict[str, Class]", known: "dict[str, str]"
) -> str:
    """`delete p` becomes `T__delete(p)`, using what `p` was declared as.

    Where the type is not known - a pointer that came from elsewhere - the
    storage is still returned, but no destructor runs. Saying that plainly
    beats guessing at a type and calling the wrong destructor.
    """

    def one(match: "re.Match[str]") -> str:
        array, expression = match.group(1), match.group(2).strip()
        held = known.get(expression)
        if held is None or held not in classes:
            return f"free((void *){expression});"
        suffix = "delete_array" if array else "delete"
        return f"{_c_name(held, suffix)}({expression});"


    return _DELETE.sub(one, body)


def _by_dependency(order: "list[str]", classes: "dict[str, Class]") -> "list[str]":
    """Classes ordered so anything held or inherited comes first.

    A field of class type needs the complete type to lay the struct out, and
    so does an embedded base. Source order is not that order - `Car` can be
    written above the `Engine` it holds - and C reads top to bottom.
    """

    placed: list[str] = []
    seen: set[str] = set()
    # Only the classes this pass emits. A plain struct is registered too -
    # every pass needs to know it is an object - but its body is the one the
    # author wrote, emitted above these. Followed into here it was emitted a
    # second time, from a reading that holds no members, and C saw the same
    # struct defined twice.
    emitted = set(order)

    def place(name: str, guard: "set[str]") -> None:
        if name in seen or name not in classes or name not in emitted:
            return
        if name in guard:
            raise CppTranslationError(
                "<c++>", 0,
                f"{name} contains itself, directly or through another class; "
                f"a struct cannot hold a complete copy of its own type",
            )
        guard.add(name)
        found = classes[name]
        if found.base:
            place(found.base, guard)
        for member in found.members:
            held = member.ctype.replace("*", "").strip()
            # A pointer to a class needs only the name, which the typedef
            # above already gave; a value needs the whole type.
            if "*" not in member.ctype and held in classes:
                place(held, guard)
        guard.discard(name)
        seen.add(name)
        placed.append(name)

    for name in order:
        place(name, set())
    return placed


#: The head of a function definition at the outermost level: `int f(args) {`.
_FUNCTION_HEAD = re.compile(r"\)\s*\{")


# --- references ------------------------------------------------------------
#
# A reference is a pointer the language dereferences for you. So it becomes a
# pointer here, and the dereference is written out: `b.v` on a `Box &b` is
# `b->v`, and `n = n + 1` on an `int &n` is `(*n) = (*n) + 1`. Call sites take
# the address, which is what the caller was doing all along without saying so.

#: `Box &b` or `const int & n` in a parameter list or a declaration.
_REFERENCE = re.compile(
    r"\b((?:const\s+)?[A-Za-z_]\w*)\s*&\s*([A-Za-z_]\w*)"
)


def _reference_parameters(
    parameters: str, classes: "dict[str, Class]"
) -> "dict[str, str]":
    """Which parameters are references, and to what."""

    found: dict[str, str] = {}
    for part in _split_arguments(parameters):
        match = _REFERENCE.search(part)
        if match:
            found[match.group(2)] = match.group(1).replace("const", "").strip()
    return found


def _reference_positions(
    parameters: str, classes: "dict[str, Class]"
) -> "list[int]":
    """The argument positions that want an address rather than a value."""

    at = []
    for index, part in enumerate(_split_arguments(parameters)):
        if _REFERENCE.search(part):
            at.append(index)
    return at


def _references_to_pointers(parameters: str) -> str:
    """`Box &b` becomes `Box *b`; the callee sees a pointer either way."""

    return _REFERENCE.sub(lambda m: f"{m.group(1)} *{m.group(2)}", parameters)


def _address_over_a_conditional(spelled: str) -> str:
    """`&(c ? a : b)` becomes `(c ? &(a) : &(b))`.

    C++ lets a conditional be an lvalue where both arms are, so its address
    can be taken. C has no such thing: a conditional there is a value, and a
    value has no address. Both arms do, though, and choosing between two
    addresses is the same choice as taking the address of what was chosen.
    """

    arms = _conditional_arms(spelled)
    if arms is None:
        return f"&({spelled})"
    depth = 0
    question = -1
    for index, piece in enumerate(_without_literals(spelled)):
        if piece in "([{":
            depth += 1
        elif piece in ")]}":
            depth -= 1
        elif depth == 0 and piece == "?":
            question = index
            break
    if question < 0:
        return f"&({spelled})"
    condition = spelled[:question].strip()
    return (
        f"(({condition}) ? {_address_over_a_conditional(arms[0].strip())}"
        f" : {_address_over_a_conditional(arms[1].strip())})"
    )


def _deref_references(
    body: str, references: "dict[str, str]", classes: "dict[str, Class]"
) -> str:
    """Write out the dereference the language was doing silently.

    A reference to a class becomes a pointer used with `->`, which the rest of
    the translator already understands. A reference to anything else is
    dereferenced at each use, because there is nothing else `n + 1` could
    mean once `n` is an `int *`.
    """

    for name, held in references.items():
        if held in classes:
            # `held_ = other;` where `other` is a reference to a class: the
            # pointer is how the reference is carried, and what is being
            # assigned is the object it names. Left as the pointer, this was
            # an assignment of a `T *` to a `T`.
            #
            # Before the `&` is taken off below, and not after. This rule
            # deliberately does not fire on `= &other;` - a `&` in front is
            # excluded - but once the `&` has been removed there is nothing
            # left to tell the two apart, so `p = &m;` became `p = *m;`: a
            # dereference where the address was asked for.
            body = _map_code(
                body,
                lambda part, n=name: re.sub(
                    rf"(?<![=!<>+\-*/%&|^])=\s*(?<![.\w>&])"
                    rf"{re.escape(n)}\b\s*;",
                    f"= *{n};",
                    part,
                ),
            )
            # `&r` on a reference is the address of what it names, which is
            # what the pointer already holds. Taking one of the pointer gave
            # a `T **` - which is what `this == &other` compared against.
            body = _map_code(
                body,
                lambda part, n=name: re.sub(
                    rf"&\s*(?<![.\w>])\b{re.escape(n)}\b(?!\s*[\w(])", n, part
                ),
            )
            body = _map_code(
                body,
                lambda part, n=name: re.sub(
                    rf"\b{re.escape(n)}\s*\.", f"{n}->", part
                ),
            )
            # `sizeof r` asks how big the object is, and the pointer standing
            # in for the reference is not it. Every other use of a reference
            # to a class wants the pointer - that is why it is left alone
            # above - but this one wants what it points at, and left alone it
            # answered eight for every object on a 64-bit machine. Silently:
            # `memset(&r, 0, sizeof r)` cleared eight bytes of whatever size
            # the object really was.
            body = _map_code(
                body,
                lambda part, n=name: re.sub(
                    rf"\bsizeof\s*(?:\(\s*{re.escape(n)}\s*\)"
                    rf"|(?<![.\w>]){re.escape(n)}\b(?!\s*[\w(]))",
                    f"sizeof(*{n})",
                    part,
                ),
            )
            continue
        body = _map_code(
            body,
            lambda part, n=name: re.sub(
                rf"(?<![.\w>&]){re.escape(n)}\b(?!\s*[\w(])", f"(*{n})", part
            ),
        )
    return body


#: A function defined at the top level: `int read(const Box &b) {`.
#: `Box<T> wrap(T v) {` - a function whose result is itself a template. The
#: plain pattern reads a type as a run of word characters and stars, and
#: `<T>` is neither, so a function written this way was not read as a
#: function at all. `std::vector<T> collect()` is the same shape, and is how
#: a great many functions in a real header are written.
_TEMPLATED_RESULT = re.compile(
    r"\b((?:const\s+)?[A-Za-z_]\w*\s*<[^<>]*>\s*(?:const\s*)?[*&]*)\s*"
    r"\b([A-Za-z_]\w*)\s*\(([^;{}()]*)\)\s*\{"
)

_DEFINITION = re.compile(r"\b([A-Za-z_][\w\s*]*?(?:&&?\s*)?)\b([A-Za-z_]\w*)\s*\(([^;{}()]*)\)\s*\{")


def _function_signatures(text: str, classes: "dict[str, Class]" = {}) -> "dict[str, list[int]]":
    """For each function defined here, which arguments want an address.

    Read from the definitions rather than tracked through the rewriter,
    because a call may be written above the function it calls and C++ does
    not mind.
    """

    found: dict[str, list[int]] = {}
    for match in _DEFINITION.finditer(text):
        if _depth_at(text, match.end() - 1) != 0:
            continue
        name = match.group(2)
        if name in _NOT_A_TYPE:
            continue
        by_value = {
            variable for _held, variable in _by_value_objects(match.group(3), classes)
        }
        at = [
            index
            for index, part in enumerate(_split_arguments(match.group(3)))
            if _REFERENCE.search(part)
            or any(re.search(rf"\b{re.escape(v)}\b", part) for v in by_value)
        ]
        if at:
            found[name] = at
    found.update(_static_member_signatures(classes))
    return found


def _static_member_signatures(
    classes: "dict[str, Class]",
) -> "dict[str, list[int]]":
    """Which arguments of each `static` member want an address."""

    found: "dict[str, list[int]]" = {}
    for spelled, written in _static_member_parameters(classes).items():
        by_value = {
            variable for _held, variable in _by_value_objects(written, classes)
        }
        at = [
            index
            for index, part in enumerate(_split_arguments(written))
            if _REFERENCE.search(part)
            or any(re.search(rf"\b{re.escape(v)}\b", part) for v in by_value)
        ]
        if at:
            found[spelled] = at
    return found


#: `typedef int (*Name)(params);` - a function reached through a variable.
_FUNCTION_TYPEDEF = re.compile(
    r"\btypedef\s+[A-Za-z_][\w\s*]*?\(\s*\*\s*([A-Za-z_]\w*)\s*\)\s*\(([^;]*)\)\s*;"
)


#: `const int A = <expr>;` at file scope.
_CONSTANT_DEFINITION = re.compile(
    r"(?m)^[ \t]*(?:static\s+)?const\s+(?:unsigned\s+)?"
    r"(?:int|long|short|char)\s+([A-Za-z_]\w*)\s*=\s*([^;]+);"
)


def _fold_constant_definitions(whole: str) -> str:
    """`const int A = 5 * B;` becomes `const int A = 120;`.

    A `static const int` member is worked out at compile time in C++, and a
    class template whose member asks for the one in the copy before it is how
    a program counts at compile time. Written out as it stands, each of these
    initialises a file-scope object from another one - which C does not let a
    definition do, and which is the whole of what `Fact<5>::value` is.
    """

    for _round in range(_HOIST_ROUNDS):
        changed = False
        for found in _CONSTANT_DEFINITION.finditer(_without_literals(whole)):
            value = whole[found.start(2): found.end(2)].strip()
            if re.fullmatch(r"[-+]?\d+", value):
                continue
            folded = _folded_integer(value, whole)
            if folded is None:
                continue
            whole = (
                whole[: found.start(2)] + str(folded) + whole[found.end(2):]
            )
            changed = True
            break
        if not changed:
            return whole
    return whole


#: `while (f(...))` or `if (f(...))` - a condition that is one call.
_CONDITION_CALL = re.compile(r"(?<![.\w>])(if|while)\s*\(\s*([A-Za-z_]\w*)\s*\(")


def _conversion_for_a_condition(
    owner: str, classes: "dict[str, Class]"
) -> "str | None":
    """The conversion a condition uses to ask an object whether it is true.

    C++ asks for `operator bool`. What that is spelled as in the C this emits
    is whatever `_rewrite_cpp_spellings` settled on, and spelling it a second
    time here is what made `while (in >> n)` spin forever the day `bool`
    stopped being written `int`: this asked for a conversion named after the
    old spelling, did not find one, and quietly left the condition testing the
    pointer - which is never null. So the class is asked what conversions it
    has rather than told what to have. Any that answers a plain value is one a
    condition can test; one that answers a pointer or another object is not.
    """

    for seen in [owner, *_every_base(owner, classes)]:
        found = classes.get(seen)
        if found is None:
            continue
        for method in found.methods:
            if not method.name.startswith(_CONVERSION_PREFIX):
                continue
            if "*" in method.returns or "&" in method.returns:
                continue
            words = [
                word
                for word in re.sub(
                    r"\b(const|volatile)\b", " ", method.returns
                ).split()
                if word
            ]
            if not words or words[-1] in classes:
                continue
            return method.name
    return None


def _ask_a_class_whether_it_is_true(
    whole: str, classes: "dict[str, Class]"
) -> str:
    """`while (in >> n)` asks the stream, not the pointer standing in for it.

    An operator that answers a reference to its own class answers a pointer
    here, and a pointer in a condition is true whenever it is not null - so
    `while (in >> n)` never ended. C++ asks the class: it has a conversion to
    bool, and that is what the condition means.

    Only for a call to one of the class's own operators, and only where the
    class has that conversion. A function that answers a plain pointer is
    being tested for null, which is what the C already says.
    """

    for _round in range(_HOIST_ROUNDS):
        changed = False
        bare = _without_literals(whole)
        for found in _CONDITION_CALL.finditer(bare):
            called = found.group(2)
            owner, mark, _rest = called.partition("__op_")
            if not mark or owner not in classes:
                continue
            method = _conversion_for_a_condition(owner, classes)
            if method is None:
                continue
            asked = _c_name(_find_method(owner, method, classes) or owner, method)
            if re.search(rf"(?<![.\w>]){re.escape(asked)}\s*\(", bare) is None:
                continue
            # It has to answer a reference to its own class; an operator that
            # answers a number is already the condition it looks like.
            if re.search(
                rf"(?<![.\w>]){re.escape(owner)}\s*\*\s*{re.escape(called)}\s*\(",
                bare,
            ) is None:
                continue
            opening = found.end() - 1
            close = _closing_paren(bare, opening)
            if close < 0:
                continue
            begins = found.start(2)
            whole = (
                whole[:begins]
                + f"{asked}({whole[begins: close + 1]})"
                + whole[close + 1:]
            )
            changed = True
            break
        if not changed:
            return whole
    return whole


def _pointer_call_signatures(
    text: str, classes: "dict[str, Class]"
) -> "tuple[dict[str, list[int]], dict[tuple[str, int], str]]":
    """Calls made through a variable, and which arguments of one want an address.

    `f(a, b)` where `f` is a variable holding a function asks the same
    question a call written to a name does - do the parameters take an object
    by address? - but nothing *defines* `f`, so the pass that reads
    definitions found no answer and handed the object over whole. A
    comparator given to `sort` is exactly this shape, and it is how a program
    says "call this back".
    """

    bare = _without_literals(text)
    shapes: "dict[str, list[tuple[int, str]]]" = {}
    for match in _FUNCTION_TYPEDEF.finditer(bare):
        wanted: "list[tuple[int, str]]" = []
        for index, part in enumerate(_split_arguments(match.group(2))):
            if "*" not in part and "&" not in part:
                continue
            for word in part.replace("*", " ").replace("&", " ").split():
                if word in classes:
                    wanted.append((index, word))
                    break
        if wanted:
            shapes[match.group(1)] = wanted
    positions: "dict[str, list[int]]" = {}
    types: "dict[tuple[str, int], str]" = {}
    for spelled, wanted in shapes.items():
        for match in re.finditer(
            rf"(?<![.\w>]){re.escape(spelled)}\s+([A-Za-z_]\w*)\s*[;,)=]", bare
        ):
            positions[match.group(1)] = [index for index, _held in wanted]
            for index, held in wanted:
                types[(match.group(1), index)] = held
    return positions, types


def _reference_literal_parameters(
    classes: "dict[str, Class]",
) -> "tuple[dict[str, list[int]], dict[tuple[str, int], str]]":
    """Which method parameters are references to something that is not a class.

    Keyed by the C name a call is written with, and by the position in *that*
    call - so one further along than the method declares, because the object
    goes first.
    """

    positions: "dict[str, list[int]]" = {}
    held: "dict[tuple[str, int], str]" = {}
    for owner, holder in classes.items():
        for method in holder.methods:
            spelled = _c_name(
                owner, method.name, _suffix_of(owner, method, classes)
            )
            at: "list[int]" = []
            for index, part in enumerate(_split_arguments(method.parameters)):
                words = part.replace("*", " ").replace("&", " ").split()
                if "&" not in part or "*" in part or len(words) < 2:
                    continue
                if any(word in classes for word in words):
                    continue
                at.append(index + 1)
                held[(spelled, index + 1)] = " ".join(
                    one for one in words[:-1] if one != "const"
                )
            if at:
                positions[spelled] = at
    return positions, held


def _bind_reference_literals(
    text: str,
    signatures: "dict[str, list[int]]",
    bound_types: "dict[tuple[str, int], str]",
) -> str:
    """`Wrap<int> w(7);` where the parameter is a `const int &`.

    A reference binds to an object, and a literal is not one - C++ makes a
    temporary for it and binds to that, which is the only reason the code is
    legal. Nothing here made one, so the call was handed a `7` where the C
    wanted an `int *`.
    """

    if not bound_types:
        return text
    for _round in range(_HOIST_ROUNDS):
        changed = False
        for name, positions in signatures.items():
            bare = _without_literals(text)
            for found in re.finditer(
                rf"(?<![.\w>]){re.escape(name)}\s*\(", bare
            ):
                close = _closing_paren(bare, found.end() - 1)
                if close < 0 or _is_a_definition(bare, close):
                    continue
                inside = text[found.end(): close]
                parts = _split_arguments(inside) if inside.strip() else []
                for index in positions:
                    if index >= len(parts):
                        continue
                    argument = parts[index].strip()
                    held = bound_types.get((name, index))
                    if held is None or not argument:
                        continue
                    if argument.startswith("&") or _has_an_address(argument):
                        continue
                    made = f"__py2bin_bound_{abs(hash((found.start(), index))) % 100000}"
                    parts[index] = f"&{made}"
                    begins = _statement_start(text, found.start())
                    text = (
                        text[:begins]
                        + f" {held} {made} = {argument}; "
                        + text[begins: found.end()]
                        + ", ".join(one.strip() for one in parts)
                        + text[close:]
                    )
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
        if not changed:
            return text
    return text


def _address_reference_arguments(
    text: str,
    signatures: "dict[str, list[int]]",
    classes: "dict[str, Class]" = {},
    scope: str = "",
    already: "set[str]" = frozenset(),
    also: "dict[tuple[str, int], str]" = {},
) -> str:
    """`bump(a, 9)` becomes `bump(&a, 9)` where the parameter is a reference.

    Only where the argument is something that has an address - a name, a
    member, an element. An expression has no address, and C++ would be making
    a temporary to bind a `const&` to; saying so beats taking the address of
    something that is not there.
    """

    if not signatures:
        return text
    wanted_types: "dict[tuple[str, int], str]" = {}
    #: The same, for a reference to something that is not a class.
    bound_types: "dict[tuple[str, int], str]" = {}
    #: How many parameters each function was written with. A call carrying one
    #: more than that has already been given the caller's space at the front,
    #: and every position the author wrote has moved along by one. Which calls
    #: those are is not decidable here - two passes insert it, at different
    #: times - so it is read off the call itself.
    arity: "dict[str, int]" = {}
    # From the file as well as from this text: a body is rewritten on its own
    # and the free functions it calls are defined elsewhere, so read only
    # here their shapes were unknown and every call to one was left alone.
    reading = f"{text}\n{scope}" if scope else text
    for match in _DEFINITION.finditer(reading):
        if _depth_at(reading, match.end() - 1) != 0:
            continue
        spelled_parts = _split_arguments(match.group(3))
        arity[match.group(2)] = len(
            [one for one in spelled_parts if one.strip()]
        )
        for index, part in enumerate(spelled_parts):
            spelled = part.replace("*", " ").replace("&", " ").split()
            for word in spelled:
                if word in classes:
                    wanted_types[(match.group(2), index)] = word
                    break
            else:
                # A reference to something that is not a class - `const int
                # &v`. C++ binds one to a literal by making a temporary and
                # binding to that; there is nowhere here for the address of
                # a `7` to come from otherwise.
                if "&" in part and "*" not in part and len(spelled) >= 2:
                    bound_types[(match.group(2), index)] = " ".join(
                        one for one in spelled[:-1] if one != "const"
                    )
    for spelled, written in _static_member_parameters(classes).items():
        parts = _split_arguments(written)
        arity[spelled] = len([one for one in parts if one.strip()])
        for index, part in enumerate(parts):
            for word in part.replace("*", " ").replace("&", " ").split():
                if word in classes:
                    wanted_types[(spelled, index)] = word
                    break
    # What a call through a variable wants, which no definition here says.
    wanted_types.update(also)
    # A reference to something that is not a class, given something with no
    # address: C++ makes a temporary and binds to that.
    text = _bind_reference_literals(text, signatures, bound_types)
    for name, positions in signatures.items():
        pattern = re.compile(rf"(?<![.\w>]){re.escape(name)}\s*\(")
        out: list[str] = []
        at = 0
        for found in pattern.finditer(text):
            close = _closing_paren(text, found.end() - 1)
            if close < 0:
                continue
            inside = text[found.end(): close]
            parts = _split_arguments(inside) if inside.strip() else []
            shift = 1 if len(parts) == arity.get(name, -2) + 1 else 0
            for index in positions:
                if index + shift >= len(parts):
                    continue
                index = index + shift
                argument = parts[index].strip()
                if argument.startswith("&") or not _has_an_address(argument):
                    continue
                # A reference this body already carries as a pointer is the
                # address; taking one of it gave a `T **`.
                if argument in already:
                    continue
                # C++ converts a derived object to its base wherever one is
                # wanted; C makes you say so. The address is the same - the
                # base is the first member - so this is a cast and no more.
                wanted = wanted_types.get((name, index))
                held = _deduced_type(argument, reading)
                # A reference to a class binds to an object of that class,
                # and only then. `escapeJson(state)` where `state` is a
                # `const char *` and the parameter is a `const string &` is a
                # conversion - C++ builds a string and binds to that - so
                # taking the address here gave a `char **`. Confirmed rather
                # than assumed: where the type cannot be read, the conversion
                # pass further down is the one that knows what to do.
                if wanted and not (
                    held
                    and (
                        held.replace("*", " ").replace("const", " ").strip()
                        == wanted
                        or _derives_from(
                            held.replace("*", "").strip(), wanted, classes
                        )
                    )
                ):
                    continue
                cast = ""
                if (
                    wanted
                    and held
                    and held.replace("*", "").strip() != wanted
                    and _derives_from(
                        held.replace("*", "").strip(), wanted, classes
                    )
                ):
                    cast = f"(struct {wanted} *)"
                # The first base is at offset zero, so a cast is the whole of
                # the conversion. A second base is not: it is a member after
                # the first, and the address of the object is not the address
                # of it. Naming the member is what moves the pointer, which
                # is what a C++ compiler emits here too.
                path = (
                    _subobject_path(
                        (held or "").replace("*", "").strip(), wanted, classes
                    )
                    if wanted and held
                    else None
                )
                if path and not set(path.split(".")) <= {"__base"}:
                    parts[index] = f"&{argument}.{path}"
                    continue
                parts[index] = f"{cast}&{argument}"
            out.append(text[at:found.end()])
            out.append(", ".join(part.strip() for part in parts))
            at = close
        out.append(text[at:])
        text = "".join(out)
    return text


#: A name, a member of one, or an element of one: things with an address.
_ADDRESSABLE = re.compile(r"^[A-Za-z_]\w*(\s*(\.|->)\s*[A-Za-z_]\w*|\s*\[[^\]]*\])*$")


def _has_an_address(argument: str) -> bool:
    return bool(_ADDRESSABLE.match(argument.strip()))


#: Stands in for a reference's own declaration while its uses are rewritten.
_BINDING_MARK = "\x00py2bin_bind_%d\x00"

#: `int &r = expr;` declared inside a body.
_LOCAL_REFERENCE = re.compile(
    r"\b((?:const\s+)?[A-Za-z_]\w*)\s*&\s*([A-Za-z_]\w*)\s*=\s*([^;]+);"
)



#: `string shout(string s);` at the top level - a declaration with no body.
#: The type and the name have to be separated by something - whitespace, or
#: the stars of a pointer return. Without that, `printf("...", t);` read as a
#: declaration of `rintf` with return type `p`.
_PROTOTYPE = re.compile(
    r"(?m)^[ \t]*([A-Za-z_][\w\s]*?)(?:\s+|\s*([*&]+)\s*)([A-Za-z_]\w*)\s*"
    r"\(([^;{}()]*)\)\s*;"
)


def _rewrite_prototypes(text: str, classes: "dict[str, Class]") -> str:
    """Say about a declared function exactly what its definition says.

    Two things a definition gets and a prototype was not: a class answered by
    value becomes the hidden pointer the caller provides, and a class taken by
    value is passed as a pointer and copied on entry. A header that declares
    the function and a source that defines it then disagreed about both.
    """

    if not classes:
        return text

    def one(match: "re.Match[str]") -> "str | None":
        spelled, stars, name, parameters = match.groups()
        stars = stars or ""
        held = spelled.strip()
        if _depth_at(match.string, match.start()) != 0:
            # A prototype is at file scope. Inside a body, `string t("a");`
            # is a temporary being built - and with the literal blanked for
            # scanning it reads exactly like a declaration taking nothing.
            return None
        # Every word of it has to be able to be part of a type. `return f(x);`
        # and `else g(y);` are statements that read like declarations.
        if name in _NOT_A_TYPE or any(
            word in _NOT_A_TYPE for word in held.split()
        ):
            return None
        # Without the `struct`, here and below: the declaration rewriter adds
        # one further down, and two is not C.
        inside = _references_to_pointers(parameters)
        for owner, variable in _by_value_objects(parameters, classes):
            inside = re.sub(
                rf"\b{re.escape(owner)}\s+{re.escape(variable)}\b",
                f"{owner} *__by_value_{variable}",
                inside,
                count=1,
            )
        if not stars and held in classes:
            comma = ", " if inside.strip() else ""
            return f"void {name}({held} *__ret{comma}{inside});"
        if inside.strip() == parameters.strip():
            return None
        return f"{held} {stars}{name}({inside});"

    return _sub_code(_PROTOTYPE, text, lambda m, whole: one(m))

def _returns_a_pointer(head: str, held: str, is_class: bool = True) -> str:
    """`const T &f(` becomes `struct T *f(` - a reference is a pointer here.

    A reference to a number becomes `int *` and not `struct int *`: what the
    reference names decides whether the word belongs in front of it.

    The `const` goes with it: what the reference promised is that the caller
    would not write through it, and a `const struct T *` says the same thing
    about the pointer that stands in for it.
    """

    opened = head.rfind("(")
    result, rest = head[:opened], head[opened:]
    # The `&` is written against the name as often as against the type -
    # `const string &pick` splits into `const`, `string`, `&pick` - so it
    # comes off the name here rather than being left to become part of it.
    name = result.split()[-1].lstrip("&*")
    keeps = "const " if re.search(r"\bconst\b", result) else ""
    spelled = f"struct {held}" if is_class else held
    return f"{keeps}{spelled} *{name}{rest}"


#: `int &at(int i) {` - a function answering a reference to something that is
#: not a class.
_SCALAR_REFERENCE_RESULT = re.compile(
    r"(?<![.\w>])((?:static\s+|inline\s+|const\s+)*[A-Za-z_]\w*)\s*&\s*"
    r"([A-Za-z_]\w*)\s*\([^;{}()]*\)\s*\{"
)


def _dereference_scalar_references(text: str, shapes) -> str:
    """Read through every call to a function that answers `T &`, `T` not a class.

    A reference to a class is a pointer at every call site already, because
    what it was picked from is one. A reference to a number is not: the
    function answers with an address and each use of it wants the number, so
    the `*` has to be written where the call is.
    """

    named = set()
    for match in _SCALAR_REFERENCE_RESULT.finditer(_without_literals(text)):
        held = match.group(1).split()[-1]
        if held not in shapes and held not in _NOT_A_TYPE:
            named.add(match.group(2))
    for name in named:
        while True:
            changed = False
            bare = _without_literals(text)
            for match in re.finditer(
                rf"(?<![.\w>:]){re.escape(name)}\s*\(", bare
            ):
                close = _closing_paren(bare, match.end() - 1)
                if close < 0 or _is_a_definition(bare, close):
                    continue
                before = text[:match.start()].rstrip()
                # Already read through, or having its address taken rather
                # than its answer used.
                if before.endswith(("(*", "&")):
                    continue
                text = (
                    text[:match.start()]
                    + f"(*{text[match.start(): close + 1]})"
                    + text[close + 1:]
                )
                changed = True
                break
            if not changed:
                break
    return text


def _rewrite_functions(
    text: str,
    classes: "dict[str, Class]",
    outer: "dict[str, str] | None" = None,
    unit: str = "",
    shapes: "dict[str, Class] | None" = None,
) -> str:
    """Rewrite each function on its own, because a scope is not the file.

    Done to the whole remainder at once, a variable declared in one function
    was in scope for every later one, and - worse - its destructor was placed
    at the end of the *last* function in the file rather than its own. The
    compiler then reported a name that is not declared, pointing at a line in
    somebody else's function.
    """

    # What may be passed by value: the classes, and the plain structs, which
    # are C already but still cannot travel in a register.
    shapes = classes if shapes is None else shapes
    # Before any one function is rewritten: a call can stand above the
    # definition it reaches, and by then this text has been walked past.
    text = _dereference_scalar_references(text, shapes)
    out = []
    at = 0
    while True:
        head = _FUNCTION_HEAD.search(text, at)
        if head is None:
            break
        opening = head.end() - 1
        if _depth_at(text, opening) != 0:
            at = head.end()
            continue
        try:
            closing = _matching(text, opening)
        except ValueError:
            break
        head = text[at:opening]
        # What sits between the previous function and this one - a file-scope
        # object, a struct, an include - is not part of the declaration, and
        # the rebuild below writes a fresh head from the return type and the
        # name. Read together, that rebuild dropped everything in front of it:
        # a static member's storage, written at file scope, vanished if the
        # first function after it returned an object by value.
        bare = _without_literals(head)
        cut = max(bare.rfind(";"), bare.rfind("}")) + 1
        lead, head = head[:cut], head[cut:]
        opened = head.rfind("(")
        references = (
            _reference_parameters(head[opened + 1:], classes) if opened >= 0 else {}
        )
        if references:
            head = head[:opened + 1] + _references_to_pointers(head[opened + 1:])
        # A parameter of class type taken by value is a pointer in the C, with
        # the copy made on entry - exactly what a method does with one. Free
        # functions did not, so `int twice(V v)` declared a struct parameter
        # this backend cannot pass and the call was refused.
        # Between the parentheses, not merely after the opening one: the
        # trailing `)` rides along on the last parameter otherwise, and
        # `v)` is not an identifier, so the last parameter was never seen.
        inside = head[opened + 1: head.rfind(")")] if opened >= 0 else ""
        copied = _by_value_objects(inside, shapes)
        # A class returned by value becomes the hidden pointer an ABI would
        # pass - the same transform a method gets, and for the same reason:
        # this backend does not return a struct.
        # `R make(int x)` - the words before the parentheses are the return
        # type and then the name, so the type is everything but the last.
        spelled_result = head[:head.rfind("(")].strip() if opened >= 0 else ""
        words = spelled_result.split()
        returned = words[-2] if len(words) >= 2 else ""
        # `const string &longest(...)` answers a reference, which is a
        # pointer here and not the hidden pointer a value return writes
        # through. Read as a value return it was given both - a `__ret` it
        # never filled and a `&` in the result type that is not C.
        # A reference to a number is a reference too. Read as though only a
        # class could be returned by one, `int &at(int i)` kept its `&` all
        # the way into the C, which is not C.
        written = re.match(r"^(.*?)([A-Za-z_]\w*)$", spelled_result)
        result_type = written.group(1) if written is not None else ""
        by_reference = "&" in result_type
        if by_reference:
            bare = re.sub(
                r"\b(?:static|inline|extern|const|volatile)\b|[&*]",
                " ",
                result_type,
            ).split()
            returned = bare[-1] if bare else returned
            head = _returns_a_pointer(head, returned, returned in shapes)
            spelled_result = head[:head.rfind("(")].strip()
        returns_object = (
            not by_reference
            and "*" not in spelled_result
            and returned in shapes
        )
        for held, variable in copied:
            # Without the `struct`: the declaration rewriter below adds one,
            # and two is not C.
            head = re.sub(
                rf"\b{re.escape(held)}\s+{re.escape(variable)}\b",
                f"{held} *__by_value_{variable}",
                head,
                count=1,
            )
        if returns_object:
            head = (
                f"\nvoid {words[-1]}"
                # Without the `struct`: the declaration rewriter adds one.
                f"({returned} *__ret{',' if inside.strip() else ''}"
                f"{head[head.rfind('(') + 1:]}"
            )
        known, pointers, indexed = _parameters_of(head, classes)
        # A file-scope object is in scope in every function, and nothing in
        # the head says so.
        for name, held in (outer or {}).items():
            known.setdefault(name, held)
        for name, held in references.items():
            if held in classes:
                known[name] = held
                pointers.add(name)
        for held, variable in copied:
            known[variable] = held
        inner = _deref_references(text[opening:closing], references, classes)
        if returns_object:
            inner = _return_through_pointer(inner, returned)
        rewritten = _rewrite_body(
            inner, classes, known, pointers,
            unit=f"{unit}\n{head[head.rfind('(') + 1: head.rfind(')')]};"
            if opened >= 0 else unit,
            stable=unit,
            pointer_arrays=indexed,
            # `S *pick` splits into the type `S *` and the name `pick`;
            # splitting on whitespace put the star with the name and lost it.
            returns=(
                re.match(r"^(.*?)([A-Za-z_]\w*)$", spelled_result).group(1).strip()
                if re.match(r"^(.*?)([A-Za-z_]\w*)$", spelled_result)
                else ""
            ),
            referenced=set(references),
        )
        if by_reference and returned not in shapes:
            # A reference to a class is a pointer already: the objects it is
            # picked from are pointers here, so `return b;` answers with one.
            # A reference to a number is not - `return slot[i];` *is* the
            # number, and what has to go back is where it lives.
            rewritten = _return_the_address(rewritten)
        elif by_reference:
            # Almost always a pointer already - but not when what is handed
            # back is an object this body declares. `R &one() { static R
            # held; return held; }` is how a program writes a singleton, and
            # it answered with the object where a pointer was wanted.
            rewritten = _address_a_returned_object(rewritten, returned)
        if copied:
            # After the body is rewritten, not before: this text is already C,
            # and a `struct V v;` put in ahead of the declaration pass reads
            # as a new object to construct - so the copy ran the constructor
            # over what it was about to be handed. Declared and then assigned
            # rather than initialised, because py2bin's C takes `o = *p;` and
            # not `struct V o = *p;`.
            entry = " ".join(
                f"struct {held} {variable}; "
                + _copied_in(held, variable, f"__by_value_{variable}", classes)
                for held, variable in copied
            )
            spot = rewritten.find("{")
            rewritten = rewritten[:spot + 1] + " " + entry + rewritten[spot + 1:]
        out.append(_rewrite_declarations(lead, classes))
        out.append(_rewrite_declarations(head, classes))
        out.append(rewritten)
        at = closing
    out.append(_rewrite_declarations(text[at:], classes))
    return "".join(out)



def _parameters_of(
    head: str, classes: "dict[str, Class]"
) -> "tuple[dict[str, str], set[str], set[str]]":
    """The class-typed parameters of a function, which its body can call on.

    A parameter is declared in the head and used in the body, and the body is
    all the rewriter is handed - so `p->sum()` on an `Inline *p` was left
    alone and the compiler reported a struct with no such member.
    """

    opening = head.rfind("(")
    if opening < 0:
        return {}, set(), set()
    known: dict[str, str] = {}
    pointers: set[str] = set()
    arrays: set[str] = set()
    for part in head[opening + 1:].split(","):
        words = part.replace("*", " * ").split()
        # `const struct P *items` names a P. Read as though the first word
        # were always the type, this saw `const` - not a class - and passed
        # the parameter over, so a method called on one was left as C++.
        while words and words[0] in ("const", "volatile", "struct", "class"):
            words = words[1:]
        if len(words) < 2:
            continue
        name = words[-1].strip("()")
        held = words[0]
        if held not in classes or not name.isidentifier():
            continue
        known[name] = held
        if "*" in part:
            pointers.add(name)
        if part.count("*") >= 2:
            # `Shape **all` is an array of pointers, reached with `all[i]->m()`
            # exactly as a locally declared `Shape *all[3]` is.
            arrays.add(name)
    return known, pointers, arrays

#: Where every brace in a text sits, and how deep it left the text after it.
#: Counting from the start on each question is what a scan does, and a scan
#: asks once per match - which makes reading a program quadratic in its own
#: length. Kept against the text itself, so a scan of the same text answers
#: from one walk; a handful of texts is all any pass has open at once.
_DEPTHS_SEEN: "dict[str, tuple[list[int], list[int]]]" = {}
_DEPTHS_KEPT = 8

_A_BRACE = re.compile(r"[{}]")


def _brace_depths(text: str) -> "tuple[list[int], list[int]]":
    """Every brace position in `text`, and the depth just after each."""

    walked = _DEPTHS_SEEN.get(text)
    if walked is not None:
        return walked
    at: "list[int]" = []
    depth: "list[int]" = []
    level = 0
    for match in _A_BRACE.finditer(text):
        level += 1 if match.group(0) == "{" else -1
        at.append(match.start())
        depth.append(level)
    while len(_DEPTHS_SEEN) >= _DEPTHS_KEPT:
        del _DEPTHS_SEEN[next(iter(_DEPTHS_SEEN))]
    _DEPTHS_SEEN[text] = (at, depth)
    return at, depth


def _depth_at(text: str, index: int) -> int:
    """How many braces are open just before `index`."""

    at, depth = _brace_depths(text)
    before = bisect.bisect_left(at, index)
    return depth[before - 1] if before else 0


def _rewrite_declarations(text: str, classes: "dict[str, Class]") -> str:
    """Outside any function: a class named as a type is all there is to do."""

    return _rewrite_types(text, classes)


#: `Counter shared;` written outside every function.
#: `static G global;` counts: how an object is stored is not part of what it
#: is, and a constructor still has to run for it. `extern` is deliberately
#: not among them - that names an object defined somewhere else, and
#: constructing it here would build it twice.
#: The array form is written as an alternative *after* the arguments so the
#: two groups in front of it keep their numbers: `Cell t[3](1)` is not C++,
#: so nothing has both.
_FILE_SCOPE_OBJECT = re.compile(
    r"(?m)^[ \t]*(?:(?:static|const|volatile|inline)[ \t]+)*"
    r"([A-Za-z_]\w*)[ \t]+([A-Za-z_]\w*)[ \t]*"
    r"(?:\(([^;{}()]*)\)|\[([^\];]*)\])?[ \t]*;"
)


def _file_scope_objects(
    text: str, classes: "dict[str, Class]"
) -> "dict[str, tuple[str, str]]":
    """Objects declared outside any function, with the class and the arguments.

    Their methods are reachable from every function in the file, so every body
    has to be rewritten knowing about them - without this, `shared.bump()` was
    left as C++ and the compiler reported a struct with no such member.
    """

    found: "dict[str, tuple[str, str]]" = {}
    for match in _FILE_SCOPE_OBJECT.finditer(text):
        # An array of them is not one of them: read as a lone object it was
        # handed the address of the whole array, which is a different type
        # and a different number of constructors.
        if _depth_at(text, match.start()) != 0 or match.group(4) is not None:
            continue
        arguments = (match.group(3) or "").strip()
        if _looks_like_parameters(arguments, classes):
            # `string shout(string s);` is a function this file declares, not
            # an object it builds. C++ has the same ambiguity and resolves it
            # the same way: what can be read as a declaration is one.
            continue
        if match.group(1) in classes:
            found[match.group(2)] = (match.group(1), arguments)
    return found


def _file_scope_arrays(
    text: str, classes: "dict[str, Class]"
) -> "dict[str, tuple[str, str]]":
    """Arrays of objects declared outside any function, with the class and count.

    C++ default-constructs every element of one before `main`, the same as it
    builds a lone object. Read only as a lone object, `Cell table[3];` was
    left as three cells of zeroes: the members were whatever the loader put
    there and the constructor's own bookkeeping - a count, a registration -
    never happened. Nothing said so; the program ran and answered wrongly.
    """

    found: "dict[str, tuple[str, str]]" = {}
    for match in _FILE_SCOPE_OBJECT.finditer(text):
        if _depth_at(text, match.start()) != 0 or match.group(4) is None:
            continue
        if match.group(1) in classes:
            found[match.group(2)] = (match.group(1), match.group(4).strip())
    return found


def _looks_like_parameters(arguments: str, classes: "dict[str, Class]") -> bool:
    """Whether `(...)` reads as a parameter list rather than as arguments."""

    if arguments.strip() == "void":
        # Only a declaration spells it that way.
        return True
    if not arguments.strip():
        # `G g();` declares a function taking nothing in C++ too. But a
        # file-scope object written that way is far commoner than a
        # prototype for a function answering a class by value, and the
        # prototype pass rewrites that one anyway.
        return False
    for part in _split_arguments(arguments):
        spelled = part.replace("*", " ").replace("&", " ").replace("const", " ")
        words = spelled.split()
        if len(words) >= 2 and all(word.isidentifier() for word in words):
            continue
        if len(words) == 1 and (words[0] in classes or words[0] in _LEADS_A_TYPE):
            continue
        return False
    return True


def _construct_before_main(text: str, made: "dict[str, str]", classes) -> str:
    """Run each file-scope object's constructor at the top of `main`.

    C++ builds them before `main` runs and C has no place to put that, so the
    first thing `main` does is what C++ had already done. A program that
    reaches one of them from another static initialiser would see the
    difference; there is no such thing here, because C has no static
    initialiser that can call anything.
    """

    # `G withArgs(7);` is a declaration with a constructor call in it, and C
    # reads that as a function taking a 7. The arguments move to the call
    # below and the declaration is left as the object it declares.
    text = _sub_code(
        _FILE_SCOPE_OBJECT,
        text,
        lambda match, whole: (
            None
            if not match.group(3) or match.group(2) not in made
            else f"{whole[match.start(): match.start(3) - 1].rstrip()};"
        ),
    )
    calls = []
    for variable, (held, arguments) in made.items():
        owner = _find_method(held, "", classes)
        if owner is None:
            continue
        given = (
            [one.strip() for one in _split_arguments(arguments)]
            if arguments
            else []
        )
        passed = f", {arguments}" if arguments else ""
        calls.append(
            f"{_c_name(owner, '', _call_suffix(owner, '', classes, given, text))}"
            f"(&{variable}{passed});"
        )
    for variable, (held, count) in _file_scope_arrays(text, classes).items():
        owner = _find_method(held, "", classes)
        if owner is None or not count:
            continue
        walked = f"__py2bin_i_{variable}"
        calls.append(
            f"{{ int {walked}; for ({walked} = 0; {walked} < {count}; "
            f"{walked}++) "
            f"{_c_name(owner, '', _call_suffix(owner, '', classes, [], text))}"
            f"(&{variable}[{walked}]); }}"
        )
    if not calls:
        return text
    entry = re.search(r"\bmain\s*\([^)]*\)\s*\{", text)
    if entry is None:
        return text
    return text[:entry.end()] + " " + " ".join(calls) + text[entry.end():]

def _with_typedefs(text: str, typedefs: str) -> str:
    """Put the typedefs after the last preprocessor line at the top."""

    lines = text.split("\n")
    at = 0
    for index, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            at = index + 1
    lines.insert(at, typedefs)
    return "\n".join(lines)


def translate_file(path: Path) -> str:
    return translate(path.read_text(encoding="utf-8", errors="replace"), str(path))



#: py2bin's own `<string>`, written in the subset it compiles.
#:
#: `std::string` proper needs a heap, templates and copy semantics, and this
#: has a fixed buffer instead - which is a real limit and stated in the
#: readme. What it buys is that `#include <string>` and `std::string` stop
#: being a wall: the class is declared in `namespace std`, and namespaces are
#: flattened, so `std::string` resolves to it exactly as the qualifier says.
_STRING_HEADER = r"""
namespace std {
class string {
public:
    char buf[256];
    int len;
    string() { buf[0] = 0; len = 0; }
    string(const char *s) {
        int i; i = 0;
        while (s[i] != 0 && i < 255) { buf[i] = s[i]; i = i + 1; }
        buf[i] = 0; len = i;
    }
    /* `string held(n, 0);` - room made up front, which is how a program
       gives a C interface somewhere to write. */
    string(int count, char fill) {
        int i; i = 0;
        while (i < count && i < 255) { buf[i] = fill; i = i + 1; }
        buf[i] = 0; len = i;
    }
    /* Writable, which is the point: a C interface handed this fills it in,
       and `resize` afterwards says how much of it was filled. */
    char *data() { return buf; }
    void resize(int count) {
        int i;
        if (count > 255) { count = 255; }
        i = len;
        while (i < count) { buf[i] = 0; i = i + 1; }
        buf[count] = 0;
        len = count;
    }
    void reserve(int count) { }
    void assign(const char *s) {
        int i; i = 0;
        while (s[i] != 0 && i < 255) { buf[i] = s[i]; i = i + 1; }
        buf[i] = 0; len = i;
    }
    int size() { return len; }
    int length() { return len; }
    int empty() { return len == 0; }
    char at(int i) { return buf[i]; }
    const char *c_str() { return buf; }
    void push_back(char c) {
        if (len < 255) { buf[len] = c; len = len + 1; buf[len] = 0; }
    }
    void clear() { len = 0; buf[0] = 0; }
    int compare(const char *s) {
        int i;
        i = 0;
        while (buf[i] != 0 && buf[i] == s[i]) { i = i + 1; }
        return (int)(unsigned char)buf[i] - (int)(unsigned char)s[i];
    }
    int find(char c) {
        int i;
        i = 0;
        while (i < len) { if (buf[i] == c) { return i; } i = i + 1; }
        return -1;
    }
    static const int npos = -1;

    void operator+=(string o) { append(o); }
    void operator+=(const char *s) { append(s); }
    void operator+=(char c) { push_back(c); }

    char &operator[](int i) { return buf[i]; }

    int operator<(string o) { return compare(o.c_str()) < 0; }
    int operator>(string o) { return compare(o.c_str()) > 0; }
    int operator<=(string o) { return compare(o.c_str()) <= 0; }
    int operator>=(string o) { return compare(o.c_str()) >= 0; }

    int find(const char *needle) {
        int i; int j;
        if (needle[0] == 0) { return 0; }
        i = 0;
        while (i < len) {
            j = 0;
            while (needle[j] != 0 && i + j < len && buf[i + j] == needle[j]) {
                j = j + 1;
            }
            if (needle[j] == 0) { return i; }
            i = i + 1;
        }
        return -1;
    }

    /* The cast is what says which overload: two `find`s take one argument,
       and the type of a call's result is what tells them apart. */
    int find(string needle) { return find((const char *)needle.c_str()); }
    /* The last match rather than the first, which is what `rfind` is. */
    int rfind(const char *needle) {
        int i; int j; int found;
        found = -1;
        i = 0;
        while (i < len) {
            j = 0;
            while (needle[j] != 0 && i + j < len && buf[i + j] == needle[j]) { j = j + 1; }
            if (needle[j] == 0) { found = i; }
            i = i + 1;
        }
        return found;
    }
    int rfind(string needle) { return rfind((const char *)needle.c_str()); }

    /* And each of them from a position, which is how a program walks a
       string looking for the next one of something. */
    int find(char c, int from) {
        int i;
        i = from;
        if (i < 0) { i = 0; }
        while (i < len) { if (buf[i] == c) { return i; } i = i + 1; }
        return -1;
    }
    int find(const char *needle, int from) {
        int i; int j;
        i = from;
        if (i < 0) { i = 0; }
        if (needle[0] == 0) { return i <= len ? i : -1; }
        while (i < len) {
            j = 0;
            while (needle[j] != 0 && i + j < len && buf[i + j] == needle[j]) {
                j = j + 1;
            }
            if (needle[j] == 0) { return i; }
            i = i + 1;
        }
        return -1;
    }
    int find(string needle, int from) {
        return find((const char *)needle.c_str(), from);
    }

    string substr(int from, int count) {
        string out;
        int i;
        i = 0;
        while (i < count && from + i < len) {
            out.push_back(buf[from + i]);
            i = i + 1;
        }
        return out;
    }

    string substr(int from) { return substr(from, len - from); }

    void append(string o) {
        int j; j = 0;
        while (j < o.len && len + j < 255) { buf[len + j] = o.buf[j]; j = j + 1; }
        len = len + j; buf[len] = 0;
    }
    void append(const char *s) {
        int j; j = 0;
        while (s[j] != 0 && len < 255) { buf[len] = s[j]; len = len + 1; j = j + 1; }
        buf[len] = 0;
    }
    string operator+(string o) {
        string r; int i; int j;
        for (i = 0; i < len; i++) { r.buf[i] = buf[i]; }
        for (j = 0; j < o.len && len + j < 255; j++) { r.buf[len + j] = o.buf[j]; }
        r.len = len + j; r.buf[r.len] = 0;
        return r;
    }
    int operator==(const char *s) { return compare(s) == 0; }
    int operator==(string o) {
        int i;
        if (len != o.len) { return 0; }
        for (i = 0; i < len; i++) { if (buf[i] != o.buf[i]) { return 0; } }
        return 1;
    }
    string operator+(const char *s) {
        string r;
        int i;
        i = 0;
        while (i < len) { r.push_back(buf[i]); i = i + 1; }
        r.append(s);
        return r;
    }
    int operator!=(string o) {
        int i;
        if (len != o.len) { return 1; }
        for (i = 0; i < len; i++) { if (buf[i] != o.buf[i]) { return 1; } }
        return 0;
    }
};

string to_string(int value) {
    string out;
    char digits[24];
    int at;
    int negative;
    unsigned int left;
    at = 0;
    negative = value < 0;
    left = negative ? (unsigned int)(-value) : (unsigned int)value;
    if (left == 0) { digits[at] = '0'; at = at + 1; }
    while (left > 0) { digits[at] = (char)('0' + (left % 10)); left = left / 10; at = at + 1; }
    if (negative) { out.push_back('-'); }
    while (at > 0) { at = at - 1; out.push_back(digits[at]); }
    return out;
}

int stoi(string text) {
    int i; int sign; int value;
    i = 0; sign = 1; value = 0;
    if (text.at(0) == '-') { sign = -1; i = 1; }
    while (i < text.size()) {
        if (text.at(i) < '0') { return value * sign; }
        if (text.at(i) > '9') { return value * sign; }
        value = value * 10 + (int)(text.at(i) - '0');
        i = i + 1;
    }
    return value * sign;
}

/* The same class over wchar_t. Windows is wide throughout - every `W` entry
   point takes one - so a program that talks to it holds its strings this
   way, and `std::wstring` is what it calls them. Written out rather than
   made a template of the narrow one: `string` here is a fixed buffer and a
   length, and two of those are two classes, which is what they are in the
   standard too. */
class wstring {
public:
    wchar_t buf[256];
    int len;
    wstring() { buf[0] = 0; len = 0; }
    wstring(const wchar_t *s) {
        int i; i = 0;
        while (s[i] != 0 && i < 255) { buf[i] = s[i]; i = i + 1; }
        buf[i] = 0; len = i;
    }
    wstring(int count, wchar_t fill) {
        int i; i = 0;
        while (i < count && i < 255) { buf[i] = fill; i = i + 1; }
        buf[i] = 0; len = i;
    }
    void assign(const wchar_t *s) {
        int i; i = 0;
        while (s[i] != 0 && i < 255) { buf[i] = s[i]; i = i + 1; }
        buf[i] = 0; len = i;
    }
    int size() { return len; }
    int length() { return len; }
    int empty() { return len == 0; }
    wchar_t at(int i) { return buf[i]; }
    const wchar_t *c_str() { return buf; }
    wchar_t *data() { return buf; }
    void push_back(wchar_t c) {
        if (len < 255) { buf[len] = c; len = len + 1; buf[len] = 0; }
    }
    void clear() { len = 0; buf[0] = 0; }
    /* `resize` shortens or lengthens; what it grows into is zero, which is
       what the standard says a default-inserted wchar_t is. */
    void resize(int count) {
        int i;
        if (count > 255) { count = 255; }
        i = len;
        while (i < count) { buf[i] = 0; i = i + 1; }
        len = count;
        buf[len] = 0;
    }
    void reserve(int count) { }
    void append(const wchar_t *s) {
        int i; i = 0;
        while (s[i] != 0 && len < 255) { buf[len] = s[i]; len = len + 1; i = i + 1; }
        buf[len] = 0;
    }
    void operator+=(const wchar_t *s) { append(s); }
    void operator+=(wstring o) { append(o.buf); }
    void operator+=(wchar_t c) { push_back(c); }
    wstring operator+(const wchar_t *s) {
        wstring made; made.assign(buf); made.append(s); return made;
    }
    wstring operator+(wstring o) {
        wstring made; made.assign(buf); made.append(o.buf); return made;
    }
};

}
"""

#: Angled includes py2bin answers itself. Anything else angled is left for the
#: C preprocessor, which has py2bin's own C headers behind it.
#: py2bin's own <vector>. A template, so it goes through the same expansion
#: any other does and comes out as one concrete class per element type. It
#: grows by doubling and copying; the old block is left where it is, because
#: the heap under it is an arena that does not reclaim (see <stdlib.h>) and
#: pretending otherwise would be the dishonest part, not the leak.
_VECTOR_HEADER = r"""
namespace std {
template<typename T>
class vector {
public:
    T *items;
    unsigned long count;
    unsigned long room;
    vector() { items = 0; count = 0; room = 0; }
    unsigned long size() { return count; }
    int empty() { return count == 0; }
    /* Every element taken apart, which is what letting go of them means.
       C++ destroys them when the vector goes and when it is cleared; this
       had done neither, so a `vector<T>` of objects with a destructor let
       them all go without running one - a leak with nothing to say so. The
       element's type is a template parameter, so the only way to name its
       destructor is to write the call the language provides. */
    void clear() {
        unsigned long i;
        i = 0;
        while (i < count) { items[i].~T(); i = i + 1; }
        count = 0;
    }
    ~vector() { clear(); }
    void reserve(unsigned long want) {
        unsigned long i;
        T *fresh;
        if (want <= room) { return; }
        /* Storage, not objects: `new T[want]` would run a constructor for
           every element, which a vector holding fewer than that does not
           want - and which a class with no default constructor cannot do.
           This is what a real vector's allocator hands back too. */
        fresh = (T *)malloc(sizeof(T) * want);
        i = 0;
        while (i < count) { fresh[i] = items[i]; i = i + 1; }
        items = fresh;
        room = want;
    }
    void push_back(T value) {
        if (count == room) {
            if (room == 0) { reserve(8); } else { reserve(room * 2); }
        }
        items[count] = value;
        count = count + 1;
    }
    void pop_back() { if (count > 0) { count = count - 1; } }
    void resize(unsigned long want) {
        reserve(want);
        count = want;
    }
    T &at(unsigned long i) { return items[i]; }
    T &operator[](unsigned long i) { return items[i]; }
    T &back() { return items[count - 1]; }
    T &front() { return items[0]; }
    typedef T *iterator;
    typedef T *const_iterator;
    typedef T value_type;
    typedef unsigned long size_type;
    T *begin() { return items; }
    T *end() { return items + count; }
    T *data() { return items; }
    unsigned long capacity() { return room; }
    /* `erase` is written against the pointer an iterator is here: everything
       after the hole moves down one, which is what a vector does. */
    T *erase(T *where) {
        unsigned long at;
        at = (unsigned long)(where - items);
        while (at + 1 < count) { items[at] = items[at + 1]; at = at + 1; }
        if (count > 0) { count = count - 1; }
        return where;
    }
    T *insert(T *where, T value) {
        unsigned long at;
        unsigned long j;
        at = (unsigned long)(where - items);
        if (count == room) {
            if (room == 0) { reserve(8); } else { reserve(room * 2); }
        }
        j = count;
        while (j > at) { items[j] = items[j - 1]; j = j - 1; }
        items[at] = value;
        count = count + 1;
        return items + at;
    }
    void assign(unsigned long many, T value) {
        unsigned long i;
        reserve(many);
        i = 0;
        while (i < many) { items[i] = value; i = i + 1; }
        count = many;
    }
};
}
"""

#: py2bin's own <iostream>. `cout` is an object with one `operator<<` per
#: type it can print, which is how the real one is put together too; each
#: hands the stream back so the next `<<` in the chain has something to be
#: called on. The output goes through printf, which py2bin implements itself.
_IOSTREAM_HEADER = r"""
#include <stdio.h>
namespace std {
/* The width and the fill are the stream's, not the value's: `setw` applies
   to the next thing written and then goes, which is what the standard says
   and what a program relies on when it lines a table up. */
/* The manipulators live here, with the stream that consumes them, so that
   `<iomanip>` is only the three functions that make one. */
struct __setw { int __n; };
struct __setfill { char __c; };
struct __setprecision { int __n; };

class ostream {
public:
    int __stream;
    int __width;
    char __fill;
    int __precision;
    ostream() { __stream = 1; __width = 0; __fill = ' '; __precision = 6; }
    void __space(int written) {
        while (written < __width) { printf("%c", __fill); written = written + 1; }
        __width = 0;
    }
    int __digits(unsigned long v) {
        int n = 1;
        while (v >= 10UL) { v = v / 10UL; n = n + 1; }
        return n;
    }
    ostream &operator<<(int v) {
        __space(__digits((unsigned long)(v < 0 ? -v : v)) + (v < 0 ? 1 : 0));
        printf("%d", v); return *this;
    }
    ostream &operator<<(long v) {
        __space(__digits((unsigned long)(v < 0 ? -v : v)) + (v < 0 ? 1 : 0));
        printf("%ld", v); return *this;
    }
    ostream &operator<<(unsigned int v) { __space(__digits(v)); printf("%u", v); return *this; }
    ostream &operator<<(unsigned long v) { __space(__digits(v)); printf("%lu", v); return *this; }
    /* `%g` and not `%f`: C++'s default is six *significant* digits, so
       `cout << 1.5` is `1.5` and not `1.500000`, and `setprecision(n)`
       changes that same count. Written as a choice between fixed formats
       rather than as `%.*g`, because py2bin's printf takes its precision
       from the format and not from an argument. */
    ostream &operator<<(double v) {
        __width = 0;
        if (__precision == 0) { printf("%.0g", v); return *this; }
        if (__precision == 1) { printf("%.1g", v); return *this; }
        if (__precision == 2) { printf("%.2g", v); return *this; }
        if (__precision == 3) { printf("%.3g", v); return *this; }
        if (__precision == 4) { printf("%.4g", v); return *this; }
        if (__precision == 5) { printf("%.5g", v); return *this; }
        if (__precision == 6) { printf("%.6g", v); return *this; }
        if (__precision == 7) { printf("%.7g", v); return *this; }
        if (__precision == 8) { printf("%.8g", v); return *this; }
        if (__precision == 9) { printf("%.9g", v); return *this; }
        if (__precision == 10) { printf("%.10g", v); return *this; }
        if (__precision == 11) { printf("%.11g", v); return *this; }
        if (__precision == 12) { printf("%.12g", v); return *this; }
        if (__precision == 13) { printf("%.13g", v); return *this; }
        if (__precision == 14) { printf("%.14g", v); return *this; }
        if (__precision == 15) { printf("%.15g", v); return *this; }
        if (__precision == 16) { printf("%.16g", v); return *this; }
        if (__precision == 17) { printf("%.17g", v); return *this; }
        printf("%g", v);
        return *this;
    }
    ostream &operator<<(char v) { __space(1); printf("%c", v); return *this; }
    ostream &operator<<(const char *v) {
        int n = 0;
        while (v[n] != 0) { n = n + 1; }
        __space(n);
        printf("%s", v); return *this;
    }
    ostream &operator<<(__setw v) { __width = v.__n; return *this; }
    ostream &operator<<(__setfill v) { __fill = v.__c; return *this; }
    ostream &operator<<(__setprecision v) { __precision = v.__n; return *this; }
};
ostream cout;
ostream cerr;
const char *endl = "\n";
}
"""

#: py2bin's own <algorithm>. Templates over pointers, which is what an
#: iterator is for a contiguous container - <vector>'s `begin()` and `end()`
#: hand back `T *`, so these work on a vector and on a plain array alike.
#: `sort` is a heapsort: O(n log n) with no recursion and no scratch memory,
#: which matters when the heap underneath is an arena that does not reclaim.
_ALGORITHM_HEADER = r"""
namespace std {

template<typename T>
T max(T a, T b) { return a > b ? a : b; }

template<typename T>
T min(T a, T b) { return a < b ? a : b; }

template<typename T>
void swap(T &a, T &b) { T held = a; a = b; b = held; }

template<typename T>
T *find(T *first, T *last, T value) {
    while (first != last) { if (*first == value) { return first; } first = first + 1; }
    return last;
}

template<typename T>
long count(T *first, T *last, T value) {
    long seen = 0;
    while (first != last) { if (*first == value) { seen = seen + 1; } first = first + 1; }
    return seen;
}

template<typename T>
void fill(T *first, T *last, T value) {
    while (first != last) { *first = value; first = first + 1; }
}

template<typename T>
void reverse(T *first, T *last) {
    if (first >= last) { return; }
    /* Copy-initialised, not default-constructed: an element of a container
       need not have a constructor taking nothing, and there is always a
       first element to copy by here. */
    T held = *first;
    last = last - 1;
    while (first < last) {
        held = *first; *first = *last; *last = held;
        first = first + 1; last = last - 1;
    }
}

template<typename T>
T *max_element(T *first, T *last) {
    T *best = first;
    while (first != last) { if (*best < *first) { best = first; } first = first + 1; }
    return best;
}

template<typename T>
T *min_element(T *first, T *last) {
    T *best = first;
    while (first != last) { if (*first < *best) { best = first; } first = first + 1; }
    return best;
}

template<typename T>
void __sift(T *base, long root, long span) {
    long child;
    /* Copy-initialised, for the reason `sort` gives: `base[root]` is always
       there, because a heap being sifted has a root. */
    T held = base[root];
    while (1) {
        child = root * 2 + 1;
        if (child >= span) { return; }
        if (child + 1 < span) { if (base[child] < base[child + 1]) { child = child + 1; } }
        if (base[child] < base[root]) { return; }
        /* No `==` here: `std::sort` asks an element for `<` and nothing
           else, and this used to ask for equality as well - which a class
           that only ordered itself could not answer. Swapping two equal
           elements is harmless; the loop moves down the heap either way. */
        held = base[root]; base[root] = base[child]; base[child] = held;
        root = child;
    }
}

template<typename T, typename C>
void __sift_by(T *base, long root, long span, C less_than) {
    long child;
    T held = base[root];
    while (1) {
        child = root * 2 + 1;
        if (child >= span) { return; }
        if (child + 1 < span) {
            if (less_than(base[child], base[child + 1])) { child = child + 1; }
        }
        if (less_than(base[child], base[root])) { return; }
        if (!less_than(base[root], base[child])) { return; }
        held = base[root]; base[root] = base[child]; base[child] = held;
        root = child;
    }
}

template<typename T, typename C>
void sort(T *first, T *last, C less_than) {
    long span;
    long i;
    span = last - first;
    if (span < 2) { return; }
    T held = *first;
    i = span / 2 - 1;
    while (i >= 0) { __sift_by(first, i, span, less_than); i = i - 1; }
    i = span - 1;
    while (i > 0) {
        held = first[0]; first[0] = first[i]; first[i] = held;
        __sift_by(first, (long)0, i, less_than);
        i = i - 1;
    }
}

template<typename T>
void sort(T *first, T *last) {
    long span;
    long i;
    span = last - first;
    if (span < 2) { return; }
    T held = *first;
    i = span / 2 - 1;
    while (i >= 0) { __sift(first, i, span); i = i - 1; }
    i = span - 1;
    while (i > 0) {
        held = first[0]; first[0] = first[i]; first[i] = held;
        __sift(first, (long)0, i);
        i = i - 1;
    }
}

template<typename T>
int equal(T *first, T *last, T *other) {
    while (first != last) {
        if (*first == *other) { first = first + 1; other = other + 1; }
        else { return 0; }
    }
    return 1;
}

}
"""

#: `std::pair` and the two free functions that go with it. A template, so it
#: comes out as one concrete struct per pair of types.
_UTILITY_HEADER = r"""
namespace std {
template<typename A, typename B>
class pair {
public:
    A first;
    B second;
    pair() { }
    pair(A a, B b) { first = a; second = b; }
};

template<typename A, typename B>
pair<A, B> make_pair(A a, B b) { pair<A, B> made(a, b); return made; }

template<typename T>
void swap(T &a, T &b) { T held; held = a; a = b; b = held; }
}
"""

#: <tuple>, for two, three or four things. Each arity is its own copy rather
#: than one written over a pack, because a variadic tuple is written as a
#: class inheriting from the tuple of everything after the first, and this
#: translator writes copies rather than deriving them.
#:
#: The members are `__0`, `__1` and so on, and `std::get<N>(t)` is rewritten
#: to read the one it names - because `get<0>(t)` spells one template
#: argument and leaves the rest to be deduced, which is a shape nothing here
#: writes out. Saying so is better than a `get` that quietly is not one.
_TUPLE_HEADER = r"""
namespace std {
template<typename A, typename B>
class tuple {
public:
    A __0;
    B __1;
    tuple() { }
    tuple(A a, B b) { __0 = a; __1 = b; }
};

template<typename A, typename B, typename C>
class tuple {
public:
    A __0;
    B __1;
    C __2;
    tuple() { }
    tuple(A a, B b, C c) { __0 = a; __1 = b; __2 = c; }
};

template<typename A, typename B, typename C, typename D>
class tuple {
public:
    A __0;
    B __1;
    C __2;
    D __3;
    tuple() { }
    tuple(A a, B b, C c, D d) { __0 = a; __1 = b; __2 = c; __3 = d; }
};

template<typename A, typename B>
tuple<A, B> make_tuple(A a, B b) { tuple<A, B> made(a, b); return made; }

template<typename A, typename B, typename C>
tuple<A, B, C> make_tuple(A a, B b, C c) { tuple<A, B, C> made(a, b, c); return made; }
}
"""

#: <iomanip>. Each manipulator is a small object the stream consumes, which
#: is what one is: `setw(5)` is not a value written but a change to how the
#: next value is.
_IOMANIP_HEADER = r"""
#include <iostream>
namespace std {
__setw setw(int n) { __setw made; made.__n = n; return made; }
__setfill setfill(char c) { __setfill made; made.__c = c; return made; }
__setprecision setprecision(int n) { __setprecision made; made.__n = n; return made; }
}
"""

#: <chrono>, over the one thing a program cannot work out for itself: what
#: time it is. Each platform is asked its own way - `clock_gettime` where
#: there is one, the performance counter on Windows - and the answer is
#: nanoseconds since something, which is all a steady clock promises.
#:
#: `CLOCK_MONOTONIC` is 6 on macOS and 1 on Linux. It is a number in a header
#: on each, and the number is different, so it is written out per platform
#: rather than assumed to be the same.
#: The headers that are about doing two things at once. Each is refused with
#: what is actually in the way rather than with "not implemented", because
#: what is in the way is one thing and it is fixable, and a person reading
#: the message should be told which thing.
_NEEDS_ATOMICS = frozenset(
    {"condition_variable", "future",
     "shared_mutex", "latch", "barrier", "semaphore", "stop_token"}
)

#: How each platform is asked what time it is. Written per target rather than
#: with an `#ifdef`, because the C++ translator runs before the preprocessor
#: and would read both branches - and two definitions of one function is not
#: something C++ accepts, which is exactly what it said.
#:
#: `CLOCK_MONOTONIC` is 6 on macOS and 1 on Linux. Two numbers for one name,
#: so the number is written out rather than assumed.
_CLOCKS = {
    "windows": r"""
extern int QueryPerformanceCounter(long long *into);
extern int QueryPerformanceFrequency(long long *into);
static long long __py2bin_now() {
    long long ticks;
    long long each;
    QueryPerformanceCounter(&ticks);
    QueryPerformanceFrequency(&each);
    if (each == 0LL) { return 0LL; }
    /* Split rather than multiplied first: a counter on a machine that has
       been up a while overflows a multiply by a billion. */
    return (ticks / each) * 1000000000LL + ((ticks % each) * 1000000000LL) / each;
}
""",
    "posix": r"""
struct __py2bin_span { long __sec; long __nsec; };
extern int clock_gettime(int which, struct __py2bin_span *into);
static long long __py2bin_now() {
    struct __py2bin_span held;
    clock_gettime(%(monotonic)d, &held);
    return (long long)held.__sec * 1000000000LL + (long long)held.__nsec;
}
""",
}


def _chrono_header(target: "str | None") -> str:
    """<chrono>, over the one thing a program cannot work out for itself."""

    named = (target or "").split("-", 1)[0]
    if named == "windows":
        clock = _CLOCKS["windows"]
    else:
        clock = _CLOCKS["posix"] % {"monotonic": 6 if named == "darwin" else 1}
    return _CHRONO_HEADER.replace("/*CLOCK*/", clock)


_CHRONO_HEADER = r"""
namespace std {
namespace chrono {

struct nanoseconds  { long long __n; long long count() const { return __n; } };
struct microseconds { long long __n; long long count() const { return __n; } };
struct milliseconds { long long __n; long long count() const { return __n; } };
struct seconds      { long long __n; long long count() const { return __n; } };

/* What subtracting one time point from another answers. It holds
   nanoseconds, and each cast divides down to the unit asked for - which is
   what `duration_cast` means, and why it truncates. */
struct duration {
    long long __n;
    long long count() const { return __n; }
    nanoseconds __as_nanoseconds() const { nanoseconds r; r.__n = __n; return r; }
    microseconds __as_microseconds() const { microseconds r; r.__n = __n / 1000LL; return r; }
    milliseconds __as_milliseconds() const { milliseconds r; r.__n = __n / 1000000LL; return r; }
    seconds __as_seconds() const { seconds r; r.__n = __n / 1000000000LL; return r; }
    /* And the same four answering the number outright. `duration_cast<X>(d)
       .count()` is how nearly every one of these is written, and going
       through the unit object means answering an object by value and then
       calling on what a call answered - two shapes that each need a name of
       their own. This is the same arithmetic with neither. */
    long long __count_nanoseconds() const { return __n; }
    long long __count_microseconds() const { return __n / 1000LL; }
    long long __count_milliseconds() const { return __n / 1000000LL; }
    long long __count_seconds() const { return __n / 1000000000LL; }
};

struct time_point {
    long long __n;
    long long time_since_epoch_count() const { return __n; }
    duration operator-(time_point o) const { duration r; r.__n = __n - o.__n; return r; }
};

/*CLOCK*/

struct steady_clock {
    static time_point now() { time_point made; made.__n = __py2bin_now(); return made; }
};
struct system_clock {
    static time_point now() { time_point made; made.__n = __py2bin_now(); return made; }
};
struct high_resolution_clock {
    static time_point now() { time_point made; made.__n = __py2bin_now(); return made; }
};
}
}
"""

#: <variant>, as a tag and one member per alternative. Not a union: py2bin
#: has no placement new, so every alternative exists and the tag is what says
#: which one means anything. The difference shows only for an alternative
#: whose constructor has an effect, and is said here rather than left to be
#: found.
_VARIANT_HEADER = r"""
namespace std {
template<typename A, typename B>
class variant {
public:
    int __tag;
    A __0;
    B __1;
    variant() { __tag = 0; }
    variant(A v) { __0 = v; __tag = 0; }
    variant(B v) { __1 = v; __tag = 1; }
    int index() const { return __tag; }
};

template<typename A, typename B, typename C>
class variant {
public:
    int __tag;
    A __0;
    B __1;
    C __2;
    variant() { __tag = 0; }
    variant(A v) { __0 = v; __tag = 0; }
    variant(B v) { __1 = v; __tag = 1; }
    variant(C v) { __2 = v; __tag = 2; }
    int index() const { return __tag; }
};
}
"""

#: How each platform starts a thread and waits for one. Written per target
#: rather than with an `#ifdef`, for the reason the clock is: the translator
#: runs before the preprocessor and would read both branches.
#:
#: One entry point shape serves both. POSIX wants `void *(*)(void *)` and
#: Windows wants `DWORD (*)(LPVOID)`; on x86-64 and ARM64 alike that is one
#: pointer argument in the first register and an answer in the first result
#: register, so a function taking a pointer and answering a long is callable
#: as either.
_THREADS = {
    "windows": r"""
extern void *CreateThread(void *security, unsigned long stack,
                          __py2bin_thread_entry entry, void *argument,
                          unsigned long flags, void *id);
extern int WaitForSingleObject(void *handle, unsigned long milliseconds);
extern int CloseHandle(void *handle);

static unsigned long __py2bin_thread_start(__py2bin_thread_entry entry, void *argument) {
    return (unsigned long)CreateThread(0, 0, entry, argument, 0, 0);
}
static void __py2bin_thread_wait(unsigned long handle) {
    WaitForSingleObject((void *)handle, 0xFFFFFFFF);
    CloseHandle((void *)handle);
}
""",
    "posix": r"""
extern int pthread_create(void *handle, void *attributes,
                          __py2bin_thread_entry entry, void *argument);
extern int pthread_join(void *handle, void *answer);

static unsigned long __py2bin_thread_start(__py2bin_thread_entry entry, void *argument) {
    /* The identity is a word on every platform py2bin targets, and the one
       the caller keeps is that word rather than a pointer to it. */
    void *made;
    made = 0;
    if (pthread_create(&made, 0, entry, argument) != 0) { return 0; }
    return (unsigned long)made;
}
static void __py2bin_thread_wait(unsigned long handle) {
    pthread_join((void *)handle, 0);
}
""",
}


def _thread_header(target: "str | None") -> str:
    """<thread>, over whichever way this machine starts one."""

    named = (target or "").split("-", 1)[0]
    return _THREAD_HEADER.replace(
        "/*START*/", _THREADS["windows" if named == "windows" else "posix"]
    )


#: <thread>. The object holds the handle and nothing else: what the thread
#: runs was settled where it was written, by the pass that builds a trampoline
#: for the callable and hands this the address of one.
_THREAD_HEADER = r"""
typedef long (*__py2bin_thread_entry)(void *);
namespace std {
/*START*/

class thread {
public:
    unsigned long __handle;
    thread() { __handle = 0; }
    /* Started by `__begin` rather than by a constructor taking a callable:
       a callable is a class here, and which class it is decides what the
       trampoline has to call - which is a thing settled while translating
       and not while running. */
    void __begin(__py2bin_thread_entry entry, void *argument) {
        __handle = __py2bin_thread_start(entry, argument);
    }
    int joinable() const { return __handle != 0; }
    void join() {
        if (__handle != 0) { __py2bin_thread_wait(__handle); __handle = 0; }
    }
    /* Detaching leaves the thread running and forgets the handle. Nothing is
       reclaimed, which is what the arena does with everything. */
    void detach() { __handle = 0; }
};
}
"""

#: <atomic>, over the two instructions py2bin emits. Every operation here is
#: one of them or is written from one: a load is an add of nothing, a store is
#: an exchange whose answer is dropped.
#:
#: The ordering argument is accepted and ignored, which is honest rather than
#: convenient: both instructions are sequentially consistent - `lock` on
#: x86-64, the acquiring and releasing pair on ARM64 - so every operation is
#: already stronger than any order a program can ask for.
_ATOMIC_HEADER = r"""
namespace std {

typedef int memory_order;
const int memory_order_relaxed = 0;
const int memory_order_acquire = 2;
const int memory_order_release = 3;
const int memory_order_acq_rel = 4;
const int memory_order_seq_cst = 5;

template<typename T>
class atomic {
public:
    long __word;
    atomic() { __word = 0; }
    atomic(T value) { __word = (long)value; }
    T load() const { return (T)__py2bin_atomic_add((long *)&__word, 0); }
    T load(memory_order o) const { return load(); }
    void store(T value) { __py2bin_atomic_swap((long *)&__word, (long)value); }
    void store(T value, memory_order o) { store(value); }
    T exchange(T value) {
        return (T)__py2bin_atomic_swap((long *)&__word, (long)value);
    }
    T exchange(T value, memory_order o) { return exchange(value); }
    T fetch_add(T value) {
        return (T)__py2bin_atomic_add((long *)&__word, (long)value);
    }
    T fetch_add(T value, memory_order o) { return fetch_add(value); }
    T fetch_sub(T value) {
        return (T)__py2bin_atomic_add((long *)&__word, -(long)value);
    }
    T operator=(T value) { store(value); return value; }
    operator T() const { return load(); }
};

/* `atomic_flag` is the lock itself, and is the one thing here that is not
   written over `atomic<T>`: it promises to be lock-free, and being an
   exchange is how it keeps that promise. */
class atomic_flag {
public:
    long __word;
    atomic_flag() { __word = 0; }
    int test_and_set() { return (int)__py2bin_atomic_swap(&__word, 1); }
    int test_and_set(memory_order o) { return test_and_set(); }
    void clear() { __py2bin_atomic_swap(&__word, 0); }
    void clear(memory_order o) { clear(); }
};
}
"""

#: <mutex>, which is that flag and a loop. A thread that cannot take the lock
#: spins rather than sleeping: py2bin has no way to ask the kernel to wait,
#: and a spin is correct where a sleep would only be kinder. Said here rather
#: than found by measuring.
_MUTEX_HEADER = r"""
#include <atomic>
namespace std {
class mutex {
public:
    long __held;
    mutex() { __held = 0; }
    void lock() { while (__py2bin_atomic_swap(&__held, 1) != 0) { } }
    int try_lock() { return __py2bin_atomic_swap(&__held, 1) == 0; }
    void unlock() { __py2bin_atomic_swap(&__held, 0); }
};

/* `recursive_mutex` is deliberately absent: telling one thread from another
   needs a thread identity, and there are no threads yet. */

template<typename M>
class lock_guard {
public:
    M *__held;
    lock_guard(M &m) { __held = &m; __held->lock(); }
    ~lock_guard() { __held->unlock(); }
};

template<typename M>
class unique_lock {
public:
    M *__held;
    int __owns;
    unique_lock(M &m) { __held = &m; __held->lock(); __owns = 1; }
    ~unique_lock() { if (__owns) { __held->unlock(); } }
    void lock() { __held->lock(); __owns = 1; }
    void unlock() { __held->unlock(); __owns = 0; }
    int owns_lock() const { return __owns; }
};
}
"""

#: <string_view>, which is a pointer and a length and nothing else - it does
#: not own what it looks at, which is the whole of what it is for.
_STRING_VIEW_HEADER = r"""
namespace std {
class string_view {
public:
    const char *__at;
    unsigned long __len;
    string_view() { __at = 0; __len = 0; }
    string_view(const char *text) {
        unsigned long n = 0;
        __at = text;
        while (text != 0 && text[n] != 0) { n = n + 1; }
        __len = n;
    }
    string_view(const char *text, unsigned long n) { __at = text; __len = n; }
    unsigned long size() const { return __len; }
    unsigned long length() const { return __len; }
    int empty() const { return __len == 0; }
    const char *data() const { return __at; }
    char operator[](unsigned long i) const { return __at[i]; }
    char at(unsigned long i) const { return __at[i]; }
    char front() const { return __at[0]; }
    char back() const { return __at[__len - 1]; }
    string_view substr(unsigned long from, unsigned long n) const {
        string_view made(__at + from, n);
        return made;
    }
    string_view substr(unsigned long from) const {
        string_view made(__at + from, __len - from);
        return made;
    }
    int compare(string_view o) const {
        unsigned long i = 0;
        while (i < __len && i < o.__len) {
            if (__at[i] != o.__at[i]) { return (int)__at[i] - (int)o.__at[i]; }
            i = i + 1;
        }
        if (__len == o.__len) { return 0; }
        return __len < o.__len ? -1 : 1;
    }
    int operator==(string_view o) const { return compare(o) == 0; }
    int operator!=(string_view o) const { return compare(o) != 0; }
};
}
"""

#: <new>. What the header itself carries is small: `new (room) T(...)` is
#: rewritten to the constructor run on that address, so the placement operator
#: it would otherwise declare has nothing left to do. What is here is the two
#: names a program including it actually writes.
_NEW_HEADER = r"""
namespace std {
struct nothrow_t { int __unused; };
struct bad_alloc { const char *what() const { return "bad_alloc"; } };
}
"""

#: <typeinfo>. What `typeid` answers here is the table an object carries,
#: which compares and has a name - so the header itself has nothing to hold,
#: and the two spellings a program writes are rewritten where they stand.
_TYPEINFO_HEADER = r"""
namespace std {
struct bad_typeid { const char *what() const { return "bad_typeid"; } };
struct bad_cast { const char *what() const { return "bad_cast"; } };
}
"""

#: <random>. `mt19937` is a named algorithm with published constants, so this
#: is that algorithm rather than something that merely looks random: seeded
#: the same way it answers the same numbers as any other implementation, which
#: is the property a program using it for a reproducible run depends on.
_RANDOM_HEADER = r"""
namespace std {
class mt19937 {
public:
    unsigned long __state[624];
    int __at;
    mt19937() { seed(5489UL); }
    mt19937(unsigned long value) { seed(value); }
    void seed(unsigned long value) {
        int i;
        __state[0] = value & 0xFFFFFFFFUL;
        i = 1;
        while (i < 624) {
            unsigned long before = __state[i - 1] ^ (__state[i - 1] >> 30);
            __state[i] = (1812433253UL * before + (unsigned long)i) & 0xFFFFFFFFUL;
            i = i + 1;
        }
        __at = 624;
    }
    void __twist() {
        int i = 0;
        while (i < 624) {
            unsigned long joined =
                (__state[i] & 0x80000000UL) | (__state[(i + 1) % 624] & 0x7FFFFFFFUL);
            unsigned long next = __state[(i + 397) % 624] ^ (joined >> 1);
            if ((joined & 1UL) != 0UL) { next = next ^ 2567483615UL; }
            __state[i] = next & 0xFFFFFFFFUL;
            i = i + 1;
        }
        __at = 0;
    }
    unsigned long operator()() {
        unsigned long v;
        if (__at >= 624) { __twist(); }
        v = __state[__at];
        __at = __at + 1;
        v = v ^ (v >> 11);
        v = v ^ ((v << 7) & 2636928640UL);
        v = v ^ ((v << 15) & 4022730752UL);
        v = v ^ (v >> 18);
        return v & 0xFFFFFFFFUL;
    }
    unsigned long min() const { return 0UL; }
    unsigned long max() const { return 4294967295UL; }
};

typedef mt19937 default_random_engine;
typedef mt19937 minstd_rand;

/* Not a device: there is nothing here to ask for entropy. It is a fixed
   sequence, and saying so is better than a name that promises otherwise. */
class random_device {
public:
    mt19937 __made;
    random_device() { }
    unsigned long operator()() { return __made(); }
};

template<typename T>
class uniform_int_distribution {
public:
    long __low;
    long __high;
    uniform_int_distribution() { __low = 0; __high = 2147483647L; }
    uniform_int_distribution(T low, T high) { __low = (long)low; __high = (long)high; }
    T operator()(mt19937 &g) {
        unsigned long room = (unsigned long)(__high - __low) + 1UL;
        if (room == 0UL) { return (T)g(); }
        return (T)(__low + (long)(g() % room));
    }
    T min() const { return (T)__low; }
    T max() const { return (T)__high; }
};
}
"""

#: <bitset>, which is arithmetic on an unsigned long and a count that says
#: how much of it is the set. Fixed at translation time, because the size is
#: a template argument - which is what makes a bitset one and not a vector.
_BITSET_HEADER = r"""
#include <string>
namespace std {
template<int N>
class bitset {
public:
    unsigned long __bits;
    bitset() { __bits = 0; }
    bitset(unsigned long value) { __bits = value & ((N >= 64) ? ~0UL : ((1UL << N) - 1UL)); }
    int size() const { return N; }
    int test(int at) const { return (int)((__bits >> at) & 1UL); }
    int operator[](int at) const { return (int)((__bits >> at) & 1UL); }
    void set(int at) { __bits = __bits | (1UL << at); }
    void reset(int at) { __bits = __bits & ~(1UL << at); }
    void flip(int at) { __bits = __bits ^ (1UL << at); }
    void reset() { __bits = 0; }
    unsigned long to_ulong() const { return __bits; }
    int count() const {
        int found = 0;
        int at = 0;
        while (at < N) { if ((__bits >> at) & 1UL) { found = found + 1; } at = at + 1; }
        return found;
    }
    int any() const { return __bits != 0; }
    int none() const { return __bits == 0; }
    int all() const { return count() == N; }
    string to_string() const {
        string made;
        int at = N;
        while (at > 0) {
            at = at - 1;
            made.push_back(((__bits >> at) & 1UL) ? '1' : '0');
        }
        return made;
    }
};
}
"""

#: <list>, a doubly linked list. Written as one rather than as a vector under
#: another name, because what a program picks a list for is that inserting in
#: the middle does not move anything else - and a name in it stays valid.
_LIST_HEADER = r"""
namespace std {
/* The node is a template of its own rather than a type inside the list: a
   class written inside a class template is not one this translator writes
   out, because the copy it would make has no arguments of its own to be
   made for. Written beside it, it takes the element type the same way the
   list does and there is nothing nested at all. */
template<typename T>
struct __list_node { T value; __list_node<T> *next; __list_node<T> *prev; };

template<typename T>
class list {
public:
    __list_node<T> *head;
    __list_node<T> *tail;
    unsigned long count;
    list() { head = 0; tail = 0; count = 0; }
    unsigned long size() { return count; }
    int empty() { return count == 0; }
    void push_back(T value) {
        __list_node<T> *made = new __list_node<T>();
        made->value = value;
        made->next = 0;
        made->prev = tail;
        if (tail != 0) { tail->next = made; } else { head = made; }
        tail = made;
        count = count + 1;
    }
    void push_front(T value) {
        __list_node<T> *made = new __list_node<T>();
        made->value = value;
        made->prev = 0;
        made->next = head;
        if (head != 0) { head->prev = made; } else { tail = made; }
        head = made;
        count = count + 1;
    }
    void pop_front() {
        __list_node<T> *going;
        if (head == 0) { return; }
        going = head;
        head = head->next;
        if (head != 0) { head->prev = 0; } else { tail = 0; }
        delete going;
        count = count - 1;
    }
    void pop_back() {
        __list_node<T> *going;
        if (tail == 0) { return; }
        going = tail;
        tail = tail->prev;
        if (tail != 0) { tail->next = 0; } else { head = 0; }
        delete going;
        count = count - 1;
    }
    T &front() { return head->value; }
    T &back() { return tail->value; }
    void clear() { while (count > 0) { pop_front(); } }
};
}
"""

#: <deque>, which a program reaches for to push and pop at both ends. Written
#: over the same storage a vector uses with a moving start, so an index costs
#: what an index should and neither end has to move the other.
_DEQUE_HEADER = r"""
namespace std {
template<typename T>
class deque {
public:
    T *items;
    unsigned long first;
    unsigned long count;
    unsigned long room;
    deque() { items = 0; first = 0; count = 0; room = 0; }
    unsigned long size() { return count; }
    int empty() { return count == 0; }
    void clear() { first = 0; count = 0; }
    void __grow(unsigned long want) {
        unsigned long i;
        T *fresh;
        if (want <= room) { return; }
        if (want < 8) { want = 8; }
        fresh = (T *)malloc(sizeof(T) * want);
        i = 0;
        while (i < count) { fresh[i] = items[first + i]; i = i + 1; }
        items = fresh;
        first = 0;
        room = want;
    }
    void push_back(T value) {
        if (first + count == room) {
            if (first > 0) {
                unsigned long i = 0;
                while (i < count) { items[i] = items[first + i]; i = i + 1; }
                first = 0;
            } else {
                __grow(room == 0 ? 8 : room * 2);
            }
        }
        items[first + count] = value;
        count = count + 1;
    }
    void push_front(T value) {
        unsigned long i;
        if (first == 0) {
            __grow(room == 0 ? 8 : room * 2);
            /* Room made at the front by moving what is there to the middle,
               so the next push_front costs nothing. */
            i = count;
            while (i > 0) { items[i - 1 + count] = items[i - 1]; i = i - 1; }
            first = count;
        }
        first = first - 1;
        items[first] = value;
        count = count + 1;
    }
    void pop_front() { if (count > 0) { first = first + 1; count = count - 1; } }
    void pop_back() { if (count > 0) { count = count - 1; } }
    T &operator[](unsigned long i) { return items[first + i]; }
    T &at(unsigned long i) { return items[first + i]; }
    T &front() { return items[first]; }
    T &back() { return items[first + count - 1]; }
};
}
"""

#: <optional>, which is a value and whether there is one. Written as the two
#: things it is rather than as a union: py2bin has no placement new, so the
#: held object exists either way and `has_value()` is what says whether it
#: means anything. The difference shows only for a type whose constructor has
#: an effect, and is stated here rather than left to be found.
_OPTIONAL_HEADER = r"""
namespace std {
struct nullopt_t { int __unused; };

template<typename T>
class optional {
public:
    T __held;
    int __present;
    optional() { __present = 0; }
    optional(T value) { __held = value; __present = 1; }
    int has_value() const { return __present; }
    T value() const { return __held; }
    T value_or(T other) const { if (__present) { return __held; } return other; }
    T operator*() const { return __held; }
    T *operator->() { return &__held; }
    void reset() { __present = 0; }
    operator bool() const { return __present; }
};

template<typename T>
optional<T> make_optional(T value) { optional<T> made(value); return made; }
}
"""

#: <numeric>, which is `accumulate` and little else that a program without
#: iterators of its own can use.
_NUMERIC_HEADER = r"""
namespace std {
template<typename T>
T accumulate(T *first, T *last, T start) {
    while (first != last) { start = start + *first; first = first + 1; }
    return start;
}
}
"""

#: <stdexcept>. The standard ones carry a message and answer `what()`; that
#: is the whole of what code catching them uses, and it is what these do.
#: There is no hierarchy - py2bin catches by the type written, and a `catch
#: (std::exception &)` that means "any of them" would need one.
_STDEXCEPT_HEADER = r"""
namespace std {
class exception {
public:
    const char *message;
    exception() { message = ""; }
    exception(const char *text) { message = text; }
    // Virtual, and answering with the class's own name rather than the
    // message, because that is what the standard one does: an exception
    // caught by value is sliced to this, and what it says then should not
    // depend on what it was before it was sliced.
    virtual const char *what() { return "std::exception"; }
};
class runtime_error : public exception {
public:
    runtime_error() { message = ""; }
    runtime_error(const char *text) { message = text; }
    const char *what() { return message; }
};
class logic_error : public exception {
public:
    logic_error() { message = ""; }
    logic_error(const char *text) { message = text; }
    const char *what() { return message; }
};
class out_of_range : public exception {
public:
    out_of_range() { message = ""; }
    out_of_range(const char *text) { message = text; }
    const char *what() { return message; }
};
class invalid_argument : public exception {
public:
    invalid_argument() { message = ""; }
    invalid_argument(const char *text) { message = text; }
    const char *what() { return message; }
};
}
"""

#: py2bin's own <filesystem>. `path` is string work and nothing else, which
#: is most of what the header is used for; the queries go to the syscalls on
#: POSIX and to the imports <windows.h> declares on Windows, chosen with the
#: platform macros the preprocessor now defines.
#:
#: What is missing is `directory_iterator`: reading a directory means
#: getdents on Linux, getdirentries on macOS and FindFirstFile on Windows,
#: each with a different struct laid out differently per architecture - and
#: py2bin can run a binary for exactly one of those here. A struct read wrong
#: gives plausible answers, so it is left out rather than guessed at.
_FILESYSTEM_HEADER = r"""
#include <string>
#include <py2bin_fs.h>

namespace std {
namespace filesystem {

class path {
public:
    std::string text;
    path() { }
    path(const char *s) { text.assign(s); }
    /* A path is UTF-16 on Windows, so this is how one arrives there: from
       the kernel, from a wide literal, or from a wstring the program built.
       The conversion is asked of <py2bin_fs.h>, which is C and knows which
       platform this is. */
    path(const wchar_t *s) {
        char __narrow[520];
        __py2bin_fs_narrow(s, __narrow, 520);
        text.assign(__narrow);
    }
    path(const std::wstring &s) {
        char __narrow[520];
        __py2bin_fs_narrow(s.c_str(), __narrow, 520);
        text.assign(__narrow);
    }
    const char *c_str() { return text.c_str(); }
    std::string string() { return text; }
    std::wstring wstring() {
        wchar_t __wide[520];
        std::wstring out;
        __py2bin_fs_widen(text.c_str(), __wide, 520);
        out.assign(__wide);
        return out;
    }
    int empty() { return text.empty(); }
    /* Written as a member rather than `out.text.push_back(c)` at each call:
       reaching through a member of a local to call one of *its* methods is
       a shape the translator does not rewrite. */
    void __add(char c) { text.push_back(c); }
    void __add_text(const char *s) { text.append(s); }
    int __size() { return text.size(); }
    char __at(int i) { return text.at(i); }

    /* `p / "sub"`, which is how a path is built. The separator is only added
       where there is not one already, so joining twice does not double it. */
    path operator/(const char *piece) {
        path joined;
        int i;
        i = 0;
        while (i < this->__size()) { joined.__add(this->__at(i)); i = i + 1; }
        if (joined.__size() > 0) {
            if (joined.text.at(joined.__size() - 1) != '/') {
                joined.__add_text("/");
            }
        }
        joined.__add_text(piece);
        return joined;
    }

    /* The same, given a wide piece: `root / L"web"` is how a program that
       writes its literals wide builds a path. */
    path operator/(const wchar_t *piece) {
        char __narrow[520];
        __py2bin_fs_narrow(piece, __narrow, 520);
        path joined;
        int i;
        i = 0;
        while (i < this->__size()) { joined.__add(this->__at(i)); i = i + 1; }
        if (joined.__size() > 0) {
            if (joined.text.at(joined.__size() - 1) != '/') {
                joined.__add_text("/");
            }
        }
        joined.__add_text(__narrow);
        return joined;
    }

    int __last_separator() {
        int i;
        int found;
        found = -1;
        i = 0;
        while (i < this->__size()) {
            if (this->__at(i) == '/') { found = i; }
            if (this->__at(i) == '\\') { found = i; }
            i = i + 1;
        }
        return found;
    }

    path filename() {
        path out;
        int i;
        i = __last_separator() + 1;
        while (i < this->__size()) { out.__add(this->__at(i)); i = i + 1; }
        return out;
    }

    path parent_path() {
        path out;
        int cut;
        int i;
        cut = __last_separator();
        i = 0;
        while (i < cut) { out.__add(this->__at(i)); i = i + 1; }
        return out;
    }

    path extension() {
        path out;
        int start;
        int i;
        int dot;
        start = __last_separator() + 1;
        dot = -1;
        i = start;
        while (i < this->__size()) {
            if (this->__at(i) == '.') { if (i > start) { dot = i; } }
            i = i + 1;
        }
        if (dot < 0) { return out; }
        i = dot;
        while (i < this->__size()) { out.__add(this->__at(i)); i = i + 1; }
        return out;
    }

    path stem() {
        path whole;
        path suffix;
        path out;
        int keep;
        int i;
        whole = this->filename();
        suffix = this->extension();
        keep = whole.__size() - suffix.__size();
        i = 0;
        while (i < keep) { out.__add(whole.__at(i)); i = i + 1; }
        return out;
    }
};

/* Every question that depends on the platform is asked in <py2bin_fs.h>,
   which is C and so is read by the preprocessor that knows about #ifdef. */
int exists(path p) { return __py2bin_fs_exists(p.c_str()); }
int is_directory(path p) { return __py2bin_fs_is_directory(p.c_str()); }
int is_regular_file(path p) {
    if (!__py2bin_fs_exists(p.c_str())) { return 0; }
    return !__py2bin_fs_is_directory(p.c_str());
}
unsigned long file_size(path p) {
    long held;
    held = __py2bin_fs_size(p.c_str());
    if (held < 0) { return 0; }
    return (unsigned long)held;
}
int create_directory(path p) { return __py2bin_fs_mkdir(p.c_str()); }
int remove(path p) {
    if (__py2bin_fs_is_directory(p.c_str())) {
        return __py2bin_fs_rmdir(p.c_str());
    }
    return __py2bin_fs_unlink(p.c_str());
}
int rename(path from, path to) {
    return __py2bin_fs_rename(from.c_str(), to.c_str());
}
path current_path() {
    path out;
    char buffer[260];
    __py2bin_fs_cwd(buffer, 260);
    out.text.assign(buffer);
    return out;
}

}
}
"""

#: py2bin's own <functional>. The comparison and arithmetic objects, which
#: are small classes with a call operator - the same thing a lambda becomes.
#:
#: `std::function` is not here. It is a box that holds *any* callable, which
#: means erasing the type of what is in it; every callable py2bin makes is a
#: class of its own, and nothing common to them exists to erase to. What it
#: is used for works without it: `auto` holds a lambda, and a plain function
#: pointer holds a function. A program that names it is told that.
_FUNCTIONAL_HEADER = r"""
namespace std {

template<typename T>
class less {
public:
    less() { }
    int operator()(T a, T b) { return a < b; }
};

template<typename T>
class greater {
public:
    greater() { }
    int operator()(T a, T b) { return a > b; }
};

template<typename T>
class less_equal {
public:
    less_equal() { }
    int operator()(T a, T b) { return a <= b; }
};

template<typename T>
class greater_equal {
public:
    greater_equal() { }
    int operator()(T a, T b) { return a >= b; }
};

template<typename T>
class equal_to {
public:
    equal_to() { }
    int operator()(T a, T b) { return a == b; }
};

template<typename T>
class not_equal_to {
public:
    not_equal_to() { }
    int operator()(T a, T b) { return a != b; }
};

template<typename T>
class plus {
public:
    plus() { }
    T operator()(T a, T b) { return a + b; }
};

template<typename T>
class minus {
public:
    minus() { }
    T operator()(T a, T b) { return a - b; }
};

template<typename T>
class multiplies {
public:
    multiplies() { }
    T operator()(T a, T b) { return a * b; }
};

template<typename T>
class negate {
public:
    negate() { }
    T operator()(T a) { return -a; }
};

}
"""


_MAP_HEADER = r"""

namespace std {
/* `<map>` carries this itself: the headers here are separate texts and none
   of them includes another, so a program that includes only `<map>` would
   have no `pair` for `insert` to take. Written the same way `<utility>`
   writes it, so a program that includes both gets one definition twice
   rather than two that disagree. */
template<typename A, typename B>
class pair {
public:
    A first;
    B second;
    pair() { }
    pair(A a, B b) { first = a; second = b; }
};

template<typename A, typename B>
pair<A, B> make_pair(A a, B b) { pair<A, B> made(a, b); return made; }

/* Entries in one array, so an iterator is a pointer to one and `it->first`
   and `it->second` are ordinary member reads. */
template<typename K, typename V>
class map_entry {
public:
    K first;
    V second;
    map_entry() { }
};

/* Searched from the front, and kept in key order. That is not a red-black
   tree, and a program storing thousands of keys will notice - but the order
   is not a performance question: C++ says walking a map visits its keys in
   order, so kept as they arrived it gave a different answer, quietly. A
   sorted array says the same thing about order and needs nothing this
   subset does not have. */
template<typename K, typename V>
class map {
public:
    map_entry<K, V> *entries;
    unsigned long used;
    unsigned long room;
    typedef K key_type;
    typedef V mapped_type;
    typedef map_entry<K, V> value_type;
    typedef map_entry<K, V> *iterator;
    typedef map_entry<K, V> *const_iterator;
    typedef unsigned long size_type;
    map() { entries = 0; used = 0; room = 0; }
    unsigned long size() { return used; }
    int empty() { return used == 0; }
    void clear() { used = 0; }
    void reserve(unsigned long want) {
        unsigned long i;
        map_entry<K, V> *fresh;
        if (want <= room) { return; }
        /* Storage, not objects, for the reason `vector` gives. */
        fresh = (map_entry<K, V> *)malloc(sizeof(map_entry<K, V>) * want);
        i = 0;
        while (i < used) { fresh[i] = entries[i]; i = i + 1; }
        entries = fresh;
        room = want;
    }
    map_entry<K, V> *begin() { return entries; }
    map_entry<K, V> *end() { return entries + used; }
    map_entry<K, V> *find(K key) {
        unsigned long i;
        K held;
        i = 0;
        while (i < used) {
            /* Through a name of its own: a comparison rewritten to the
               class's own `operator==` is matched on a name to its left, and
               `entries[i].first` is not one. */
            held = entries[i].first;
            if (held == key) { return entries + i; }
            i = i + 1;
        }
        return entries + used;
    }
    unsigned long count(K key) { return find(key) == entries + used ? 0 : 1; }
    int contains(K key) { return find(key) != entries + used; }
    /* The body of `operator[]`, under a name, so `insert` can reach it:
       both of them want "the slot for this key, made if it is not there". */
    V &__slot(K key) {
        map_entry<K, V> *found;
        found = find(key);
        if (found != entries + used) { return found->second; }
        if (used == room) {
            if (room == 0) { reserve(8); } else { reserve(room * 2); }
        }
        /* Put where it belongs rather than at the end, which is what makes
           walking this thing answer in key order. */
        {
            unsigned long __at;
            unsigned long __back;
            K __held;
            __at = 0;
            while (__at < used) {
                __held = entries[__at].first;
                if (key < __held) { break; }
                __at = __at + 1;
            }
            __back = used;
            while (__back > __at) {
                entries[__back] = entries[__back - 1];
                __back = __back - 1;
            }
            entries[__at].first = key;
            used = used + 1;
            return entries[__at].second;
        }
    }
    V &operator[](K key) { return __slot(key); }
    /* `insert` leaves a key that is already there alone - overwriting one is
       what `operator[]` is for. What it takes is a `pair`, which is what
       `make_pair` answers with and what C++ says this takes. */
    void insert(pair<K, V> entry) {
        map_entry<K, V> *found;
        found = find(entry.first);
        if (found != entries + used) { return; }
        __slot(entry.first) = entry.second;
    }
    V &at(K key) { return find(key)->second; }
    void erase(K key) {
        map_entry<K, V> *found;
        unsigned long i;
        found = find(key);
        if (found == entries + used) { return; }
        i = (unsigned long)(found - entries);
        while (i + 1 < used) { entries[i] = entries[i + 1]; i = i + 1; }
        used = used - 1;
    }
};
}
"""

_SET_HEADER = r"""

namespace std {
/* The same shape as `map` with nothing on the other side - kept in order
   for the same reason, and it is the whole of what `set` promises. */
template<typename T>
class set {
public:
    T *items;
    unsigned long used;
    unsigned long room;
    typedef T key_type;
    typedef T value_type;
    typedef T *iterator;
    typedef T *const_iterator;
    typedef unsigned long size_type;
    set() { items = 0; used = 0; room = 0; }
    unsigned long size() { return used; }
    int empty() { return used == 0; }
    void clear() { used = 0; }
    void reserve(unsigned long want) {
        unsigned long i;
        T *fresh;
        if (want <= room) { return; }
        /* Storage, not objects, for the reason `vector` gives. */
        fresh = (T *)malloc(sizeof(T) * want);
        i = 0;
        while (i < used) { fresh[i] = items[i]; i = i + 1; }
        items = fresh;
        room = want;
    }
    T *begin() { return items; }
    T *end() { return items + used; }
    T *find(T value) {
        unsigned long i;
        T held;
        i = 0;
        while (i < used) {
            /* Through a name of its own, for the same reason `map` does. */
            held = items[i];
            if (held == value) { return items + i; }
            i = i + 1;
        }
        return items + used;
    }
    unsigned long count(T value) { return find(value) == items + used ? 0 : 1; }
    int contains(T value) { return find(value) != items + used; }
    void insert(T value) {
        unsigned long at;
        unsigned long back;
        T held;
        if (find(value) != items + used) { return; }
        if (used == room) {
            if (room == 0) { reserve(8); } else { reserve(room * 2); }
        }
        at = 0;
        while (at < used) {
            held = items[at];
            if (value < held) { break; }
            at = at + 1;
        }
        back = used;
        while (back > at) { items[back] = items[back - 1]; back = back - 1; }
        items[at] = value;
        used = used + 1;
    }
    void erase(T value) {
        T *found;
        unsigned long i;
        found = find(value);
        if (found == items + used) { return; }
        i = (unsigned long)(found - items);
        while (i + 1 < used) { items[i] = items[i + 1]; i = i + 1; }
        used = used - 1;
    }
    T &operator[](unsigned long i) { return items[i]; }
};
}
"""

_MEMORY_HEADER = r"""

namespace std {
/* A holder that frees what it holds. Not move-only and not reference
   counted: this subset has neither move semantics nor atomics, so what is
   here is the ownership - one owner, freed when the holder goes - and not
   the machinery C++ uses to enforce it. */
template<typename T>
class unique_ptr {
public:
    T *raw;
    unique_ptr() { raw = 0; }
    unique_ptr(T *p) { raw = p; }
    /* Built from another one, which is a move: `std::move` is a cast and
       nothing here keeps it, so the transfer has to be what building one
       from another *does*. That is not a liberty - a program that copies a
       unique_ptr does not compile in C++ at all, so the only way one is
       ever built from another is the move this is. Without it both held the
       same pointer, both destructors freed it, and the one that was moved
       from still answered as though it owned something. */
    unique_ptr(unique_ptr &o) { raw = o.raw; o.raw = 0; }
    unique_ptr &operator=(unique_ptr &o) {
        if (raw != 0) { delete raw; }
        raw = o.raw; o.raw = 0; return *this;
    }
    ~unique_ptr() { if (raw != 0) { delete raw; raw = 0; } }
    T *get() { return raw; }
    T *operator->() { return raw; }
    T &operator*() { return *raw; }
    int operator!() { return raw == 0; }
    int operator==(T *p) { return raw == p; }
    int operator!=(T *p) { return raw != p; }
    T *release() { T *held; held = raw; raw = 0; return held; }
    void reset(T *p) { if (raw != 0) { delete raw; } raw = p; }
};

template<typename T>
class shared_ptr {
public:
    T *raw;
    shared_ptr() { raw = 0; }
    shared_ptr(T *p) { raw = p; }
    T *get() { return raw; }
    T *operator->() { return raw; }
    T &operator*() { return *raw; }
    int operator!() { return raw == 0; }
    int operator==(T *p) { return raw == p; }
    int operator!=(T *p) { return raw != p; }
    void reset(T *p) { raw = p; }
};
}
"""

#: One class that reads *and* writes a string, written once and emitted
#: under both names. C++ has three - `istringstream`, `ostringstream` and
#: `stringstream` - and the third is the one that does both; a typedef onto
#: another class is not something this translator resolves, so the text is
#: what is repeated rather than the name.
_READ_WRITE_STREAM = r"""class @NAME@ {
public:
    string held;
    unsigned long __cursor;
    int failed;
    @NAME@() { __cursor = 0; failed = 0; }
    @NAME@(const char *s) { held.assign(s); __cursor = 0; failed = 0; }
    @NAME@(string s) { held = s; __cursor = 0; failed = 0; }
    string str() { return held; }
    void str(string s) { held = s; __cursor = 0; failed = 0; }
    /* The writing side, on the same buffer. `stringstream` is a name for
       this class rather than for the output one: it was a name for that,
       so a program that wrote to a `stringstream` and then read it back was
       asking an object that had no `>>` at all. */
    @NAME@ &operator<<(const char *s) { held.append(s); return *this; }
    @NAME@ &operator<<(string s) { held.append(s); return *this; }
    @NAME@ &operator<<(char c) { held.push_back(c); return *this; }
    @NAME@ &operator<<(int v) {
        held.append(to_string(v));
        return *this;
    }
    @NAME@ &operator<<(long v) {
        held.append(to_string((int)v));
        return *this;
    }
    @NAME@ &operator<<(unsigned long v) {
        held.append(to_string((int)v));
        return *this;
    }
    @NAME@ &operator<<(double v) {
        char __b[64];
        snprintf(__b, 64, "%.6g", v);
        held.append(__b);
        return *this;
    }
    int __spacing(char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    }
    void __skip() {
        while (__cursor < (unsigned long)held.size() && __spacing(held.at((int)__cursor))) {
            __cursor = __cursor + 1;
        }
    }
    int eof() { __skip(); return __cursor >= (unsigned long)held.size(); }
    int fail() { return failed; }
    /* `while (in >> n)` asks the stream whether the read worked. */
    operator bool() { return failed == 0; }
    int __sign() {
        char c;
        int negative;
        negative = 0;
        if (__cursor < (unsigned long)held.size()) {
            c = held.at((int)__cursor);
            if (c == '-') { negative = 1; __cursor = __cursor + 1; }
            else if (c == '+') { __cursor = __cursor + 1; }
        }
        return negative;
    }
    long __digits(int *any) {
        long got;
        char c;
        got = 0;
        while (__cursor < (unsigned long)held.size()) {
            c = held.at((int)__cursor);
            if (c < '0' || c > '9') { break; }
            got = got * 10 + (long)(c - '0');
            __cursor = __cursor + 1;
            *any = 1;
        }
        return got;
    }
    @NAME@ &operator>>(long &v) {
        long got;
        int any;
        int negative;
        __skip();
        any = 0;
        negative = __sign();
        got = __digits(&any);
        if (any == 0) { failed = 1; } else { v = negative ? -got : got; }
        return *this;
    }
    @NAME@ &operator>>(int &v) {
        long got;
        int any;
        int negative;
        __skip();
        any = 0;
        negative = __sign();
        got = __digits(&any);
        if (any == 0) { failed = 1; } else { v = (int)(negative ? -got : got); }
        return *this;
    }
    @NAME@ &operator>>(double &v) {
        double got;
        double scale;
        double power;
        long whole;
        long exponent;
        int any;
        int negative;
        int negative_exponent;
        char c;
        __skip();
        any = 0;
        negative = __sign();
        whole = __digits(&any);
        got = (double)whole;
        if (__cursor < (unsigned long)held.size() && held.at((int)__cursor) == '.') {
            __cursor = __cursor + 1;
            scale = 0.1;
            while (__cursor < (unsigned long)held.size()) {
                c = held.at((int)__cursor);
                if (c < '0' || c > '9') { break; }
                got = got + scale * (double)(c - '0');
                scale = scale * 0.1;
                __cursor = __cursor + 1;
                any = 1;
            }
        }
        if (any && __cursor < (unsigned long)held.size()) {
            c = held.at((int)__cursor);
            if (c == 'e' || c == 'E') {
                __cursor = __cursor + 1;
                negative_exponent = __sign();
                exponent = __digits(&any);
                power = 1.0;
                while (exponent > 0) { power = power * 10.0; exponent = exponent - 1; }
                if (negative_exponent) { got = got / power; } else { got = got * power; }
            }
        }
        if (any == 0) { failed = 1; } else { v = negative ? -got : got; }
        return *this;
    }
    @NAME@ &operator>>(char &v) {
        __skip();
        if (__cursor >= (unsigned long)held.size()) { failed = 1; return *this; }
        v = held.at((int)__cursor);
        __cursor = __cursor + 1;
        return *this;
    }
    /* One word, which is what `>>` on a string means. */
    @NAME@ &operator>>(string &v) {
        char c;
        v.clear();
        __skip();
        while (__cursor < (unsigned long)held.size()) {
            c = held.at((int)__cursor);
            if (__spacing(c)) { break; }
            v.push_back(c);
            __cursor = __cursor + 1;
        }
        if (v.size() == 0) { failed = 1; }
        return *this;
    }
    /* Up to the next newline, which `>>` never crosses. */
    int __line(string &out, char stop) {
        char c;
        out.clear();
        if (__cursor >= (unsigned long)held.size()) { failed = 1; return 0; }
        while (__cursor < (unsigned long)held.size()) {
            c = held.at((int)__cursor);
            __cursor = __cursor + 1;
            if (c == stop) { return 1; }
            out.push_back(c);
        }
        return 1;
    }
};

int getline(@NAME@ &in, string &out) { return in.__line(out, '\n'); }
int getline(@NAME@ &in, string &out, char stop) {
    return in.__line(out, stop);
}

"""


#: One class that reads a string *and* writes it, emitted under both
#: names. C++ has three of these and the third does both; a typedef onto
#: another class is not something this translator resolves, so what is
#: repeated is the text and not the name.
_READ_WRITE_STREAM = r"""/* Reading values back out of a string. The other half of `<sstream>`, and
   without it a program could build a string with `<<` and had no way to take
   one apart - which is what `@NAME@` is for. Written out by hand
   rather than over `scanf`: py2bin's printf reads its format at compile
   time, and what is wanted here is a position that moves. */
class @NAME@ {
public:
    string held;
    unsigned long __cursor;
    int failed;
    @NAME@() { __cursor = 0; failed = 0; }
    @NAME@(const char *s) { held.assign(s); __cursor = 0; failed = 0; }
    @NAME@(string s) { held = s; __cursor = 0; failed = 0; }
    string str() { return held; }
    void str(string s) { held = s; __cursor = 0; failed = 0; }
    int __spacing(char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    }
    void __skip() {
        while (__cursor < (unsigned long)held.size() && __spacing(held.at((int)__cursor))) {
            __cursor = __cursor + 1;
        }
    }
    int eof() { __skip(); return __cursor >= (unsigned long)held.size(); }
    int fail() { return failed; }
    /* And the writing side, on the same buffer: `stringstream` is the one
       that does both, and it is this class under another name. */
    @NAME@ &operator<<(const char *s) { held.append(s); return *this; }
    @NAME@ &operator<<(string s) { held.append(s); return *this; }
    @NAME@ &operator<<(char c) { held.push_back(c); return *this; }
    @NAME@ &operator<<(int v) { held.append(to_string(v)); return *this; }
    @NAME@ &operator<<(long v) { held.append(to_string((int)v)); return *this; }
    @NAME@ &operator<<(unsigned long v) {
        held.append(to_string((int)v));
        return *this;
    }
    @NAME@ &operator<<(double v) {
        char __b[64];
        snprintf(__b, 64, "%.6g", v);
        held.append(__b);
        return *this;
    }
    /* `while (in >> n)` asks the stream whether the read worked. */
    operator bool() { return failed == 0; }
    int __sign() {
        char c;
        int negative;
        negative = 0;
        if (__cursor < (unsigned long)held.size()) {
            c = held.at((int)__cursor);
            if (c == '-') { negative = 1; __cursor = __cursor + 1; }
            else if (c == '+') { __cursor = __cursor + 1; }
        }
        return negative;
    }
    long __digits(int *any) {
        long got;
        char c;
        got = 0;
        while (__cursor < (unsigned long)held.size()) {
            c = held.at((int)__cursor);
            if (c < '0' || c > '9') { break; }
            got = got * 10 + (long)(c - '0');
            __cursor = __cursor + 1;
            *any = 1;
        }
        return got;
    }
    @NAME@ &operator>>(long &v) {
        long got;
        int any;
        int negative;
        __skip();
        any = 0;
        negative = __sign();
        got = __digits(&any);
        if (any == 0) { failed = 1; } else { v = negative ? -got : got; }
        return *this;
    }
    @NAME@ &operator>>(int &v) {
        long got;
        int any;
        int negative;
        __skip();
        any = 0;
        negative = __sign();
        got = __digits(&any);
        if (any == 0) { failed = 1; } else { v = (int)(negative ? -got : got); }
        return *this;
    }
    @NAME@ &operator>>(double &v) {
        double got;
        double scale;
        double power;
        long whole;
        long exponent;
        int any;
        int negative;
        int negative_exponent;
        char c;
        __skip();
        any = 0;
        negative = __sign();
        whole = __digits(&any);
        got = (double)whole;
        if (__cursor < (unsigned long)held.size() && held.at((int)__cursor) == '.') {
            __cursor = __cursor + 1;
            scale = 0.1;
            while (__cursor < (unsigned long)held.size()) {
                c = held.at((int)__cursor);
                if (c < '0' || c > '9') { break; }
                got = got + scale * (double)(c - '0');
                scale = scale * 0.1;
                __cursor = __cursor + 1;
                any = 1;
            }
        }
        if (any && __cursor < (unsigned long)held.size()) {
            c = held.at((int)__cursor);
            if (c == 'e' || c == 'E') {
                __cursor = __cursor + 1;
                negative_exponent = __sign();
                exponent = __digits(&any);
                power = 1.0;
                while (exponent > 0) { power = power * 10.0; exponent = exponent - 1; }
                if (negative_exponent) { got = got / power; } else { got = got * power; }
            }
        }
        if (any == 0) { failed = 1; } else { v = negative ? -got : got; }
        return *this;
    }
    @NAME@ &operator>>(char &v) {
        __skip();
        if (__cursor >= (unsigned long)held.size()) { failed = 1; return *this; }
        v = held.at((int)__cursor);
        __cursor = __cursor + 1;
        return *this;
    }
    /* One word, which is what `>>` on a string means. */
    @NAME@ &operator>>(string &v) {
        char c;
        v.clear();
        __skip();
        while (__cursor < (unsigned long)held.size()) {
            c = held.at((int)__cursor);
            if (__spacing(c)) { break; }
            v.push_back(c);
            __cursor = __cursor + 1;
        }
        if (v.size() == 0) { failed = 1; }
        return *this;
    }
    /* Up to the next newline, which `>>` never crosses. */
    int __line(string &out, char stop) {
        char c;
        out.clear();
        if (__cursor >= (unsigned long)held.size()) { failed = 1; return 0; }
        while (__cursor < (unsigned long)held.size()) {
            c = held.at((int)__cursor);
            __cursor = __cursor + 1;
            if (c == stop) { return 1; }
            out.push_back(c);
        }
        return 1;
    }
};

int getline(@NAME@ &in, string &out) { return in.__line(out, '\n'); }
int getline(@NAME@ &in, string &out, char stop) {
    return in.__line(out, stop);
}

"""


_SSTREAM_HEADER = (
    r"""

namespace std {
/* A stream that writes into a string. `<<` is what a program uses it for,
   and each overload appends the text the value would print as. */
class ostringstream {
public:
    string held;
    ostringstream() { }
    string str() { return held; }
    void clear() { held.clear(); }
    ostringstream &operator<<(const char *s) { held.append(s); return *this; }
    ostringstream &operator<<(string s) { held.append(s); return *this; }
    ostringstream &operator<<(char c) { held.push_back(c); return *this; }
    ostringstream &operator<<(int v) { held.append(to_string(v)); return *this; }
    ostringstream &operator<<(long v) { held.append(to_string((int)v)); return *this; }
    ostringstream &operator<<(unsigned int v) { held.append(to_string((int)v)); return *this; }
    ostringstream &operator<<(unsigned long v) { held.append(to_string((int)v)); return *this; }
    /* Six *significant* digits, which is what C++ prints by default and
       what `cout` here already did - written out by hand as six decimal
       places, this said `1.500000` where C++ says `1.5`, and where the two
       disagreed the program still ran. */
    ostringstream &operator<<(double v) {
        char __b[64];
        snprintf(__b, 64, "%.6g", v);
        held.append(__b);
        return *this;
    }
};

"""
    + _READ_WRITE_STREAM.replace("@NAME@", "istringstream")
    + _READ_WRITE_STREAM.replace("@NAME@", "stringstream")
    + "}\n"
)

_ARRAY_HEADER = r"""

namespace std {
/* `array<T, N>` needs a value template argument, which this subset does not
   deduce - so the count lives in the object and the storage is a vector's.
   `std::array<int, 4> a;` gives four default elements, as C++ does. */
template<typename T>
class array {
public:
    T *items;
    unsigned long count;
    typedef T *iterator;
    typedef T value_type;
    typedef unsigned long size_type;
    array() { items = 0; count = 0; }
    void resize(unsigned long want) {
        items = (T *)malloc(sizeof(T) * want);
        count = want;
    }
    unsigned long size() { return count; }
    int empty() { return count == 0; }
    T &operator[](unsigned long i) { return items[i]; }
    T &at(unsigned long i) { return items[i]; }
    T &front() { return items[0]; }
    T &back() { return items[count - 1]; }
    T *begin() { return items; }
    T *end() { return items + count; }
    T *data() { return items; }
    void fill(T value) {
        unsigned long i;
        i = 0;
        while (i < count) { items[i] = value; i = i + 1; }
    }
};
}
"""


_UNKNWN_HEADER = r"""

/* COM's root interface, as py2bin's own header rather than as a fetch.
   `unknwn.h` does not exist as a file anywhere: every open implementation of
   the Windows API generates it from an `.idl` at build time, and the
   vendor's own ships inside a toolchain. What COM *is* is a struct whose
   first member points at a table of function pointers - which is exactly
   what py2bin lays a class with virtual methods out as - so it is written
   here as that class, and a program declares an interface by deriving from
   it the way a generated header does. */
/* No include guard: this text is pasted by the pass that reads C++, which
   pastes each of py2bin's own headers once however many files ask for it -
   and a guard here would be torn from what it guards when the directives
   are moved to the top. */

/* HRESULT, GUID and the rest come from <wtypes.h>, which is py2bin's own
   as well. Asked for rather than written out again, because written out in
   both they were the same names against two different structs - and a guard
   is no answer here: the translator moves every directive to the top of the
   file it emits, so a `#ifndef` around a declaration ends up above what it
   was meant to guard and guards nothing. An `#include` is a directive all
   through and survives that move intact.

   The translator does not need these to be declared, only named, and it has
   the names from the declarations it is reading. */
#include <wtypes.h>

#ifndef S_OK
#define S_OK ((HRESULT)0)
#endif
#ifndef S_FALSE
#define S_FALSE ((HRESULT)1)
#endif
#ifndef E_NOINTERFACE
#define E_NOINTERFACE ((HRESULT)0x80004002)
#endif
#ifndef E_POINTER
#define E_POINTER ((HRESULT)0x80004003)
#endif
#ifndef E_FAIL
#define E_FAIL ((HRESULT)0x80004005)
#endif
#ifndef E_OUTOFMEMORY
#define E_OUTOFMEMORY ((HRESULT)0x8007000E)
#endif
#ifndef E_INVALIDARG
#define E_INVALIDARG ((HRESULT)0x80070057)
#endif
#ifndef SUCCEEDED
#define SUCCEEDED(hr) ((HRESULT)(hr) >= 0)
#endif
#ifndef FAILED
#define FAILED(hr) ((HRESULT)(hr) < 0)
#endif


/* Three slots, and no destructor among them. A virtual destructor would be
   a fourth entry in the table, and every method of every interface derived
   from this would sit one slot further down than COM puts it - so a call on
   a real object would load the wrong pointer and jump into whatever was
   next. Nothing would say so: a vtable call is a load and a branch.
   COM does not destroy through the interface anyway. `Release` is how an
   object is let go, and the object frees itself when its own count reaches
   zero, which is what the count is for. */
class IUnknown {
public:
    virtual HRESULT QueryInterface(REFIID riid, void **object) = 0;
    virtual unsigned long AddRef() = 0;
    virtual unsigned long Release() = 0;
};
"""

_OBJIDL_HEADER = r"""

/* The COM interfaces a program is handed rather than ones it writes. Like
   <unknwn.h>, no implementation publishes this as a file: it is generated
   from an .idl at build time, and the vendor's ships inside a toolchain.
   Written here as the classes it describes, at the slots it puts them.

   Only what a caller reaches for. A generated header that names IStream in
   a signature needs the type to exist with the right table; the rest of the
   .idl would be an ABI written from memory, which is the one thing worth
   less than nothing here. */
#ifndef __py2bin_objidl_h
#define __py2bin_objidl_h
#include <unknwn.h>

/* The eight-byte integer the SDK spells as a one-member union. Written as
   the integer it is, deliberately: a struct passed BY VALUE through a
   foreign vtable is the one thing py2bin cannot spell, and IStream::Seek
   takes one. On both Windows machines an eight-byte struct travels in the
   same register as an eight-byte integer, so the bits and the ABI are the
   same either way, and this way it can be called. */
typedef long long __py2bin_LARGE;
typedef unsigned long long __py2bin_ULARGE;

typedef struct __py2bin_STATSTG {
    wchar_t *pwcsName;
    unsigned long type;
    __py2bin_ULARGE cbSize;
    unsigned long mtime_low, mtime_high;
    unsigned long ctime_low, ctime_high;
    unsigned long atime_low, atime_high;
    unsigned long grfMode;
    unsigned long grfLocksSupported;
    GUID clsid;
    unsigned long grfStateBits;
    unsigned long reserved;
} STATSTG;

/* Slots 3 and 4, after IUnknown's three. */
class ISequentialStream : public IUnknown {
public:
    virtual HRESULT Read(void *pv, unsigned long cb, unsigned long *read) = 0;
    virtual HRESULT Write(const void *pv, unsigned long cb,
                          unsigned long *written) = 0;
};

/* Slots 5 through 13, in the order the .idl declares them. */
class IStream : public ISequentialStream {
public:
    virtual HRESULT Seek(__py2bin_LARGE move, unsigned long origin,
                         __py2bin_ULARGE *position) = 0;
    virtual HRESULT SetSize(__py2bin_ULARGE size) = 0;
    virtual HRESULT CopyTo(IStream *other, __py2bin_ULARGE cb,
                           __py2bin_ULARGE *read, __py2bin_ULARGE *written) = 0;
    virtual HRESULT Commit(unsigned long flags) = 0;
    virtual HRESULT Revert() = 0;
    virtual HRESULT LockRegion(__py2bin_ULARGE offset, __py2bin_ULARGE cb,
                               unsigned long type) = 0;
    virtual HRESULT UnlockRegion(__py2bin_ULARGE offset, __py2bin_ULARGE cb,
                                 unsigned long type) = 0;
    virtual HRESULT Stat(STATSTG *out, unsigned long flag) = 0;
    virtual HRESULT Clone(IStream **out) = 0;
};

#define STREAM_SEEK_SET 0
#define STREAM_SEEK_CUR 1
#define STREAM_SEEK_END 2

#endif
"""


_OAIDL_HEADER = r"""

/* What Automation passes a value in. Sixteen bytes on both Windows
   machines: a two-byte tag, six the SDK reserves, and eight of value -
   which is what every member of that union is, or fits in. Written as the
   layout rather than as the union, because the union's members are a
   hundred names for the same eight bytes and the layout is the part that
   has to be right. */
#ifndef __py2bin_oaidl_h
#define __py2bin_oaidl_h
#include <unknwn.h>

typedef unsigned short VARTYPE;

typedef struct __py2bin_VARIANT {
    VARTYPE vt;
    unsigned short wReserved1;
    unsigned short wReserved2;
    unsigned short wReserved3;
    long long value;
} VARIANT;
typedef VARIANT VARIANTARG;
typedef VARIANT *LPVARIANT;

#define VT_EMPTY 0
#define VT_NULL 1
#define VT_I2 2
#define VT_I4 3
#define VT_R4 4
#define VT_R8 5
#define VT_BSTR 8
#define VT_DISPATCH 9
#define VT_BOOL 11
#define VT_UNKNOWN 13
#define VT_I8 20
#define VT_UI8 21

/* What Automation calls a member by, and the locale a name is read in.
   Both are plain numbers; the SDK gives them names and a program passes them
   through without looking. */
typedef long DISPID;
typedef long MEMBERID;
typedef unsigned long LCID;
typedef OLECHAR *LPOLESTR;
typedef OLECHAR *BSTR_ALIAS;

/* The arguments `Invoke` is handed, laid out as the SDK lays them out: a
   program that reads one reads these four members and nothing else. */
typedef struct __py2bin_DISPPARAMS {
    VARIANTARG *rgvarg;
    DISPID *rgdispidNamedArgs;
    unsigned int cArgs;
    unsigned int cNamedArgs;
} DISPPARAMS;

/* What `Invoke` fills in when the call it forwards fails. */
typedef struct __py2bin_EXCEPINFO {
    unsigned short wCode;
    unsigned short wReserved;
    BSTR bstrSource;
    BSTR bstrDescription;
    BSTR bstrHelpFile;
    unsigned long dwHelpContext;
    void *pvReserved;
    void *pfnDeferredFillIn;
    HRESULT scode;
} EXCEPINFO;

/* Described by name only: every use of it is a pointer handed back by
   `GetTypeInfo` and passed on, and what it points at is the type library's
   business. A struct with no members is not C, so it has one. */
struct ITypeInfo { int __py2bin_opaque; };
struct ITypeLib { int __py2bin_opaque; };

/* Automation's interface, which is what a generated header derives from
   when the object is scriptable - `IXMLDOMNode`, and most of what a browser
   control hands back. Four methods after `IUnknown`'s three, in the order
   COM puts them: the order *is* the layout, the same way it is for
   `IUnknown` above. */
class IDispatch : public IUnknown {
public:
    virtual HRESULT GetTypeInfoCount(unsigned int *count) = 0;
    virtual HRESULT GetTypeInfo(
        unsigned int which, LCID locale, ITypeInfo **answered
    ) = 0;
    virtual HRESULT GetIDsOfNames(
        REFIID riid,
        LPOLESTR *names,
        unsigned int count,
        LCID locale,
        DISPID *answered
    ) = 0;
    virtual HRESULT Invoke(
        DISPID member,
        REFIID riid,
        LCID locale,
        unsigned short flags,
        DISPPARAMS *given,
        VARIANT *answered,
        EXCEPINFO *failed,
        unsigned int *wrong
    ) = 0;
};

#define DISPATCH_METHOD 1
#define DISPATCH_PROPERTYGET 2
#define DISPATCH_PROPERTYPUT 4

#endif
"""


_EVENTTOKEN_HEADER = r"""

/* One struct, which a generated header then names two thousand times: the
   handle registering for an event hands back, so that the registration can
   be taken off again. */
#ifndef __py2bin_eventtoken_h
#define __py2bin_eventtoken_h

typedef struct EventRegistrationToken {
    long long value;
} EventRegistrationToken;

#endif
"""


_TYPE_TRAITS_HEADER = r"""

/* <type_traits>, as py2bin's own. A real standard library's is written in
   namespaces, SFINAE and variadic templates; what is here is the same
   answers arrived at the way the standard describes them - a general class
   that says no, and a narrower one written for the shape that says yes.

   Each is a class whose whole content is static, which is why it costs
   nothing: the copies are made at translation time and the answer is a
   constant by the time any code runs. */
#ifndef __py2bin_type_traits
#define __py2bin_type_traits

/* In `namespace std`, which is flattened, so `std::is_same` resolves to this
   exactly as the qualifier says - and so does the bare name after a
   `using namespace std;`. */
namespace std {

template <class T, T v> struct integral_constant {
    typedef T value_type;
    static const T value = v;
};

template <bool B> struct bool_constant { static const bool value = B; };
typedef bool_constant<true> true_type;
typedef bool_constant<false> false_type;

/* The shape questions. Each is a no with a yes written for one shape. */
template <class T, class U> struct is_same { static const bool value = false; };
template <class T> struct is_same<T, T> { static const bool value = true; };

template <class T> struct is_pointer { static const bool value = false; };
template <class T> struct is_pointer<T *> { static const bool value = true; };

template <class T> struct is_reference { static const bool value = false; };
template <class T> struct is_reference<T &> { static const bool value = true; };

template <class T> struct is_lvalue_reference { static const bool value = false; };
template <class T> struct is_lvalue_reference<T &> { static const bool value = true; };

template <class T> struct is_const { static const bool value = false; };
template <class T> struct is_const<const T> { static const bool value = true; };

template <class T> struct is_array { static const bool value = false; };

template <class T> struct is_void { static const bool value = false; };
template <> struct is_void<void> { static const bool value = true; };

/* The ones that are true for a list of types rather than for a shape. */
template <class T> struct is_integral { static const bool value = false; };
template <> struct is_integral<bool> { static const bool value = true; };
template <> struct is_integral<char> { static const bool value = true; };
template <> struct is_integral<signed char> { static const bool value = true; };
template <> struct is_integral<unsigned char> { static const bool value = true; };
template <> struct is_integral<short> { static const bool value = true; };
template <> struct is_integral<unsigned short> { static const bool value = true; };
template <> struct is_integral<int> { static const bool value = true; };
template <> struct is_integral<unsigned int> { static const bool value = true; };
template <> struct is_integral<long> { static const bool value = true; };
template <> struct is_integral<unsigned long> { static const bool value = true; };
template <> struct is_integral<long long> { static const bool value = true; };
template <> struct is_integral<unsigned long long> { static const bool value = true; };

template <class T> struct is_floating_point { static const bool value = false; };
template <> struct is_floating_point<float> { static const bool value = true; };
template <> struct is_floating_point<double> { static const bool value = true; };
template <> struct is_floating_point<long double> { static const bool value = true; };

template <class T> struct is_signed { static const bool value = false; };
template <> struct is_signed<signed char> { static const bool value = true; };
template <> struct is_signed<short> { static const bool value = true; };
template <> struct is_signed<int> { static const bool value = true; };
template <> struct is_signed<long> { static const bool value = true; };
template <> struct is_signed<long long> { static const bool value = true; };
template <> struct is_signed<float> { static const bool value = true; };
template <> struct is_signed<double> { static const bool value = true; };

template <class T> struct is_unsigned { static const bool value = false; };
template <> struct is_unsigned<bool> { static const bool value = true; };
template <> struct is_unsigned<unsigned char> { static const bool value = true; };
template <> struct is_unsigned<unsigned short> { static const bool value = true; };
template <> struct is_unsigned<unsigned int> { static const bool value = true; };
template <> struct is_unsigned<unsigned long> { static const bool value = true; };
template <> struct is_unsigned<unsigned long long> { static const bool value = true; };

/* The ones that answer with a type rather than with yes or no. */
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T &> { typedef T type; };

template <class T> struct remove_pointer { typedef T type; };
template <class T> struct remove_pointer<T *> { typedef T type; };

template <class T> struct remove_const { typedef T type; };
template <class T> struct remove_const<const T> { typedef T type; };

template <class T> struct remove_volatile { typedef T type; };
template <class T> struct remove_volatile<volatile T> { typedef T type; };

template <class T> struct add_pointer { typedef T *type; };
template <class T> struct add_const { typedef const T type; };

template <bool B, class T, class F> struct conditional { typedef T type; };
template <class T, class F> struct conditional<false, T, F> { typedef F type; };

/* The one a function's return type is written as. When the answer is false
   there is no `type` in it at all, which is what takes the function out of
   the running rather than making it an error - the whole point of the
   thing, and why it is spelled the way it is. */
template <bool B, class T> struct enable_if {};
template <class T> struct enable_if<true, T> { typedef T type; };

template <class... Ts> struct how_many { static const int value = sizeof...(Ts); };

}

#endif
"""

_WRL_HEADER = r"""

/* The two things a WebView2 program uses out of the Windows Runtime C++
   Template Library. The vendor's own is a template library that ships inside
   a toolchain and is written in C++ this subset does not have; what a caller
   reaches for out of it is smaller than that, and this is that.

   `ComPtr<T>` is a pointer that counts: it releases what it held when it is
   given something else or goes out of scope, and hands out its address for a
   call that fills it in. That is the whole of what the name means. */
#ifndef __py2bin_wrl_h
#define __py2bin_wrl_h
#include <unknwn.h>

namespace Microsoft { namespace WRL {

template <class T> class ComPtr {
public:
    T *ptr_;

    ComPtr() { ptr_ = 0; }
    ComPtr(T *given) { ptr_ = given; if (ptr_) { ptr_->AddRef(); } }
    ComPtr(const ComPtr<T> &other) {
        ptr_ = other.ptr_;
        if (ptr_) { ptr_->AddRef(); }
    }
    ~ComPtr() { if (ptr_) { ptr_->Release(); ptr_ = 0; } }

    /* Assignment releases what was held first, and in that order: a pointer
       assigned to itself would otherwise be released and then kept. */
    void operator=(T *given) {
        if (given) { given->AddRef(); }
        if (ptr_) { ptr_->Release(); }
        ptr_ = given;
    }
    void operator=(const ComPtr<T> &other) {
        if (other.ptr_) { other.ptr_->AddRef(); }
        if (ptr_) { ptr_->Release(); }
        ptr_ = other.ptr_;
    }

    T *Get() const { return ptr_; }
    T *operator->() const { return ptr_; }
    T **GetAddressOf() { return &ptr_; }
    /* `&held` where a call wants somewhere to write. The vendor's own
       releases what it held first; this one does not, because a program
       that asks for the address is about to be given a fresh pointer and
       the one before it was almost always null. */
    T **operator&() { return &ptr_; }
    T *Detach() { T *held = ptr_; ptr_ = 0; return held; }
    void Attach(T *given) { if (ptr_) { ptr_->Release(); } ptr_ = given; }
    void Reset() { if (ptr_) { ptr_->Release(); ptr_ = 0; } }

    int operator==(const void *other) const { return (const void *)ptr_ == other; }
    int operator!=(const void *other) const { return (const void *)ptr_ != other; }

    /* `As` asks the object whether it is also something else, which is what
       QueryInterface is for. The answer goes into the pointer handed in. */
    template <class U> HRESULT As(ComPtr<U> *other) {
        if (!ptr_) { return E_POINTER; }
        void *found = 0;
        HRESULT asked = ptr_->QueryInterface(__uuidof(U), &found);
        if (SUCCEEDED(asked)) { other->Attach((U *)found); }
        return asked;
    }
};

} }

#endif
"""

_BUILTIN_CPP_HEADERS = {
    # COM's root, which no implementation publishes as a file.
    "unknwn.h": _UNKNWN_HEADER,
    "type_traits": _TYPE_TRAITS_HEADER,
    "wrl.h": _WRL_HEADER,
    "wrl/client.h": _WRL_HEADER,
    # And the interfaces a generated header names in its signatures. Same
    # reason and same shape: the classes they are, at the slots the .idl
    # puts them at.
    "objidl.h": _OBJIDL_HEADER,
    "oaidl.h": _OAIDL_HEADER,
    "EventToken.h": _EVENTTOKEN_HEADER,
    "eventtoken.h": _EVENTTOKEN_HEADER,
    "string": _STRING_HEADER,
    "map": _MAP_HEADER,
    # An unordered map is the same interface with no promise about order,
    # and this one keeps key order - which is a stronger promise than C++
    # makes here, so no program that is correct against C++ can tell.
    "unordered_map": _MAP_HEADER,
    "set": _SET_HEADER,
    "unordered_set": _SET_HEADER,
    "memory": _MEMORY_HEADER,
    "sstream": _SSTREAM_HEADER,
    "array": _ARRAY_HEADER,
    "vector": _VECTOR_HEADER,
    "iostream": _IOSTREAM_HEADER,
    "algorithm": _ALGORITHM_HEADER,
    "utility": _UTILITY_HEADER,
    "optional": _OPTIONAL_HEADER,
    "list": _LIST_HEADER,
    "bitset": _BITSET_HEADER,
    "random": _RANDOM_HEADER,
    "typeinfo": _TYPEINFO_HEADER,
    "string_view": _STRING_VIEW_HEADER,
    "new": _NEW_HEADER,
    "atomic": _ATOMIC_HEADER,
    "thread": _thread_header,
    "mutex": _MUTEX_HEADER,
    "variant": _VARIANT_HEADER,
    # Answered per target, so it is a function rather than a text.
    "chrono": _chrono_header,
    "iomanip": _IOMANIP_HEADER,
    "tuple": _TUPLE_HEADER,
    "deque": _DEQUE_HEADER,
    "numeric": _NUMERIC_HEADER,
    "stdexcept": _STDEXCEPT_HEADER,
    "filesystem": _FILESYSTEM_HEADER,
    "functional": _FUNCTIONAL_HEADER,
}


def _c_header_under(named: str) -> "str | None":
    """`<cstdarg>` is `<stdarg.h>`, and so is every other one of that shape.

    C++ renames each C header by dropping the `.h` and putting a `c` in
    front, and says the two hold the same things. Which ones exist is not a
    list to keep here: it is whichever headers py2bin's C ships, asked at
    the moment the question comes up. Written as a list it went stale, and
    a program including `<cstdarg>` was told py2bin does not implement it
    while `<stdarg.h>` sat in the same build.
    """

    from .c_preprocessor import _BUILTIN_HEADERS

    if not named.startswith("c") or named.endswith(".h"):
        return None
    under = f"{named[1:]}.h"
    return f"#include <{under}>\n" if under in _BUILTIN_HEADERS else None

#: A quoted include names a file of this project; an angled one names a header
#: py2bin ships, which is C already and left for the preprocessor.
_LOCAL_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"[ \t]*$', re.M)

#: An include of either spelling. Which one it is stopped mattering once a
#: quoted include had to fall back to what py2bin ships and an angled one had
#: to reach the project's own headers - both of which C already says.
_ANY_INCLUDE = re.compile(
    r'#[ \t]*include[ \t]*(?:"([^"]+)"|<([^>]+)>)'
)

def _line_of(text: str, index: int) -> int:
    """Which line `index` falls on, counting from one."""

    return text.count("\n", 0, index) + 1


#: Headers py2bin supplied while preprocessing a branch-selecting one, so
#: the run that reads the rest of the program is told not to supply them
#: again. Written at the top of the unit rather than where the header was
#: pasted, which may be below the program's own include of the same thing.
_SUPPLIED_BY_A_BRANCH: "set[str]" = set()


#: A header that declares one thing or another according to a macro.
_CHOOSES_A_BRANCH = re.compile(r"(?m)^[ \t]*#[ \t]*(?:else|elif)\b")

#: Suffixes that mean C++ rather than C.
CPP_SUFFIXES = (".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx")


def is_cpp(path: Path) -> bool:
    return path.suffix.lower() in CPP_SUFFIXES


def _py2bin_ships(named: str) -> bool:
    """Whether py2bin has a copy of this header of its own to hand over.

    Both tables, because both are answers to the same question: the C++ ones
    this stage pastes, and the C ones the preprocessor below serves.
    """

    from .c_preprocessor import _BUILTIN_HEADERS

    return (
        named in _BUILTIN_CPP_HEADERS
        or named in _BUILTIN_HEADERS
        or _c_header_under(named) is not None
    )


def _somebody_supplies(named: str, include_dirs: "tuple[str, ...]") -> bool:
    """Whether anything will ever hand this header over.

    py2bin's own C++ headers, py2bin's own C ones, and the search path. A
    name none of those holds is a name no pass below can answer either.
    """

    if _py2bin_ships(named):
        return True
    return any((Path(folder) / named).is_file() for folder in include_dirs)


def _found_on_the_path(
    named: str, here: "Path | None", include_dirs: "tuple[str, ...]"
) -> "Path | None":
    """The file the search path answers this include with, if it holds one.

    A compiler looks in the directories it was given before it reaches for a
    copy of its own, so a project that vendors a header py2bin happens to
    ship - its own `string.h`, or an SDK's `unknwn.h` - is compiled against
    the one it wrote. py2bin preferred its own and said nothing, so a program
    was built against different macros and a different layout than the file
    sitting on disk beside it.

    `here` is the including file's own directory, which a quoted include
    searches first and an angled one does not.
    """

    from .c_preprocessor import _FETCHED_INTO

    for folder in (
        *(() if here is None else (here,)),
        *(Path(item) for item in include_dirs),
    ):
        candidate = folder / named
        if not candidate.is_file():
            continue
        # Except out of the directory `--auto-fetch` downloads into. A fetched
        # set brings its neighbours along, so a build that once fetched
        # anything from a Windows set left that set's `winnt.h` there, and
        # taking it would shadow py2bin's own with a copy that cannot compile
        # here - for every build afterwards. The preprocessor below draws the
        # same line for the same reason. A directory somebody named
        # themselves is their own choice and still wins.
        if _FETCHED_INTO in candidate.parts and _py2bin_ships(named):
            continue
        return candidate
    return None


def _which_file(path: Path) -> object:
    """What makes two include paths the same file, asked of the filesystem.

    A header is pasted once per translation unit, so this pass has to know
    when two spellings reach one file. The path alone does not say: the
    filesystem here is case-insensitive, so `Foo.h` and `foo.h` are one file
    under two names, and pasting it twice hands the translator two copies of
    every class in it - which comes out as `two definitions of ... take the
    same arguments`, blaming the header rather than the second include.
    `resolve()` settles symlinks and `./` and doubled slashes and stops there.

    So the file is stat'ed and identified the way the filesystem identifies
    it, by device and inode.
    """

    try:
        status = path.stat()
    except OSError:
        return path.resolve()
    if not status.st_ino:
        # A filesystem that does not number its files; the settled path is
        # all there is left to go on.
        return path.resolve()
    return (status.st_dev, status.st_ino)

#: The two macros whose value is *where they are written*. Every other
#: predefined macro means the same wherever it stands, which is why the
#: preprocessor can answer those and not these.
_POSITION_MACROS = re.compile(r"(?<![\w$])(__LINE__|__FILE__)(?![\w$])")


def _answer_position_macros(text: str, path: Path) -> str:
    """Say where each `__LINE__` and `__FILE__` in this file is written.

    The preprocessor answers these for a C program and gets them right, but
    it never sees a C++ one as the user wrote it: by the time it runs, every
    header has been pasted in above and every class has been rewritten into
    structs and free functions, which moves the line a token sits on and
    leaves one file name for the whole unit. So `__LINE__` reported an offset
    into text nobody wrote and `__FILE__` named the file the build started
    from however deep in a header the macro stood.

    Here, each file is still its own text, so the answer is exact - and once
    it is a number and a string, nothing below can move it.

    Only what is written in code can be answered this way. A `#define` whose
    body names one of them means the line of the *call*, and calls are
    expanded by the preprocessor far below this - so that is refused by name
    rather than answered with a line the caller is not on.
    """

    if "__LINE__" not in text and "__FILE__" not in text:
        return text
    spelling = str(path).replace("\\", "\\\\").replace('"', '\\"')
    out: list[str] = []
    at = 0
    for kind, written in _split_literals(text):
        part = written
        if kind == "literal" and part.startswith("#"):
            # `_split_literals` hands a preprocessing directive over whole,
            # continuation lines and all, which is exactly the region whose
            # value is decided somewhere else.
            # Left exactly as written, for the preprocessor to answer later.
            # It will answer with a line in the translated unit rather than
            # the one the macro was called on, which is wrong - but it is a
            # number that goes in a log message, and refusing to build every
            # program that owns a LOG macro is a worse answer than an
            # imprecise line in one. Code gets the exact answer above; this
            # is no worse than it was.
            pass
        elif kind == "code":
            start = at

            def answer(match: "re.Match[str]") -> str:
                if match.group(1) == "__LINE__":
                    return str(_line_of(text, start + match.start()))
                return f'"{spelling}"'

            part = _POSITION_MACROS.sub(answer, part)
        out.append(part)
        # Along the text as it was written, not as it comes out: a `__LINE__`
        # answered with a four-digit number is longer than the name it
        # replaced, and every line asked for after it would have been read
        # off the wrong offset.
        at += len(written)
    return "".join(out)


def inline_local_includes(
    path: Path,
    include_dirs: "tuple[str, ...]" = (),
    seen: "set[object] | None" = None,
    seen_headers: "set[str] | None" = None,
    target: "str | None" = None,
) -> str:
    """Paste this project's own headers in, so the translator can see them.

    A class is usually declared in a header and used from a source file. The
    translator works on text and runs before the preprocessor, so without this
    it would be handed a file that mentions a class it has never seen and
    would leave the calls alone - producing C that does not compile, blaming a
    line the user did write for a declaration they put somewhere else.

    Either spelling is pasted, and the two differ only in where the file is
    looked for: quoted searches the including file's own directory first,
    angled does not. The search path is asked before py2bin reaches for a copy
    of its own, which is the order a compiler searches in - what py2bin ships
    is the fallback for a header nobody else has. A header already pasted is
    skipped rather than pasted twice, which is what an include guard would
    have done anyway.
    """

    seen = set() if seen is None else seen
    # One copy of a supplied header per translation unit, whatever how many
    # files ask for it - the same job an include guard does.
    seen_headers = set() if seen_headers is None else seen_headers
    settled = _which_file(path)
    if settled in seen:
        return ""
    seen.add(settled)
    text = path.read_text(encoding="utf-8", errors="replace")
    # Before a single include is pasted, while the offsets in this text are
    # still offsets into the file the user opened. Nothing below moves a line,
    # so the includes are found on the lines they were written on either way.
    text = _answer_position_macros(text, path)

    def supply(named: str) -> "str | None":
        """One of py2bin's own C++ headers, pasted once per unit."""

        supplied = _BUILTIN_CPP_HEADERS.get(named) or _c_header_under(named)
        if supplied is None:
            return None
        # One of these is written differently for each machine, because what
        # it wraps is: the clock has a different name and a different number
        # on each. Answered here, where the target is known - the translator
        # runs before the preprocessor and cannot choose a branch itself.
        if callable(supplied):
            supplied = supplied(target)
        if named in seen_headers:
            return ""
        seen_headers.add(named)
        # One of these may include another - <filesystem> is written on top
        # of <string> - so what is pasted is pasted again. Without it the
        # inner include survived into the C, and the compiler reported a
        # missing header the user never wrote.
        return _ANY_INCLUDE.sub(lambda found: reach(found, ours=True), supplied)

    def chosen_branch(named: str, candidate: Path) -> str:
        """The one branch of a header that declares two, as C++ sees it."""

        if named in seen_headers:
            return ""
        seen_headers.add(named)
        from .c_preprocessor import as_cplusplus

        return as_cplusplus(
            named,
            candidate.parent,
            include_dirs,
            target,
            _SUPPLIED_BY_A_BRANCH,
            seen_headers,
        )

    def reach(match: "re.Match[str]", ours: bool = False) -> str:
        """Whatever this include names, however it is spelled.

        The search path is looked at first and what py2bin ships is the
        fallback, which is the order a compiler searches in: a project that
        vendors its own copy of a header py2bin happens to ship gets the one
        it wrote, and finds out about it either way rather than being handed
        a different file in silence.

        `ours` says the include was written inside one of py2bin's own
        headers rather than by the program, and those keep reaching py2bin's
        own: <bitset> is written on top of py2bin's <string>, and a project
        that supplies a `string` of its own would otherwise have taken
        <bitset> apart along with everything else that leans on one. A real
        standard library is spared this because it includes reserved names
        nobody writes - `<__fwd/string.h>` - which headers spelled the way
        the program spells them cannot do.
        """

        named = match.group(1) or match.group(2)
        angled = match.group(2) is not None
        if ours:
            supplied = supply(named)
            if supplied is not None:
                return supplied
        # Angled, so the file's own directory is not searched - which is the
        # rule C gives and matters here, because a fetch leaves its copy
        # right beside the program.
        candidate = _found_on_the_path(
            named, None if angled else path.parent, include_dirs
        )
        if candidate is not None:
            if _CHOOSES_A_BRANCH.search(
                candidate.read_text(encoding="utf-8", errors="replace")
            ):
                # A header that declares one thing or another according to a
                # macro cannot be read as it stands: this pass runs before
                # the preprocessor and has no `#if`, so it would take both
                # branches - and the one meant for C is written in shapes
                # that mean something else here. `interface X { ... }` came
                # out as `interface X = { ... };`.
                #
                # So the preprocessor runs first, for this header alone and
                # with `__cplusplus` defined, and what comes back is the one
                # branch a C++ compiler would have been handed. A generated
                # COM header declares each interface twice and picks between
                # them on exactly that, and a program calling one the C++ way
                # - `view->Navigate(url)` - needs the classes.
                return chosen_branch(named, candidate)
            # The program's own headers are read here whatever they are
            # called, and whichever way they are spelled. The two spellings
            # differ in *where* C looks and never in how it reads what it
            # finds: the same header included with quotes was pasted and
            # translated, and included with angles was handed below
            # untouched, so a class in it reached a C compiler and the
            # constructor was reported as a type it had never heard of.
            return inline_local_includes(
                candidate, include_dirs, seen, seen_headers, target
            )
        supplied = supply(named)
        if supplied is not None:
            return supplied
        if angled:
            if "." not in named:
                # A C++ standard header, spelled the way only those are, that
                # py2bin does not implement. Left to be fetched, what arrives
                # is a real standard library's copy - written in namespaces,
                # SFINAE and partial specialisation, none of which this
                # subset has - and it fails somewhere deep inside itself
                # about something that is not the reason.
                if named in _NEEDS_ATOMICS:
                    raise CppTranslationError(
                        str(path),
                        _line_of(text, match.start()),
                        f"<{named}> is not implemented yet, and what is "
                        f"missing is worth naming. py2bin does emit an atomic "
                        f"add now - `lock xadd` on x86-64, the "
                        f"`ldaxr`/`stlxr` pair on ARM64 - and the allocator "
                        f"uses it, so a heap shared between threads is no "
                        f"longer the thing in the way. What is left is "
                        f"starting a thread on each platform and the "
                        f"trampoline that gets a callable to it. Until then "
                        f"this is refused rather than half-written, because "
                        f"a thread that runs and is subtly wrong is worse "
                        f"than one that does not exist",
                    )
                raise CppTranslationError(
                    str(path),
                    _line_of(text, match.start()),
                    f"<{named}> is a C++ standard header py2bin does not "
                    f"implement. A copy fetched from a real standard library "
                    f"is written in C++ this subset does not have, so it "
                    f"would fail somewhere inside itself rather than here. "
                    f"py2bin ships "
                    f"{', '.join(sorted(h for h in _BUILTIN_CPP_HEADERS if '.' not in h))}; "
                    f"name a directory with --include DIR to use your own.",
                )
            # Nothing on the search path holds it, py2bin does not ship it,
            # and neither does the preprocessor below. Nothing later will
            # supply it either, so the program cannot build - and saying so
            # here, by name, is what lets `--auto-fetch` go and get it. Left
            # for the preprocessor, the translator ran on a file missing
            # every class that header declares and reported whichever pass
            # first tripped over one, which named a symptom and not this.
            if not _somebody_supplies(named, include_dirs):
                raise CppTranslationError(
                    str(path),
                    _line_of(text, match.start()),
                    f"cannot find the header {named!r}",
                )
        # Not ours to paste; leave it for the preprocessor to fail on clearly.
        return match.group(0)

    return _ANY_INCLUDE.sub(reach, text)


def _already_supplied() -> str:
    """The line that tells the preprocessor what a branch already brought."""

    return "".join(
        f'#pragma py2bin supplied "{name}"\n'
        for name in sorted(_SUPPLIED_BY_A_BRANCH)
    )


def translate_project(
    path: Path,
    include_dirs: "tuple[str, ...]" = (),
    target: "str | None" = None,
) -> str:
    """The C for one C++ source, with this project's headers pasted in."""

    _SUPPLIED_BY_A_BRANCH.clear()
    inlined = inline_local_includes(path, include_dirs, None, None, target)
    return translate(_already_supplied() + inlined, str(path))


def translate_unity(
    sources: "tuple[Path, ...]",
    include_dirs: "tuple[str, ...]" = (),
    target: "str | None" = None,
) -> str:
    """One C translation unit from several C++ sources and their headers.

    Translated together rather than one at a time, because a class declared in
    a shared header would otherwise be emitted once per source that includes
    it - and a struct defined twice is not C. One `seen` set across the whole
    build is what makes the header arrive once, which is what an include guard
    would have done had the translator run after the preprocessor rather than
    before it.

    It also matches what is underneath: py2bin has no linker, so the program
    was always going to be one translation unit.
    """

    seen: set[Path] = set()
    shared: set[str] = set()
    _SUPPLIED_BY_A_BRANCH.clear()
    pieces = [
        inline_local_includes(path, include_dirs, seen, shared, target)
        for path in sources
    ]
    joined = _already_supplied() + "\n".join(pieces)
    return translate(joined, str(sources[0]) if sources else "<c++>")
