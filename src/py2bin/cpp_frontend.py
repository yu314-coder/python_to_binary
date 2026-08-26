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
_OPERATOR_NAMES = {
    "+": "op_add", "-": "op_sub", "*": "op_mul", "/": "op_div", "%": "op_mod",
    "==": "op_eq", "!=": "op_ne", "<": "op_lt", ">": "op_gt",
    "<=": "op_le", ">=": "op_ge", "[]": "op_index", "()": "op_call",
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
#: `c++` rather than `++c`. Spelled `operator++(int)`, whose parameter is
#: never given a value and exists only to be different.
_POSTFIX = {"op_inc": "op_inc_post", "op_dec": "op_dec_post"}

#: Longest first, so `<=` is not read as `<`. `->` is not among them: it takes
#: no right operand of its own, so the two-operand pass has nothing to match.
_OPERATOR_SYMBOLS = [
    symbol
    for symbol in sorted(_OPERATOR_NAMES, key=len, reverse=True)
    if symbol not in ("->", "!", "++", "--")
]


@dataclass
class Member:
    name: str
    ctype: str
    array: str = ""   # "[8]" for `int items[8]`, kept for the struct field


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


@dataclass
class Class:
    name: str
    base: str | None = None
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
            out.append(change("".join(chunk)))
            chunk = []
            quote = char
            literal = [char]
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
    raise ValueError("unbalanced braces")


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


#: `final` after the name says nothing may derive from it, which C++ checks
#: and C has no way to state - so it is read and dropped, the way `override`
#: is on a member.
_CLASS_HEAD = re.compile(
    r"\b(class|struct)\s+([A-Za-z_]\w*)\s*(?:final\s*)?"
    r"(?::\s*(?:public|private|protected)?\s*(?:virtual\s+)?"
    r"([A-Za-z_]\w*)\s*)?\{"
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

def _split_members(body: str, name: str, filename: str, at: int) -> Class:
    """Read a class body into its data members and its member functions."""

    found = Class(name)
    index = 0
    while index < len(body):
        char = body[index]
        if char in " \t\n;":
            index += 1
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
    words = declaration.replace("*", " * ").split()
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
    return Member(name=spelled, ctype=" ".join(words[:-1]), array=array)


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

    def decorated(method: Method) -> Method:
        method.virtual = virtual
        method.pure = pure
        method.shared = shared
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
            raise CppTranslationError(
                filename, at_line,
                f"py2bin's C++ subset does not know operator{symbol!r}; it "
                f"knows {', '.join(sorted(_OPERATOR_NAMES))}",
            )
        # `*x` and `x * y` are written the same and are not the same member.
        # The parameter list is what says which: a unary operator takes none.
        named = _OPERATOR_NAMES[symbol]
        if not parameters.strip():
            named = {"*": _DEREFERENCE, "-": _NEGATE}.get(symbol, named)
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


def _template_parameters(spelled: str) -> "list[tuple[str, bool]]":
    """Each parameter as (name, is_a_type)."""

    found: list[tuple[str, bool]] = []
    for part in _split_arguments(spelled):
        words = part.strip().split()
        if not words:
            continue
        is_a_type = words[0] in ("typename", "class")
        found.append((words[-1], is_a_type))
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


def _instantiated_name(name: str, arguments: "list[str]") -> str:
    """`Stack<int *>` becomes `Stack__int_p`, which is a C identifier."""

    spelled = []
    for argument in arguments:
        cleaned = argument.strip().replace("*", " p").replace("&", " r")
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", cleaned).strip("_")
        spelled.append(cleaned or "x")
    return f"{name}__" + "_".join(spelled)


def _substituted(body: str, names: "list[str]", arguments: "list[str]") -> str:
    """Replace each template parameter with the argument given for it."""

    for parameter, argument in zip(names, arguments):
        body = _map_code(
            body,
            lambda part, p=parameter, a=argument.strip(): re.sub(
                rf"\b{re.escape(p)}\b", a, part
            ),
        )
    return body


#: What a literal is. `1` is an int, `1.0` a double, `"s"` a `const char *`.
_LITERAL_TYPES = (
    (re.compile(r"^[+-]?\d+[uUlL]*$"), "int"),
    (re.compile(r"^[+-]?0[xX][0-9a-fA-F]+[uUlL]*$"), "int"),
    (re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?[fF]?$"), "double"),
    (re.compile(r'^".*"$', re.S), "const char *"),
    (re.compile(r"^'.*'$", re.S), "char"),
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
    pattern = re.compile(
        rf"\b((?:const\s+)?[A-Za-z_]\w*)\s*(\*?)\s*\b{re.escape(spelled)}\b\s*([=;,)\[])"
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
        return _declared_here(text).get(spelled)
    # The declaration nearest above the call, which is the one C++ would have
    # in scope; falling back to the first anywhere when the call comes first.
    earlier = [match for match in found if before < 0 or match.start() < before]
    declared = earlier[-1] if earlier else found[0]
    stars = declared.group(2) + ("*" if declared.group(3) == "[" else "")
    return (declared.group(1) + " " + stars).strip()



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


def _member_result(text: str, owner: str, spelled: str) -> "str | None":
    """What the member matching `spelled` on that class is declared to answer."""

    for head in _CLASS_HEAD.finditer(text):
        if head.group(2) != owner:
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            return None
        inside = text[head.end() - 1: closing]
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
    # Arithmetic: `raw + 8` is where `raw` points moved along, and `x * 2.0`
    # is whichever of the two is wider - which is what C++ does with them.
    binary = re.match(r"^(.+?)\s*[-+*/%]\s*([^-+*/%]+)$", spelled)
    if binary is not None:
        left = _deduced_type(binary.group(1).strip(), text, before)
        right = _deduced_type(binary.group(2).strip(), text, before)
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
#: `a[i]`, whatever `a` is.
_INDEXED = re.compile(r"^(.+)\[[^\]]*\]$", re.S)

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
    plain = _PLAIN_CALL.match(spelled)
    if plain is not None:
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
        return " ".join(words[:-1]).replace("&", "*").strip()
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
    if len(wanted) != len(given):
        return None
    found: dict[str, str] = {}
    names = {name for name, is_type in parameters if is_type}
    for part, argument in zip(wanted, given):
        stars = part.count("*")
        words = part.replace("*", " * ").replace("&", " ").split()
        if len(words) < 2:
            continue
        named = words[0] if words[0] not in ("const",) else (words[1] if len(words) > 2 else "")
        if named not in names:
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
    if len(found) != len([1 for _n, is_type in parameters if is_type]):
        return None
    return [found[name] for name, _is_type in parameters]


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
_LAMBDA = re.compile(
    r"\[([^\]\[]*)\]\s*\(([^()]*)\)\s*(?:mutable\s*)?(?:->\s*([^{;]+?))?\s*\{"
)


#: `friend class X;` or `friend int peek(X &);` - an access grant.
_FRIEND = re.compile(r"\bfriend\b[^;{}]*;")

#: `class B;` - a forward declaration. The typedefs this emits already name
#: every class before any body, so there is nothing left for one to do.
_FORWARD_DECLARATION = re.compile(
    r"(?m)^[ \t]*(?:class|struct|union)\s+[A-Za-z_]\w*\s*;[ \t]*$"
)

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



def _hoist_plain_structs(text: str, plain: "list[str]") -> "tuple[str, list[str]]":
    """Take each plain struct's body out, to be emitted above the classes.

    A `struct` with no methods is C already and is left exactly as written -
    but a class holding one is emitted above whatever is left of the file, so
    the class named a type C had not seen. The bodies come up in the order
    they were written, which keeps one holding another after it.
    """

    if not plain:
        return text, []
    wanted = set(plain)
    bodies: "list[str]" = []
    out: list[str] = []
    at = 0
    for head in _CLASS_HEAD.finditer(text):
        if head.start() < at or head.group(2) not in wanted:
            continue
        try:
            closing = _matching(text, head.end() - 1)
        except ValueError:
            continue
        end = closing
        while end < len(text) and text[end] in " \t":
            end += 1
        if end < len(text) and text[end] == ";":
            end += 1
        bodies.append(text[head.start():end])
        out.append(text[at:head.start()])
        at = end
    out.append(text[at:])
    return "".join(out), bodies

def _hoist_tagged_types(text: str) -> "tuple[str, str]":
    """Take every top-level `enum`/`union` definition out, with its typedef.

    Returns what is left and what was taken. They are put back above the
    struct definitions, because a struct holding one needs the complete type
    and C reads a file top to bottom.
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
        end = closing
        while end < len(text) and text[end] in " \t":
            end += 1
        if end < len(text) and text[end] == ";":
            end += 1
        # The typedef this file's own spelling pass wrote just after it.
        following = re.match(
            r"\s*typedef\s+(?:enum|union)\s+\w+\s+\w+\s*;", text[end:]
        )
        if following:
            end += following.end()
        taken.append(text[match.start():end])
        out.append(text[at:match.start()])
        at = end
    out.append(text[at:])
    return "".join(out), "\n".join(taken)


def _tag_typedef(name: str, text: str) -> str:
    """`typedef enum Colour Colour;` - whichever tag the name was declared with."""

    kind = "union" if re.search(rf"\bunion\s+{re.escape(name)}\s*\{{", text) else "enum"
    return f"typedef {kind} {name} {name};"



#: `using Number = int;` and `using Fn = int (*)(int);` - a typedef with the
#: name in front. `using namespace x;` and `using B::f;` have no `=` and are
#: not this.
_ALIAS = re.compile(r"(?<![.\w>])using\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")

def _rewrite_cpp_spellings(text: str) -> "tuple[str, set[str]]":
    """The C++ that is a different spelling of C, spelled the C way.

    Returns the text and the tag names that need a typedef, since C++ lets a
    bare `Colour` or `U` name a type and C wants `enum Colour` or `union U`.
    """

    # `using Number = int;` is a typedef written the other way round, which
    # is the only way C++11 and later spell one in new code.
    text = _map_code(text, lambda part: _ALIAS.sub(r"typedef \2 \1;", part))
    text = _FORWARD_DECLARATION.sub("", text)
    # `friend class X;` and `friend int f();` grant access to what is private.
    # py2bin emits a plain struct and enforces no access at all, so a friend
    # declaration asks for something already given and has nothing to become.
    text = _map_code(text, lambda part: _FRIEND.sub("", part))
    # Keywords in C++, and in C either a macro from a header the program did
    # not include or nothing at all. Spelled out here so a program need not
    # remember which.
    text = _map_code(
        text,
        lambda part: re.sub(r"\bnullptr\b", "0", re.sub(r"\bbool\b", "int", part)),
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
        while end < len(text) and text[end] in " \t":
            end += 1
        if end < len(text) and text[end] == ";":
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

#: `C(int x, int y) : a(x), b(y) {` - a constructor's initialiser list.
_INITIALISER_LIST = re.compile(
    r"\)\s*:\s*((?:[A-Za-z_]\w*\s*\([^()]*\)\s*,?\s*)+)\{"
)

#: `a(x)` inside one.
_ONE_INITIALISER = re.compile(r"([A-Za-z_]\w*)\s*\(([^()]*)\)")


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
        head.group(2): head.group(3)
        for head in _CLASS_HEAD.finditer(text)
        if head.group(3)
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
            name, value = found.group(1), found.group(2).strip()
            if name == base:
                assignments.append(f"{_BASE_INIT}({value});")
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

    while True:
        found = re.search(r"= \(?([^()\[]*)\)?\[(__py2bin_each_\d+)\];", text)
        if found is None or "\x00closed" in text[found.end(): found.end() + 8]:
            break
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
_BRACE_INIT = re.compile(
    r"(?<![.\w>])([A-Za-z_]\w*)\s+(\*?)\s*([A-Za-z_]\w*)\s*\{([^{}]*)\}\s*;"
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

    def one(match: "re.Match[str]") -> str:
        held, star, name, inside = match.groups()
        if held in _NOT_A_TYPE or star:
            return match.group(0)
        inside = inside.strip()
        if held in constructed:
            return f"{held} {name}({inside});"
        return f"{held} {name} = {{{inside}}};" if inside else f"{held} {name};"

    return _map_code(text, lambda part: _BRACE_INIT.sub(one, part))


def _has_a_constructor(text: str, head: "re.Match[str]") -> bool:
    """Whether a class body declares a constructor of its own."""

    try:
        closing = _matching(text, head.end() - 1)
    except ValueError:
        return False
    body = text[head.end() - 1: closing]
    return re.search(rf"(?<![.\w>~]){re.escape(head.group(2))}\s*\(", body) is not None

def _rewrite_static_members(text: str, filename: str) -> str:
    """A static member is one object for the class, so it becomes one object.

    C has no such thing inside a struct, and there is nowhere for it to live
    but file scope. The name carries the class, so two classes may each have
    a `count` without either being the other's.
    """

    owners: "dict[str, str]" = {}
    given: "dict[str, str]" = {}
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
        body = text[head.end() - 1: closing]

        def taken(match: "re.Match[str]", o=owner) -> str:
            owners[match.group(2)] = o
            if match.group(3):
                # Given its value here, so this is where it is defined; there
                # is no `int C::limit = 10;` anywhere else to find.
                given[_c_name(o, match.group(2))] = (
                    f"{match.group(1).strip()} {_c_name(o, match.group(2))} "
                    f"{match.group(3).strip()};"
                )
            return ""

        out.append(text[at:head.end() - 1])
        out.append(_STATIC_MEMBER.sub(taken, body))
        at = closing
    out.append(text[at:])
    text = "".join(out)
    if not owners:
        return text

    # `int C::count = 0;` becomes the file-scope object itself.
    def defined(match: "re.Match[str]") -> str:
        spelled, owner, name, value = match.groups()
        if owners.get(name) != owner:
            return match.group(0)
        return f"{spelled} {_c_name(owner, name)} {value or '= 0'};"

    text = _STATIC_DEFINITION.sub(defined, text)
    # Those that were given a value in the class need their storage written.
    if given:
        text = "\n".join(given.values()) + "\n" + text
    # And every mention of it - `C::count` from outside, `count` from within -
    # is that object.
    for name, owner in owners.items():
        spelled = _c_name(owner, name)
        text = _map_code(
            text,
            lambda part, o=owner, n=name, s=spelled: re.sub(
                rf"\b{re.escape(o)}\s*::\s*{re.escape(n)}\b", s, part
            ),
        )
        text = _map_code(
            text,
            lambda part, n=name, s=spelled: re.sub(
                rf"(?<![.\w>:]){re.escape(n)}\b(?!\s*::)", s, part
            ),
        )
    return text

#: `int add(int a, int b = 10)` - a parameter with a default.
_DEFAULTED = re.compile(r"([A-Za-z_]\w*)\s*=\s*([^,)]+)")

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
            lifted.append(text[start:end])
            text = text[:start] + text[end:]
            owner = head.group(2)
            text = _map_code(
                text,
                lambda part, o=owner: re.sub(
                    rf"\b{re.escape(o)}\s*::\s*", "", part
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

    lifted: list[str] = []
    while True:
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
            if end < len(text) and text[end] == ";":
                end += 1
            outer, name = head.group(2), inner.group(2)
            spelled = f"{outer}__{name}"
            taken = text[start:end].replace(name, spelled, 1)
            lifted.append(taken)
            text = text[:start] + text[end:]
            text = _map_code(
                text,
                lambda part, o=outer, n=name, s=spelled: re.sub(
                    rf"\b{re.escape(o)}\s*::\s*{re.escape(n)}\b", s, part
                ),
            )
            text = _map_code(
                text,
                lambda part, n=name, s=spelled: re.sub(
                    rf"(?<![.\w>:]){re.escape(n)}\b(?!\s*::)", s, part
                ),
            )
            moved = True
            break
        if not moved:
            return "\n".join(lifted) + ("\n" if lifted else "") + text


def _settled_parameters(
    parameters: str,
    holder: str,
    text: str,
    closing: int,
    filename: str,
    at: int,
) -> str:
    """Give a generic lambda's `auto` parameters the types it is called with.

    `[](auto a, auto b)` is a member template in C++ - one copy per set of
    argument types. py2bin writes a lambda out as one class, so the types are
    read from the calls instead. Where the calls disagree, that is a template
    and this says so rather than compiling the first one and running it for
    both.
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
            "writes one copy of a lambda. Nothing here says what it is called "
            "with, so give the parameters their types",
        )
    if any(one != settled[0] for one in settled[1:]):
        raise CppTranslationError(
            filename, _line_of(text, at),
            "a lambda whose parameters are `auto` is called here with more "
            "than one set of types, which is a template; py2bin writes one "
            "copy of a lambda, so write one for each set",
        )
    out = []
    for spelled, wanted in zip(_split_arguments(parameters), settled[0]):
        spelled = spelled.strip()
        out.append(re.sub(r"(?<![.\w>])auto\b", wanted, spelled, count=1))
    return ", ".join(out)


#: `std::function<int(int)>` - by the time this runs the `std::` is gone.
_STD_FUNCTION = re.compile(r"(?<![.\w>])function\s*<")

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


def _erased_holders(text: str, name: str) -> "set[str]":
    """Every name declared to hold this signature: a variable, a member."""

    return {
        match.group(1)
        for match in re.finditer(
            rf"(?<![.\w>]){re.escape(name)}\s*&?\s*([A-Za-z_]\w*)\s*[;=,)]", text
        )
    }


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
            if re.match(rf"\s*(?:const\s+)?{re.escape(name)}\s*&?\s*\w*\s*$", part)
        ]
        if at:
            found[match.group(2)] = at
    return found


def _assigned_to(text: str, name: str, filename: str) -> "list[str]":
    """Every value this signature is given: assigned, declared, or passed."""

    holders = _erased_holders(text, name)
    found: "list[str]" = []
    bare = _without_literals(text)
    # Passed to a function that takes one.
    for called, positions in _erased_parameters(text, name).items():
        for match in re.finditer(rf"(?<![.\w>]){re.escape(called)}\s*\(", bare):
            close = _closing_paren(bare, match.end() - 1)
            if close < 0 or _is_a_definition(bare, close):
                continue
            given = _split_arguments(bare[match.end(): close])
            for index in positions:
                if index >= len(given):
                    continue
                value = given[index].strip()
                if value.isidentifier() and value not in holders:
                    found.append(value)
    # `NAME f = value;` - given where it is declared.
    for match in re.finditer(
        rf"(?<![.\w>]){re.escape(name)}\s+[A-Za-z_]\w*\s*=\s*([A-Za-z_]\w*)\s*;",
        bare,
    ):
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
        text = _erased_given(text, name, holders, places, by_value)
        text = _erased_truth(text, holders)
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
        for called, positions in _erased_parameters(text, name).items():
            bare = _without_literals(text)
            for match in re.finditer(
                rf"(?<![.\w>]){re.escape(called)}\s*\(", bare
            ):
                close = _closing_paren(bare, match.end() - 1)
                if close < 0 or _is_a_definition(bare, close):
                    continue
                given = _split_arguments(text[match.end(): close])
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
                    made = f"__py2bin_given_{abs(hash((called, index, value))) % 100000}"
                    given[index] = made
                    start = _statement_start(text, match.start())
                    text = (
                        text[:start]
                        + f" {name} {made}; {made}.__which = {which}; "
                        f"{made}.__held{which} = {value}; "
                        + text[start:match.end()]
                        + ", ".join(one.strip() for one in given)
                        + text[close:]
                    )
                    holders.add(made)
                    changed = True
                    break
                if changed:
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
        if re.search(r"(?<![.\w>])auto\b", parameters):
            parameters = _settled_parameters(
                parameters, holder, text, closing, filename, found.start()
            )
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
        made.append(
            f"class {name} {{\npublic:\n{members}"
            f"    {name}() {{ }}\n"
            f"    {result} operator()({parameters}) {body}\n}};\n"
        )
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
    names = _names_of_class(text, owner)
    for name in sorted(names, key=len, reverse=True):
        body = _map_code(
            body,
            lambda part, n=name: re.sub(
                rf"(?<![.\w>]){re.escape(n)}\b", f"{_SELF}->{n}", part
            ),
        )
    return body


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
        # A name that is called is a function, not a capture.
        after = body[match.end():].lstrip()
        if after.startswith("("):
            continue
        if name not in _declared_here(scope):
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
    r"(?:([A-Za-z_][\w\s*&]*?)\s+)?\b([A-Za-z_]\w*)\s*<([^<>]*)>\s*"
    r"\(([^()]*)\)\s*(?:const\s*)?\{"
)

#: `template<typename T> T Box<T>::get() { ... }` - a member of a class
#: template, defined outside it. Which is how a header usually writes one.
_TEMPLATE_MEMBER = re.compile(
    r"\btemplate\s*<([^<>]*)>\s*"
    r"(?:([A-Za-z_][\w\s*&]*?)\s+)?"
    r"\b([A-Za-z_]\w*)\s*<([^<>]*)>\s*::\s*(~?[A-Za-z_]\w*)\s*"
    r"\(([^()]*)\)\s*(?:const\s*)?\{"
)

#: A backstop on the folding below, not a budget.
_OUT_OF_LINE_ROUNDS = 512



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
            if _depth_at(text, match.start()) > 0:
                found = match
                break
        if found is None:
            return text
        rest = text[found.end():]
        definition = _DEFINITION.match(rest)
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
        copies, without = _member_copies(
            without, name, parameters, definition.group(3), pattern
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


def _member_copies(
    text: str,
    name: str,
    parameters: "list[tuple[str, bool]]",
    declared: str,
    pattern: str,
) -> "tuple[list[str], str]":
    """One copy per set of argument types the calls to `name` ask for."""

    made: "dict[str, str]" = {}
    out: list[str] = []
    at = 0
    for call in re.finditer(
        rf"(\.|->)\s*{re.escape(name)}\s*\(", _without_literals(text)
    ):
        if call.start() < at:
            continue
        close = _closing_paren(text, call.end() - 1)
        if close < 0:
            continue
        given = _call_arguments(text, call.end() - 1)
        deduced = _deduce_arguments(parameters, declared, given, text, call.start())
        if deduced is None:
            continue
        named = _instantiated_name(name, deduced)
        if named not in made:
            copy = _substituted(
                pattern, [held for held, _is_type in parameters], deduced
            )
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
_NESTED_TYPEDEF = re.compile(
    r"\btypedef\s+([A-Za-z_][\w\s]*?)\s*(\**)\s*([A-Za-z_]\w*)\s*;"
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
                f"{match.group(1).strip()} {match.group(2)}".strip()
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
    return text

def _expand_templates(text: str, filename: str) -> str:
    """Replace every template with the copies this file actually asks for."""

    patterns: dict[str, tuple[list[tuple[str, bool]], str, str]] = {}
    #: Copies the author wrote out by hand, under the name the expander would
    #: have used. They seed `made`, so the pattern is never copied over them.
    written: "dict[str, str]" = {}
    cut: list[tuple[int, int]] = []
    for match in _TEMPLATE.finditer(text):
        if _depth_at(text, match.start()) != 0:
            continue
        parameters = _template_parameters(match.group(1))
        rest = text[match.end():]
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
        definition = _DEFINITION.match(rest)
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

    if not patterns and not written:
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

    made: "dict[str, str]" = dict(written)
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

        asked: list[tuple[str, list[str]]] = []
        unread: list[tuple[str, int]] = []
        for name, entries in patterns.items():
          for parameters, kind, pattern_text in entries:
            if kind != "function":
                continue
            # A call that did not spell the arguments out: `twice(5)` rather
            # than `twice<int>(5)`. What it means is read off the arguments.
            declared = _DEFINITION.match(pattern_text)
            if declared is None:
                continue
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
                given = _call_arguments(region, call.end() - 1)
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
                asked.append((name, deduced, (parameters, kind, pattern_text)))
        for name in patterns:
            for found in re.finditer(rf"(?<![.\w>]){re.escape(name)}\s*<", region):
                close = _closing_angle(region, found.end() - 1)
                if close < 0:
                    continue
                arguments = _split_arguments(region[found.end(): close])
                if any(
                    re.search(
                        rf"(?<![.\w>]){re.escape(other)}\s*<", ",".join(arguments)
                    )
                    for other in patterns
                ):
                    continue  # an inner template first; next round
                # Spelled out, so the entry is whichever takes that many
                # template parameters.
                for entry in patterns[name]:
                    # `entry[0]` is the parameter list already read, not the
                    # text it was read from.
                    if len(entry[0]) == len(arguments):
                        asked.append(
                            (name, [a.strip() for a in arguments], entry)
                        )
                        break
        return asked, unread

    def point(region: str, scope: str) -> str:
        """Send every use in this region to the copy written for it.

        Same reasoning as in :func:`uses` about which text is searched first.
        """

        scope = region + "\n" + scope

        out: list[str] = []
        at = 0
        spelled_out = re.compile(
            r"(?<![.\w>])(" + "|".join(re.escape(n) for n in patterns) + r")\s*<"
        )
        for found in spelled_out.finditer(region):
            if found.start() < at:
                continue
            close = _closing_angle(region, found.end() - 1)
            if close < 0:
                continue
            arguments = [
                a.strip() for a in _split_arguments(region[found.end(): close])
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
          for parameters, kind, pattern_text in entries:
            if kind != "function":
                continue
            declared = _DEFINITION.match(pattern_text)
            if declared is None:
                continue
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

        fresh = [
            (name, arguments, entry)
            for name, arguments, entry in wanted
            if _instantiated_name(name, arguments) not in made
        ]
        for name, arguments, entry in fresh:
            parameters, kind, pattern = entry
            if len(arguments) != len(parameters):
                raise CppTranslationError(
                    filename,
                    _line_of(text, text.index(name)),
                    f"{name} is a template taking {len(parameters)} argument(s) "
                    f"and is used here with {len(arguments)}",
                )
            named = _instantiated_name(name, arguments)
            copy = _substituted(
                pattern, [p for p, _is_type in parameters], arguments
            )
            # Rename the pattern itself, and the constructors and destructors
            # inside it, which are spelled with the class's own name.
            copy = _map_code(
                copy,
                lambda part, n=name, s=named: re.sub(
                    rf"(?<![.\w>]){re.escape(n)}\b(?!\s*<)", s, part
                ),
            )
            made[named] = copy

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
_IN_FLIGHT = "__py2bin_in_flight"

#: Declared once, at the top of any file that throws.
_EXCEPTION_STATE = f"""
static int {_THROWN} = 0;
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
        while head.split() and head.split()[0] in _STORAGE:
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

    bodies = {
        name: text[match.end() - 1: closing]
        for match, closing, name, _returns in _every_body(text)
    }
    throwing = {
        name for name, body in bodies.items() if _THROW.search(_without_literals(body))
    }
    while True:
        grown = set(throwing)
        for name, body in bodies.items():
            if name in grown:
                continue
            code = _without_literals(body)
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
        found[name] = (spelled or "long").replace("&", "*")
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
        handler_open = offset + catch.end() - 1
        handler_close = _matching(body, handler_open)

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
        handled = _guarded(
            body[handler_open + 1: handler_close - 1],
            landing,
            throwing,
            classes,
            filename,
            counter,
        )
        caught = _catch_binding(
            catch.group(1).strip(), filename, body, offset, classes
        )
        made = (
            f"{{ {guarded} }} goto {after}; {label}: ; "
            f"{{ {_THROWN} = 0; {caught}{handled} }} {after}: ;"
        )
        finished.append(made)
        body = (
            body[:found.start()]
            + (_TRY_MARK % (len(finished) - 1))
            + body[handler_close:]
        )


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
    if held.strip() in _CLASS_NAMES:
        if by_reference:
            # A reference to what is in flight, not a copy of it - which is
            # what `catch (std::exception &e)` is for. Written as a C++
            # reference so the pass that turns those into pointers does it,
            # and `e.what()` reaches the object that was actually thrown
            # rather than the base it was sliced to.
            return f"{held} &{named} = *({held} *){_IN_FLIGHT}; "
        if held.strip() in _POLYMORPHIC and held.strip() in _INHERITED_FROM:
            raise CppTranslationError(
                filename,
                _line_of(body, at),
                f"catching {held.strip()} by value, and something in this "
                f"file derives from it. C++ slices the object to that class "
                f"here, so a virtual function called on it answers as the "
                f"base rather than as what was thrown - py2bin's copy keeps "
                f"the object it was made from and would answer differently. "
                f"Write `catch ({held.strip()} &{named})`, which is what the "
                f"slicing is a reason to write anyway",
            )
        # Declared and then assigned, not initialised: py2bin's C takes
        # `o = *p;` and not `struct V o = *p;`.
        return (
            f"{held} {named}; {named} = *({held} *){_IN_FLIGHT}; "
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
            f"{{ {_THROWN} = 1; "
            f"{made.group(1)} *__py2bin_raised = new {spelled}; "
            f"{_IN_FLIGHT} = (long)__py2bin_raised; {landing.leave()} }}"
        )
    held = _deduced_type(spelled, body, match.start())
    if held is not None and held.replace("*", "").strip() in _CLASS_NAMES:
        named = held.replace("*", "").strip()
        return (
            f"{{ {_THROWN} = 1; "
            f"{named} *__py2bin_raised = ({named} *)malloc(sizeof({named})); "
            f"*__py2bin_raised = {spelled}; "
            f"{_IN_FLIGHT} = (long)__py2bin_raised; {landing.leave()} }}"
        )
    return (
        f"{{ {_THROWN} = 1; {_IN_FLIGHT} = (long)({spelled}); {landing.leave()} }}"
    )


#: `Err(1, 2)` - a temporary built where it is thrown.
_CONSTRUCTED = re.compile(r"^([A-Za-z_]\w*)\s*\(")

#: The classes this file declares. Read before the exception pass, which runs
#: before classes are taken apart and so has nothing else to ask.
_CLASS_NAMES: "set[str]" = set()

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
        name, base = head.group(2), head.group(3)
        bases[name] = base or ""
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
    pattern = re.compile(
        r"(?<![.\w>])((?:[A-Za-z_]\w*\s*(?:\.|->)\s*)*)([A-Za-z_]\w*)\s*\("
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
    name = call.split("(", 1)[0].strip().replace("->", ".").split(".")[-1].strip()
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

    def rename(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in overloaded:
            return match.group(0)
        given = _call_arguments(match.string, match.end() - 1)
        arity = len(given)
        same = [p for p in overloaded[name] if _arity(p) == arity]
        if not same:
            return match.group(0)
        if len(same) == 1:
            return f"{name}__{arity}("
        wanted = [_deduced_type(value, match.string, match.start()) for value in given]
        if any(item is None for item in wanted):
            raise CppTranslationError(
                filename,
                _line_of(match.string, match.start()),
                f"more than one {name}() takes {arity} argument(s), and "
                f"py2bin cannot tell which is meant here. It reads the type "
                f"of a literal and of a variable it can see declared; cast "
                f"the argument to the type of the one you want",
            )
        codes = [_type_code(item) for item in wanted]
        for parameters in same:
            if _parameter_types(parameters) == codes:
                return f"{name}__{suffix_of(name, parameters)}("
        for parameters in same:
            if all(
                declared == code or declared in _PROMOTIONS.get(code, ())
                for declared, code in zip(_parameter_types(parameters), codes)
            ):
                return f"{name}__{suffix_of(name, parameters)}("
        return match.group(0)

    return _map_code(
        text, lambda part: re.sub(r"(?<![.\w>])([A-Za-z_]\w*)\s*\(", rename, part)
    )


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
    """

    known: set[str] = set()
    declared: dict[str, str] = {}

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
            for spelled in _declared_names(_without_nested(inner)):
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
    for header in re.finditer(r"#\s*include\s*<([^>]+)>", text):
        name = header.group(1)
        if name in _BUILTIN_CPP_HEADERS:
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



def _without_literals(text: str) -> str:
    """The text with every literal blanked, for scanning rather than editing.

    Kept the same length so an offset still means something, and emptied so
    nothing inside a literal can be read as code.
    """

    return "".join(
        part if kind == "code" else " " * len(part)
        for kind, part in _split_literals(text)
    )


def _split_literals(text: str) -> "list[tuple[str, str]]":
    """The text as alternating ("code", ...) and ("literal", ...) pieces."""

    pieces: list[tuple[str, str]] = []
    chunk: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in "\"'":
            pieces.append(("code", "".join(chunk)))
            chunk = []
            quote = char
            literal = [char]
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
    base = owner.base
    depth = 1
    while base and base in classes:
        # One `__base.` per level. A name inherited through two classes lives
        # two bases down, and a single hop named a member the middle class
        # does not have.
        for name in classes[base].field_names():
            names.setdefault(name, "__base." * depth)
        base = classes[base].base
        depth += 1

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

    seen = name
    while seen and seen in classes:
        for method in classes[seen].methods:
            if _slot_key(method) == key and not method.pure:
                return seen
        seen = classes[seen].base
    return None


def _slot_method(
    name: str, key: "tuple[str, int]", classes: "dict[str, Class]"
) -> "Method | None":
    """The declaration for a slot, wherever it was first written."""

    seen = name
    while seen and seen in classes:
        for method in classes[seen].methods:
            if _slot_key(method) == key:
                return method
        seen = classes[seen].base
    return None


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
        held = part.replace("*", "").strip().split()
        if "*" not in part and held and held[0] in classes:
            spelled = f"struct {held[0]} *"
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

    steps = []
    seen = name
    while seen and seen in classes and not _carries_vptr(seen, classes):
        steps.append("__base")
        seen = classes[seen].base
    return ".".join([*steps, "__vptr"])


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
    return "\n".join(lines)


def _emit_class(found: Class, classes: "dict[str, Class]") -> str:
    """The struct, and the free functions its methods become."""

    lines = [f"struct {found.name} {{"]
    if found.base:
        # First, so a pointer to the derived object is a pointer to the base.
        lines.append(f"    struct {found.base} __base;")
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
    if not found.members and not found.base:
        # C has no empty struct; give it something so the type exists.
        lines.append("    int __empty;")
    lines.append("};")
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
    parameters = "" if method.shared else f"struct {found.name} *this"
    # A value return becomes the hidden pointer a compiler would pass: the
    # caller supplies the space and the callee writes through it. py2bin's C
    # cannot return a struct, and does not have to - this is the same
    # transform an ABI performs, done where the C is written.
    returned = _returns_object(method, classes)
    if returned:
        parameters += f", struct {returned} *__ret"
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
        returns=method.returns,
        referenced=set(references),
    )
    if returned:
        body = _return_through_pointer(body)
    if method.name == "":
        body, base_arguments = _base_initialiser(body, found)
        body, member_arguments = _member_initialisers(body)
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
        body = _return_the_address(body)
    return f"static {returns} {name}({parameters}) {body}"


def _return_the_address(body: str) -> str:
    """`return items[i];` becomes `return &(items[i]);` for a reference."""

    return re.sub(
        r"\breturn\s+([^;]+);",
        lambda m: f"return &({m.group(1).strip()});",
        body,
    )


def _return_through_pointer(body: str) -> str:
    """`return v;` becomes `*__ret = v; return;` - the value goes to the caller.

    Written as two statements rather than one so the expression is evaluated
    before anything else happens, which is the order C++ promises.
    """

    def replace(match: "re.Match[str]") -> str:
        value = match.group(1).strip()
        if not value:
            return match.group(0)
        return f"{{ *__ret = {value}; return; }}"

    return _map_code(body, lambda part: re.sub(r"\breturn\b([^;]*);", replace, part))


def _subobjects(found: Class, classes: "dict[str, Class]") -> "list[tuple[str, str]]":
    """The base and the class-typed members, each with the address to use.

    C++ builds these before the constructor body runs and takes them apart
    after the destructor body does. C does nothing at all, so a class holding
    another read whatever was on the stack - which is a wrong answer rather
    than a failure, and the worst kind to ship.
    """

    parts: list[tuple[str, str]] = []
    if found.base and found.base in classes:
        parts.append((found.base, "&this->__base"))
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
    spelled = " ".join(f"this->{name} = {value};" for name, value in values)
    opening = body.find("{")
    return body[:opening + 1] + " " + spelled + body[opening + 1:]


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
    for held, address in _subobjects(found, classes):
        if base_arguments is not None and address == "&this->__base":
            # Named in the initialiser list, so it is built with what was
            # written there rather than with nothing.
            owner = _find_method(held, "", classes)
            given = (
                [a.strip() for a in _split_arguments(base_arguments)]
                if base_arguments else []
            )
            if owner is not None:
                calls.append(
                    f"{_c_name(owner, '', _call_suffix(owner, '', classes, given, reading))}"
                    f"({address}{', ' + base_arguments if base_arguments else ''});"
                )
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
            if len(given) == 1 and _same_class(given[0], held, reading, classes):
                # `Person(string n) : name(n)` names the copy constructor,
                # and a class that wrote none has the memberwise copy py2bin
                # already does everywhere else. Nothing to construct.
                calls.append(
                    _copied_in(held, address.lstrip("&"), f"&{given[0]}", classes)
                )
                continue
            calls.append(
                f"{_c_name(owner, '', _call_suffix(owner, '', classes, given, reading))}"
                f"({address}{', ' + arguments if arguments.strip() else ''});"
            )
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
        calls.append(
            f"{_c_name(owner, '', _call_suffix(owner, '', classes, []))}({address});"
        )
    if _is_polymorphic(found.name, classes):
        # After the base constructor, which set the pointer to *its* table.
        # Overwriting it here is what C++ means by the object becoming its own
        # type as construction proceeds, and it is why a virtual call made
        # from a base constructor reaches the base's version.
        calls.append(
            f"this->{_vptr_path(found.name, classes)} = "
            f"{_vtable_name(found.name)};"
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
    base = owner.base
    depth = 1
    while base and base in classes:
        for member in classes[base].members:
            found.append((member, "__base." * depth))
        base = classes[base].base
        depth += 1
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
            depth = _base_depth(owner.name, provider, classes)
            reached = "this" if depth == 0 else (
                "&this->" + "__base." * (depth - 1) + "__base"
            )
            pattern = (
                rf"(?<![.\w>]){re.escape(named)}\s*::\s*"
                rf"{re.escape(method)}\s*\("
            )
            body = _rewrite_calls(
                body, pattern, _name_for(provider, method, classes), reached
            )
    return body


def _bare_method_calls(
    body: str, owner: Class, classes: "dict[str, Class]", scope: str = ""
) -> str:
    """`sum()` inside a member is a call on `this`, and C has no such thing.

    Written out here rather than left to the `this->` pass above, which points
    *names* at the object: a call is the name plus its argument list, and the
    object has to be threaded through as the first argument.
    """

    for method in sorted(_reachable_methods(owner.name, classes), key=len, reverse=True):
        provider = _find_method(owner.name, method, classes)
        if provider is None:
            continue
        reached = "this"
        if provider != owner.name:
            depth = _base_depth(owner.name, provider, classes)
            reached = "&this->" + "__base." * (depth - 1) + "__base"
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

    def chosen(given: "list[str]"):
        through = virtual(given) if virtual is not None else ""
        if through:
            return through, "this"
        return (direct if isinstance(direct, str) else direct(given)), reached

    return chosen

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
_PROMOTIONS = {
    "int": ("long", "unsigned_long", "double", "float", "unsigned_int", "char", "short"),
    "char": ("int", "long", "unsigned_long", "double"),
    "double": ("float",),
    "char_p": ("void_p",),
}


def _chosen_overload(
    set_of: "list[Method]", given: "list[str]", text: str, before: int
) -> "Method | None":
    """Which member of an overload set a call with these arguments means."""

    candidates = [m for m in set_of if _arity(m.parameters) == len(given)]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    wanted = [_deduced_type(value, text, before) for value in given]
    if any(item is None for item in wanted):
        return None
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
    return near[0] if len(near) == 1 else None


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

    seen = name
    while seen:
        found = classes.get(seen)
        if found is None:
            return None
        if any(m.name == method for m in found.methods):
            return seen
        seen = found.base
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

    seen = derived
    while seen:
        if seen == base:
            return True
        found = classes.get(seen)
        if found is None:
            return False
        seen = found.base
    return False



def _upcast_pointers(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
    returns: str,
    scope: str,
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
    for match in _DEFINITION.finditer(scope):
        if _depth_at(scope, match.end() - 1) != 0 or match.group(2) in _NOT_A_TYPE:
            continue
        wanted_at = [
            (index, part.replace("*", " ").replace("const", " ").strip())
            for index, part in enumerate(_split_arguments(match.group(3)))
            if "*" in part
            and part.replace("*", " ").replace("const", " ").split()
            and part.replace("*", " ").replace("const", " ").split()[0] in classes
        ]
        if wanted_at:
            body = _cast_at_positions(
                body, match.group(2), wanted_at, classes, scope, None
            )

    # And the same for a method, whose parameters the class table has.
    signatures = _call_signatures(classes)
    for name, (owner, method) in signatures.items():
        wanted_at = [
            (index, part.replace("*", "").strip())
            for index, part in enumerate(_split_arguments(method.parameters))
            if "*" in part
            and part.replace("*", "").replace("const", "").strip().split()
            and part.replace("*", "").replace("const", "").strip().split()[0] in classes
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
            wanted = spelled.replace("const", "").strip().split()[0]
            where = index + skip
            if where >= len(given):
                continue
            value = given[where].strip()
            if value.startswith("(struct"):
                continue
            held = (_deduced_type(value, scope) or "").replace("*", "").strip()
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
    for pattern in (_OBJECT, _OBJECT_ARRAY, _OBJECT_POINTER, _VALUE_INIT):
        for match in pattern.finditer(body):
            held, name = match.group(1), match.group(2)
            if held in classes:
                found[name] = held
    return found



def _already_a_declaration(text: str, match: "re.Match[str]") -> bool:
    """Whether this call is already the whole initialiser of a declaration.

    That form needs no temporary: the object being declared *is* the space
    the callee writes through. It also covers what this pass writes itself -
    the same thing with a `&` in it - which was otherwise hoisted again, and
    again, until the round limit stopped it.
    """

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
    if not returning:
        return body
    pattern = re.compile(
        r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\("
    )

    def taken(match: "re.Match[str]") -> str:
        spelled, variable, called = match.groups()
        if returning.get(called) != spelled or spelled not in classes:
            return match.group(0)
        close = _closing_paren(match.string, match.end() - 1)
        if close < 0:
            return match.group(0)
        arguments = match.string[match.end(): close]
        passed = f", {arguments}" if arguments.strip() else ""
        return f"{spelled} {variable}; {called}(&{variable}{passed})" + "\x00drop"

    body = _map_code(body, lambda part: pattern.sub(taken, part))
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
    returning: "dict[str, str]" = {}
    for definition in _DEFINITION.finditer(f"{body}\n{scope}"):
        held = definition.group(1).strip()
        if held in classes and "*" not in held and "&" not in held:
            returning[definition.group(2)] = held

    def settled(text: str, match: "re.Match[str]", close: int) -> bool:
        """Whether this call still needs somewhere to write its answer."""

        return not _already_a_declaration(text, match)

    def self_standing(text: str) -> "tuple | None":
        """The first call to a free function that answers an object."""

        for name, held in returning.items():
            for match in re.finditer(
                rf"(?<![.\w>]){re.escape(name)}\s*\(", text
            ):
                close = _closing_paren(text, match.end() - 1)
                if close < 0 or _is_a_definition(text, close):
                    continue
                if not settled(text, match, close):
                    continue
                if _stands_alone(text, match, close):
                    # Nothing wants the answer, so there is nowhere it has to
                    # go. An earlier pass has usually already given a call in
                    # this position the hidden pointer it writes through, and
                    # wrapping it again handed a temporary the `void` that a
                    # rewritten call returns.
                    continue
                return (match, close, held, False)
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
            found = (match, close, declared, by_reference)
            break
        if found is None:
            found = self_standing(body)
        if found is None:
            return body
        match, close, held, by_reference = found
        counter[0] += 1
        name = f"__py2bin_value_{counter[0]}"
        known[name] = held
        start = _statement_start(body, match.start())
        call = body[match.start(): close + 1]
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


def _statement_start(body: str, at: int) -> int:
    """Where the statement holding `at` begins.

    A temporary has to be declared before the statement that uses it, not in
    the middle of one - `printf("%s", f.filename().c_str())` has no room for
    a declaration inside the argument list.
    """

    depth = 0
    index = at
    while index > 0:
        index -= 1
        piece = body[index]
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
_TEMPORARY = re.compile(r"(?<![.\w>])([A-Za-z_]\w*)\s*\(")



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
            if re.search(r"\bnew$", before):
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
        owner = _find_method(type_name, "", classes)
        if owner is None and arguments:
            raise CppTranslationError(
                "<c++>", 0, f"{type_name} has no constructor to take arguments"
            )
        if owner is not None:
            given = (
                [a.strip() for a in _split_arguments(arguments)]
                if (arguments or "").strip()
                else []
            )
            suffix = _call_suffix(owner, "", classes, given, scope())
            passed = f", {arguments}" if arguments else ""
            constructed += f" {_c_name(owner, '', suffix)}(&{variable}{passed});"
        if _find_method(type_name, "~", classes):
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

    # Before the declaration passes: `Node *n = new Node(3);` has to become a
    # call first, or the pointer declaration reads `new Node` as the type.
    # `new int(5)` stores as well as answers, and C has no one expression
    # that does both. Written out as its own statement first, so what reaches
    # the rewrite below is the storage on its own.
    body = _hoist_new_initialisers(body, classes, [0])
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

    # `V(5)` written where a value goes: C has no expression that constructs,
    # so each temporary becomes an object with a name ahead of the statement.
    body = _rewrite_temporaries(body, classes, [0])

    # `f.filename().c_str()`: a call on what a value return handed back. The
    # declarations have not been read yet - they are rewritten below, and
    # this has to run before that - so what this body declares is scanned for
    # first, without touching it.
    body = _hoist_value_returns(
        body, classes, {**_declared_objects(body, classes), **known}, [0], unit
    )
    # And the declarations that hoist just wrote: a free function answering an
    # object fills a space the caller provides, here as everywhere else. The
    # file-scope pass does not reach a method body, which is emitted on its
    # own - so `held.append(to_string(v))` inside one was left calling a
    # function with one argument too few.
    body = _free_value_initialisers(body, classes, f"{body}\n{unit}")

    # `int &r = a.v;` is a pointer whose uses are dereferenced. Done before
    # anything else reads the body, so the rest sees an ordinary pointer.
    local_references: dict[str, str] = {}
    bindings: list[str] = []

    def bind(match: "re.Match[str]") -> str:
        spelled, variable, source = match.groups()
        held = spelled.replace("const", "").strip()
        if held in _NOT_A_TYPE:
            return match.group(0)
        if not _could_start_a_declaration(match.string, match.start()):
            # `flags & mask` is an operator, not a reference, and only where
            # the statement begins can a declaration be what was meant.
            return match.group(0)
        local_references[variable] = held
        if held in classes:
            known[variable] = held
            pointers.add(variable)
            made = f"struct {held} *{variable} = &({source.strip()});"
        else:
            made = f"{spelled} *{variable} = &({source.strip()});"
        # Held aside while the uses are dereferenced: the declaration is the
        # one place the name means the pointer and not what it points at, and
        # rewriting it too gave `int *(*alias)`.
        bindings.append(made)
        return _BINDING_MARK % (len(bindings) - 1)

    body = _map_code(body, lambda part: _LOCAL_REFERENCE.sub(bind, part))
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
    body, from_operators = _rewrite_value_operators(body, classes, known, pointers)
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

    body = _rewrite_operators(
        body, classes, known, pointers, scope(), referenced or set()
    )
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
                depth = _base_depth(holds, owner, classes)
                inner = variable if variable in pointers else f"{variable}"
                path = "".join(["->" if variable in pointers else "."] +
                               ["__base."] * (depth - 1) + ["__base"])
                reached = f"&{inner}{path}"
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
    body = _upcast_pointers(body, classes, known, pointers, returns, scope())
    body = _fix_call_arguments(body, classes, known, pointers)

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
            [*enclosing, *((name, known[name]) for name in destroyed)],
            # An array of pointers declared in an enclosing scope is still an
            # array of pointers inside a block. Without this `all[i]->get()`
            # in a `for` body was left as C++, while the identical statement
            # written without the braces translated - which is the sort of
            # difference nobody would think to test for.
            dict(pointer_arrays),
            returns,
            {**outer_references, **local_references},
        )
        for inner in blocks
    ]
    body = _close_with_destructors(body, destroyed, known, classes, enclosing)
    return _restore_nested(body, rewritten_blocks)




#: Stands in for a nested block while the enclosing scope is rewritten, so a
#: declaration inside one is not mistaken for a declaration in this one.
_BLOCK_MARK = "\x00py2bin_block_%d\x00"



def _initialiser_brace(body: str, index: int) -> bool:
    """Whether the `{` at `index` opens a list of values rather than a scope."""

    before = body[:index].rstrip()
    return before.endswith("=") or before.endswith(",")

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
    while index < len(body):
        char = body[index]
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
        out.append(char)
        index += 1
    return "".join(out), blocks


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
) -> "tuple[str, dict[str, str]]":
    """`V c = a + b;` - an operator answering an object, given somewhere to put it."""

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
            suffix = _call_suffix(owner, name, classes, [right], body, match.start())
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

def _rewrite_operators(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
    scope: str = "",
    referenced: "set[str] | None" = None,
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
            body = _rewrite_binary_operator(
                body, variable, symbol, owner, name, address, known, pointers,
                classes, scope,
            )
            if variable in pointers:
                # `base[child] < base[root]` - an element of an array of
                # objects is an object, and comparing two of them is the
                # class's own operator. This is how a container's own code
                # is written, and a pattern matching a name never saw it.
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
        pattern = re.compile(
            rf"(?<![.\w>=!<]){re.escape(variable)}\s*=(?!=)\s*([A-Za-z_]\w*)\s*;"
        )

        def assigned(match: "re.Match[str]", o=owner, a=address) -> str:
            source = match.group(1)
            if known.get(source) != holds:
                return match.group(0)
            passed = source if source in pointers else f"&{source}"
            return f"{_c_name(o, 'op_assign')}({a}, {passed});"

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
    return body


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
                    body = _rewrite_calls(
                        body,
                        rf"(?<![.\w>]){re.escape(variable)}\s*->\s*"
                        rf"{re.escape(method)}\s*\(",
                        _name_for(provider, method, classes, body),
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
        if before and (before[-1].isalnum() or before[-1] in "_)]\"'"):
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


def _call_signatures(
    classes: "dict[str, Class]"
) -> "dict[str, tuple[str, Method]]":
    """Every method by the C name it is emitted under."""

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
        # `T__new` passes no receiver; everything else passes `this` first,
        # and a value return puts the caller's space after it.
        made = found.group(1)
        skip = 0 if made.endswith("new") or "__new__" in made else 1
        if skip and _returns_object(method, classes):
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
            words = declared.replace("*", " * ").split()
            by_value = (
                "*" not in declared
                and "&" not in declared
                and len(words) == 2
                and words[0] in classes
            )
            if value in pointers:
                continue
            if by_value or _REFERENCE.search(declared):
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
        separator = "" if rest.startswith(")") else ", "
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
        separator = "" if rest.startswith(")") else ", "
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
        result, parameters = _c_signature(holds, declared, classes)
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
            return through, receiver
        spelled = named if isinstance(named, str) else named(given)
        return (spelled, direct) if direct is not None else spelled

    return chosen


def _reachable_methods(name: str, classes: "dict[str, Class]") -> "list[str]":
    found: list[str] = []
    seen = name
    while seen and seen in classes:
        found.extend(m.name for m in classes[seen].methods if m.name)
        seen = classes[seen].base
    return found


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
        reached = f"&{expression}." + "__base." * (_base_depth(holds, owner, classes) - 1) + "__base"
    return f"{_c_name(owner, '~')}({reached});"


def _close_with_destructors(
    body: str,
    destroyed: "list[str]",
    known: "dict[str, str]",
    classes: "dict[str, Class]",
    enclosing: "list[tuple[str, str]]" = (),
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
    leaving = calls + "".join(
        f" {_destructor_call(held, name, classes)}"
        for name, held in reversed(list(enclosing))
    )

    out = []
    at = 0
    for found in re.finditer(r"\breturn\b([^;]*);", body):
        value = found.group(1).strip()
        for name in destroyed:
            if re.search(rf"\b{re.escape(name)}\b", value):
                raise CppTranslationError(
                    "<c++>",
                    0,
                    f"this `return` uses {name}, which has a destructor. "
                    f"Running it first would return a destroyed object and "
                    f"running it after needs a temporary whose type this "
                    f"translator does not know - assign to a variable and "
                    f"return that instead",
                )
        out.append(body[at:found.start()])
        out.append(leaving.strip() + " " + found.group(0))
        at = found.end()
    out.append(body[at:])
    body = "".join(out)

    closing = body.rfind("}")
    if closing < 0:
        return body
    # And at the end, for a path that simply falls off it.
    return body[:closing] + calls + " " + body[closing:]



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
    r"(?<![#\w])(?:([A-Za-z_][\w \t]*?)\s*([*&]*)\s*)?"
    r"\b([A-Za-z_]\w*)::(~?[A-Za-z_]\w*)\s*"
    r"\(([^)]*)\)\s*(?:const\s*)?\{"
)


def translate(source: str, filename: str = "<c++>") -> str:
    """Translate the C++ subset in `source` into C.

    The result is ordinary C: structs where the classes were, free functions
    where the methods were, and calls rewritten to pass the object.
    """

    text = _strip_comments(source)
    # Before anything else reads the text: a class inside a namespace is a
    # class, and every pass below looks for classes at the top level.
    text, namespaces = _flatten_namespaces(text, filename)
    text, aliases = _namespace_aliases(text)
    text = _strip_namespace_qualifiers(text, namespaces | aliases)
    _refuse_unsupported(text, filename)
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
    text = _expand_templates(text, filename)
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
    if throws:
        # Before the classes are taken apart: what this leaves behind is a
        # `return`, and only the pass that reads a class body knows which
        # destructors a return has to run first. The names are read first,
        # because a thrown object is copied to the heap and the copy needs
        # its type spelled.
        global _CLASS_NAMES, _POLYMORPHIC, _INHERITED_FROM
        _CLASS_NAMES = {m.group(2) for m in _CLASS_HEAD.finditer(text)}
        _POLYMORPHIC = _polymorphic_names(text)
        _INHERITED_FROM = {
            m.group(3) for m in _CLASS_HEAD.finditer(text) if m.group(3)
        }
        text = _rewrite_exceptions_early(text, filename)
        patterned = True

    classes: dict[str, Class] = {}
    order: list[str] = []
    plain: list[str] = []
    pieces: list[tuple[int, int, str]] = []   # start, end, replacement

    for head in _CLASS_HEAD.finditer(text):
        keyword, name, base = head.group(1), head.group(2), head.group(3)
        opening = head.end() - 1
        try:
            closing = _matching(text, opening)
        except ValueError:
            raise CppTranslationError(
                filename, _line_of(text, opening), f"{keyword} {name} is not closed"
            ) from None
        # A `struct` with no methods is C already; leave it exactly as it is.
        inner = text[opening + 1: closing - 1]
        if keyword == "struct" and "(" not in inner:
            # A struct with no methods is C already and is left exactly as it
            # is - but C++ lets the bare name be a type and C does not, so it
            # still needs the typedef emitted below.
            plain.append(name)
            # It is still an object as far as *passing* goes: py2bin's C can
            # neither pass nor answer a struct by value, and `Point add(Point
            # a, Point b)` is as ordinary in C++ as it is impossible here. So
            # it is registered with nothing in it - which is all these passes
            # need to know - and is kept out of `order`, so the body it was
            # written with is emitted rather than one rebuilt from a reading
            # that a bitfield or an array would not survive.
            classes[name] = Class(name)
            continue
        found = _split_members(inner, name, filename, _line_of(text, opening))
        found.base = base
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
        if found.base and found.base not in classes:
            raise CppTranslationError(
                filename,
                found.methods[0].line if found.methods else 0,
                f"{name} inherits from {found.base}, which is not a class this "
                f"translation unit declares",
            )

    # Members defined outside their class, folded back into it.
    for out in _OUT_OF_LINE.finditer(text):
        spelled, stars, owner, method, parameters = out.groups()
        returns = f"{(spelled or '').strip()} {stars or ''}".strip()
        if owner not in classes:
            continue
        closing = _matching(text, out.end() - 1)
        body = text[out.end() - 1: closing]
        held = classes[owner]
        for existing in held.methods:
            spelled = "~" if method.startswith("~") else method
            if existing.name == spelled or (
                spelled == owner and existing.name == ""
            ):
                existing.body = body
                existing.parameters = "" if parameters.strip() in ("", "void") else parameters.strip()
                break
        else:
            spelled = "~" if method.startswith("~") else ("" if method == owner else method)
            held.methods.append(
                Method(spelled, returns or "void",
                       "" if parameters.strip() in ("", "void") else parameters.strip(),
                       body, _line_of(text, out.start()))
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
    directives = []
    kept_lines = []
    for line in remainder.split("\n"):
        (directives if line.lstrip().startswith("#") else kept_lines).append(line)
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
    typedefs = "\n".join(
        [tagged] * bool(tagged)
        + [f"typedef struct {name} {name};" for name in [*order, *plain]]
        # After the structs: one of these may name a struct in its parameters.
        + function_types
    )
    # A class holding another must be defined after it: C needs the complete
    # type to lay out the field. Emitting in source order put `Car` before
    # `Engine` whenever that is how they were written.
    declarations = "\n".join(
        [*plain_bodies]
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

    # `M::twice(21)` - a static member function is reached by the class's
    # name, which is a qualifier and not a namespace, so nothing else strips
    # it. There is no object to pass, which is what makes it static.
    for name in order:
        for method in classes[name].methods:
            if not method.shared or not method.name:
                continue
            spelled = _c_name(name, method.name, _suffix_of(name, method, classes))
            remainder = _map_code(
                remainder,
                lambda part, o=name, m=method.name, s=spelled: re.sub(
                    rf"\b{re.escape(o)}\s*::\s*{re.escape(m)}\b", s, part
                ),
            )

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
        definitions += "\n" + "\n".join(
            _emit_heap_helpers(classes[name], classes) for name in order
        )
    return f"{head}\n{typedefs}\n{declarations}\n\n{definitions}\n\n{rewritten}\n"






#: `new T`, `new T(a, b)` and `new T[n]`. The type is read as a name so a
#: qualified one has already been flattened by the time this runs.
#: The stars are part of the type: `new T[n]` inside a container of pointers
#: reads `new Row *[n]` once T has been substituted, and taken without them
#: the `*[n]` was left behind for the C compiler to choke on.
_NEW = re.compile(r"\bnew\s+([A-Za-z_]\w*(?:\s*\*)*)\s*(\[|\()?")
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
        arity = _arity(constructor.parameters)
        suffix = arity if len(constructors) > 1 else None
        parameters = constructor.parameters or "void"
        passed = ", ".join(
            _parameter_name(part) for part in _split_arguments(constructor.parameters)
            if part.strip()
        )
        ctor = _c_name(name, "", _suffix_of(name, constructor, classes))
        made.append(
            f"static struct {name} *{_c_name(name, 'new', suffix)}({parameters}) {{\n"
            f"    struct {name} *__p = (struct {name} *)malloc({size});\n"
            f"    if (__p == 0) return 0;\n"
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

    def place(name: str, guard: "set[str]") -> None:
        if name in seen or name not in classes:
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
            continue
        body = _map_code(
            body,
            lambda part, n=name: re.sub(
                rf"(?<![.\w>&]){re.escape(n)}\b(?!\s*[\w(])", f"(*{n})", part
            ),
        )
    return body


#: A function defined at the top level: `int read(const Box &b) {`.
_DEFINITION = re.compile(r"\b([A-Za-z_][\w\s*]*?)\b([A-Za-z_]\w*)\s*\(([^;{}()]*)\)\s*\{")


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
    return found


def _address_reference_arguments(
    text: str,
    signatures: "dict[str, list[int]]",
    classes: "dict[str, Class]" = {},
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
    #: How many parameters each function was written with. A call carrying one
    #: more than that has already been given the caller's space at the front,
    #: and every position the author wrote has moved along by one. Which calls
    #: those are is not decidable here - two passes insert it, at different
    #: times - so it is read off the call itself.
    arity: "dict[str, int]" = {}
    for match in _DEFINITION.finditer(text):
        if _depth_at(text, match.end() - 1) != 0:
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
                # C++ converts a derived object to its base wherever one is
                # wanted; C makes you say so. The address is the same - the
                # base is the first member - so this is a cast and no more.
                wanted = wanted_types.get((name, index))
                held = _deduced_type(argument, text)
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
        returns_object = "*" not in spelled_result and returned in shapes
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
            inner = _return_through_pointer(inner)
        rewritten = _rewrite_body(
            inner, classes, known, pointers,
            unit=f"{unit}\n{head[head.rfind('(') + 1: head.rfind(')')]};"
            if opened >= 0 else unit,
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

def _depth_at(text: str, index: int) -> int:
    """How many braces are open just before `index`."""

    return text.count("{", 0, index) - text.count("}", 0, index)


def _rewrite_declarations(text: str, classes: "dict[str, Class]") -> str:
    """Outside any function: a class named as a type is all there is to do."""

    return _rewrite_types(text, classes)


#: `Counter shared;` written outside every function.
#: `static G global;` counts: how an object is stored is not part of what it
#: is, and a constructor still has to run for it. `extern` is deliberately
#: not among them - that names an object defined somewhere else, and
#: constructing it here would build it twice.
_FILE_SCOPE_OBJECT = re.compile(
    r"(?m)^[ \t]*(?:(?:static|const|volatile|inline)[ \t]+)*"
    r"([A-Za-z_]\w*)[ \t]+([A-Za-z_]\w*)[ \t]*(?:\(([^;{}()]*)\))?[ \t]*;"
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
        if _depth_at(text, match.start()) != 0:
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
    void clear() { count = 0; }
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
class ostream {
public:
    int __stream;
    ostream() { __stream = 1; }
    ostream &operator<<(int v) { printf("%d", v); return *this; }
    ostream &operator<<(long v) { printf("%ld", v); return *this; }
    ostream &operator<<(unsigned int v) { printf("%u", v); return *this; }
    ostream &operator<<(unsigned long v) { printf("%lu", v); return *this; }
    ostream &operator<<(double v) { printf("%g", v); return *this; }
    ostream &operator<<(char v) { printf("%c", v); return *this; }
    ostream &operator<<(const char *v) { printf("%s", v); return *this; }
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
};

template<typename T>
void swap(T &a, T &b) { T held; held = a; a = b; b = held; }
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
    const char *c_str() { return text.c_str(); }
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
/* Entries in one array, so an iterator is a pointer to one and `it->first`
   and `it->second` are ordinary member reads. */
template<typename K, typename V>
class map_entry {
public:
    K first;
    V second;
    map_entry() { }
};

/* Searched from the front, and kept in insertion order. That is not a
   red-black tree, and a program storing thousands of keys will notice; it is
   the same interface, and it needs nothing this subset does not have. */
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
    V &operator[](K key) {
        map_entry<K, V> *found;
        found = find(key);
        if (found != entries + used) { return found->second; }
        if (used == room) {
            if (room == 0) { reserve(8); } else { reserve(room * 2); }
        }
        entries[used].first = key;
        used = used + 1;
        return entries[used - 1].second;
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
/* The same shape as `map` with nothing on the other side. */
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
        if (find(value) != items + used) { return; }
        if (used == room) {
            if (room == 0) { reserve(8); } else { reserve(room * 2); }
        }
        items[used] = value;
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
    ~unique_ptr() { if (raw != 0) { delete raw; raw = 0; } }
    T *get() { return raw; }
    T *operator->() { return raw; }
    T &operator*() { return *raw; }
    int operator!() { return raw == 0; }
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
    void reset(T *p) { raw = p; }
};
}
"""

_SSTREAM_HEADER = r"""

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
    ostringstream &operator<<(double v) {
        int whole;
        int frac;
        whole = (int)v;
        frac = (int)((v - (double)whole) * 1000000.0);
        if (frac < 0) { frac = -frac; }
        held.append(to_string(whole));
        held.push_back('.');
        held.append(to_string(frac));
        return *this;
    }
};

typedef ostringstream stringstream;
}
"""

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

_BUILTIN_CPP_HEADERS = {
    "string": _STRING_HEADER,
    "map": _MAP_HEADER,
    # An unordered map is the same interface with no promise about order,
    # and this one keeps insertion order - which is a stronger promise than
    # C++ makes, so no program that is correct against C++ can tell.
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
    "numeric": _NUMERIC_HEADER,
    "stdexcept": _STDEXCEPT_HEADER,
    "filesystem": _FILESYSTEM_HEADER,
    "functional": _FUNCTIONAL_HEADER,
    "cassert": "#include <assert.h>\n",
    "climits": "#include <limits.h>\n",
    "cfloat": "#include <float.h>\n",
    "cctype": "#include <ctype.h>\n",
    "cstdio": '#include <stdio.h>\n',
    "cstdlib": '#include <stdlib.h>\n',
    "cstring": '#include <string.h>\n',
    "cmath": '#include <math.h>\n',
    "cstdint": '#include <stdint.h>\n',
}

#: A quoted include names a file of this project; an angled one names a header
#: py2bin ships, which is C already and left for the preprocessor.
_LOCAL_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"[ \t]*$', re.M)

#: Suffixes that mean C++ rather than C.
CPP_SUFFIXES = (".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx")


def is_cpp(path: Path) -> bool:
    return path.suffix.lower() in CPP_SUFFIXES


def inline_local_includes(
    path: Path,
    include_dirs: "tuple[str, ...]" = (),
    seen: "set[Path] | None" = None,
    seen_headers: "set[str] | None" = None,
) -> str:
    """Paste this project's own headers in, so the translator can see them.

    A class is usually declared in a header and used from a source file. The
    translator works on text and runs before the preprocessor, so without this
    it would be handed a file that mentions a class it has never seen and
    would leave the calls alone - producing C that does not compile, blaming a
    line the user did write for a declaration they put somewhere else.

    Only quoted includes are pasted. An angled one names a header py2bin
    ships, which is C already, and the preprocessor handles it as it always
    has. A header already pasted is skipped rather than pasted twice, which is
    what an include guard would have done anyway.
    """

    seen = set() if seen is None else seen
    # One copy of a supplied header per translation unit, whatever how many
    # files ask for it - the same job an include guard does.
    seen_headers = set() if seen_headers is None else seen_headers
    settled = path.resolve()
    if settled in seen:
        return ""
    seen.add(settled)
    text = path.read_text(encoding="utf-8", errors="replace")

    def paste(match: "re.Match[str]") -> str:
        named = match.group(1)
        for folder in (path.parent, *(Path(d) for d in include_dirs)):
            candidate = folder / named
            if candidate.is_file():
                return inline_local_includes(
                    candidate, include_dirs, seen, seen_headers
                )
        # Not ours to paste; leave it for the preprocessor to fail on clearly.
        return match.group(0)

    text = _LOCAL_INCLUDE.sub(paste, text)

    def paste_builtin(match: "re.Match[str]") -> str:
        named = match.group(1)
        supplied = _BUILTIN_CPP_HEADERS.get(named)
        if supplied is None:
            return match.group(0)
        if named in seen_headers:
            return ""
        seen_headers.add(named)
        # One of these may include another - <filesystem> is written on top
        # of <string> - so what is pasted is pasted again. Without it the
        # inner include survived into the C, and the compiler reported a
        # missing header the user never wrote.
        return re.sub(r'#\s*include\s*<([^>]+)>', paste_builtin, supplied)

    return re.sub(r'#\s*include\s*<([^>]+)>', paste_builtin, text)


def translate_project(
    path: Path, include_dirs: "tuple[str, ...]" = ()
) -> str:
    """The C for one C++ source, with this project's headers pasted in."""

    return translate(inline_local_includes(path, include_dirs), str(path))


def translate_unity(
    sources: "tuple[Path, ...]", include_dirs: "tuple[str, ...]" = ()
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
    pieces = [
        inline_local_includes(path, include_dirs, seen, shared) for path in sources
    ]
    joined = "\n".join(pieces)
    return translate(joined, str(sources[0]) if sources else "<c++>")
