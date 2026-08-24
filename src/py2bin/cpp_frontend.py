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
}
#: Longest first, so `<=` is not read as `<`.
_OPERATOR_SYMBOLS = sorted(_OPERATOR_NAMES, key=len, reverse=True)


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


@dataclass
class Class:
    name: str
    base: str | None = None
    members: list[Member] = field(default_factory=list)
    methods: list[Method] = field(default_factory=list)

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
    """The index just past the brace that closes the one at `opening`."""

    depth = 0
    index = opening
    while index < len(text):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise ValueError("unbalanced braces")


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


_CLASS_HEAD = re.compile(
    r"\b(class|struct)\s+([A-Za-z_]\w*)\s*(?::\s*(?:public|private|protected)?\s*"
    r"([A-Za-z_]\w*)\s*)?\{"
)


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
            method = _method_from(head, body[brace:close], filename, at + index)
            found.methods.append(method)
            index = close
            continue
        if statement_end < 0:
            break
        declaration = body[index:statement_end].strip()
        if declaration:
            if "(" in declaration:
                # A member function declared here and defined outside.
                found.methods.append(
                    _method_from(declaration, "", filename, at + index)
                )
            else:
                found.members.append(_member_from(declaration, filename, at + index))
        index = statement_end + 1
    return found


def _member_from(declaration: str, filename: str, at: int) -> Member:
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
    head = re.sub(r"\b(override|final)\b\s*$", "", head).strip()
    pure = False
    if re.search(r"=\s*0\s*$", head):
        pure = True
        head = re.sub(r"=\s*0\s*$", "", head).strip()

    def decorated(method: Method) -> Method:
        method.virtual = virtual
        method.pure = pure
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
        return decorated(Method(
            _OPERATOR_NAMES[symbol], returns or "void", parameters, body, at_line
        ))
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
_NAMESPACE = re.compile(r"\bnamespace\s*([A-Za-z_]\w*)?\s*\{")
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
        return None
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
        return _declared_return(text, None, plain.group(1))
    return None


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


def _expand_lambdas(text: str, filename: str) -> str:
    """Turn each lambda into a class, and its use into an object of it."""

    made: list[str] = []
    count = 0
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
        count += 1
        name = f"__py2bin_lambda_{count}"
        captures, parameters, declared = found.groups()
        try:
            closing = _matching(text, found.end() - 1)
        except ValueError:
            raise CppTranslationError(
                filename, _line_of(text, found.start()), "a lambda is not closed"
            ) from None
        body = text[found.end() - 1: closing]
        result = (declared or "").strip() or _lambda_result(body, parameters, text)
        held = _lambda_captures(captures, text, found.start(), filename)
        members = "".join(f"    {spelled} {variable};\n" for variable, spelled in held)
        made.append(
            f"class {name} {{\npublic:\n{members}"
            f"    {name}() {{ }}\n"
            f"    {result} operator()({parameters}) {body}\n}};\n"
        )
        # Where it is used: a declaration of one, and a member per capture.
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
        setup = "".join(f" {holder}.{v} = {v};" for v, _s in held)
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
    return "".join(made) + text


def _looks_like_an_index(text: str, found: "re.Match[str]") -> bool:
    """Whether `[...](...)  {` is really a subscript rather than a lambda.

    `a[i](x) { ... }` is not C++, so the only way to be fooled is a subscript
    on something callable followed by a block - which does not happen. What
    does happen is a name immediately before the bracket, which a lambda
    never has.
    """

    before = text[:found.start()].rstrip()
    return bool(before) and (before[-1].isalnum() or before[-1] in "_)]")


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


def _lambda_captures(
    captures: str, text: str, at: int, filename: str
) -> "list[tuple[str, str]]":
    """Each captured name and the type it is held as."""

    spelled = captures.strip()
    if not spelled:
        return []
    if spelled in ("=", "&"):
        raise CppTranslationError(
            filename,
            _line_of(text, at),
            "a lambda that captures everything by writing `[=]` or `[&]` does "
            "not say what it captures, and py2bin writes a member per capture "
            "- name them, as in `[x, y]`",
        )
    held: list[tuple[str, str]] = []
    for part in _split_arguments(spelled):
        name = part.strip()
        if name.startswith("&"):
            raise CppTranslationError(
                filename,
                _line_of(text, at),
                f"a lambda capturing {name[1:].strip()!r} by reference outlives "
                "nothing here - py2bin copies each capture into a member, so "
                "write it as a copy",
            )
        if not name.isidentifier():
            raise CppTranslationError(
                filename, _line_of(text, at), f"cannot read the capture {name!r}"
            )
        found = _deduced_type(name, text[:at])
        held.append((name, found or "int"))
    return held

def _expand_templates(text: str, filename: str) -> str:
    """Replace every template with the copies this file actually asks for."""

    patterns: dict[str, tuple[list[tuple[str, bool]], str, str]] = {}
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

    if not patterns:
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

    made: "dict[str, str]" = {}
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
            known.add(name)
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


def _emit_one(
    found: Class, method: Method, classes: "dict[str, Class]", unit: str = ""
) -> str:
    name = _c_name(found.name, method.name, _suffix_of(found.name, method, classes))
    parameters = f"struct {found.name} *this"
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
    if method.parameters:
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
    body = _bare_method_calls(body, found, classes)
    known = {"this": found.name}
    for referenced, held in references.items():
        if held in classes:
            known[referenced] = held
    receivers: dict[str, str] = {}
    for holder, prefix in _reachable_members(found, classes):
        held = holder.ctype.replace("*", "").strip()
        if held not in classes:
            continue
        # Already qualified above, so the name in the text is `this->motor`.
        spelled = f"this->{prefix}{holder.name}"
        known[spelled] = held
        receivers[spelled] = f"&{spelled}"
    body = _rewrite_body(
        body,
        classes,
        known,
        pointers={"this", *(n for n, h in references.items() if h in classes)},
        receivers=receivers,
        unit=unit,
    )
    if copied:
        # Declared and then assigned, not initialised: py2bin's C takes
        # `o = *p;` and not `struct V o = *p;`.
        entry = " ".join(
            f"struct {held} {variable}; {variable} = *__by_value_{variable};"
            for held, variable in copied
        )
        opening = body.find("{")
        body = body[:opening + 1] + " " + entry + body[opening + 1:]
    if returned:
        body = _return_through_pointer(body)
    if method.name == "":
        body = _open_with_subobjects(body, found, classes)
    elif method.name == "~":
        body = _close_with_subobjects(body, found, classes)
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


def _open_with_subobjects(body: str, found: Class, classes: "dict[str, Class]") -> str:
    calls = []
    for held, address in _subobjects(found, classes):
        owner = _find_method(held, "", classes)
        if owner is None:
            continue
        if any(m.name == "" and m.parameters for m in classes[held].methods) and not any(
            m.name == "" and not m.parameters for m in classes[held].methods
        ):
            raise CppTranslationError(
                "<c++>", 0,
                f"{found.name} holds a {held}, whose only constructor takes "
                f"arguments. C++ would name it in an initialiser list, which "
                f"this subset does not read - give {held} a constructor taking "
                f"nothing, or hold a pointer",
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

def _bare_method_calls(body: str, owner: Class, classes: "dict[str, Class]") -> str:
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
            _dispatched_here(owner.name, method, classes, provider, reached, body),
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
        if word in classes:
            return f"struct {word}"
        return word

    return _map_code(text, lambda part: _WORD.sub(replace, part))


#: `Vec v(1, 2);` and `Vec v;` - an object with automatic storage.
_OBJECT = re.compile(r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(\(([^;{}]*)\))?\s*;")
#: `Vec *p = ...;`
_OBJECT_POINTER = re.compile(
    r"(?<!struct )\b([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)"
)
#: `Vec c = a.add(b);` - a declaration whose value comes from a method that
#: returns an object. The space is the caller's to provide.
_VALUE_INIT = re.compile(
    r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=\s*"
    r"([A-Za-z_]\w*)\s*(\.|->)\s*([A-Za-z_]\w*)\s*\(([^;]*)\)\s*;"
)

#: `Vec bank[3];` - an array of objects, each of which C++ default-constructs.
_OBJECT_ARRAY = re.compile(r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;")
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

    def assigned(match: "re.Match[str]") -> str:
        variable, made = match.group(1), match.group(2)
        wanted = known.get(variable) if variable in pointers else None
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
#: The same, assigning to a pointer declared earlier.
_ASSIGNED_FROM_NEW = re.compile(
    r"\b([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)__new(?:__\d+|_array)?\s*\("
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


def _hoist_value_returns(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    counter: "list[int]",
) -> str:
    """`f.filename().c_str()` becomes a temporary and two calls.

    A method returning an object by value returns nothing in the C - the
    caller provides the space and the callee writes through a hidden pointer -
    so its result is not an expression that anything can be called on. C++
    calls that a temporary, and this writes the temporary out.

    Done here, on the C++, so everything after sees an ordinary object with
    an ordinary name and needs to know nothing about where it came from.
    """

    if not known:
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
            declared = _method_named_returning(owner, method, classes)
            if declared is None:
                continue
            close = _closing_paren(body, match.end() - 1)
            if close < 0:
                continue
            # Anything except the one form that is already handled: a
            # declaration whose whole initialiser is this call, where the
            # caller's own space is the variable being declared. Everywhere
            # else - assigned to something that exists, called on, passed as
            # an argument - there is no space to write through until one is
            # made, so one is.
            begins = _statement_start(body, match.start())
            while begins < len(body) and body[begins] in " \t\n":
                begins += 1
            if _VALUE_INIT.match(body, begins):
                continue
            found = (match, close, declared)
            break
        if found is None:
            return body
        match, close, held = found
        counter[0] += 1
        name = f"__py2bin_value_{counter[0]}"
        known[name] = held
        start = _statement_start(body, match.start())
        call = body[match.start(): close + 1]
        body = (
            body[:start]
            + f"{held} {name} = {call}; "
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

def _rewrite_body(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]" = frozenset(),
    receivers: "dict[str, str] | None" = None,
    inherited_arrays: "dict[str, str] | None" = None,
    unit: str = "",
    enclosing: "list[tuple[str, str]]" = (),
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
    body = _rewrite_new(body, classes, scope())

    # `f.filename().c_str()`: a call on what a value return handed back. The
    # declarations have not been read yet - they are rewritten below, and
    # this has to run before that - so what this body declares is scanned for
    # first, without touching it.
    body = _hoist_value_returns(
        body, classes, {**_declared_objects(body, classes), **known}, [0]
    )

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

    body = _OBJECT_ARRAY.sub(declare_array, body)
    body = _OBJECT.sub(declare, body)

    def declare_pointer(match: "re.Match[str]") -> str:
        type_name, variable = match.groups()
        if type_name not in classes:
            return match.group(0)
        known[variable] = type_name
        pointers.add(variable)
        return f"struct {type_name} *{variable}"

    pointer_arrays: dict[str, str] = {}

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

    body = _rewrite_operators(body, classes, known, pointers)
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
                _dispatched(holds, method, classes, reached, owner, scope()),
                reached,
            )
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
        )
        for inner in blocks
    ]
    body = _close_with_destructors(body, destroyed, known, classes, enclosing)
    return _restore_nested(body, rewritten_blocks)




#: Stands in for a nested block while the enclosing scope is rewritten, so a
#: declaration inside one is not mistaken for a declaration in this one.
_BLOCK_MARK = "\x00py2bin_block_%d\x00"


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
            if depth == 2:
                closing = _matching(body, index)
                out.append(_BLOCK_MARK % len(blocks))
                blocks.append(body[index:closing])
                depth -= 1
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
        pattern = re.compile(
            rf"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*"
            rf"{re.escape(symbol)}\s*([A-Za-z_]\w*)\s*;"
        )

        def replace(match: "re.Match[str]", n=name) -> str:
            type_name, variable, left, right = match.groups()
            if type_name not in classes or left not in known:
                return match.group(0)
            owner = _find_method(known[left], n, classes)
            if owner is None or not _method_named(owner, n, classes, True):
                return match.group(0)
            added[variable] = type_name
            address = left if left in pointers else f"&{left}"
            passed = f"&{right}" if right in known and right not in pointers else right
            return (
                f"struct {type_name} {variable}; "
                f"{_c_name(owner, n)}({address}, &{variable}, {passed});"
            )

        body = _map_code(body, lambda part: pattern.sub(replace, part))
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


def _rewrite_operators(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]",
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
            pattern = (
                rf"\b{re.escape(variable)}\s*{re.escape(symbol)}\s*"
                rf"([A-Za-z_]\w*)\b"
            )

            def replace(match: "re.Match[str]", o=owner, n=name, a=address) -> str:
                right = match.group(1)
                passed = f"&{right}" if right in known and right not in pointers else right
                return f"{_c_name(o, n)}({a}, {passed})"

            body = _map_code(body, lambda part: re.sub(pattern, replace, part))
    # `d(5)` where `d` is an object with a call operator. A name that holds
    # an object is never a function, so a call on it is that operator and
    # nothing else - which is what makes this safe to do by name.
    for variable in sorted(known, key=len, reverse=True):
        owner = _find_method(known[variable], "op_call", classes)
        if owner is None:
            continue
        address = variable if variable in pointers else f"&{variable}"
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
        address = variable if variable in pointers else f"&{variable}"
        pattern = rf"\b{re.escape(variable)}\s*\[([^\]]*)\]"
        body = _map_code(
            body,
            lambda part, o=owner, a=address, p=pattern: re.sub(
                p,
                lambda m: f"{_c_name(o, 'op_index')}({a}, {m.group(1)})",
                part,
            ),
        )
    return body

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
        chosen = chosen.replace("__I__", found.group(1))
        reached = (
            receiver.replace("__I__", found.group(1))
            if receiver
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
        chosen = chosen.replace("__I__", found.group(1))
        out.append(body[at:found.start()])
        out.append(f"{chosen}(&{variable}[{found.group(1)}]{separator}")
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
):
    """`_dispatch` where it applies, otherwise the direct name."""

    virtual = _dispatch(holds, method, classes, receiver)
    if virtual is None:
        return _name_for(static, method, classes, text)
    direct = _name_for(static, method, classes, text)

    def chosen(given: "list[str]") -> str:
        through = virtual(given)
        if through:
            return through
        return direct if isinstance(direct, str) else direct(given)

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
    r"(?<![#\w])(?:([A-Za-z_][\w \t*]*?)\s+)?\b([A-Za-z_]\w*)::(~?[A-Za-z_]\w*)\s*"
    r"\(([^)]*)\)\s*\{"
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
    before_patterns = text
    # Before the templates: a lambda becomes a class, and a class may be a
    # template argument.
    text = _expand_lambdas(text, filename)
    text = _expand_templates(text, filename)
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
        typedefs = "\n".join(f"typedef struct {name} {name};" for name in plain)
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
        returns, owner, method, parameters = out.groups()
        returns = (returns or "").strip()
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

    # A polymorphic class always gets a constructor, even one that does
    # nothing visible: something has to install the table, and a derived class
    # that borrowed its base's constructor would be left pointing at the
    # base's - so its objects would answer as the base.
    for name in order:
        found = classes[name]
        if not _is_polymorphic(name, classes):
            continue
        if not any(m.name == "" and not m.parameters for m in found.methods):
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
    typedefs = "\n".join(
        f"typedef struct {name} {name};" for name in [*order, *plain]
    )
    # A class holding another must be defined after it: C needs the complete
    # type to lay out the field. Emitting in source order put `Car` before
    # `Engine` whenever that is how they were written.
    declarations = "\n".join(
        _emit_class(classes[name], classes) for name in _by_dependency(order, classes)
    )
    tables = _emit_vtables(order, classes)
    if tables:
        declarations += "\n" + tables
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
    remainder = _address_reference_arguments(
        remainder, _function_signatures(remainder, classes)
    )
    rewritten = _rewrite_functions(remainder, classes, made, scope)
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
_DELETE = re.compile(r"\bdelete\s*(\[\s*\])?\s*([^;]+);")

#: The header on a `new[]` block: the element count, so `delete[]` knows how
#: many destructors to run. Sixteen bytes rather than eight, because malloc
#: hands back 16-byte-aligned memory and the array has to stay that way.
_ARRAY_COOKIE = 16


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
                close = _closing_paren(body, found.end() - 1)
                if body[found.end(): close].strip():
                    raise CppTranslationError(
                        "<c++>",
                        _line_of(body, found.start()),
                        f"new {type_name}(value) has to store through the "
                        f"pointer as well as return it, which is not one C "
                        f"expression; write `{type_name} *p = new {type_name}; "
                        f"*p = value;`",
                    )
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
    text: str, signatures: "dict[str, list[int]]"
) -> str:
    """`bump(a, 9)` becomes `bump(&a, 9)` where the parameter is a reference.

    Only where the argument is something that has an address - a name, a
    member, an element. An expression has no address, and C++ would be making
    a temporary to bind a `const&` to; saying so beats taking the address of
    something that is not there.
    """

    if not signatures:
        return text
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
            for index in positions:
                if index >= len(parts):
                    continue
                argument = parts[index].strip()
                if argument.startswith("&") or not _has_an_address(argument):
                    continue
                parts[index] = f"&{argument}"
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


def _rewrite_functions(
    text: str,
    classes: "dict[str, Class]",
    outer: "dict[str, str] | None" = None,
    unit: str = "",
) -> str:
    """Rewrite each function on its own, because a scope is not the file.

    Done to the whole remainder at once, a variable declared in one function
    was in scope for every later one, and - worse - its destructor was placed
    at the end of the *last* function in the file rather than its own. The
    compiler then reported a name that is not declared, pointing at a line in
    somebody else's function.
    """

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
        copied = _by_value_objects(inside, classes)
        for held, variable in copied:
            # Without the `struct`: the declaration rewriter below adds one,
            # and two is not C.
            head = re.sub(
                rf"\b{re.escape(held)}\s+{re.escape(variable)}\b",
                f"{held} *__by_value_{variable}",
                head,
                count=1,
            )
        known, pointers = _parameters_of(head, classes)
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
        rewritten = _rewrite_body(inner, classes, known, pointers, unit=unit)
        if copied:
            # After the body is rewritten, not before: this text is already C,
            # and a `struct V v;` put in ahead of the declaration pass reads
            # as a new object to construct - so the copy ran the constructor
            # over what it was about to be handed. Declared and then assigned
            # rather than initialised, because py2bin's C takes `o = *p;` and
            # not `struct V o = *p;`.
            entry = " ".join(
                f"struct {held} {variable}; {variable} = *__by_value_{variable};"
                for held, variable in copied
            )
            spot = rewritten.find("{")
            rewritten = rewritten[:spot + 1] + " " + entry + rewritten[spot + 1:]
        out.append(_rewrite_declarations(head, classes))
        out.append(rewritten)
        at = closing
    out.append(_rewrite_declarations(text[at:], classes))
    return "".join(out)



def _parameters_of(
    head: str, classes: "dict[str, Class]"
) -> "tuple[dict[str, str], set[str]]":
    """The class-typed parameters of a function, which its body can call on.

    A parameter is declared in the head and used in the body, and the body is
    all the rewriter is handed - so `p->sum()` on an `Inline *p` was left
    alone and the compiler reported a struct with no such member.
    """

    opening = head.rfind("(")
    if opening < 0:
        return {}, set()
    known: dict[str, str] = {}
    pointers: set[str] = set()
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
    return known, pointers

def _depth_at(text: str, index: int) -> int:
    """How many braces are open just before `index`."""

    return text.count("{", 0, index) - text.count("}", 0, index)


def _rewrite_declarations(text: str, classes: "dict[str, Class]") -> str:
    """Outside any function: a class named as a type is all there is to do."""

    return _rewrite_types(text, classes)


#: `Counter shared;` written outside every function.
_FILE_SCOPE_OBJECT = re.compile(
    r"(?m)^[ \t]*([A-Za-z_]\w*)[ \t]+([A-Za-z_]\w*)[ \t]*;"
)


def _file_scope_objects(
    text: str, classes: "dict[str, Class]"
) -> "dict[str, str]":
    """Objects declared outside any function, and what class each holds.

    Their methods are reachable from every function in the file, so every body
    has to be rewritten knowing about them - without this, `shared.bump()` was
    left as C++ and the compiler reported a struct with no such member.
    """

    found: dict[str, str] = {}
    for match in _FILE_SCOPE_OBJECT.finditer(text):
        if _depth_at(text, match.start()) != 0:
            continue
        if match.group(1) in classes:
            found[match.group(2)] = match.group(1)
    return found


def _construct_before_main(text: str, made: "dict[str, str]", classes) -> str:
    """Run each file-scope object's constructor at the top of `main`.

    C++ builds them before `main` runs and C has no place to put that, so the
    first thing `main` does is what C++ had already done. A program that
    reaches one of them from another static initialiser would see the
    difference; there is no such thing here, because C has no static
    initialiser that can call anything.
    """

    calls = []
    for variable, held in made.items():
        owner = _find_method(held, "", classes)
        if owner is None:
            continue
        calls.append(
            f"{_c_name(owner, '', _call_suffix(owner, '', classes, []))}(&{variable});"
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
    void append(string o) {
        int j; j = 0;
        while (j < o.len && len + j < 255) { buf[len + j] = o.buf[j]; j = j + 1; }
        len = len + j; buf[len] = 0;
    }
    string operator+(string o) {
        string r; int i; int j;
        for (i = 0; i < len; i++) { r.buf[i] = buf[i]; }
        for (j = 0; j < o.len && len + j < 255; j++) { r.buf[len + j] = o.buf[j]; }
        r.len = len + j; r.buf[r.len] = 0;
        return r;
    }
    int operator==(string o) {
        int i;
        if (len != o.len) { return 0; }
        for (i = 0; i < len; i++) { if (buf[i] != o.buf[i]) { return 0; } }
        return 1;
    }
    int operator!=(string o) { int same; same = 0; return same; }
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
    void clear() { count = 0; }
    void reserve(unsigned long want) {
        unsigned long i;
        T *fresh;
        if (want <= room) { return; }
        fresh = new T[want];
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
    T *begin() { return items; }
    T *end() { return items + count; }
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
void swap(T &a, T &b) { T held; held = a; a = b; b = held; }

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
    T held;
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
    T held;
    while (1) {
        child = root * 2 + 1;
        if (child >= span) { return; }
        if (child + 1 < span) { if (base[child] < base[child + 1]) { child = child + 1; } }
        if (base[child] < base[root]) { return; }
        if (base[child] == base[root]) { return; }
        held = base[root]; base[root] = base[child]; base[child] = held;
        root = child;
    }
}

template<typename T, typename C>
void __sift_by(T *base, long root, long span, C less_than) {
    long child;
    T held;
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
    T held;
    span = last - first;
    if (span < 2) { return; }
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
    T held;
    span = last - first;
    if (span < 2) { return; }
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

_BUILTIN_CPP_HEADERS = {
    "string": _STRING_HEADER,
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
