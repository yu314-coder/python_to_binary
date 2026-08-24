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
_REFUSED = (
    ("template", "templates"),
    ("virtual", "virtual functions"),
    ("operator", "operator overloading"),
    ("throw", "exceptions"),
    ("catch", "exceptions"),
    ("namespace", "namespaces"),
    ("new", "`new` - py2bin's C compiler has no malloc, so there is no heap"),
    ("delete", "`delete` - there is no heap to return an object to"),
)

_WORD = re.compile(r"\b[A-Za-z_]\w*\b")


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
    head = head.strip()
    open_paren = head.find("(")
    close_paren = head.rfind(")")
    if open_paren < 0 or close_paren < 0:
        raise CppTranslationError(filename, at, f"cannot read the member {head!r}")
    before = head[:open_paren].strip()
    parameters = head[open_paren + 1: close_paren].strip()
    if parameters in ("void", ""):
        parameters = ""
    if before.startswith("~"):
        return Method("~", "void", parameters, body, at)
    words = before.replace("*", " * ").split()
    if len(words) == 1:
        # A constructor: the name is the class's own, with no return type.
        return Method("", "void", parameters, body, at)
    return Method(words[-1], " ".join(words[:-1]), parameters, body, at)


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
        if name in ("iostream", "string", "vector", "map", "memory", "algorithm"):
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
    true false""".split()
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
    for found in _DECLARED_HERE.finditer(_without_literals(body)):
        if found.group(1) in _NOT_A_TYPE:
            continue
        hidden.add(found.group(2))
    return hidden


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


def _emit_class(found: Class, classes: "dict[str, Class]") -> str:
    """The struct, and the free functions its methods become."""

    lines = [f"struct {found.name} {{"]
    if found.base:
        # First, so a pointer to the derived object is a pointer to the base.
        lines.append(f"    struct {found.base} __base;")
    for member in found.members:
        lines.append(f"    {member.ctype} {member.name}{member.array};")
    if not found.members and not found.base:
        # C has no empty struct; give it something so the type exists.
        lines.append("    int __empty;")
    lines.append("};")
    return "\n".join(lines)


def _emit_methods(found: Class, classes: "dict[str, Class]") -> str:
    out = []
    for method in found.methods:
        if not method.body:
            continue  # declared here, defined outside; emitted with that body
        out.append(_emit_one(found, method, classes))
    return "\n".join(out)


def _emit_one(found: Class, method: Method, classes: "dict[str, Class]") -> str:
    name = _c_name(found.name, method.name)
    parameters = f"struct {found.name} *this"
    if method.parameters:
        parameters += ", " + _rewrite_types(method.parameters, classes)
    body = _this_qualified(
        method.body, found, classes, _shadowing(method.body, method.parameters)
    )
    body = _bare_method_calls(body, found, classes)
    known = {"this": found.name}
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
        body, classes, known, pointers={"this"}, receivers=receivers
    )
    if method.name == "":
        body = _open_with_subobjects(body, found, classes)
    elif method.name == "~":
        body = _close_with_subobjects(body, found, classes)
    returns = "void" if method.name in ("", "~") else method.returns
    return f"static {returns} {name}({parameters}) {body}"


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
        calls.append(f"{_c_name(owner, '')}({address});")
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
        body = _rewrite_calls(body, pattern, _c_name(provider, method), reached)
    return body

def _c_name(class_name: str, method: str) -> str:
    if method == "":
        return f"{class_name}__ctor"
    if method == "~":
        return f"{class_name}__dtor"
    return f"{class_name}__{method}"


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
_OBJECT_POINTER = re.compile(r"\b([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)")
#: `Vec bank[3];` - an array of objects, each of which C++ default-constructs.
_OBJECT_ARRAY = re.compile(r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;")


def _rewrite_body(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]" = frozenset(),
    receivers: "dict[str, str] | None" = None,
    inherited_arrays: "dict[str, str] | None" = None,
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
            passed = f", {arguments}" if arguments else ""
            constructed += f" {_c_name(owner, '')}(&{variable}{passed});"
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
            f" {_c_name(owner, '')}(&{variable}[{index}]); }}"
        )
        return kept

    # A nested block is its own scope. Handled first, and on its own, because
    # C++ destroys what a block declared at the end of *that* block - and a
    # name declared there is not in scope after it either, so a destructor
    # placed at the end of the function named a variable C says is not there.
    body, blocks = _lift_nested(body)

    body = _OBJECT_ARRAY.sub(declare_array, body)
    body = _OBJECT.sub(declare, body)

    def declare_pointer(match: "re.Match[str]") -> str:
        type_name, variable = match.groups()
        if type_name not in classes:
            return match.group(0)
        known[variable] = type_name
        pointers.add(variable)
        return f"struct {type_name} *{variable}"

    body = _OBJECT_POINTER.sub(declare_pointer, body)

    # Calls, longest name first so `ab.m()` is not matched inside `xab.m()`.
    # An element of an array of objects is a receiver like any other; its
    # address is `&bank[i]`, whatever the index expression happens to be.
    for variable, holds in arrays.items():
        for method in _reachable_methods(holds, classes):
            owner = _find_method(holds, method, classes)
            if owner is None:
                continue
            pattern = (
                rf"\b{re.escape(variable)}\s*\[([^\]]*)\]\s*\.\s*"
                rf"{re.escape(method)}\s*\("
            )
            body = _rewrite_indexed(body, pattern, _c_name(owner, method), variable)

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
            body = _rewrite_calls(body, pattern, _c_name(owner, method), reached)
    # Now that this scope is known, each block is rewritten inside it.
    rewritten_blocks = [
        _rewrite_body(
            inner, classes, dict(known), set(pointers), dict(given), dict(arrays)
        )
        for inner in blocks
    ]
    body = _close_with_destructors(body, destroyed, known, classes)
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

def _rewrite_indexed(body: str, pattern: str, function: str, variable: str) -> str:
    """`bank[i].rate(` becomes `function(&bank[i]`, keeping the index."""

    out = []
    at = 0
    for found in re.finditer(pattern, body):
        rest = body[found.end():].lstrip()
        separator = "" if rest.startswith(")") else ", "
        out.append(body[at:found.start()])
        out.append(f"{function}(&{variable}[{found.group(1)}]{separator}")
        at = found.end()
    out.append(body[at:])
    return "".join(out)

def _rewrite_calls(body: str, pattern: str, function: str, receiver: str) -> str:
    """Turn each match into `function(receiver` plus a comma only if needed.

    The comma is the whole reason this is not a one-line `re.sub`: a method
    taking nothing becomes a call taking only the object, and `f(&v, )` is not
    C. Whether an argument follows is a property of the text after the match,
    which a replacement string cannot see.
    """

    out = []
    at = 0
    for found in re.finditer(pattern, body):
        rest = body[found.end():].lstrip()
        separator = "" if rest.startswith(")") else ", "
        out.append(body[at:found.start()])
        out.append(f"{function}({receiver}{separator}")
        at = found.end()
    out.append(body[at:])
    return "".join(out)


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


def _close_with_destructors(
    body: str,
    destroyed: "list[str]",
    known: "dict[str, str]",
    classes: "dict[str, Class]",
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

    if not destroyed:
        return body
    calls = "".join(
        f" {_c_name(_find_method(known[name], '~', classes), '~')}(&{name});"
        for name in reversed(destroyed)
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
        out.append(calls.strip() + " " + found.group(0))
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
    _refuse_unsupported(text, filename)

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
        if not plain:
            return source
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
    definitions = "\n".join(_emit_methods(classes[name], classes) for name in order)
    rewritten = _rewrite_functions(remainder, classes)
    head = "\n".join(directives)
    return f"{head}\n{typedefs}\n{declarations}\n\n{definitions}\n\n{rewritten}\n"



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


def _rewrite_functions(text: str, classes: "dict[str, Class]") -> str:
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
        known, pointers = _parameters_of(head, classes)
        out.append(_rewrite_declarations(head, classes))
        out.append(
            _rewrite_body(text[opening:closing], classes, known, pointers)
        )
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
    """Outside any function: a class named as a type is all there is to do.

    A file-scope object would want its constructor run before `main`, which C
    has no place to put, so only the type is rewritten here - the typedef makes
    the bare name legal and the object is left uninitialised, exactly as the
    same declaration in C would be.
    """

    return _rewrite_types(text, classes)

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
                return inline_local_includes(candidate, include_dirs, seen)
        # Not ours to paste; leave it for the preprocessor to fail on clearly.
        return match.group(0)

    return _LOCAL_INCLUDE.sub(paste, text)


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
    pieces = [inline_local_includes(path, include_dirs, seen) for path in sources]
    joined = "\n".join(pieces)
    return translate(joined, str(sources[0]) if sources else "<c++>")
