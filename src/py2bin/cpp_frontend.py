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


def _this_qualified(body: str, owner: Class, classes: "dict[str, Class]") -> str:
    """Point bare member names at `this`, the way C++ resolves them.

    Inherited names count: the base is embedded as the first member, so a
    name the base declares is reached through it.
    """

    names = dict.fromkeys(owner.field_names(), "")
    base = owner.base
    while base and base in classes:
        for name in classes[base].field_names():
            names.setdefault(name, "__base.")
        base = classes[base].base

    def replace(match: "re.Match[str]") -> str:
        word = match.group(0)
        if word not in names:
            return word
        start = match.start()
        # Not after `.` or `->`, which already name an object, and not a
        # member of some other struct.
        before = body[:start].rstrip()
        if before.endswith(".") or before.endswith("->"):
            return word
        return f"this->{names[word]}{word}"

    return _WORD.sub(replace, body)


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
    body = _this_qualified(method.body, found, classes)
    body = _bare_method_calls(body, found, classes)
    body = _rewrite_body(body, classes, {"this": found.name}, pointers={"this"})
    returns = "void" if method.name in ("", "~") else method.returns
    return f"static {returns} {name}({parameters}) {body}"



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

    return _WORD.sub(replace, text)


#: `Vec v(1, 2);` and `Vec v;` - an object with automatic storage.
_OBJECT = re.compile(r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(\(([^;{}]*)\))?\s*;")
#: `Vec *p = ...;`
_OBJECT_POINTER = re.compile(r"\b([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)")


def _rewrite_body(
    body: str,
    classes: "dict[str, Class]",
    known: "dict[str, str]",
    pointers: "set[str]" = frozenset(),
) -> str:
    """Rewrite declarations and calls inside one function body.

    `known` maps a variable to the class it holds; `pointers` says which of
    those are pointers, because a pointer is already the address a method
    wants and an object has to have one taken.
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
    for variable in sorted(known, key=len, reverse=True):
        holds = known[variable]
        arrow = "->" if variable in pointers else r"\."
        address = variable if variable in pointers else f"&{variable}"
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
    body = _close_with_destructors(body, destroyed, known, classes)
    return body


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
        return source

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

    declarations = "\n".join(_emit_class(classes[name], classes) for name in order)
    definitions = "\n".join(_emit_methods(classes[name], classes) for name in order)
    rewritten = _rewrite_body(remainder, classes, {})
    head = "\n".join(directives)
    return f"{head}\n{declarations}\n\n{definitions}\n\n{rewritten}\n"


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
