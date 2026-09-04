"""py2bin's C preprocessor: translation phases 1 to 4, in pure Python.

This is a real preprocessor, not a pattern match over ``#include`` lines. It
splices continued lines, replaces comments with a space, splits the result into
preprocessing tokens, and then interprets the directives: ``#define`` of both
object-like and function-like macros (variadic ones included), ``#undef``,
``#include`` of files it can find and read, the whole conditional family with a
real constant-expression evaluator, ``#error`` and ``#pragma once``. Macro
expansion follows the standard's algorithm, with hide sets, so a macro cannot
expand into itself, and arguments are substituted TEXTUALLY -- an argument that
is used twice in a replacement list really is written twice, exactly as C says,
and an argument used once is written once.

The result is a list of :class:`py2bin.c_frontend.Token`, so the front end that
parses C never sees a directive at all.

What is accepted
----------------
* ``#define NAME body`` and ``#define NAME(a, b) body``, with ``#`` (stringify)
  and ``##`` (paste), and ``#define NAME(a, ...) body`` with ``__VA_ARGS__``;
  a macro may be redefined only with an identical replacement list, as C
  requires;
* ``#undef NAME``;
* ``#include "file"`` and ``#include <file>``. A quoted name is looked for
  beside the file that included it and then along the search path; an angled
  name only along the search path. A name that is not found but is one of the
  standard headers py2bin knows (``stdio.h``, ``stdlib.h``, ``string.h``,
  ``stdint.h``, ``stddef.h``, ``stdbool.h``, ``limits.h``, ``math.h``,
  ``inttypes.h``) is served from py2bin's OWN built-in copy, which defines the
  macros of that header and nothing else -- the types and the ``printf`` those
  headers declare are built into the compiler. The form
  ``#include MACRO`` is expanded first and then read;
* ``#if``/``#ifdef``/``#ifndef``/``#elif``/``#else``/``#endif``, where the
  controlling expression is a real C constant expression evaluated in 64-bit
  ``intmax_t``/``uintmax_t`` arithmetic, with ``defined X``, ``defined(X)``,
  short-circuiting ``&&``, ``||`` and ``?:`` (an unevaluated operand is never
  diagnosed), and every remaining identifier replaced by ``0``;
* ``#error message``, which fails the compilation with that message;
* ``#pragma once``; ``#pragma pack``; and every other pragma, which is read
  and dropped, because C says an implementation ignores a pragma it does not
  recognise. The few that would make py2bin emit something else if it obeyed
  are refused by name instead -- see ``_CHANGES_THE_PROGRAM``;
* ``_Pragma("...")``, the operator spelling of ``#pragma``, which is what a
  macro has to use because a directive is not a token;
* the null directive ``#``;
* the predefined macros ``__FILE__``, ``__LINE__``, ``__STDC__``,
  ``__STDC_VERSION__``, ``__STDC_HOSTED__`` (0 -- py2bin has no hosted library),
  ``__py2bin__``, ``__py2bin_target__``, and one each of
  ``__py2bin_arm64__``/``__py2bin_x86_64__`` and
  ``__py2bin_darwin__``/``__py2bin_linux__``/``__py2bin_windows__``.
  ``__DATE__`` and ``__TIME__`` are the fixed ``"Jan  1 1970"`` and
  ``"00:00:00"``: C11 6.10.8.1 allows an implementation-defined constant when
  the date of translation is not available, and py2bin would rather compile the
  same source to the same bytes than read the clock;
* ``#warning``, whose text goes to standard error and whose whole purpose is
  not to stop the compilation -- so it does not.

What is rejected
----------------
``#line`` (py2bin reports the line the token was really written on),
a ``#pragma`` that would move a struct's members, rename a definition
or place one in a section of its own -- ignoring one of those is how a
compiler silently changes an ABI, so py2bin refuses it by name -- the GNU
``, ## __VA_ARGS__`` comma-deletion extension, ``#`` and ``##`` that survive
into the output, a ``##`` that pastes two tokens into something that is not one
token, and any header that cannot be found. Real system headers use extensions
this compiler does not have; py2bin will say so rather than guess.

Trigraphs (``??=``) and digraphs (``%:``, ``<:``) are not recognised, which is
what every C compiler does by default; spell the punctuation the normal way.
"""

from __future__ import annotations

import pathlib

import collections
import dataclasses
import re
import sys
from pathlib import Path

from .c_frontend import ARENA_BYTES, CCompileError, Lexer, Token

__all__ = ["preprocess"]


# --- preprocessing tokens ----------------------------------------------------


#: What `#pragma pack` is handed to the parser as. A name no program can
#: write, so nothing else can be mistaken for one. Stated in `c_frontend`,
#: which is what reads it, and checked against that here.
from .c_frontend import _PACK_MARKER


@dataclasses.dataclass(frozen=True, slots=True)
class PPToken:
    """One preprocessing token, with where it was written and what hides it.

    ``hides`` is the standard's hide set: the macros whose expansion this token
    came out of. A macro name is never expanded again while it is in the hide
    set of its own token, which is what stops ``#define A A`` from running
    forever without banning self-reference outright.
    """

    kind: str  # identifier | number | character | string | punctuator | other
    spelling: str
    line: int
    column: int
    origin: str
    spaced: bool = False
    hides: frozenset[str] = frozenset()


#: A placemarker: what an empty macro argument becomes when it is an operand of
#: ``##`` (C11 6.10.3.3p2). It is deleted once every paste is done.
_PLACEMARKER = "placemarker"

_PUNCTUATORS = tuple(
    sorted(
        (
            "...",
            "<<=",
            ">>=",
            "->",
            "++",
            "--",
            "<<",
            ">>",
            "<=",
            ">=",
            "==",
            "!=",
            "&&",
            "||",
            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "&=",
            "|=",
            "^=",
            "##",
            *"{}[]();,?:=+-*/%~!<>&|^.#",
        ),
        key=len,
        reverse=True,
    )
)


def _clean(source: str, origin: str) -> tuple[str, list[tuple[int, int]]]:
    """Run translation phases 1-3 and report where every character came from.

    Line continuations disappear, every line ending becomes ``\\n``, and each
    comment becomes a single space -- including a block comment that spanned
    lines, which is why a directive may be continued through one. The returned
    positions are the real line and column in ``source``, so a diagnostic about
    a spliced or comment-separated token still points at what was typed.
    """

    text: list[str] = []
    where: list[tuple[int, int]] = []
    index = 0
    line = 1
    column = 1
    total = len(source)
    state = "normal"
    started_at = (1, 1)
    quote = ""
    while index < total:
        character = source[index]
        if character == "\\":
            step = 0
            if source.startswith("\\\r\n", index):
                step = 3
            elif source.startswith("\\\n", index) or source.startswith("\\\r", index):
                step = 2
            if step:
                index += step
                line += 1
                column = 1
                continue
        if character in "\r\n":
            at = (line, column)
            index += 2 if source.startswith("\r\n", index) else 1
            line += 1
            column = 1
            if state == "block":
                continue
            if state in ("string", "character"):
                # An unterminated literal ends here; the scanner reports it.
                state = "normal"
            elif state == "line":
                state = "normal"
            text.append("\n")
            where.append(at)
            continue
        if state == "line":
            index += 1
            column += 1
            continue
        if state == "block":
            if source.startswith("*/", index):
                index += 2
                column += 2
                state = "normal"
                continue
            index += 1
            column += 1
            continue
        if state == "normal":
            if source.startswith("//", index):
                text.append(" ")
                where.append((line, column))
                state = "line"
                index += 2
                column += 2
                continue
            if source.startswith("/*", index):
                text.append(" ")
                where.append((line, column))
                started_at = (line, column)
                state = "block"
                index += 2
                column += 2
                continue
            if character in "\"'":
                state = "string" if character == '"' else "character"
                quote = character
        elif character == "\\" and index + 1 < total:
            # Inside a literal a backslash takes the next character with it, so
            # that '\"' does not end the string.
            text.append(character)
            where.append((line, column))
            index += 1
            column += 1
            character = source[index]
        elif character == quote:
            state = "normal"
        text.append(character)
        where.append((line, column))
        index += 1
        column += 1
    if state == "block":
        raise CCompileError(origin, started_at[0], started_at[1], "unterminated block comment")
    text.append("\n")
    where.append((line, column))
    return "".join(text), where


#: What a compiler on each platform defines, as far as a program that only
#: wants to know where it is being built needs. Deliberately not the whole set
#: - `__GNUC__` would promise extensions this compiler does not have.
_PLATFORM_MACROS = {
    "windows": (
        # _WIN32 is defined on 64-bit Windows too - it means "Windows", not
        # "32-bit" - and _WIN64 alongside it says which.
        "#define _WIN32 1",
        "#define _WIN64 1",
        "#define __WIN32__ 1",
        "#define WIN32 1",
    ),
    "darwin": (
        "#define __APPLE__ 1",
        "#define __MACH__ 1",
        "#define __unix__ 1",
        "#define __unix 1",
    ),
    "linux": (
        "#define __linux__ 1",
        "#define __linux 1",
        "#define __unix__ 1",
        "#define __unix 1",
    ),
}

#: And which machine. `_M_X64`/`_M_ARM64` are Microsoft's spellings, which
#: Windows code uses as often as the GNU ones.
_MACHINE_MACROS = {
    "x86_64": (
        "#define __x86_64__ 1",
        "#define __x86_64 1",
        "#define __amd64__ 1",
        "#define _M_X64 100",
        "#define _LP64 1",
    ),
    "arm64": (
        "#define __aarch64__ 1",
        "#define _M_ARM64 1",
        "#define _LP64 1",
    ),
}


def _respaced(tokens: "list[PPToken]") -> str:
    """Put a run of preprocessing tokens back together as it was written.

    Each token remembers whether whitespace came before it, which is what
    makes this possible - the alternative is one space between everything.
    """

    out: list[str] = []
    for index, token in enumerate(tokens):
        if index and token.spaced:
            out.append(" ")
        out.append(token.spelling)
    return "".join(out)


def _as_written(tokens: "list[PPToken]") -> str:
    """The tokens back as text, keeping the lines they came from.

    `_respaced` puts a run back together on one line, which is right for one
    directive and wrong for a file: the C++ translator reads line by line in
    places, and an error it reports names a line. So a token that came from
    further down the file starts a new line here, and a run of them that came
    from the same one stays on it.
    """

    out: "list[str]" = []
    line = 0
    for token in tokens:
        if token.line != line:
            if out:
                out.append("\n")
            line = token.line
        elif out and token.spaced:
            out.append(" ")
        out.append(token.spelling)
    return "".join(out) + "\n"


#: The headers of py2bin's own that declare a COM interface rather than a
#: plain C type. Under `__cplusplus` each of these gives classes, and the
#: translator has to see one to lay out anything derived from it.
_DECLARES_INTERFACES = frozenset({"unknwn.h", "objidl.h", "oaidl.h"})

#: Of those, the ones a branching header's run hands to the C++ stage to
#: paste its own spelling of - a class the translator can derive from -
#: rather than emitting the C shape read here. The C++ stage's <objidl.h>
#: is this file's own text, so nothing is lost in the hand-over.
_HANDED_TO_CPLUSPLUS = frozenset({"unknwn.h", "objidl.h", "oaidl.h"})


def as_cplusplus(
    named: str,
    directory: "Path | None",
    include_dirs: "tuple[str, ...]",
    target: "str | None",
    supplied: "set[str]",
    already: "set[str]",
    cplusplus: "frozenset[str]" = frozenset(),
) -> str:
    """One header, preprocessed as a C++ compiler would see it.

    A header that declares one thing or another according to a macro cannot
    be handed to the C++ translator as it stands: the translator runs before
    the preprocessor and has no `#if`, so it would read both branches. A
    generated COM header is exactly that - `#if defined(__cplusplus) &&
    !defined(CINTERFACE)` picks between classes and a table of function
    pointers - and a program that calls one the C++ way needs the first.

    So the preprocessor runs first, for this header alone, with
    `__cplusplus` defined. What comes back is one branch, with its own
    includes taken and its macros expanded, which is what a C++ compiler's
    parser is handed too.
    """

    engine = Preprocessor(include_dirs, target)
    engine.cplusplus_supplies = cplusplus
    # Everything pasted so far, except the header this run is for.
    engine.already_pasted = set(already) - {named}
    engine.run(
        "#define __cplusplus 201703L\n#define __py2bin_translating 1\n",
        "<c++>",
        None,
    )
    if (target or "").startswith("windows-"):
        # The plain-C types first, so that what is kept below does not
        # declare them. py2bin's own <unknwn.h> writes out GUID and HRESULT
        # only where <wtypes.h> has not - and <wtypes.h> is one of the ones
        # left to the other run, so it has to have been seen here for that
        # guard to hold.
        engine.run("#include <wtypes.h>\n", "<c++>", None)
    engine.run(f'#include "{named}"\n', f"<{named}>", directory)

    # py2bin's own headers were read for their macros - a generated header is
    # written in `STDMETHODCALLTYPE` and `MIDL_INTERFACE` and says nothing
    # without them. What they *declare* is another matter, and the two kinds
    # part company here.
    #
    # The ones that declare COM interfaces stay, because the translator is
    # about to read `struct ICoreWebView2 : public IUnknown` and a base it
    # cannot see is a base it cannot lay out. They are reported as supplied,
    # so the run that reads the rest of the program leaves them alone.
    #
    # The rest are plain C - types and prototypes - and are dropped and left
    # to that run entirely. Not reported, so it reads them itself, at the top
    # where it puts every directive. That is the order they have to be in:
    # <shellapi.h> asks for HINSTANCE, and the answer has to be above it.
    ours = set(engine._builtins_read)
    # One py2bin already pasted into this unit is not put in again: the pass
    # that reads C++ pastes each of its own headers once, and a program
    # reaching <wrl.h> has had <unknwn.h> from it before this header was
    # touched. Kept as well, IUnknown was declared twice and its table
    # written out twice.
    pasted = ours & already
    kept = (ours & _DECLARES_INTERFACES) - pasted
    dropped = ours - kept - pasted
    supplied.update(kept)
    handed = kept & _HANDED_TO_CPLUSPLUS
    already.update(kept - handed)
    # The search-path headers this run expanded are in the unit now too.
    already.update(engine.search_path_read)
    # The dropped ones are asked for by name, so the other run reads them -
    # at the top, where it puts every directive, which is above every use.
    # Dropped without asking, a generated header named a type nothing had
    # declared: `EventRegistrationToken` is in one of them and appears in
    # two thousand of its own signatures.
    # And the C++ ones this run met and could not answer, asked the same
    # way. The caller tells the two apart by the table each name is in.
    asked = "".join(
        f"#include <{name}>\n"
        # And the ones that declare COM interfaces: read here for their
        # macros, then handed to the C++ stage to paste its own spelling of,
        # ahead of this header. What this run emitted for them was the C
        # shape - a table of function pointers - and the translator reading
        # `IXMLDOMNode : public IDispatch` after it found no class of that
        # name unless the program had happened to include <oaidl.h> itself.
        for name in sorted(dropped | handed | engine.left_to_cplusplus)
    )
    return asked + _as_written(
        [
            item
            for item in engine.output
            if item.origin.strip("<>") not in dropped | pasted | handed
            and item.origin not in engine.pasted_origins
        ]
    )


def _scan(source: str, origin: str) -> list[list[PPToken]]:
    """Split cleaned source into logical lines of preprocessing tokens."""

    text, where = _clean(source, origin)
    lines: list[list[PPToken]] = []
    current: list[PPToken] = []
    index = 0
    total = len(text)
    spaced = False
    while index < total:
        character = text[index]
        if character == "\n":
            lines.append(current)
            current = []
            spaced = False
            index += 1
            continue
        if character.isspace():
            spaced = True
            index += 1
            continue
        line, column = where[index]
        if character.isalpha() or character == "_":
            start = index
            while index < total and (text[index].isalnum() or text[index] == "_"):
                index += 1
            name = text[start:index]
            if name in {"L", "u", "U", "u8"} and index < total and text[index] in "'\"":
                # A prefix belongs to the literal, not to an identifier that
                # happens to sit next to one. Kept in the spelling and handed
                # on whole: the compiler's own lexer reads the prefix again
                # and decides what the literal means, which is where knowing
                # the target - and so how wide a wchar_t is - belongs.
                quote = text[index]
                index += 1
                while index < total and text[index] != quote:
                    if text[index] == "\n":
                        break
                    index += 2 if text[index] == "\\" else 1
                if index >= total or text[index] != quote:
                    raise CCompileError(
                        origin,
                        line,
                        column,
                        "unterminated string literal"
                        if quote == '"'
                        else "unterminated character constant",
                    )
                index += 1
                current.append(
                    PPToken(
                        "string" if quote == '"' else "character",
                        text[start:index],
                        line,
                        column,
                        origin,
                        spaced,
                    )
                )
                spaced = False
                continue
            current.append(PPToken("identifier", name, line, column, origin, spaced))
            spaced = False
            continue
        if character.isdigit() or (
            character == "." and index + 1 < total and text[index + 1].isdigit()
        ):
            # A preprocessing number (C11 6.4.8) is deliberately looser than a
            # real constant: it is checked when it is converted, not here.
            start = index
            index += 1
            while index < total:
                following = text[index]
                if following in "eEpP" and index + 1 < total and text[index + 1] in "+-":
                    index += 2
                    continue
                if following.isalnum() or following in "._":
                    index += 1
                    continue
                break
            current.append(PPToken("number", text[start:index], line, column, origin, spaced))
            spaced = False
            continue
        if character in "\"'":
            start = index
            index += 1
            while index < total and text[index] != character:
                if text[index] == "\n":
                    break
                index += 2 if text[index] == "\\" else 1
            if index >= total or text[index] != character:
                raise CCompileError(
                    origin,
                    line,
                    column,
                    "unterminated string literal"
                    if character == '"'
                    else "unterminated character constant",
                )
            index += 1
            current.append(
                PPToken(
                    "string" if character == '"' else "character",
                    text[start:index],
                    line,
                    column,
                    origin,
                    spaced,
                )
            )
            spaced = False
            continue
        operator = next(
            (candidate for candidate in _PUNCTUATORS if text.startswith(candidate, index)),
            None,
        )
        if operator is not None:
            index += len(operator)
            current.append(PPToken("punctuator", operator, line, column, origin, spaced))
            spaced = False
            continue
        index += 1
        current.append(PPToken("other", character, line, column, origin, spaced))
        spaced = False
    if current:
        lines.append(current)
    return lines


# --- macros ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Macro:
    name: str
    parameters: tuple[str, ...] | None  # None for an object-like macro
    variadic: bool
    body: tuple[PPToken, ...]
    token: PPToken
    builtin: str = ""


@dataclasses.dataclass(slots=True)
class _Condition:
    """One live ``#if`` group."""

    outer: bool  # was the enclosing group emitting?
    taken: bool  # has a branch of this group been taken?
    active: bool  # is the current branch emitting?
    seen_else: bool
    token: PPToken
    #: What each branch of this group asked for, in order, and whether it
    #: held. Kept so a `#error` at the end of a chain can say why it was
    #: reached: a header that falls through every branch is telling you what
    #: it wanted, and reading that off the source is the whole answer.
    tried: "list[tuple[str, bool]]" = dataclasses.field(default_factory=list)


_MAXIMUM_INCLUDE_DEPTH = 64
#: A ceiling on expansion work, so a pathological macro fails instead of hanging.
_MAXIMUM_EXPANSIONS = 2_000_000

_PREDEFINED = frozenset(
    {
        "__FILE__",
        "__LINE__",
        "__DATE__",
        "__TIME__",
        "__STDC__",
        "__STDC_VERSION__",
        "__STDC_HOSTED__",
        "__py2bin__",
    }
)


class Preprocessor:
    def __init__(
        self,
        include_dirs: tuple[str, ...] = (),
        target: str | None = None,
    ):
        self.macros: dict[str, Macro] = {}
        self.include_dirs = [Path(item) for item in include_dirs]
        #: Standard headers already supplied. See the include path below.
        self._builtins_read: set[str] = set()
        #: Names one of py2bin's own headers defined, which a real platform
        #: header may redefine. They are defaults so a program that never
        #: reaches a fetched set still has them, not claims about how the
        #: platform spells them.
        self._defaults: set[str] = set()
        #: Whether the file being read is one py2bin supplies.
        self._reading_a_builtin = False
        #: Headers the C++ stage supplies and this run cannot - <string>,
        #: <filesystem> and the rest of the C++ standard library py2bin
        #: writes. Empty except when a single header is being read ahead of
        #: the translator, which is the only run that can hand them back.
        self.cplusplus_supplies: "frozenset[str]" = frozenset()
        #: Search-path headers the C++ stage has already pasted into the unit.
        #: A run that reads one branching header alone must not expand them
        #: again, or the unit holds two copies; and the ones it does expand
        #: are reported in `search_path_read`, so a later direct include of
        #: one of them is skipped in turn.
        self.already_pasted: "set[str]" = set()
        self.search_path_read: "set[str]" = set()
        self.pasted_origins: "set[str]" = set()
        #: The ones such a run met, to be asked of the C++ stage afterwards.
        self.left_to_cplusplus: "set[str]" = set()
        #: Whether this text is C the C++ translator wrote. It matters to
        #: `pack` alone: that translator reads a pack out of the text and
        #: carries it with the struct, so a pack it could not read there is
        #: one the classes have already been laid out without.
        self.translated_cplusplus = False
        self.output: list[PPToken] = []
        self.depth = 0
        self.once: set[str] = set()
        #: Which file each device-and-inode pair turned out to be, under the
        #: first name that reached it. Two spellings of one file have to come
        #: out as one origin here, or `#pragma once` cannot recognise the
        #: second one. Kept per run: an inode is reused once its file is gone.
        self._files_seen: dict[tuple[int, int], str] = {}
        self.expansions = 0
        self.priming = True
        self._predefine(target)
        self.priming = False

    # --- diagnostics ---

    @staticmethod
    def error(message: str, token: PPToken):
        raise CCompileError(token.origin, token.line, token.column, message)

    # --- the built-in macros ---

    def _predefine(self, target: str | None) -> None:
        origin = "<built-in>"
        for name in ("__FILE__", "__LINE__"):
            token = PPToken("identifier", name, 1, 1, origin)
            self.macros[name] = Macro(name, None, False, (), token, builtin=name)
        text = [
            "#define __STDC__ 1",
            "#define __STDC_VERSION__ 201112L",
            "#define __STDC_HOSTED__ 0",
            "#define __py2bin__ 1",
            # GCC's `__extension__` marks a construct as deliberately beyond
            # the standard and changes nothing else; mingw's headers write it
            # in front of `__int64`. Nothing here needs to hear it.
            "#define __extension__",
            # C11 6.10.8.1 lets an implementation supply a fixed date and time
            # when the real ones are not available, and py2bin says they are
            # not: a compiler that read the clock would give a different binary
            # every time it ran, and this one is reproducible on purpose.
            '#define __DATE__ "Jan  1 1970"',
            '#define __TIME__ "00:00:00"',
        ]
        if target:
            text.append(f'#define __py2bin_target__ "{target}"')
            system, _, machine = target.partition("-")
            if machine in ("arm64", "x86_64"):
                text.append(f"#define __py2bin_{machine}__ 1")
            if system in ("darwin", "linux", "windows"):
                text.append(f"#define __py2bin_{system}__ 1")
            # The names real code actually guards on. py2bin's own
            # `__py2bin_windows__` says the same thing, but no program in the
            # world is written against it - a file that picks its headers with
            # `#ifdef _WIN32` took the wrong branch on every target until
            # these were defined, and the failure was a missing header rather
            # than anything that pointed at the cause.
            text.extend(_PLATFORM_MACROS.get(system, ()))
            text.extend(_MACHINE_MACROS.get(machine, ()))
            # What a compiler tells a header about its own data model, in the
            # spelling GCC and clang settled on. These say nothing about
            # which compiler is reading - a header that writes
            # `__UINTPTR_TYPE__` in a prototype wants a type name and not an
            # extension - and a published C runtime does write one, in the
            # file the rest of the set includes. Windows is LLP64, so the
            # pointer-wide integer there is `long long` and not `long`.
            if machine in ("arm64", "x86_64"):
                wide = "long long" if system == "windows" else "long"
                text.extend(
                    (
                        f"#define __SIZE_TYPE__ unsigned {wide}",
                        f"#define __PTRDIFF_TYPE__ {wide}",
                        f"#define __INTPTR_TYPE__ {wide}",
                        f"#define __UINTPTR_TYPE__ unsigned {wide}",
                        "#define __WCHAR_TYPE__ "
                        + ("unsigned short" if system == "windows" else "int"),
                    )
                )
        self._file("\n".join(text) + "\n", origin, None)

    # --- running over a file ---

    def run(self, source: str, origin: str, directory: Path | None) -> None:
        self._file(source, origin, directory)

    def _file(self, source: str, origin: str, directory: Path | None) -> None:
        conditions: list[_Condition] = []
        run: list[PPToken] = []
        for line in _scan(source, origin):
            if not line:
                continue
            if line[0].kind == "punctuator" and line[0].spelling == "#":
                # A macro invocation never spans a directive, so everything
                # gathered so far can be expanded with the macros as they are.
                if run:
                    self.output.extend(self.expand(run))
                    run = []
                self._directive(line, conditions, directory)
                continue
            if conditions and not conditions[-1].active:
                continue
            run.extend(line)
        if run:
            self.output.extend(self.expand(run))
        if conditions:
            self.error("this #if was never closed with #endif", conditions[-1].token)

    def _directive(
        self,
        line: list[PPToken],
        conditions: list[_Condition],
        directory: Path | None,
    ) -> None:
        if len(line) == 1:
            return  # the null directive
        name_token = line[1]
        if name_token.kind != "identifier":
            if conditions and not conditions[-1].active:
                return
            self.error(
                f"expected a preprocessing directive, found {name_token.spelling!r}",
                name_token,
            )
        name = name_token.spelling
        rest = line[2:]
        active = conditions[-1].active if conditions else True
        if name in ("if", "ifdef", "ifndef"):
            if not active:
                # A skipped group's conditions are not evaluated at all; only
                # the nesting has to be tracked. 'outer' being false is what
                # keeps every branch of this group dark, whatever it says.
                conditions.append(_Condition(False, False, False, False, name_token))
                return
            value = self._condition(name, rest, name_token, directory)
            conditions.append(
                _Condition(
                    True, value, value, False, name_token,
                    [(f"#{name} {_respaced(rest)}".strip(), value)],
                )
            )
            return
        if name == "elif":
            if not conditions:
                self.error("#elif without a matching #if", name_token)
            condition = conditions[-1]
            if condition.seen_else:
                self.error("#elif after #else", name_token)
            if not condition.outer or condition.taken:
                condition.active = False
                if condition.outer:
                    condition.tried.append((f"#elif {_respaced(rest)}", False))
                return
            value = self._condition("if", rest, name_token, directory)
            condition.active = value
            condition.taken = condition.taken or value
            condition.tried.append((f"#elif {_respaced(rest)}", value))
            return
        if name == "else":
            if not conditions:
                self.error("#else without a matching #if", name_token)
            condition = conditions[-1]
            if condition.seen_else:
                self.error("#else after #else", name_token)
            if rest and condition.outer:
                self.error("#else takes no tokens", rest[0])
            condition.seen_else = True
            condition.active = condition.outer and not condition.taken
            condition.taken = condition.taken or condition.active
            return
        if name == "endif":
            if not conditions:
                self.error("#endif without a matching #if", name_token)
            if rest and conditions[-1].outer:
                self.error("#endif takes no tokens", rest[0])
            conditions.pop()
            return
        if not active:
            return
        if name == "define":
            self._define(rest, name_token)
            return
        if name == "undef":
            if not rest or rest[0].kind != "identifier":
                self.error("#undef needs a macro name", name_token)
            if len(rest) > 1:
                self.error("#undef takes one macro name", rest[1])
            if rest[0].spelling in _PREDEFINED:
                self.error(
                    f"{rest[0].spelling} is predefined and cannot be undefined",
                    rest[0],
                )
            self.macros.pop(rest[0].spelling, None)
            return
        if name == "include":
            self._include(rest, name_token, directory)
            return
        if name == "error":
            # Rebuilt with the spacing the author wrote, not one space per
            # token: `#error <windows.h> is for Windows` came back as
            # `< windows . h > is for Windows`, which is a worse message than
            # the one the author took the trouble to write.
            message = _respaced(rest)
            spelled = f"#error {message}" if message else "#error"
            spelled += self._why_here(conditions)
            self.error(spelled, name_token)
        if name == "pragma":
            self._pragma(rest, name_token)
            return
        if name == "line":
            self.error(
                "#line is not implemented; py2bin reports the line a token was "
                "really written on",
                name_token,
            )
        if name == "warning":
            # A `#warning` exists in order not to stop a compilation, so
            # stopping one is the one response to it that cannot be right.
            # Every compiler reports it and carries on, and C23 wrote that
            # down; a published set puts one at the top of each header it has
            # superseded, so refusing it stopped nine of the 1350 headers in
            # one of them over a line that changes nothing emitted. Written
            # where a compiler writes a diagnostic, there being no other
            # channel out of here.
            spelled = _respaced(rest)
            print(
                f"py2bin: warning: {name_token.origin}:{name_token.line}:"
                f"{name_token.column}: {spelled}",
                file=sys.stderr,
            )
            return
        self.error(f"unknown preprocessing directive #{name}", name_token)

    # --- #define ---

    def _define(self, rest: list[PPToken], at: PPToken) -> None:
        if not rest or rest[0].kind != "identifier":
            self.error("#define needs a macro name", at)
        name_token = rest[0]
        name = name_token.spelling
        if name == "defined":
            self.error("'defined' cannot be used as a macro name", name_token)
        if not self.priming and (name in _PREDEFINED or name in ("__FILE__", "__LINE__")):
            self.error(f"{name} is predefined and cannot be redefined", name_token)
        index = 1
        parameters: tuple[str, ...] | None = None
        variadic = False
        if index < len(rest) and rest[index].spelling == "(" and not rest[index].spaced:
            index += 1
            names: list[str] = []
            if index < len(rest) and rest[index].spelling == ")":
                index += 1
            else:
                while True:
                    if index >= len(rest):
                        self.error("this parameter list is not closed", name_token)
                    token = rest[index]
                    if token.spelling == "..." and token.kind == "punctuator":
                        variadic = True
                        names.append("__VA_ARGS__")
                        index += 1
                    elif token.kind == "identifier":
                        if token.spelling == "__VA_ARGS__":
                            self.error(
                                "__VA_ARGS__ is only the name of the variable "
                                "arguments and cannot be a parameter",
                                token,
                            )
                        if token.spelling in names:
                            self.error(
                                f"parameter {token.spelling!r} is named twice", token
                            )
                        names.append(token.spelling)
                        index += 1
                    else:
                        self.error(
                            f"expected a parameter name, found {token.spelling!r}", token
                        )
                    if index >= len(rest):
                        self.error("this parameter list is not closed", name_token)
                    if rest[index].spelling == ")":
                        index += 1
                        break
                    if rest[index].spelling != ",":
                        self.error(
                            f"expected ',' or ')', found {rest[index].spelling!r}",
                            rest[index],
                        )
                    if variadic:
                        self.error("'...' must be the last parameter", rest[index])
                    index += 1
            parameters = tuple(names)
        body = list(rest[index:])
        if body:
            # The whitespace before the first body token is not part of the
            # replacement list, and must not show up in a stringification.
            body[0] = dataclasses.replace(body[0], spaced=False)
        self._check_body(body, parameters, name_token)
        macro = Macro(name, parameters, variadic, tuple(body), name_token)
        existing = self.macros.get(name)
        if existing is not None and not self._same(existing, macro):
            # What py2bin's own header said is a default, not a definition:
            # it is there so a program that never sees a real platform header
            # still has the name. A program that does see one has the real
            # thing, and it wins. Every implementation spells these
            # differently - `S_OK` is `((HRESULT)0)` here and
            # `((HRESULT)0x00000000)` in the set a fetch brings down - and
            # neither is wrong, so this is not a mistake to report.
            if name in self._defaults:
                self._defaults.discard(name)
                self.macros[name] = macro
                return
            if self._reading_a_builtin:
                # A program, or a header on its path, defined this name and
                # then reached one of py2bin's own headers that defines it
                # too: <apisetcconv.h> writes WINBASEAPI and then includes a
                # piece of <windows.h>, which is py2bin's. C's answer is that
                # the last definition wins and the compiler says so, and that
                # is what every compiler does. Keeping the *first* instead -
                # which is what this branch did - was a quiet disagreement
                # with all of them: `#define EOF 0` and then <stdio.h> left
                # EOF at 0 where clang says -1, and a DECLARE_HANDLE spelled
                # by the program before <windows.h> made HKL and HFONT
                # four-byte ints where the platform has eight-byte pointers.
                # Nothing was said, and the program ran. So the last one wins
                # here as well, with the diagnostic clang gives.
                print(
                    f"py2bin: warning: {name_token.origin}:{name_token.line}:"
                    f"{name_token.column}: macro {name!r} redefined; the "
                    f"earlier definition at {existing.token.origin}:"
                    f"{existing.token.line}:{existing.token.column} is replaced",
                    file=sys.stderr,
                )
                self._defaults.add(name)
                self.macros[name] = macro
                return
            self.error(
                f"macro {name!r} is redefined with a different replacement list; "
                "C requires every definition of a macro to be identical "
                f"(the first was at {existing.token.origin}:{existing.token.line}:"
                f"{existing.token.column})",
                name_token,
            )
        if self._reading_a_builtin:
            self._defaults.add(name)
        self.macros[name] = macro

    def _check_body(
        self,
        body: list[PPToken],
        parameters: tuple[str, ...] | None,
        at: PPToken,
    ) -> None:
        for position, token in enumerate(body):
            if (
                token.kind == "identifier"
                and token.spelling == "__VA_ARGS__"
                and (parameters is None or "__VA_ARGS__" not in parameters)
            ):
                self.error(
                    "__VA_ARGS__ means something only in the replacement list of a "
                    "macro whose parameter list ends with '...'",
                    token,
                )
            if token.kind != "punctuator":
                continue
            if token.spelling == "##":
                if position == 0:
                    self.error("'##' cannot begin a replacement list", token)
                if position == len(body) - 1:
                    self.error("'##' cannot end a replacement list", token)
                if (
                    parameters is not None
                    and "__VA_ARGS__" in parameters
                    and body[position - 1].spelling == ","
                    and body[position + 1].spelling == "__VA_ARGS__"
                ):
                    self.error(
                        "', ## __VA_ARGS__' is a GNU extension that deletes the "
                        "comma when the variable arguments are empty; py2bin "
                        "implements standard C only",
                        token,
                    )
            if token.spelling == "#" and parameters is not None:
                following = body[position + 1] if position + 1 < len(body) else None
                if following is None or following.spelling not in parameters:
                    self.error(
                        "'#' in a function-like macro must be followed by a "
                        "parameter name",
                        token,
                    )

    @staticmethod
    def _same(left: Macro, right: Macro) -> bool:
        if left.parameters != right.parameters or left.variadic != right.variadic:
            return False
        if len(left.body) != len(right.body):
            return False
        return all(
            one.spelling == two.spelling and one.spaced == two.spaced
            for one, two in zip(left.body, right.body)
        )

    # --- #include ---

    #: The pragmas that would make py2bin emit something else, and what each
    #: of them would change. Everything not named here is read and dropped,
    #: which is what C11 6.10.6 and C++ [cpp.pragma] ask for: a pragma an
    #: implementation does not recognise is ignored. Naming the ones it did
    #: recognise instead - and refusing the rest - was the wrong way round,
    #: and it stopped ordinary portable source on its first line, because
    #: the pragmas a compiler is told about are unbounded and the ones that
    #: mean anything to *this* compiler are not.
    #:
    #: The question each entry answers is "what would py2bin have to emit
    #: differently if it obeyed?" - where the members of a struct sit, which
    #: definition a name reaches, which section something lands in. A pragma
    #: with no answer to that is talking to a compiler about diagnostics,
    #: folding, inlining or unrolling, and saying nothing about the program.
    #: `pack` is absent because it is not refused: it is implemented, below.
    _CHANGES_THE_PROGRAM = {
        "ms_struct": "lays every struct out by another ABI's rules",
        "align": "says where the members of a struct sit",
        "options": "carries `align=`, which says where the members of a "
                   "struct sit",
        "scalar_storage_order": "stores the members of a struct in the "
                                "other byte order",
        "pointers_to_members": "says how wide a pointer to a member is",
        "vtordisp": "adds a hidden field to a class with a virtual base",
        "weak": "makes a definition weak, which says which one a call "
                "reaches",
        "redefine_extname": "gives a definition a different symbol name",
        "init_seg": "sets the order objects with static storage are "
                    "constructed in",
        "section": "declares a section of its own for what follows",
        "code_seg": "puts the functions that follow in a section of their own",
        "data_seg": "puts the objects that follow in a section of their own",
        "const_seg": "puts the constants that follow in a section of their "
                     "own",
        "bss_seg": "puts the zeroed objects that follow in a section of "
                   "their own",
    }

    def _why_here(self, conditions: "list[_Condition]") -> str:
        """What the chain around a `#error` asked for, and did not get.

        A header that falls through every branch of a chain and stops is
        telling you what it wanted. Naming the branches is the whole answer:
        it says which conditions would have to hold, and whether any of them
        is one you can arrange - which guessing at `-D` cannot say, and got
        wrong when it tried.
        """

        if not conditions or not conditions[-1].tried:
            return ""
        tried = [spelled for spelled, held in conditions[-1].tried if not held]
        if not tried:
            return ""
        lines = "\n".join(f"      {one}" for one in tried[:12])
        more = (
            f"\n      ... and {len(tried) - 12} more" if len(tried) > 12 else ""
        )
        return (
            f"\n  Reached because none of these held:\n{lines}{more}\n"
            f"  Each names what that branch needed. py2bin defines the "
            f"platform and architecture macros for the target it is building "
            f"for, and does not define another compiler's - a header that "
            f"believed it was being compiled by one would reach for builtins "
            f"that are not here. Where a branch asks for something you can "
            f"arrange, `-D NAME` (build.py) or `--define NAME` (py2bin cc) "
            f"is how."
        )

    def _pragma(
        self,
        rest: "list[PPToken]",
        name_token: "PPToken",
        into: "list[PPToken] | None" = None,
    ) -> None:
        """`#pragma once`, the few that change the program, and the rest.

        A pragma is where an implementation is allowed to be told anything at
        all, so nearly all of them say nothing to this compiler and are
        dropped where they stand - which is what C asks for, and what every
        other compiler does. The exceptions are the ones that would change
        what py2bin emits if it obeyed: `pack`, which is implemented, and
        those in `_CHANGES_THE_PROGRAM`, which are refused by name. Dropping
        one of *those* would lay the program out differently and say nothing,
        and that is worse than not building it.

        `into` is where `pack` puts the tokens it hands the parser. It is the
        output when a `#pragma` line is read, and the run being expanded when
        a `_Pragma` is - a pack written that way belongs where the operator
        was, not in front of everything gathered since the last directive.
        """

        output = self.output if into is None else into
        spelled = [item.spelling for item in rest]
        if spelled == ["once"]:
            self.once.add(name_token.origin)
            return
        if len(spelled) == 3 and spelled[:2] == ["py2bin", "supplied"]:
            # py2bin's own, and only py2bin writes it: the header named here
            # has already been read out into this text by another run of this
            # preprocessor. A header of py2bin's own is remembered by name
            # rather than by a guard, and that memory does not survive from
            # one run to the next - so it is written down instead. Without
            # it, a program that includes <windows.h> and a header that was
            # preprocessed separately gets both copies and is told its
            # structs are defined twice.
            self._builtins_read.add(spelled[2].strip('"'))
            return
        if not spelled:
            # `#pragma` with nothing after it. C says an implementation may
            # do what it likes, and there is nothing here to do.
            return
        if spelled[0] == "pack":
            # It changes how every struct after it is laid out, so it cannot
            # be dropped the way the rest are. It is passed on to the parser
            # as tokens - a directive is not one, and this is the only
            # channel there is between the two.
            output.append(
                dataclasses.replace(
                    name_token, kind="identifier", spelling=_PACK_MARKER
                )
            )
            output.extend(rest[1:])
            output.append(
                dataclasses.replace(name_token, kind="punctuator", spelling=";")
            )
            return
        changes = self._CHANGES_THE_PROGRAM.get(spelled[0])
        if changes is not None:
            self.error(
                f"#pragma {spelled[0]} is not implemented, and py2bin will "
                f"not ignore it the way it ignores a pragma that says "
                f"nothing to it: this one {changes}. A program built as if "
                f"it were not there is laid out differently from the one "
                f"that was written, and nothing would say so",
                name_token,
            )
        # Everything else. Only the first word is read, because everything
        # after it belongs to that pragma rather than to this compiler:
        # `#pragma region section` names a region, and reading its second
        # word as a pragma of its own would refuse a fold marker.
        return

    def _pragma_operator(
        self,
        pending: "collections.deque[PPToken]",
        at: "PPToken",
        out: "list[PPToken]",
    ) -> None:
        """`_Pragma("...")`, which is the only way a macro can write a pragma.

        A directive is not a token, so nothing a macro expands to can be a
        `#pragma`; C99 gave the same thing an operator spelling for exactly
        that reason, and headers use it to wrap a pragma in a name of their
        own. It is read here rather than beside the directives because the
        string is usually built by `#` out of a macro's argument, and there
        is nothing to read until the expansion has run.

        Which is also why a `pack` written this way is refused in a C++
        translation unit and only there. The C++ translator reads a pack out
        of the text, before any macro has been replaced, so that a class can
        carry its packing to wherever the struct is written out - and by the
        time the string exists to be read here, the classes have been laid
        out already. In C nothing moves and this is simply the pack.
        """

        opening = pending[0] if pending else None
        if opening is None or opening.kind != "punctuator" or opening.spelling != "(":
            self.error("_Pragma takes a parenthesised string literal", at)
        pending.popleft()
        literal = pending[0] if pending else None
        if literal is None or literal.kind != "string":
            self.error("_Pragma takes a string literal", at)
        pending.popleft()
        closing = pending[0] if pending else None
        if closing is None or closing.kind != "punctuator" or closing.spelling != ")":
            self.error("_Pragma is not closed with ')'", at)
        pending.popleft()
        written = self._destringized(literal, at)
        if (
            self.translated_cplusplus
            and written
            and written[0].spelling == "pack"
        ):
            self.error(
                "a `pack` written as `_Pragma(\"pack(...)\")` is not "
                "translated from C++; write it as `#pragma pack(...)`, which "
                "is. The C++ translator reads a pack out of the text so a "
                "class can carry its packing to wherever the struct ends up, "
                "and it runs before any macro has been replaced - so the "
                "classes above have been laid out unpacked already",
                at,
            )
        self._pragma(written, at, out)

    @staticmethod
    def _destringized(literal: "PPToken", at: "PPToken") -> "list[PPToken]":
        """The tokens `_Pragma`'s operand stands for (C11 6.10.9).

        Any prefix and the quotes go, `\\"` becomes a quote and `\\\\` becomes
        one backslash, and what is left is read as the tokens that would have
        followed a `#pragma`. Read left to right in one pass, the way the
        standard states it, so a backslash this produces is what the pragma
        says and never the start of an escape the author did not write.

        Every token is put back where the operator was written. What comes
        out of here is a position inside a string nobody can point at, and a
        `pack` handed to the parser has to say which line it came from.
        """

        spelling = literal.spelling
        inside = spelling[spelling.index('"') + 1: spelling.rindex('"')]
        text: "list[str]" = []
        index = 0
        while index < len(inside):
            if inside[index] == "\\" and inside[index + 1: index + 2] in ('"', "\\"):
                text.append(inside[index + 1])
                index += 2
                continue
            text.append(inside[index])
            index += 1
        return [
            dataclasses.replace(
                token, line=at.line, column=at.column, origin=at.origin
            )
            for line in _scan("".join(text), at.origin)
            for token in line
        ]

    def _search_path(self, directory: "Path | None") -> "list[Path]":
        """Every directory an include is looked for in, nearest first.

        The asking file's own comes first, the way it does for an include
        written with quotes - which is what `__has_include` is asking about.
        """

        return [*([] if directory is None else [directory]), *self.include_dirs]

    def _include(self, rest: list[PPToken], at: PPToken, directory: Path | None) -> None:
        name, angled = self._header_name(rest, at)
        if name is None:
            expanded = self.expand(list(rest))
            name, angled = self._header_name(expanded, at)
            if name is None:
                self.error(
                    "#include needs <a header> or \"a file\", or a macro that "
                    "expands to one",
                    at,
                )
        if not name:
            self.error("#include needs a file name", at)
        candidates: list[Path] = []
        if not angled and directory is not None:
            candidates.append(directory / name)
        candidates.extend(item / name for item in self.include_dirs)
        builtin = _BUILTIN_HEADERS.get(name)
        for candidate in candidates:
            if not candidate.is_file():
                continue
            # A header py2bin ships is never taken from the directory py2bin
            # downloads into. A fetched set brings its neighbours along, so a
            # build that once fetched anything from a Windows set left that
            # set's `winnt.h` in the cache - and an include directory is
            # searched before a built-in, so py2bin's own was shadowed by a
            # copy that cannot compile here, for every build after. A header
            # named with -I is somebody's own choice and still wins.
            if builtin is not None and _FETCHED_INTO in candidate.parts:
                continue
            if builtin is None:
                # A header the C++ stage pasted already is read for its
                # macros - the header being read alone here is written in
                # them - and what it declares is dropped from this run's
                # output, the way py2bin's own headers are treated. A fetched
                # <ws2tcpip.h> asked for <winsock2.h>, which the program had
                # included directly and which was already in the unit, so
                # its enum arrived twice and the C compiler said so.
                lowered = name.lower()
                if name in self.already_pasted or any(
                    lowered == other.lower() for other in self.already_pasted
                ):
                    self.pasted_origins.add(self._origin_of(candidate))
                self.search_path_read.add(name)
            self._read(candidate, at)
            return
        if builtin is not None:
            if name in self._builtins_read:
                # C says a standard header may be included more than once, and
                # programs rely on it: two headers of a project each include
                # <stdlib.h> and the second is meant to be nothing. The ones
                # that are only #defines survived that; <math.h> and <stdlib.h>
                # carry functions, and a second copy is a redefinition.
                return
            self._builtins_read.add(name)
            was = self._reading_a_builtin
            self._reading_a_builtin = True
            # py2bin's own C headers were written as C. Two of them carry a
            # C++ arm behind `#ifdef __cplusplus` - a table of function
            # pointers on one side and classes on the other - and the C run
            # has to be handed the table, whatever the program's own text was
            # allowed to see. Reading one is synchronous, so the name is
            # simply absent for the duration and back afterwards.
            saw_cplusplus = self.macros.pop("__cplusplus", None)
            try:
                self._enter(builtin, f"<{name}>", None, at)
            finally:
                self._reading_a_builtin = was
                if saw_cplusplus is not None:
                    self.macros["__cplusplus"] = saw_cplusplus
            return
        if name in self.cplusplus_supplies:
            # Not this run's to answer. A header with an `#else` in it is read
            # here ahead of the C++ translator, so that the branch a C++
            # compiler takes is the one it is handed - and a project's own
            # <fstream> on the search path did exactly that and then asked
            # for <filesystem>, which only the translator has. Refused here,
            # it said "cannot find the header" and listed every C header
            # py2bin ships, none of which was the one asked for. Noted
            # instead, and asked of the stage that has it.
            self.left_to_cplusplus.add(name)
            return
        # Deduped and one per line: the list repeated itself where two search
        # roots coincided, and a wall of comma-separated paths is hard to read
        # at the moment someone most needs to read it.
        seen: list[str] = []
        for item in candidates:
            spelled = str(item)
            if spelled not in seen:
                seen.append(spelled)
        where = "\n    ".join(seen) or "nowhere"
        self.error(
            f"cannot find the header {name!r}. py2bin looked in:\n    {where}\n"
            "  Add a directory with --include DIR (build.py) or --include-dir "
            "(py2bin cc); a folder called include, inc, headers or src beside "
            "the program is searched anyway.\n"
            "  py2bin ships the standard headers "
            f"({', '.join(sorted(_BUILTIN_HEADERS))}) and has no system include "
            "path. The core of a platform SDK is among them, written by py2bin "
            "rather than taken from a set that asks which compiler is reading "
            "it; the rest of a set can be brought down with --auto-fetch and "
            "compiles against that core",
            at,
        )

    @staticmethod
    def _header_name(rest: list[PPToken], at: PPToken) -> tuple[str | None, bool]:
        """Read ``<file>`` or ``"file"`` back out of preprocessing tokens."""

        if not rest:
            return None, False
        if rest[0].kind == "string" and len(rest) == 1:
            return rest[0].spelling[1:-1], False
        if rest[0].spelling == "<" and rest[-1].spelling == ">":
            inner = rest[1:-1]
            if any(item.spaced for item in inner):
                return None, True
            return "".join(item.spelling for item in inner), True
        return None, False

    def _origin_of(self, path: Path) -> str:
        """What this file is called here, whatever the include spelled.

        `#pragma once` has to decide whether the path in hand names a file
        already read, and comparing the spelled path cannot answer that. The
        filesystem this is running on is case-insensitive, so `Foo.h` and
        `foo.h` are one file under two names - and a header read twice is a
        wall of duplicate definitions, which is the one thing `#pragma once`
        exists to prevent. A symlink is the same question wearing a different
        hat, and `resolve()` alone answers only that half of it.

        So the filesystem is asked instead of the string: the device and inode
        a stat reports are what makes two paths one file. The first spelling
        that reached the file is the name every later one answers to, and it
        is a real path, so a diagnostic still points at something that exists.
        """

        settled = str(path.resolve())
        try:
            status = path.stat()
        except OSError:
            return settled
        if not status.st_ino:
            # A filesystem that does not number its files. There is nothing
            # left to compare but the path, and `resolve()` has at least
            # followed the symlinks and dropped the `./` and the doubled
            # slashes from it.
            return settled
        return self._files_seen.setdefault((status.st_dev, status.st_ino), settled)

    def _read(self, path: Path, at: PPToken) -> None:
        resolved = self._origin_of(path)
        if resolved in self.once:
            # _enter would notice too; catching it here means a header that
            # said '#pragma once' is not even read a second time.
            return
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as problem:
            self.error(f"cannot read {path}: {problem}", at)
        except UnicodeDecodeError:
            self.error(f"{path} is not UTF-8 text", at)
        self._enter(source, resolved, path.parent, at)

    def _enter(self, source: str, origin: str, directory: Path | None, at: PPToken) -> None:
        if origin in self.once:
            return
        if self.depth >= _MAXIMUM_INCLUDE_DEPTH:
            self.error(
                f"#include nested more than {_MAXIMUM_INCLUDE_DEPTH} deep; this is "
                "almost always a cycle without an include guard",
                at,
            )
        self.depth += 1
        try:
            self._file(source, origin, directory)
        finally:
            self.depth -= 1

    # --- conditionals ---

    def _condition(
        self,
        name: str,
        rest: list[PPToken],
        at: PPToken,
        directory: "Path | None" = None,
    ) -> bool:
        if name in ("ifdef", "ifndef"):
            if not rest or rest[0].kind != "identifier":
                self.error(f"#{name} needs a macro name", at)
            if len(rest) > 1:
                self.error(f"#{name} takes one macro name", rest[1])
            defined = rest[0].spelling in self.macros
            return defined if name == "ifdef" else not defined
        if not rest:
            self.error("#if needs an expression", at)
        tokens = self.expand(self._defined(rest, at))
        return _Evaluator(tokens, at, self._search_path(directory)).run() != 0

    def _defined(self, rest: list[PPToken], at: PPToken) -> list[PPToken]:
        """Apply the ``defined`` operator before anything is macro-expanded."""

        result: list[PPToken] = []
        index = 0
        while index < len(rest):
            token = rest[index]
            if not (token.kind == "identifier" and token.spelling == "defined"):
                result.append(token)
                index += 1
                continue
            index += 1
            wrapped = index < len(rest) and rest[index].spelling == "("
            if wrapped:
                index += 1
            if index >= len(rest) or rest[index].kind != "identifier":
                self.error("'defined' needs a macro name", token)
            name = rest[index].spelling
            index += 1
            if wrapped:
                if index >= len(rest) or rest[index].spelling != ")":
                    self.error("'defined(' needs a closing ')'", token)
                index += 1
            result.append(
                PPToken(
                    "number",
                    "1" if name in self.macros else "0",
                    token.line,
                    token.column,
                    token.origin,
                    token.spaced,
                )
            )
        return result

    # --- macro expansion ---

    def expand(self, tokens: list[PPToken]) -> list[PPToken]:
        """Expand every macro invocation in ``tokens`` (C11 6.10.3.4)."""

        out: list[PPToken] = []
        pending = collections.deque(tokens)
        while pending:
            token = pending.popleft()
            if token.kind != "identifier" or token.spelling in token.hides:
                out.append(token)
                continue
            macro = self.macros.get(token.spelling)
            if macro is None:
                if token.spelling == "_Pragma":
                    self._pragma_operator(pending, token, out)
                    continue
                out.append(token)
                continue
            self.expansions += 1
            if self.expansions > _MAXIMUM_EXPANSIONS:
                self.error(
                    "macro expansion did not finish; py2bin stopped after "
                    f"{_MAXIMUM_EXPANSIONS} expansions",
                    token,
                )
            if macro.builtin:
                out.append(self._builtin(macro, token))
                continue
            if macro.parameters is None:
                replacement = self._substitute(macro, token, None, token.hides | {macro.name})
                pending.extendleft(reversed(replacement))
                continue
            if not pending or pending[0].spelling != "(" or pending[0].kind != "punctuator":
                # A function-like macro that is not called is just an identifier.
                out.append(token)
                continue
            arguments, closing = self._arguments(pending, macro, token)
            hides = (token.hides & closing.hides) | {macro.name}
            replacement = self._substitute(macro, token, arguments, hides)
            pending.extendleft(reversed(replacement))
        return out

    @staticmethod
    def _builtin(macro: Macro, at: PPToken) -> PPToken:
        if macro.builtin == "__LINE__":
            return PPToken("number", str(at.line), at.line, at.column, at.origin, at.spaced)
        spelling = at.origin.replace("\\", "\\\\").replace('"', '\\"')
        return PPToken("string", f'"{spelling}"', at.line, at.column, at.origin, at.spaced)

    def _arguments(
        self,
        pending: collections.deque[PPToken],
        macro: Macro,
        at: PPToken,
    ) -> tuple[list[list[PPToken]], PPToken]:
        pending.popleft()  # the '('
        arguments: list[list[PPToken]] = [[]]
        depth = 0
        closing: PPToken | None = None
        named = len(macro.parameters or ()) - (1 if macro.variadic else 0)
        while pending:
            token = pending.popleft()
            if token.kind == "punctuator":
                if token.spelling == "(":
                    depth += 1
                elif token.spelling == ")":
                    if depth == 0:
                        closing = token
                        break
                    depth -= 1
                elif token.spelling == "," and depth == 0:
                    # Once the variable arguments have started, a comma is part
                    # of __VA_ARGS__ rather than a separator.
                    if not (macro.variadic and len(arguments) > named):
                        arguments.append([])
                        continue
            arguments[-1].append(token)
        if closing is None:
            self.error(
                f"the argument list of macro {macro.name!r} is not closed", at
            )
        if arguments == [[]] and not macro.parameters:
            arguments = []
        expected = len(macro.parameters or ())
        if macro.variadic:
            if len(arguments) == named:
                arguments = list(arguments) + [[]]
            if len(arguments) != expected:
                self.error(
                    f"macro {macro.name!r} takes at least {named} argument(s) but "
                    f"{len(arguments)} were given",
                    at,
                )
        elif len(arguments) != expected:
            self.error(
                f"macro {macro.name!r} takes {expected} argument(s) but "
                f"{len(arguments)} were given",
                at,
            )
        return arguments, closing

    def _substitute(
        self,
        macro: Macro,
        at: PPToken,
        arguments: list[list[PPToken]] | None,
        hides: frozenset[str],
    ) -> list[PPToken]:
        """Build a macro's replacement list: C11 6.10.3.1 through 6.10.3.3."""

        values: dict[str, list[PPToken]] = {}
        if arguments is not None:
            for name, argument in zip(macro.parameters or (), arguments):
                values[name] = argument
        body = macro.body
        result: list[PPToken] = []
        index = 0
        while index < len(body):
            token = body[index]
            following = body[index + 1] if index + 1 < len(body) else None
            if (
                token.kind == "punctuator"
                and token.spelling == "#"
                and arguments is not None
                and following is not None
                and following.spelling in values
            ):
                result.append(_stringify(values[following.spelling], self._at(token, at)))
                index += 2
                continue
            if token.kind == "punctuator" and token.spelling == "##" and following is not None:
                if following.spelling in values:
                    argument = values[following.spelling]
                    right = (
                        [self._at(item, at) for item in argument]
                        if argument
                        else [self._at(_placemarker(following), at)]
                    )
                else:
                    right = [self._at(following, at)]
                self._glue(result, right, at)
                index += 2
                continue
            if token.spelling in values and token.kind == "identifier":
                argument = values[token.spelling]
                if following is not None and following.spelling == "##":
                    substituted = list(argument) or [self._at(_placemarker(token), at)]
                else:
                    # Everywhere but next to # or ##, an argument is macro
                    # expanded first, and then substituted.
                    substituted = self.expand(list(argument))
                result.extend(_spaced_like(substituted, token))
                index += 1
                continue
            result.append(self._at(token, at))
            index += 1
        # What replaces the macro sits where the macro name sat, whitespace
        # included: that is what makes the standard's own 'join' example
        # stringify as "x ## y" rather than "x## y".
        return [
            dataclasses.replace(item, hides=item.hides | hides)
            for item in _spaced_like(result, at)
            if item.kind != _PLACEMARKER
        ]

    @staticmethod
    def _at(token: PPToken, at: PPToken) -> PPToken:
        """Move a replacement-list token to where the macro was written."""

        return dataclasses.replace(token, line=at.line, column=at.column, origin=at.origin)

    def _glue(self, result: list[PPToken], right: list[PPToken], at: PPToken) -> None:
        if not result:  # unreachable: '##' cannot begin a replacement list
            result.extend(right)
            return
        left = result.pop()
        first = right[0]
        if left.kind == _PLACEMARKER:
            pasted = first
        elif first.kind == _PLACEMARKER:
            pasted = left
        else:
            spelling = left.spelling + first.spelling
            lines = _scan(spelling, left.origin)
            if len(lines) != 1 or len(lines[0]) != 1:
                self.error(
                    f"pasting {left.spelling!r} and {first.spelling!r} does not "
                    "make a single preprocessing token",
                    at,
                )
            pasted = dataclasses.replace(
                lines[0][0],
                line=left.line,
                column=left.column,
                origin=left.origin,
                spaced=left.spaced,
                hides=left.hides & first.hides,
            )
        result.append(pasted)
        result.extend(right[1:])

    # --- handing the result to the C front end ---

    def tokens(self) -> list[Token]:
        result: list[Token] = []
        for token in self.output:
            result.append(_convert(token, self.error))
        line, column, origin = 1, 1, ""
        if self.output:
            last = self.output[-1]
            line, column, origin = last.line, last.column + len(last.spelling), last.origin
        result.append(Token("eof", "", line, column, origin=origin))
        return result


def _spaced_like(tokens: list[PPToken], at: PPToken) -> list[PPToken]:
    """Give a replacement the whitespace of the token it replaces."""

    if not tokens or tokens[0].spaced == at.spaced:
        return tokens
    return [dataclasses.replace(tokens[0], spaced=at.spaced)] + tokens[1:]


def _placemarker(token: PPToken) -> PPToken:
    return PPToken(_PLACEMARKER, "", token.line, token.column, token.origin, token.spaced)


def _stringify(argument: list[PPToken], at: PPToken) -> PPToken:
    """The ``#`` operator: C11 6.10.3.2."""

    pieces: list[str] = []
    for position, token in enumerate(argument):
        if position and token.spaced:
            pieces.append(" ")
        if token.kind in ("string", "character"):
            pieces.append(token.spelling.replace("\\", "\\\\").replace('"', '\\"'))
        else:
            pieces.append(token.spelling)
    spelling = '"' + "".join(pieces) + '"'
    try:
        lines = _scan(spelling, at.origin)
    except CCompileError:
        lines = []
    if len(lines) != 1 or len(lines[0]) != 1 or lines[0][0].kind != "string":
        # C11 6.10.3.2p2 leaves this undefined: a lone backslash, say, would
        # make a string literal that does not end.
        raise CCompileError(
            at.origin,
            at.line,
            at.column,
            "stringifying this argument with '#' does not make a valid string "
            f"literal ({spelling})",
        )
    return PPToken("string", spelling, at.line, at.column, at.origin, at.spaced)


def _convert(token: PPToken, error) -> Token:
    """Turn one preprocessing token into a token the C parser understands.

    The conversion is done by py2bin's own C lexer, so a number, a character
    constant or a string literal is read by exactly the code that reads it when
    no macro is involved.
    """

    if token.kind == "punctuator" and token.spelling in ("#", "##"):
        error(
            f"{token.spelling!r} is a preprocessing operator and means nothing in "
            "C; it is only valid inside a #define",
            token,
        )
    if token.kind == "other":
        error(f"unsupported character {token.spelling!r}", token)
    lexer = Lexer(token.spelling, token.origin)
    lexer.line, lexer.column = token.line, token.column
    produced = lexer.tokens()
    if len(produced) != 2:
        error(f"{token.spelling!r} is not a valid C token", token)
    return dataclasses.replace(produced[0], origin=token.origin)


# --- the standard headers py2bin serves itself -------------------------------

#: py2bin's C compiler has the fixed-width types, ``size_t``, ``printf`` and the
#: rest built in, so these copies carry only what a header can really give a
#: The math functions, as the C py2bin compiles them from. Written here
#: rather than in a file beside this one so that the package is Python and
#: nothing else: every other body of C or C++ py2bin ships - the standard
#: headers, the COM ones, the C++ library - is a string in a module, and a
#: lone `.c` file was the one thing that had to be packaged specially and
#: found at run time.
_MATH_H = r"""/* A libm written in the C py2bin compiles, so the compiler supplies the math
 * functions by compiling them rather than by linking anything. Every routine
 * reduces its argument to a small interval and evaluates a polynomial there;
 * the reductions use a split constant (Cody-Waite) so the reduced argument
 * keeps full precision. */

union __PY2BIN_BITS { double d; long long i; };

double __py2bin_scalbn(double v, long long k)
{
    /* 2^k built straight from the exponent field: no library, no loop. */
    union __PY2BIN_BITS u;
    if (k > 1023) { k = 1023; }
    if (k < -1022) { k = -1022; }
    u.i = (k + 1023) << 52;
    return v * u.d;
}

double exp(double x)
{
    if (x != x) { return x; }
    if (x > 709.782712893384) { return 1.0e308 * 10.0; }
    if (x < -745.1332191019411) { return 0.0; }
    double n = x * 1.4426950408889634;              /* x / ln 2 */
    long long k = (long long)(n < 0.0 ? n - 0.5 : n + 0.5);
    double kd = (double)k;
    /* ln 2 split so k * ln2 is exact to more than 53 bits. */
    double r = x - kd * 0.693147180369123816490 - kd * 1.90821492927058770002e-10;
    double p = 1.0 + r * (1.0 + r * (0.5 + r * (0.1666666666666666574
        + r * (0.04166666666666666435 + r * (0.008333333333333333218
        + r * (0.001388888888888888785 + r * (0.0001984126984126984125
        + r * (0.00002480158730158730495 + r * (0.000002755731922398589
        + r * (2.75573192239858883e-07 + r * (2.50521083854417202e-08
        + r * (2.08767569878681002e-09 + r * (1.60590438368216133e-10
        + r * 1.14707455977297245e-11)))))))))))));
    return __py2bin_scalbn(p, k);
}

double log(double x)
{
    if (x != x) { return x; }
    if (x < 0.0) { return 0.0 / 0.0; }
    if (x == 0.0) { return -1.0 / 0.0; }
    union __PY2BIN_BITS u;
    u.d = x;
    long long k = ((u.i >> 52) & 2047) - 1023;
    if (k == -1023) {           /* subnormal: scale into the normal range */
        u.d = x * 4503599627370496.0;
        k = (((u.i >> 52) & 2047) - 1023) - 52;
    }
    u.i = (u.i & 4503599627370495LL) | (1023LL << 52);   /* m in [1, 2) */
    double m = u.d;
    if (m > 1.4142135623730951) { m = m * 0.5; k = k + 1; }
    /* log m = 2 atanh t with t = (m-1)/(m+1); |t| <= 0.1716 */
    double t = (m - 1.0) / (m + 1.0);
    double s = t * t;
    double p = 1.0 + s * (0.3333333333333333 + s * (0.2 + s * (0.14285714285714285
        + s * (0.1111111111111111 + s * (0.09090909090909091 + s * (0.07692307692307693
        + s * (0.06666666666666667 + s * (0.058823529411764705 + s * (0.05263157894736842
        + s * 0.047619047619047616)))))))));
    return (double)k * 0.6931471805599453094 + 2.0 * t * p;
}

double __py2bin_sinpoly(double r)
{
    double s = r * r;
    return r * (
        1.0 + s * (-0.16666666666666666 + s * (0.008333333333333333 + s * (-0.0001984126984126984 + s * (2.7557319223985893e-06 + s * (-2.505210838544172e-08 + s * (1.6059043836821613e-10 + s * (-7.647163731819816e-13 + s * (2.8114572543455206e-15 + s * (-8.22063524662433e-18))))))))));
}

double __py2bin_cospoly(double r)
{
    double s = r * r;
    return (
        1.0 + s * (-0.5 + s * (0.041666666666666664 + s * (-0.001388888888888889 + s * (2.48015873015873e-05 + s * (-2.755731922398589e-07 + s * (2.08767569878681e-09 + s * (-1.1470745597729725e-11 + s * (4.779477332387385e-14 + s * (-1.5619206968586225e-16))))))))));
}

double __py2bin_trig(double x, long long want_cos)
{
    if (x != x) { return x; }
    if (x < -1.0e9 || x > 1.0e9) { return 0.0 / 0.0; }
    double n = x * 0.6366197723675814;              /* x / (pi/2) */
    long long k = (long long)(n < 0.0 ? n - 0.5 : n + 0.5);
    double kd = (double)k;
    /* pi/2 in three pieces, so the reduced argument stays accurate. */
    /* pi/2 split so that kd * (each part) is exact: the leading part has 33
     * significant bits and the rest are zero, which is what stops the
     * subtraction from cancelling away the low bits of r for large x. */
    /* pi/2 as a leading part with 33 significant bits (so kd * it is exact)
     * plus its tail. Subtracting a third piece here would double-count the
     * tail: the further stages of the classic reduction refine the REMAINDER,
     * they are not additional terms. */
    double r = x - kd * 1.57079632673412561417e+00;
    r = r - kd * 6.07710050650619224932e-11;
    long long q = (k + (want_cos ? 1 : 0)) & 3;
    if (q < 0) { q = q + 4; }
    if (q == 0) { return __py2bin_sinpoly(r); }
    if (q == 1) { return __py2bin_cospoly(r); }
    if (q == 2) { return -__py2bin_sinpoly(r); }
    return -__py2bin_cospoly(r);
}

double sin(double x) { return __py2bin_trig(x, 0); }
double cos(double x) { return __py2bin_trig(x, 1); }
double tan(double x) { return sin(x) / cos(x); }

double pow(double x, double y)
{
    if (y == 0.0) { return 1.0; }
    /* An integral exponent is exact by repeated squaring; going through
     * exp(y log x) would lose the last bit or two. */
    double ay = y < 0.0 ? -y : y;
    if (ay <= 1024.0 && ay == (double)(long long)ay && x == x) {
        long long e = (long long)ay;
        double base = x;
        double acc = 1.0;
        while (e > 0) {
            if (e & 1) { acc = acc * base; }
            base = base * base;
            e = e >> 1;
        }
        return y < 0.0 ? 1.0 / acc : acc;
    }
    if (x != x || y != y) { return 0.0 / 0.0; }
    if (x == 0.0) { return y < 0.0 ? 1.0 / 0.0 : 0.0; }
    if (x > 0.0) { return exp(y * log(x)); }
    /* A negative base is defined only for an integral exponent. */
    double t = y < 0.0 ? -y : y;
    if (t != (double)(long long)t) { return 0.0 / 0.0; }
    double magnitude = exp(y * log(-x));
    return ((long long)t & 1) ? -magnitude : magnitude;
}
"""

#: program that has no library behind it: the macros.
#: py2bin's own <windows.h>. Not Microsoft's - that one is tens of thousands
#: of declarations written in extensions this compiler does not have
#: (`__declspec`, `__stdcall`, SAL annotations, packed unions). This is the
#: part a program usually wants: the types, the constants, and prototypes for
#: functions py2bin can import from a DLL. Each prototype is an extern the
#: loader binds, so calling one still needs no toolchain.
#:
#: It is only meaningful on a Windows target, and says so on any other rather
#: than letting a program compile against declarations that cannot resolve.
_WINDOWS_H = """
#ifndef __py2bin_windows__
#error <windows.h> is for Windows targets; build with --target windows-x86_64 \
or windows-arm64, or guard the include with #ifdef _WIN32
#endif

/* How a published Windows set spells the things it asks of a compiler:
   `__C89_NAMELESS` before an anonymous member, `__LONG32` where a long has
   to stay 32 bits wide, `__MINGW_EXTENSION` before a declaration only some
   compilers take. A fetched set's own headers reach the core through
   <windef.h> or <minwindef.h>, which are this file - and the file they
   would have read those from is the one py2bin replaced, so each of them
   stopped on the first such word as if it were a type name. py2bin ships
   _mingw.h, so this resolves whether or not anything has been fetched. */
#include <_mingw.h>
/* The set's own <windows.h> reads <stdarg.h> on its first page, and the
   headers below it take that for granted: a console or string function that
   takes a `va_list` names the type without asking anyone for it. */
#include <stdarg.h>
/* The SDK's <winnt.h> includes <string.h> and <ctype.h>, and headers built on
   it lean on that: <propidl.h>'s PropVariantInit is memset. */
#include <string.h>
#include <ctype.h>
/* Which slice of the API this program is being built for.
   `#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)` opens a great many
   of a set's headers, and with nothing to expand it that line is not an
   expression at all. */
#include <winapifamily.h>

#ifndef NULL
#define NULL ((void *)0)
#endif
#define WINAPI
#define APIENTRY
#define CALLBACK
/* The rest of the set's names for how a call is made. py2bin has one
   convention per target and decides it itself, so each of these says
   nothing here - but a declaration that spells one and finds it undefined
   reads as two names in a row, and stops. */
#define CDECL
#define WINAPIV
#define PASCAL
#define APIPRIVATE
#define CONST const
#define FALSE 0
#define TRUE 1
#define MAX_PATH 260

typedef int BOOL;
typedef unsigned char BYTE;
typedef unsigned short WORD;
/* `unsigned long`, which on Windows is four bytes - the set spells it that
   way and then says `typedef DWORD ULONG;` in a second header, and with
   `unsigned int` here those two were the same width under different names
   and the second was reported as a clash. */
typedef unsigned long DWORD;
typedef unsigned int UINT;
typedef int INT;
typedef long LONG;
typedef unsigned long ULONG;
typedef long long LONGLONG;
typedef unsigned long long ULONGLONG;
typedef short SHORT;
typedef unsigned short USHORT;
typedef unsigned char UCHAR;
typedef unsigned char BOOLEAN;
typedef void *PVOID;
typedef long long LONG_PTR;
typedef unsigned long long ULONG_PTR;
typedef long long INT_PTR;
typedef unsigned long long UINT_PTR;
typedef unsigned long long DWORD_PTR;
typedef BYTE *PBYTE;
typedef BYTE *LPBYTE;
/* The fixed-width spellings. A generated COM header uses these throughout,
   because an .idl says how wide a field is rather than what a C compiler
   happens to make of `int`. */
typedef signed char INT8;
typedef short INT16;
typedef int INT32;
typedef long long INT64;
typedef unsigned char UINT8;
typedef unsigned short UINT16;
typedef unsigned int UINT32;
typedef unsigned long long UINT64;
typedef int LONG32;
typedef long long LONG64;
typedef unsigned int ULONG32;
typedef unsigned long long ULONG64;
typedef unsigned int DWORD32;
typedef unsigned long long DWORD64;
typedef float FLOAT;
typedef double DOUBLE;
typedef void *HGLOBAL;
typedef void *HLOCAL;
typedef void *HDC;
typedef void *HRGN;
typedef void *HBITMAP;
typedef void *HKEY;
typedef char *PSTR;
typedef wchar_t *PWSTR;
typedef const wchar_t *PCWSTR;
typedef const char *PCSTR;
/* Written as the struct alone rather than the union the SDK spells, because
   the two anonymous members of that union are one name for the halves and
   another for the whole, and `QuadPart` is what a program reaches for. */
typedef struct _LARGE_INTEGER { LONGLONG QuadPart; } LARGE_INTEGER;
typedef struct _ULARGE_INTEGER { ULONGLONG QuadPart; } ULARGE_INTEGER;
typedef struct _LUID { DWORD LowPart; LONG HighPart; } LUID;
/* The guards go with the definitions. A set writes its own copy of a
   handful of these small structs in whichever of its headers needs one, each
   behind `#ifndef _FILETIME_` - which is how the set keeps from defining it
   twice itself, and is what tells a second copy to stand down. py2bin
   supplies the struct, so py2bin supplies the guard: without it 12 of the
   1350 headers in one set defined `_FILETIME` a second time and were refused
   for it. */
#define _FILETIME_
typedef struct _FILETIME {
    DWORD dwLowDateTime;
    DWORD dwHighDateTime;
} FILETIME;
/* As wide as a pointer, which on Windows is not `unsigned long`: it is
   LLP64, so a `long` there is four bytes and a pointer is eight. */
typedef unsigned long long SIZE_T;
typedef long long SSIZE_T;
typedef void *HANDLE;
typedef void *HWND;
typedef void *HINSTANCE;
typedef void *HMODULE;
typedef void *LPVOID;
typedef const void *LPCVOID;
typedef char CHAR;
typedef char *LPSTR;
typedef const char *LPCSTR;
typedef wchar_t WCHAR;
typedef wchar_t *LPWSTR;
typedef const wchar_t *LPCWSTR;
typedef DWORD *LPDWORD;
typedef WORD *LPWORD;
typedef BOOL *LPBOOL;

/* The rest of a published set's own spellings for what is already above.
   Each of these is one line in <winnt.h>, <windef.h> or <minwindef.h> - the
   headers this file stands in for - and a header that names one is not
   asking for anything py2bin cannot compile: it is calling a type by the
   name the set gave it. Measured over a published set of 1350 headers,
   `WINBOOL` alone is where 290 of them stopped. */
typedef int WINBOOL;
#define VOID void
typedef DWORD LCID;
typedef WORD LANGID;
typedef DWORD ACCESS_MASK;
typedef DWORD COLORREF;
typedef COLORREF *LPCOLORREF;
/* A security identifier is variable-length and always reached through a
   pointer; the set says so itself, and gives the pointer no more type than
   this. */
typedef PVOID PSID;
typedef CHAR *PCHAR;
typedef UCHAR *PUCHAR;
typedef SHORT *PSHORT;
typedef USHORT *PUSHORT;
typedef INT *PINT;
typedef UINT *PUINT;
typedef LONG *PLONG;
typedef ULONG *PULONG;
typedef WORD *PWORD;
typedef DWORD *PDWORD;
typedef BOOL *PBOOL;
typedef FLOAT *PFLOAT;
typedef INT *LPINT;
typedef UINT *LPUINT;
typedef LONG *LPLONG;
typedef ULONG *LPULONG;
typedef CHAR *PSZ;
typedef HANDLE *PHANDLE;
typedef HANDLE *LPHANDLE;
typedef HANDLE *SPHANDLE;
typedef HANDLE GLOBALHANDLE;
typedef HANDLE LOCALHANDLE;
typedef HKEY *PHKEY;
typedef int HFILE;
typedef struct _SYSTEMTIME {
    WORD wYear;
    WORD wMonth;
    WORD wDayOfWeek;
    WORD wDay;
    WORD wHour;
    WORD wMinute;
    WORD wSecond;
    WORD wMilliseconds;
} SYSTEMTIME, *PSYSTEMTIME, *LPSYSTEMTIME;
#define _SYSTEMTIME_
typedef struct _LIST_ENTRY {
    struct _LIST_ENTRY *Flink;
    struct _LIST_ENTRY *Blink;
} LIST_ENTRY, *PLIST_ENTRY;

/* A handle is a pointer the program never looks inside, and the set writes
   every one of them with this. It gives each its own struct type only under
   STRICT, which is a promise about type checking rather than about layout:
   either way what the program holds is one pointer. */
#define DECLARE_HANDLE(name) typedef HANDLE name
#define ANYSIZE_ARRAY 1
/* What is left of the 16-bit memory models. Every set still writes them on
   a pointer here and there, in both cases, and every set defines all four to
   nothing. */
#define FAR
#define NEAR
#define far
#define near
#define FIELD_OFFSET(type, field) ((LONG)(LONG_PTR)&(((type *)0)->field))

/* What a set writes in front of a function the loader binds. Each names the
   library the declaration came from and each expands to `dllimport` where a
   compiler has it; py2bin binds an import by finding the name in a DLL's own
   export table, so none of them has anything to say here. */
#define WINBASEAPI
#define WINUSERAPI
#define WINGDIAPI
#define WINADVAPI
#define WINCOMMCTRLAPI
#define NTSYSAPI
#define NTSYSCALLAPI
#define NTAPI
/* A function the set writes out in the header itself rather than importing.
   `static` because py2bin compiles one translation unit: that is what a
   definition in a header has to be to belong to it. */
#define FORCEINLINE static __inline
#define DECLSPEC_SELECTANY
#define DECLSPEC_NORETURN
#define DECLSPEC_NOTHROW
#define DECLSPEC_NOINLINE
#define DECLSPEC_DEPRECATED
#define DECLSPEC_CACHEALIGN
/* Not defined away, on purpose. This one decides where a member sits, and a
   struct laid out differently from the one written runs and is wrong with
   nothing said - so it is left spelled as the `__declspec` it is, and the
   refusal a program gets names alignment rather than the macro. */
#define DECLSPEC_ALIGN(bytes) __declspec(align(bytes))

/* Narrow or wide, decided by UNICODE exactly as the replaced header decides
   it. `TEXT()` has to put an L in front of a literal in the wide case, which
   is a paste and so needs the second macro to expand its argument first. */
#ifdef UNICODE
typedef WCHAR TCHAR;
#define __TEXT(quoted) L##quoted
#else
typedef char TCHAR;
#define __TEXT(quoted) quoted
#endif
typedef TCHAR *PTSTR;
typedef TCHAR *LPTSTR;
typedef const TCHAR *PCTSTR;
typedef const TCHAR *LPCTSTR;
#define TEXT(quoted) __TEXT(quoted)

/* The rest of the handles <windef.h> and <minwindef.h> declare, in the
   spelling they declare them with. Each is a pointer nothing looks inside,
   and a header that names one wants a parameter type. */
DECLARE_HANDLE(HHOOK);
DECLARE_HANDLE(HACCEL);
DECLARE_HANDLE(HCOLORSPACE);
DECLARE_HANDLE(HGLRC);
DECLARE_HANDLE(HDESK);
DECLARE_HANDLE(HENHMETAFILE);
DECLARE_HANDLE(HMETAFILE);
DECLARE_HANDLE(HFONT);
DECLARE_HANDLE(HPALETTE);
DECLARE_HANDLE(HPEN);
DECLARE_HANDLE(HMONITOR);
#define HMONITOR_DECLARED 1
DECLARE_HANDLE(HWINEVENTHOOK);
DECLARE_HANDLE(HKL);
DECLARE_HANDLE(HRSRC);
DECLARE_HANDLE(HTASK);
DECLARE_HANDLE(HWINSTA);
typedef void *HGDIOBJ;

/* And the rest of the spellings for what is already here. `PWCHAR` is a
   wide string the program may write into; `DWORDLONG` is the set's name for
   the 64-bit unsigned it already has; a security descriptor and a security
   identifier are variable-length and reached through a pointer the set gives
   no more type than this. */
typedef WCHAR *PWCHAR;
typedef WCHAR *LPWCH;
typedef const WCHAR *LPCWCH;
typedef ULONGLONG DWORDLONG;
typedef DWORDLONG *PDWORDLONG;
typedef DWORD SECURITY_INFORMATION;
typedef DWORD *PSECURITY_INFORMATION;
typedef PVOID PSECURITY_DESCRIPTOR;
typedef ULONG_PTR KAFFINITY;
typedef DWORD FOURCC;
typedef FILETIME *PFILETIME;
typedef FILETIME *LPFILETIME;
typedef SYSTEMTIME *LPSYSTEMTIME;
typedef LUID *PLUID;
typedef LARGE_INTEGER *PLARGE_INTEGER;
typedef ULARGE_INTEGER *PULARGE_INTEGER;
/* Two structs from <minwinbase.h> that a header names in a prototype far
   more often than a program fills one in. Both are laid out as the platform
   lays them: three fields and a pointer that has to be eight-aligned, so
   `SECURITY_ATTRIBUTES` is 24 bytes; and in `OVERLAPPED` the union is one
   name for the pair of DWORDs and another for a pointer over the same eight
   bytes, which is what makes it 32. */
typedef struct _SECURITY_ATTRIBUTES {
    DWORD nLength;
    LPVOID lpSecurityDescriptor;
    BOOL bInheritHandle;
} SECURITY_ATTRIBUTES, *PSECURITY_ATTRIBUTES, *LPSECURITY_ATTRIBUTES;
#define _SECURITY_ATTRIBUTES_
typedef struct _OVERLAPPED {
    ULONG_PTR Internal;
    ULONG_PTR InternalHigh;
    union {
        struct {
            DWORD Offset;
            DWORD OffsetHigh;
        } DUMMYSTRUCTNAME;
        PVOID Pointer;
    } DUMMYUNIONNAME;
    HANDLE hEvent;
} OVERLAPPED, *LPOVERLAPPED;

#define INVALID_HANDLE_VALUE ((HANDLE)-1)
#define STD_INPUT_HANDLE ((DWORD)-10)
#define STD_OUTPUT_HANDLE ((DWORD)-11)
#define STD_ERROR_HANDLE ((DWORD)-12)

#define GENERIC_READ 0x80000000
#define GENERIC_WRITE 0x40000000
#define FILE_SHARE_READ 0x00000001
#define FILE_SHARE_WRITE 0x00000002
#define CREATE_ALWAYS 2
#define CREATE_NEW 1
#define OPEN_EXISTING 3
#define OPEN_ALWAYS 4
#define TRUNCATE_EXISTING 5
#define FILE_ATTRIBUTE_NORMAL 0x00000080

#define CP_ACP 0
#define CP_UTF8 65001

#define MB_OK 0x00000000
#define MB_OKCANCEL 0x00000001
#define MB_YESNO 0x00000004
#define MB_ICONERROR 0x00000010
#define MB_ICONWARNING 0x00000030
#define MB_ICONINFORMATION 0x00000040
#define IDOK 1
#define IDCANCEL 2
#define IDYES 6
#define IDNO 7

#define SM_CXSCREEN 0
#define SM_CYSCREEN 1

#define FOREGROUND_BLUE 0x0001
#define FOREGROUND_GREEN 0x0002
#define FOREGROUND_RED 0x0004
#define FOREGROUND_INTENSITY 0x0008
#define BACKGROUND_BLUE 0x0010
#define BACKGROUND_GREEN 0x0020
#define BACKGROUND_RED 0x0040
#define BACKGROUND_INTENSITY 0x0080

/* Everything below is imported from a DLL by the loader. The set is what
   py2bin has vetted signatures for; anything else is a name this header does
   not declare, and the compiler says so rather than pretending. */
extern void Sleep(DWORD);
extern DWORD GetTickCount(void);
extern DWORD GetTickCount64(void);
extern DWORD GetLastError(void);
extern void SetLastError(DWORD);
extern DWORD GetCurrentProcessId(void);
extern DWORD GetCurrentThreadId(void);
extern HANDLE GetStdHandle(DWORD);
extern BOOL CloseHandle(HANDLE);
extern BOOL WriteFile(HANDLE, LPCVOID, DWORD, LPDWORD, LPVOID);
extern BOOL ReadFile(HANDLE, LPVOID, DWORD, LPDWORD, LPVOID);
extern HANDLE CreateFileA(LPCSTR, DWORD, DWORD, LPVOID, DWORD, DWORD, HANDLE);
extern BOOL DeleteFileA(LPCSTR);
extern DWORD GetFileSize(HANDLE, LPDWORD);
extern BOOL SetConsoleOutputCP(UINT);
extern BOOL SetConsoleCP(UINT);
extern BOOL SetConsoleTextAttribute(HANDLE, WORD);
extern BOOL SetConsoleTitleA(LPCSTR);
extern DWORD GetModuleFileNameA(HMODULE, LPSTR, DWORD);
extern DWORD GetEnvironmentVariableA(LPCSTR, LPSTR, DWORD);
extern int MultiByteToWideChar(UINT, DWORD, LPCSTR, int, LPWSTR, int);
extern int WideCharToMultiByte(UINT, DWORD, LPCWSTR, int, LPSTR, int, LPCSTR, LPBOOL);
extern BOOL QueryPerformanceCounter(LPVOID);
extern BOOL QueryPerformanceFrequency(LPVOID);
extern int MessageBoxA(HWND, LPCSTR, LPCSTR, UINT);
extern int MessageBoxW(HWND, LPCWSTR, LPCWSTR, UINT);
extern int GetSystemMetrics(int);
extern BOOL SetWindowPos(HWND, HWND, int, int, int, int, UINT);

/* A DLL somebody else wrote: load it, ask for the entry point, call the
   pointer. This is how a vendor component is reached - WebView2Loader.dll
   is one - without naming anybody's product in py2bin's import table. */
typedef int (*FARPROC)(void);
/* The set's other two names for the same thing, from when they were not the
   same thing. */
typedef FARPROC NEARPROC;
typedef FARPROC PROC;
extern HMODULE LoadLibraryW(LPCWSTR);
extern HMODULE LoadLibraryA(LPCSTR);
extern BOOL FreeLibrary(HMODULE);
extern FARPROC GetProcAddress(HMODULE, LPCSTR);
extern HMODULE GetModuleHandleA(LPCSTR);

/* COM. Calling through a vtable is something py2bin has always been able to
   express; these are how a program comes by the pointer to call it on. */
/* HRESULT, GUID, BSTR and the S_/E_ codes come from <wtypes.h>, which is
   py2bin's own too. Written out again here they were the same types under
   the same names twice, and a program that included both was told its GUID
   was defined twice - which it was, by us. */
#include <wtypes.h>
/* The vendor's <windows.h> includes <winerror.h>, and a set that has one
   relies on that order: <urlmon.h> writes `#ifndef E_PENDING` around its own
   spelling, which only does its job if the real one has been read already.
   Taken where a fetch has brought one down and skipped where it has not,
   which is what __has_include is for. */
#if __has_include(<winerror.h>)
#include <winerror.h>
#endif

/* The SDK defines these to nothing when the reader is not a C++ compiler,
   which py2bin is not: it defines no __cplusplus, so a generated header
   takes its C branch throughout. */
#ifndef DEFINE_ENUM_FLAG_OPERATORS
#define DEFINE_ENUM_FLAG_OPERATORS(T)
#endif
#ifndef DECLSPEC_XFGVIRT
#define DECLSPEC_XFGVIRT(base, func)
#endif
#ifndef EXTERN_C
#define EXTERN_C
#endif
#ifndef DECLSPEC_IMPORT
#define DECLSPEC_IMPORT
#endif
/* COM has one home here, which is <objbase.h>: the entry points, the
   threading models and the class contexts are declared there, and declaring
   them again under this name was the same functions twice. */
#include <objbase.h>
#ifndef CLSCTX_LOCAL_SERVER
#define CLSCTX_LOCAL_SERVER 0x4
#endif

extern HRESULT CoInitialize(LPVOID);
extern BSTR SysAllocString(LPCWSTR);
extern void SysFreeString(BSTR);
extern UINT SysStringLen(BSTR);
extern BOOL MessageBeep(UINT);
extern BOOL CreateDirectoryA(LPCSTR, LPVOID);
extern BOOL RemoveDirectoryA(LPCSTR);
extern BOOL MoveFileA(LPCSTR, LPCSTR);
extern DWORD GetFileAttributesA(LPCSTR);
extern DWORD SetFilePointer(HANDLE, LONG, LPVOID, DWORD);
extern DWORD GetCurrentDirectoryA(DWORD, LPSTR);

#define INVALID_FILE_ATTRIBUTES ((DWORD)-1)
#define FILE_ATTRIBUTE_DIRECTORY 0x00000010
#define FILE_BEGIN 0
#define FILE_CURRENT 1
#define FILE_END 2

/* Windowing. A program that wants a window rather than a console had no way
   to ask for one here: the vendor's headers declare all of this, and they
   cannot be compiled by anything but the two compilers they are written for.
   These are ordinary imports the loader binds, like everything above. */
typedef void *HINSTANCE;
typedef void *HICON;
typedef void *HCURSOR;
typedef void *HBRUSH;
typedef void *HMENU;
typedef void *HMODULE;
typedef void *WPARAM;
typedef void *LPARAM;
typedef void *LRESULT;
typedef unsigned short ATOM;

typedef struct tagPOINT { LONG x; LONG y; } POINT;
typedef struct tagRECT { LONG left; LONG top; LONG right; LONG bottom; } RECT;
/* And the pointer spellings of both, and the two the set writes with a
   trailing L - the same fields, under the names a header that came from an
   .idl uses. */
typedef RECT *PRECT;
typedef RECT *LPRECT;
typedef const RECT *LPCRECT;
typedef POINT *PPOINT;
typedef POINT *LPPOINT;
typedef struct _RECTL { LONG left; LONG top; LONG right; LONG bottom; }
    RECTL, *PRECTL, *LPRECTL;
typedef const RECTL *LPCRECTL;
typedef struct _POINTL { LONG x; LONG y; } POINTL, *PPOINTL;
typedef struct tagPOINTS { SHORT x; SHORT y; } POINTS, *PPOINTS, *LPPOINTS;
typedef struct tagSIZE { LONG cx; LONG cy; } SIZE, *PSIZE, *LPSIZE;
typedef SIZE SIZEL;
typedef SIZE *PSIZEL, *LPSIZEL;
typedef struct tagMSG {
    HWND hwnd; UINT message; WPARAM wParam; LPARAM lParam;
    DWORD time; POINT pt;
} MSG, *PMSG, *LPMSG;

/* The window procedure a program writes, and the loader calls back into. */
typedef LRESULT (*WNDPROC)(HWND, UINT, WPARAM, LPARAM);

typedef struct tagWNDCLASSEXW {
    UINT cbSize; UINT style; WNDPROC lpfnWndProc;
    int cbClsExtra; int cbWndExtra;
    HINSTANCE hInstance; HICON hIcon; HCURSOR hCursor; HBRUSH hbrBackground;
    LPCWSTR lpszMenuName; LPCWSTR lpszClassName; HICON hIconSm;
} WNDCLASSEXW;

/* What WM_NCCREATE carries: the window's own creation parameters, which is
   how a program written in classes finds its object from inside a window
   procedure - the procedure is static, so nothing else could. */
typedef struct tagCREATESTRUCTW {
    LPVOID lpCreateParams;
    HINSTANCE hInstance; HMENU hMenu; HWND hwndParent;
    int cy; int cx; int y; int x;
    LONG style; LPCWSTR lpszName; LPCWSTR lpszClass; DWORD dwExStyle;
} CREATESTRUCTW, *LPCREATESTRUCTW;

extern ATOM RegisterClassExW(LPVOID);
extern BOOL GetClassInfoExW(HINSTANCE, LPCWSTR, LPVOID);
extern DWORD GetModuleFileNameW(HMODULE, LPWSTR, DWORD);
extern BOOL UnregisterClassW(LPCWSTR, HINSTANCE);
extern HWND CreateWindowExW(DWORD, LPCWSTR, LPCWSTR, DWORD,
                            int, int, int, int,
                            HWND, HMENU, HINSTANCE, LPVOID);
extern BOOL DestroyWindow(HWND);
extern BOOL ShowWindow(HWND, int);
extern BOOL UpdateWindow(HWND);
extern BOOL GetMessageW(LPVOID, HWND, UINT, UINT);
extern BOOL PeekMessageW(LPVOID, HWND, UINT, UINT, UINT);
extern BOOL TranslateMessage(LPVOID);
extern LRESULT DispatchMessageW(LPVOID);
extern void PostQuitMessage(int);
extern LRESULT DefWindowProcW(HWND, UINT, WPARAM, LPARAM);
extern BOOL PostMessageW(HWND, UINT, WPARAM, LPARAM);
extern LRESULT SendMessageW(HWND, UINT, WPARAM, LPARAM);
extern BOOL GetClientRect(HWND, LPVOID);
extern BOOL GetWindowRect(HWND, LPVOID);
extern BOOL MoveWindow(HWND, int, int, int, int, BOOL);
extern BOOL SetWindowTextW(HWND, LPCWSTR);
extern HCURSOR LoadCursorW(HINSTANCE, LPCWSTR);
extern HICON LoadIconW(HINSTANCE, LPCWSTR);
/* LONG_PTR, not a pointer: the SDK says so, and a program storing its own
   object here casts to it rather than to `void *`. Wide enough to hold
   either, which is the point of the type. */
extern LONG_PTR SetWindowLongPtrW(HWND, int, LONG_PTR);
extern LONG_PTR GetWindowLongPtrW(HWND, int);
extern LONG_PTR SetWindowLongW(HWND, int, LONG);
extern LONG_PTR GetWindowLongW(HWND, int);
extern HMODULE GetModuleHandleW(LPCWSTR);
extern LPWSTR GetCommandLineW(void);
extern char *GetCommandLineA(void);

#define WS_OVERLAPPED 0x00000000
#define WS_CAPTION 0x00C00000
#define WS_SYSMENU 0x00080000
#define WS_THICKFRAME 0x00040000
#define WS_MINIMIZEBOX 0x00020000
#define WS_MAXIMIZEBOX 0x00010000
#define WS_VISIBLE 0x10000000
#define WS_CHILD 0x40000000
#define WS_OVERLAPPEDWINDOW 0x00CF0000
#define CW_USEDEFAULT ((int)0x80000000)
#define SW_HIDE 0
#define SW_SHOWNORMAL 1
#define SW_SHOW 5
#define WM_DESTROY 0x0002
#define WM_SIZE 0x0005
#define WM_CLOSE 0x0010
#define WM_QUIT 0x0012
#define WM_PAINT 0x000F
#define WM_KEYDOWN 0x0100
#define WM_COMMAND 0x0111
#define WM_USER 0x0400
#define GWLP_USERDATA (-21)
#define CS_VREDRAW 0x0001
#define CS_HREDRAW 0x0002
#define CS_DBLCLKS 0x0008
#define CS_OWNDC 0x0020
#define WM_NCCREATE 0x0081
#define WM_NCDESTROY 0x0082
#define WM_DPICHANGED 0x02E0
#define WM_GETMINMAXINFO 0x0024
#define WM_SETFOCUS 0x0007
#define WM_ERASEBKGND 0x0014
#define SW_SHOWDEFAULT 10
#define SW_SHOWMAXIMIZED 3
#define SW_SHOWMINIMIZED 2
#define SW_RESTORE 9
#define MB_ERR_INVALID_CHARS 0x00000008
#define WC_ERR_INVALID_CHARS 0x00000080

/* Per-monitor DPI. The context is an opaque handle whose well-known values
   are small negative numbers cast to it, which is how the SDK spells them. */
typedef HANDLE DPI_AWARENESS_CONTEXT;
#define DPI_AWARENESS_CONTEXT_UNAWARE ((DPI_AWARENESS_CONTEXT)-1)
#define DPI_AWARENESS_CONTEXT_SYSTEM_AWARE ((DPI_AWARENESS_CONTEXT)-2)
#define DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE ((DPI_AWARENESS_CONTEXT)-3)
#define DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 ((DPI_AWARENESS_CONTEXT)-4)
extern BOOL SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT);
extern UINT GetDpiForWindow(HWND);
#define IDC_ARROW ((LPCWSTR)32512)
#define IDI_APPLICATION ((LPCWSTR)32512)
#define COLOR_WINDOW 5
#define PM_REMOVE 0x0001

/* The two GDI shapes py2bin's own headers lean on, written here rather than
   pulled in: <oleidl.h> names LOGPALETTE in a prototype and takes it for
   granted. The pull this stood in for - `#if __has_include(<wingdi.h>)` -
   fired for any directory holding a wingdi.h at all, and a package brings
   that file along without its neighbours as often as not, so a hello-world
   that never named it stopped on a header wingdi.h includes and nobody had
   fetched. A program that wants the rest of GDI includes <wingdi.h> itself.
   Guarded with the names the published header guards them with, so that
   header stands down instead of defining them a second time. Sizes and
   offsets checked against clang on four targets: 4 and 8 bytes, aligned to
   1 and 2. */
#ifndef _PALETTEENTRY_DEFINED
#define _PALETTEENTRY_DEFINED
typedef void *HFONT;
typedef void *HPALETTE;
/* GDI's font description and text metrics, laid out as <wingdi.h> lays them:
   LOGFONTW is 92 bytes and TEXTMETRICW 60, and <ocidl.h>'s IFont hands
   both back. */
/* The face-name length <wingdi.h> defines; a header carrying its own
   fallback LOGFONT (<shtypes.h> does) tests for it. */
#define LF_FACESIZE 32
#define LF_FULLFACESIZE 64
typedef struct tagLOGFONTA {
    LONG lfHeight; LONG lfWidth; LONG lfEscapement; LONG lfOrientation; LONG lfWeight;
    BYTE lfItalic; BYTE lfUnderline; BYTE lfStrikeOut; BYTE lfCharSet;
    BYTE lfOutPrecision; BYTE lfClipPrecision; BYTE lfQuality; BYTE lfPitchAndFamily;
    CHAR lfFaceName[LF_FACESIZE];
} LOGFONTA, *PLOGFONTA, *LPLOGFONTA;
typedef struct tagLOGFONTW {
    LONG lfHeight; LONG lfWidth; LONG lfEscapement; LONG lfOrientation; LONG lfWeight;
    BYTE lfItalic; BYTE lfUnderline; BYTE lfStrikeOut; BYTE lfCharSet;
    BYTE lfOutPrecision; BYTE lfClipPrecision; BYTE lfQuality; BYTE lfPitchAndFamily;
    WCHAR lfFaceName[LF_FACESIZE];
} LOGFONTW, *PLOGFONTW, *LPLOGFONTW;
typedef struct tagTEXTMETRICA {
    LONG tmHeight; LONG tmAscent; LONG tmDescent; LONG tmInternalLeading; LONG tmExternalLeading;
    LONG tmAveCharWidth; LONG tmMaxCharWidth; LONG tmWeight; LONG tmOverhang;
    LONG tmDigitizedAspectX; LONG tmDigitizedAspectY;
    BYTE tmFirstChar; BYTE tmLastChar; BYTE tmDefaultChar; BYTE tmBreakChar;
    BYTE tmItalic; BYTE tmUnderlined; BYTE tmStruckOut; BYTE tmPitchAndFamily; BYTE tmCharSet;
} TEXTMETRICA, *PTEXTMETRICA, *LPTEXTMETRICA;
typedef struct tagTEXTMETRICW {
    LONG tmHeight; LONG tmAscent; LONG tmDescent; LONG tmInternalLeading; LONG tmExternalLeading;
    LONG tmAveCharWidth; LONG tmMaxCharWidth; LONG tmWeight; LONG tmOverhang;
    LONG tmDigitizedAspectX; LONG tmDigitizedAspectY;
    WCHAR tmFirstChar; WCHAR tmLastChar; WCHAR tmDefaultChar; WCHAR tmBreakChar;
    BYTE tmItalic; BYTE tmUnderlined; BYTE tmStruckOut; BYTE tmPitchAndFamily; BYTE tmCharSet;
} TEXTMETRICW, *PTEXTMETRICW, *LPTEXTMETRICW;
#ifdef UNICODE
typedef LOGFONTW LOGFONT;
typedef TEXTMETRICW TEXTMETRIC;
#else
typedef LOGFONTA LOGFONT;
typedef TEXTMETRICA TEXTMETRIC;
#endif
typedef struct tagPALETTEENTRY {
    BYTE peRed;
    BYTE peGreen;
    BYTE peBlue;
    BYTE peFlags;
} PALETTEENTRY, *PPALETTEENTRY, *LPPALETTEENTRY;
#endif
#ifndef _LOGPALETTE_DEFINED
#define _LOGPALETTE_DEFINED
typedef struct tagLOGPALETTE {
    WORD palVersion;
    WORD palNumEntries;
    PALETTEENTRY palPalEntry[1];
} LOGPALETTE, *PLOGPALETTE, *NPLOGPALETTE, *LPLOGPALETTE;
#endif
"""

#: The platform half of <filesystem>, in C. It lives here rather than in the
#: C++ header because `#ifdef` is read by *this* preprocessor, and the C++
#: translator runs before it - a `#ifdef _WIN32` written in a C++ header is
#: still there when the translator reads the file, and it sees both branches.
#: So the C++ side is the `path` class and nothing conditional, and every
#: question that depends on the platform is asked here.
_PY2BIN_FS_H = """
#ifdef _WIN32
#include <windows.h>

int __py2bin_fs_exists(const char *__p) {
    return GetFileAttributesA(__p) != INVALID_FILE_ATTRIBUTES;
}

int __py2bin_fs_is_directory(const char *__p) {
    unsigned int __held = GetFileAttributesA(__p);
    if (__held == INVALID_FILE_ATTRIBUTES) { return 0; }
    return (__held & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

long __py2bin_fs_size(const char *__p) {
    void *__handle;
    unsigned int __held;
    __handle = CreateFileA(__p, GENERIC_READ, FILE_SHARE_READ, 0,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (__handle == INVALID_HANDLE_VALUE) { return -1; }
    __held = GetFileSize(__handle, 0);
    CloseHandle(__handle);
    return (long)__held;
}

int __py2bin_fs_mkdir(const char *__p) { return CreateDirectoryA(__p, 0) != 0; }
int __py2bin_fs_rmdir(const char *__p) { return RemoveDirectoryA(__p) != 0; }
int __py2bin_fs_unlink(const char *__p) { return DeleteFileA(__p) != 0; }
int __py2bin_fs_rename(const char *__a, const char *__b) {
    return MoveFileA(__a, __b) != 0;
}
int __py2bin_fs_cwd(char *__into, int __room) {
    return (int)GetCurrentDirectoryA((unsigned int)__room, __into);
}

/* A path on Windows is UTF-16 and py2bin's `path` holds UTF-8, so a path
   that came from the kernel or is going back to it is converted rather than
   truncated: a user directory is as likely to hold a character outside
   ASCII as not, and dropping the high byte gives a name that does not
   exist. */
int __py2bin_fs_narrow(const wchar_t *__wide, char *__into, int __room) {
    int __got;
    __got = WideCharToMultiByte(65001, 0, __wide, -1, __into, __room, 0, 0);
    if (__got <= 0) { if (__room > 0) { __into[0] = 0; } return 0; }
    return __got - 1;
}

int __py2bin_fs_widen(const char *__narrow, wchar_t *__into, int __room) {
    int __got;
    __got = MultiByteToWideChar(65001, 0, __narrow, -1, __into, __room);
    if (__got <= 0) { if (__room > 0) { __into[0] = 0; } return 0; }
    return __got - 1;
}

#else

int __py2bin_fs_exists(const char *__p) {
    return __py2bin_access(__p, 0) == 0;
}

int __py2bin_fs_is_directory(const char *__p) {
    long __fd;
    long __back;
    char __probe[1];
    __fd = __py2bin_open(__p, 0, 0);
    if (__fd < 0) { return 0; }
    /* Reading a directory descriptor fails - EISDIR on Linux, EINVAL on
       macOS - and reading a file of any kind does not. That is enough to
       tell them apart without reading a `struct stat`, whose layout differs
       between the platforms this has to work on and between architectures
       on one of them. An empty file reads zero bytes, which is not a
       failure, so the test is on the sign and not on the count. */
    __back = __py2bin_read(__fd, __probe, 1);
    __py2bin_close(__fd);
    return __back < 0;
}

long __py2bin_fs_size(const char *__p) {
    long __fd;
    long __size;
    __fd = __py2bin_open(__p, 0, 0);
    if (__fd < 0) { return -1; }
    __size = __py2bin_lseek(__fd, 0, 2);
    __py2bin_close(__fd);
    return __size;
}

int __py2bin_fs_mkdir(const char *__p) {
    return __py2bin_mkdir(__p, 0755) == 0;
}
int __py2bin_fs_rmdir(const char *__p) { return __py2bin_rmdir(__p) == 0; }
int __py2bin_fs_unlink(const char *__p) { return __py2bin_unlink(__p) == 0; }
int __py2bin_fs_rename(const char *__a, const char *__b) {
    return __py2bin_rename(__a, __b) == 0;
}
int __py2bin_fs_cwd(char *__into, int __room) {
    /* No getcwd syscall is wired, and the one that exists differs enough
       between the kernels to be worth leaving alone. `.` is what a relative
       path is resolved against, which is the answer callers use it for. */
    if (__room > 1) { __into[0] = '.'; __into[1] = 0; return 1; }
    return 0;
}

/* Everywhere else a `wchar_t` holds one code point, so the conversion is
   UTF-8 against UTF-32 and is written out rather than asked of the
   platform. Both stop one short of the room they are given, for the
   terminator, and both answer with the length they wrote. */
int __py2bin_fs_narrow(const wchar_t *__wide, char *__into, int __room) {
    int __at;
    int __out;
    unsigned int __c;
    __at = 0;
    __out = 0;
    while (__wide[__at] != 0) {
        __c = (unsigned int)__wide[__at];
        if (__c < 0x80) {
            if (__out + 1 >= __room) { break; }
            __into[__out] = (char)__c; __out = __out + 1;
        } else if (__c < 0x800) {
            if (__out + 2 >= __room) { break; }
            __into[__out] = (char)(0xC0 | (__c >> 6));
            __into[__out + 1] = (char)(0x80 | (__c & 0x3F));
            __out = __out + 2;
        } else if (__c < 0x10000) {
            if (__out + 3 >= __room) { break; }
            __into[__out] = (char)(0xE0 | (__c >> 12));
            __into[__out + 1] = (char)(0x80 | ((__c >> 6) & 0x3F));
            __into[__out + 2] = (char)(0x80 | (__c & 0x3F));
            __out = __out + 3;
        } else {
            if (__out + 4 >= __room) { break; }
            __into[__out] = (char)(0xF0 | (__c >> 18));
            __into[__out + 1] = (char)(0x80 | ((__c >> 12) & 0x3F));
            __into[__out + 2] = (char)(0x80 | ((__c >> 6) & 0x3F));
            __into[__out + 3] = (char)(0x80 | (__c & 0x3F));
            __out = __out + 4;
        }
        __at = __at + 1;
    }
    if (__room > 0) { __into[__out] = 0; }
    return __out;
}

int __py2bin_fs_widen(const char *__narrow, wchar_t *__into, int __room) {
    int __at;
    int __out;
    unsigned int __c;
    int __more;
    __at = 0;
    __out = 0;
    while (__narrow[__at] != 0) {
        __c = (unsigned int)(unsigned char)__narrow[__at];
        __more = 0;
        if (__c >= 0xF0) { __c = __c & 0x07; __more = 3; }
        else if (__c >= 0xE0) { __c = __c & 0x0F; __more = 2; }
        else if (__c >= 0xC0) { __c = __c & 0x1F; __more = 1; }
        while (__more > 0) {
            __at = __at + 1;
            if (__narrow[__at] == 0) { __more = 0; break; }
            __c = (__c << 6) | ((unsigned int)(unsigned char)__narrow[__at] & 0x3F);
            __more = __more - 1;
        }
        if (__out + 1 >= __room) { break; }
        __into[__out] = (wchar_t)__c;
        __out = __out + 1;
        __at = __at + 1;
    }
    if (__room > 0) { __into[__out] = 0; }
    return __out;
}

#endif
"""

_CTYPE_H = """
/* The C locale, which is the only one py2bin has: these answer for ASCII and
   say no to everything above it rather than guessing at an encoding. */
int isdigit(int __c) { return __c >= 48 && __c <= 57; }
int isupper(int __c) { return __c >= 65 && __c <= 90; }
int islower(int __c) { return __c >= 97 && __c <= 122; }
int isalpha(int __c) { return isupper(__c) || islower(__c); }
int isalnum(int __c) { return isalpha(__c) || isdigit(__c); }
int isspace(int __c) {
    return __c == 32 || (__c >= 9 && __c <= 13);
}
int isblank(int __c) { return __c == 32 || __c == 9; }
int ispunct(int __c) {
    if (__c <= 32 || __c >= 127) { return 0; }
    return !isalnum(__c);
}
int isprint(int __c) { return __c >= 32 && __c < 127; }
int isgraph(int __c) { return __c > 32 && __c < 127; }
int iscntrl(int __c) { return __c < 32 || __c == 127; }
int isxdigit(int __c) {
    if (isdigit(__c)) { return 1; }
    if (__c >= 97 && __c <= 102) { return 1; }
    return __c >= 65 && __c <= 70;
}
int tolower(int __c) { return isupper(__c) ? __c + 32 : __c; }
int toupper(int __c) { return islower(__c) ? __c - 32 : __c; }
"""

_ASSERT_H = """
#include <stdio.h>
#include <stdlib.h>

/* A real assert, not a no-op: it says which condition failed and stops.
   py2bin cannot raise a signal, so it exits with the status a shell reports
   for one instead of aborting. Writing it as a call rather than an `if`
   keeps `assert(x);` a single statement wherever it is written. */
void __py2bin_assert(int __ok, const char *__text) {
    if (__ok) { return; }
    printf("Assertion failed: %s\\n", __text);
    abort();
}
#define assert(condition) __py2bin_assert((condition) ? 1 : 0, #condition)
"""

_FLOAT_H = """
#define FLT_RADIX 2
#define FLT_MANT_DIG 24
#define FLT_DIG 6
#define FLT_MIN_EXP (-125)
#define FLT_MAX_EXP 128
#define FLT_EPSILON 1.19209290e-07F
#define FLT_MIN 1.17549435e-38F
#define FLT_MAX 3.40282347e+38F
#define DBL_MANT_DIG 53
#define DBL_DIG 15
#define DBL_MIN_EXP (-1021)
#define DBL_MAX_EXP 1024
#define DBL_EPSILON 2.2204460492503131e-16
#define DBL_MIN 2.2250738585072014e-308
#define DBL_MAX 1.7976931348623157e+308
#define LDBL_MANT_DIG DBL_MANT_DIG
#define LDBL_DIG DBL_DIG
#define LDBL_EPSILON DBL_EPSILON
#define LDBL_MIN DBL_MIN
#define LDBL_MAX DBL_MAX
#define DECIMAL_DIG 17
"""

_STRING_H = """
#ifndef NULL
#define NULL ((void *)0)
#endif

unsigned long strlen(const char *__s) {
    unsigned long __n = 0;
    while (__s[__n] != 0) { __n = __n + 1; }
    return __n;
}

int strcmp(const char *__a, const char *__b) {
    unsigned long __i = 0;
    while (__a[__i] != 0 && __a[__i] == __b[__i]) { __i = __i + 1; }
    /* C fixes the sign and leaves the magnitude open; the difference is what
       every implementation returns, so a program that prints it prints the
       same thing here as it does anywhere else. Unsigned, because C compares
       these as unsigned char and a plain char is signed in this dialect. */
    return (int)(unsigned char)__a[__i] - (int)(unsigned char)__b[__i];
}

int strncmp(const char *__a, const char *__b, unsigned long __n) {
    unsigned long __i = 0;
    while (__i < __n) {
        if (__a[__i] != __b[__i]) {
            return (int)(unsigned char)__a[__i] - (int)(unsigned char)__b[__i];
        }
        if (__a[__i] == 0) { return 0; }
        __i = __i + 1;
    }
    return 0;
}

char *strcpy(char *__to, const char *__from) {
    unsigned long __i = 0;
    while (__from[__i] != 0) { __to[__i] = __from[__i]; __i = __i + 1; }
    __to[__i] = 0;
    return __to;
}

char *strncpy(char *__to, const char *__from, unsigned long __n) {
    unsigned long __i = 0;
    while (__i < __n && __from[__i] != 0) { __to[__i] = __from[__i]; __i = __i + 1; }
    /* C pads the rest with zeros and does not terminate a full copy. */
    while (__i < __n) { __to[__i] = 0; __i = __i + 1; }
    return __to;
}

char *strcat(char *__to, const char *__from) {
    unsigned long __at = strlen(__to);
    unsigned long __i = 0;
    while (__from[__i] != 0) { __to[__at + __i] = __from[__i]; __i = __i + 1; }
    __to[__at + __i] = 0;
    return __to;
}

char *strncat(char *__to, const char *__from, unsigned long __n) {
    unsigned long __at = strlen(__to);
    unsigned long __i = 0;
    while (__i < __n && __from[__i] != 0) {
        __to[__at + __i] = __from[__i];
        __i = __i + 1;
    }
    __to[__at + __i] = 0;
    return __to;
}

char *strchr(const char *__s, int __c) {
    unsigned long __i = 0;
    while (1) {
        if (__s[__i] == (char)__c) { return (char *)(__s + __i); }
        if (__s[__i] == 0) { return NULL; }
        __i = __i + 1;
    }
}

char *strrchr(const char *__s, int __c) {
    char *__found = NULL;
    unsigned long __i = 0;
    while (1) {
        if (__s[__i] == (char)__c) { __found = (char *)(__s + __i); }
        if (__s[__i] == 0) { return __found; }
        __i = __i + 1;
    }
}

char *strstr(const char *__hay, const char *__needle) {
    unsigned long __i = 0;
    unsigned long __j;
    if (__needle[0] == 0) { return (char *)__hay; }
    while (__hay[__i] != 0) {
        __j = 0;
        while (__needle[__j] != 0 && __hay[__i + __j] == __needle[__j]) {
            __j = __j + 1;
        }
        if (__needle[__j] == 0) { return (char *)(__hay + __i); }
        __i = __i + 1;
    }
    return NULL;
}

void *memcpy(void *__to, const void *__from, unsigned long __n) {
    unsigned char *__d = (unsigned char *)__to;
    const unsigned char *__s = (const unsigned char *)__from;
    unsigned long __i = 0;
    while (__i < __n) { __d[__i] = __s[__i]; __i = __i + 1; }
    return __to;
}

void *memmove(void *__to, const void *__from, unsigned long __n) {
    unsigned char *__d = (unsigned char *)__to;
    const unsigned char *__s = (const unsigned char *)__from;
    unsigned long __i;
    /* Backwards when they overlap the wrong way, which is the whole of what
       memmove promises over memcpy. */
    if (__d > __s) {
        __i = __n;
        while (__i > 0) { __i = __i - 1; __d[__i] = __s[__i]; }
        return __to;
    }
    __i = 0;
    while (__i < __n) { __d[__i] = __s[__i]; __i = __i + 1; }
    return __to;
}

void *memset(void *__to, int __value, unsigned long __n) {
    unsigned char *__d = (unsigned char *)__to;
    unsigned long __i = 0;
    while (__i < __n) { __d[__i] = (unsigned char)__value; __i = __i + 1; }
    return __to;
}

int memcmp(const void *__a, const void *__b, unsigned long __n) {
    const unsigned char *__x = (const unsigned char *)__a;
    const unsigned char *__y = (const unsigned char *)__b;
    unsigned long __i = 0;
    while (__i < __n) {
        if (__x[__i] != __y[__i]) { return (int)__x[__i] - (int)__y[__i]; }
        __i = __i + 1;
    }
    return 0;
}
"""

_WCHAR_H = """
#ifndef NULL
#define NULL ((void *)0)
#endif
#define WEOF ((wchar_t)-1)

unsigned long wcslen(const wchar_t *__s) {
    unsigned long __n = 0;
    while (__s[__n] != 0) { __n = __n + 1; }
    return __n;
}

int wcscmp(const wchar_t *__a, const wchar_t *__b) {
    unsigned long __i = 0;
    while (__a[__i] != 0 && __a[__i] == __b[__i]) { __i = __i + 1; }
    /* C fixes the sign and leaves the magnitude open. The difference is what
       every implementation returns, so a program that prints the result
       prints the same thing here as it does anywhere else. */
    return (int)__a[__i] - (int)__b[__i];
}

wchar_t *wcscpy(wchar_t *__to, const wchar_t *__from) {
    unsigned long __i = 0;
    while (__from[__i] != 0) { __to[__i] = __from[__i]; __i = __i + 1; }
    __to[__i] = 0;
    return __to;
}
"""

_STDLIB_H = f"#define __PY2BIN_ARENA_BYTES {ARENA_BYTES}UL\n" + """
#ifndef NULL
#define NULL ((void *)0)
#endif
#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

/* The bump pointer and the end of the reservation. Both zero until the first
   allocation, which is what makes the mapping happen on demand: a program
   that includes this header and never allocates reserves nothing.

   `size_t` and not `unsigned long`. Windows is LLP64, where a `long` is four
   bytes and a pointer is eight, so an address held in one is an address with
   its top half cut off - and the arena is mapped wherever the kernel likes,
   which on a 64-bit process is usually above the four gigabytes that fit.
   Every pointer this handed out was then a low address that belongs to
   nobody, which is a program that either faults or quietly writes over
   something else. `size_t` is the width of a pointer on every target py2bin
   has, which is the property being relied on. */
static size_t __py2bin_heap_bump = 0;
static size_t __py2bin_heap_end = 0;
static size_t __py2bin_heap_claimed = 0;

/* The bump moves with an atomic add, so two callers are handed two blocks
   rather than the same one. A read and then a write is not enough: both read
   the same address before either writes, and both get it.

   The reservation is claimed the same way. Whoever adds one to the claim and
   sees a zero does the mapping; everybody else waits for the end to be
   published, reading it atomically because that is the only read this
   compiler promises anything about. The add that publishes it releases, and
   the add that reads it acquires, so the base written before the publication
   is there for whoever sees it. */
void *malloc(size_t __n) {
    size_t __p;
    if ((size_t)__py2bin_atomic_add((long *)&__py2bin_heap_end, 0) == 0) {
        if (__py2bin_atomic_add((long *)&__py2bin_heap_claimed, 1) == 0) {
            __py2bin_heap_bump = (size_t)__py2bin_arena();
            __py2bin_atomic_add(
                (long *)&__py2bin_heap_end,
                (long)(__py2bin_heap_bump + __PY2BIN_ARENA_BYTES));
        } else {
            while ((size_t)__py2bin_atomic_add(
                       (long *)&__py2bin_heap_end, 0) == 0) { }
        }
    }
    /* Round up to 16, which is the alignment any C object may ask for, so
       every block this returns is aligned for every type. A request of zero
       still gets a distinct address, as C says it may. */
    __n = (__n + (size_t)15) & ~(size_t)15;
    if (__n == (size_t)0) __n = (size_t)16;
    /* Written as a subtraction so a size near the top of the range cannot
       wrap the sum past the end and be let through. */
    /* Taken first and checked after. The other order is a read of the bump,
       a comparison, and then a write - and between the read and the write
       another caller may have taken the same block. Overshooting the end
       costs the arena nothing that was not already gone: the only way past
       the check is that there was no room. */
    __p = (size_t)__py2bin_atomic_add((long *)&__py2bin_heap_bump, (long)__n);
    if (__p + __n > __py2bin_heap_end || __p + __n < __p) return NULL;
    return (void *)__p;
}

void *calloc(size_t __count, size_t __size) {
    size_t __total;
    unsigned char *__block;
    size_t __i;
    if (__count != (size_t)0 && __size > ((size_t)-1) / __count) return NULL;
    __total = __count * __size;
    __block = (unsigned char *)malloc(__total);
    if (__block == NULL) return NULL;
    /* The arena is zero-filled when it is mapped, but a block reused after a
       realloc is not, so this clears rather than assuming. */
    for (__i = (size_t)0; __i < __total; __i++) __block[__i] = 0;
    return (void *)__block;
}

void free(void *__block) {
    /* An arena does not reclaim. Saying so plainly is better than a free()
       that appears to work and silently does nothing about fragmentation. */
    (void)__block;
}

void *realloc(void *__block, size_t __size) {
    unsigned char *__old;
    unsigned char *__new;
    size_t __i;
    if (__block == NULL) return malloc(__size);
    __new = (unsigned char *)malloc(__size);
    if (__new == NULL) return NULL;
    /* Nothing records how big the old block was, so this copies the smaller
       of the two -- the new size -- and reads no more of the old block than
       the arena holds. Growing is exact; shrinking copies only what stays. */
    __old = (unsigned char *)__block;
    for (__i = (size_t)0; __i < __size; __i++) {
        if ((size_t)(__old + __i) >= __py2bin_heap_bump) break;
        __new[__i] = __old[__i];
    }
    return (void *)__new;
}

int abs(int __value) { return __value < 0 ? -__value : __value; }
long labs(long __value) { return __value < 0 ? -__value : __value; }
"""

#: How py2bin gives a program `NULL`. Guarded, because C says a redefinition
#: has to be identical and a vendored header spelling it `0` is just as valid
#: a null pointer constant - so whichever got there first keeps it, which is
#: what every real header does.
_NULL = "#ifndef NULL\n#define NULL ((void *)0)\n#endif\n"

#: `offsetof`, which is the standard way a program asks where a member sits -
#: and the standard way a program checks that a header's layout is what it
#: expects. Written as the expansion every implementation writes, because C
#: gives no other way to say it: the address of a member of the object at
#: address zero *is* the offset. Guarded like `NULL`, so a header that
#: defines its own keeps it.
_OFFSETOF = (
    "#ifndef offsetof\n"
    "#define offsetof(TYPE, MEMBER) ((unsigned long)&((TYPE *)0)->MEMBER)\n"
    "#endif\n"
)

#: The types a header reaches for when it wants the ones a platform is
#: expected to have. py2bin has no system to take them from, so it says what
#: they are for the target it is building for - which is the same thing every
#: other header here does. Each is guarded: a header that defines its own
#: first keeps it, the way `NULL` does.
_SYS_TYPES_H = """
/* No `ssize_t` here on purpose. It is one of the typedefs the compiler
   settles from the target's data model - see `typedefs_for` in
   py2bin.c_frontend - because how wide it is depends on which model is in
   force. Written out here as well it was `long`, which is right on the LP64
   targets and four bytes on Windows, where the model makes it eight: the two
   answers met and `<sys/types.h>` stopped compiling for Windows altogether.
   `size_t` is left out of this header for the same reason and always was. */
#ifndef __py2bin_off_t_defined
#define __py2bin_off_t_defined
typedef long off_t;
#endif
#ifndef __py2bin_time_t_defined
#define __py2bin_time_t_defined
typedef long time_t;
#endif
#ifndef __py2bin_clock_t_defined
#define __py2bin_clock_t_defined
typedef long clock_t;
#endif
#ifndef __py2bin_pid_t_defined
#define __py2bin_pid_t_defined
typedef int pid_t;
typedef int mode_t;
typedef unsigned int uid_t;
typedef unsigned int gid_t;
typedef unsigned long ino_t;
typedef unsigned long dev_t;
typedef unsigned long nlink_t;
#endif
/* The short names a socket header still asks for by tradition. */
#ifndef __py2bin_u_char_defined
#define __py2bin_u_char_defined
typedef unsigned char u_char;
typedef unsigned short u_short;
typedef unsigned int u_int;
typedef unsigned long u_long;
#endif
"""

#: `<time.h>`: the types and the shape of a broken-down time. The functions
#: are not here - py2bin has no clock to read without asking the system for
#: one - so a program that calls `time()` is told at the call rather than
#: given something that answers zero.
_TIME_H = """
#ifndef __py2bin_time_t_defined
#define __py2bin_time_t_defined
typedef long time_t;
#endif
#ifndef __py2bin_clock_t_defined
#define __py2bin_clock_t_defined
typedef long clock_t;
#endif
#ifndef __py2bin_tm_defined
#define __py2bin_tm_defined
struct tm {
    int tm_sec; int tm_min; int tm_hour;
    int tm_mday; int tm_mon; int tm_year;
    int tm_wday; int tm_yday; int tm_isdst;
};
#endif
#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC 1000000
#endif
"""



#: COM, as py2bin's own headers rather than as a fetch.
#:
#: `wtypes.h`, `unknwn.h`, `objidl.h` and the rest do not exist as files
#: anywhere: every open implementation of the Windows API generates them from
#: `.idl` with a tool that runs at build time, and the vendor's own set ships
#: inside a toolchain nothing can fetch. Checked against three of those
#: implementations - none publishes one.
#:
#: So they are written here, the way `<windows.h>` is. What COM actually *is*
#: is a struct whose first member points at a table of function pointers, and
#: py2bin builds those already: a class with virtual methods is exactly that
#: layout. These headers say so in C, so a program can declare an interface
#: and call through it with nothing underneath but py2bin.
_WTYPES_H = """
#ifndef __py2bin_wtypes_h
#define __py2bin_wtypes_h
/* COM's own types are the same shape wherever they are described, and this
   header is asked for on every target: py2bin's C++ <unknwn.h> asks it for
   HRESULT and GUID rather than writing them out again. So the platform's
   own header is taken where there is one and the handful of spellings it
   would have given are written out where there is not - which is what lets
   a program declare a COM interface and build for six machines. */
/* Before the platform's header and not after it: <windows.h> asks
   <objbase.h> for the COM entry points, <objbase.h> is written in HRESULT,
   and its own way back here is closed by the guard above. Whatever a header
   includes may include it back, so what it needs first has to be first. */
typedef long HRESULT;
/* The rest of the spellings <wtypesbase.h> and <wtypes.h> give: a status
   code, the tag of a VARIANT, the id of a method on a dispatch interface,
   and the id of a property. Each is one of the integers above under the name
   an .idl gave it. */
typedef long SCODE;
typedef SCODE *PSCODE;
typedef unsigned short VARTYPE;
typedef long DISPID;
typedef long MEMBERID;
typedef unsigned long PROPID;
typedef unsigned short VARIANT_BOOL;
typedef const wchar_t *LPCOLESTR;
typedef wchar_t *LPOLESTR;
typedef wchar_t OLECHAR;
typedef wchar_t *BSTR;
typedef double DATE;
/* The guard the set writes around its own copy, which every header that has
   one writes: py2bin supplies the struct, so it supplies the guard too, and
   a set's copy stands down rather than being reported as a second
   definition. */
#define GUID_DEFINED
typedef struct _GUID {
    unsigned int Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char Data4[8];
} GUID;

typedef GUID IID;
typedef GUID CLSID;
typedef const GUID *REFGUID;
typedef const GUID *REFIID;
typedef const GUID *REFCLSID;
typedef GUID FMTID;
typedef const GUID *REFFMTID;
typedef GUID *LPFMTID;
/* The pointer spellings <guiddef.h> gives, which is this file under the
   name a set asks for. A header taking a GUID it may write into says LPGUID
   and one that only reads says LPCGUID; both are how the set writes what is
   above, and a header that used either stopped as if it were a type nobody
   had declared. */
typedef GUID *LPGUID;
typedef const GUID *LPCGUID;
typedef IID *LPIID;
typedef CLSID *LPCLSID;

#ifdef _WIN32
#include <windows.h>
#else
typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned int DWORD;
typedef unsigned int UINT;
typedef int BOOL;
typedef long LONG;
typedef unsigned long ULONG;
typedef void *LPVOID;
typedef char *LPSTR;
typedef const char *LPCSTR;
typedef wchar_t *LPWSTR;
typedef const wchar_t *LPCWSTR;
typedef unsigned long long SIZE_T;
typedef struct _LARGE_INTEGER { long long QuadPart; } LARGE_INTEGER;
typedef struct _ULARGE_INTEGER { unsigned long long QuadPart; } ULARGE_INTEGER;
#endif



/* A generated header names its interface IDs with this. The SDK writes it
   two ways - a declaration everywhere, and a definition in the one file that
   defines INITGUID - because a program is several translation units and the
   value has to live in exactly one of them. py2bin has one, so the value
   always lives here: there is no other file for it to clash with, and a
   declaration with no definition anywhere would be a name nothing could
   resolve, py2bin having no linker either. */
typedef BSTR *LPBSTR;
/* The aggregate automation types <wtypes.h> declares, laid out as the SDK
   lays them: CY and DECIMAL are 8 and 16 bytes, BLOB and CLIPDATA hold a
   count and a pointer, SAFEARRAY is 32 bytes on x64. <propidl.h>'s
   PROPVARIANT is a union of every one of them. */
typedef union tagCY {
    struct { unsigned long Lo; long Hi; };
    long long int64;
} CY, *LPCY;
typedef struct tagDEC {
    unsigned short wReserved;
    union { struct { unsigned char scale; unsigned char sign; }; unsigned short signscale; };
    unsigned long Hi32;
    union { struct { unsigned long Lo32; unsigned long Mid32; }; unsigned long long Lo64; };
} DECIMAL, *LPDECIMAL;
typedef struct tagBLOB { unsigned long cbSize; unsigned char *pBlobData; } BLOB, *LPBLOB;
/* The guards the SDK's <wtypes.h> defines for these, so a header that
   carries its own copy - <ws2def.h> does - skips it. */
#define __BLOB_T_DEFINED
#define _tagBLOB_DEFINED
#define _BLOB_DEFINED
#define _LPBLOB_DEFINED
typedef struct tagBSTRBLOB { unsigned long cbSize; unsigned char *pData; } BSTRBLOB, *LPBSTRBLOB;
typedef struct tagCLIPDATA { unsigned long cbSize; long ulClipFmt; unsigned char *pClipData; } CLIPDATA;
typedef struct tagSAFEARRAYBOUND { unsigned long cElements; long lLbound; } SAFEARRAYBOUND, *LPSAFEARRAYBOUND;
typedef struct tagSAFEARRAY {
    unsigned short cDims;
    unsigned short fFeatures;
    unsigned long cbElements;
    unsigned long cLocks;
    void * pvData;
    SAFEARRAYBOUND rgsabound[1];
} SAFEARRAY, *LPSAFEARRAY;
/* The RPC wire shapes <wtypes.h> declares and <oleidl.h> and <ocidl.h> name
   in their remote variants: a count and the data, laid out as the SDK lays
   them. */
/* A property's identity: the GUID of its set and a number in it; twenty
   bytes, aligned to four. <propsys.h> and <propkey.h> speak in these. */
typedef struct _tagpropertykey { GUID fmtid; unsigned int pid; } PROPERTYKEY;
typedef const PROPERTYKEY *REFPROPERTYKEY;
typedef long long hyper;
typedef unsigned long long MIDL_uhyper;
typedef struct _BYTE_BLOB { unsigned long clSize; unsigned char abData[1]; } BYTE_BLOB, *UP_BYTE_BLOB;
typedef struct _WORD_BLOB { unsigned long clSize; unsigned short asData[1]; } WORD_BLOB, *UP_WORD_BLOB;
typedef struct _DWORD_BLOB { unsigned long clSize; unsigned long alData[1]; } DWORD_BLOB, *UP_DWORD_BLOB;
typedef struct _FLAGGED_BYTE_BLOB { unsigned long fFlags; unsigned long clSize; unsigned char abData[1]; } FLAGGED_BYTE_BLOB, *UP_FLAGGED_BYTE_BLOB;
typedef struct _FLAGGED_WORD_BLOB { unsigned long fFlags; unsigned long clSize; unsigned short asData[1]; } FLAGGED_WORD_BLOB, *UP_FLAGGED_WORD_BLOB;
typedef struct _BYTE_SIZEDARR { unsigned long clSize; unsigned char *pData; } BYTE_SIZEDARR;
typedef struct _SHORT_SIZEDARR { unsigned long clSize; unsigned short *pData; } WORD_SIZEDARR;
typedef struct _LONG_SIZEDARR { unsigned long clSize; unsigned long *pData; } DWORD_SIZEDARR;
typedef struct _HYPER_SIZEDARR { unsigned long clSize; hyper *pData; } HYPER_SIZEDARR;
#define DEFINE_GUID(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \\
    const GUID name = {l, w1, w2, {b1, b2, b3, b4, b5, b6, b7, b8}}
#define DEFINE_OLEGUID(name, l, w1, w2) \\
    DEFINE_GUID(name, l, w1, w2, 0xC0, 0, 0, 0, 0, 0, 0, 0x46)

#define IsEqualGUID(a, b) (__py2bin_same_guid(&(a), &(b)))
#define IsEqualIID(a, b) IsEqualGUID(a, b)
#define IsEqualCLSID(a, b) IsEqualGUID(a, b)

static int __py2bin_same_guid(const GUID *__a, const GUID *__b) {
    int __i;
    if (__a->Data1 != __b->Data1) { return 0; }
    if (__a->Data2 != __b->Data2) { return 0; }
    if (__a->Data3 != __b->Data3) { return 0; }
    for (__i = 0; __i < 8; __i = __i + 1) {
        if (__a->Data4[__i] != __b->Data4[__i]) { return 0; }
    }
    return 1;
}

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
#endif
"""

#: What a generated interface header spells its declarations with. Every one
#: of these is punctuation to a compiler that lays a vtable out itself: the
#: calling convention is the one py2bin uses for every call, and the rest say
#: things about linkage and documentation tools.
_RPCNDR_H = """
#ifndef __py2bin_rpcndr_h
#define __py2bin_rpcndr_h
#include <wtypes.h>

/* MIDL output checks this before it will compile, and says only "this stub
   requires an updated version of <rpcndr.h>" when it is missing. 500 is past
   every __REQUIRED_RPCNDR_H_VERSION__ a generated header asks for. */
/* `__uuidof(T)` is how a program written for one compiler asks for an
   interface's id. A generated header writes that id out beside the interface
   as `IID_T`, which is the same thing under a name any compiler can read. */
#ifndef __uuidof
#define __uuidof(T) (&IID_##T)
#endif

#define __RPCNDR_H_VERSION__ 500
#define __MIDL_user_allocate_free_DEFINED__

#include <sal.h>

/* The types a generated header names in its proxy and stub prototypes, at
   the end of the file. A proxy marshals a call to an object in another
   process; a program calling one in its own never reaches any of this, and
   the prototypes only ever take a pointer to them - so an opaque struct is
   the whole of what is needed, and is honest about the rest. */
typedef struct IRpcStubBuffer IRpcStubBuffer;
typedef struct IRpcChannelBuffer IRpcChannelBuffer;
typedef struct __py2bin_RPC_MESSAGE RPC_MESSAGE;
typedef RPC_MESSAGE *PRPC_MESSAGE;
typedef long RPC_STATUS;
typedef void *RPC_BINDING_HANDLE;
typedef void *handle_t;
/* What <rpcdce.h> gives a generated header, which asks for <rpc.h> and gets
   this file. An interface handle is the address of the description MIDL
   wrote for that interface and is only ever passed along, so the set gives
   it no more type than a pointer either. */
typedef void *RPC_IF_HANDLE;
/* An .idl says `boolean` and `byte` where C says how wide, and MIDL passes
   both straight through into the header it generates. */
typedef unsigned char boolean;
typedef unsigned char byte;

/* What a generated header puts before a vtable pointer. `const` in the SDK,
   because the table is never written; the spelling is all that differs. */
#define CONST_VTBL const

#define __RPC_STUB
#define __RPC_FAR
#define __RPC_API
#define __RPC_USER
#define RPC_ENTRY
#define STDMETHODCALLTYPE
#define STDMETHODVCALLTYPE
#define STDAPICALLTYPE
#define WINOLEAPI HRESULT
#define EXTERN_C
#define DECLSPEC_UUID(x)
#define DECLSPEC_NOVTABLE
#define MIDL_INTERFACE(x) struct
#define interface struct
#define BEGIN_INTERFACE
#define END_INTERFACE
#define PURE = 0
#define THIS_
#define THIS void
#define STDMETHOD(name) virtual HRESULT name
#define STDMETHOD_(type, name) virtual type name
#define STDMETHODIMP HRESULT
#define STDMETHODIMP_(type) type
#define STDAPI HRESULT
#define STDAPI_(type) type
#define STDAPIV HRESULT
#define STDAPIV_(type) type
#define WINOLEAUTAPI HRESULT
#define WINOLEAUTAPI_(type) type
#endif
"""

_OBJBASE_H = """
#ifndef __py2bin_objbase_h
#define __py2bin_objbase_h
#include <rpcndr.h>

#ifndef COINIT_APARTMENTTHREADED
#define COINIT_APARTMENTTHREADED 0x2
#endif
#ifndef COINIT_MULTITHREADED
#define COINIT_MULTITHREADED 0x0
#endif
#ifndef CLSCTX_INPROC_SERVER
#define CLSCTX_INPROC_SERVER 0x1
#endif
#ifndef CLSCTX_ALL
#define CLSCTX_ALL 0x17
#endif

extern HRESULT CoInitializeEx(void *reserved, unsigned int model);
extern void CoUninitialize(void);
extern HRESULT CoCreateInstance(REFCLSID clsid, void *outer,
                                unsigned int context, REFIID riid,
                                void **object);
extern void CoTaskMemFree(void *block);
extern void *CoTaskMemAlloc(SIZE_T bytes);
#endif
"""

#: Every SDK header that is a piece of <windows.h> is that header. Written
#: as an include so the text is entered once however many of its names a
#: program uses: a built-in is remembered by the name it was read under, and
#: the same content under two names would be read twice and redefine
#: everything in it.
_PART_OF_WINDOWS_H = "#include <windows.h>\n"

#: How a published C runtime asks for the varargs machinery, which py2bin has
#: under C's own names. `_ADDRESSOF` is there because the C++ half of that
#: header takes the address through a `reinterpret_cast`; in C it is `&`.
_VADEFS_H = """
#ifndef __py2bin_vadefs_h
#define __py2bin_vadefs_h
#define _INC_VADEFS
#include <stdarg.h>
#define _VA_LIST_DEFINED
#define _crt_va_start(list, last) va_start(list, last)
#define _crt_va_arg(list, type) va_arg(list, type)
#define _crt_va_end(list) va_end(list)
#define _crt_va_copy(to, from) va_copy(to, from)
#define _ADDRESSOF(v) (&(v))
#endif
"""

#: Which slice of the Windows API a program is being built against. Every set
#: has this file and every set agrees on what the numbers are; what differs is
#: only which one the build picks, and a build that picks none stops at the
#: first `#if WINAPI_FAMILY_PARTITION(...)` - a line that is not an expression
#: until something defines the macro. py2bin's answer is the desktop, which is
#: what every set defaults to and what a program compiled to a .exe is; a
#: program that wants another says so with `-D WINAPI_FAMILY=...`, which the
#: guards below leave alone.
_WINAPIFAMILY_H = """
#ifndef __py2bin_winapifamily_h
#define __py2bin_winapifamily_h
#define WINAPI_FAMILY_PC_APP 2
#define WINAPI_FAMILY_PHONE_APP 3
#define WINAPI_FAMILY_SYSTEM 4
#define WINAPI_FAMILY_SERVER 5
#define WINAPI_FAMILY_DESKTOP_APP 100
#define WINAPI_FAMILY_APP WINAPI_FAMILY_PC_APP
#ifndef WINAPI_FAMILY
#define WINAPI_FAMILY WINAPI_FAMILY_DESKTOP_APP
#endif
#ifndef WINAPI_PARTITION_DESKTOP
#define WINAPI_PARTITION_DESKTOP (WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP)
#endif
#ifndef WINAPI_PARTITION_APP
#define WINAPI_PARTITION_APP (WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP || \\
                              WINAPI_FAMILY == WINAPI_FAMILY_PC_APP || \\
                              WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP)
#endif
#ifndef WINAPI_PARTITION_PC_APP
#define WINAPI_PARTITION_PC_APP (WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP || \\
                                 WINAPI_FAMILY == WINAPI_FAMILY_PC_APP)
#endif
#ifndef WINAPI_PARTITION_PHONE_APP
#define WINAPI_PARTITION_PHONE_APP (WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP)
#endif
#ifndef WINAPI_PARTITION_SYSTEM
#define WINAPI_PARTITION_SYSTEM (WINAPI_FAMILY == WINAPI_FAMILY_SYSTEM || \\
                                 WINAPI_FAMILY == WINAPI_FAMILY_SERVER)
#endif
#define WINAPI_PARTITION_PHONE WINAPI_PARTITION_PHONE_APP
#define WINAPI_FAMILY_PARTITION(slice) slice
#endif
"""

_MINGW_H = """
#ifndef __py2bin_mingw_h
#define __py2bin_mingw_h
#define _INC__MINGW_H

/* The header the mingw-w64 set writes at the top of every file it has, and
   the one file in that set that does not exist: it is generated from
   `_mingw.h.in` by a configure step, so a fetch finds `_mingw.h.in` and no
   `_mingw.h` anywhere. What it holds is a description of the compiler
   reading it - which extensions it has, how it spells an attribute, how wide
   its `long` is - so py2bin is the one that knows the answers, and this is
   py2bin's answers.

   Nearly all of it is nothing. An attribute that asks for an optimisation,
   a deprecation warning or a calling convention has nothing to say to a
   compiler that has one convention and makes its own decisions about
   inlining, and a macro that expands to nothing is how a compiler without
   the extension has always been told about it. */

#define __MINGW_EXTENSION
#define __MINGW_NOTHROW
#define __MINGW_IMPORT extern
#define __MINGW_INTRIN_INLINE static
#define __MINGW_ATTRIB_NORETURN
#define __MINGW_ATTRIB_CONST
#define __MINGW_ATTRIB_MALLOC
#define __MINGW_ATTRIB_PURE
#define __MINGW_ATTRIB_UNUSED
#define __MINGW_ATTRIB_USED
#define __MINGW_ATTRIB_NONNULL(x)
#define __MINGW_ATTRIB_DEPRECATED
#define __MINGW_ATTRIB_DEPRECATED_MSG(x)
#define __MINGW_ATTRIB_DEPRECATED_MSVC2005
#define __MINGW_ASM_CRT_CALL(x)
#define __MINGW_SELECTANY
/* Asked in an `#if`, so leaving it undefined is not the same as answering
   no: an undefined name with an argument list after it is not an expression
   at all. py2bin is neither of those compilers, and says so. */
#define __MINGW_GNUC_PREREQ(major, minor) 0
#define __MINGW_MSC_PREREQ(major, minor) 0
#define __MINGW_FORTIFY_LEVEL 0
/* What a header puts on a printf-shaped function so a compiler can check the
   format string against the arguments. py2bin checks its own printf calls
   from the format string itself, and takes nothing from these. */
#define __MINGW_GNU_PRINTF(fmt, args)
#define __MINGW_GNU_SCANF(fmt, args)
#define __MINGW_MS_PRINTF(fmt, args)
#define __MINGW_MS_SCANF(fmt, args)
#define __MINGW_ATTRIB_DEPRECATED_STR(x)
#define __MINGW_ATTRIB_DEPRECATED_SEC_WARN
#define __MINGW_ATTRIB_NO_OPTIMIZE
#define __MINGW_BROKEN_INTERFACE(x)
#define __MINGW_PRAGMA_PARAM(x)
#define __MINGW_SEC_WARN_STR(x)
#define __MINGW_MSVC2005_DEPRECATE_STR(x)
#define __MINGW_CXX11_CONSTEXPR
#define __MINGW_CXX14_CONSTEXPR
#define __MINGW_POISON_NAME(x) x
#define __MINGW_ASM_CALL(x)
#define __MINGW_UCRT_ASM_CALL(x)

/* How the set spells the things a compiler is asked to do to a declaration.
   py2bin decides its own linkage, its own inlining and its own convention,
   so each of these says nothing. */
#define _CRTIMP
#define _CRTIMP2
#define _CRTIMP_ALTERNATIVE
#define _CRTIMP_NOIA64
#define _CRTIMP_PURE
#define _MCRTIMP
#define _MRTIMP2
#define _CRTDECL
#define __CRTDECL
#define __CRT_INLINE static
#define _CRT_ALIGN(x)
#define _CRT_DEPRECATE_TEXT(x)
#define _CRT_INSECURE_DEPRECATE(x)
#define _CRT_INSECURE_DEPRECATE_GLOBALS(x)
#define _CRT_INSECURE_DEPRECATE_MEMORY(x)
#define _CRT_MANAGED_HEAP_DEPRECATE
#define _CRT_OBSOLETE(x)
#define _CRT_BEGIN_C_HEADER
#define _CRT_END_C_HEADER
#define _CRT_UNUSED(x) (void)(x)
#define __CRT_UUID_DECL(type, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8)
#define _SECURECRT_FILL_BUFFER_PATTERN 0xFD
#define _TRUNCATE ((size_t)-1)
#define __CRT__NO_INLINE 1

/* Spellings of a type or a qualifier that only one compiler has. */
#define __LONG32 long
#define __int8 char
#define __int16 short
#define __int32 int
#define __int64 long long
#define __ptr32
#define __ptr64
#define __unaligned
#define UNALIGNED
#define __w64
#define __nothrow
#define __restrict__
#define __restrict_arr
#define __forceinline __inline

/* An anonymous member is one py2bin has, so these say to leave it
   anonymous - which is what the set does for every compiler that has them. */
#define _ANONYMOUS_UNION
#define _ANONYMOUS_STRUCT
#define __ANONYMOUS_DEFINED
#define _UNION_NAME(x)
#define _STRUCT_NAME(x)
#define DUMMYUNIONNAME
#define DUMMYUNIONNAME1
#define DUMMYUNIONNAME2
#define DUMMYUNIONNAME3
#define DUMMYUNIONNAME4
#define DUMMYUNIONNAME5
#define DUMMYUNIONNAME6
#define DUMMYUNIONNAME7
#define DUMMYUNIONNAME8
#define DUMMYUNIONNAME9
#define DUMMYSTRUCTNAME
#define DUMMYSTRUCTNAME1
#define DUMMYSTRUCTNAME2
#define DUMMYSTRUCTNAME3
#define DUMMYSTRUCTNAME4
#define DUMMYSTRUCTNAME5
#define __C89_NAMELESS
#define __C89_NAMELESSSTRUCTNAME
#define __C89_NAMELESSUNIONNAME

/* How the set writes "the narrow one or the wide one, whichever this build
   asked for". It keeps these in `_mingw_unicode.h`, which its own <windef.h>
   includes - and <windef.h> here is py2bin's <windows.h>, so a header that
   declared its types with `__MINGW_TYPEDEF_AW(MCI_OPEN_PARMS)` found nothing
   to expand it. UNICODE decides, the way it decides TCHAR. */
#ifndef _INC_CRT_UNICODE_MACROS
#ifdef UNICODE
#define _INC_CRT_UNICODE_MACROS 1
#define __MINGW_NAME_AW(func) func##W
#define __MINGW_NAME_AW_EXT(func, ext) func##W##ext
#define WINELIB_NAME_AW(func) func##W
#define __MINGW_NAME_UAW(func) func##_W
#define __MINGW_NAME_UAW_EXT(func, ext) func##_W_##ext
#define __MINGW_STRING_AW(str) L##str
#define __MINGW_PROCNAMEEXT_AW "W"
#else
#define _INC_CRT_UNICODE_MACROS 2
#define __MINGW_NAME_AW(func) func##A
#define __MINGW_NAME_AW_EXT(func, ext) func##A##ext
#define WINELIB_NAME_AW(func) func##A
#define __MINGW_NAME_UAW(func) func##_A
#define __MINGW_NAME_UAW_EXT(func, ext) func##_A_##ext
#define __MINGW_STRING_AW(str) str
#define __MINGW_PROCNAMEEXT_AW "A"
#endif
#define __MINGW_TYPEDEF_AW(type) typedef __MINGW_NAME_AW(type) type;
#define DECL_WINELIB_TYPE_AW(type) typedef WINELIB_NAME_AW(type) type;
#define __MINGW_TYPEDEF_UAW(type) typedef __MINGW_NAME_UAW(type) type;
#endif

#define __CRT_STRINGIZE(x) #x
#define _CRT_STRINGIZE(x) __CRT_STRINGIZE(x)
#define __CRT_WIDE(x) L ## x
#define _CRT_WIDE(x) __CRT_WIDE(x)

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef WINVER
#define WINVER _WIN32_WINNT
#endif
#ifndef __MSVCRT_VERSION__
#define __MSVCRT_VERSION__ 0x700
#endif
#define __USE_MINGW_ANSI_STDIO 0
#define _INT128_DEFINED
#define USE___UUIDOF 0
#define __DECLSPEC_SUPPORTED 1
#define _ARGMAX 100

#endif
"""

_EVENTTOKEN_H = """
#ifndef __py2bin_eventtoken_h
#define __py2bin_eventtoken_h
/* One struct, which a generated COM header then names two thousand times:
   the handle registering for an event hands back, so the registration can
   be taken off again. */
typedef struct EventRegistrationToken {
    long long value;
} EventRegistrationToken;
#endif
"""

_OAIDL_H = """
#ifndef __py2bin_oaidl_h
#define __py2bin_oaidl_h
#include <wtypes.h>
#include <unknwn.h>
/* What Automation passes a value in. Sixteen bytes on both Windows
   machines: a two-byte tag, six the SDK reserves, and eight of value -
   which is what every member of that union is, or fits in. Written as the
   layout rather than as the union, because the union's members are a
   hundred names for the same eight bytes and the layout is the part that
   has to be right. */
typedef unsigned short VARTYPE;
typedef struct IDispatch IDispatch;
typedef struct IRecordInfo IRecordInfo;
/* What Automation passes a value in, laid out as the SDK lays it: a two-byte
   tag, six reserved, then sixteen bytes of value - the record pair is the
   widest member - so twenty-four on both 64-bit Windows machines. The union
   is written out, since a program reads `lVal` or `bstrVal` by name, and
   `rgvarg[1]` is twenty-four bytes along. */
typedef struct tagVARIANT {
    union {
        struct {
            VARTYPE vt;
            unsigned short wReserved1;
            unsigned short wReserved2;
            unsigned short wReserved3;
            union {
                long long llVal; long lVal; unsigned char bVal; short iVal;
                float fltVal; double dblVal; VARIANT_BOOL boolVal; SCODE scode;
                CY cyVal; DATE date; BSTR bstrVal; IUnknown *punkVal;
                IDispatch *pdispVal; SAFEARRAY *parray; unsigned char *pbVal;
                short *piVal; long *plVal; long long *pllVal; float *pfltVal;
                double *pdblVal; VARIANT_BOOL *pboolVal; SCODE *pscode;
                CY *pcyVal; DATE *pdate; BSTR *pbstrVal; IUnknown **ppunkVal;
                IDispatch **ppdispVal; SAFEARRAY **pparray;
                struct tagVARIANT *pvarVal; void *byref; char cVal;
                unsigned short uiVal; unsigned long ulVal;
                unsigned long long ullVal; int intVal; unsigned int uintVal;
                DECIMAL *pdecVal; char *pcVal; unsigned short *puiVal;
                unsigned long *pulVal; unsigned long long *pullVal;
                int *pintVal; unsigned int *puintVal;
                struct { void *pvRecord; IRecordInfo *pRecInfo; };
            };
        };
        DECIMAL decVal;
    };
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
typedef DWORD LCID;
typedef long MEMBERID;
/* The arguments `Invoke` is handed, and what it fills in on failure: twenty-
   four and sixty-four bytes on x64. */
typedef struct __py2bin_DISPPARAMS {
    VARIANTARG *rgvarg;
    DISPID *rgdispidNamedArgs;
    unsigned int cArgs;
    unsigned int cNamedArgs;
} DISPPARAMS;
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
typedef struct ITypeInfo ITypeInfo;
typedef struct ITypeLib ITypeLib;
/* Automation's interface: four methods after IUnknown's three, in COM's
   order - the order is the layout. Both shapes, chosen as <unknwn.h>
   chooses: the class for the translator, the table for C. IErrorLog and
   IPropertyBag beside it, which <ocidl.h>'s IPersistPropertyBag takes. */
#ifdef __cplusplus
class IDispatch : public IUnknown {
public:
    virtual HRESULT GetTypeInfoCount(unsigned int *count) = 0;
    virtual HRESULT GetTypeInfo(unsigned int which, LCID locale, ITypeInfo **answered) = 0;
    virtual HRESULT GetIDsOfNames(REFIID riid, LPOLESTR *names, unsigned int count, LCID locale, DISPID *answered) = 0;
    virtual HRESULT Invoke(DISPID member, REFIID riid, LCID locale, unsigned short flags, DISPPARAMS *given, VARIANT *answered, EXCEPINFO *failed, unsigned int *wrong) = 0;
};
class IErrorLog : public IUnknown {
public:
    virtual HRESULT AddError(LPCOLESTR name, EXCEPINFO *failed) = 0;
};
class IPropertyBag : public IUnknown {
public:
    virtual HRESULT Read(LPCOLESTR name, VARIANT *value, IErrorLog *log) = 0;
    virtual HRESULT Write(LPCOLESTR name, VARIANT *value) = 0;
};
#else
typedef struct IDispatchVtbl {
    HRESULT (*QueryInterface)(IDispatch *, REFIID, void **);
    unsigned long (*AddRef)(IDispatch *);
    unsigned long (*Release)(IDispatch *);
    HRESULT (*GetTypeInfoCount)(IDispatch *, unsigned int *);
    HRESULT (*GetTypeInfo)(IDispatch *, unsigned int, LCID, ITypeInfo **);
    HRESULT (*GetIDsOfNames)(IDispatch *, REFIID, LPOLESTR *, unsigned int, LCID, DISPID *);
    HRESULT (*Invoke)(IDispatch *, DISPID, REFIID, LCID, unsigned short, DISPPARAMS *, VARIANT *, EXCEPINFO *, unsigned int *);
} IDispatchVtbl;
struct IDispatch { const IDispatchVtbl *lpVtbl; };
typedef struct IErrorLog IErrorLog;
typedef struct IPropertyBag IPropertyBag;
#endif
#define DISPATCH_METHOD 1
#define DISPATCH_PROPERTYGET 2
#define DISPATCH_PROPERTYPUT 4
#endif
"""

_UNKNWN_H = """
#ifndef __py2bin_unknwn_h_c
#define __py2bin_unknwn_h_c
#include <wtypes.h>
/* COM's root interface in the shape C sees one: a pointer to a table of
   function pointers, each taking the interface as its first argument. That
   is what the object is - the C++ spelling of it is the same bytes - and it
   is the branch a generated header takes here, py2bin defining no
   __cplusplus. Three slots, which is what IUnknown has. */
/* Both shapes, chosen the way a generated header chooses: the same object
   either way - a pointer to a table of function pointers is what a class
   with virtual methods is laid out as - and which spelling a program uses
   decides which it wants to see. */
#ifdef __cplusplus
class IUnknown {
public:
    virtual HRESULT QueryInterface(REFIID riid, void **object) = 0;
    virtual unsigned long AddRef() = 0;
    virtual unsigned long Release() = 0;
};
typedef IUnknown *LPUNKNOWN;
#else
typedef struct IUnknown IUnknown;
typedef struct IUnknownVtbl {
    HRESULT (*QueryInterface)(IUnknown *, REFIID, void **);
    unsigned long (*AddRef)(IUnknown *);
    unsigned long (*Release)(IUnknown *);
} IUnknownVtbl;
struct IUnknown { const IUnknownVtbl *lpVtbl; };
typedef IUnknown *LPUNKNOWN;
#endif
#endif
"""

_OBJIDL_H = """
#ifndef __py2bin_objidl_h
#define __py2bin_objidl_h
#include <wtypes.h>
#include <unknwn.h>
/* IStream, in the shape C sees a COM interface in: a pointer to a table of
   function pointers, each taking the interface as its first argument. That
   is what the object actually is - the C++ spelling of it is the same
   bytes - and it is the shape a generated header uses, because py2bin
   defines no __cplusplus and so the C branch of one is the branch taken.

   Slot order is the .idl's: IUnknown's three, ISequentialStream's two, then
   IStream's nine. A slot out of place is a call to the wrong function, and
   nothing reports it.

   The eight-byte value the SDK spells as a one-member union is written as
   the integer it is. A struct passed BY VALUE through a foreign vtable is
   the one thing py2bin cannot spell, and Seek takes one; on both Windows
   machines an eight-byte struct travels in the register an eight-byte
   integer travels in, so this is the same ABI and can be called. */
#ifndef __cplusplus
typedef struct IStream IStream;
typedef struct ISequentialStream ISequentialStream;
/* The other interfaces this header forward-declares, and the names the set
   spells a pointer to each of them with. Declared and not defined, which is
   what the set's own `#ifndef __IDataObject_FWD_DEFINED__` blocks do: a
   neighbouring header names these in a prototype far more often than
   anything calls one, and a pointer to an incomplete struct is a complete
   type. A program that does call one needs the vtable, and the header that
   declares that interface properly is the one to bring it - so this is
   honest about the difference rather than inventing a table whose slot
   order nothing here could check. */
typedef struct IDataObject IDataObject;
typedef struct IAdviseSink IAdviseSink;
typedef struct IBindCtx IBindCtx;
typedef struct IMoniker IMoniker;
typedef struct IStorage IStorage;
typedef struct ILockBytes ILockBytes;
typedef struct IPersist IPersist;
typedef struct IPersistStream IPersistStream;
typedef struct IPersistFile IPersistFile;
typedef struct IPersistStorage IPersistStorage;
typedef struct IRunningObjectTable IRunningObjectTable;
typedef struct IRootStorage IRootStorage;
typedef struct IMalloc IMalloc;
typedef struct IMarshal IMarshal;
typedef struct IMessageFilter IMessageFilter;
typedef struct IEnumUnknown IEnumUnknown;
typedef struct IEnumString IEnumString;
typedef struct IEnumMoniker IEnumMoniker;
typedef struct IEnumFORMATETC IEnumFORMATETC;
typedef struct IEnumSTATDATA IEnumSTATDATA;
typedef struct IEnumSTATSTG IEnumSTATSTG;
typedef struct IDataAdviseHolder IDataAdviseHolder;
typedef IDataObject *LPDATAOBJECT;
typedef IAdviseSink *LPADVISESINK;
typedef IBindCtx *LPBC;
typedef IBindCtx *LPBINDCTX;
typedef IMoniker *LPMONIKER;
typedef IStorage *LPSTORAGE;
typedef ILockBytes *LPLOCKBYTES;
typedef IPersist *LPPERSIST;
typedef IPersistStream *LPPERSISTSTREAM;
typedef IPersistFile *LPPERSISTFILE;
typedef IPersistStorage *LPPERSISTSTORAGE;
typedef IRunningObjectTable *LPRUNNINGOBJECTTABLE;
typedef IRootStorage *LPROOTSTORAGE;
typedef IMalloc *LPMALLOC;
typedef IMarshal *LPMARSHAL;
typedef IMessageFilter *LPMESSAGEFILTER;
typedef IEnumUnknown *LPENUMUNKNOWN;
typedef IEnumString *LPENUMSTRING;
typedef IEnumMoniker *LPENUMMONIKER;
typedef IEnumFORMATETC *LPENUMFORMATETC;
typedef IEnumSTATDATA *LPENUMSTATDATA;
typedef IEnumSTATSTG *LPENUMSTATSTG;
typedef IStream *LPSTREAM;
#endif

typedef struct __py2bin_STATSTG {
    LPOLESTR pwcsName;
    unsigned long type;
    unsigned long long cbSize;
    unsigned long mtime_low, mtime_high;
    unsigned long ctime_low, ctime_high;
    unsigned long atime_low, atime_high;
    unsigned long grfMode;
    unsigned long grfLocksSupported;
    CLSID clsid;
    unsigned long grfStateBits;
    unsigned long reserved;
} STATSTG;

/* Both shapes, the way <unknwn.h> gives IUnknown both. Same object, same
   slots; which spelling a program uses decides which it is shown. */
#ifdef __cplusplus
class ISequentialStream : public IUnknown {
public:
    virtual HRESULT Read(void *pv, unsigned long cb, unsigned long *read) = 0;
    virtual HRESULT Write(const void *pv, unsigned long cb,
                          unsigned long *written) = 0;
};
#else
typedef struct ISequentialStreamVtbl {
    HRESULT (*QueryInterface)(ISequentialStream *, REFIID, void **);
    unsigned long (*AddRef)(ISequentialStream *);
    unsigned long (*Release)(ISequentialStream *);
    HRESULT (*Read)(ISequentialStream *, void *, unsigned long, unsigned long *);
    HRESULT (*Write)(ISequentialStream *, const void *, unsigned long, unsigned long *);
} ISequentialStreamVtbl;
struct ISequentialStream { const ISequentialStreamVtbl *lpVtbl; };
#endif

#ifdef __cplusplus
class IStream : public ISequentialStream {
public:
    virtual HRESULT Seek(long long move, unsigned long origin,
                         unsigned long long *position) = 0;
    virtual HRESULT SetSize(unsigned long long size) = 0;
    virtual HRESULT CopyTo(IStream *other, unsigned long long cb,
                           unsigned long long *read,
                           unsigned long long *written) = 0;
    virtual HRESULT Commit(unsigned long flags) = 0;
    virtual HRESULT Revert() = 0;
    virtual HRESULT LockRegion(unsigned long long offset,
                               unsigned long long cb, unsigned long type) = 0;
    virtual HRESULT UnlockRegion(unsigned long long offset,
                                 unsigned long long cb, unsigned long type) = 0;
    virtual HRESULT Stat(STATSTG *out, unsigned long flag) = 0;
    virtual HRESULT Clone(IStream **out) = 0;
};
#else
typedef struct IStreamVtbl {
    HRESULT (*QueryInterface)(IStream *, REFIID, void **);
    unsigned long (*AddRef)(IStream *);
    unsigned long (*Release)(IStream *);
    HRESULT (*Read)(IStream *, void *, unsigned long, unsigned long *);
    HRESULT (*Write)(IStream *, const void *, unsigned long, unsigned long *);
    HRESULT (*Seek)(IStream *, long long, unsigned long, unsigned long long *);
    HRESULT (*SetSize)(IStream *, unsigned long long);
    HRESULT (*CopyTo)(IStream *, IStream *, unsigned long long,
                      unsigned long long *, unsigned long long *);
    HRESULT (*Commit)(IStream *, unsigned long);
    HRESULT (*Revert)(IStream *);
    HRESULT (*LockRegion)(IStream *, unsigned long long, unsigned long long,
                          unsigned long);
    HRESULT (*UnlockRegion)(IStream *, unsigned long long, unsigned long long,
                            unsigned long);
    HRESULT (*Stat)(IStream *, STATSTG *, unsigned long);
    HRESULT (*Clone)(IStream *, IStream **);
} IStreamVtbl;
struct IStream { const IStreamVtbl *lpVtbl; };
#endif

#define STREAM_SEEK_SET 0
#define STREAM_SEEK_CUR 1
#define STREAM_SEEK_END 2

/* Named in signatures and passed along, never called by the programs that
   name them: a generated header takes one of these and hands it back. Left
   incomplete rather than written out, because each of their tables is a
   dozen methods over structs whose layout would be being guessed at here -
   and a program that does want to call one is told the type is incomplete,
   which is true, rather than handed a table that might be wrong.

   Written out, with real tables, when something needs to call one. IStream
   above is the shape that takes. */
/* What a COM object passes data through. Transcribed from the set rather
   than written from memory, because the whole worth of a struct here is that
   every member sits where the platform puts it: `LONG lindex` is four bytes
   on Windows, and eight of them would move `tymed` and make the struct the
   wrong size to hand to anything.

   `hMetaFilePict` and the rest of the union are handles, which are pointers,
   and the union is as wide as the widest of them either way. */
typedef struct IDataObject IDataObject;
typedef struct IAdviseSink IAdviseSink;
typedef struct IBindCtx IBindCtx;
typedef struct IClassFactory IClassFactory;
typedef struct IDispatch IDispatch;
typedef struct IEnumFORMATETC IEnumFORMATETC;
typedef struct IEnumSTATDATA IEnumSTATDATA;
typedef struct IEnumString IEnumString;
typedef struct IEnumUnknown IEnumUnknown;
typedef struct IMalloc IMalloc;
typedef struct IMoniker IMoniker;
typedef struct IStorage IStorage;
typedef struct ITypeInfo ITypeInfo;
typedef struct ITypeLib ITypeLib;
typedef struct IRunningObjectTable IRunningObjectTable;
typedef struct IPersist IPersist;
typedef struct IMessageFilter IMessageFilter;
typedef struct IMarshal IMarshal;
/* The factory a COM class is created through: two methods after IUnknown's
   three, in COM's order. <ocidl.h>'s IClassFactory2 derives from it. */
#ifdef __cplusplus
class IClassFactory : public IUnknown {
public:
    virtual HRESULT CreateInstance(IUnknown *outer, REFIID riid, void **object) = 0;
    virtual HRESULT LockServer(BOOL lock) = 0;
};
#else
typedef struct IClassFactoryVtbl {
    HRESULT (*QueryInterface)(IClassFactory *, REFIID, void **);
    unsigned long (*AddRef)(IClassFactory *);
    unsigned long (*Release)(IClassFactory *);
    HRESULT (*CreateInstance)(IClassFactory *, IUnknown *, REFIID, void **);
    HRESULT (*LockServer)(IClassFactory *, BOOL);
} IClassFactoryVtbl;
struct IClassFactory { const IClassFactoryVtbl *lpVtbl; };
#endif
/* The interfaces the rest of the SDK derives from - <ocidl.h>'s
   IPersistMemory is an IPersist - written for the translator; C keeps the
   forward declarations above, as it always has. Each method list is COM's,
   in COM's order. */
#ifdef __cplusplus
class IPersist : public IUnknown {
public:
    virtual HRESULT GetClassID(CLSID *id) = 0;
};
class IPersistStream : public IPersist {
public:
    virtual HRESULT IsDirty() = 0;
    virtual HRESULT Load(IStream *from) = 0;
    virtual HRESULT Save(IStream *to, BOOL clear) = 0;
    virtual HRESULT GetSizeMax(ULARGE_INTEGER *size) = 0;
};
class IPersistFile : public IPersist {
public:
    virtual HRESULT IsDirty() = 0;
    virtual HRESULT Load(LPCOLESTR name, DWORD mode) = 0;
    virtual HRESULT Save(LPCOLESTR name, BOOL remember) = 0;
    virtual HRESULT SaveCompleted(LPCOLESTR name) = 0;
    virtual HRESULT GetCurFile(LPOLESTR *name) = 0;
};
class IPersistStorage : public IPersist {
public:
    virtual HRESULT IsDirty() = 0;
    virtual HRESULT InitNew(IStorage *in) = 0;
    virtual HRESULT Load(IStorage *from) = 0;
    virtual HRESULT Save(IStorage *to, BOOL same) = 0;
    virtual HRESULT SaveCompleted(IStorage *now) = 0;
    virtual HRESULT HandsOffStorage() = 0;
};
class IEnumUnknown : public IUnknown {
public:
    virtual HRESULT Next(unsigned long wanted, IUnknown **out, unsigned long *got) = 0;
    virtual HRESULT Skip(unsigned long count) = 0;
    virtual HRESULT Reset() = 0;
    virtual HRESULT Clone(IEnumUnknown **out) = 0;
};
class IEnumString : public IUnknown {
public:
    virtual HRESULT Next(unsigned long wanted, LPOLESTR *out, unsigned long *got) = 0;
    virtual HRESULT Skip(unsigned long count) = 0;
    virtual HRESULT Reset() = 0;
    virtual HRESULT Clone(IEnumString **out) = 0;
};
class IMalloc : public IUnknown {
public:
    virtual void *Alloc(SIZE_T size) = 0;
    virtual void *Realloc(void *block, SIZE_T size) = 0;
    virtual void Free(void *block) = 0;
    virtual SIZE_T GetSize(void *block) = 0;
    virtual int DidAlloc(void *block) = 0;
    virtual void HeapMinimize() = 0;
};
class IAdviseSink : public IUnknown {
public:
    virtual void OnDataChange(FORMATETC *format, STGMEDIUM *medium) = 0;
    virtual void OnViewChange(DWORD aspect, LONG index) = 0;
    virtual void OnRename(IMoniker *moniker) = 0;
    virtual void OnSave() = 0;
    virtual void OnClose() = 0;
};
#endif

typedef unsigned short CLIPFORMAT;
/* The handles `STGMEDIUM` chooses between. These are declared here rather
   than taken from <windows.h> because that header refuses a target that is
   not Windows, and the point of these COM headers is that an interface can
   be declared and built for six machines. Three of the five were written out
   here already; `HBITMAP` and `HGLOBAL` were left to <windows.h>, so
   <objidl.h> compiled on Windows and nowhere else. */
typedef void *HBITMAP;
typedef void *HGLOBAL;
typedef void *HMETAFILEPICT;
typedef void *HENHMETAFILE;
typedef void *HMETAFILE;

typedef struct tagDVTARGETDEVICE {
    DWORD tdSize;
    WORD tdDriverNameOffset;
    WORD tdDeviceNameOffset;
    WORD tdPortNameOffset;
    WORD tdExtDevmodeOffset;
    BYTE tdData[1];
} DVTARGETDEVICE;

typedef struct tagFORMATETC {
    CLIPFORMAT cfFormat;
    DVTARGETDEVICE *ptd;
    DWORD dwAspect;
    LONG lindex;
    DWORD tymed;
} FORMATETC;
typedef struct tagFORMATETC *LPFORMATETC;

typedef struct tagSTGMEDIUM {
    DWORD tymed;
    union {
        HBITMAP hBitmap;
        HMETAFILEPICT hMetaFilePict;
        HENHMETAFILE hEnhMetaFile;
        HGLOBAL hGlobal;
        LPOLESTR lpszFileName;
        IStream *pstm;
        IStorage *pstg;
    };
    IUnknown *pUnkForRelease;
} STGMEDIUM;
typedef struct tagSTGMEDIUM *LPSTGMEDIUM;
typedef STGMEDIUM uSTGMEDIUM;

typedef struct tagSTATDATA {
    FORMATETC formatetc;
    DWORD advf;
    IAdviseSink *pAdvSink;
    DWORD dwConnection;
} STATDATA;

#define TYMED_NULL 0
#define TYMED_HGLOBAL 1
#define TYMED_FILE 2
#define TYMED_ISTREAM 4
#define TYMED_ISTORAGE 8
#define TYMED_GDI 16
#define TYMED_MFPICT 32
#define TYMED_ENHMF 64

#define DVASPECT_CONTENT 1
#define DVASPECT_THUMBNAIL 2
#define DVASPECT_ICON 4
#define DVASPECT_DOCPRINT 8


#endif
"""


_SHELLAPI_H = """
#ifndef __py2bin_shellapi_h
#define __py2bin_shellapi_h
#include <windows.h>

/* What a program reaches for out of the shell: opening a document or a URL
   with whatever is registered for it. The rest of the header is the file
   operations, the tray and the drag-and-drop plumbing, none of which can be
   called without the structs they take - written out when something needs
   one, rather than guessed at now. */
extern HINSTANCE ShellExecuteW(HWND, LPCWSTR, LPCWSTR, LPCWSTR, LPCWSTR, int);
extern HINSTANCE ShellExecuteA(HWND, LPCSTR, LPCSTR, LPCSTR, LPCSTR, int);

#define SE_ERR_FNF 2
#define SE_ERR_PNF 3
#define SE_ERR_ACCESSDENIED 5
#define SE_ERR_OOM 8
#define SE_ERR_NOASSOC 31
#endif
"""

_SAL_H = """
#ifndef __py2bin_sal_h
#define __py2bin_sal_h
/* The annotations a platform header writes on its parameters. They say what
   a function does with a pointer - reads it, writes it, may be handed null -
   for a static analyser to check against. They are not types and they emit
   nothing, which is what they expand to under every compiler that is not
   that analyser, and what they expand to here. */
#define _In_
#define _In_opt_
#define _In_z_
#define _In_opt_z_
#define _Out_
#define _Out_opt_
#define _Out_writes_bytes_(n)
#define _Inout_
#define _Inout_opt_
#define _Outptr_
#define _Outptr_opt_
#define _Outptr_result_maybenull_
#define _COM_Outptr_
#define _COM_Outptr_opt_
#define _COM_Outptr_result_maybenull_
#define _Ret_maybenull_
#define _Ret_notnull_
#define _Result_nullonfailure_
#define _Field_size_(n)
#define _Field_size_opt_(n)
#define _In_reads_(n)
#define _In_reads_opt_(n)
#define _In_reads_bytes_(n)
#define _In_reads_bytes_opt_(n)
#define _Out_writes_(n)
#define _Out_writes_opt_(n)
#define _Out_writes_to_(n, c)
#define _Inout_updates_(n)
#define _Inout_updates_bytes_(n)
#define _Success_(expr)
#define _Check_return_
#define _Must_inspect_result_
#define _Reserved_
#define _Null_terminated_
#define _Notnull_
#define _Maybenull_
#define _Deref_out_
#define _Deref_opt_out_
#define _Analysis_noreturn_
#define _When_(c, a)
#define _Post_
#define _Pre_
#define __in
#define __out
#define __inout
#define __in_opt
#define __out_opt
#define __reserved
#define __deref_out
#define SAL_H
#endif
"""


#: The operators a header uses to ask a compiler what it can do. Each takes
#: an argument, so none of them can be left to the rule that turns an unknown
#: identifier into 0 - that leaves the `(` behind, and a standard C++ header
#: stops on its first line of feature detection.
_ASKS_WHAT_THIS_COMPILER_HAS = frozenset(
    {
        "__has_include",
        "__has_include_next",
        "__has_feature",
        "__has_extension",
        "__has_builtin",
        "__has_attribute",
        "__has_c_attribute",
        "__has_cpp_attribute",
        "__has_declspec_attribute",
        "__has_keyword",
        "__is_identifier",
        "__has_unique_object_representations",
    }
)

#: Where `--auto-fetch` keeps what it downloaded. Named here rather than
#: imported, because the fetcher imports this module for the list of headers
#: py2bin ships and the two would chase each other.
_FETCHED_INTO = ".py2bin-headers"

#: `typedef struct tagRECT { ... } RECT;` and `} RECT, *LPRECT;` - the name a
#: platform header gives a struct, which is the name a program writes.
_A_STRUCT_TYPEDEF = re.compile(
    r"\}\s*((?:\s*\*?\s*[A-Za-z_]\w*\s*,?)+)\s*;"
)


def platform_structs() -> "frozenset[str]":
    """Every name py2bin's own headers give to a struct.

    The C++ translator runs before the preprocessor, so it never sees these
    declarations and cannot tell `RECT` - a struct - from `HRESULT`, which is
    a `long`. It has to: a parameter of struct type is passed as a pointer in
    the C this emits, and one of arithmetic type is passed as it is written.
    Read from the headers rather than listed, so a type added to one of them
    is known here without being named twice.
    """

    global _PLATFORM_STRUCTS
    if _PLATFORM_STRUCTS is not None:
        return _PLATFORM_STRUCTS
    found: "set[str]" = set()
    for text in _BUILTIN_HEADERS.values():
        if not isinstance(text, str):
            continue
        for match in re.finditer(r"\btypedef\s+(?:struct|union)\b", text):
            opening = text.find("{", match.end())
            semicolon = text.find(";", match.end())
            if opening < 0 or (0 <= semicolon < opening):
                # `typedef struct IUnknown IUnknown;` - a name for a type
                # declared elsewhere, with no body here to read.
                continue
            depth = 0
            at = opening
            while at < len(text):
                if text[at] == "{":
                    depth += 1
                elif text[at] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                at += 1
            if at >= len(text):
                continue
            names = _A_STRUCT_TYPEDEF.match(text, at)
            if names is None:
                continue
            for spelled in names.group(1).split(","):
                spelled = spelled.strip()
                # Only the plain name: `*LPRECT` is a pointer to one, and a
                # parameter declared with it is a pointer already.
                if spelled and not spelled.startswith("*"):
                    found.add(spelled)
    _PLATFORM_STRUCTS = frozenset(found)
    return _PLATFORM_STRUCTS


#: Worked out once; the headers do not change while the process runs.
_PLATFORM_STRUCTS: "frozenset[str] | None" = None

_BUILTIN_HEADERS = {
    "sys/types.h": _SYS_TYPES_H,
    "time.h": _TIME_H,
    "wtypes.h": _WTYPES_H,
    "rpcndr.h": _RPCNDR_H,
    # MIDL output opens with <rpc.h> and <rpcndr.h> together. What a *caller*
    # needs from the first is what the second already gives: the calling
    # convention spellings and the interface macros. The RPC runtime itself
    # is for a program that serves an interface over a wire, which is not
    # what a program calling a COM object in its own process is doing.
    "rpc.h": "#include <rpcndr.h>\n",
    # The three a generated COM header names in its signatures. Written for
    # the C branch of one, which is the branch taken here: py2bin defines no
    # __cplusplus, so `#if defined(__cplusplus) && !defined(CINTERFACE)` is
    # false and the interface arrives as a table of function pointers - which
    # is what the object is, and what py2bin's C compiles.
    "sal.h": _SAL_H,
    # The shell's own header, which is a piece of the platform like the rest
    # and is included by a great deal that never calls anything in it.
    "shellapi.h": _SHELLAPI_H,
    # The one file in the mingw-w64 set that does not exist as a file: it is
    # generated by a configure step, and what it holds is a description of
    # the compiler reading it - which py2bin is the one that knows.
    "_mingw.h": _MINGW_H,
    # Which slice of the API this build wants, which is a decision rather
    # than a description of a machine - so py2bin makes it, the way it makes
    # the one _mingw.h holds.
    "winapifamily.h": _WINAPIFAMILY_H,
    # The same header under the name a generated one asks for.
    "guiddef.h": "#include <wtypes.h>\n",
    "unknwn.h": _UNKNWN_H,
    "specstrings.h": _SAL_H,
    "objidl.h": _OBJIDL_H,
    "oaidl.h": _OAIDL_H,
    "EventToken.h": _EVENTTOKEN_H,
    "eventtoken.h": _EVENTTOKEN_H,
    "objbase.h": _OBJBASE_H,
    "combaseapi.h": _OBJBASE_H,
    "ole2.h": _OBJBASE_H,
    "stdio.h": "#define EOF (-1)\n" + _NULL,
    # <stdlib.h> brings the heap, and brings it as C source for the same
    # reason <math.h> does: an allocator you can read is one you can check.
    # The compiler itself supplies exactly one thing, __py2bin_arena(), which
    # is a single anonymous mapping; everything below is ordinary C compiled
    # like any other. It is an arena - free() keeps its promise not to touch
    # what you hand it, and the memory comes back when the process ends.
    "stdlib.h": _STDLIB_H,
    "string.h": _STRING_H,
    "ctype.h": _CTYPE_H,
    "windows.h": _WINDOWS_H,
    # The SDK splits <windows.h> across a dozen files and a program is as
    # likely to name one of those as the whole. Each is py2bin's own
    # <windows.h>, written as an include rather than the same text under a
    # second name, so a program that asks for both gets it once.
    #
    # Not fetching them is the point. The published sets are written for a
    # compiler that is GCC or MSVC and say so: Wine's `winnt.h` runs a chain
    # of nine branches looking for one of those two paired with an
    # architecture, and where none holds it stops with "You must define
    # NtCurrentTeb() for your architecture" - every branch that would have
    # matched needing inline assembly or an MSVC intrinsic. py2bin is neither
    # compiler, so it brings its own, the way it does for COM.
    "winnt.h": _PART_OF_WINDOWS_H,
    "windef.h": _PART_OF_WINDOWS_H,
    "minwindef.h": _PART_OF_WINDOWS_H,
    "minwinbase.h": _PART_OF_WINDOWS_H,
    "winbase.h": _PART_OF_WINDOWS_H,
    "winuser.h": _PART_OF_WINDOWS_H,
    "basetsd.h": _PART_OF_WINDOWS_H,
    "py2bin_fs.h": _PY2BIN_FS_H,
    "assert.h": _ASSERT_H,
    "float.h": _FLOAT_H,
    "stddef.h": _NULL + _OFFSETOF,
    # A `va_list` is a pointer to the cells the call wrote its extra arguments
    # into, and the four names that walk one are compiled rather than called -
    # so this header is the typedef and nothing else.
    "stdarg.h": "typedef char *va_list;\n",
    # What a published C runtime asks for varargs with, and the one place in
    # such a set that says outright it does not know this compiler:
    #
    #   vadefs.h:27: #error VARARGS not implemented for this compiler
    #
    # It is written as a choice between two - `__builtin_va_list` for GCC,
    # `char *` plus pointer arithmetic for MSVC - and the second of those is
    # the shape py2bin already has. So this is py2bin's answer to the same
    # question, under the name the set asks it by, and the four `_crt_va_`
    # names hand straight back to the four the compiler implements.
    "vadefs.h": _VADEFS_H,
    # wchar_t, char16_t and char32_t are keywords in py2bin's C, the way they
    # are in C++, so these headers have no typedefs to give. What they do
    # bring is the handful of functions that go with them, written in C.
    "wchar.h": _WCHAR_H,
    "uchar.h": _NULL,
    "stdbool.h": "#define bool _Bool\n#define true 1\n#define false 0\n"
    "#define __bool_true_false_are_defined 1\n",
    # <math.h> supplies its functions as C source that py2bin then
    # compiles. Nothing is linked: the transcendentals are argument
    # reduction plus a polynomial, written in the same C the compiler
    # already accepts, while sqrt/fabs/floor/ceil/trunc remain single
    # hardware instructions.
    "math.h": _MATH_H,
    "limits.h": """
#define CHAR_BIT 8
#define SCHAR_MIN (-128)
#define SCHAR_MAX 127
#define UCHAR_MAX 255
#define CHAR_MIN SCHAR_MIN
#define CHAR_MAX SCHAR_MAX
#define MB_LEN_MAX 1
#define SHRT_MIN (-32768)
#define SHRT_MAX 32767
#define USHRT_MAX 65535
#define INT_MIN (-2147483647 - 1)
#define INT_MAX 2147483647
#define UINT_MAX 4294967295U
#define LONG_MIN (-9223372036854775807L - 1)
#define LONG_MAX 9223372036854775807L
#define ULONG_MAX 18446744073709551615UL
#define LLONG_MIN (-9223372036854775807LL - 1)
#define LLONG_MAX 9223372036854775807LL
#define ULLONG_MAX 18446744073709551615ULL
""",
    "stdint.h": """
#define INT8_MIN (-128)
#define INT8_MAX 127
#define UINT8_MAX 255
#define INT16_MIN (-32768)
#define INT16_MAX 32767
#define UINT16_MAX 65535
#define INT32_MIN (-2147483647 - 1)
#define INT32_MAX 2147483647
#define UINT32_MAX 4294967295U
#define INT64_MIN (-9223372036854775807LL - 1)
#define INT64_MAX 9223372036854775807LL
#define UINT64_MAX 18446744073709551615ULL
#define INTMAX_MIN INT64_MIN
#define INTMAX_MAX INT64_MAX
#define UINTMAX_MAX UINT64_MAX
#define INTPTR_MIN (-9223372036854775807L - 1)
#define INTPTR_MAX 9223372036854775807L
#define UINTPTR_MAX 18446744073709551615UL
#define PTRDIFF_MIN INTPTR_MIN
#define PTRDIFF_MAX INTPTR_MAX
#define SIZE_MAX UINTPTR_MAX
#define INT8_C(value) value
#define INT16_C(value) value
#define INT32_C(value) value
#define INT64_C(value) value ## LL
#define UINT8_C(value) value
#define UINT16_C(value) value
#define UINT32_C(value) value ## U
#define UINT64_C(value) value ## ULL
""",
    "inttypes.h": """
#define PRId8 "d"
#define PRIi8 "i"
#define PRIu8 "u"
#define PRIx8 "x"
#define PRIX8 "X"
#define PRId16 "d"
#define PRIi16 "i"
#define PRIu16 "u"
#define PRIx16 "x"
#define PRIX16 "X"
#define PRId32 "d"
#define PRIi32 "i"
#define PRIu32 "u"
#define PRIx32 "x"
#define PRIX32 "X"
#define PRId64 "lld"
#define PRIi64 "lli"
#define PRIu64 "llu"
#define PRIx64 "llx"
#define PRIX64 "llX"
#define PRIdMAX "lld"
#define PRIuMAX "llu"
#define PRIdPTR "ld"
#define PRIuPTR "lu"
""",
}


# --- the #if expression evaluator --------------------------------------------

_MASK = (1 << 64) - 1
#: ``#if`` arithmetic is done in intmax_t/uintmax_t, which are 64 bits wide here.
_SIGN = 1 << 63


def _signed(value: int) -> int:
    value &= _MASK
    return value - (1 << 64) if value & _SIGN else value


@dataclasses.dataclass(frozen=True, slots=True)
class _Number:
    value: int
    unsigned: bool


#: C's binary operators, tightest last. ``?:`` and the unary operators are
#: parsed on their own; assignment and the comma are not constant expressions.
_PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "|": 3,
    "^": 4,
    "&": 5,
    "==": 6,
    "!=": 6,
    "<": 7,
    ">": 7,
    "<=": 7,
    ">=": 7,
    "<<": 8,
    ">>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
}


class _Evaluator:
    """A C constant expression as ``#if`` sees it: 64-bit, integers only."""

    #: C11 5.2.4.1 asks a compiler to manage 63 levels of parenthesised
    #: expression; beyond this one py2bin says so rather than exhausting the
    #: Python stack it is written on.
    MAXIMUM_NESTING = 200

    def __init__(
        self,
        tokens: list[PPToken],
        at: PPToken,
        look_in: "list[Path] | None" = None,
    ):
        self.tokens = tokens
        self.index = 0
        self.at = at
        self.depth = 0
        #: Where `__has_include` looks, in order. The directory of the file
        #: asking comes first, as it does for an include written with quotes.
        self.look_in = look_in if look_in is not None else []

    def deeper(self) -> None:
        self.depth += 1
        if self.depth > self.MAXIMUM_NESTING:
            self.error(
                f"this #if expression nests more than {self.MAXIMUM_NESTING} deep"
            )

    def error(self, message: str, token: PPToken | None = None):
        located = token or (self.tokens[self.index] if self.index < len(self.tokens) else self.at)
        raise CCompileError(located.origin, located.line, located.column, message)

    @property
    def token(self) -> PPToken | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def spelling(self) -> str:
        token = self.token
        return "" if token is None else token.spelling

    def run(self) -> int:
        value = self.conditional(True)
        if self.token is not None:
            self.error(f"unexpected {self.spelling()!r} in a #if expression")
        return value.value

    def conditional(self, evaluate: bool) -> _Number:
        self.deeper()
        try:
            return self._conditional(evaluate)
        finally:
            self.depth -= 1

    def _conditional(self, evaluate: bool) -> _Number:
        condition = self.binary(0, evaluate)
        if self.spelling() != "?":
            return condition
        self.index += 1
        taken = condition.value != 0
        left = self.conditional(evaluate and taken)
        if self.spelling() != ":":
            self.error("this '?' has no ':'")
        self.index += 1
        right = self.conditional(evaluate and not taken)
        chosen = left if taken else right
        return _Number(chosen.value, left.unsigned or right.unsigned)

    def binary(self, minimum: int, evaluate: bool) -> _Number:
        """Every binary operator, by precedence climbing.

        One function rather than one per level, so that a parenthesised
        expression costs a handful of Python frames instead of a dozen. All of
        C's binary operators here are left-associative, so the right operand is
        parsed at the next precedence up.
        """

        left = self.unary(evaluate)
        while True:
            token = self.token
            if token is None or token.kind != "punctuator":
                break
            precedence = _PRECEDENCE.get(token.spelling)
            if precedence is None or precedence < minimum:
                break
            self.index += 1
            if token.spelling == "&&":
                right = self.binary(precedence + 1, evaluate and left.value != 0)
                left = _Number(int(left.value != 0 and right.value != 0), False)
                continue
            if token.spelling == "||":
                right = self.binary(precedence + 1, evaluate and left.value == 0)
                left = _Number(int(left.value != 0 or right.value != 0), False)
                continue
            right = self.binary(precedence + 1, evaluate)
            left = self.apply(token.spelling, left, right, evaluate, token)
        return left

    def apply(
        self,
        operator: str,
        left: _Number,
        right: _Number,
        evaluate: bool,
        token: PPToken,
    ) -> _Number:
        if operator in ("<<", ">>"):
            if evaluate and not 0 <= right.value < 64:
                self.error(
                    f"shifting by {right.value} is undefined in C", token
                )
            count = right.value if 0 <= right.value < 64 else 0
            if operator == "<<":
                return _Number(self.wrap(left.value << count, left.unsigned), left.unsigned)
            return _Number(self.wrap(left.value >> count, left.unsigned), left.unsigned)
        unsigned = left.unsigned or right.unsigned
        first = left.value & _MASK if unsigned else left.value
        second = right.value & _MASK if unsigned else right.value
        if operator in ("==", "!=", "<", ">", "<=", ">="):
            result = {
                "==": first == second,
                "!=": first != second,
                "<": first < second,
                ">": first > second,
                "<=": first <= second,
                ">=": first >= second,
            }[operator]
            return _Number(int(result), False)
        if operator in ("/", "%"):
            if second == 0:
                if not evaluate:
                    return _Number(0, unsigned)
                self.error("division by zero in a #if expression", token)
            if unsigned:
                value = first // second if operator == "/" else first % second
            else:
                # C truncates toward zero; Python floors.
                quotient = abs(first) // abs(second)
                if (first < 0) != (second < 0):
                    quotient = -quotient
                value = quotient if operator == "/" else first - quotient * second
            return _Number(self.wrap(value, unsigned), unsigned)
        value = {
            "+": first + second,
            "-": first - second,
            "*": first * second,
            "&": first & second,
            "|": first | second,
            "^": first ^ second,
        }[operator]
        return _Number(self.wrap(value, unsigned), unsigned)

    @staticmethod
    def wrap(value: int, unsigned: bool) -> int:
        return (value & _MASK) if unsigned else _signed(value)

    def unary(self, evaluate: bool) -> _Number:
        self.deeper()
        try:
            return self._unary(evaluate)
        finally:
            self.depth -= 1

    def _unary(self, evaluate: bool) -> _Number:
        token = self.token
        if token is None:
            self.error("a #if expression ends too early")
        if token.kind == "punctuator" and token.spelling in ("+", "-", "~", "!"):
            self.index += 1
            operand = self.unary(evaluate)
            if token.spelling == "+":
                return operand
            if token.spelling == "-":
                return _Number(self.wrap(-operand.value, operand.unsigned), operand.unsigned)
            if token.spelling == "~":
                return _Number(self.wrap(~operand.value, operand.unsigned), operand.unsigned)
            return _Number(int(operand.value == 0), False)
        if token.kind == "punctuator" and token.spelling == "(":
            self.index += 1
            value = self.conditional(evaluate)
            if self.spelling() != ")":
                self.error("this '(' has no ')'")
            self.index += 1
            return value
        self.index += 1
        if token.kind == "identifier":
            if token.spelling == "defined":
                self.error(
                    "'defined' here came out of a macro expansion, which C leaves "
                    "undefined; write it directly in the #if",
                    token,
                )
            if token.spelling in _ASKS_WHAT_THIS_COMPILER_HAS:
                return self.what_this_compiler_has(token)
            # C11 6.10.1p4: every identifier that is left is replaced by 0.
            return _Number(0, False)
        if token.kind in ("number", "character"):
            return self.constant(token)
        self.error(f"unexpected {token.spelling!r} in a #if expression", token)

    def what_this_compiler_has(self, token: PPToken) -> _Number:
        """`__has_feature(x)` and its family, which take an argument.

        Left as an ordinary identifier each became 0, and the `(` after it
        was then a stray - which is what stopped a standard C++ header at its
        first line of feature detection. They are operators, not macros: a
        compiler answers them about itself, and every compiler that does not
        have one still has to read past the argument.

        py2bin answers no to all of them but `__has_include`, which it can
        answer truthfully by looking. Saying no is what makes a library take
        the portable path it keeps for compilers without the extension, which
        is the path py2bin wants.
        """

        if self.spelling() != "(":
            self.error(f"{token.spelling!r} takes a parenthesised argument", token)
        self.index += 1
        depth = 1
        inside: "list[PPToken]" = []
        while depth:
            here = self.token
            if here is None:
                self.error(f"this {token.spelling!r} has no ')'", token)
            if here.kind == "punctuator" and here.spelling == "(":
                depth += 1
            elif here.kind == "punctuator" and here.spelling == ")":
                depth -= 1
                if not depth:
                    self.index += 1
                    break
            inside.append(here)
            self.index += 1
        if token.spelling != "__has_include":
            return _Number(0, False)
        return _Number(int(self.can_find_header(inside)), False)

    def can_find_header(self, inside: "list[PPToken]") -> bool:
        """Whether `__has_include`'s argument names a header that is there."""

        spelled = "".join(item.spelling for item in inside).strip()
        if spelled.startswith("<") and spelled.endswith(">"):
            name, angled = spelled[1:-1], True
        elif len(spelled) >= 2 and spelled[0] == spelled[-1] == '"':
            name, angled = spelled[1:-1], False
        else:
            return False
        if name in _BUILTIN_HEADERS:
            return True
        return any((folder / name).is_file() for folder in self.look_in)

    def constant(self, token: PPToken) -> _Number:
        converted = _convert(token, lambda message, item: self.error(message, item))
        if converted.kind != "integer":
            self.error(
                "a #if expression is an integer constant expression; "
                f"{token.spelling!r} is not an integer",
                token,
            )
        value = converted.value
        assert isinstance(value, int)
        unsigned = "u" in converted.suffix
        if not unsigned and converted.radix != 10 and value > 0x7FFFFFFFFFFFFFFF:
            # A hexadecimal or octal constant takes an unsigned type when it no
            # longer fits a signed one.
            unsigned = True
        if value > _MASK:
            self.error(
                f"the constant {token.spelling!r} does not fit in 64 bits, which is "
                "what #if computes in",
                token,
            )
        return _Number(value & _MASK if unsigned else value, unsigned)


def preprocess(
    source: str,
    filename: str,
    *,
    target: str | None = None,
    include_dirs: tuple[str, ...] = (),
    defines: tuple[str, ...] = (),
    cplusplus: bool = False,
) -> list[Token]:
    """Preprocess C source text and return the tokens the C parser reads."""

    engine = Preprocessor(include_dirs, target)
    engine.translated_cplusplus = cplusplus
    if cplusplus:
        # This text was C++ and its conditionals were written for a C++
        # compiler, where `__cplusplus` is defined. Left undefined here, a
        # program's own `#ifdef __cplusplus` took the arm meant for C: the
        # class in the first arm had already been lifted out and written as
        # a struct, and then the `typedef int Tag;` in the `#else` arm was
        # read as well - "'Tag' is already a different type" - or a function
        # defined under the guard simply went missing at its call. Inside
        # py2bin's own C headers it stays undefined; see where one is read.
        engine.run("#define __cplusplus 201703L\n", "<c++>", None)
    if defines:
        text = []
        for item in defines:
            name, separator, value = item.partition("=")
            text.append(f"#define {name} {value if separator else '1'}")
        engine.run("\n".join(text) + "\n", "<command line>", None)
    engine.run(source, filename, Path(filename).parent)
    return engine.tokens()
