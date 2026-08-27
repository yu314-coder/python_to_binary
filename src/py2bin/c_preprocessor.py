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
* ``#pragma once``;
* the null directive ``#``;
* the predefined macros ``__FILE__``, ``__LINE__``, ``__STDC__``,
  ``__STDC_VERSION__``, ``__STDC_HOSTED__`` (0 -- py2bin has no hosted library),
  ``__py2bin__``, ``__py2bin_target__``, and one each of
  ``__py2bin_arm64__``/``__py2bin_x86_64__`` and
  ``__py2bin_darwin__``/``__py2bin_linux__``/``__py2bin_windows__``.
  ``__DATE__`` and ``__TIME__`` are the fixed ``"Jan  1 1970"`` and
  ``"00:00:00"``: C11 6.10.8.1 allows an implementation-defined constant when
  the date of translation is not available, and py2bin would rather compile the
  same source to the same bytes than read the clock.

What is rejected
----------------
``#line`` (py2bin reports the line the token was really written on),
``#warning`` (py2bin's C compiler has no channel for a diagnostic that is not
fatal), any ``#pragma`` other than ``once`` -- ignoring an unknown pragma is
how a compiler silently changes an ABI, so py2bin refuses instead -- the GNU
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
        self.output: list[PPToken] = []
        self.depth = 0
        self.once: set[str] = set()
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
            self.error(
                "#warning is not implemented; py2bin's C compiler has no channel "
                "for a diagnostic that does not stop the compilation",
                name_token,
            )
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

    #: Pragmas that say something to a *compiler* and nothing about the
    #: program: which warnings to show, where an editor may fold, which
    #: library a linker should look in. None of them can change the layout or
    #: the ABI of what follows, which is the reason the rest are refused - so
    #: these are read and dropped rather than being refused along with them.
    _INERT_PRAGMAS = frozenset(
        {
            "warning",       # MSVC: push, pop, disable, suppress
            "region",        # editor folding
            "endregion",
            "message",       # a note to whoever is compiling
            "comment",       # comment(lib, ...), which needs a linker to mean
            "push_macro",    # saves and restores a macro this preprocessor
            "pop_macro",     # keeps by name anyway
            "component",
            "function",
            "intrinsic",
            "auto_inline",
            "inline_depth",
            "inline_recursion",
            "optimize",
            "float_control",
            "fenv_access",
            "STDC",          # C99 says an implementation may ignore these
        }
    )

    #: Pragmas whose *second* word is what says they are inert. Every
    #: compiler that has these spells them `<who> diagnostic ...`, and which
    #: compiler it is addressed to does not matter: a pragma about
    #: diagnostics says nothing about the program whoever is being told.
    #: Read this way rather than by naming the compilers, which is a thing
    #: no module here is allowed to do.
    _INERT_SECOND = frozenset({"diagnostic", "system_header", "poison"})

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

    def _pragma(self, rest: "list[PPToken]", name_token: "PPToken") -> None:
        """`#pragma once`, the ones that mean nothing here, and the rest.

        A pragma is where an implementation is allowed to be told anything at
        all, so most of them are refused: one can change the layout or the ABI
        of everything after it, and a compiler that ignored `#pragma pack`
        would lay every struct out wrong and say nothing. The ones accepted
        here are the ones that provably cannot - they speak to a compiler
        about diagnostics, folding, or linking, and say nothing about the
        program itself.
        """

        spelled = [item.spelling for item in rest]
        if spelled == ["once"]:
            self.once.add(name_token.origin)
            return
        if not spelled:
            # `#pragma` with nothing after it. C says an implementation may
            # do what it likes, and there is nothing here to do.
            return
        if spelled[0] in self._INERT_PRAGMAS:
            return
        if len(spelled) > 1 and spelled[1] in self._INERT_SECOND:
            return
        if spelled[0] == "pack":
            # It changes how every struct after it is laid out, so it cannot
            # be dropped the way the ones above are. It is passed on to the
            # parser as tokens - a directive is not one, and this is the only
            # channel there is between the two.
            self.output.append(
                dataclasses.replace(
                    name_token, kind="identifier", spelling=_PACK_MARKER
                )
            )
            self.output.extend(rest[1:])
            self.output.append(
                dataclasses.replace(name_token, kind="punctuator", spelling=";")
            )
            return
        self.error(
            f"#pragma {spelled[0]} is not implemented; py2bin refuses a pragma "
            f"it does not know, because a pragma can change the layout or the "
            f"ABI of everything after it. The ones it accepts are 'once' and "
            f"those that speak only to a compiler - "
            f"{', '.join(sorted(self._INERT_PRAGMAS))}",
            name_token,
        )

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
            try:
                self._enter(builtin, f"<{name}>", None, at)
            finally:
                self._reading_a_builtin = was
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
            "path: a platform SDK header is written in compiler extensions this "
            "C compiler does not implement, so finding one would not help",
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

    def _read(self, path: Path, at: PPToken) -> None:
        resolved = str(path.resolve())
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
_MATH_H = (
    pathlib.Path(__file__).with_name("libm.c").read_text(encoding="utf-8")
)

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

#ifndef NULL
#define NULL ((void *)0)
#endif
#define WINAPI
#define APIENTRY
#define CALLBACK
#define CONST const
#define FALSE 0
#define TRUE 1
#define MAX_PATH 260

typedef int BOOL;
typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned int DWORD;
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
typedef struct tagMSG {
    HWND hwnd; UINT message; WPARAM wParam; LPARAM lParam;
    DWORD time; POINT pt;
} MSG;

/* The window procedure a program writes, and the loader calls back into. */
typedef LRESULT (*WNDPROC)(HWND, UINT, WPARAM, LPARAM);

typedef struct tagWNDCLASSEXW {
    UINT cbSize; UINT style; WNDPROC lpfnWndProc;
    int cbClsExtra; int cbWndExtra;
    HINSTANCE hInstance; HICON hIcon; HCURSOR hCursor; HBRUSH hbrBackground;
    LPCWSTR lpszMenuName; LPCWSTR lpszClassName; HICON hIconSm;
} WNDCLASSEXW;

extern ATOM RegisterClassExW(LPVOID);
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
extern LRESULT SetWindowLongPtrW(HWND, int, LPVOID);
extern LRESULT GetWindowLongPtrW(HWND, int);
extern HMODULE GetModuleHandleW(LPCWSTR);

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
#define IDC_ARROW ((LPCWSTR)32512)
#define IDI_APPLICATION ((LPCWSTR)32512)
#define COLOR_WINDOW 5
#define PM_REMOVE 0x0001
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
   that includes this header and never allocates reserves nothing. */
static unsigned long __py2bin_heap_bump = 0;
static unsigned long __py2bin_heap_end = 0;

void *malloc(unsigned long __n) {
    unsigned long __p;
    if (__py2bin_heap_end == 0) {
        __py2bin_heap_bump = (unsigned long)__py2bin_arena();
        __py2bin_heap_end = __py2bin_heap_bump + __PY2BIN_ARENA_BYTES;
    }
    /* Round up to 16, which is the alignment any C object may ask for, so
       every block this returns is aligned for every type. A request of zero
       still gets a distinct address, as C says it may. */
    __n = (__n + 15UL) & ~15UL;
    if (__n == 0UL) __n = 16UL;
    /* Written as a subtraction so a size near the top of the range cannot
       wrap the sum past the end and be let through. */
    if (__n > __py2bin_heap_end - __py2bin_heap_bump) return NULL;
    __p = __py2bin_heap_bump;
    __py2bin_heap_bump = __p + __n;
    return (void *)__p;
}

void *calloc(unsigned long __count, unsigned long __size) {
    unsigned long __total;
    unsigned char *__block;
    unsigned long __i;
    if (__count != 0UL && __size > 0xFFFFFFFFFFFFFFFFUL / __count) return NULL;
    __total = __count * __size;
    __block = (unsigned char *)malloc(__total);
    if (__block == NULL) return NULL;
    /* The arena is zero-filled when it is mapped, but a block reused after a
       realloc is not, so this clears rather than assuming. */
    for (__i = 0UL; __i < __total; __i++) __block[__i] = 0;
    return (void *)__block;
}

void free(void *__block) {
    /* An arena does not reclaim. Saying so plainly is better than a free()
       that appears to work and silently does nothing about fragmentation. */
    (void)__block;
}

void *realloc(void *__block, unsigned long __size) {
    unsigned char *__old;
    unsigned char *__new;
    unsigned long __i;
    if (__block == NULL) return malloc(__size);
    __new = (unsigned char *)malloc(__size);
    if (__new == NULL) return NULL;
    /* Nothing records how big the old block was, so this copies the smaller
       of the two -- the new size -- and reads no more of the old block than
       the arena holds. Growing is exact; shrinking copies only what stays. */
    __old = (unsigned char *)__block;
    for (__i = 0UL; __i < __size; __i++) {
        if ((unsigned long)(__old + __i) >= __py2bin_heap_bump) break;
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

#: The types a header reaches for when it wants the ones a platform is
#: expected to have. py2bin has no system to take them from, so it says what
#: they are for the target it is building for - which is the same thing every
#: other header here does. Each is guarded: a header that defines its own
#: first keeps it, the way `NULL` does.
_SYS_TYPES_H = """
#ifndef __py2bin_ssize_t_defined
#define __py2bin_ssize_t_defined
typedef long ssize_t;
#endif
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
#include <windows.h>

typedef long HRESULT;
typedef unsigned short VARIANT_BOOL;
typedef const wchar_t *LPCOLESTR;
typedef wchar_t *LPOLESTR;
typedef wchar_t OLECHAR;
typedef wchar_t *BSTR;
typedef double DATE;

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

/* A generated header names its interface IDs with this. The SDK writes it
   two ways - a declaration everywhere, and a definition in the one file that
   defines INITGUID - because a program is several translation units and the
   value has to live in exactly one of them. py2bin has one, so the value
   always lives here: there is no other file for it to clash with, and a
   declaration with no definition anywhere would be a name nothing could
   resolve, py2bin having no linker either. */
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

/* What a generated header puts before a vtable pointer. `const` in the SDK,
   because the table is never written; the spelling is all that differs. */
#define CONST_VTBL const

#define __RPC_STUB
#define __RPC_FAR
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
/* What Automation passes a value in. Sixteen bytes on both Windows
   machines: a two-byte tag, six the SDK reserves, and eight of value -
   which is what every member of that union is, or fits in. Written as the
   layout rather than as the union, because the union's members are a
   hundred names for the same eight bytes and the layout is the part that
   has to be right. */
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
typedef struct IUnknown IUnknown;
typedef struct IUnknownVtbl {
    HRESULT (*QueryInterface)(IUnknown *, REFIID, void **);
    unsigned long (*AddRef)(IUnknown *);
    unsigned long (*Release)(IUnknown *);
} IUnknownVtbl;
struct IUnknown { const IUnknownVtbl *lpVtbl; };
typedef IUnknown *LPUNKNOWN;
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
typedef struct IStream IStream;
typedef struct ISequentialStream ISequentialStream;

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

typedef struct ISequentialStreamVtbl {
    HRESULT (*QueryInterface)(ISequentialStream *, REFIID, void **);
    unsigned long (*AddRef)(ISequentialStream *);
    unsigned long (*Release)(ISequentialStream *);
    HRESULT (*Read)(ISequentialStream *, void *, unsigned long, unsigned long *);
    HRESULT (*Write)(ISequentialStream *, const void *, unsigned long, unsigned long *);
} ISequentialStreamVtbl;
struct ISequentialStream { const ISequentialStreamVtbl *lpVtbl; };

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

typedef unsigned short CLIPFORMAT;
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
    # The one file in the mingw-w64 set that does not exist as a file: it is
    # generated by a configure step, and what it holds is a description of
    # the compiler reading it - which py2bin is the one that knows.
    "_mingw.h": _MINGW_H,
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
    "stddef.h": _NULL,
    # A `va_list` is a pointer to the cells the call wrote its extra arguments
    # into, and the four names that walk one are compiled rather than called -
    # so this header is the typedef and nothing else.
    "stdarg.h": "typedef char *va_list;\n",
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
) -> list[Token]:
    """Preprocess C source text and return the tokens the C parser reads."""

    engine = Preprocessor(include_dirs, target)
    if defines:
        text = []
        for item in defines:
            name, separator, value = item.partition("=")
            text.append(f"#define {name} {value if separator else '1'}")
        engine.run("\n".join(text) + "\n", "<command line>", None)
    engine.run(source, filename, Path(filename).parent)
    return engine.tokens()
