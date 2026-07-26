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
from pathlib import Path

from .c_frontend import CCompileError, Lexer, Token

__all__ = ["preprocess"]


# --- preprocessing tokens ----------------------------------------------------


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
                raise CCompileError(
                    origin,
                    line,
                    column,
                    "wide and Unicode string/character literals are not supported",
                )
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
            value = self._condition(name, rest, name_token)
            conditions.append(_Condition(True, value, value, False, name_token))
            return
        if name == "elif":
            if not conditions:
                self.error("#elif without a matching #if", name_token)
            condition = conditions[-1]
            if condition.seen_else:
                self.error("#elif after #else", name_token)
            if not condition.outer or condition.taken:
                condition.active = False
                return
            value = self._condition("if", rest, name_token)
            condition.active = value
            condition.taken = condition.taken or value
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
            message = " ".join(item.spelling for item in rest)
            self.error(f"#error {message}" if message else "#error", name_token)
        if name == "pragma":
            if [item.spelling for item in rest] == ["once"]:
                self.once.add(name_token.origin)
                return
            self.error(
                "the only #pragma py2bin implements is 'once'; it refuses to "
                "ignore a pragma it does not know, because a pragma can change "
                "the layout or the ABI of everything after it",
                name_token,
            )
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
            self.error(
                f"macro {name!r} is redefined with a different replacement list; "
                "C requires every definition of a macro to be identical "
                f"(the first was at {existing.token.origin}:{existing.token.line}:"
                f"{existing.token.column})",
                name_token,
            )
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
        for candidate in candidates:
            if candidate.is_file():
                self._read(candidate, at)
                return
        builtin = _BUILTIN_HEADERS.get(name)
        if builtin is not None:
            self._enter(builtin, f"<{name}>", None, at)
            return
        searched = ", ".join(str(item) for item in candidates) or "nowhere"
        self.error(
            f"cannot find the header {name!r}; py2bin looked in {searched}. It has "
            "built-in copies of the standard headers "
            f"({', '.join(sorted(_BUILTIN_HEADERS))}) and no system include path, "
            "because a real system header uses compiler extensions py2bin's C "
            "compiler does not implement",
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

    def _condition(self, name: str, rest: list[PPToken], at: PPToken) -> bool:
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
        return _Evaluator(tokens, at).run() != 0

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
_BUILTIN_HEADERS = {
    "stdio.h": "#define EOF (-1)\n#define NULL ((void *)0)\n",
    "stdlib.h": "#define NULL ((void *)0)\n"
    "#define EXIT_SUCCESS 0\n#define EXIT_FAILURE 1\n",
    "string.h": "#define NULL ((void *)0)\n",
    "stddef.h": "#define NULL ((void *)0)\n",
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

    def __init__(self, tokens: list[PPToken], at: PPToken):
        self.tokens = tokens
        self.index = 0
        self.at = at
        self.depth = 0

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
            # C11 6.10.1p4: every identifier that is left is replaced by 0.
            return _Number(0, False)
        if token.kind in ("number", "character"):
            return self.constant(token)
        self.error(f"unexpected {token.spelling!r} in a #if expression", token)

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
