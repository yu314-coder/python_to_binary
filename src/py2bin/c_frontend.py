"""py2bin's C compiler: C source text straight to py2bin's native IR.

This module is a real (small) C compiler written in pure Python. It lexes and
parses C, builds a typed syntax tree of its own, applies C's integer promotions
and conversions, and emits py2bin's native IR -- which the handwritten ARM64 and
x86-64 encoders turn into machine code. The directives are run first by
:mod:`py2bin.c_preprocessor`, which is py2bin's own, so no external compiler,
assembler, linker, preprocessor or toolchain is involved, and no process is
ever started.

It deliberately does NOT reuse Python's ``ast`` module. An earlier bridge did,
and C's semantics kept leaking away: a C ``for`` acquired Python ``range``
behaviour, and nothing in a Python tree can express a narrow integer type, the
address of a local, or ``goto``. C gets its own front end here, and the pieces
of C that cannot yet be compiled correctly are rejected with a file:line:column
error instead of being approximated.

What is implemented
-------------------
* the integer type zoo -- ``char``/``short``/``int``/``long``/``long long`` and
  their unsigned forms, plus the ``<stdint.h>`` fixed-width names -- with C's
  integer promotions, usual arithmetic conversions, and exact truncation and
  sign/zero extension on every assignment, cast and narrow-typed operation;
* ``float`` and ``double``: decimal and hexadecimal floating constants,
  arithmetic, IEEE comparisons (including the unordered case NaN produces),
  the usual arithmetic conversions between the integer and floating types, and
  conversions in both directions -- with ``float`` objects really four bytes
  wide. Every floating expression is EVALUATED in double precision and rounded
  to ``float`` only where C requires the extra precision removed, which is
  ``FLT_EVAL_METHOD == 1``. A floating argument or result crosses a call as its
  IEEE bit pattern in an integer register; that ABI is py2bin's own and never
  meets a platform C function. ``long double`` is rejected rather than quietly
  aliased to ``double``;
* real memory: local arrays (including multi-dimensional), ``&x``, ``*p``,
  pointer arithmetic, and ``a[i]``, all with loads and stores at the right
  width;
* casts between integer types, between pointer types, and between the two, and
  ``sizeof`` for every complete type;
* the full expression grammar: ``++``/``--`` (prefix and postfix), the comma
  operator, ``?:``, short-circuit ``&&``/``||``, compound assignment, and
  signed/unsigned division and remainder;
* statements: ``if``/``else``, ``while``, ``do``/``while``, ``for``,
  ``switch``/``case``/``default`` with fallthrough, ``break``, ``continue``,
  ``goto`` with labels, and ``return``;
* functions, called through a real machine call ABI on the targets whose
  encoder implements one (see ``CALL_CAPABLE_TARGETS``): each call gets its own
  stack frame with a saved link register, so **recursion works** -- direct,
  deep, and mutual. On the remaining targets the call ABI is not implemented,
  so a call is still inlined at its site and recursion is rejected there rather
  than miscompiled;
* function pointers, on the same targets: C's real declarator grammar, so
  ``int (*p)(int)``, ``int (*ops[3])(void)`` and ``int *(*f)(char *)`` all read
  the way C says they do; a function designator decaying to a pointer; ``&f``;
  ``(*p)(x)``; casts naming a function-pointer type; and a genuine indirect
  machine call. The called expression is evaluated exactly once and BEFORE the
  arguments. A function type must state its parameters -- C's empty ``()``
  leaves them unspecified, and py2bin will not emit a call it cannot check;
* file-scope objects -- objects with static storage duration -- on the targets
  whose encoder establishes the static block (see ``STATIC_CAPABLE_TARGETS``).
  They live in one contiguous zero-filled block that outlives every frame, so
  the same object really is the same object in ``main`` and in everything
  ``main`` calls. ``static`` at file scope is accepted (it limits a linkage a
  single translation unit cannot escape). An initializer must be a constant
  expression, which is what C requires of static storage: arithmetic constants
  and address constants such as ``&x``, an array name, or a string literal;
* ``printf`` with real runtime formatting, and the vetted ``extern`` adapter
  ABI that lets compiled C drive an embedded CPython;
* the directives, which :mod:`py2bin.c_preprocessor` has already run by the
  time anything here sees a token: macros with ``#`` and ``##``, ``#include``,
  and the conditional family. That module documents exactly what it accepts.

What is rejected
----------------
``long double``, a variadic function *type* (a pointer to one), a
function type with an unspecified ``()`` parameter list, a function whose own
declarator is not the plain ``TYPE *... name(params)`` (write a typedef for the
result type), ``extern`` objects (py2bin compiles one translation unit and has
no linker), more than eight arguments to a function (py2bin passes
arguments only in registers), and recursion, function pointers or file-scope
objects on the targets that have no call ABI or static block yet.
"""

from __future__ import annotations

import dataclasses
import struct

from .native.frontend import _CABI_RESULT_WIDTH
# The C front end may call more than a Python program may: a Windows API
# function is an import the loader binds, and has no CPython shim because it
# could not have one anywhere else. See `C_EXTERN_SYMBOLS`.
from .native.frontend import C_EXTERN_RESULTS as _CABI_RESULTS
from .native.frontend import C_EXTERN_SYMBOLS as _CABI_SYMBOLS
from .native.compiler import CALL_CAPABLE_TARGETS
from .native.ir import (
    MAXIMUM_STACK_SLOTS,
    BitsFloat,
    Call as IRCall,
    CStringConstant,
    Exit,
    ExitValue,
    ExternCall,
    FloatBinary,
    FloatBits,
    FloatCompare,
    FloatConstant,
    FloatExpression,
    FloatLoad,
    FloatStore,
    FloatToInt,
    FloatUnary,
    Function as IRFunction,
    FunctionAddress,
    GlobalAddress,
    FileCall,
    HeapInit,
    HeapLoad,
    HeapStore,
    IndirectCall,
    IntBinary,
    IntCompare,
    IntConstant,
    IntExpression,
    IntLoad,
    IntToFloat,
    IntUnary,
    Jump,
    JumpIfFalse,
    Label,
    Module,
    Operation,
    Return as IRReturn,
    SlotAddress,
    Store,
    Write,
    WriteRuntime,
)


#: Targets whose encoder can hand back the address of a string literal. Each
#: places the bytes after the code and reaches them with a PC-relative
#: reference; the back ends not named here have nowhere to put them yet.
#: The targets whose writers place a string literal's bytes where a pointer
#: to them can be formed. On ARM64 the encoder appends them to the code image
#: itself and reaches them PC-relatively, so the writer needs to do nothing -
#: which is why the two ARM64 targets can be here without further work.
_STRING_VALUE_TARGETS = frozenset(
    {
        "darwin-arm64",
        "darwin-x86_64",
        "windows-x86_64",
        "windows-arm64",
        "linux-arm64",
        "linux-x86_64",
    }
)


class CCompileError(ValueError):
    """A source-located rejection from py2bin's C compiler."""

    def __init__(self, filename: str, line: int, column: int, message: str):
        self.filename = filename
        self.line = line
        self.column = column
        self.message = message
        super().__init__(f"{filename}:{line}:{column}: {message}")


# --- tokens ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Token:
    kind: str  # "identifier" | "integer" | "string" | "symbol" | "eof"
    value: object
    line: int
    column: int
    # Integer literals carry the type their suffix, base and value demand: C
    # gives a hexadecimal or octal constant an unsigned type when it no longer
    # fits a signed one, but a plain decimal constant never gets one.
    suffix: str = ""
    radix: int = 10
    # The file the token was really written in, which is not the file being
    # compiled once #include has brought another one in.
    origin: str = ""
    # For a string or character literal, its prefix: "", "u8", "L", "u", "U".
    # The plain and u8 kinds carry bytes; the rest carry code points, because
    # what a code unit is depends on the target.
    encoding: str = ""


_OPERATORS = (
    "<<=",
    ">>=",
    "...",
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
)
_PUNCTUATION = set("{}[]();,?:=+-*/%~!<>&|^.")

_SIMPLE_ESCAPES = {
    "\\": 0x5C,
    "'": 0x27,
    '"': 0x22,
    "?": 0x3F,
    "a": 0x07,
    "b": 0x08,
    "f": 0x0C,
    "n": 0x0A,
    "r": 0x0D,
    "t": 0x09,
    "v": 0x0B,
}


class Lexer:
    """C's tokens, from text that the preprocessor has already been over.

    ``line`` and ``column`` may be set before lexing starts, which is how the
    preprocessor converts one preprocessing token at a time and still reports
    the position it was written at.
    """

    def __init__(self, source: str, filename: str):
        self.source = source
        self.filename = filename
        self.index = 0
        self.line = 1
        self.column = 1

    def error(self, message: str, line: int | None = None, column: int | None = None):
        raise CCompileError(
            self.filename,
            self.line if line is None else line,
            self.column if column is None else column,
            message,
        )

    def advance(self) -> str:
        character = self.source[self.index]
        self.index += 1
        if character == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return character

    def startswith(self, text: str) -> bool:
        return self.source.startswith(text, self.index)

    def skip_layout(self) -> None:
        while self.index < len(self.source):
            character = self.source[self.index]
            if character.isspace():
                self.advance()
                continue
            if self.startswith("//"):
                while self.index < len(self.source) and self.source[self.index] != "\n":
                    self.advance()
                continue
            if self.startswith("/*"):
                line, column = self.line, self.column
                self.advance()
                self.advance()
                while self.index < len(self.source) and not self.startswith("*/"):
                    self.advance()
                if self.index >= len(self.source):
                    self.error("unterminated block comment", line, column)
                self.advance()
                self.advance()
                continue
            if character == "#":
                # The preprocessor consumes every directive before this lexer
                # runs, so a '#' that reaches here is one it left behind.
                self.error(
                    "'#' is a preprocessing operator and means nothing in C; it "
                    "is only valid at the start of a directive or inside a #define"
                )
            return

    def escape(self, quote: str) -> "tuple[int, bool]":
        """Consume one character or escape and say what it is.

        Returns the value and whether it is a *byte*. The distinction is the
        whole of what C says here: `\\xFF` names the byte 0xFF whatever the
        literal's kind, while `é` written in the source names a character,
        and what that becomes depends on the kind - three bytes in a plain
        literal, one code unit in a wide one. Conflating them made a source
        character above 127 unrepresentable and refused the literal.
        """

        character = self.advance()
        if character != "\\":
            return ord(character), False
        if self.index >= len(self.source):
            self.error("unterminated escape sequence")
        escaped = self.advance()
        if escaped in _SIMPLE_ESCAPES:
            return _SIMPLE_ESCAPES[escaped], True
        if escaped in "uU":
            # A universal character name: `\\u00e9` is the character, not a
            # byte, so it encodes the same way one written in the source does.
            width = 4 if escaped == "u" else 8
            digits = ""
            while len(digits) < width and self.index < len(self.source) and (
                self.source[self.index] in "0123456789abcdefABCDEF"
            ):
                digits += self.advance()
            if len(digits) != width:
                self.error(f"\\{escaped} needs exactly {width} hexadecimal digits")
            value = int(digits, 16)
            if value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
                self.error(f"\\{escaped}{digits} is not a character")
            return value, False
        if escaped == "x":
            digits = ""
            while self.index < len(self.source) and self.source[self.index] in (
                "0123456789abcdefABCDEF"
            ):
                digits += self.advance()
            if not digits:
                self.error("\\x needs at least one hexadecimal digit")
            value = int(digits, 16)
            if value > 0xFF:
                self.error("\\x escape does not fit in one byte")
            return value, True
        if escaped in "01234567":
            digits = escaped
            while len(digits) < 3 and self.index < len(self.source) and (
                self.source[self.index] in "01234567"
            ):
                digits += self.advance()
            value = int(digits, 8)
            if value > 0xFF:
                self.error("octal escape does not fit in one byte")
            return value, True
        self.error(f"unsupported escape sequence \\{escaped} in a {quote} literal")

    def digits(self, allowed: str) -> int:
        """Consume a run of ``allowed`` characters and report how many."""

        seen = 0
        while self.index < len(self.source) and self.source[self.index] in allowed:
            self.advance()
            seen += 1
        return seen

    def floating(self, start: int, line: int, column: int, *, hexadecimal: bool) -> Token:
        """Finish scanning a floating constant that began at ``start``.

        C's grammar is followed exactly: a decimal constant needs either a
        period or an exponent, a hexadecimal one always needs its ``p``
        exponent, and the suffix selects the type. The value itself is produced
        by Python's own correctly-rounded decimal-to-binary64 conversion, so a
        literal is the nearest double to what was written rather than the
        result of a hand-rolled parse.
        """

        allowed = "0123456789abcdefABCDEF" if hexadecimal else "0123456789"
        if self.index < len(self.source) and self.source[self.index] == ".":
            self.advance()
            self.digits(allowed)
        markers = "pP" if hexadecimal else "eE"
        if self.index < len(self.source) and self.source[self.index] in markers:
            self.advance()
            if self.index < len(self.source) and self.source[self.index] in "+-":
                self.advance()
            if not self.digits("0123456789"):
                self.error("this floating constant's exponent has no digits", line, column)
        elif hexadecimal:
            self.error(
                "a hexadecimal floating constant needs a binary exponent, as in "
                "0x1.8p3",
                line,
                column,
            )
        text = self.source[start : self.index]
        suffix = ""
        if self.index < len(self.source) and self.source[self.index] in "fFlL":
            suffix = self.advance().lower()
        if self.index < len(self.source) and (
            self.source[self.index].isalnum() or self.source[self.index] == "_"
        ):
            self.error("unsupported floating literal suffix", line, column)
        if suffix == "l":
            self.error(
                "'long double' is not implemented by py2bin's C compiler; it has "
                "'float' and 'double', and would have to pretend a wider type was "
                "wider than double to accept this",
                line,
                column,
            )
        try:
            value = float.fromhex(text) if hexadecimal else float(text)
        except (ValueError, OverflowError):
            self.error(f"{text!r} is not a valid floating constant", line, column)
        if suffix == "f":
            try:
                value = struct.unpack("<f", struct.pack("<f", value))[0]
            except OverflowError:
                value = float("inf")
        if value in (float("inf"), float("-inf")):
            self.error(
                f"the floating constant {text!r} overflows the type it is written "
                "with; C leaves that undefined, so py2bin refuses it",
                line,
                column,
            )
        return Token("float", value, line, column, suffix)

    def number(self) -> Token:
        line, column = self.line, self.column
        start = self.index
        radix = 10
        if self.startswith("0x") or self.startswith("0X"):
            radix = 16
            self.advance()
            self.advance()
            digits = self.index
            while self.index < len(self.source) and self.source[self.index] in (
                "0123456789abcdefABCDEF"
            ):
                self.advance()
            if self.index < len(self.source) and self.source[self.index] in ".pP":
                return self.floating(start, line, column, hexadecimal=True)
            if self.index == digits:
                self.error("hexadecimal integer needs at least one digit", line, column)
            value = int(self.source[start : self.index], 16)
        else:
            while self.index < len(self.source) and self.source[self.index].isdigit():
                self.advance()
            text = self.source[start : self.index]
            if self.index < len(self.source) and self.source[self.index] in ".eE":
                if self.source[self.index] == "." or (
                    self.index + 1 < len(self.source)
                    and (
                        self.source[self.index + 1].isdigit()
                        or self.source[self.index + 1] in "+-"
                    )
                ):
                    return self.floating(start, line, column, hexadecimal=False)
            if text.startswith("0") and len(text) > 1:
                radix = 8
                if any(digit not in "01234567" for digit in text):
                    self.error(
                        f"{text!r} starts with 0, so C reads it as octal, but it "
                        "has a digit that is not octal",
                        line,
                        column,
                    )
                value = int(text, 8)
            else:
                value = int(text)
        suffix = ""
        while self.index < len(self.source) and self.source[self.index] in "uUlL":
            suffix += self.advance().lower()
        if self.index < len(self.source) and (
            self.source[self.index].isalnum() or self.source[self.index] == "_"
        ):
            self.error("unsupported integer literal suffix", line, column)
        if suffix.replace("u", "", 1).count("u") or suffix.count("l") > 2:
            self.error("unsupported integer literal suffix", line, column)
        return Token("integer", value, line, column, suffix, radix)

    def character(self, kind: str = "") -> Token:
        # The prefix has already been read - it lexed as an identifier, which
        # is what a prefix looks like until the quote after it says otherwise.
        line, column = self.line, self.column
        self.advance()  # opening quote
        if self.index < len(self.source) and self.source[self.index] == "'":
            self.error("empty character constant", line, column)
        value, is_byte = self.escape("character")
        if self.index >= len(self.source) or self.advance() != "'":
            self.error("multi-character constants are not supported", line, column)
        if kind:
            # A wide character constant is the code point, not a byte, and it
            # is not sign-adjusted: `L'\u00e9'` is 233 and not -23.
            if kind == "u" and value > 0xFFFF:
                self.error(
                    "u'...' holds one UTF-16 code unit, and this character "
                    "does not fit in one; write U'...'",
                    line,
                    column,
                )
            return Token("integer", value, line, column, "", 10, "", kind)
        if not is_byte and value > 0x7F:
            self.error(
                "a character constant holds one byte; write it as a wide one "
                "(L'x', u'x' or U'x') or put it in a string, where it is UTF-8",
                line,
                column,
            )
        # A character constant has type int in C, and a plain 'char' is signed
        # in this dialect, so \xFF is -1 exactly as it is on Apple's ABI.
        if value >= 0x80:
            value -= 0x100
        return Token("integer", value, line, column, "")

    def string(self, kind: str = "") -> Token:
        """One string literal, of whichever kind its prefix asked for.

        A plain or `u8` literal is bytes; the others are code units, which
        cannot be encoded until the target is known - `wchar_t` is two bytes
        on Windows and four elsewhere - so they are carried as code points
        and encoded where that is known.
        """

        # As in `character`, the prefix has already been read.
        line, column = self.line, self.column
        self.advance()  # opening quote
        points: list[int] = []
        data = bytearray()
        while True:
            if self.index >= len(self.source):
                self.error("unterminated string literal", line, column)
            if self.source[self.index] == '"':
                self.advance()
                break
            if self.source[self.index] == "\n":
                self.error("newline in string literal", line, column)
            value, is_byte = self.escape("string")
            if kind in ("", "u8"):
                # A byte goes in as it is; a character goes in as the UTF-8 it
                # is written as, which is what the source file already held.
                data.extend(
                    bytes([value]) if is_byte else chr(value).encode("utf-8")
                )
                continue
            points.append(value)
        if kind in ("", "u8"):
            return Token("string", bytes(data), line, column, "", 10, "", kind)
        return Token("string", tuple(points), line, column, "", 10, "", kind)

    def tokens(self) -> list[Token]:
        result: list[Token] = []
        while True:
            self.skip_layout()
            if self.index >= len(self.source):
                result.append(Token("eof", "", self.line, self.column))
                return result
            character = self.source[self.index]
            line, column = self.line, self.column
            if character.isalpha() or character == "_":
                start = self.index
                while self.index < len(self.source) and (
                    self.source[self.index].isalnum() or self.source[self.index] == "_"
                ):
                    self.advance()
                name = self.source[start : self.index]
                if name in {"L", "u8", "u", "U"} and self.index < len(self.source) and (
                    self.source[self.index] in "'\""
                ):
                    if self.source[self.index] == '"':
                        result.append(self.string(name))
                    else:
                        if name == "u8":
                            self.error(
                                "u8 character constants are C23; write the "
                                "character in a u8 string, which is UTF-8",
                                line,
                                column,
                            )
                        result.append(self.character(name))
                    continue
                result.append(Token("identifier", name, line, column))
                continue
            if character.isdigit():
                result.append(self.number())
                continue
            if (
                character == "."
                and self.index + 1 < len(self.source)
                and self.source[self.index + 1].isdigit()
            ):
                # C lets a floating constant start with its period, as in '.5'.
                result.append(self.floating(self.index, line, column, hexadecimal=False))
                continue
            if character == "'":
                result.append(self.character())
                continue
            if character == '"':
                result.append(self.string())
                continue
            operator = next(
                (candidate for candidate in _OPERATORS if self.startswith(candidate)),
                None,
            )
            if operator is not None:
                for _ in operator:
                    self.advance()
                result.append(Token("symbol", operator, line, column))
                continue
            if character in _PUNCTUATION:
                self.advance()
                result.append(Token("symbol", character, line, column))
                continue
            self.error(f"unsupported character {character!r}")


# --- the C type system -------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class VoidType:
    def __str__(self) -> str:
        return "void"


@dataclasses.dataclass(frozen=True, slots=True)
class IntegerType:
    name: str
    size: int
    signed: bool
    rank: int

    def __str__(self) -> str:
        return self.name


@dataclasses.dataclass(frozen=True, slots=True)
class FloatingType:
    """``float`` or ``double``, both held in IEEE-754 binary64 registers.

    py2bin's C evaluates every floating expression in double precision and
    rounds to ``float`` only where C says the extra precision must be removed
    -- assignment, cast, argument passing and return. That is exactly
    ``FLT_EVAL_METHOD == 1``, which C11 6.3.1.8p2 explicitly permits, and it is
    what lets one register file and one set of instructions serve both types.
    ``size`` is still the real storage size, so a ``float`` object occupies four
    bytes in memory and a ``float`` array indexes by four.
    """

    name: str
    size: int
    rank: int

    def __str__(self) -> str:
        return self.name


@dataclasses.dataclass(frozen=True, slots=True)
class PointerType:
    target: "CType"

    def __str__(self) -> str:
        if isinstance(self.target, FunctionType):
            # C writes a pointer to a function with the star inside the
            # parentheses, and a diagnostic that says 'int (*)(int)' is one a
            # reader can paste straight back into the program.
            inside = ", ".join(str(item) for item in self.target.parameters) or "void"
            return f"{self.target.result} (*)({inside})"
        return f"{self.target} *"


@dataclasses.dataclass(frozen=True, slots=True)
class ArrayType:
    element: "CType"
    count: int | None

    def __str__(self) -> str:
        return f"{self.element}[{'' if self.count is None else self.count}]"


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionType:
    """A function type: what ``f`` names, and what a function pointer points at.

    C never lets a value have this type -- a function designator immediately
    becomes a pointer to the function everywhere but ``sizeof`` and ``&`` -- so
    it has no size and no alignment here either. ``parameters`` is the full
    prototype: py2bin does not accept the unprototyped ``()`` form in a function
    type, because the calls it would then have to emit could not be checked.
    """

    result: "CType"
    parameters: tuple["CType", ...]

    def __str__(self) -> str:
        inside = ", ".join(str(item) for item in self.parameters) or "void"
        return f"{self.result} ({inside})"


#: What an unnamed bitfield is called, so the passes that walk members can
#: tell padding from something a program can write to.
#: What an anonymous struct or union member is called while it is laid out.
#: It has no name in the program, so this cannot collide with one; `member`
#: looks through anything called this.
_ANONYMOUS_MEMBER = "\x00anonymous"

#: What a parameter with no name is called while the frame is laid out. Not
#: a name a program can write, so nothing can reach it by accident.
_UNNAMED_PARAMETER = "\x00parameter"

_UNNAMED_BITFIELD = "__py2bin_pad_"

#: What `#pragma pack` reaches the parser as. The preprocessor emits it: a
#: directive is not a token, and this is the only channel between the two.
_PACK_MARKER = "__py2bin_pragma_pack"


@dataclasses.dataclass(frozen=True, slots=True)
class Member:
    name: str
    ctype: "CType"
    offset: int
    #: For a bitfield: how wide it is, and where in the storage unit at
    #: `offset` its low bit sits. `width` is None for an ordinary member,
    #: which is the whole of what tells the two apart.
    width: "int | None" = None
    bit: int = 0


@dataclasses.dataclass(slots=True)
class StructType:
    """A struct or union, laid out by C's alignment and padding rules.

    Each member starts at the next offset satisfying its own alignment, and the
    whole object is padded to a multiple of the strictest member alignment so
    that arrays of it stay aligned. A union puts every member at offset 0 and
    takes the size of its largest. ``members`` is None until the body is seen,
    which is what makes a forward-declared ``struct T;`` an incomplete type
    that only a pointer may refer to.
    """

    name: str | None
    is_union: bool = False
    members: tuple[Member, ...] | None = None
    size: int = 0
    alignment: int = 1

    def member(self, name: str) -> Member | None:
        for item in self.members or ():
            if item.name == name:
                return item
        # An anonymous member's members are this one's members. C11 says so
        # and the Windows SDK is written that way throughout: `STGMEDIUM`
        # holds an unnamed union of handles, and `m.hGlobal` reaches into it
        # without naming it. The offset is the anonymous member's plus the
        # inner one's, which is all that being inside it means.
        for item in self.members or ():
            if not item.name.startswith(_ANONYMOUS_MEMBER):
                continue
            if not isinstance(item.ctype, StructType):
                continue
            inner = item.ctype.member(name)
            if inner is not None:
                return dataclasses.replace(
                    inner, offset=item.offset + inner.offset
                )
        return None

    def __str__(self) -> str:
        keyword = "union" if self.is_union else "struct"
        return f"{keyword} {self.name}" if self.name else keyword


@dataclasses.dataclass(frozen=True, slots=True)
class _Hole:
    """A placeholder for the type an enclosing declarator has not built yet.

    ``int (*p)(void)`` is parsed inside-out: ``*p`` is read first, against a
    hole, and the ``(void)`` that follows the closing parenthesis is then
    substituted into it by :func:`_fill`. ``key`` keeps nested holes distinct.
    """

    key: int

    def __str__(self) -> str:  # pragma: no cover - only reachable on a bug
        return "?"


def _fill(ctype: "CType", hole: _Hole, actual: "CType") -> "CType":
    """Replace ``hole`` inside ``ctype`` with ``actual``."""

    if ctype == hole:
        return actual
    if isinstance(ctype, PointerType):
        return PointerType(_fill(ctype.target, hole, actual))
    if isinstance(ctype, ArrayType):
        return ArrayType(_fill(ctype.element, hole, actual), ctype.count)
    if isinstance(ctype, FunctionType):
        return FunctionType(_fill(ctype.result, hole, actual), ctype.parameters)
    return ctype


def align_of(ctype: "CType") -> int:
    """The alignment ``ctype`` requires, in bytes."""

    if isinstance(ctype, (IntegerType, FloatingType)):
        return ctype.size
    if isinstance(ctype, PointerType):
        return 8
    if isinstance(ctype, ArrayType):
        return align_of(ctype.element)
    if isinstance(ctype, StructType):
        return ctype.alignment
    return 1


def lay_out(
    struct: StructType,
    members: "list[tuple[str, CType] | tuple[str, CType, int]]",
    pack: "int | None" = None,
) -> None:
    """Assign member offsets and set the struct's size and alignment.

    A member given a width is a bitfield: it is packed into a storage unit of
    its own declared type, and the next one continues in the same unit while
    it fits. That is what every ABI py2bin targets does, and it is the only
    part of a bitfield's layout C says anything about beyond "it fits".

    `pack` is what `#pragma pack` last said: a cap on how far any member may
    be padded forward, and on the whole struct's own alignment. It is a cap
    and not a setting, which is what the directive means - a member whose
    type is already narrower than it keeps its own alignment.
    """

    placed: list[Member] = []
    offset = 0
    alignment = 1
    #: Where the bitfield being filled starts, and how much of it is used.
    unit_at = -1
    unit_bits = 0
    unit_size = 0
    for entry in members:
        name, ctype = entry[0], entry[1]
        width = entry[2] if len(entry) > 2 else None
        member_alignment = align_of(ctype)
        if pack is not None:
            member_alignment = min(member_alignment, pack)
        member_size = size_of(ctype) or 0
        alignment = max(alignment, member_alignment)
        if width is not None and not struct.is_union:
            if width == 0:
                # Closes whatever unit is being filled and takes no bits of
                # its own, which is the only thing C gives a zero width to
                # mean. It does not reserve a unit: the next field starts one.
                offset = (offset + member_alignment - 1) & ~(member_alignment - 1)
                unit_at, unit_bits, unit_size = -1, 0, 0
                continue
            if unit_at < 0 or unit_size != member_size or (
                unit_bits + width > member_size * 8
            ):
                # A new storage unit: this one is full, or the first, or of a
                # different width.
                offset = (offset + member_alignment - 1) & ~(member_alignment - 1)
                unit_at, unit_bits, unit_size = offset, 0, member_size
                offset += member_size
            placed.append(Member(name, ctype, unit_at, width, unit_bits))
            unit_bits += width
            continue
        unit_at, unit_bits, unit_size = -1, 0, 0
        if struct.is_union:
            placed.append(Member(name, ctype, 0, width, 0))
            offset = max(offset, member_size)
            continue
        # Pad forward to this member's own alignment.
        offset = (offset + member_alignment - 1) & ~(member_alignment - 1)
        placed.append(Member(name, ctype, offset))
        offset += member_size
    # Tail padding keeps an array of this type aligned.
    struct.members = tuple(placed)
    struct.alignment = alignment
    struct.size = (offset + alignment - 1) & ~(alignment - 1)


@dataclasses.dataclass(frozen=True, slots=True)
class OpaqueType:
    """A named type py2bin knows only as something a pointer can point at.

    ``PyObject`` is the motivating case: generated C never dereferences a Python
    object, it only passes the handle back to the interpreter. Modelling these
    as incomplete types is what lets the compiler reject ``*handle`` and
    ``handle + 1`` instead of inventing a layout it cannot know.
    """

    name: str

    def __str__(self) -> str:
        return self.name


CType = (
    VoidType
    | IntegerType
    | FloatingType
    | PointerType
    | ArrayType
    | FunctionType
    | OpaqueType
    | StructType
)

VOID = VoidType()

#: The file system calls a C program may make, and how many arguments each
#: takes. Named with the prefix because they are py2bin's own primitives, not
#: POSIX's own spelling: `open` is a name a program may use for its own.
_FILE_BUILTINS: dict[str, tuple[str, int]] = {
    "__py2bin_open": ("open", 3),
    "__py2bin_read": ("read", 3),
    "__py2bin_write": ("write", 3),
    "__py2bin_close": ("close", 1),
    "__py2bin_lseek": ("lseek", 3),
    "__py2bin_mkdir": ("mkdir", 2),
    "__py2bin_rmdir": ("rmdir", 1),
    "__py2bin_unlink": ("unlink", 1),
    "__py2bin_rename": ("rename", 2),
    "__py2bin_access": ("access", 2),
}

#: The names that stop the program rather than returning from it.
_EXIT_BUILTINS = frozenset({"exit", "_Exit", "abort"})

#: What `abort()` leaves behind. A shell reports 134 for SIGABRT, which is
#: what a caller looking for an abort will be testing for.
_ABORT_STATUS = 134

#: Bytes the heap reserves the first time a program asks for memory. One
#: anonymous mapping, made on demand, never grown and never given back -- see
#: :class:`py2bin.native.ir.HeapInit`. <stdlib.h> quotes this same number as
#: __PY2BIN_ARENA_BYTES, so the allocator's idea of where the arena ends and
#: the reservation itself cannot drift apart.
ARENA_BYTES = 64 * 1024 * 1024
CHAR = IntegerType("char", 1, True, 1)
SCHAR = IntegerType("signed char", 1, True, 1)
UCHAR = IntegerType("unsigned char", 1, False, 1)
#: C says char16_t and char32_t are exactly these widths, and unsigned.
CHAR16 = IntegerType("char16_t", 2, False, 2)
CHAR32 = IntegerType("char32_t", 4, False, 3)
#: `wchar_t` is whatever the platform says it is: two bytes on Windows,
#: four everywhere else. A wide literal is encoded to match, so a program
#: that builds for both gets the platform's own answer on each.
WCHAR_NARROW = IntegerType("wchar_t", 2, False, 2)
WCHAR_WIDE = IntegerType("wchar_t", 4, True, 3)


def wchar_for(target: str) -> IntegerType:
    return WCHAR_NARROW if target.startswith("windows-") else WCHAR_WIDE


#: What each prefix means, except `L`, whose answer the target decides.
_WIDE_CHAR_TYPES = {"u": CHAR16, "U": CHAR32}
SHORT = IntegerType("short", 2, True, 2)
USHORT = IntegerType("unsigned short", 2, False, 2)
INT = IntegerType("int", 4, True, 3)
UINT = IntegerType("unsigned int", 4, False, 3)
LONG = IntegerType("long", 8, True, 4)
ULONG = IntegerType("unsigned long", 8, False, 4)
LLONG = IntegerType("long long", 8, True, 5)
ULLONG = IntegerType("unsigned long long", 8, False, 5)
BOOL = IntegerType("_Bool", 1, False, 0)
# Floating ranks sit above every integer rank, which is what makes the usual
# arithmetic conversions pick the floating type in a mixed expression.
FLOAT = FloatingType("float", 4, 6)
DOUBLE = FloatingType("double", 8, 7)

#: `long` on Windows, which is four bytes there and eight everywhere else.
#: Windows is LLP64: it widened its pointers and left `long` where it was, so
#: `LONG` in a platform header is a 32-bit field. py2bin was LP64 on every
#: target, on the reasoning that it never shared a layout with a platform C
#: library - which was true while it compiled nobody's headers but its own,
#: and stopped being true the day it compiled a vendor's. `FORMATETC` holds a
#: `LONG`, and eight bytes where the platform has four moves every member
#: after it and makes the struct the wrong size to hand to anything.
#:
#: The rank stays where `long`'s rank is. Rank is the order the usual
#: arithmetic conversions go in, not a width, and `long` outranks `int` on
#: Windows as everywhere else even though the two are the same size there.
LONG_LLP64 = IntegerType("long", 4, True, 4)
ULONG_LLP64 = IntegerType("unsigned long", 4, False, 4)


def long_for(target: str) -> IntegerType:
    return LONG_LLP64 if target.startswith("windows-") else LONG


def ulong_for(target: str) -> IntegerType:
    return ULONG_LLP64 if target.startswith("windows-") else ULONG


def typedefs_for(target: str) -> "dict[str, CType]":
    """The standard typedefs, with the ones a data model decides settled.

    `size_t` and the pointer-width integers are as wide as a pointer on every
    target; on Windows that is not `unsigned long`, which is why they are
    named here rather than left to whatever `long` turned out to be.
    """

    found = dict(_TYPEDEFS)
    if target.startswith("windows-"):
        found.update(
            {
                "ssize_t": LLONG,
                "size_t": ULLONG,
                "ptrdiff_t": LLONG,
                "intptr_t": LLONG,
                "uintptr_t": ULLONG,
            }
        )
    return found


# The model everywhere but Windows: LP64, where a `long` and a pointer are
# both eight bytes.
_TYPEDEFS: dict[str, CType] = {
    "Py_ssize_t": LLONG,
    "ssize_t": LONG,
    "size_t": ULONG,
    "ptrdiff_t": LONG,
    "intptr_t": LONG,
    "uintptr_t": ULONG,
    "intmax_t": LLONG,
    "uintmax_t": ULLONG,
    "int8_t": SCHAR,
    "uint8_t": UCHAR,
    "int16_t": SHORT,
    "uint16_t": USHORT,
    "int32_t": INT,
    "uint32_t": UINT,
    "int64_t": LLONG,
    "uint64_t": ULLONG,
}

# Types that exist only behind a pointer. Every one of them is a real CPython or
# C library object whose layout py2bin deliberately does not model.
_OPAQUE_NAMES = frozenset(
    {"PyObject", "PyTypeObject", "PyThreadState", "PyCodeObject", "FILE"}
)

_UNSIGNED_COUNTERPART = {
    INT: UINT,
    LONG: ULONG,
    LLONG: ULLONG,
    # And the Windows `long`, which is its own type at its own width.
    LONG_LLP64: ULONG_LLP64,
}

_TYPE_KEYWORDS = frozenset(
    {
        "void",
        "char",
        "short",
        "int",
        "long",
        "signed",
        "unsigned",
        "_Bool",
        "float",
        "double",
        "wchar_t",
        "char16_t",
        "char32_t",
    }
)
# Qualifiers py2bin can honour by ignoring them. ``const`` and ``restrict``
# constrain the program, not the generated code, and every C local here lives
# in addressable stack memory that is loaded and stored on each access, which
# is what ``volatile`` asks for. ``_Atomic`` is deliberately absent: accepting
# it would promise an atomicity this backend does not emit.
_QUALIFIERS = frozenset({"const", "volatile", "restrict"})

_SPECIFIER_COMBINATIONS: dict[tuple[str, ...], CType] = {
    ("void",): VOID,
    ("_Bool",): BOOL,
    ("char16_t",): CHAR16,
    ("char32_t",): CHAR32,
    ("char",): CHAR,
    ("signed", "char"): SCHAR,
    ("char", "signed"): SCHAR,
    ("unsigned", "char"): UCHAR,
    ("char", "unsigned"): UCHAR,
    ("short",): SHORT,
    ("short", "int"): SHORT,
    ("signed", "short"): SHORT,
    ("signed", "short", "int"): SHORT,
    ("unsigned", "short"): USHORT,
    ("unsigned", "short", "int"): USHORT,
    ("int",): INT,
    ("signed",): INT,
    ("signed", "int"): INT,
    ("unsigned",): UINT,
    ("unsigned", "int"): UINT,
    ("long",): LONG,
    ("long", "int"): LONG,
    ("signed", "long"): LONG,
    ("signed", "long", "int"): LONG,
    ("unsigned", "long"): ULONG,
    ("unsigned", "long", "int"): ULONG,
    ("long", "long"): LLONG,
    ("long", "long", "int"): LLONG,
    ("signed", "long", "long"): LLONG,
    ("signed", "long", "long", "int"): LLONG,
    ("unsigned", "long", "long"): ULLONG,
    ("unsigned", "long", "long", "int"): ULLONG,
    ("float",): FLOAT,
    ("double",): DOUBLE,
}

#: Specifier lists that name a real C type py2bin will not pretend to have.
#: ``long double`` is the whole list: accepting it would promise a precision
#: wider than double that this backend does not compute in.
_REJECTED_COMBINATIONS: dict[tuple[str, ...], str] = {
    ("long", "double"): "'long double' is not implemented by py2bin's C compiler; "
    "it evaluates floating expressions in double precision, and accepting the "
    "name would promise a wider type than it computes with",
}


def size_of(ctype: CType) -> int | None:
    """The size in bytes of ``ctype``, or None when it is incomplete."""

    if isinstance(ctype, (IntegerType, FloatingType)):
        return ctype.size
    if isinstance(ctype, PointerType):
        return 8
    if isinstance(ctype, ArrayType):
        element = size_of(ctype.element)
        if element is None or ctype.count is None:
            return None
        return element * ctype.count
    if isinstance(ctype, StructType):
        return ctype.size if ctype.members is not None else None
    return None


def is_signed(ctype: CType) -> bool:
    return isinstance(ctype, IntegerType) and ctype.signed


def is_integer(ctype: CType) -> bool:
    return isinstance(ctype, IntegerType)


def is_floating(ctype: CType) -> bool:
    return isinstance(ctype, FloatingType)


def is_arithmetic(ctype: CType) -> bool:
    """C's arithmetic types: the integer types and the floating ones."""

    return isinstance(ctype, (IntegerType, FloatingType))


def is_scalar(ctype: CType) -> bool:
    return isinstance(ctype, (IntegerType, FloatingType, PointerType))


def promote(ctype: CType) -> CType:
    """C's integer promotions: everything narrower than int becomes int."""

    if isinstance(ctype, IntegerType) and ctype.rank < INT.rank:
        return INT
    return ctype


def arithmetic_conversions(left: CType, right: CType) -> CType:
    """C11 6.3.1.8 with the floating types in front, as the standard orders it.

    If either operand is floating, the common type is the wider of the two
    floating types (or the one floating type when the other operand is an
    integer); only when both are integers do the integer rules apply.
    """

    if is_floating(left) or is_floating(right):
        if left == DOUBLE or right == DOUBLE:
            return DOUBLE
        return FLOAT
    return usual_conversions(left, right)


def usual_conversions(left: IntegerType, right: IntegerType) -> IntegerType:
    """C11 6.3.1.8 for two integer operands."""

    left = promote(left)
    right = promote(right)
    if left == right:
        return left
    if left.signed == right.signed:
        return left if left.rank > right.rank else right
    unsigned, signed = (left, right) if not left.signed else (right, left)
    if unsigned.rank >= signed.rank:
        return unsigned
    if signed.size > unsigned.size:
        return signed
    return _UNSIGNED_COUNTERPART[signed]


def compatible(left: CType, right: CType) -> bool:
    """Assignment compatibility for pointers, ignoring qualifiers."""

    if left == right:
        return True
    if isinstance(left, PointerType) and isinstance(right, PointerType):
        # C11 6.3.2.3 converts between void * and a pointer to an OBJECT only.
        # A pointer to a function is not one, and the standard gives no
        # conversion at all in either direction, so it needs an explicit cast.
        if isinstance(left.target, FunctionType) or isinstance(
            right.target, FunctionType
        ):
            return False
        if isinstance(left.target, VoidType) or isinstance(right.target, VoidType):
            return True
        return left.target == right.target
    return False


# --- the C syntax tree -------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Node:
    token: Token


@dataclasses.dataclass(slots=True)
class IntLiteral(Node):
    value: int
    ctype: CType


@dataclasses.dataclass(slots=True)
class FloatLiteral(Node):
    value: float
    ctype: CType


@dataclasses.dataclass(slots=True)
class StringLiteral(Node):
    #: Bytes for a plain or `u8` literal; code points for a wide one, which
    #: cannot become code units until the target says how wide one is.
    data: "bytes | tuple[int, ...]"
    kind: str = ""

    def bytes_for(self, target: str) -> bytes:
        """The literal's bytes, terminator included, for this target."""

        if isinstance(self.data, bytes):
            return self.data + b"\0"
        width = _unit_width(self.kind, target)
        out = bytearray()
        for point in self.data:
            out.extend(_code_units(point, width))
        out.extend(bytes(width))
        return bytes(out)

    def element_for(self, target: str) -> "CType":
        """The type of one element of the array this literal is."""

        if isinstance(self.data, bytes):
            return CHAR
        if self.kind == "u":
            return CHAR16
        if self.kind == "U":
            return CHAR32
        return wchar_for(target)


def _unit_width(kind: str, target: str) -> int:
    if kind == "u":
        return 2
    if kind == "U":
        return 4
    return wchar_for(target).size


def _code_units(point: int, width: int) -> bytes:
    """One code point as code units of `width` bytes, little-endian.

    Two bytes means UTF-16, and a character outside the basic plane becomes
    a surrogate pair - which is what makes `L"..."` on Windows different
    from `L"..."` anywhere else, rather than merely narrower.
    """

    if width == 2 and point > 0xFFFF:
        point -= 0x10000
        high = 0xD800 + (point >> 10)
        low = 0xDC00 + (point & 0x3FF)
        return high.to_bytes(2, "little") + low.to_bytes(2, "little")
    return point.to_bytes(width, "little")


@dataclasses.dataclass(slots=True)
class Identifier(Node):
    name: str


@dataclasses.dataclass(slots=True)
class Unary(Node):
    operator: str
    operand: Node


@dataclasses.dataclass(slots=True)
class IncDec(Node):
    operator: str  # "++" or "--"
    operand: Node
    prefix: bool


@dataclasses.dataclass(slots=True)
class Binary(Node):
    operator: str
    left: Node
    right: Node


@dataclasses.dataclass(slots=True)
class Logical(Node):
    operator: str  # "&&" or "||"
    left: Node
    right: Node


@dataclasses.dataclass(slots=True)
class Conditional(Node):
    test: Node
    body: Node
    alternative: Node


@dataclasses.dataclass(slots=True)
class Assignment(Node):
    operator: str  # "=", "+=", ...
    target: Node
    value: Node


@dataclasses.dataclass(slots=True)
class Comma(Node):
    left: Node
    right: Node


@dataclasses.dataclass(slots=True)
class Call(Node):
    name: str
    arguments: list[Node]


@dataclasses.dataclass(slots=True)
class CallThrough(Node):
    """``EXPR(args)`` where ``EXPR`` is not a bare name.

    ``(*p)(1)``, ``ops[i](1)`` and ``s.op(1)`` all land here. A call written as
    a bare name is a :class:`Call`, whose name may still turn out to name an
    object of function-pointer type rather than a function.
    """

    target: Node
    arguments: list[Node]


@dataclasses.dataclass(slots=True)
class MemberAccess(Node):
    base: Node
    name: str
    through_pointer: bool


@dataclasses.dataclass(slots=True)
class Index(Node):
    base: Node
    offset: Node


@dataclasses.dataclass(slots=True)
class Cast(Node):
    ctype: CType
    operand: Node


@dataclasses.dataclass(slots=True)
class TypeArgument(Node):
    """A type written where an argument goes, which only `va_arg` does."""

    ctype: "CType"


@dataclasses.dataclass(slots=True)
class SizeofType(Node):
    ctype: CType


@dataclasses.dataclass(slots=True)
class SizeofExpression(Node):
    operand: Node


# --- statements --------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Declaration(Node):
    entries: list[tuple[CType, str, object]]  # (type, name, initializer)
    #: `static int n = 0;` inside a block. One object for the whole program,
    #: named only where it was written.
    stored: bool = False


@dataclasses.dataclass(slots=True)
class ExpressionStatement(Node):
    expression: Node | None


@dataclasses.dataclass(slots=True)
class Compound(Node):
    body: list[Node]


@dataclasses.dataclass(slots=True)
class If(Node):
    test: Node
    body: Node
    alternative: Node | None


@dataclasses.dataclass(slots=True)
class While(Node):
    test: Node
    body: Node


@dataclasses.dataclass(slots=True)
class DoWhile(Node):
    body: Node
    test: Node


@dataclasses.dataclass(slots=True)
class For(Node):
    initializer: Node | None
    test: Node | None
    step: Node | None
    body: Node


@dataclasses.dataclass(slots=True)
class Switch(Node):
    control: Node
    body: Node


@dataclasses.dataclass(slots=True)
class Labeled(Node):
    kind: str  # "case" | "default" | "label"
    value: object  # constant expression node, or the label name
    statement: Node | None


@dataclasses.dataclass(slots=True)
class Goto(Node):
    name: str


@dataclasses.dataclass(slots=True)
class Break(Node):
    pass


@dataclasses.dataclass(slots=True)
class Continue(Node):
    pass


@dataclasses.dataclass(slots=True)
class Return(Node):
    value: Node | None


@dataclasses.dataclass(slots=True)
class Function:
    name: str
    result: CType
    parameters: list[tuple[CType, str]]
    #: ``None`` while only a prototype has been seen. py2bin has no linker, so a
    #: prototype that never acquires a body cannot be called -- but declaring
    #: one is still the ordinary way to write mutual recursion in C.
    body: Compound | None
    token: Token
    #: Whether the parameter list ended with `...`. The extra arguments are
    #: promoted and written into a run of 8-byte cells, whose address is what
    #: `va_start` hands back - so `va_arg` is a load and a step forward.
    variadic: bool = False


@dataclasses.dataclass(slots=True)
class GlobalObject:
    """A file-scope object: one with static storage duration.

    C initializes such an object before the program starts and gives it zero
    when no initializer is written, so ``initializer`` may only be a constant
    expression -- there is nothing running yet that could evaluate anything
    else.
    """

    name: str
    ctype: CType
    initializer: object
    token: Token


@dataclasses.dataclass(slots=True)
class TranslationUnit:
    functions: dict[str, Function]
    externs: dict[str, CType]  # local name -> declared C result type
    enumerators: dict[str, int] = dataclasses.field(default_factory=dict)
    globals: dict[str, GlobalObject] = dataclasses.field(default_factory=dict)
    #: A function this program declares, never defines, and that a library
    #: the program named is claimed to hold: the symbol, the kind of each
    #: argument, and the kind of the result. Read alongside the vetted table.
    library_symbols: dict[str, tuple[str, tuple[str, ...], str]] = (
        dataclasses.field(default_factory=dict)
    )


# --- parser ------------------------------------------------------------------


_BINARY_PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "|": 3,
    "^": 4,
    "&": 5,
    "==": 6,
    "!=": 6,
    "<": 7,
    "<=": 7,
    ">": 7,
    ">=": 7,
    "<<": 8,
    ">>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
}

_ASSIGNMENTS = {"=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^="}

_RESERVED = frozenset(
    {
        "auto",
        "break",
        "case",
        "continue",
        "default",
        "do",
        "else",
        "extern",
        "for",
        "goto",
        "if",
        "register",
        "return",
        "sizeof",
        "static",
        "switch",
        "while",
        *_TYPE_KEYWORDS,
        *_QUALIFIERS,
    }
)

_UNSUPPORTED_KEYWORDS = {
    "static": "'static' is accepted on a declaration - at file scope, where all "
    "it limits is a linkage py2bin's single translation unit has no way to "
    "escape, and inside a block, where it gives the object static storage. It "
    "is not accepted here, which is not a place C puts it either",
    "register": "the 'register' storage class is not accepted",
    "auto": "the 'auto' storage class is not accepted",
    "_Complex": "complex types are not implemented",
    "_Atomic": "atomic types are not implemented; py2bin emits no atomic instructions",
    "_Thread_local": "thread-local storage is not implemented",
}

#: Specifiers that say how a call should be made rather than what anything
#: is. py2bin decides that for itself, so each is read and dropped. Refusing
#: them instead bought nothing and cost everything: `static inline` is how a
#: platform header writes every small function, so a fetched SDK header set
#: stopped at its first one.
_IGNORED_SPECIFIERS = frozenset(
    {"inline", "__inline", "__inline__", "__forceinline"}
)



def _stored_declarations(node: object) -> "list[Declaration]":
    """Every `static` declaration inside a body, at any depth.

    Walked over the dataclass fields rather than through a visitor per node
    type: what is wanted is one kind of node, and every other kind only has
    to be looked through.
    """

    found: "list[Declaration]" = []
    seen: "set[int]" = set()
    pending: "list[object]" = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, (list, tuple)):
            pending.extend(current)
            continue
        if not isinstance(current, Node) or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Declaration) and current.stored:
            found.append(current)
        for field in dataclasses.fields(current):
            if field.name == "token":
                continue
            pending.append(getattr(current, field.name))
    return found

class Parser:
    def __init__(self, tokens: list[Token], filename: str, target: str = ""):
        self.tokens = tokens
        self.filename = filename
        #: Needed here only for `wchar_t`, whose width the platform decides.
        self.target = target
        self.index = 0
        self.functions: dict[str, Function] = {}
        self.externs: dict[str, CType] = {}
        #: Shared libraries the program named, and which symbols each claims.
        self.libraries: "list[tuple[str, frozenset[str]]]" = []
        #: Where each symbol taken from one of them comes from.
        self.symbol_libraries: dict[str, str] = {}
        #: The shape of a call to each, read off the prototype.
        self.library_symbols: "dict[str, tuple[str, tuple[str, ...], str]]" = {}
        #: What `#pragma pack` last said, and what `push` saved. None is the
        #: ABI's own answer, which is what a file without one gets.
        self.pack: "int | None" = None
        self.pack_stack: "list[int | None]" = []
        # struct/union tags are shared across the translation unit, so a
        # tag mentioned inside its own body refers to the same type.
        self.struct_tags: dict[str, StructType] = {}
        # Copied per parse: a typedef in one translation unit must not
        # leak into the next compilation in the same process.
        self.typedefs: dict[str, CType] = typedefs_for(target)
        self.enum_tags: dict[str, CType] = {}
        self.enumerators: dict[str, int] = {}
        self.globals: dict[str, GlobalObject] = {}
        # Set by declarator() to the token its name came from, so a caller that
        # needs to point an error at the name does not have to guess.
        self.declared_token: Token = self.tokens[0]
        self.holes = 0

    # --- token helpers ---

    @property
    def token(self) -> Token:
        return self.tokens[self.index]

    def peek(self, distance: int = 1) -> Token:
        return self.tokens[min(self.index + distance, len(self.tokens) - 1)]

    def error(self, message: str, token: Token | None = None):
        location = token or self.token
        raise CCompileError(
            location.origin or self.filename, location.line, location.column, message
        )

    def at(self, value: str) -> bool:
        return self.token.value == value and self.token.kind in {"symbol", "identifier"}

    def accept(self, value: str) -> bool:
        if not self.at(value):
            return False
        self.index += 1
        return True

    def take(self, value: str | None = None) -> Token:
        token = self.token
        if value is not None and not self.at(value):
            self.error(f"expected {value!r}, found {self.describe(token)}")
        self.index += 1
        return token

    @staticmethod
    def describe(token: Token) -> str:
        if token.kind == "eof":
            return "end of file"
        if token.kind == "string":
            return "a string literal"
        return repr(token.value)

    def identifier(self) -> Token:
        token = self.token
        if token.kind != "identifier":
            self.error(f"expected an identifier, found {self.describe(token)}")
        if token.value in _RESERVED or token.value in _UNSUPPORTED_KEYWORDS:
            self.error(f"{token.value!r} is a keyword and cannot be used as a name")
        self.index += 1
        return token

    # --- types ---

    def at_type(self) -> bool:
        token = self.token
        if token.kind != "identifier":
            return False
        name = str(token.value)
        if name in _UNSUPPORTED_KEYWORDS:
            return True
        if name in _TYPE_KEYWORDS or name in _QUALIFIERS or name in self.typedefs:
            return True
        if name in {"struct", "union", "enum", "typedef"}:
            return True
        if name in _OPAQUE_NAMES:
            # 'PyObject x' is not something py2bin can lay out, but 'PyObject *x'
            # is a handle. Only the pointer form is a type here.
            return self.peek().value == "*"
        return False

    def enum_specifier(self) -> CType:
        """Parse ``enum`` with an optional tag and optional enumerator list.

        C makes each enumerator an ``int`` constant in the ordinary namespace,
        and an unadorned enumeration compatible with ``int``, so the type here
        is simply ``int``. Values continue from the previous enumerator unless
        one is given explicitly.
        """

        keyword = self.take()  # 'enum'
        tag: str | None = None
        if self.token.kind == "identifier" and self.token.value not in _RESERVED:
            tag = str(self.take().value)
        if not self.at("{"):
            if tag is None:
                self.error("an anonymous enum needs a body", keyword)
            if tag not in self.enum_tags:
                self.error(f"enum {tag} has not been defined", keyword)
            return self.enum_tags[tag]
        self.take("{")
        next_value = 0
        while not self.accept("}"):
            if self.token.kind == "eof":
                self.error("unterminated enum body", keyword)
            name_token = self.identifier()
            name = str(name_token.value)
            if name in self.enumerators:
                self.error(f"duplicate enumerator {name!r}", name_token)
            if self.accept("="):
                value_token = self.token
                next_value = ConstantEvaluator(self.filename, self.enumerators).value(
                    self.assignment_expression()
                )
                # A flag enum's "all bits" entry is written 0xffffffff, which
                # is past the signed range and which every real compiler
                # takes - C23 says so outright, and the ones before it took
                # it anyway. Kept as the int of the same bits, because what a
                # value like this is for is masking, and the bits are what
                # masking reads.
                if not -(1 << 31) <= next_value < (1 << 32):
                    self.error(
                        "an enumerator must fit in an int, signed or unsigned",
                        value_token,
                    )
                if next_value >= (1 << 31):
                    next_value -= 1 << 32
            self.enumerators[name] = next_value
            next_value += 1
            if not self.accept(","):
                self.take("}")
                break
        if tag is not None:
            self.enum_tags[tag] = INT
        return INT

    def typedef_declaration(self) -> None:
        """Record ``typedef <type> <name>;`` for the rest of this unit."""

        keyword = self.take()  # 'typedef'
        base = self.type_specifier()
        while True:
            name_token = self.token
            ctype, name = self.declarator(base)
            if not name:
                self.error("a typedef needs a name", keyword)
            existing = self.typedefs.get(name)
            if existing is not None and existing != ctype:
                self.error(f"{name!r} is already a different type", name_token)
            self.typedefs[name] = ctype
            if not self.accept(","):
                break
        self.take(";")

    def struct_specifier(self, is_union: bool) -> "StructType":
        """Parse ``struct``/``union`` with an optional tag and optional body.

        A tag names one type for the whole translation unit, so ``struct T *``
        inside ``struct T`` refers to the same object being defined. A tag with
        no body is a forward declaration: the type stays incomplete, which is
        what lets a pointer to it exist while ``sizeof`` and member access are
        still rejected.
        """

        keyword = self.take()  # 'struct' or 'union'
        tag: str | None = None
        if self.token.kind == "identifier" and self.token.value not in _RESERVED:
            tag = str(self.take().value)
        if tag is not None and tag in self.struct_tags:
            struct = self.struct_tags[tag]
            if struct.is_union != is_union:
                self.error(
                    f"{tag!r} was declared as a "
                    f"{'union' if struct.is_union else 'struct'}",
                    keyword,
                )
        else:
            struct = StructType(tag, is_union)
            if tag is not None:
                self.struct_tags[tag] = struct
        if not self.at("{"):
            if tag is None:
                self.error("an anonymous struct needs a body", keyword)
            return struct
        if struct.members is not None:
            self.error(f"{struct} is defined twice", keyword)
        self.take("{")
        members: "list[tuple[str, CType] | tuple[str, CType, int]]" = []
        seen: set[str] = set()
        while not self.accept("}"):
            if self.token.kind == "eof":
                self.error("unterminated struct body", keyword)
            member_start = self.token
            base = self.type_specifier()
            if self.at(";") and isinstance(base, StructType) and base.name is None:
                # `union { ... };` with no name of its own. Laid out as one
                # member so its size and alignment count, and looked through
                # by `member` so its members are reachable without it.
                if base.members is None:
                    self.error("an anonymous member needs a body", member_start)
                members.append((f"{_ANONYMOUS_MEMBER}{len(members)}", base))
                self.take(";")
                continue
            while True:
                name_token = self.token
                if self.at(":"):
                    # `unsigned int : 3;` - an unnamed bitfield, which pads and
                    # is not reachable. It still takes its bits.
                    self.take(":")
                    width = self.bitfield_width(base, "", member_start)
                    members.append(
                        (f"{_UNNAMED_BITFIELD}{len(members)}", base, width)
                    )
                    if not self.accept(","):
                        break
                    continue
                ctype, member_name = self.declarator(base)
                if self.at(":"):
                    self.take(":")
                    width = self.bitfield_width(ctype, member_name, name_token)
                    if member_name in seen:
                        self.error(f"duplicate member {member_name!r}", name_token)
                    seen.add(member_name)
                    members.append((member_name, ctype, width))
                    if not self.accept(","):
                        break
                    continue
                if member_name in seen:
                    self.error(
                        f"duplicate member {member_name!r}", name_token
                    )
                if isinstance(ctype, StructType) and ctype.members is None:
                    self.error(
                        f"member {member_name!r} has incomplete type {ctype}",
                        member_start,
                    )
                if isinstance(ctype, VoidType):
                    self.error(f"member {member_name!r} cannot be void", member_start)
                if isinstance(ctype, FunctionType):
                    # C has no member of function type. Without this the member
                    # would be laid out with a size of zero and silently alias
                    # whatever followed it.
                    self.error(
                        f"member {member_name!r} has function type {ctype}; C has "
                        f"no such member. Write '{ctype.result} (*{member_name})"
                        "(...)' for a pointer to a function",
                        member_start,
                    )
                if isinstance(ctype, ArrayType) and size_of(ctype) is None:
                    self.error(
                        f"member {member_name!r} has the incomplete type {ctype}",
                        member_start,
                    )
                seen.add(member_name)
                members.append((member_name, ctype))
                if not self.accept(","):
                    break
            self.take(";")
        if not members:
            self.error("an empty struct has no size in C", keyword)
        lay_out(struct, members, self.pack)
        return struct

    def bitfield_width(self, ctype: CType, name: str, token: Token) -> int:
        """The `: 3` on a member: how many bits of its storage unit it takes."""

        spelled = ConstantEvaluator(self.filename, self.enumerators).value(
            self.assignment_expression()
        )
        held = size_of(ctype)
        if not isinstance(ctype, IntegerType) or held is None:
            self.error(
                f"a bitfield has to have an integer type, and {name or 'this one'} "
                f"is declared {ctype}",
                token,
            )
        if spelled < 0 or spelled > held * 8:
            self.error(
                f"a bitfield of {ctype} holds between 0 and {held * 8} bits, "
                f"and {name or 'this one'} asks for {spelled}",
                token,
            )
        if spelled == 0 and name:
            self.error(
                "only an unnamed bitfield may be zero bits wide; that is what "
                "says the next one starts a new storage unit",
                token,
            )
        return spelled

    def type_specifier(self) -> CType:
        """Parse a declaration specifier list into one type."""

        start = self.token
        words: list[str] = []
        base: CType | None = None
        while self.token.kind == "identifier":
            name = str(self.token.value)
            if name in {"struct", "union"} and base is None and not words:
                base = self.struct_specifier(name == "union")
                continue
            if name == "enum" and base is None and not words:
                base = self.enum_specifier()
                continue
            if name in _IGNORED_SPECIFIERS:
                self.index += 1
                continue
            if name == "__declspec":
                self.skip_declspec()
                continue
            if name in _UNSUPPORTED_KEYWORDS:
                self.error(_UNSUPPORTED_KEYWORDS[name])
            if name in _QUALIFIERS:
                self.index += 1
                continue
            if name in _TYPE_KEYWORDS:
                words.append(name)
                self.index += 1
                continue
            if base is None and not words and name in self.typedefs:
                base = self.typedefs[name]
                self.index += 1
                continue
            if base is None and not words and name in _OPAQUE_NAMES:
                base = OpaqueType(name)
                self.index += 1
                continue
            break
        if words:
            if base is not None:
                self.error("conflicting type specifiers", start)
            key = tuple(words)
            rejection = _REJECTED_COMBINATIONS.get(key) or _REJECTED_COMBINATIONS.get(
                tuple(sorted(key))
            )
            if rejection is not None:
                self.error(rejection, start)
            if key == ("wchar_t",):
                resolved = wchar_for(self.target)
            else:
                resolved = _SPECIFIER_COMBINATIONS.get(key)
                if resolved is None:
                    resolved = _SPECIFIER_COMBINATIONS.get(tuple(sorted(key)))
                if resolved is LONG:
                    resolved = long_for(self.target)
                elif resolved is ULONG:
                    resolved = ulong_for(self.target)
            if resolved is None:
                self.error(f"unsupported type specifier {' '.join(words)!r}", start)
            base = resolved
        if base is None:
            self.error(f"expected a type name, found {self.describe(start)}", start)
        return base

    def skip_declspec(self) -> None:
        """Read `__declspec(...)` and drop it - except where it decides layout.

        What a generated COM header puts in one is `selectany`, `novtable`,
        `uuid` and `xfg_virtual`: a linkage hint for a definition repeated
        across translation units, two optimisation hints, and a GUID that the
        same header also writes out as an ordinary constant. None of them
        changes a byte of what is emitted here, and a single translation unit
        has no linkage to choose anyway.

        `align` does change layout, so it is refused rather than dropped.
        Quietly ignoring it would move every member after it and the program
        would run and be wrong, which is the failure worth the most care.
        """

        keyword = self.take()  # '__declspec'
        if not self.at("("):
            self.error("__declspec takes a parenthesised attribute", keyword)
        depth = 0
        while True:
            if self.token.kind == "eof":
                self.error("unterminated __declspec", keyword)
            if self.at("("):
                depth += 1
            elif self.at(")"):
                depth -= 1
            elif self.token.kind == "identifier" and self.token.value == "align":
                self.error(
                    "__declspec(align) decides where every member after it "
                    "sits, and py2bin does not implement it. Dropping it "
                    "would build a struct of a different shape than the one "
                    "written, so it is refused instead",
                    self.token,
                )
            self.index += 1
            if depth == 0:
                return

    def pointer_suffix(self, base: CType) -> CType:
        while True:
            if self.accept("*"):
                base = PointerType(base)
                while self.token.kind == "identifier" and self.token.value in _QUALIFIERS:
                    self.index += 1
                continue
            return base

    def declarator(
        self, base: CType, *, abstract: bool = False, optional: bool = False
    ) -> tuple[CType, str]:
        """Parse a full C declarator and return the type it builds, plus its name.

        This is C's real declarator grammar, so ``int (*p)(int)``, ``int
        (*ops[3])(void)`` and ``int *(*f)(char *)`` all read correctly: the
        prefix ``*``s bind loosest, the postfix ``[]`` and ``()`` bind tighter
        and apply left to right, and parentheses regroup. The inner declarator
        is parsed against a hole and the outer type is substituted into it,
        which is exactly what "declaration mirrors use" means.

        ``optional`` accepts either form, which is what a prototype's parameter
        list needs: ``int f(int, int *);`` names nothing, ``int f(int a);`` does.
        ``self.declared_token`` is left holding the token the name came from, so
        a caller can point an error at it.
        """

        self.declared_token = self.token
        base = self.pointer_suffix(base)
        if self.at("(") and self.at_nested_declarator():
            self.take("(")
            hole = _Hole(self.next_hole())
            inner, name = self.declarator(hole, abstract=abstract, optional=optional)
            self.take(")")
            return _fill(inner, hole, self.declarator_suffix(base)), name
        name = ""
        if not abstract and not (optional and self.token.kind != "identifier"):
            self.declared_token = self.token
            name = str(self.identifier().value)
        return self.declarator_suffix(base), name

    def next_hole(self) -> int:
        self.holes += 1
        return self.holes

    def at_nested_declarator(self) -> bool:
        """Whether the ``(`` here regroups a declarator rather than starting a
        parameter list.

        ``int (*p)(void)`` regroups; ``int (void)`` and ``int (int, char *)``
        are parameter lists. The two are told apart by what follows the ``(``:
        a ``*``, another ``(``, or a name that is not a type name can only
        begin a declarator.
        """

        saved = self.index
        self.index += 1
        try:
            if self.at("*") or self.at("("):
                return True
            if self.token.kind != "identifier" or self.token.value in _RESERVED:
                return False
            return not self.at_type()
        finally:
            self.index = saved

    def declarator_suffix(self, base: CType) -> CType:
        """Apply a declarator's postfix ``[N]`` and ``(params)`` groups.

        They bind tighter than the prefix ``*``s and apply left to right, so
        ``int a[2][3]`` is an array of 2 arrays of 3 ints and the list is
        applied in reverse.
        """

        suffixes: list[tuple[str, object]] = []
        while True:
            if self.accept("["):
                if self.accept("]"):
                    suffixes.append(("array", None))
                    continue
                length = self.array_length()
                self.take("]")
                suffixes.append(("array", length))
                continue
            if self.at("("):
                suffixes.append(("function", self.parameter_type_list()))
                continue
            break
        for kind, payload in reversed(suffixes):
            if kind == "array":
                if isinstance(base, FunctionType):
                    self.error(f"an array of {base} is not a type C has")
                base = ArrayType(base, payload)  # type: ignore[arg-type]
            else:
                if isinstance(base, (ArrayType, FunctionType)):
                    self.error(f"a function cannot return {base}")
                base = FunctionType(base, payload)  # type: ignore[arg-type]
        return base

    def parameter_type_list(self) -> tuple[CType, ...]:
        """``(void)`` or ``(TYPE, TYPE, ...)`` in a function declarator."""

        token = self.take("(")
        if self.accept(")"):
            self.error(
                "a function type must state its parameter types; write '(void)' "
                "for a function that takes none. C's empty '()' means the "
                "parameters are UNSPECIFIED, and py2bin will not emit a call it "
                "cannot check",
                token,
            )
        if self.at("void") and self.peek().value == ")":
            self.take("void")
            self.take(")")
            return ()
        parameters: list[CType] = []
        while True:
            if self.at("..."):
                self.error("a variadic function type is not implemented")
            parameter, _name = self.declarator(self.type_specifier(), optional=True)
            if isinstance(parameter, ArrayType):
                # A parameter of array type is adjusted to a pointer, and one of
                # function type to a pointer to the function (C11 6.7.6.3p7-8).
                parameter = PointerType(parameter.element)
            elif isinstance(parameter, FunctionType):
                parameter = PointerType(parameter)
            if isinstance(parameter, VoidType):
                self.error("a parameter cannot have type void", token)
            parameters.append(parameter)
            if self.accept(")"):
                break
            self.take(",")
        return tuple(parameters)

    def array_suffix(self, base: CType) -> CType:
        """The ``[N]`` part of a declarator whose name has already been taken."""

        return self.declarator_suffix(base)

    def array_length(self) -> int:
        token = self.token
        value = ConstantEvaluator(self.filename, self.enumerators).value(self.assignment_expression())
        if value <= 0:
            self.error("an array needs a positive constant length", token)
        return value

    def type_name(self) -> CType:
        """A type in a cast or ``sizeof``: specifiers plus an abstract declarator.

        The abstract declarator is the ordinary one with the name left out, so
        ``int (*)(int)`` -- the type a cast to a function pointer names -- reads
        through exactly the same code that reads ``int (*p)(int)``.
        """

        ctype, _name = self.declarator(self.type_specifier(), abstract=True)
        return ctype

    # --- expressions ---

    def expression(self) -> Node:
        node = self.assignment_expression()
        while self.at(","):
            token = self.take(",")
            node = Comma(token, node, self.assignment_expression())
        return node

    def assignment_expression(self) -> Node:
        node = self.conditional_expression()
        if self.token.kind == "symbol" and self.token.value in _ASSIGNMENTS:
            token = self.take()
            return Assignment(
                token, str(token.value), node, self.assignment_expression()
            )
        return node

    def conditional_expression(self) -> Node:
        node = self.binary_expression(1)
        if self.at("?"):
            token = self.take("?")
            body = self.expression()
            self.take(":")
            return Conditional(token, node, body, self.conditional_expression())
        return node

    def binary_expression(self, minimum: int) -> Node:
        node = self.unary_expression()
        while True:
            if self.token.kind != "symbol":
                return node
            operator = str(self.token.value)
            precedence = _BINARY_PRECEDENCE.get(operator)
            if precedence is None or precedence < minimum:
                return node
            token = self.take()
            right = self.binary_expression(precedence + 1)
            if operator in {"&&", "||"}:
                node = Logical(token, operator, node, right)
            else:
                node = Binary(token, operator, node, right)

    def unary_expression(self) -> Node:
        token = self.token
        if token.kind == "symbol" and token.value in {"++", "--"}:
            self.take()
            return IncDec(token, str(token.value), self.unary_expression(), True)
        if token.kind == "symbol" and token.value in {"+", "-", "!", "~", "*", "&"}:
            self.take()
            return Unary(token, str(token.value), self.unary_expression())
        if token.kind == "identifier" and token.value == "sizeof":
            self.take()
            if self.at("(") and self.at_type_after_parenthesis():
                self.take("(")
                ctype = self.type_name()
                self.take(")")
                return SizeofType(token, ctype)
            return SizeofExpression(token, self.unary_expression())
        if self.at("(") and self.at_type_after_parenthesis():
            self.take("(")
            ctype = self.type_name()
            self.take(")")
            return Cast(token, ctype, self.unary_expression())
        return self.postfix_expression()

    def at_type_after_parenthesis(self) -> bool:
        saved = self.index
        self.index += 1
        result = self.at_type()
        self.index = saved
        return result

    def postfix_expression(self) -> Node:
        node = self.primary_expression()
        while True:
            token = self.token
            if self.accept("["):
                offset = self.expression()
                self.take("]")
                node = Index(token, node, offset)
                continue
            if token.kind == "symbol" and token.value in {"++", "--"}:
                self.take()
                node = IncDec(token, str(token.value), node, False)
                continue
            if token.kind == "symbol" and token.value in {".", "->"}:
                self.take()
                member_token = self.identifier()
                node = MemberAccess(
                    token, node, str(member_token.value), token.value == "->"
                )
                continue
            if self.at("("):
                self.take("(")
                arguments: list[Node] = []
                if not self.accept(")"):
                    while True:
                        arguments.append(self.assignment_expression())
                        if self.accept(")"):
                            break
                        self.take(",")
                node = CallThrough(token, node, arguments)
                continue
            return node

    def primary_expression(self) -> Node:
        token = self.token
        if token.kind == "integer":
            self.take()
            if token.encoding:
                # `L'x'` has type wchar_t, `u'x'` char16_t, `U'x'` char32_t -
                # not int, which is what an unprefixed one has.
                return IntLiteral(
                    token, int(token.value), _WIDE_CHAR_TYPES.get(
                        token.encoding, wchar_for(self.target)
                    ) if token.encoding != "L" else wchar_for(self.target),
                )
            return IntLiteral(
                token,
                int(token.value),
                _literal_type(token, self.filename, self.target),
            )
        if token.kind == "float":
            self.take()
            # An unsuffixed floating constant has type double; 'f' makes it a
            # float, and the lexer already rounded its value to binary32.
            return FloatLiteral(
                token, float(token.value), FLOAT if token.suffix == "f" else DOUBLE
            )
        if token.kind == "string":
            self.take()
            kind = token.encoding
            if kind in ("", "u8"):
                data = bytes(token.value)  # type: ignore[arg-type]
                while self.token.kind == "string":  # adjacent ones concatenate
                    if self.token.encoding not in ("", "u8"):
                        self.error(
                            "a plain string literal joined to a wide one; C "
                            "leaves what that means undefined, so py2bin will "
                            "not choose",
                            self.token,
                        )
                    data += bytes(self.take().value)  # type: ignore[arg-type]
                return StringLiteral(token, data, kind)
            points = tuple(token.value)  # type: ignore[arg-type]
            while self.token.kind == "string":
                if self.token.encoding != kind:
                    self.error(
                        f"a {kind or 'plain'} string literal joined to a "
                        f"{self.token.encoding or 'plain'} one; C leaves what "
                        "that means undefined, so py2bin will not choose",
                        self.token,
                    )
                points += tuple(self.take().value)  # type: ignore[arg-type]
            return StringLiteral(token, points, kind)
        if token.kind == "identifier":
            if token.value in _UNSUPPORTED_KEYWORDS:
                self.error(_UNSUPPORTED_KEYWORDS[str(token.value)])
            if token.value in _RESERVED:
                self.error(f"unexpected keyword {token.value!r} in an expression")
            name = str(self.identifier().value)
            if self.accept("("):
                arguments: list[Node] = []
                if not self.accept(")"):
                    while True:
                        if name == "va_arg" and len(arguments) == 1:
                            # The second argument names a type, which is where
                            # `va_arg` differs from every other call in C. Read
                            # the way `sizeof(int)` is read.
                            arguments.append(
                                TypeArgument(self.token, self.type_name())
                            )
                        else:
                            arguments.append(self.assignment_expression())
                        if self.accept(")"):
                            break
                        self.take(",")
                return Call(token, name, arguments)
            return Identifier(token, name)
        if self.accept("("):
            node = self.expression()
            self.take(")")
            return node
        self.error(f"expected an expression, found {self.describe(token)}")

    # --- statements ---

    def compound_statement(self) -> Compound:
        token = self.take("{")
        body: list[Node] = []
        while not self.accept("}"):
            if self.token.kind == "eof":
                self.error("unterminated block")
            body.append(self.statement())
        return Compound(token, body)

    def statement(self) -> Node:
        token = self.token
        if self.at("{"):
            return self.compound_statement()
        if token.kind == "identifier" and token.value == "typedef":
            self.typedef_declaration()
            return Declaration(token, [])
        if (
            token.kind == "identifier"
            and token.value in {"struct", "union", "enum"}
            and self.declares_type_only()
        ):
            self.type_specifier()
            self.take(";")
            return Declaration(token, [])
        if self.at_type():
            return self.declaration_statement()
        if token.kind == "identifier":
            keyword = str(token.value)
            if keyword == "if":
                self.take()
                self.take("(")
                test = self.expression()
                self.take(")")
                body = self.statement()
                alternative = self.statement() if self.accept("else") else None
                return If(token, test, body, alternative)
            if keyword == "while":
                self.take()
                self.take("(")
                test = self.expression()
                self.take(")")
                return While(token, test, self.statement())
            if keyword == "do":
                self.take()
                body = self.statement()
                self.take("while")
                self.take("(")
                test = self.expression()
                self.take(")")
                self.take(";")
                return DoWhile(token, body, test)
            if keyword == "for":
                return self.for_statement()
            if keyword == "switch":
                self.take()
                self.take("(")
                control = self.expression()
                self.take(")")
                return Switch(token, control, self.statement())
            if keyword == "break":
                self.take()
                self.take(";")
                return Break(token)
            if keyword == "continue":
                self.take()
                self.take(";")
                return Continue(token)
            if keyword == "return":
                self.take()
                if self.accept(";"):
                    return Return(token, None)
                value = self.expression()
                self.take(";")
                return Return(token, value)
            if keyword == "goto":
                self.take()
                name = str(self.identifier().value)
                self.take(";")
                return Goto(token, name)
            if keyword == "case":
                self.take()
                value = self.conditional_expression()
                self.take(":")
                return Labeled(token, "case", value, self.labeled_body())
            if keyword == "default":
                self.take()
                self.take(":")
                return Labeled(token, "default", None, self.labeled_body())
            if self.peek().value == ":" and self.peek().kind == "symbol":
                name = str(self.identifier().value)
                self.take(":")
                return Labeled(token, "label", name, self.labeled_body())
        if self.accept(";"):
            return ExpressionStatement(token, None)
        expression = self.expression()
        self.take(";")
        return ExpressionStatement(token, expression)

    def labeled_body(self) -> Node | None:
        """The statement a label introduces, which may be absent before '}'."""

        if self.at("}"):
            return None
        return self.statement()

    def declaration_statement(self) -> Declaration:
        token = self.token
        # `static int n = 0;` - one object, initialised once, named only
        # here. Read off before the type, which is where C++ and C both put
        # it and where the type reader would otherwise refuse it.
        stored = False
        while self.token.kind == "identifier" and (
            self.token.value == "static" or self.token.value in _IGNORED_SPECIFIERS
        ):
            stored = stored or self.token.value == "static"
            self.index += 1
        base = self.type_specifier()
        entries: list[tuple[CType, str, object]] = []
        while True:
            ctype, name = self.declarator(base)
            initializer: object = None
            if self.accept("="):
                initializer = self.initializer()
            entries.append((ctype, name, initializer))
            if self.accept(";"):
                break
            self.take(",")
        return Declaration(token, entries, stored)

    def initializer(self) -> object:
        if self.at("{"):
            token = self.take("{")
            items: list[object] = []
            if not self.accept("}"):
                while True:
                    items.append(self.initializer())
                    if self.accept("}"):
                        break
                    self.take(",")
                    if self.accept("}"):  # a trailing comma is legal C
                        break
            return (token, items)
        return self.assignment_expression()

    def for_statement(self) -> For:
        token = self.take("for")
        self.take("(")
        initializer: Node | None
        if self.accept(";"):
            initializer = None
        elif self.at_type():
            initializer = self.declaration_statement()
        else:
            initializer = ExpressionStatement(self.token, self.expression())
            self.take(";")
        test = None if self.at(";") else self.expression()
        self.take(";")
        step = None if self.at(")") else self.expression()
        self.take(")")
        return For(token, initializer, test, step, self.statement())

    # --- translation unit ---

    def pragma_pack(self) -> None:
        """`#pragma pack(...)`, handed here as tokens by the preprocessor.

        It caps how far a member may be padded forward, which is how every
        binary format and every platform header describes a layout that is
        not the one the ABI would choose. The forms are the ones MSVC and GCC
        both take: a number, `push`/`pop` with or without one, and nothing at
        all, which goes back to the ABI's own answer.
        """

        marker = self.take()
        given: "list[object]" = []
        if self.accept("("):
            while not self.accept(")"):
                if self.token.kind == "eof":
                    self.error("#pragma pack is not closed", marker)
                item = self.take()
                if item.value != ",":
                    given.append(item.value)
        self.accept(";")
        words = [one for one in given if isinstance(one, str)]
        numbers = [one for one in given if isinstance(one, int)]
        if "show" in words:
            return
        if "pop" in words:
            self.pack = self.pack_stack.pop() if self.pack_stack else None
            if numbers:
                self.pack = self.checked_pack(numbers[-1], marker)
            return
        if "push" in words:
            self.pack_stack.append(self.pack)
            if numbers:
                self.pack = self.checked_pack(numbers[-1], marker)
            return
        if not given:
            # `#pragma pack()` goes back to what the ABI says.
            self.pack = None
            return
        if numbers:
            self.pack = self.checked_pack(numbers[-1], marker)
            return
        self.error(
            f"py2bin does not know what `#pragma pack` means with "
            f"{', '.join(str(one) for one in given)}; it takes a number, "
            f"`push`, `pop`, or nothing",
            marker,
        )

    def checked_pack(self, value: object, token: Token) -> int:
        """The alignment a pack directive named, which C says is a power of two."""

        if not isinstance(value, int) or value <= 0 or value & (value - 1):
            self.error(
                f"#pragma pack takes a power of two, not {value!r}", token
            )
        if value > 16:
            self.error(
                f"#pragma pack({value}) is wider than any alignment py2bin "
                f"gives a type, so it would change nothing; py2bin says so "
                f"rather than accepting a number it cannot honour",
                token,
            )
        return int(value)

    def translation_unit(self) -> TranslationUnit:
        while self.token.kind != "eof":
            if self.token.kind == "identifier" and self.token.value == _PACK_MARKER:
                self.pragma_pack()
                continue
            if self.accept("extern"):
                self.extern_prototype()
                continue
            # A struct or union definition with no declarator introduces a type
            # and nothing else: `struct P { int x; };`
            if self.token.kind == "identifier" and self.token.value == "typedef":
                self.typedef_declaration()
                continue
            if (
                self.token.kind == "identifier"
                and self.token.value in {"struct", "union", "enum"}
                and self.declares_type_only()
            ):
                self.type_specifier()
                self.take(";")
                continue
            self.external_declaration()
        self.bind_libraries()
        return TranslationUnit(
            self.functions,
            self.externs,
            self.enumerators,
            self.globals,
            self.library_symbols,
        )

    def bind_libraries(self) -> None:
        """Bind each undefined function a named library is claimed to hold.

        py2bin has no linker, so a function this unit declares and never
        defines is normally a mistake worth reporting. Where the program has
        named the library it lives in, it is not a mistake: it is an import,
        and the shape of the call is read off the prototype the program
        wrote - which is the same thing a linker reads out of an import
        library, said in the source instead.
        """

        if not self.libraries:
            return
        for name, function in self.functions.items():
            if function.body is not None or name in self.externs:
                continue
            library = self.library_for(name)
            if library is None:
                continue
            kinds = tuple(
                _abi_kind(held) for held, _spelled in function.parameters
            )
            if any(kind is None for kind in kinds):
                continue
            result = _abi_kind(function.result, result=True)
            if result is None:
                continue
            self.library_symbols[name] = (name, kinds, result)
            self.externs[name] = function.result
            self.symbol_libraries[name] = library

    def library_for(self, name: str) -> "str | None":
        """Which named library claims that symbol, if any."""

        for library, claimed in self.libraries:
            if not claimed or name in claimed:
                return library
        return None

    def declares_type_only(self) -> bool:
        """Whether a struct/union specifier is followed straight by ``;``.

        Scans ahead without consuming: `struct P { ... };` declares a type,
        while `struct P p;` and `struct P f(void)` declare an object or a
        function and must go through the normal path.
        """

        index = self.index + 1
        if (
            index < len(self.tokens)
            and self.tokens[index].kind == "identifier"
            and self.tokens[index].value not in _RESERVED
        ):
            index += 1
        if index < len(self.tokens) and self.tokens[index].value == "{":
            depth = 0
            while index < len(self.tokens):
                value = self.tokens[index].value
                if value == "{":
                    depth += 1
                elif value == "}":
                    depth -= 1
                    if depth == 0:
                        index += 1
                        break
                index += 1
        return index < len(self.tokens) and self.tokens[index].value == ";"

    def external_declaration(self) -> None:
        """One file-scope declaration: a function, or an object.

        A plain ``TYPE *... name (`` is a function declaration or definition and
        keeps its own path, because that path also needs the parameter NAMES.
        Everything else -- including ``int (*handler)(int);``, whose declarator
        also contains a parameter list -- is an object. ``static`` is accepted
        and ignored here: it limits linkage, and a single translation unit with
        no linker has no linkage to limit.
        """

        while self.token.kind == "identifier" and (
            self.token.value == "static" or self.token.value in _IGNORED_SPECIFIERS
        ):
            self.take()
        base = self.type_specifier()
        if self.at_function_declarator():
            declared = self.pointer_suffix(base)
            name_token = self.identifier()
            self.function_definition(declared, name_token)
            return
        self.object_declaration(base)

    def at_function_declarator(self) -> bool:
        """Whether what follows is ``*``... ``name`` ``(`` -- a function, not an
        object. ``int (*p)(void)`` fails this test at its very first token."""

        index = self.index
        while index < len(self.tokens):
            token = self.tokens[index]
            if token.value == "*" or (
                token.kind == "identifier" and token.value in _QUALIFIERS
            ):
                index += 1
                continue
            break
        if index + 1 >= len(self.tokens):
            return False
        name = self.tokens[index]
        if name.kind != "identifier" or name.value in _RESERVED:
            return False
        return self.tokens[index + 1].value == "("

    def object_declaration(self, base: CType) -> None:
        """``TYPE name[= init], *other, third[3];`` at file scope."""

        while True:
            ctype, name = self.declarator(base)
            name_token = self.declared_token
            if not name:
                self.error("a file-scope declaration needs a name", name_token)
            if isinstance(ctype, FunctionType):
                # `int (*f(int))(int)` -- a function whose own declarator is not
                # simply `TYPE *... name(params)`. The parameter NAMES are what
                # a definition needs, and this shape does not deliver them here.
                self.error(
                    f"{name!r} is declared as a function here, and py2bin only "
                    "implements the plain declarator form 'TYPE *... name"
                    "(params)'. Give the result type a typedef and write "
                    f"'<typedef> {name}(params)'",
                    name_token,
                )
            initializer: object = None
            if self.accept("="):
                initializer = self.initializer()
            self.declare_global(
                GlobalObject(name, ctype, initializer, name_token)
            )
            if self.accept(";"):
                return
            self.take(",")

    def declare_global(self, entry: GlobalObject) -> None:
        if entry.name in self.functions or entry.name in self.externs:
            self.error(
                f"{entry.name!r} is already declared as a function", entry.token
            )
        if entry.name in self.enumerators:
            self.error(
                f"{entry.name!r} is already an enumeration constant", entry.token
            )
        previous = self.globals.get(entry.name)
        if previous is not None:
            # C's tentative definitions: `int x; int x = 1;` is one object. Two
            # initializers for it are not, and neither is a change of type.
            if previous.ctype != entry.ctype:
                self.error(
                    f"{entry.name!r} was declared {previous.ctype} and is now "
                    f"declared {entry.ctype}",
                    entry.token,
                )
            if previous.initializer is not None and entry.initializer is not None:
                self.error(f"{entry.name!r} is initialized twice", entry.token)
            if entry.initializer is None:
                return
        self.globals[entry.name] = entry

    def function_definition(self, result: CType, name_token: Token) -> None:
        name = str(name_token.value)
        self.take("(")
        parameters: list[tuple[CType, str]] = []
        variadic = False
        if not self.accept(")"):
            if self.at("void") and self.peek().value == ")":
                self.take("void")
                self.take(")")
            else:
                while True:
                    if self.at("..."):
                        # `int f(int n, ...)`. The extra arguments are promoted
                        # and written into a run of cells at the call, and the
                        # address of that run is the `va_list`.
                        self.take("...")
                        if not parameters:
                            self.error(
                                "a variadic function needs a named parameter "
                                "before the '...', because `va_start` names it"
                            )
                        variadic = True
                        self.take(")")
                        break
                    parameter_type, parameter_name = self.declarator(
                        self.type_specifier(), optional=True
                    )
                    if isinstance(parameter_type, ArrayType):
                        # A parameter of array type is adjusted to a pointer.
                        parameter_type = PointerType(parameter_type.element)
                    parameters.append((parameter_type, parameter_name))
                    if self.accept(")"):
                        break
                    self.take(",")
        if name in self.externs or name in self.globals:
            self.error(f"{name!r} is already declared", name_token)
        previous = self.functions.get(name)
        if previous is not None:
            self.check_redeclaration(previous, result, parameters, name_token)
        if self.accept(";"):
            # A prototype. It carries no body, and repeating it is legal C as
            # long as the signature agrees, which check_redeclaration enforced.
            if previous is None:
                self.functions[name] = Function(
                    name, result, parameters, None, name_token, variadic
                )
            return
        if previous is not None and previous.body is not None:
            self.error(f"{name!r} is already defined", name_token)
        # A parameter with no name is one the body does not use, which C++
        # has always allowed and C allows as of C23 - and which a generated
        # callback is full of: an event handler takes the object it fired on
        # and usually wants only the arguments. Given a name nothing can
        # spell, so it takes its place in the frame and nothing reaches it.
        parameters = [
            (held, spelled or f"{_UNNAMED_PARAMETER}{index}")
            for index, (held, spelled) in enumerate(parameters)
        ]
        seen: set[str] = set()
        for parameter_type, parameter_name in parameters:
            if parameter_name in seen:
                self.error(f"duplicate parameter {parameter_name!r}", name_token)
            seen.add(parameter_name)
            if isinstance(parameter_type, VoidType):
                self.error("a parameter cannot have type void", name_token)
        # The signature is registered before the body is parsed so that the
        # function's own name -- and any name a prototype introduced -- resolves
        # inside it. That is what makes direct and mutual recursion parseable.
        self.functions[name] = Function(
            name, result, parameters, None, name_token, variadic
        )
        body = self.compound_statement()
        self.functions[name] = Function(
            name, result, parameters, body, name_token, variadic
        )

    def check_redeclaration(
        self,
        previous: Function,
        result: CType,
        parameters: list[tuple[CType, str]],
        name_token: Token,
    ) -> None:
        """C requires a redeclaration to agree with what came before."""

        if previous.result != result:
            self.error(
                f"{previous.name!r} was declared to return {previous.result} and is "
                f"now declared to return {result}",
                name_token,
            )
        if len(previous.parameters) != len(parameters):
            self.error(
                f"{previous.name!r} was declared with {len(previous.parameters)} "
                f"parameter(s) and is now declared with {len(parameters)}",
                name_token,
            )
        for position, ((old, _old_name), (new, _new_name)) in enumerate(
            zip(previous.parameters, parameters), 1
        ):
            if old != new:
                self.error(
                    f"parameter {position} of {previous.name!r} was declared "
                    f"{old} and is now declared {new}",
                    name_token,
                )

    def extern_prototype(self) -> None:
        """``extern TYPE name(TYPE, ...);`` bound to py2bin's vetted adapter ABI.

        Writing the prototype out is what removes the need for a preprocessor:
        the source states the exact ABI it uses, and the compiler checks that
        statement against the table it will actually emit a call for.
        """

        result = self.pointer_suffix(self.type_specifier())
        name_token = self.identifier()
        name = str(name_token.value)
        if not self.at("("):
            self.error(
                f"'extern {name}' declares an object defined in another "
                "translation unit; py2bin compiles exactly one translation unit "
                "and has no linker, so only 'extern' function prototypes bound "
                "to the vetted adapter ABI are accepted. Drop the 'extern' to "
                "define the object here.",
                name_token,
            )
        self.take("(")
        declared: list[CType] = []
        if not self.accept(")"):
            while True:
                parameter_type = self.pointer_suffix(self.type_specifier())
                if isinstance(parameter_type, VoidType) and not declared:
                    pass
                else:
                    declared.append(parameter_type)
                if self.token.kind == "identifier" and not self.at_type():
                    self.identifier()
                if self.accept(")"):
                    break
                self.take(",")
        self.take(";")
        if name not in _CABI_SYMBOLS:
            self.error(
                f"external symbol {name!r} is not in py2bin's vetted adapter ABI; "
                f"choose one of {', '.join(sorted(_CABI_SYMBOLS))}",
                name_token,
            )
        _symbol, signature = _CABI_SYMBOLS[name]
        if len(declared) != len(signature):
            self.error(
                f"prototype for {name!r} declares {len(declared)} parameter(s) but "
                f"its vetted adapter ABI takes {len(signature)}",
                name_token,
            )
        for position, (declared_type, kind) in enumerate(zip(declared, signature), 1):
            if not _matches_abi(declared_type, kind):
                self.error(
                    f"parameter {position} of {name!r} is declared "
                    f"{declared_type!r} but its vetted adapter ABI passes {kind!r}",
                    name_token,
                )
        if not _matches_abi(result, _CABI_RESULTS[name], result=True):
            self.error(
                f"prototype for {name!r} returns {str(result)!r} but its vetted "
                f"adapter ABI returns {_CABI_RESULTS[name]!r}",
                name_token,
            )
        if name in self.functions or name in self.externs:
            self.error(f"{name!r} is already declared", name_token)
        self.externs[name] = result


def _abi_kind(held: CType, result: bool = False) -> "str | None":
    """How a value of this type travels: a pointer, a word, or a double.

    None where py2bin cannot pass it at all - a struct by value, which no
    part of this compiler does - so the caller leaves the function alone and
    the ordinary diagnostic reports it.
    """

    if isinstance(held, VoidType):
        return "void" if result else None
    if isinstance(held, (PointerType, ArrayType)):
        return "ptr"
    if isinstance(held, FloatingType):
        return "double"
    if isinstance(held, IntegerType):
        return "int"
    return None


def _literal_type(token: Token, filename: str, target: str = "") -> CType:
    """The type C gives an integer constant: C11 6.4.4.1 table.

    The first type in the list that can represent the value wins. A ``u``
    suffix keeps the list unsigned; an ``l``/``ll`` suffix drops the narrower
    entries; and only a hexadecimal or octal constant may pick an unsigned type
    it was not asked for, which is what gives ``0xFFFFFFFFFFFFFFFF`` the value
    18446744073709551615 rather than -1.
    """

    suffix = token.suffix
    value = int(token.value)
    longs = min(suffix.count("l"), 2)
    # `long` is as wide as the target says, so which type a constant lands in
    # is the target's answer too: on Windows a value past 32 bits written
    # with one `l` is a `long long`, because a `long` there cannot hold it.
    long_type = long_for(target)
    ulong_type = ulong_for(target)
    if "u" in suffix:
        candidates = [UINT, ulong_type, ULLONG][longs:]
    elif token.radix == 10:
        candidates = [INT, long_type, LLONG][longs:]
    else:
        candidates = [INT, UINT, long_type, ulong_type, LLONG, ULLONG][longs * 2 :]
    for candidate in candidates:
        bits = candidate.size * 8
        low = 0 if not candidate.signed else -(1 << (bits - 1))
        high = (1 << bits) - 1 if not candidate.signed else (1 << (bits - 1)) - 1
        if low <= value <= high:
            return candidate
    raise CCompileError(
        filename,
        token.line,
        token.column,
        f"the integer constant {value} does not fit any C integer type; a "
        "decimal constant never becomes unsigned on its own, so write the 'u' "
        "suffix if that is what you meant",
    )


def _matches_abi(ctype: CType, kind: str, *, result: bool = False) -> bool:
    if kind == "void":
        return isinstance(ctype, VoidType)
    if kind == "int":
        return isinstance(ctype, IntegerType)
    if kind == "ptr":
        return isinstance(ctype, PointerType)
    if kind == "f64":
        # Specifically a double. A C ``float`` is passed in the same register
        # class but is half the width, so accepting it would hand the callee
        # a value it reads as twice the bits it was given.
        return isinstance(ctype, FloatingType) and ctype.size == 8
    if kind in {"cstr", "cfmt", "cdata"}:
        return not result and isinstance(ctype, PointerType) and ctype.target in {
            CHAR,
            SCHAR,
            UCHAR,
        }
    return False


# --- compile-time constant evaluation ----------------------------------------


class ConstantEvaluator:
    """Evaluates the constant expressions C needs before code generation.

    Array lengths and ``case`` labels must be known at compile time, and they
    are parsed before any lowering context exists, so this is a small standalone
    interpreter over the syntax tree rather than a pass over the IR.
    """

    def __init__(self, filename: str, names: "dict[str, int] | None" = None):
        self.filename = filename
        #: The enumeration constants in scope. C makes each one an integer
        #: constant expression, so it may stand in an array length, a `case`
        #: label, or the value of a later enumerator - which is how every
        #: enum a COM header generates is written: each entry is the one
        #: before it plus one.
        self.names = names if names is not None else {}

    def error(self, message: str, token: Token):
        raise CCompileError(
            token.origin or self.filename, token.line, token.column, message
        )

    def value(self, node: Node) -> int:
        result = self.evaluate(node)
        return result

    def evaluate(self, node: Node) -> int:
        if isinstance(node, IntLiteral):
            return node.value
        if isinstance(node, Identifier):
            if node.name in self.names:
                return self.names[node.name]
            self.error(
                f"{node.name!r} is not a constant; only an enumeration "
                "constant may stand in a constant expression",
                node.token,
            )
        if isinstance(node, FloatLiteral):
            self.error(
                "a floating constant is not an integer constant expression; an "
                "array length and a 'case' label must be integers",
                node.token,
            )
        if isinstance(node, SizeofType):
            size = size_of(node.ctype)
            if size is None:
                self.error(f"sizeof({node.ctype}) needs a complete type", node.token)
            return size
        if isinstance(node, Unary):
            operand = self.evaluate(node.operand)
            if node.operator == "+":
                return operand
            if node.operator == "-":
                return -operand
            if node.operator == "~":
                return ~operand
            if node.operator == "!":
                return int(operand == 0)
            self.error(
                f"unary {node.operator!r} is not allowed in a constant expression",
                node.token,
            )
        if isinstance(node, Cast):
            value = self.evaluate(node.operand)
            if not isinstance(node.ctype, IntegerType):
                self.error(
                    "only integer casts are allowed in a constant expression",
                    node.token,
                )
            return _wrap(value, node.ctype.size, node.ctype.signed)
        if isinstance(node, Conditional):
            return (
                self.evaluate(node.body)
                if self.evaluate(node.test)
                else self.evaluate(node.alternative)
            )
        if isinstance(node, Logical):
            left = self.evaluate(node.left)
            if node.operator == "&&":
                return int(bool(left) and bool(self.evaluate(node.right)))
            return int(bool(left) or bool(self.evaluate(node.right)))
        if isinstance(node, Binary):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            if node.operator in {"/", "%"} and right == 0:
                self.error("division by zero in a constant expression", node.token)
            operations = {
                "+": lambda a, b: a + b,
                "-": lambda a, b: a - b,
                "*": lambda a, b: a * b,
                "/": lambda a, b: abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1),
                "%": lambda a, b: a - (abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)) * b,
                "&": lambda a, b: a & b,
                "|": lambda a, b: a | b,
                "^": lambda a, b: a ^ b,
                "<<": lambda a, b: a << b,
                ">>": lambda a, b: a >> b,
                "<": lambda a, b: int(a < b),
                "<=": lambda a, b: int(a <= b),
                ">": lambda a, b: int(a > b),
                ">=": lambda a, b: int(a >= b),
                "==": lambda a, b: int(a == b),
                "!=": lambda a, b: int(a != b),
            }
            operation = operations.get(node.operator)
            if operation is None:
                self.error(
                    f"{node.operator!r} is not allowed in a constant expression",
                    node.token,
                )
            return operation(left, right)
        self.error(
            "this expression is not a constant expression; an array length and a "
            "'case' label must be known at compile time",
            node.token,
        )


def _wrap(value: int, size: int, signed: bool) -> int:
    modulus = 1 << (size * 8)
    value &= modulus - 1
    if signed and value >= modulus >> 1:
        value -= modulus
    return value


def _s64(value: int) -> int:
    return _wrap(value, 8, True)


def _u64(value: int) -> int:
    return value & 0xFFFFFFFFFFFFFFFF


# --- IR construction helpers -------------------------------------------------


def _fold(operator: str, left: int, right: int) -> int | None:
    """Constant-fold one IR operation with exact 64-bit wrapping semantics."""

    if operator == "add":
        return _s64(left + right)
    if operator == "sub":
        return _s64(left - right)
    if operator == "mul":
        return _s64(left * right)
    if operator == "and":
        return _s64(_u64(left) & _u64(right))
    if operator == "or":
        return _s64(_u64(left) | _u64(right))
    if operator == "xor":
        return _s64(_u64(left) ^ _u64(right))
    if operator == "lshift":
        return _s64(_u64(left) << (right & 63))
    if operator == "rshift":
        return _s64(left >> (right & 63))
    if operator == "urshift":
        return _s64(_u64(left) >> (right & 63))
    if right == 0:
        return None  # division by zero is undefined; let it happen at runtime
    if operator == "sdiv":
        quotient = abs(left) // abs(right)
        return _s64(quotient if (left < 0) == (right < 0) else -quotient)
    if operator == "smod":
        remainder = abs(left) % abs(right)
        return _s64(remainder if left >= 0 else -remainder)
    if operator == "udiv":
        return _s64(_u64(left) // _u64(right))
    if operator == "umod":
        return _s64(_u64(left) % _u64(right))
    return None


_COMPARISONS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "ult": lambda a, b: _u64(a) < _u64(b),
    "ule": lambda a, b: _u64(a) <= _u64(b),
    "ugt": lambda a, b: _u64(a) > _u64(b),
    "uge": lambda a, b: _u64(a) >= _u64(b),
}


def _constant(value: int) -> IntConstant:
    """Build an ``IntConstant`` in the IR's one canonical form.

    Every integer in the IR is a 64-bit bit pattern read as SIGNED, because
    that is what the machine registers hold. A constant that arrives as an
    unsigned quantity -- ``0xFFFFFFFFFFFFFFFFull``, or a ``case`` label of an
    unsigned type -- has to be converted here. The encoders mask to 64 bits
    anyway, so skipping this looks harmless; it is not, because the constant
    folder then compares and divides Python integers of arbitrary width, and
    ``(long)0xFFFFFFFFFFFFFFFE >= 79490271399379139`` folds to true when the
    same expression computed at runtime is false.
    """

    return IntConstant(_s64(value))


def _value_of(expression: IntExpression) -> int | None:
    return expression.value if isinstance(expression, IntConstant) else None


def _binary(operator: str, left: IntExpression, right: IntExpression) -> IntExpression:
    a, b = _value_of(left), _value_of(right)
    if a is not None and b is not None:
        folded = _fold(operator, a, b)
        if folded is not None:
            return IntConstant(folded)
    if b == 0 and operator in {"add", "sub", "or", "xor", "lshift", "rshift", "urshift"}:
        return left
    if a == 0 and operator == "add":
        return right
    if b == 1 and operator in {"mul", "sdiv", "udiv"}:
        return left
    if a == 1 and operator == "mul":
        return right
    if (a == 0 or b == 0) and operator == "mul":
        return IntConstant(0)
    return IntBinary(operator, left, right)


def _compare(operator: str, left: IntExpression, right: IntExpression) -> IntExpression:
    a, b = _value_of(left), _value_of(right)
    if a is not None and b is not None:
        return IntConstant(int(_COMPARISONS[operator](a, b)))
    return IntCompare(operator, left, right)


def _is_link_constant(expression: object) -> bool:
    """Whether ``expression`` is a value C may initialize static storage with.

    C11 6.7.9p4: the initializer for an object with static storage duration is
    an arithmetic constant expression, or an address constant -- the address of
    an object or function, optionally offset by a constant. Everything here is
    known before the first instruction runs, which is exactly why start-up can
    place it without evaluating anything.
    """

    if isinstance(
        expression,
        (IntConstant, FloatConstant, CStringConstant, GlobalAddress, FunctionAddress),
    ):
        return True
    if isinstance(expression, IntBinary) and expression.operator in {"add", "sub"}:
        return _is_link_constant(expression.left) and _is_link_constant(
            expression.right
        )
    if isinstance(expression, (FloatBits, BitsFloat)):
        return _is_link_constant(expression.value)
    return False


def _contains_call(value: object) -> bool:
    """True when an IR expression embeds a call, so re-emitting it would repeat it.

    ``Lowerer.extern_call`` and ``Lowerer.direct_call`` pin every call in a slot
    as soon as it is lowered, so nothing this compiler builds should ever trip
    this check. It stays as the guard on the two places that reuse an expression
    -- the address of a read-modify-write target -- because the failure it
    prevents (a call happening twice because its value appeared twice in a tree)
    is the defect this backend has produced most often.
    """

    if isinstance(value, (ExternCall, IRCall, IndirectCall)):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_call(item) for item in value)
    for name in getattr(type(value), "__slots__", ()) or ():
        if _contains_call(getattr(value, name)):
            return True
    return False


@dataclasses.dataclass(frozen=True, slots=True)
class Value:
    """A lowered C expression: its type and its canonical IR value.

    Every integer value is kept sign-extended (signed types) or zero-extended
    (unsigned types) into 64 bits, so one representation serves both the
    register file and memory of any width. ``null`` marks the null pointer
    constant, the one integer C lets stand in for a pointer.

    A value whose type is floating carries a ``FloatExpression`` instead, always
    in binary64 -- including one of type ``float``, whose extra precision C11
    6.3.1.8p2 lets an implementation keep until an assignment, cast, argument
    or return removes it. ``ctype`` is what says which of the two ``expr`` is,
    so nothing has to guess.
    """

    ctype: CType
    expr: IntExpression | FloatExpression
    null: bool = False


@dataclasses.dataclass(slots=True)
class Local:
    """A named object and where it lives.

    ``slot`` is a stack-slot index for an automatic object, and a byte offset
    into the module's static storage block when ``static`` is set. The two are
    deliberately different address spaces: a frame dies when its function
    returns, and the static block does not.
    """

    ctype: CType
    slot: int
    static: bool = False


@dataclasses.dataclass(slots=True)
class FunctionContext:
    function: Function
    result_slot: int | None
    return_label: str
    is_main: bool
    #: True while lowering a body that will become a real IR ``Function``, so a
    #: ``return`` becomes a machine return instead of a store-and-jump.
    call_body: bool = False
    labels: dict[str, str] = dataclasses.field(default_factory=dict)
    defined: set[str] = dataclasses.field(default_factory=set)
    pending: list[tuple[str, Token]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class SwitchContext:
    ctype: IntegerType
    cases: list[tuple[int, str]]
    default: str | None
    seen: set[int]


# Every printf conversion py2bin implements, and the C type its argument is
# converted to before it is formatted.
_CONVERSIONS = {
    "d": ("signed", INT),
    "i": ("signed", INT),
    "u": ("unsigned", UINT),
    "x": ("hex", UINT),
    "X": ("HEX", UINT),
    "c": ("char", INT),
    "s": ("string", PointerType(CHAR)),
}
#: printf's floating conversions. py2bin emits the whole formatter itself, so
#: each one is a compile-time choice of shape plus a runtime precision.
_FLOAT_CONVERSIONS = frozenset({"f", "F", "e", "E", "g", "G"})

#: A conversion's precision bounds the output, and the output goes into a fixed
#: frame buffer. 120 leaves room for the 309 integer digits the largest double
#: needs in %f, with the sign and the point, inside _TEXT_BYTES.
_MAXIMUM_PRECISION = 120

#: `%*d` - the width is the argument before the value. Stands for a width in
#: the parsed format where a number would be, so every pass below can tell the
#: two apart by asking rather than by guessing at a sentinel number.
def _negated(width: object) -> IntExpression:
    """The other sign of a width handed over at run time.

    C7.21.6.1p5: a negative width argument means the `-` flag and that width
    without its sign. Both pads are written for a width nobody wrote down,
    and each loop stops at once when its count is not positive - so the sign
    of the number chooses between them without a branch of its own.
    """

    return _binary("sub", IntConstant(0), _width_of(width))


def _width_of(width: object) -> IntExpression:
    """A field width, whether it was written down or handed over."""

    return IntConstant(width) if isinstance(width, int) else width


class _FromAnArgument:
    def __repr__(self) -> str:  # pragma: no cover - a marker, shown in traces
        return "<width from an argument>"


_FROM_AN_ARGUMENT = _FromAnArgument()

#: The widest field printf will pad to. The formatter builds its answer in a
#: fixed frame buffer, and a width past that would write off the end of it.
_MAXIMUM_FIELD = 120

#: The two that format into a buffer rather than onto stdout. Compiled like
#: `printf` is - the format is read here, and the code that writes it out is
#: emitted - so there is no C library underneath these either.
_INTO_A_BUFFER = frozenset(
    {"sprintf", "snprintf", "swprintf", "swprintf_s", "_snwprintf"}
)

#: The ones that write wide characters. Same formatting, wider stores: what
#: a program reaches for when the thing it is talking to takes UTF-16.
_WIDE_BUFFER = frozenset({"swprintf", "swprintf_s", "_snwprintf"})

#: The ones told how much room they have. `sprintf` is not; the rest are,
#: except where C++'s array overload took the room from the array instead.
_BOUNDED_BUFFER = frozenset({"snprintf", "swprintf", "swprintf_s", "_snwprintf"})

#: What `<stdarg.h>` gives a program. Each is compiled rather than called:
#: a `va_list` is a pointer into the cells the call wrote, `va_arg` reads one
#: and steps past it, and `va_end` has nothing to undo.
_VARIADIC_BUILTINS = frozenset({"va_start", "va_arg", "va_end", "va_copy"})

#: The hidden parameter a variadic function is given: where its extra
#: arguments were written. Named so no program can collide with it.
_VARIADIC_PARAMETER = "__py2bin_va_area"

#: Bytes of frame the floating formatter needs. The digit array holds the EXACT
#: decimal expansion of a double: 767 digits for the smallest subnormal (its
#: mantissa times 5**1074), plus one the final rounding carry can add.
_DIGIT_BYTES = 1024
_TEXT_BYTES = 512

#: Stands in for `wchar_t *` and `wchar_t` in the table below, because how
#: wide one is depends on the target and the table is built once.
_WIDE_STRING = PointerType(VoidType())
_WIDE_CHAR = IntegerType("__py2bin_wchar", 0, True, 0)

_LENGTHS = {
    "": {},
    "hh": {"d": SCHAR, "i": SCHAR, "u": UCHAR, "x": UCHAR, "X": UCHAR},
    "h": {"d": SHORT, "i": SHORT, "u": USHORT, "x": USHORT, "X": USHORT},
    # `%ls` and `%lc` name a wide string and a wide character. The type they
    # want depends on the platform, so it is filled in where the target is
    # known rather than named here.
    "l": {
        "d": LONG, "i": LONG, "u": ULONG, "x": ULONG, "X": ULONG,
        "s": _WIDE_STRING, "c": _WIDE_CHAR,
    },
    "ll": {"d": LLONG, "i": LLONG, "u": ULLONG, "x": ULLONG, "X": ULLONG},
    "z": {"d": LONG, "i": LONG, "u": ULONG, "x": ULONG, "X": ULONG},
    "j": {"d": LLONG, "i": LLONG, "u": ULLONG, "x": ULLONG, "X": ULLONG},
}

#: How many stack slots one translation unit's frame may take. This is the
#: same budget the IR enforces rather than a second, tighter one: two numbers
#: for the same limit meant a program could be rejected here at 32 KB while
#: the backend was prepared to give it 512 KB, and the message named a
#: restriction that was not the real one.
_MAXIMUM_SLOTS = MAXIMUM_STACK_SLOTS

#: Bytes of static storage one translation unit may declare. The block is a
#: single mapping obtained at start-up, so the limit is a sanity bound rather
#: than an architectural one; it is stated because exceeding it must be a
#: compile-time rejection and never a mapping that silently fails at run time.
_MAXIMUM_STATIC_BYTES = 8 << 20

#: py2bin's call ABI passes every argument in a register. AAPCS64 has eight
#: integer parameter registers, and stack argument passing is not implemented,
#: so a longer parameter list is rejected rather than silently truncated.
#: Arguments a target can pass. ARM64 implements the AAPCS64 memory
#: argument area; the x86 encoders stop at their register count.
_MAXIMUM_ARGUMENTS = 8
#: Every target now implements its convention's memory argument area.
_STACK_ARGUMENT_TARGETS = frozenset(
    {
        "darwin-arm64",
        "linux-arm64",
        "windows-arm64",
        "darwin-x86_64",
        "linux-x86_64",
        "windows-x86_64",
    }
)
_ARGUMENT_CEILING = 64


# The C math functions the target implements as a single instruction. No
# library is linked and no libm is bundled: py2bin emits the hardware
# operation, which is why these work with no linker at all. Anything needing a
# software implementation (sin, cos, exp, log, pow) is deliberately absent.
_MATH_BUILTINS = {
    "sqrt": "sqrt",
    "fabs": "abs",
    "floor": "floor",
    "ceil": "ceil",
    "trunc": "trunc",
    "round": "round",
}


class Lowerer:
    """Lowers a parsed translation unit to py2bin's native IR."""

    def __init__(self, unit: TranslationUnit, filename: str, target: str):
        self.enumerators = dict(getattr(unit, "enumerators", {}) or {})
        self.unit = unit
        self.filename = filename
        self.target = target
        self.operations: list[Operation] = []
        self.stack_slots = 0
        #: The largest the frame ever was, which is what it must be built to.
        #: `stack_slots` falls back at every statement boundary now, so it is
        #: no longer the answer.
        self.peak_slots = 0
        #: One past the highest slot holding something that outlives its
        #: statement. Reclamation never goes below this.
        self.reserved_slots = 0
        self.scopes: list[dict[str, Local]] = []
        self.counter = 0
        self.break_targets: list[str] = []
        self.continue_targets: list[str] = []
        self.switches: list[SwitchContext] = []
        self.functions: list[FunctionContext] = []
        self.active: list[str] = []
        self.buffer_slot: int | None = None
        #: Where formatted output goes when it is not going to stdout:
        #: (buffer slot, limit slot, count slot) for `snprintf`.
        self.sink: "tuple[int, int, int] | None" = None
        #: Bytes per character the sink stores. One for a `char` buffer, two
        #: or four for a `wchar_t` one - the platform decides which.
        self.sink_width = 1
        self.digit_slot: int | None = None
        self.text_slot: int | None = None
        self.float_scratch: dict[str, int] = {}
        # The shared floating formatter of the body being lowered: its entry
        # label, the label its dispatch chain lives at, and every site that
        # jumped into it and must be returned to.
        self.float_entry: str | None = None
        self.float_dispatch: str | None = None
        self.float_returns: list[tuple[int, str]] = []
        # Real calls: every function reached from main, lowered once into its
        # own IR body, plus the set currently being lowered so a call that
        # arrives while its own body is still open (that is, recursion) emits a
        # call rather than trying to lower the body a second time.
        self.calls_are_real = target in CALL_CAPABLE_TARGETS
        self.lowered: dict[str, IRFunction] = {}
        self.lowering: set[str] = set()
        # File-scope objects. They live in the module's static storage block
        # rather than any frame, which is what lets one object be the same
        # object in the entry point and in every function body.
        self.statics: dict[str, Local] = {}
        #: `static int n;` inside a block, by the declaration that wrote it.
        #: Keyed that way so a body inlined into several call sites still
        #: names one object, which is what C says a static local is.
        self.stored_locals: "dict[tuple[int, str], Local]" = {}
        #: The bitfield the last `lvalue` answered about, if it was one. A
        #: bitfield has no address, so the caller gets its unit's and this.
        self.packed: "Member | None" = None
        self.static_bytes = 0

    # --- bookkeeping ---

    def error(self, message: str, token: Token):
        raise CCompileError(
            token.origin or self.filename, token.line, token.column, message
        )

    def emit(self, operation: Operation) -> None:
        self.operations.append(operation)

    def take(self, size: int) -> int:
        """Slots for something whose lifetime ends with the statement.

        The frame is one fixed allocation, so what matters for its size is the
        high-water mark rather than what is outstanding now - :attr:`peak_slots`
        carries that, and is what the emitted frame is built from. Reading the
        live count instead would hand the function a frame smaller than the
        offsets written into its own code.
        """

        slots = max(1, (size + 7) // 8)
        base = self.stack_slots
        self.stack_slots += slots
        if self.stack_slots > _MAXIMUM_SLOTS:
            raise CCompileError(
                self.filename,
                1,
                1,
                f"this function needs more than {_MAXIMUM_SLOTS * 8} bytes of "
                "stack frame; py2bin's native frames are a single fixed "
                "allocation, so split it into smaller functions",
            )
        self.peak_slots = max(self.peak_slots, self.stack_slots)
        return base

    def allocate(self, size: int) -> int:
        """Slots for something that outlives the statement that made it.

        A local, or one of the function-wide scratch areas. Reclaiming these
        at a statement boundary would hand the next statement a slot something
        still holds - which is why the two allocators are separate rather than
        one with a flag nobody remembers to pass.
        """

        base = self.take(size)
        self.reserved_slots = self.stack_slots
        return base

    def new_temp(self) -> int:
        return self.take(8)

    def release_temporaries(self, mark: int) -> None:
        """Give back every temporary slot taken since `mark`.

        Never below :attr:`reserved_slots`: a statement that declared a local
        or first touched the float formatter's scratch raised that floor, and
        those slots are still live even though the statement has finished.
        """

        self.stack_slots = max(mark, self.reserved_slots)

    def new_label(self, prefix: str) -> str:
        self.counter += 1
        return f"c_{prefix}_{self.counter}"

    def declare(self, name: str, ctype: CType, token: Token) -> Local:
        scope = self.scopes[-1]
        if name in scope:
            self.error(f"{name!r} is already declared in this scope", token)
        if isinstance(ctype, FunctionType):
            self.error(
                f"{name!r} is declared as a function inside a block; py2bin "
                "accepts a function declaration only at file scope. Write "
                f"'{ctype.result} (*{name})(...)' for a pointer to a function",
                token,
            )
        size = size_of(ctype)
        if size is None:
            self.error(
                f"cannot declare {name!r} with the incomplete type {ctype}", token
            )
        local = Local(ctype, self.allocate(size))
        scope[name] = local
        return local

    def lookup(self, name: str) -> Local | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        # File scope is the outermost scope, so a local of the same name wins.
        return self.statics.get(name)

    def allocate_static(self, ctype: CType, token: Token) -> int:
        """Reserve aligned space for one file-scope object and return its offset."""

        size = size_of(ctype)
        if size is None:
            self.error(
                f"a file-scope object cannot have the incomplete type {ctype}", token
            )
        alignment = align_of(ctype)
        offset = (self.static_bytes + alignment - 1) & ~(alignment - 1)
        self.static_bytes = offset + size
        if self.static_bytes > _MAXIMUM_STATIC_BYTES:
            self.error(
                f"this translation unit declares more than "
                f"{_MAXIMUM_STATIC_BYTES} bytes of file-scope objects, which is "
                "more static storage than py2bin reserves",
                token,
            )
        return offset

    def address_of(self, local: Local) -> IntExpression:
        """Where an object lives: a frame slot, or the static storage block."""

        if local.static:
            return GlobalAddress(local.slot)
        return SlotAddress(local.slot)

    # --- conversions ---

    def fit(self, expression: IntExpression, ctype: CType) -> IntExpression:
        """Reduce ``expression`` to the canonical 64-bit form of ``ctype``."""

        if ctype == BOOL:
            return _compare("ne", expression, IntConstant(0))
        size = size_of(ctype)
        constant = _value_of(expression)
        if size is None or size == 8:
            # Widening to 64 bits is a no-op on the value, but a constant that
            # came in as an unsigned quantity still has to be renormalized so
            # the folder keeps seeing the signed reading of the bit pattern.
            return expression if constant is None else _constant(constant)
        signed = is_signed(ctype)
        if constant is not None:
            return _constant(_wrap(constant, size, signed))
        bits = size * 8
        if signed:
            shift = IntConstant(64 - bits)
            return _binary("rshift", _binary("lshift", expression, shift), shift)
        return _binary("and", expression, IntConstant((1 << bits) - 1))

    # --- floating conversions ---
    #
    # Every floating value in flight is a binary64 double, whatever its C type.
    # These four helpers are the only places that cross between the integer and
    # floating worlds, so the rules live in one place instead of at each use.

    def widen(self, value: Value) -> FloatExpression:
        """The double a floating or integer arithmetic value stands for."""

        if is_floating(value.ctype):
            return value.expr
        assert isinstance(value.ctype, IntegerType)
        # A canonical unsigned value narrower than 64 bits is already a
        # non-negative i64, so only the 64-bit unsigned types need the unsigned
        # conversion -- but they really need it, or 2**64-1 converts to -1.0.
        signed = value.ctype.signed or value.ctype.size < 8
        return IntToFloat(self.fit(value.expr, value.ctype), signed=signed)

    def narrow(self, expression: FloatExpression, target: FloatingType) -> FloatExpression:
        """Remove the extra precision C requires a conversion to remove."""

        if target.size == 4:
            return BitsFloat(FloatBits(expression, 4), 4)
        return expression

    def to_integer(self, expression: FloatExpression, target: IntegerType) -> IntExpression:
        """C's conversion of a floating value to an integer type: truncate."""

        if target == BOOL:
            return FloatCompare("ne", expression, FloatConstant(0.0))
        # A destination whose range runs past 2**63-1 needs the unsigned
        # instruction; everything narrower fits in a signed 64-bit result and is
        # then reduced by fit() exactly as an integer conversion is.
        signed = target.signed or target.size < 8
        return self.fit(FloatToInt(expression, signed=signed), target)

    def stored_bits(self, expression: object, ctype: CType) -> IntExpression:
        """The integer image an object of ``ctype`` holds in memory.

        A C floating object is its IEEE-754 bit pattern, four bytes wide for a
        ``float`` and eight for a ``double``, so every store goes through the
        same ``HeapStore`` every other C object uses.
        """

        if is_floating(ctype):
            return FloatBits(expression, ctype.size)
        return expression

    def from_bits(self, expression: IntExpression, ctype: CType) -> object:
        """Read back what :meth:`stored_bits` wrote."""

        if is_floating(ctype):
            return BitsFloat(expression, ctype.size)
        return self.fit(expression, ctype)

    def truth(self, value: Value) -> IntExpression:
        """The 0/1 an ``if``, ``while`` or ``&&`` tests a scalar with."""

        if is_floating(value.ctype):
            return FloatCompare("ne", value.expr, FloatConstant(0.0))
        return value.expr

    def assign_convert(
        self, value: Value, target: CType, token: Token, what: str
    ) -> IntExpression | FloatExpression:
        if isinstance(target, FloatingType):
            if is_arithmetic(value.ctype):
                return self.narrow(self.widen(value), target)
            if isinstance(value.ctype, PointerType):
                self.error(
                    f"{what} needs {target}, but this is a pointer; C has no "
                    "conversion between a pointer and a floating type at all",
                    token,
                )
            self.error(f"{what} needs {target}, but this is {value.ctype}", token)
        if isinstance(target, IntegerType):
            if isinstance(value.ctype, FloatingType):
                return self.to_integer(value.expr, target)
            if isinstance(value.ctype, IntegerType):
                return self.fit(value.expr, target)
            if isinstance(value.ctype, PointerType):
                self.error(
                    f"{what} needs {target}, but this is a pointer; C requires an "
                    "explicit cast to convert one to an integer",
                    token,
                )
            self.error(f"{what} needs {target}, but this expression has no value", token)
        if isinstance(target, PointerType):
            if value.null:
                return IntConstant(0)
            if isinstance(value.ctype, PointerType) and compatible(target, value.ctype):
                return value.expr
            if isinstance(value.ctype, PointerType):
                self.error(
                    f"{what} needs {target}, but this is {value.ctype}; C requires "
                    "an explicit cast between incompatible pointer types",
                    token,
                )
            self.error(f"{what} needs {target}, but this is {value.ctype}", token)
        self.error(f"{what} cannot be given a value of type {value.ctype}", token)

    # --- lvalues and loads ---

    def load(self, ctype: CType, address: IntExpression) -> Value:
        if isinstance(ctype, ArrayType):
            # An array used as a value decays to a pointer to its first element.
            return Value(PointerType(ctype.element), address)
        if isinstance(ctype, FunctionType):
            # A function designator used as a value decays to a pointer to the
            # function (C11 6.3.2.1p4); its "address" already IS that pointer.
            return Value(PointerType(ctype), address)
        if isinstance(ctype, StructType):
            # A struct value is carried as the address of the object. Only
            # assignment, member access and & consume one, and each of those
            # wants the address rather than a word-sized load.
            return Value(ctype, address)
        size = size_of(ctype)
        if size is None:
            raise AssertionError("incomplete lvalue reached the loader")
        if isinstance(ctype, FloatingType):
            # The object's bytes ARE its IEEE bit pattern, so an ordinary
            # integer load reaches it and BitsFloat reinterprets the result.
            return Value(ctype, BitsFloat(HeapLoad(address, size, False), size))
        return Value(ctype, HeapLoad(address, size, is_signed(ctype)))

    def lvalue(self, node: Node) -> tuple[CType, IntExpression]:
        if isinstance(node, Identifier):
            local = self.lookup(node.name)
            if local is None:
                if node.name == "NULL":
                    self.error("the null pointer constant is not an lvalue", node.token)
                if node.name in self.unit.functions:
                    # A function designator. It is not an lvalue in C either,
                    # but '&' and a call both want exactly this pair, and every
                    # other use goes through load(), which decays it.
                    return self.function_designator(node.name, node.token)
                self.error(
                    f"{node.name!r} is not a declared local or parameter", node.token
                )
            return local.ctype, self.address_of(local)
        if isinstance(node, Unary) and node.operator == "*":
            pointer = self.rvalue(node.operand)
            if not isinstance(pointer.ctype, PointerType):
                self.error(
                    f"cannot dereference a value of type {pointer.ctype}", node.token
                )
            target = pointer.ctype.target
            if isinstance(target, FunctionType):
                # *fp is the function itself, which decays straight back to the
                # pointer -- which is why (*fp)(x) and fp(x) are the same call,
                # and why **fp is still legal C.
                return target, pointer.expr
            if size_of(target) is None:
                self.error(
                    f"cannot dereference {pointer.ctype}: {target} is an incomplete "
                    "type whose layout py2bin deliberately does not model",
                    node.token,
                )
            return target, pointer.expr
        if isinstance(node, Index):
            return self.lvalue(
                Unary(
                    node.token,
                    "*",
                    Binary(node.token, "+", node.base, node.offset),
                )
            )
        if isinstance(node, MemberAccess):
            # `a.m` needs the address of a, and `p->m` the value of p. Both
            # then add the member's constant offset.
            if node.through_pointer:
                pointer = self.rvalue(node.base)
                if not isinstance(pointer.ctype, PointerType):
                    self.error(
                        f"'->' needs a pointer to a struct or union, not "
                        f"{pointer.ctype}",
                        node.token,
                    )
                owner = pointer.ctype.target
                address = pointer.expr
            else:
                owner, address = self.lvalue(node.base)
            if not isinstance(owner, StructType):
                self.error(
                    f"{'->' if node.through_pointer else '.'} needs a struct or "
                    f"union, not {owner}",
                    node.token,
                )
            if owner.members is None:
                self.error(
                    f"{owner} is incomplete here, so its members are unknown",
                    node.token,
                )
            member = owner.member(node.name)
            if member is None:
                self.error(
                    f"{owner} has no member named {node.name!r}", node.token
                )
            # A bitfield is some of the bits of the unit it shares, so what
            # is answered here is that unit's address and the member is left
            # where the caller can find it. Everything that reads or writes
            # one looks; `&` refuses.
            self.packed = member if member.width is not None else None
            if member.offset == 0:
                return member.ctype, address
            return member.ctype, IntBinary(
                "add", address, IntConstant(member.offset)
            )
        self.error("this expression is not an lvalue", node.token)

    def stabilize(self, expression: IntExpression) -> IntExpression:
        """Pin a value in a slot when re-emitting it would repeat a call."""

        if isinstance(expression, (IntConstant, IntLoad, SlotAddress, GlobalAddress)):
            return expression
        if not _contains_call(expression):
            return expression
        slot = self.new_temp()
        self.emit(Store(slot, expression))
        return IntLoad(slot)

    def materialize(self, expression: IntExpression) -> IntExpression:
        """Pin a value in a slot so later stores cannot change what it reads."""

        if isinstance(expression, (IntConstant, IntLoad)):
            return expression
        slot = self.new_temp()
        self.emit(Store(slot, expression))
        return IntLoad(slot)

    def materialize_float(self, expression: FloatExpression) -> FloatExpression:
        """The floating counterpart of :meth:`materialize`.

        The slot holds the full binary64 value, not the object's storage
        format, so pinning a ``float`` expression here does not round it -- C
        rounds at the assignment itself, which happens after this.
        """

        if isinstance(expression, (FloatConstant, FloatLoad)):
            return expression
        slot = self.new_temp()
        self.emit(FloatStore(slot, expression))
        return FloatLoad(slot)

    def pin(self, value: Value) -> Value:
        """Pin either flavour of value in a slot, keeping its C type."""

        if is_floating(value.ctype):
            return Value(value.ctype, self.materialize_float(value.expr))
        return Value(value.ctype, self.materialize(value.expr), value.null)

    # --- expressions ---

    def rvalue(self, node: Node) -> Value:
        if isinstance(node, IntLiteral):
            return Value(
                node.ctype,
                _constant(node.value),
                null=node.value == 0 and node.ctype in {INT, LONG, LLONG},
            )
        if isinstance(node, FloatLiteral):
            return Value(node.ctype, FloatConstant(node.value))
        if isinstance(node, StringLiteral):
            if self.target not in _STRING_VALUE_TARGETS:
                self.error(
                    "using a string literal as a pointer value needs the "
                    "constant bytes an image writer places after the "
                    f"code; it is not implemented for {self.target!r} "
                    "(printf of a literal is)",
                    node.token,
                )
            return Value(
                PointerType(node.element_for(self.target)),
                CStringConstant(node.bytes_for(self.target)),
            )
        if isinstance(node, Identifier):
            local = self.lookup(node.name)
            if local is None:
                if node.name == "NULL":
                    return Value(PointerType(VOID), IntConstant(0), null=True)
                if node.name in self.enumerators:
                    return Value(INT, IntConstant(self.enumerators[node.name]))
                if node.name in self.unit.functions:
                    ctype, address = self.function_designator(node.name, node.token)
                    return self.load(ctype, address)
                self.error(
                    f"{node.name!r} is not a declared local or parameter", node.token
                )
            return self.load(local.ctype, self.address_of(local))
        if isinstance(node, (Index, MemberAccess)):
            self.packed = None
            ctype, address = self.lvalue(node)
            if self.packed is not None:
                return self.load_bitfield(self.packed, address)
            return self.load(ctype, address)
        if isinstance(node, Unary):
            return self.unary(node)
        if isinstance(node, IncDec):
            return self.increment(node)
        if isinstance(node, Binary):
            return self.binary(node)
        if isinstance(node, Logical):
            return self.logical(node)
        if isinstance(node, Conditional):
            return self.conditional(node)
        if isinstance(node, Assignment):
            return self.assignment(node)
        if isinstance(node, Comma):
            self.discard(node.left)
            return self.rvalue(node.right)
        if isinstance(node, Cast):
            return self.cast(node)
        if isinstance(node, SizeofType):
            size = size_of(node.ctype)
            if size is None:
                self.error(
                    f"sizeof({node.ctype}) needs a complete type", node.token
                )
            return Value(ULONG, IntConstant(size))
        if isinstance(node, SizeofExpression):
            return Value(ULONG, IntConstant(self.sizeof_expression(node.operand)))
        if isinstance(node, Call):
            return self.call(node)
        if isinstance(node, CallThrough):
            return self.call_through(node.target, node.arguments, node.token)
        self.error("unsupported expression", node.token)

    def discard(self, node: Node) -> None:
        """Evaluate an expression for its side effects only."""

        self.rvalue(node)

    def scalar(self, node: Node, what: str) -> Value:
        value = self.rvalue(node)
        if not is_scalar(value.ctype):
            self.error(f"{what} needs an integer or pointer value", node.token)
        return value

    def sizeof_expression(self, node: Node) -> int:
        """``sizeof e`` does not evaluate ``e``; it needs only its type."""

        if isinstance(node, StringLiteral):
            # sizeof a literal is its whole array, terminator included - which
            # for a wide one is code units, not characters.
            return len(node.bytes_for(self.target))
        if isinstance(node, Identifier):
            local = self.lookup(node.name)
            if local is not None:
                size = size_of(local.ctype)
                if size is None:
                    self.error(
                        f"sizeof({node.name}) needs a complete type", node.token
                    )
                return size
            if node.name in self.unit.functions:
                # C11 6.5.3.4p1: sizeof may not be applied to a function type.
                # Without this the designator would decay and quietly answer 8.
                self.error(
                    f"sizeof({node.name}) applies sizeof to a function, which C "
                    f"does not define; write sizeof(&{node.name}) for the size "
                    "of a pointer to it",
                    node.token,
                )
        saved_operations = self.operations
        saved_slots = self.stack_slots
        self.operations = []
        try:
            if isinstance(node, (Index, MemberAccess)) or (
                isinstance(node, Unary) and node.operator == "*"
            ):
                ctype, _address = self.lvalue(node)
            else:
                ctype = self.rvalue(node).ctype
        finally:
            self.operations = saved_operations
            self.stack_slots = saved_slots
        size = size_of(ctype)
        if size is None:
            self.error(f"sizeof needs a complete type, not {ctype}", node.token)
        return size

    def unary(self, node: Unary) -> Value:
        if node.operator == "&":
            self.packed = None
            ctype, address = self.lvalue(node.operand)
            if self.packed is not None:
                self.error(
                    f"{self.packed.name!r} is a bitfield, which has no address "
                    f"of its own: it is some of the bits of the unit it shares. "
                    f"Copy it into an object of its own first",
                    node.token,
                )
            return Value(PointerType(ctype), address)
        if node.operator == "*":
            ctype, address = self.lvalue(node)
            return self.load(ctype, address)
        if node.operator == "!":
            value = self.scalar(node.operand, "the operand of '!'")
            if is_floating(value.ctype):
                return Value(INT, FloatCompare("eq", value.expr, FloatConstant(0.0)))
            return Value(INT, _compare("eq", value.expr, IntConstant(0)))
        value = self.rvalue(node.operand)
        if not is_arithmetic(value.ctype):
            self.error(
                f"unary {node.operator!r} needs an arithmetic operand, not "
                f"{value.ctype}",
                node.token,
            )
        if is_floating(value.ctype):
            if node.operator == "~":
                self.error(
                    f"unary '~' needs an integer operand, not {value.ctype}",
                    node.token,
                )
            if node.operator == "+":
                return value
            return Value(value.ctype, FloatUnary("neg", value.expr))
        ctype = promote(value.ctype)
        expression = self.fit(value.expr, ctype)
        if node.operator == "+":
            return Value(ctype, expression)
        if node.operator == "-":
            if isinstance(expression, IntConstant):
                return Value(ctype, self.fit(IntConstant(_s64(-expression.value)), ctype))
            return Value(ctype, self.fit(IntUnary("neg", expression), ctype))
        if isinstance(expression, IntConstant):
            return Value(ctype, self.fit(IntConstant(_s64(~expression.value)), ctype))
        return Value(ctype, self.fit(IntUnary("invert", expression), ctype))

    def increment(self, node: IncDec) -> Value:
        ctype, address = self.lvalue(node.operand)
        if isinstance(ctype, ArrayType):
            self.error("an array cannot be incremented", node.token)
        address = self.stabilize(address)
        step = 1
        if isinstance(ctype, FloatingType):
            # C's ++ adds 1 to a floating object too, and the result is rounded
            # back into the object exactly as an assignment would round it.
            old_slot = self.new_temp()
            self.emit(FloatStore(old_slot, self.load(ctype, address).expr))
            operator = "add" if node.operator == "++" else "sub"
            updated = self.narrow(
                FloatBinary(operator, FloatLoad(old_slot), FloatConstant(1.0)), ctype
            )
            new_slot = self.new_temp()
            self.emit(FloatStore(new_slot, updated))
            self.emit(
                HeapStore(
                    address,
                    FloatBits(FloatLoad(new_slot), ctype.size),
                    ctype.size,
                )
            )
            return Value(ctype, FloatLoad(old_slot if not node.prefix else new_slot))
        if isinstance(ctype, PointerType):
            element = size_of(ctype.target)
            if element is None:
                self.error(
                    f"cannot step a {ctype}: {ctype.target} is an incomplete type",
                    node.token,
                )
            step = element
        elif not is_arithmetic(ctype):
            self.error(f"{ctype} cannot be incremented", node.token)
        old_slot = self.new_temp()
        self.emit(Store(old_slot, self.load(ctype, address).expr))
        operator = "add" if node.operator == "++" else "sub"
        updated = self.fit(
            _binary(operator, IntLoad(old_slot), IntConstant(step)), ctype
        )
        new_slot = self.new_temp()
        self.emit(Store(new_slot, updated))
        self.emit(HeapStore(address, IntLoad(new_slot), size_of(ctype)))
        return Value(ctype, IntLoad(old_slot if not node.prefix else new_slot))

    def binary(self, node: Binary) -> Value:
        if node.operator in {"==", "!=", "<", "<=", ">", ">="}:
            return self.comparison(node)
        left = self.rvalue(node.left)
        right = self.rvalue(node.right)
        return self.apply(node.operator, left, right, node.token)

    def apply(self, operator: str, left: Value, right: Value, token: Token) -> Value:
        if operator in {"+", "-"} and (
            isinstance(left.ctype, PointerType) or isinstance(right.ctype, PointerType)
        ):
            return self.pointer_arithmetic(operator, left, right, token)
        if not is_arithmetic(left.ctype) or not is_arithmetic(right.ctype):
            self.error(
                f"{operator!r} needs arithmetic operands, not {left.ctype} and "
                f"{right.ctype}",
                token,
            )
        if is_floating(left.ctype) or is_floating(right.ctype):
            names = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
            if operator not in names:
                self.error(
                    f"{operator!r} needs integer operands; C does not define it for "
                    f"{left.ctype} and {right.ctype}",
                    token,
                )
            ctype = arithmetic_conversions(left.ctype, right.ctype)
            # No rounding here: C11 6.3.1.8p2 lets an implementation keep the
            # extra range and precision of a wider evaluation format, which is
            # what py2bin does (FLT_EVAL_METHOD == 1). The rounding happens
            # where C requires it -- assignment, cast, argument and return.
            return Value(
                ctype, FloatBinary(names[operator], self.widen(left), self.widen(right))
            )
        if operator in {"<<", ">>"}:
            ctype = promote(left.ctype)
            value = self.fit(left.expr, ctype)
            count = self.fit(right.expr, promote(right.ctype))
            if operator == "<<":
                name = "lshift"
            else:
                name = "rshift" if ctype.signed else "urshift"
            return Value(ctype, self.fit(_binary(name, value, count), ctype))
        ctype = usual_conversions(left.ctype, right.ctype)
        first = self.fit(left.expr, ctype)
        second = self.fit(right.expr, ctype)
        names = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "&": "and",
            "|": "or",
            "^": "xor",
        }
        if operator in names:
            return Value(ctype, self.fit(_binary(names[operator], first, second), ctype))
        if operator in {"/", "%"}:
            if _value_of(second) == 0:
                self.error("division by zero", token)
            signed = ctype.signed
            name = {
                ("/", True): "sdiv",
                ("/", False): "udiv",
                ("%", True): "smod",
                ("%", False): "umod",
            }[(operator, signed)]
            return Value(ctype, self.fit(_binary(name, first, second), ctype))
        self.error(f"unsupported binary operator {operator!r}", token)

    def pointer_arithmetic(
        self, operator: str, left: Value, right: Value, token: Token
    ) -> Value:
        if isinstance(left.ctype, PointerType) and isinstance(
            right.ctype, PointerType
        ):
            if operator != "-":
                self.error("two pointers cannot be added", token)
            if not compatible(left.ctype, right.ctype):
                self.error(
                    f"cannot subtract {right.ctype} from {left.ctype}", token
                )
            element = size_of(left.ctype.target)
            if not element:
                self.error(
                    "pointer subtraction needs a complete element type", token
                )
            difference = _binary("sub", left.expr, right.expr)
            return Value(LLONG, _binary("sdiv", difference, IntConstant(element)))
        pointer, count = (left, right)
        if not isinstance(pointer.ctype, PointerType):
            pointer, count = right, left
            if operator == "-":
                self.error("an integer minus a pointer is not valid C", token)
        if not is_integer(count.ctype):
            self.error(
                f"a pointer can only be offset by an integer, not {count.ctype}",
                token,
            )
        element = size_of(pointer.ctype.target)
        if element is None:
            self.error(
                f"cannot do arithmetic on {pointer.ctype}: {pointer.ctype.target} is "
                "an incomplete type whose size py2bin does not know",
                token,
            )
        scaled = _binary("mul", count.expr, IntConstant(element))
        name = "add" if operator == "+" else "sub"
        return Value(pointer.ctype, _binary(name, pointer.expr, scaled))

    def comparison(self, node: Binary) -> Value:
        left = self.rvalue(node.left)
        right = self.rvalue(node.right)
        names = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
        name = names[node.operator]
        if isinstance(left.ctype, PointerType) or isinstance(right.ctype, PointerType):
            first, second = left, right
            if left.null and isinstance(right.ctype, PointerType):
                first = Value(right.ctype, IntConstant(0))
            elif right.null and isinstance(left.ctype, PointerType):
                second = Value(left.ctype, IntConstant(0))
            elif not (
                isinstance(left.ctype, PointerType)
                and isinstance(right.ctype, PointerType)
                and compatible(left.ctype, right.ctype)
            ):
                self.error(
                    f"cannot compare {left.ctype} with {right.ctype}; a pointer "
                    "compares with a compatible pointer or with NULL",
                    node.token,
                )
            if name not in {"eq", "ne"}:
                name = "u" + name  # addresses are unsigned
            return Value(INT, _compare(name, first.expr, second.expr))
        if not is_arithmetic(left.ctype) or not is_arithmetic(right.ctype):
            self.error(
                f"cannot compare {left.ctype} with {right.ctype}", node.token
            )
        if is_floating(left.ctype) or is_floating(right.ctype):
            # An IEEE comparison has four outcomes, and the backends give the
            # unordered one its own handling: every ordering below is false
            # when either operand is a NaN, and only '!=' is true.
            return Value(
                INT, FloatCompare(name, self.widen(left), self.widen(right))
            )
        ctype = usual_conversions(left.ctype, right.ctype)
        if not ctype.signed and ctype.size == 8 and name not in {"eq", "ne"}:
            # Only a 64-bit unsigned type needs the unsigned condition codes:
            # everything narrower is zero-extended into a non-negative i64.
            name = "u" + name
        return Value(
            INT,
            _compare(name, self.fit(left.expr, ctype), self.fit(right.expr, ctype)),
        )

    def logical(self, node: Logical) -> Value:
        slot = self.new_temp()
        end = self.new_label("logic_end")
        left = self.scalar(node.left, f"the left operand of {node.operator!r}")
        if node.operator == "&&":
            self.emit(Store(slot, IntConstant(0)))
            self.emit(JumpIfFalse(self.truth(left), end))
            right = self.scalar(node.right, "the right operand of '&&'")
            self.emit(JumpIfFalse(self.truth(right), end))
            self.emit(Store(slot, IntConstant(1)))
        else:
            taken = self.new_label("logic_true")
            other = self.new_label("logic_right")
            self.emit(Store(slot, IntConstant(0)))
            self.emit(JumpIfFalse(self.truth(left), other))
            self.emit(Jump(taken))
            self.emit(Label(other))
            right = self.scalar(node.right, "the right operand of '||'")
            self.emit(JumpIfFalse(self.truth(right), end))
            self.emit(Label(taken))
            self.emit(Store(slot, IntConstant(1)))
        self.emit(Label(end))
        return Value(INT, IntLoad(slot))

    def conditional(self, node: Conditional) -> Value:
        """``a ? b : c``, with only the selected arm evaluated.

        Each arm stores its own canonical value into one slot and the common
        type is applied once at the merge point. That is equivalent to
        converting each arm separately -- the common type is never narrower
        than either arm -- and it means the arms can be lowered in the order C
        requires instead of both being evaluated to pick between them.

        Whether the slot holds an integer or a double is only known once BOTH
        arms have been lowered, so each arm's store is emitted as a placeholder
        and rewritten in place afterwards. Rewriting rather than re-lowering is
        what keeps each arm evaluated exactly once.
        """

        test = self.scalar(node.test, "the condition of '?:'")
        otherwise = self.new_label("select_else")
        end = self.new_label("select_end")
        slot = self.new_temp()
        self.emit(JumpIfFalse(self.truth(test), otherwise))
        body = self.rvalue(node.body)
        body_store = len(self.operations)
        self.emit(Store(slot, IntConstant(0)))
        self.emit(Jump(end))
        self.emit(Label(otherwise))
        alternative = self.rvalue(node.alternative)
        alternative_store = len(self.operations)
        self.emit(Store(slot, IntConstant(0)))
        self.emit(Label(end))
        if isinstance(body.ctype, VoidType) or isinstance(alternative.ctype, VoidType):
            if not (
                isinstance(body.ctype, VoidType)
                and isinstance(alternative.ctype, VoidType)
            ):
                self.error(
                    "one arm of '?:' has a value and the other does not", node.token
                )
            return Value(VOID, IntConstant(0))
        if is_arithmetic(body.ctype) and is_arithmetic(alternative.ctype):
            result: CType = arithmetic_conversions(body.ctype, alternative.ctype)
        elif isinstance(body.ctype, PointerType) and alternative.null:
            result = body.ctype
        elif isinstance(alternative.ctype, PointerType) and body.null:
            result = alternative.ctype
        elif (
            isinstance(body.ctype, PointerType)
            and isinstance(alternative.ctype, PointerType)
            and compatible(body.ctype, alternative.ctype)
        ):
            result = (
                alternative.ctype
                if isinstance(body.ctype.target, VoidType)
                else body.ctype
            )
        else:
            self.error(
                f"the arms of '?:' have incompatible types {body.ctype} and "
                f"{alternative.ctype}",
                node.token,
            )
        # Now that the common type is known, give each arm's placeholder store
        # the form that type needs. A floating result keeps the full binary64
        # value in the slot, so an integer arm is converted here and a 'float'
        # arm is not rounded -- the extra precision is removed later, where C
        # says it must be.
        for index, arm in ((body_store, body), (alternative_store, alternative)):
            if is_floating(result):
                self.operations[index] = FloatStore(slot, self.widen(arm))
            else:
                self.operations[index] = Store(slot, arm.expr)
        if is_floating(result):
            return Value(result, FloatLoad(slot))
        return Value(result, self.fit(IntLoad(slot), result))

    def cast(self, node: Cast) -> Value:
        value = self.rvalue(node.operand)
        target = node.ctype
        if isinstance(target, VoidType):
            return Value(VOID, IntConstant(0))
        if isinstance(target, ArrayType):
            self.error("a cast cannot name an array type", node.token)
        if isinstance(target, OpaqueType):
            self.error(f"a cast cannot name the incomplete type {target}", node.token)
        if not is_scalar(value.ctype):
            self.error(
                f"cannot cast a value of type {value.ctype} to {target}", node.token
            )
        if isinstance(target, FloatingType):
            if not is_arithmetic(value.ctype):
                self.error(
                    f"cannot cast {value.ctype} to {target}; C has no conversion "
                    "between a pointer and a floating type",
                    node.token,
                )
            # A cast is one of the places C requires the extra precision to go.
            return Value(target, self.narrow(self.widen(value), target))
        if isinstance(target, IntegerType):
            if isinstance(value.ctype, FloatingType):
                return Value(target, self.to_integer(value.expr, target))
            return Value(target, self.fit(value.expr, target))
        if isinstance(value.ctype, FloatingType):
            self.error(
                f"cannot cast {value.ctype} to {target}; C has no conversion "
                "between a floating type and a pointer",
                node.token,
            )
        return Value(target, value.expr, null=value.null)

    def copy_struct(
        self, ctype: "StructType", address: IntExpression, node: Assignment
    ) -> Value:
        """Copy a whole struct or union, one aligned word at a time.

        C assigns aggregates by value. The source and destination are distinct
        objects here (C leaves overlapping assignment through pointers to
        memmove), so a straight forward copy is correct. Both are aligned to
        the type's own alignment, so the copy uses the widest unit that
        alignment allows and finishes any remainder with narrower stores.
        """

        source = self.rvalue(node.value)
        if not isinstance(source.ctype, StructType) or source.ctype is not ctype:
            self.error(
                f"this assignment needs {ctype}, but this is {source.ctype}",
                node.token,
            )
        destination = self.stabilize(address)
        origin = self.stabilize(source.expr)
        remaining = ctype.size
        offset = 0
        while remaining:
            unit = 8 if remaining >= 8 and ctype.alignment >= 8 else (
                4 if remaining >= 4 and ctype.alignment >= 4 else (
                    2 if remaining >= 2 and ctype.alignment >= 2 else 1
                )
            )
            self.emit(
                HeapStore(
                    IntBinary("add", destination, IntConstant(offset)),
                    HeapLoad(
                        IntBinary("add", origin, IntConstant(offset)), unit, False
                    ),
                    unit,
                )
            )
            offset += unit
            remaining -= unit
        return Value(ctype, destination)

    def load_bitfield(self, member: Member, address: IntExpression) -> Value:
        """Read the bits this member owns out of the unit it shares.

        Shifted down, masked to its width, and - for a signed one - given
        back the sign C says those bits carry. Without that last step a
        `signed int f : 3` holding -1 would read as 7.
        """

        held = member.ctype
        size = size_of(held) or 1
        unit = self.load(held, address).expr
        shifted = (
            _binary("urshift", unit, IntConstant(member.bit))
            if member.bit
            else unit
        )
        mask = (1 << member.width) - 1
        value = _binary("and", shifted, IntConstant(mask))
        if held.signed and member.width < size * 8:
            # The field's own sign bit, brought back: `(v ^ s) - s` where `s`
            # is that bit. Written this way because it is two instructions
            # and needs no branch.
            sign = 1 << (member.width - 1)
            value = _binary(
                "sub", _binary("xor", value, IntConstant(sign)), IntConstant(sign)
            )
        return Value(held, self.from_bits(value, held))

    def assign_bitfield(
        self, node: Assignment, member: Member, address: IntExpression
    ) -> Value:
        """Write this member's bits, leaving the rest of the unit alone.

        Read, clear, or in, store. C says the value is truncated to the
        field's width, and the result of the assignment is what was stored -
        so the answer is read back out rather than being what was written.
        """

        held = member.ctype
        address = self.stabilize(address)
        if node.operator == "=":
            value = self.rvalue(node.value)
            stored = self.assign_convert(value, held, node.token, "this assignment")
        else:
            current = self.load_bitfield(member, address)
            operand = self.rvalue(node.value)
            combined = self.apply(node.operator[:-1], current, operand, node.token)
            stored = self.assign_convert(
                combined, held, node.token, "this compound assignment"
            )
        mask = (1 << member.width) - 1
        bits = self.materialize(self.stored_bits(stored, held))
        unit = self.load(held, address).expr
        kept = _binary("and", unit, IntConstant(~(mask << member.bit)))
        put = _binary(
            "lshift", _binary("and", bits, IntConstant(mask)), IntConstant(member.bit)
        )
        self.emit(
            HeapStore(address, _binary("or", kept, put), size_of(held))
        )
        return self.load_bitfield(member, address)

    def assignment(self, node: Assignment) -> Value:
        self.packed = None
        ctype, address = self.lvalue(node.target)
        if self.packed is not None:
            return self.assign_bitfield(node, self.packed, address)
        if isinstance(ctype, ArrayType):
            self.error("an array is not assignable", node.token)
        address = self.stabilize(address)
        if isinstance(ctype, StructType):
            if node.operator != "=":
                self.error(
                    f"{node.operator} is not defined for {ctype}", node.token
                )
            return self.copy_struct(ctype, address, node)
        if node.operator == "=":
            value = self.rvalue(node.value)
            stored = self.assign_convert(
                value, ctype, node.token, "this assignment"
            )
        else:
            current = self.load(ctype, address)
            operand = self.rvalue(node.value)
            combined = self.apply(node.operator[:-1], current, operand, node.token)
            stored = self.assign_convert(
                combined, ctype, node.token, "this compound assignment"
            )
        # The stored expression may read the very object about to be written, so
        # it has to be pinned before the store rather than recomputed after it.
        if is_floating(ctype):
            stored = self.materialize_float(stored)
        else:
            stored = self.materialize(stored)
        self.emit(HeapStore(address, self.stored_bits(stored, ctype), size_of(ctype)))
        return Value(ctype, stored)

    # --- calls ---

    def function_designator(
        self, name: str, token: Token
    ) -> tuple[CType, IntExpression]:
        """The type and the runtime entry address of the function ``name``.

        Naming a function without calling it is what makes a function pointer,
        so this is also where the body is lowered: nothing else would have
        pulled it into the module, and the address of a body that was never
        emitted would point at whatever followed it.
        """

        function = self.unit.functions[name]
        if not self.calls_are_real:
            self.error(
                f"taking the address of {name!r} needs a real machine call, and "
                f"the call ABI is not implemented for target {self.target!r}; "
                "the compiler inlines calls there instead",
                token,
            )
        if function.body is None:
            self.error(
                f"{name!r} is declared but never defined, so it has no address; "
                "py2bin has no linker, and every function a program uses has to "
                "be defined in this translation unit",
                token,
            )
        if name == "main":
            self.error(
                "main() is the process entry point and its 'return' exits the "
                "process, so py2bin does not let a program take its address",
                token,
            )
        limit = self.argument_limit()
        if len(function.parameters) > limit:
            self.error(
                f"{name}() takes {len(function.parameters)} parameters; py2bin's "
                f"call ABI passes at most {limit} arguments in "
                "registers and does not implement stack arguments",
                token,
            )
        self.lower_callee(function)
        ctype = FunctionType(
            function.result, tuple(item for item, _name in function.parameters)
        )
        return ctype, FunctionAddress(name)


    def math_builtin(self, node: Call) -> Value:
        """Lower a one-instruction C math function.

        These are not library calls. Each maps to a single floating-point
        instruction the CPU already has, so the result is exact for the
        operations IEEE-754 defines exactly (sqrt is correctly rounded, and the
        rounding functions are exact), and nothing has to be linked.
        """

        if len(node.arguments) != 1:
            self.error(
                f"{node.name}() takes exactly one argument", node.token
            )
        if node.name == "round" and not self.target.endswith("-arm64"):
            self.error(
                "round() breaks ties away from zero, which x86-64's roundsd "
                "cannot do in one instruction and py2bin will not approximate; "
                "use trunc(), floor() or ceil(), or target arm64",
                node.token,
            )
        value = self.rvalue(node.arguments[0])
        if not is_arithmetic(value.ctype):
            self.error(
                f"{node.name}() needs a number, not {value.ctype}", node.token
            )
        return Value(DOUBLE, FloatUnary(_MATH_BUILTINS[node.name], self.widen(value)))

    def arena_builtin(self, node: Call) -> Value:
        """Reserve the arena and hand back its base.

        This is the only part of the heap the compiler has to provide: one
        anonymous mapping. `malloc` and its family are ordinary C written on
        top of it in <stdlib.h>, which is why they can be read rather than
        taken on trust.
        """

        if node.arguments:
            self.error("__py2bin_arena() takes no arguments", node.token)
        slot = self.take(8)
        self.emit(HeapInit(slot, ARENA_BYTES))
        return Value(PointerType(VOID), IntLoad(slot))

    def file_builtin(self, node: Call) -> Value:
        """One file syscall, with the kernel's own answer handed back.

        These are what <stdio.h>'s FILE layer and <filesystem> are written on
        top of, in C - the same arrangement as the allocator. A failure comes
        back as a negative errno rather than through anything hidden, because
        that is what the kernel returns and there is nowhere else to put it.

        POSIX only: the syscalls are the interface. Windows has handles and
        a different set of calls, and the headers reach those through the
        imports <windows.h> declares - which is why they are written with
        `#ifdef _WIN32` rather than pretending one shape fits both.
        """

        kind, arity = _FILE_BUILTINS[node.name]
        if self.target.startswith("windows-"):
            self.error(
                f"{node.name}() is a POSIX system call, and Windows has no "
                "such thing to make. py2bin's own headers use the functions "
                "<windows.h> imports there instead; code calling this "
                "directly needs the same `#ifdef _WIN32`",
                node.token,
            )
        if len(node.arguments) != arity:
            self.error(
                f"{node.name}() takes exactly {arity} argument(s)", node.token
            )
        arguments = []
        for argument in node.arguments:
            value = self.rvalue(argument)
            if not is_integer(value.ctype) and not isinstance(
                value.ctype, PointerType
            ):
                self.error(
                    f"{node.name}() takes integers and pointers, not "
                    f"{value.ctype}",
                    argument.token,
                )
            arguments.append(value.expr)
        slot = self.take(8)
        self.emit(FileCall(kind, slot, tuple(arguments)))
        return Value(LONG, IntLoad(slot))

    def exit_builtin(self, node: Call) -> None:
        """`exit(status)` and `abort()`: stop the process, here and now.

        The IR already ends a program this way - it is what `return` from
        `main` compiles to - and nothing about it needs to be at the end of
        `main`, so a call anywhere works.
        """

        if node.name == "abort":
            if node.arguments:
                self.error("abort() takes no arguments", node.token)
            # 134 is what a shell reports for a process killed by SIGABRT.
            # py2bin cannot raise a signal, so this is the nearest thing that
            # a caller testing the status will recognise.
            self.emit(ExitValue(IntConstant(_ABORT_STATUS)))
            return
        if len(node.arguments) != 1:
            self.error(f"{node.name}() takes exactly one argument", node.token)
        value = self.rvalue(node.arguments[0])
        if not is_integer(value.ctype):
            self.error(f"{node.name}() needs an integer status", node.token)
        self.emit(ExitValue(self.fit(value.expr, INT)))

    def argument_limit(self) -> int:
        """How many arguments this target can pass.

        ARM64 implements the AAPCS64 memory argument area, so it is bounded
        only by the frame; the x86 encoders still stop at their register
        count and reject the rest rather than truncating.
        """

        if self.target in _STACK_ARGUMENT_TARGETS:
            return _ARGUMENT_CEILING
        return _MAXIMUM_ARGUMENTS

    def call(self, node: Call) -> Value:
        if node.name in _MATH_BUILTINS and self.lookup(node.name) is None:
            if node.name not in self.unit.functions:
                return self.math_builtin(node)
        if node.name == "__py2bin_arena" and node.name not in self.unit.functions:
            return self.arena_builtin(node)
        if node.name in _FILE_BUILTINS and node.name not in self.unit.functions:
            return self.file_builtin(node)
        if node.name in _EXIT_BUILTINS and node.name not in self.unit.functions:
            self.exit_builtin(node)
            # `exit` does not come back, and C says its type is void. A caller
            # that uses the value of it is a caller that is wrong about what
            # it does, so there is nothing to hand back.
            return Value(INT, IntConstant(0))
        if node.name in _VARIADIC_BUILTINS and node.name not in self.unit.functions:
            return self.variadic_builtin(node)
        if node.name in _INTO_A_BUFFER and node.name not in self.unit.functions:
            return self.formatted_into(node, bounded=node.name in _BOUNDED_BUFFER)
        if node.name == "printf" and "printf" not in self.unit.functions:
            self.error(
                "printf's return value is not implemented; call it as a "
                "statement, or use snprintf if the count is what is wanted",
                node.token,
            )
        if self.lookup(node.name) is not None:
            # An object of function-pointer type shadows any function of the
            # same name, exactly as C's scoping says it does.
            return self.call_through(
                Identifier(node.token, node.name), node.arguments, node.token
            )
        if node.name in self.unit.externs:
            return self.extern_call(node, discarded=False)
        function = self.unit.functions.get(node.name)
        if function is None:
            self.error(
                f"call to {node.name!r}, which is not a function declared in this "
                "translation unit or a declared extern",
                node.token,
            )
        if function.body is None:
            self.error(
                f"call to {node.name!r}, which is declared but never defined; "
                "py2bin has no linker, so the body of every function a program "
                "calls has to be in this translation unit - or, where it lives "
                "in a shared library somebody else shipped, name that library "
                "with --library NAME.dll and it becomes an import",
                node.token,
            )
        if function.name == "main":
            self.error(
                "py2bin's C compiler does not support calling main(): it is the "
                "process entry point, and its 'return' exits the process",
                node.token,
            )
        if self.calls_are_real:
            return self.direct_call(function, node)
        return self.inline(function, node)

    # --- calls through a pointer -------------------------------------------
    #
    # The callee is not known until the program runs, so there is nothing to
    # inline and nothing to check at the call site beyond the POINTER's own
    # prototype -- which is precisely why C makes the prototype part of the
    # type. A target that is not a pointer to a function is rejected here.

    def call_through(
        self, target: Node, arguments: list[Node], token: Token
    ) -> Value:
        pointer = self.rvalue(target)
        ctype = pointer.ctype
        if not (
            isinstance(ctype, PointerType) and isinstance(ctype.target, FunctionType)
        ):
            self.error(
                f"this call needs a function or a pointer to one, not {ctype}",
                token,
            )
        signature = ctype.target
        if not self.calls_are_real:
            self.error(
                "a call through a function pointer needs a real machine call, "
                f"and the call ABI is not implemented for target {self.target!r}",
                token,
            )
        limit = self.argument_limit()
        if len(signature.parameters) > limit:
            self.error(
                f"{signature} takes {len(signature.parameters)} parameters; "
                f"py2bin's call ABI passes at most {limit} arguments "
                "in registers and does not implement stack arguments",
                token,
            )
        if len(arguments) != len(signature.parameters):
            self.error(
                f"a call through {ctype} takes {len(signature.parameters)} "
                f"argument(s), got {len(arguments)}",
                token,
            )
        # The pointer is evaluated first and pinned. Lowering an argument may
        # emit stores, and re-emitting the target expression after them would
        # both repeat any call inside it and let it read the wrong memory.
        address = self.materialize(pointer.expr)
        prepared = [
            self.stored_bits(
                self.assign_convert(
                    self.rvalue(argument),
                    parameter,
                    argument.token,
                    f"argument {position} of a call through {ctype}",
                ),
                parameter,
            )
            for position, (argument, parameter) in enumerate(
                zip(arguments, signature.parameters), 1
            )
        ]
        slot = self.new_temp()
        self.emit(Store(slot, IndirectCall(address, tuple(prepared))))
        if isinstance(signature.result, VoidType):
            return Value(VOID, IntConstant(0))
        return Value(
            signature.result, self.from_bits(IntLoad(slot), signature.result)
        )

    def extern_call(self, node: Call, *, discarded: bool) -> Value:
        name = node.name
        # The vetted table first, then whatever a named library claimed:
        # nothing a program says may replace a shape py2bin has checked.
        if name in _CABI_SYMBOLS:
            symbol, signature = _CABI_SYMBOLS[name]
            result_kind = _CABI_RESULTS[name]
        else:
            symbol, signature, result_kind = self.unit.library_symbols[name]
        if result_kind == "void" and not discarded:
            self.error(
                f"extern call {name}() returns void; its result is not a value and "
                "can only be discarded",
                node.token,
            )
        if len(node.arguments) != len(signature):
            self.error(
                f"{name}() takes {len(signature)} argument(s), got "
                f"{len(node.arguments)}",
                node.token,
            )
        arguments: list[IntExpression] = []
        for position, (argument, kind) in enumerate(
            zip(node.arguments, signature), 1
        ):
            what = f"argument {position} of {name}()"
            if kind in {"cstr", "cfmt", "cdata"}:
                if not isinstance(argument, StringLiteral):
                    self.error(
                        f"{what} must be a literal C string: py2bin materializes it "
                        "in the image, and a runtime pointer would need a lifetime "
                        "this compiler cannot verify",
                        argument.token,
                    )
                if kind != "cdata" and b"\0" in argument.data:
                    # A "cdata" callee is given the length separately, so it
                    # reads every byte rather than stopping at the first zero.
                    self.error(
                        f"{what} contains an embedded NUL the callee would truncate",
                        argument.token,
                    )
                if kind == "cfmt" and b"%" in argument.data:
                    self.error(
                        f"{name}() is variadic and py2bin passes no variadic "
                        "arguments, so its format string must not contain '%'",
                        argument.token,
                    )
                arguments.append(CStringConstant(argument.data + b"\0"))
                continue
            value = self.rvalue(argument)
            if kind == "ptr":
                if value.null:
                    arguments.append(IntConstant(0))
                elif isinstance(value.ctype, PointerType):
                    arguments.append(value.expr)
                else:
                    self.error(
                        f"{what} needs a pointer handle; pass a handle or NULL",
                        argument.token,
                    )
            elif kind == "f64":
                # A double travels in its own register class, so it is passed
                # as a double rather than squeezed through an integer.
                if not isinstance(value.ctype, (FloatingType, IntegerType)):
                    self.error(
                        f"{what} needs a number, not {value.ctype}", argument.token
                    )
                arguments.append(self.widen(value))
            else:
                if not is_integer(value.ctype):
                    self.error(
                        f"{what} needs an integer, not {value.ctype}", argument.token
                    )
                arguments.append(self.fit(value.expr, LLONG))
        call = ExternCall(
            symbol, tuple(arguments), _CABI_RESULT_WIDTH.get(symbol, "i64")
        )
        if discarded or result_kind == "void":
            self.emit(Store(self.new_temp(), call))
            return Value(VOID, IntConstant(0))
        slot = self.new_temp()
        self.emit(Store(slot, call))
        declared = self.unit.externs[name]
        return Value(declared, self.fit(IntLoad(slot), declared))

    def variadic_builtin(self, node: Call) -> Value:
        """`va_start`, `va_arg`, `va_end` and `va_copy`, written out here.

        A `va_list` is a pointer to the cells the call wrote its extra
        arguments into, so starting one is an assignment, reading one is a
        load and a step, and ending one is nothing at all.
        """

        if node.name == "va_end":
            return Value(VOID, IntConstant(0))
        if node.name in ("va_start", "va_copy"):
            if len(node.arguments) != 2:
                self.error(f"{node.name} takes two arguments", node.token)
            ctype, address = self.lvalue(node.arguments[0])
            if node.name == "va_copy":
                source = self.rvalue(node.arguments[1])
                self.emit(HeapStore(address, source.expr, 8))
                return Value(VOID, IntConstant(0))
            held = self.lookup(_VARIADIC_PARAMETER)
            if held is None:
                self.error(
                    "va_start is only meaningful inside a function whose "
                    "parameter list ends with '...'",
                    node.token,
                )
            self.emit(
                HeapStore(address, HeapLoad(self.address_of(held), 8), 8)
            )
            return Value(VOID, IntConstant(0))
        # `va_arg(ap, T)`: read the cell, then move past it.
        if len(node.arguments) != 2:
            self.error("va_arg takes a va_list and a type", node.token)
        wanted = node.arguments[1]
        if not isinstance(wanted, TypeArgument):
            self.error(
                "the second argument of va_arg has to be a type", node.token
            )
        ctype, address = self.lvalue(node.arguments[0])
        cell = self.new_temp()
        self.emit(Store(cell, HeapLoad(address, 8)))
        self.emit(
            HeapStore(address, _binary("add", IntLoad(cell), IntConstant(8)), 8)
        )
        held = wanted.ctype
        if is_floating(held):
            # Promoted to a double where it was passed, and rounded here if
            # the program asked for a float.
            return Value(
                held,
                self.assign_convert(
                    Value(DOUBLE, BitsFloat(HeapLoad(IntLoad(cell), 8), 8)),
                    held,
                    node.token,
                    "va_arg",
                ),
            )
        return Value(held, self.fit(HeapLoad(IntLoad(cell), 8), held))

    def gather_variadic(self, function: Function, node: Call) -> IntExpression:
        """Write the arguments past the named ones into a run of cells.

        C promotes what it passes to `...`: an integer narrower than `int`
        arrives as an `int`, and a `float` as a `double`. Each promoted value
        takes one eight-byte cell, so `va_arg` is a load and a step forward -
        which is the whole of what a `va_list` has to be here.
        """

        extra = node.arguments[len(function.parameters):]
        area = self.allocate(8 * max(1, len(extra)))
        for index, argument in enumerate(extra):
            value = self.rvalue(argument)
            if is_floating(value.ctype):
                self.emit(
                    HeapStore(
                        _binary("add", SlotAddress(area), IntConstant(index * 8)),
                        FloatBits(self.promote_float(value), 8),
                        8,
                    )
                )
                continue
            if not is_integer(value.ctype) and not isinstance(
                value.ctype, PointerType
            ):
                self.error(
                    f"a variadic argument has to be a number or a pointer, "
                    f"not {value.ctype}",
                    argument.token,
                )
            self.emit(
                HeapStore(
                    _binary("add", SlotAddress(area), IntConstant(index * 8)),
                    value.expr,
                    8,
                )
            )
        return SlotAddress(area)

    def promote_float(self, value: Value) -> FloatExpression:
        """A `float` argument arrives as a `double`, which C says it must."""

        return value.expr

    def prepare_arguments(
        self, function: Function, node: Call
    ) -> list[IntExpression]:
        """Check the argument count and convert each argument to its parameter.

        Every argument comes back as an INTEGER expression, because py2bin's
        internal call ABI passes a floating argument as the object's IEEE bit
        pattern in an integer register. That ABI is py2bin's own -- a compiled C
        function here is never called by anything but py2bin's own code -- and
        it means a double argument needs no new register class in either
        encoder while still delivering the exact value. Passing the bit pattern
        of the PARAMETER's type is what makes a 'float' parameter arrive as the
        four bytes its object holds.
        """

        if function.variadic:
            if len(node.arguments) < len(function.parameters):
                self.error(
                    f"{function.name}() takes at least "
                    f"{len(function.parameters)} argument(s), got "
                    f"{len(node.arguments)}",
                    node.token,
                )
        elif len(node.arguments) != len(function.parameters):
            self.error(
                f"{function.name}() takes {len(function.parameters)} argument(s), "
                f"got {len(node.arguments)}",
                node.token,
            )
        prepared: list[IntExpression] = []
        area: "IntExpression | None" = (
            self.gather_variadic(function, node) if function.variadic else None
        )
        for position, (argument, (parameter_type, _name)) in enumerate(
            zip(node.arguments, function.parameters), 1
        ):
            converted = self.assign_convert(
                self.rvalue(argument),
                parameter_type,
                argument.token,
                f"argument {position} of {function.name}()",
            )
            prepared.append(self.stored_bits(converted, parameter_type))
        if area is not None:
            # One more argument than the program wrote: where the extras are.
            # It travels like any other, so the callee finds them whether it
            # was inlined into this frame or called into one of its own.
            prepared.append(area)
        return prepared

    # --- real calls --------------------------------------------------------
    #
    # On a target whose encoder implements the call ABI, a call is a call: the
    # callee is lowered once into its own IR Function with its own frame, and
    # the call site branches to it. Recursion then costs nothing special -- the
    # body being lowered simply refers to itself by name.

    def direct_call(self, function: Function, node: Call) -> Value:
        limit = self.argument_limit()
        if len(function.parameters) > limit:
            self.error(
                f"{function.name}() takes {len(function.parameters)} parameters; "
                f"py2bin's call ABI passes at most {limit} arguments "
                "in registers and does not implement stack arguments",
                node.token,
            )
        prepared = self.prepare_arguments(function, node)
        self.lower_callee(function)
        call = IRCall(function.name, tuple(prepared))
        # Pin the result in a slot at once. Everything downstream may re-read a
        # value any number of times, and re-emitting a call expression would
        # make the call happen again -- the defect this backend has produced
        # more often than any other.
        slot = self.new_temp()
        self.emit(Store(slot, call))
        if isinstance(function.result, VoidType):
            return Value(VOID, IntConstant(0))
        # A floating result came back as its bit pattern in the integer result
        # register, the mirror image of how prepare_arguments passed one in.
        return Value(function.result, self.from_bits(IntLoad(slot), function.result))

    def lower_callee(self, function: Function) -> None:
        """Lower ``function`` into its own IR body, once.

        A call that arrives while the callee's own body is still being lowered
        is exactly what recursion is; the name is already reserved, so it just
        returns and the call site emits its branch.
        """

        if function.name in self.lowered or function.name in self.lowering:
            return
        assert function.body is not None
        self.lowering.add(function.name)
        saved = (
            self.operations,
            self.stack_slots,
            self.peak_slots,
            self.reserved_slots,
            self.scopes,
            self.buffer_slot,
            self.digit_slot,
            self.text_slot,
            self.float_scratch,
            self.float_entry,
            self.float_dispatch,
            self.float_returns,
            self.break_targets,
            self.continue_targets,
            self.switches,
            self.functions,
        )
        self.operations = []
        self.stack_slots = 0
        self.peak_slots = 0
        self.reserved_slots = 0
        self.scopes = [{}]
        self.buffer_slot = None
        self.digit_slot = None
        self.text_slot = None
        self.float_scratch = {}
        self.float_entry = None
        self.float_dispatch = None
        self.float_returns = []
        self.break_targets = []
        self.continue_targets = []
        self.switches = []
        self.functions = []
        try:
            for parameter_type, name in function.parameters:
                local = self.declare(name, parameter_type, function.token)
                if local.slot != self.stack_slots - 1:
                    raise AssertionError(
                        "a parameter must occupy exactly one stack slot"
                    )
            if function.variadic:
                self.declare(_VARIADIC_PARAMETER, PointerType(CHAR), function.token)
            wanted = len(function.parameters) + (1 if function.variadic else 0)
            if self.stack_slots != wanted:
                raise AssertionError("parameter slots must be slots 0..n-1")
            context = FunctionContext(
                function,
                None,
                self.new_label(f"return_{function.name}"),
                False,
                call_body=True,
            )
            self.functions.append(context)
            self.block(function.body)
            self.functions.pop()
            self.check_labels(context)
            # Falling off the end of a non-void function is undefined in C.
            # Returning a defined 0 is the one choice that cannot surprise:
            # nothing is left to run off the end of the body into.
            self.emit(
                IRReturn(
                    None if isinstance(function.result, VoidType) else IntConstant(0)
                )
            )
            # After the return, so nothing can fall into it.
            self.emit_float_dispatch()
            body = IRFunction(
                function.name,
                len(function.parameters) + (1 if function.variadic else 0),
                self.peak_slots,
                self.operations,
            )
        finally:
            (
                self.operations,
                self.stack_slots,
                self.peak_slots,
                self.reserved_slots,
                self.scopes,
                self.buffer_slot,
                self.digit_slot,
                self.text_slot,
                self.float_scratch,
                self.float_entry,
                self.float_dispatch,
                self.float_returns,
                self.break_targets,
                self.continue_targets,
                self.switches,
                self.functions,
            ) = saved
            self.lowering.discard(function.name)
        self.lowered[function.name] = body

    def inline(self, function: Function, node: Call) -> Value:
        if function.name in self.active:
            self.error(
                f"recursive call to {function.name}(): py2bin's call ABI is not "
                f"implemented for target {self.target!r}, so every function is "
                "inlined there and recursion cannot be expressed; the targets "
                f"that do support it are {', '.join(sorted(CALL_CAPABLE_TARGETS))}",
                node.token,
            )
        prepared = self.prepare_arguments(function, node)
        self.scopes.append({})
        wanted = [*function.parameters]
        if function.variadic:
            wanted.append((PointerType(CHAR), _VARIADIC_PARAMETER))
        for (parameter_type, name), expression in zip(wanted, prepared):
            local = self.declare(name, parameter_type, node.token)
            self.emit(
                HeapStore(
                    SlotAddress(local.slot), expression, size_of(parameter_type)
                )
            )
        result_slot = None
        if not isinstance(function.result, VoidType):
            result_slot = self.new_temp()
            self.emit(Store(result_slot, IntConstant(0)))
        return_label = self.new_label(f"return_{function.name}")
        context = FunctionContext(function, result_slot, return_label, False)
        self.functions.append(context)
        self.active.append(function.name)
        saved_breaks, self.break_targets = self.break_targets, []
        saved_continues, self.continue_targets = self.continue_targets, []
        saved_switches, self.switches = self.switches, []
        self.block(function.body)
        self.break_targets = saved_breaks
        self.continue_targets = saved_continues
        self.switches = saved_switches
        self.active.pop()
        self.functions.pop()
        self.check_labels(context)
        self.emit(Label(return_label))
        self.scopes.pop()
        if result_slot is None:
            return Value(VOID, IntConstant(0))
        return Value(function.result, self.from_bits(IntLoad(result_slot), function.result))

    def check_labels(self, context: FunctionContext) -> None:
        for name, token in context.pending:
            if name not in context.defined:
                self.error(
                    f"goto {name}: there is no label {name!r} in "
                    f"{context.function.name}()",
                    token,
                )

    # --- statements ---

    def block(self, node: Compound) -> None:
        self.scopes.append({})
        for statement in node.body:
            self.statement(statement)
        self.scopes.pop()

    def statement(self, node: Node) -> None:
        """One statement, then its temporaries handed back.

        Every intermediate a statement needs is dead once it finishes: C gives
        no way to name a temporary, so nothing outside can still be holding
        one. Slots were previously taken and never given back, which made the
        frame grow with the *length* of a function rather than with how much
        it needs at once - about eighteen hundred statements reached the
        512 KB limit and the build was refused.

        The mark is taken here, per statement, which is what makes this safe
        around loops. A `while` evaluates its condition into temporaries taken
        before the body is lowered, and the body's own statements can only
        reclaim down to marks above those - so the condition's slots survive
        the body, which at run time they must, since the loop comes back to
        them. The whole `while` is itself one statement in its enclosing list,
        so its condition is reclaimed once the loop is done.
        """

        mark = self.stack_slots
        self.statement_body(node)
        self.release_temporaries(mark)

    def statement_body(self, node: Node) -> None:
        if isinstance(node, Compound):
            self.block(node)
            return
        if isinstance(node, ExpressionStatement):
            if node.expression is not None:
                self.expression_statement(node.expression)
            return
        if isinstance(node, Declaration):
            self.declaration(node)
            return
        if isinstance(node, If):
            self.if_statement(node)
            return
        if isinstance(node, While):
            self.while_statement(node)
            return
        if isinstance(node, DoWhile):
            self.do_while_statement(node)
            return
        if isinstance(node, For):
            self.for_statement(node)
            return
        if isinstance(node, Switch):
            self.switch_statement(node)
            return
        if isinstance(node, Labeled):
            self.labeled_statement(node)
            return
        if isinstance(node, Goto):
            context = self.functions[-1]
            target = context.labels.setdefault(
                node.name, self.new_label(f"goto_{node.name}")
            )
            context.pending.append((node.name, node.token))
            self.emit(Jump(target))
            return
        if isinstance(node, Break):
            if not self.break_targets:
                self.error("'break' is not inside a loop or switch", node.token)
            self.emit(Jump(self.break_targets[-1]))
            return
        if isinstance(node, Continue):
            if not self.continue_targets:
                self.error("'continue' is not inside a loop", node.token)
            self.emit(Jump(self.continue_targets[-1]))
            return
        if isinstance(node, Return):
            self.return_statement(node)
            return
        self.error("unsupported statement", node.token)

    def expression_statement(self, node: Node) -> None:
        """A full expression evaluated for its effect, with its value discarded."""

        if isinstance(node, Call):
            if node.name == "printf" and "printf" not in self.unit.functions:
                self.printf(node)
                return
            if node.name in _INTO_A_BUFFER and node.name not in self.unit.functions:
                self.formatted_into(node, bounded=node.name in _BOUNDED_BUFFER)
                return
            if node.name in _VARIADIC_BUILTINS and node.name not in self.unit.functions:
                self.variadic_builtin(node)
                return
            if node.name in self.unit.externs:
                self.extern_call(node, discarded=True)
                return
        if isinstance(node, Comma):
            self.expression_statement(node.left)
            self.expression_statement(node.right)
            return
        self.rvalue(node)

    def stored_declaration(self, node: Declaration) -> None:
        """`static int n = 0;` inside a block: one object, named only here.

        The slot is keyed by the declaration itself rather than by the call,
        which is what makes this survive inlining: a body compiled into three
        call sites is the same `Declaration` node three times, so all three
        name the one object C promises - and not one object per inlining,
        which is why this used to be refused outright.

        The initial value has to be a constant, as C requires: it is written
        into the static block rather than stored on the way past, so a
        declaration reached twice does not initialise twice.
        """

        for ctype, name, initializer in node.entries:
            if isinstance(ctype, VoidType):
                self.error(f"{name!r} cannot have type void", node.token)
            if isinstance(ctype, ArrayType) and ctype.count is None:
                ctype = self.deduce_array(ctype, initializer, node.token)
            local = self.stored_locals.get((id(node), name))
            if local is None:
                # Reached without having been seen by the pass that gives
                # these their storage, which walks every function's body.
                self.error(
                    f"{name!r} is a static object in a body py2bin did not "
                    f"read before it started compiling",
                    node.token,
                )
            self.scopes[-1][name] = local

    def declaration(self, node: Declaration) -> None:
        if node.stored:
            self.stored_declaration(node)
            return
        for ctype, name, initializer in node.entries:
            if isinstance(ctype, VoidType):
                self.error(f"{name!r} cannot have type void", node.token)
            if isinstance(ctype, ArrayType) and ctype.count is None:
                ctype = self.deduce_array(ctype, initializer, node.token)
            local = self.declare(name, ctype, node.token)
            if initializer is None:
                continue
            if isinstance(ctype, ArrayType):
                self.array_initializer(
                    self.address_of(local), ctype, initializer, node.token
                )
                continue
            if isinstance(ctype, StructType) and isinstance(initializer, tuple):
                self.struct_initializer(
                    self.address_of(local), ctype, initializer, node.token
                )
                continue
            if isinstance(initializer, tuple):
                token, items = initializer
                if len(items) != 1 or isinstance(items[0], tuple):
                    self.error(
                        "a braced initializer for a scalar needs exactly one value",
                        token,
                    )
                initializer = items[0]
            value = self.rvalue(initializer)
            stored = self.assign_convert(
                value, ctype, node.token, f"the initializer for {name!r}"
            )
            self.emit(
                HeapStore(
                    self.address_of(local),
                    self.stored_bits(stored, ctype),
                    size_of(ctype),
                )
            )

    # --- file-scope objects ------------------------------------------------
    #
    # An object with static storage duration is initialized before the program
    # starts, so its initializer has to be a constant expression: there is
    # nothing running yet that could evaluate anything else. py2bin honours
    # that literally -- it lowers the initializer, then requires the result to
    # be a value the compiler already knows, and rejects anything else with a
    # file:line:col error rather than quietly running it at start-up.

    def declare_globals(self) -> None:
        """Give every file-scope object its storage, then its initial value."""

        entries = list(self.unit.globals.values())
        for entry in entries:
            ctype = entry.ctype
            if isinstance(ctype, VoidType):
                self.error(f"{entry.name!r} cannot have type void", entry.token)
            if isinstance(ctype, ArrayType) and ctype.count is None:
                ctype = self.deduce_array(ctype, entry.initializer, entry.token)
                entry.ctype = ctype
            self.statics[entry.name] = Local(
                ctype, self.allocate_static(ctype, entry.token), static=True
            )
        # Every offset is fixed before any initializer is lowered, so one
        # object's initializer may take the address of another.
        for entry in entries:
            self.initialize_global(entry)
        self.declare_stored_locals()

    def declare_stored_locals(self) -> None:
        """Give every `static` inside a block its storage and its first value.

        Here, with the file-scope objects, rather than where the declaration
        stands: C11 6.2.4p3 gives a static local its initial value before the
        program starts, and a store written where the declaration is would run
        it again on every call - which is what a static local exists not to do.
        """

        for function in self.unit.functions.values():
            if function.body is None:
                continue
            for node in _stored_declarations(function.body):
                for ctype, name, initializer in node.entries:
                    if isinstance(ctype, ArrayType) and ctype.count is None:
                        ctype = self.deduce_array(ctype, initializer, node.token)
                    key = (id(node), name)
                    if key in self.stored_locals:
                        continue
                    local = Local(
                        ctype, self.allocate_static(ctype, node.token), static=True
                    )
                    self.stored_locals[key] = local
                    if initializer is not None:
                        self.aggregate_initializer(
                            GlobalAddress(local.slot),
                            ctype,
                            initializer,
                            node.token,
                            static=True,
                        )

    def initialize_global(self, entry: GlobalObject) -> None:
        local = self.statics[entry.name]
        ctype = local.ctype
        base = GlobalAddress(local.slot)
        initializer = entry.initializer
        if initializer is None:
            # C gives a static object with no initializer the value zero, and
            # the block already holds zero everywhere.
            return
        if isinstance(ctype, ArrayType):
            self.array_initializer(base, ctype, initializer, entry.token, static=True)
            return
        if isinstance(ctype, StructType):
            self.struct_initializer(
                base, ctype, initializer, entry.token, static=True
            )
            return
        if isinstance(initializer, tuple):
            token, items = initializer
            if len(items) != 1 or isinstance(items[0], tuple):
                self.error(
                    "a braced initializer for a scalar needs exactly one value",
                    token,
                )
            initializer = items[0]
        stored = self.static_value(
            initializer, ctype, entry.token, f"the initializer for {entry.name!r}"
        )
        bits = self.stored_bits(stored, ctype)
        if bits == IntConstant(0):
            return  # already zero
        self.emit(HeapStore(base, bits, size_of(ctype)))

    def static_value(
        self, node: Node, ctype: CType, token: Token, what: str
    ) -> IntExpression | FloatExpression:
        """Lower a static-storage initializer and insist that it be constant."""

        saved_operations = self.operations
        self.operations = []
        try:
            value = self.rvalue(node)
            converted = self.assign_convert(value, ctype, token, what)
        finally:
            emitted = self.operations
            self.operations = saved_operations
        if emitted or not _is_link_constant(converted):
            self.error(
                f"{what} must be a constant expression: it initializes an object "
                "with static storage duration, which C gives its value before "
                "the program starts running",
                token,
            )
        return converted

    def deduce_array(
        self, ctype: ArrayType, initializer: object, token: Token
    ) -> ArrayType:
        if isinstance(initializer, tuple):
            return ArrayType(ctype.element, max(1, len(initializer[1])))
        if isinstance(initializer, StringLiteral) and _string_fits(
            initializer, ctype.element, self.target
        ):
            size = size_of(ctype.element) or 1
            return ArrayType(
                ctype.element, len(initializer.bytes_for(self.target)) // size
            )
        self.error(
            "an array without a length needs a braced initializer (or a string "
            "literal for a character array) to deduce it from",
            token,
        )

    def aggregate_initializer(
        self,
        base: "IntExpression",
        ctype: "CType",
        initializer: object,
        token: Token,
        *,
        static: bool = False,
    ) -> None:
        """Initialize whatever is at ``base``, of whatever shape it is.

        An array, a struct, or a scalar. Written as one entry point because a
        member may be any of the three and C nests them freely.
        """

        if isinstance(ctype, ArrayType):
            self.array_initializer(base, ctype, initializer, token, static=static)
            return
        if isinstance(ctype, StructType):
            if not isinstance(initializer, tuple):
                self.error(
                    f"{ctype} needs a braced initializer, not a single value",
                    token,
                )
            self.struct_initializer(base, ctype, initializer, token, static=static)
            return
        if isinstance(initializer, tuple):
            _brace, items = initializer
            if len(items) != 1 or isinstance(items[0], tuple):
                self.error(
                    "a braced initializer for a scalar needs exactly one value",
                    token,
                )
            initializer = items[0]
        if static:
            stored = self.static_value(initializer, ctype, token, "the initializer")
        else:
            stored = self.assign_convert(
                self.rvalue(initializer), ctype, token, "the initializer"
            )
        bits = self.stored_bits(stored, ctype)
        if static and bits == IntConstant(0):
            return
        self.emit(HeapStore(base, bits, size_of(ctype)))

    def packed_unit(
        self,
        members: "list[Member]",
        position: int,
        items: "list[object]",
        index: int,
        token: Token,
    ) -> "tuple[IntExpression, int]":
        """One store's worth of bitfields, and how many values it used."""

        offset = members[position].offset
        combined: IntExpression = IntConstant(0)
        while position < len(members) and index < len(items):
            member = members[position]
            if member.width is None or member.offset != offset:
                break
            if member.name.startswith(_UNNAMED_BITFIELD):
                position += 1
                continue
            value = self.static_value(
                items[index], member.ctype, token, f"the value for {member.name!r}"
            )
            mask = (1 << member.width) - 1
            combined = _binary(
                "or",
                combined,
                _binary(
                    "lshift",
                    _binary("and", value, IntConstant(mask)),
                    IntConstant(member.bit),
                ),
            )
            position += 1
            index += 1
        return combined, index

    def struct_initializer(
        self,
        base: "IntExpression",
        ctype: StructType,
        initializer: object,
        token: Token,
        *,
        static: bool = False,
    ) -> None:
        """`struct P a = {1, 2};` - each value goes to the member it lines up with.

        In declaration order, which is what C says the braces mean without
        designators. A union takes one value, for its first member, because
        every member of one starts at the same place.
        """

        if ctype.members is None:
            self.error(f"{ctype} is not complete here", token)
        if isinstance(initializer, StringLiteral) or not isinstance(initializer, tuple):
            self.error(f"{ctype} needs a braced initializer", token)
        _brace, items = initializer
        members = list(ctype.members)
        if ctype.is_union:
            members = members[:1]
        named = [
            one for one in members if not one.name.startswith(_UNNAMED_BITFIELD)
        ]
        if len(items) > len(named):
            self.error(
                f"the initializer has {len(items)} values but {ctype} holds "
                f"{len(named)}",
                token,
            )
        index = 0
        for position, member in enumerate(members):
            if member.name.startswith(_UNNAMED_BITFIELD):
                # Padding: C gives it no value, so it takes none of the list.
                continue
            if index >= len(items):
                break
            where = (
                base
                if member.offset == 0
                else _binary("add", base, IntConstant(member.offset))
            )
            if member.width is not None:
                # Every bitfield sharing this unit is written by one store:
                # storing them one at a time would each write the whole unit
                # and the last would be the only one left.
                taken = self.packed_unit(members, position, items, index, token)
                self.emit(HeapStore(where, taken[0], size_of(member.ctype)))
                index = taken[1]
                continue
            self.aggregate_initializer(
                where, member.ctype, items[index], token, static=static
            )
            index += 1
        # C zero-fills what the braces leave out; a static object already is.
        if static or len(items) == len(members):
            return
        filled = members[len(items)].offset if items else 0
        if ctype.size > filled:
            self.emit_bytes(base, filled, b"\0" * (ctype.size - filled))

    def array_initializer(
        self,
        base: IntExpression,
        ctype: ArrayType,
        initializer: object,
        token: Token,
        *,
        static: bool = False,
    ) -> None:
        element = ctype.element
        size = size_of(element)
        if size is None:
            self.error(
                "an array of something with no size cannot be initialized; "
                "assign the elements instead",
                token,
            )
        if isinstance(initializer, StringLiteral):
            if not _string_fits(initializer, element, self.target):
                self.error(
                    f"a {initializer.kind or 'plain'} string literal cannot "
                    f"initialize an array of {element}",
                    token,
                )
            data = initializer.bytes_for(self.target)
            width = size_of(element) or 1
            if len(data) > ctype.count * width:
                self.error(
                    f"the initializer is {len(data)} bytes but the array holds "
                    f"{ctype.count * width}",
                    token,
                )
            self.emit_bytes(
                base,
                0,
                data + b"\0" * (ctype.count * width - len(data)),
                skip_zero=static,
            )
            return
        if not isinstance(initializer, tuple):
            self.error(
                "an array needs a braced initializer, not a single value", token
            )
        _brace, items = initializer
        if len(items) > ctype.count:
            self.error(
                f"the initializer has {len(items)} values but the array holds "
                f"{ctype.count}",
                token,
            )
        for position, item in enumerate(items):
            if isinstance(item, tuple) or isinstance(element, (StructType, ArrayType)):
                # An array of structs, or of arrays: each element is itself an
                # aggregate and is initialized the same way this one is.
                self.aggregate_initializer(
                    _binary("add", base, IntConstant(position * size)),
                    element,
                    item,
                    token,
                    static=static,
                )
                continue
            what = f"initializer element {position}"
            if static:
                stored = self.static_value(item, element, token, what)
            else:
                stored = self.assign_convert(
                    self.rvalue(item), element, token, what
                )
            bits = self.stored_bits(stored, element)
            if static and bits == IntConstant(0):
                # The static block arrives zero-filled from the kernel.
                continue
            self.emit(
                HeapStore(
                    _binary("add", base, IntConstant(position * size)),
                    bits,
                    size,
                )
            )
        # C zero-fills whatever the braces leave out. A static object is
        # already zero everywhere, so only an automatic one needs the fill.
        filled = len(items) * size
        if not static:
            self.emit_bytes(base, filled, b"\0" * (ctype.count * size - filled))

    def emit_bytes(
        self,
        base: IntExpression,
        offset: int,
        data: bytes,
        *,
        skip_zero: bool = False,
    ) -> None:
        """Store a constant byte image, using the widest aligned store each time.

        A byte-at-a-time fill of ``char page[2048] = {0}`` would be two thousand
        instructions. Both architectures py2bin emits for are little-endian, so
        a chunk of the image packs into one store in source order.

        ``skip_zero`` drops the stores that would write zero, which is what the
        static storage block already holds; it must not be used for an
        automatic object, whose bytes start out as whatever the frame held.
        """

        start = offset
        end = offset + len(data)
        while offset < end:
            for width in (8, 4, 2, 1):
                if offset % width == 0 and offset + width <= end:
                    chunk = data[offset - start : offset - start + width]
                    if skip_zero and not any(chunk):
                        offset += width
                        break
                    self.emit(
                        HeapStore(
                            _binary("add", base, IntConstant(offset)),
                            _constant(int.from_bytes(chunk, "little")),
                            width,
                        )
                    )
                    offset += width
                    break

    def if_statement(self, node: If) -> None:
        test = self.scalar(node.test, "an 'if' condition")
        otherwise = self.new_label("else")
        self.emit(JumpIfFalse(self.truth(test), otherwise))
        self.statement(node.body)
        if node.alternative is None:
            self.emit(Label(otherwise))
            return
        end = self.new_label("endif")
        self.emit(Jump(end))
        self.emit(Label(otherwise))
        self.statement(node.alternative)
        self.emit(Label(end))

    def while_statement(self, node: While) -> None:
        top = self.new_label("while")
        end = self.new_label("while_end")
        self.emit(Label(top))
        test = self.scalar(node.test, "a 'while' condition")
        self.emit(JumpIfFalse(self.truth(test), end))
        self.break_targets.append(end)
        self.continue_targets.append(top)
        self.statement(node.body)
        self.break_targets.pop()
        self.continue_targets.pop()
        self.emit(Jump(top))
        self.emit(Label(end))

    def do_while_statement(self, node: DoWhile) -> None:
        top = self.new_label("do")
        again = self.new_label("do_test")
        end = self.new_label("do_end")
        self.emit(Label(top))
        self.break_targets.append(end)
        self.continue_targets.append(again)
        self.statement(node.body)
        self.break_targets.pop()
        self.continue_targets.pop()
        self.emit(Label(again))
        test = self.scalar(node.test, "a 'do while' condition")
        self.emit(JumpIfFalse(self.truth(test), end))
        self.emit(Jump(top))
        self.emit(Label(end))

    def for_statement(self, node: For) -> None:
        """A C ``for`` is a ``while`` with an initializer and a step.

        It is emphatically not Python's ``for``: the initializer runs even when
        the body never does, the body may change the counter, and the counter
        keeps the first value that failed the test.
        """

        self.scopes.append({})  # C99 scope for a declaration in the initializer
        if node.initializer is not None:
            self.statement(node.initializer)
        top = self.new_label("for")
        step = self.new_label("for_step")
        end = self.new_label("for_end")
        self.emit(Label(top))
        if node.test is not None:
            test = self.scalar(node.test, "a 'for' condition")
            self.emit(JumpIfFalse(self.truth(test), end))
        self.break_targets.append(end)
        self.continue_targets.append(step)
        self.statement(node.body)
        self.break_targets.pop()
        self.continue_targets.pop()
        self.emit(Label(step))
        if node.step is not None:
            self.expression_statement(node.step)
        self.emit(Jump(top))
        self.emit(Label(end))
        self.scopes.pop()

    def switch_statement(self, node: Switch) -> None:
        control = self.rvalue(node.control)
        if not is_integer(control.ctype):
            self.error(
                f"a 'switch' needs an integer control expression, not "
                f"{control.ctype}",
                node.token,
            )
        ctype = promote(control.ctype)
        slot = self.new_temp()
        self.emit(Store(slot, self.fit(control.expr, ctype)))
        mark = len(self.operations)
        end = self.new_label("switch_end")
        context = SwitchContext(ctype, [], None, set())
        self.switches.append(context)
        self.break_targets.append(end)
        self.statement(node.body)
        self.break_targets.pop()
        self.switches.pop()
        dispatch: list[Operation] = [
            JumpIfFalse(IntCompare("ne", IntLoad(slot), IntConstant(value)), label)
            for value, label in context.cases
        ]
        dispatch.append(Jump(context.default or end))
        self.operations[mark:mark] = dispatch
        self.emit(Label(end))

    def labeled_statement(self, node: Labeled) -> None:
        if node.kind == "label":
            context = self.functions[-1]
            name = str(node.value)
            if name in context.defined:
                self.error(f"label {name!r} is defined twice", node.token)
            context.defined.add(name)
            target = context.labels.setdefault(name, self.new_label(f"goto_{name}"))
            self.emit(Label(target))
        elif node.kind == "case":
            if not self.switches:
                self.error("'case' is not inside a switch", node.token)
            context = self.switches[-1]
            raw = ConstantEvaluator(self.filename, self.enumerators).value(node.value)
            value = _s64(_wrap(raw, context.ctype.size, context.ctype.signed))
            if value in context.seen:
                self.error(f"duplicate case value {raw}", node.token)
            context.seen.add(value)
            label = self.new_label("case")
            context.cases.append((value, label))
            self.emit(Label(label))
        else:
            if not self.switches:
                self.error("'default' is not inside a switch", node.token)
            context = self.switches[-1]
            if context.default is not None:
                self.error("a switch has at most one 'default'", node.token)
            label = self.new_label("default")
            context.default = label
            self.emit(Label(label))
        if node.statement is not None:
            self.statement(node.statement)

    def return_statement(self, node: Return) -> None:
        context = self.functions[-1]
        if node.value is None:
            if not isinstance(context.function.result, VoidType):
                self.error(
                    f"{context.function.name}() returns "
                    f"{context.function.result}, so 'return' needs a value",
                    node.token,
                )
            if context.call_body:
                self.emit(IRReturn(None))
                return
            self.emit(Jump(context.return_label))
            return
        if isinstance(context.function.result, VoidType):
            self.error(
                f"{context.function.name}() returns void and cannot return a value",
                node.token,
            )
        value = self.rvalue(node.value)
        result = context.function.result
        stored = self.assign_convert(value, result, node.token, "this 'return'")
        if context.is_main:
            self.emit(ExitValue(stored))
            return
        # A floating result travels back in the integer result register as the
        # bit pattern of the declared result type, which is what direct_call
        # and inline() both read it back as.
        stored = self.stored_bits(stored, result)
        if context.call_body:
            self.emit(IRReturn(stored))
            return
        assert context.result_slot is not None
        self.emit(Store(context.result_slot, stored))
        self.emit(Jump(context.return_label))

    # --- printf ---

    #: Where a conversion is built. Wide enough for the longest number plus
    #: the widest field a program may pad it to, because the padding is
    #: written into the same buffer, in front of what it pads.
    _BUFFER_BYTES = 32 + _MAXIMUM_FIELD

    def print_buffer(self) -> int:
        if self.buffer_slot is None:
            self.buffer_slot = self.allocate(self._BUFFER_BYTES)
        return self.buffer_slot

    def put(self, payload: bytes) -> None:
        """Write literal bytes: to stdout, or into the buffer a sink names.

        Into a buffer they go one at a time. The text between conversions in
        a format is short, and so is the widest field, so a store apiece is
        smaller than the loop and the address it would need.
        """

        if self.sink is None:
            self.emit(Write(payload))
            return
        for byte in payload:
            self.put_byte(IntConstant(byte))

    def put_byte(self, character: IntExpression) -> None:
        """One byte into the sink, kept if it fits and counted either way."""

        buffer, limit, count = self.sink
        past = self.new_label("sink_byte_past")
        self.emit(
            JumpIfFalse(
                IntCompare(
                    "lt",
                    _binary("add", IntLoad(count), IntConstant(1)),
                    IntLoad(limit),
                ),
                past,
            )
        )
        self.emit(
            HeapStore(
                _binary(
                    "add",
                    IntLoad(buffer),
                    _binary("mul", IntLoad(count), IntConstant(self.sink_width)),
                ),
                character,
                self.sink_width,
            )
        )
        self.emit(Label(past))
        self.emit(Store(count, _binary("add", IntLoad(count), IntConstant(1))))

    def put_runtime(
        self, address: IntExpression, length: IntExpression, read: int = 1
    ) -> None:
        """Write `length` bytes from `address`, wherever the output is going.

        Into a buffer, the copy stops at what the caller said it holds and the
        count keeps rising past it: `snprintf` answers the length it would
        have written, which is what lets a caller ask how much room to make.
        """

        if self.sink is None:
            self.emit(WriteRuntime(address, length))
            return
        buffer, limit, count = self.sink
        source = self.new_temp()
        left = self.new_temp()
        self.emit(Store(source, address))
        self.emit(Store(left, length))
        top = self.new_label("sink")
        end = self.new_label("sink_end")
        past = self.new_label("sink_past")
        self.emit(Label(top))
        self.emit(JumpIfFalse(IntCompare("gt", IntLoad(left), IntConstant(0)), end))
        # One short of the limit: the last byte of the buffer is the NUL.
        self.emit(
            JumpIfFalse(
                IntCompare(
                    "lt",
                    _binary("add", IntLoad(count), IntConstant(1)),
                    IntLoad(limit),
                ),
                past,
            )
        )
        self.emit(
            HeapStore(
                _binary(
                    "add",
                    IntLoad(buffer),
                    _binary("mul", IntLoad(count), IntConstant(self.sink_width)),
                ),
                HeapLoad(IntLoad(source), read),
                self.sink_width,
            )
        )
        self.emit(Label(past))
        self.emit(Store(count, _binary("add", IntLoad(count), IntConstant(1))))
        self.emit(Store(source, _binary("add", IntLoad(source), IntConstant(read))))
        # The loop below repeats from here.
        self.emit(Store(left, _binary("sub", IntLoad(left), IntConstant(1))))
        self.emit(Jump(top))
        self.emit(Label(end))

    def formatted_into(self, node: Call, bounded: bool) -> Value:
        """`sprintf` and `snprintf`: the same formatting, into a buffer.

        C answers the length that *would* have been written, so a caller can
        ask how much room to make; the copy stops at the room there is, and
        the terminator goes at the end of what was kept.
        """

        wide = node.name in _WIDE_BUFFER
        # `swprintf_s(buffer, L"...")` is C++'s array overload: the room is
        # the array's own size and is not written at the call. Told apart by
        # what sits second - a size, or the format itself.
        if wide and len(node.arguments) >= 2 and isinstance(
            node.arguments[1], StringLiteral
        ):
            bounded = False
        wanted = 3 if bounded else 2
        if len(node.arguments) < wanted:
            self.error(
                f"{node.name} needs a buffer{', a size' if bounded else ''} "
                f"and a format",
                node.token,
            )
        target = self.rvalue(node.arguments[0])
        if not isinstance(target.ctype, PointerType):
            self.error(
                f"{node.name} writes through a pointer, not {target.ctype}",
                node.token,
            )
        buffer = self.new_temp()
        self.emit(Store(buffer, target.expr))
        limit = self.new_temp()
        if bounded:
            room = self.rvalue(node.arguments[1])
            if not is_integer(room.ctype):
                self.error(
                    f"{node.name} needs a size, not {room.ctype}", node.token
                )
            self.emit(Store(limit, room.expr))
        else:
            # `sprintf` has no limit. Said as a number rather than as a
            # special case, so the copy is the same code either way. The
            # array overload does have one - the array's own length - and
            # saying so is the whole reason a program reaches for it.
            self.emit(Store(limit, IntConstant(self.room_in(node.arguments[0]))))
        count = self.new_temp()
        self.emit(Store(count, IntConstant(0)))
        held, self.sink = self.sink, (buffer, limit, count)
        was, self.sink_width = self.sink_width, (
            wchar_for(self.target).size if wide else 1
        )
        try:
            self.printf(
                Call(node.token, node.name, list(node.arguments[wanted - 1:]))
            )
        finally:
            self.sink = held
            self.sink_width = was
        # The terminator goes where the copy stopped, which is the shorter of
        # what was written and one less than the room. `snprintf(b, 0, ...)`
        # writes nothing at all, which is what C says a size of zero means.
        past = self.new_label("no_terminator")
        self.emit(JumpIfFalse(IntCompare("gt", IntLoad(limit), IntConstant(0)), past))
        stop = self.new_temp()
        self.emit(Store(stop, IntLoad(count)))
        fits = self.new_label("terminator_fits")
        self.emit(
            JumpIfFalse(
                IntCompare(
                    "gt",
                    IntLoad(count),
                    _binary("sub", IntLoad(limit), IntConstant(1)),
                ),
                fits,
            )
        )
        self.emit(Store(stop, _binary("sub", IntLoad(limit), IntConstant(1))))
        self.emit(Label(fits))
        width = wchar_for(self.target).size if wide else 1
        self.emit(
            HeapStore(
                _binary(
                    "add",
                    IntLoad(buffer),
                    _binary("mul", IntLoad(stop), IntConstant(width)),
                ),
                IntConstant(0),
                width,
            )
        )
        self.emit(Label(past))
        return Value(INT, IntLoad(count))

    def room_in(self, node: object) -> int:
        """How many characters the array being written to holds.

        Only where the buffer is a name that holds an array: that is what the
        overload taking no size is for, and anything else is refused rather
        than given a number nobody wrote.
        """

        if isinstance(node, Identifier):
            local = self.lookup(node.name)
            held = local.ctype if local is not None else None
            if isinstance(held, ArrayType) and held.count:
                return held.count
        self.error(
            "this form of the call takes its room from the array it writes "
            "to, and the first argument is not one; pass the count instead",
            getattr(node, "token", None),
        )
        return 0

    def printf(self, node: Call) -> None:
        if not node.arguments or not isinstance(node.arguments[0], StringLiteral):
            self.error(
                f"{node.name} needs a literal format string; py2bin reads the "
                "format at compile time and emits the formatting code itself",
                node.token,
            )
        segments = self.parse_format(node.arguments[0])
        arguments = node.arguments[1:]
        conversions = [payload for kind, payload in segments if kind != "text"]
        # A `%*d` wants two: the width and then the value.
        expected = sum(
            2 if one[4] is _FROM_AN_ARGUMENT else 1 for one in conversions
        )
        if expected != len(arguments):
            self.error(
                f"printf has {expected} conversion(s) but {len(arguments)} "
                "argument(s)",
                node.token,
            )
        # C11 6.5.2.2p10 puts a sequence point after every argument is
        # evaluated and before the call, so printf may produce no output until
        # all of its arguments have been computed. Evaluating them while
        # emitting the format text would let an argument that writes to stdout
        # interleave with the literal parts. Evaluate everything first, hold
        # each result in a slot, then emit the output.
        prepared: list[object] = []
        at = 0
        for style, ctype, precision, flags, width in conversions:
            if width is _FROM_AN_ARGUMENT:
                asked = arguments[at]
                at += 1
                given = self.rvalue(asked)
                if not is_integer(given.ctype):
                    self.error(
                        "a width given as '*' is read from an int argument, "
                        f"not from {given.ctype}",
                        asked.token,
                    )
                # Pinned in a slot like every other argument, so evaluating a
                # later one cannot change the width this conversion pads to.
                width = self.materialize(self.fit(given.expr, INT))
            argument = arguments[at]
            at += 1
            value = self.rvalue(argument)
            prepared.append((style, ctype, precision, value, argument, flags, width))
        held: list[object] = []
        for style, ctype, precision, value, argument, flags, width in prepared:
            if style in ("string", "wide_string"):
                held.append((style, precision, value, argument, flags, width))
                continue
            if style in _FLOAT_CONVERSIONS:
                if not is_floating(value.ctype):
                    self.error(
                        f"a %{style} conversion needs a floating value, not "
                        f"{value.ctype}; C's printf reads a double here, and an "
                        "integer argument is undefined -- write a cast if that is "
                        "what you meant",
                        argument.token,
                    )
                # materialize_float() pins the double in a slot for the same
                # reason the integer path pins its value: a later argument must
                # not be able to change what this one formats.
                held.append(
                    (
                        style,
                        precision,
                        self.materialize_float(value.expr),
                        argument,
                        flags,
                        width,
                    )
                )
                continue
            if not is_integer(value.ctype):
                self.error(
                    f"this conversion needs an integer, not {value.ctype}",
                    argument.token,
                )
            # materialize() pins the value in a slot, so evaluating a later
            # argument cannot change what this one reads.
            held.append(
                (
                    style,
                    precision,
                    self.materialize(self.fit(value.expr, ctype)),
                    argument,
                    flags,
                    width,
                )
            )

        index = 0
        for kind, payload in segments:
            if kind == "text":
                self.put(payload)
                continue
            style, precision, value, argument, flags, width = held[index]
            index += 1
            if style in _FLOAT_CONVERSIONS:
                self.emit_floating(value, style, precision, flags, width)
                continue
            if style in ("string", "wide_string"):
                if value.null or not isinstance(value.ctype, PointerType):
                    self.error(
                        "a %s conversion needs a character pointer", argument.token
                    )
                wanted = (
                    value.ctype.target == wchar_for(self.target)
                    if style == "wide_string"
                    else _is_character(value.ctype.target)
                )
                if not wanted:
                    self.error(
                        f"a %{'l' if style == 'wide_string' else ''}s "
                        f"conversion needs a "
                        f"{'wide ' if style == 'wide_string' else ''}character "
                        f"pointer, not {value.ctype}",
                        argument.token,
                    )
                self.emit_string(
                    value.expr,
                    flags,
                    width,
                    size_of(value.ctype.target) or 1
                    if style == "wide_string"
                    else 1,
                )
                continue
            expression = value
            if style == "char":
                self.emit_character(expression, flags, width)
            elif style == "signed":
                self.emit_number(
                    expression, signed=True, base=10, upper=False,
                    flags=flags, width=width,
                )
            elif style == "unsigned":
                self.emit_number(
                    expression, signed=False, base=10, upper=False,
                    flags=flags, width=width,
                )
            else:
                self.emit_number(
                    expression, signed=False, base=16, upper=style == "HEX",
                    flags=flags, width=width,
                )

    def parse_format(self, literal: StringLiteral) -> list[tuple[str, object]]:
        segments: list[tuple[str, object]] = []
        text = bytearray()
        data = literal.data
        position = 0
        while position < len(data):
            byte = data[position]
            if byte != 0x25:  # '%'
                text.append(byte)
                position += 1
                continue
            position += 1
            if position < len(data) and data[position] == 0x25:
                text.append(0x25)
                position += 1
                continue
            flags = ""
            while position < len(data) and chr(data[position]) in "-+ #0":
                flags += chr(data[position])
                position += 1
            width: object = 0
            if position < len(data) and chr(data[position]) == "*":
                # The width comes from an argument. The format is still read
                # while compiling - what is not known then is the number, and
                # the padding loops take a count worked out at run time
                # already, so the number is the only thing that has to wait.
                position += 1
                if "0" in flags:
                    self.error(
                        "a '0' flag with a width given as '*' is not "
                        "implemented; the zeros go between the sign and the "
                        "digits, and with the width unknown the sign has been "
                        "written by then. Write the width as a number, or "
                        "drop the '0'",
                        literal.token,
                    )
                width = _FROM_AN_ARGUMENT
            while position < len(data) and chr(data[position]).isdigit():
                width = width * 10 + (data[position] - 48)
                position += 1
            if isinstance(width, int) and width > _MAXIMUM_FIELD:
                self.error(
                    f"a field width of {width} is beyond the {_MAXIMUM_FIELD} "
                    f"py2bin implements; the formatter pads inside a fixed "
                    f"frame buffer",
                    literal.token,
                )
            if "#" in flags:
                self.error(
                    "the '#' flag is not implemented; py2bin emits the "
                    "formatting itself and writes no 0x or 0 prefix",
                    literal.token,
                )
            precision: int | None = None
            if position < len(data) and data[position] == 0x2E:  # '.'
                position += 1
                figures = ""
                while position < len(data) and chr(data[position]).isdigit():
                    figures += chr(data[position])
                    position += 1
                # C reads an omitted precision after the period as zero.
                precision = int(figures) if figures else 0
            length = ""
            while position < len(data) and chr(data[position]) in "hlzjtL":
                length += chr(data[position])
                position += 1
            if position >= len(data):
                self.error("printf format ends inside a conversion", literal.token)
            specifier = chr(data[position])
            position += 1
            if specifier in _FLOAT_CONVERSIONS:
                if length == "L":
                    self.error(
                        f"printf conversion %L{specifier} names a long double, "
                        "which py2bin's C compiler does not implement",
                        literal.token,
                    )
                if length not in {"", "l"}:
                    # C11 7.21.6.1: 'l' before a floating conversion has no
                    # effect, because the argument is a double either way.
                    self.error(
                        f"printf conversion %{length}{specifier} is not valid",
                        literal.token,
                    )
                if precision is None:
                    precision = 6
                if precision > _MAXIMUM_PRECISION:
                    self.error(
                        f"printf precision {precision} is beyond the "
                        f"{_MAXIMUM_PRECISION} py2bin implements; the formatter "
                        "writes into a fixed frame buffer",
                        literal.token,
                    )
                if text:
                    segments.append(("text", bytes(text)))
                    text.clear()
                segments.append(
                    ("conversion", (specifier, DOUBLE, precision, flags, width))
                )
                continue
            if specifier not in _CONVERSIONS:
                self.error(
                    f"printf conversion %{length}{specifier} is not implemented; "
                    "py2bin emits the formatting itself and supports "
                    "%d %i %u %x %X %c %s %f %F %e %E %g %G and %% with the "
                    "h/hh/l/ll/z length modifiers, and a precision on the "
                    "floating conversions (no flags or field widths)",
                    literal.token,
                )
            if precision is not None:
                self.error(
                    f"a precision on %{specifier} is not implemented; py2bin "
                    "implements one only on the floating conversions",
                    literal.token,
                )
            if length == "L":
                self.error(
                    f"printf conversion %L{specifier} is not valid", literal.token
                )
            style, default = _CONVERSIONS[specifier]
            table = _LENGTHS.get(length)
            if table is None:
                self.error(
                    f"printf length modifier {length!r} is not implemented",
                    literal.token,
                )
            if length and specifier not in table:
                self.error(
                    f"printf conversion %{length}{specifier} is not valid",
                    literal.token,
                )
            ctype = table.get(specifier, default) if length else default
            if ctype is _WIDE_STRING:
                ctype = PointerType(wchar_for(self.target))
                style = "wide_string"
            elif ctype is _WIDE_CHAR:
                ctype = wchar_for(self.target)
            if text:
                segments.append(("text", bytes(text)))
                text.clear()
            segments.append(("conversion", (style, ctype, None, flags, width)))
        if text:
            segments.append(("text", bytes(text)))
        return segments

    #: The working variables the floating formatter needs. They are allocated
    #: once per function body and REUSED by every conversion in it: each one is
    #: live only inside the formatter's own straight-line code, and a program
    #: printing a dozen doubles would otherwise need a dozen copies of them.
    _FLOAT_SCRATCH = (
        "argument",
        "mode",
        # The field a conversion is padded to, and how: 0 pads with spaces on
        # the left, 1 with spaces on the right, 2 with zeros after the sign.
        # `positive` is the character a `+` or a space flag asks for in front
        # of a number that has no sign of its own, or 0 for neither.
        "field",
        "align",
        "positive",
        "given",
        "figures_asked",
        "upper",
        "back",
        "bits",
        "sign",
        "exponent",
        "mantissa",
        "significand",
        "scale",
        "power",
        "length",
        "index",
        "carry",
        "term",
        "repeats",
        "step",
        "cut",
        "guard",
        "sticky",
        "keep",
        "roundup",
        "surviving",
        "written",
        "decimal_exponent",
        "count",
        "position",
        "digit",
        "form",
        "figures",
        "zero",
    )

    def float_buffers(self) -> tuple[int, int, dict[str, int]]:
        """The frame the floating formatter works in, allocated once."""

        if self.digit_slot is None:
            self.digit_slot = self.allocate(_DIGIT_BYTES)
            self.text_slot = self.allocate(_TEXT_BYTES)
            self.float_scratch = {
                # `allocate`, not `new_temp`: this is reached from whichever
                # statement first formats a float, and is then read by every
                # later one.
                name: self.allocate(8) for name in self._FLOAT_SCRATCH
            }
        assert self.text_slot is not None
        return self.digit_slot, self.text_slot, self.float_scratch

    def emit_floating(
        self,
        value: FloatExpression,
        style: str,
        precision: int,
        flags: str = "",
        width: int = 0,
    ) -> None:
        """Format one double for %f/%e/%g and write it.

        The formatter itself is emitted ONCE per function body and reached by a
        jump, because it is some hundreds of IR operations and a program that
        prints a table of numbers would otherwise carry a copy of it per
        conversion. The shape, precision and case are passed in slots, and a
        return identifier picks the site to jump back to -- a subroutine call
        built out of the jumps the IR has, since it has no indirect branch.
        """

        _digits, _text, scratch = self.float_buffers()
        if self.float_entry is None:
            self.emit_float_formatter()
        assert self.float_entry is not None
        kind = style.lower()
        self.emit(Store(scratch["argument"], FloatBits(value, 8)))
        self.emit(
            Store(scratch["mode"], IntConstant({"f": 0, "e": 1, "g": 2}[kind]))
        )
        self.emit(Store(scratch["given"], IntConstant(precision)))
        self.emit(Store(scratch["field"], IntConstant(width)))
        self.emit(
            Store(
                scratch["align"],
                IntConstant(1 if "-" in flags else (2 if "0" in flags else 0)),
            )
        )
        self.emit(
            Store(
                scratch["positive"],
                IntConstant(43 if "+" in flags else (32 if " " in flags else 0)),
            )
        )
        # C reads a %g precision of zero as one significant digit.
        self.emit(
            Store(scratch["figures_asked"], IntConstant(max(1, precision)))
        )
        self.emit(
            Store(
                scratch["upper"],
                IntConstant(1 if style in {"F", "E", "G"} else 0),
            )
        )
        identifier = len(self.float_returns)
        back = self.new_label("fp_back")
        self.emit(Store(scratch["back"], IntConstant(identifier)))
        self.emit(Jump(self.float_entry))
        self.emit(Label(back))
        self.float_returns.append((identifier, back))

    def emit_float_dispatch(self) -> None:
        """Close a body's shared formatter by returning to each of its sites.

        This goes after the body's own exit or return, so nothing falls into
        it, and every path into the formatter set ``back`` to one of the
        identifiers below.
        """

        if self.float_entry is None or not self.float_returns:
            return
        assert self.float_dispatch is not None
        back = self.float_scratch["back"]
        self.emit(Label(self.float_dispatch))
        for identifier, target in self.float_returns:
            self.emit(
                JumpIfFalse(
                    IntCompare("ne", IntLoad(back), IntConstant(identifier)),
                    target,
                )
            )
        # Unreachable: every site above stored one of those identifiers.
        self.emit(Jump(self.float_returns[-1][1]))

    def emit_float_formatter(self) -> None:
        """Emit the shared formatter, jumped over so it is only ever entered."""

        skip = self.new_label("fp_skip")
        self.emit(Jump(skip))
        self.float_entry = self.new_label("fp_formatter")
        self.float_dispatch = self.new_label("fp_dispatch")
        self.emit(Label(self.float_entry))
        self.float_formatter()
        self.emit(Jump(self.float_dispatch))
        self.emit(Label(skip))

    def float_formatter(self) -> None:
        """Format the double in the ``argument`` slot, with no library at all.

        The conversion is EXACT, not approximate. Every finite double is
        ``M * 2**E`` with M below 2**53, and every such number has a finite
        decimal expansion: for E >= 0 it is the integer ``M * 2**E``, and for
        E < 0 it is ``M * 5**-E`` scaled by ``10**E``. So the emitted code
        builds that expansion digit by digit in a base-10 array -- doubling it E
        times, or multiplying it by five -E times -- and then rounds the decimal
        digits themselves. No logarithm, no repeated division of the double, and
        no accumulated error: the digits printed are the digits the value has.

        The rounding is round-half-to-even ON THE EXACT VALUE, which is what
        C11 7.21.6.1p13 recommends and what makes %.0f of 0.5 print 0, of 1.5
        print 2, and of 2.5 print 2 again.

        The cost is a loop of up to 1074 multiplications over up to 767 digits
        for the smallest subnormals; a value near 1 needs a few dozen. That is
        the price of being exact without a bignum library, and it is paid only
        where a program actually prints a floating value.
        """

        digit_slot, text_slot, scratch = self.float_buffers()
        digits = SlotAddress(digit_slot)
        text = SlotAddress(text_slot)

        argument = scratch["argument"]
        mode = scratch["mode"]
        given = scratch["given"]
        significant = scratch["figures_asked"]
        uppercase = scratch["upper"]
        bits = scratch["bits"]
        sign = scratch["sign"]
        exponent = scratch["exponent"]
        mantissa = scratch["mantissa"]
        significand = scratch["significand"]
        scale = scratch["scale"]
        power = scratch["power"]
        length = scratch["length"]
        index = scratch["index"]
        carry = scratch["carry"]
        term = scratch["term"]
        repeats = scratch["repeats"]
        step = scratch["step"]
        cut = scratch["cut"]
        guard = scratch["guard"]
        sticky = scratch["sticky"]
        keep = scratch["keep"]
        roundup = scratch["roundup"]
        surviving = scratch["surviving"]
        written = scratch["written"]
        decimal_exponent = scratch["decimal_exponent"]
        count = scratch["count"]
        position = scratch["position"]
        digit = scratch["digit"]
        form = scratch["form"]
        figures = scratch["figures"]
        zero = scratch["zero"]

        def store(slot: int, expression: IntExpression) -> None:
            self.emit(Store(slot, expression))

        def label(name: str) -> str:
            return self.new_label(f"fp_{name}")

        def at(name: str) -> str:
            target = label(name)
            self.emit(Label(target))
            return target

        def put(character: IntExpression) -> None:
            """Append one byte to the output buffer."""

            self.emit(
                HeapStore(_binary("add", text, IntLoad(written)), character, 1)
            )
            store(written, _binary("add", IntLoad(written), IntConstant(1)))

        def put_word(data: bytes) -> None:
            """Append a word, lower-cased unless the conversion was uppercase.

            ASCII sets bit 5 on the lower-case letters, so one runtime OR turns
            the same constants into "inf"/"INF" and "nan"/"NAN".
            """

            fold = IntBinary(
                "mul",
                IntCompare("eq", IntLoad(uppercase), IntConstant(0)),
                IntConstant(0x20),
            )
            for byte in data:
                put(_binary("or", IntConstant(byte), fold))

        def unless(condition: IntExpression, target: str) -> None:
            """Jump to ``target`` when ``condition`` is false."""

            self.emit(JumpIfFalse(condition, target))

        def byte_at(where: IntExpression) -> IntExpression:
            return HeapLoad(where, 1, False)

        # --- take the value apart --------------------------------------
        store(bits, IntLoad(argument))
        store(sign, IntBinary("urshift", IntLoad(bits), IntConstant(63)))
        store(
            exponent,
            IntBinary(
                "and",
                IntBinary("urshift", IntLoad(bits), IntConstant(52)),
                IntConstant(0x7FF),
            ),
        )
        store(
            mantissa,
            IntBinary("and", IntLoad(bits), IntConstant((1 << 52) - 1)),
        )
        store(written, IntConstant(0))
        positive = label("positive")
        unless(IntLoad(sign), positive)
        put(IntConstant(0x2D))  # '-'
        self.emit(Label(positive))

        # Infinities and NaNs have no digits; C prints them as words.
        finite = label("finite")
        emit_text = label("emit")
        unless(IntCompare("eq", IntLoad(exponent), IntConstant(0x7FF)), finite)
        not_a_number = label("nan")
        unless(IntCompare("eq", IntLoad(mantissa), IntConstant(0)), not_a_number)
        put_word(b"INF")
        self.emit(Jump(emit_text))
        self.emit(Label(not_a_number))
        put_word(b"NAN")
        self.emit(Jump(emit_text))
        self.emit(Label(finite))

        # value == significand * 2**power, exactly.
        subnormal = label("subnormal")
        ready = label("ready")
        unless(IntLoad(exponent), subnormal)
        store(
            significand,
            IntBinary("add", IntLoad(mantissa), IntConstant(1 << 52)),
        )
        store(power, IntBinary("sub", IntLoad(exponent), IntConstant(1075)))
        self.emit(Jump(ready))
        self.emit(Label(subnormal))
        store(significand, IntLoad(mantissa))
        store(power, IntConstant(-1074))
        self.emit(Label(ready))

        # --- the exact decimal expansion -------------------------------
        # digits[0] is the LEAST significant digit, and the value is
        # sum(digits[i] * 10**i) * 10**-scale.
        store(zero, IntCompare("eq", IntLoad(significand), IntConstant(0)))
        store(length, IntConstant(0))
        split = at("split")
        split_end = label("split_end")
        unless(IntLoad(significand), split_end)
        self.emit(
            HeapStore(
                _binary("add", digits, IntLoad(length)),
                IntBinary("umod", IntLoad(significand), IntConstant(10)),
                1,
            )
        )
        store(length, _binary("add", IntLoad(length), IntConstant(1)))
        store(
            significand, IntBinary("udiv", IntLoad(significand), IntConstant(10))
        )
        self.emit(Jump(split))
        self.emit(Label(split_end))
        nonzero = label("nonzero")
        unless(IntCompare("eq", IntLoad(length), IntConstant(0)), nonzero)
        self.emit(HeapStore(digits, IntConstant(0), 1))
        store(length, IntConstant(1))
        self.emit(Label(nonzero))

        # A positive binary exponent doubles the integer; a negative one turns
        # 2**-n into 5**n over 10**n, which is where the scale comes from.
        nonnegative = label("nonnegative")
        scaled = label("scaled")
        # A zero has an exact expansion of 0 * 10**1074, but C gives it the
        # decimal exponent 0, so it takes no scaling at all -- which is also
        # what stops %e from printing "0.000000e-1074".
        store(scale, IntConstant(0))
        store(repeats, IntConstant(0))
        store(step, IntConstant(2))
        unless(IntCompare("eq", IntLoad(zero), IntConstant(0)), scaled)
        unless(IntCompare("lt", IntLoad(power), IntConstant(0)), nonnegative)
        store(step, IntConstant(5))
        store(repeats, IntUnary("neg", IntLoad(power)))
        store(scale, IntUnary("neg", IntLoad(power)))
        self.emit(Jump(scaled))
        self.emit(Label(nonnegative))
        store(step, IntConstant(2))
        store(repeats, IntLoad(power))
        store(scale, IntConstant(0))
        self.emit(Label(scaled))

        outer = at("scale")
        outer_end = label("scale_end")
        unless(IntLoad(repeats), outer_end)
        store(carry, IntConstant(0))
        store(index, IntConstant(0))
        inner = at("multiply")
        inner_end = label("multiply_end")
        unless(IntCompare("lt", IntLoad(index), IntLoad(length)), inner_end)
        store(
            term,
            IntBinary(
                "add",
                IntBinary(
                    "mul",
                    byte_at(_binary("add", digits, IntLoad(index))),
                    IntLoad(step),
                ),
                IntLoad(carry),
            ),
        )
        self.emit(
            HeapStore(
                _binary("add", digits, IntLoad(index)),
                IntBinary("umod", IntLoad(term), IntConstant(10)),
                1,
            )
        )
        store(carry, IntBinary("udiv", IntLoad(term), IntConstant(10)))
        store(index, _binary("add", IntLoad(index), IntConstant(1)))
        self.emit(Jump(inner))
        self.emit(Label(inner_end))
        spill = at("spill")
        spill_end = label("spill_end")
        unless(IntLoad(carry), spill_end)
        self.emit(
            HeapStore(
                _binary("add", digits, IntLoad(length)),
                IntBinary("umod", IntLoad(carry), IntConstant(10)),
                1,
            )
        )
        store(length, _binary("add", IntLoad(length), IntConstant(1)))
        store(carry, IntBinary("udiv", IntLoad(carry), IntConstant(10)))
        self.emit(Jump(spill))
        self.emit(Label(spill_end))
        store(repeats, _binary("sub", IntLoad(repeats), IntConstant(1)))
        self.emit(Jump(outer))
        self.emit(Label(outer_end))

        # --- round the decimal digits ----------------------------------
        # 'cut' is how many of the least significant digits are dropped. %f
        # keeps a fixed number of them after the point; %e and %g keep a fixed
        # number of significant digits.
        fixed_cut = label("fixed_cut")
        exponent_cut = label("exponent_cut")
        chose_cut = label("chose_cut")
        unless(IntCompare("eq", IntLoad(mode), IntConstant(0)), exponent_cut)
        self.emit(Label(fixed_cut))
        store(cut, _binary("sub", IntLoad(scale), IntLoad(given)))
        self.emit(Jump(chose_cut))
        self.emit(Label(exponent_cut))
        unless(IntCompare("eq", IntLoad(mode), IntConstant(1)), general_cut := label("general_cut"))
        store(
            cut,
            _binary(
                "sub",
                _binary("sub", IntLoad(length), IntLoad(given)),
                IntConstant(1),
            ),
        )
        self.emit(Jump(chose_cut))
        self.emit(Label(general_cut))
        store(cut, _binary("sub", IntLoad(length), IntLoad(significant)))
        self.emit(Label(chose_cut))
        exact = label("exact")
        unless(IntCompare("gt", IntLoad(cut), IntConstant(0)), exact)

        store(guard, IntConstant(0))
        have_guard = label("have_guard")
        unless(
            IntCompare(
                "lt", _binary("sub", IntLoad(cut), IntConstant(1)), IntLoad(length)
            ),
            have_guard,
        )
        store(
            guard,
            byte_at(
                _binary(
                    "add", digits, _binary("sub", IntLoad(cut), IntConstant(1))
                )
            ),
        )
        self.emit(Label(have_guard))

        # 'sticky' says whether anything below the guard digit was nonzero,
        # which is what separates an exact tie from a value just above one.
        store(sticky, IntConstant(0))
        store(index, IntConstant(0))
        tail = at("tail")
        tail_end = label("tail_end")
        unless(
            IntCompare(
                "lt", IntLoad(index), _binary("sub", IntLoad(cut), IntConstant(1))
            ),
            tail_end,
        )
        unless(IntCompare("lt", IntLoad(index), IntLoad(length)), tail_end)
        store(
            sticky,
            IntBinary(
                "or",
                IntLoad(sticky),
                IntCompare(
                    "ne",
                    byte_at(_binary("add", digits, IntLoad(index))),
                    IntConstant(0),
                ),
            ),
        )
        store(index, _binary("add", IntLoad(index), IntConstant(1)))
        self.emit(Jump(tail))
        self.emit(Label(tail_end))

        store(keep, IntConstant(0))
        have_keep = label("have_keep")
        unless(IntCompare("lt", IntLoad(cut), IntLoad(length)), have_keep)
        store(keep, byte_at(_binary("add", digits, IntLoad(cut))))
        self.emit(Label(have_keep))
        # Round half to even: up when the guard is above five, and on an exact
        # five only when something below it was set or the surviving digit is
        # odd.
        store(
            roundup,
            IntBinary(
                "or",
                IntCompare("gt", IntLoad(guard), IntConstant(5)),
                IntBinary(
                    "and",
                    IntCompare("eq", IntLoad(guard), IntConstant(5)),
                    IntBinary(
                        "or",
                        IntCompare("ne", IntLoad(sticky), IntConstant(0)),
                        IntBinary("and", IntLoad(keep), IntConstant(1)),
                    ),
                ),
            ),
        )

        store(surviving, _binary("sub", IntLoad(length), IntLoad(cut)))
        nonempty = label("nonempty")
        unless(IntCompare("lt", IntLoad(surviving), IntConstant(0)), nonempty)
        store(surviving, IntConstant(0))
        self.emit(Label(nonempty))
        store(index, IntConstant(0))
        shift = at("shift")
        shift_end = label("shift_end")
        unless(IntCompare("lt", IntLoad(index), IntLoad(surviving)), shift_end)
        self.emit(
            HeapStore(
                _binary("add", digits, IntLoad(index)),
                byte_at(
                    _binary(
                        "add", digits, _binary("add", IntLoad(index), IntLoad(cut))
                    )
                ),
                1,
            )
        )
        store(index, _binary("add", IntLoad(index), IntConstant(1)))
        self.emit(Jump(shift))
        self.emit(Label(shift_end))
        store(scale, _binary("sub", IntLoad(scale), IntLoad(cut)))
        store(length, IntLoad(surviving))
        survived = label("survived")
        unless(IntCompare("eq", IntLoad(length), IntConstant(0)), survived)
        self.emit(HeapStore(digits, IntConstant(0), 1))
        store(length, IntConstant(1))
        self.emit(Label(survived))

        rounded = label("rounded")
        unless(IntLoad(roundup), rounded)
        store(carry, IntConstant(1))
        store(index, IntConstant(0))
        bump = at("bump")
        bump_end = label("bump_end")
        unless(IntCompare("lt", IntLoad(index), IntLoad(length)), bump_end)
        unless(IntLoad(carry), bump_end)
        store(
            term,
            IntBinary(
                "add",
                byte_at(_binary("add", digits, IntLoad(index))),
                IntLoad(carry),
            ),
        )
        self.emit(
            HeapStore(
                _binary("add", digits, IntLoad(index)),
                IntBinary("umod", IntLoad(term), IntConstant(10)),
                1,
            )
        )
        store(carry, IntBinary("udiv", IntLoad(term), IntConstant(10)))
        store(index, _binary("add", IntLoad(index), IntConstant(1)))
        self.emit(Jump(bump))
        self.emit(Label(bump_end))
        settled = label("settled")
        unless(IntLoad(carry), settled)
        self.emit(
            HeapStore(_binary("add", digits, IntLoad(length)), IntLoad(carry), 1)
        )
        store(length, _binary("add", IntLoad(length), IntConstant(1)))
        self.emit(Label(settled))
        self.emit(Label(rounded))
        self.emit(Label(exact))

        # --- pick the shape and the number of fraction digits ----------
        store(
            decimal_exponent,
            _binary(
                "sub",
                _binary("sub", IntLoad(length), IntConstant(1)),
                IntLoad(scale),
            ),
        )
        chosen = label("chosen")
        general = label("general")
        unless(IntCompare("ne", IntLoad(mode), IntConstant(2)), general)
        # %f and %e keep the shape and precision they were written with.
        store(form, IntLoad(mode))
        store(figures, IntLoad(given))
        self.emit(Jump(chosen))
        self.emit(Label(general))
        # C11 7.21.6.1p8: %g uses %e when the exponent is below -4 or at least
        # the precision, and %f otherwise, then drops trailing zeros.
        fixed_form = label("fixed_form")
        unless(
            IntBinary(
                "or",
                IntCompare("lt", IntLoad(decimal_exponent), IntConstant(-4)),
                IntCompare("ge", IntLoad(decimal_exponent), IntLoad(significant)),
            ),
            fixed_form,
        )
        store(form, IntConstant(1))
        store(figures, _binary("sub", IntLoad(significant), IntConstant(1)))
        self.emit(Jump(chosen))
        self.emit(Label(fixed_form))
        store(form, IntConstant(0))
        store(
            figures,
            _binary(
                "sub",
                _binary("sub", IntLoad(significant), IntConstant(1)),
                IntLoad(decimal_exponent),
            ),
        )
        self.emit(Label(chosen))

        def fraction(index_of: object) -> None:
            """Emit '.' and the fraction digits, reading 0 outside the array."""

            done = label("fraction_done")
            unless(IntCompare("gt", IntLoad(figures), IntConstant(0)), done)
            put(IntConstant(0x2E))  # '.'
            store(count, IntConstant(1))
            top = at("fraction")
            stop = label("fraction_end")
            unless(IntCompare("le", IntLoad(count), IntLoad(figures)), stop)
            store(position, index_of())
            store(digit, IntConstant(0))
            outside = label("outside")
            unless(IntCompare("ge", IntLoad(position), IntConstant(0)), outside)
            unless(IntCompare("lt", IntLoad(position), IntLoad(length)), outside)
            store(digit, byte_at(_binary("add", digits, IntLoad(position))))
            self.emit(Label(outside))
            put(_binary("add", IntLoad(digit), IntConstant(48)))
            store(count, _binary("add", IntLoad(count), IntConstant(1)))
            self.emit(Jump(top))
            self.emit(Label(stop))
            self.emit(Label(done))

        exponential = label("exponential")
        after = label("after_mantissa")
        unless(IntCompare("eq", IntLoad(form), IntConstant(0)), exponential)
        # Fixed form: digits at or above the point, then the fraction.
        empty = label("no_integer")
        integer_done = label("integer_done")
        unless(
            IntCompare(
                "ge",
                _binary("sub", IntLoad(length), IntConstant(1)),
                IntLoad(scale),
            ),
            empty,
        )
        store(index, _binary("sub", IntLoad(length), IntConstant(1)))
        walk = at("integer")
        walk_end = label("integer_end")
        unless(IntCompare("ge", IntLoad(index), IntLoad(scale)), walk_end)
        put(
            _binary(
                "add",
                byte_at(_binary("add", digits, IntLoad(index))),
                IntConstant(48),
            )
        )
        store(index, _binary("sub", IntLoad(index), IntConstant(1)))
        self.emit(Jump(walk))
        self.emit(Label(walk_end))
        self.emit(Jump(integer_done))
        self.emit(Label(empty))
        put(IntConstant(48))  # a value below one still shows its leading '0'
        self.emit(Label(integer_done))
        fraction(lambda: _binary("sub", IntLoad(scale), IntLoad(count)))
        self.emit(Jump(after))
        self.emit(Label(exponential))
        put(
            _binary(
                "add",
                byte_at(
                    _binary(
                        "add", digits, _binary("sub", IntLoad(length), IntConstant(1))
                    )
                ),
                IntConstant(48),
            )
        )
        fraction(
            lambda: _binary(
                "sub",
                _binary("sub", IntLoad(length), IntConstant(1)),
                IntLoad(count),
            )
        )
        self.emit(Label(after))

        # %g drops the trailing zeros, and the point when nothing follows
        # it. There is always a digit before the point, so this cannot run
        # off the front of the buffer.
        kept = label("kept")
        unless(IntCompare("eq", IntLoad(mode), IntConstant(2)), kept)
        unless(IntCompare("gt", IntLoad(figures), IntConstant(0)), kept)
        zeros = at("zeros")
        zeros_end = label("zeros_end")
        unless(
            IntCompare(
                "eq",
                byte_at(
                    _binary(
                        "add", text, _binary("sub", IntLoad(written), IntConstant(1))
                    )
                ),
                IntConstant(48),
            ),
            zeros_end,
        )
        store(written, _binary("sub", IntLoad(written), IntConstant(1)))
        self.emit(Jump(zeros))
        self.emit(Label(zeros_end))
        point = label("point")
        unless(
            IntCompare(
                "eq",
                byte_at(
                    _binary(
                        "add", text, _binary("sub", IntLoad(written), IntConstant(1))
                    )
                ),
                IntConstant(0x2E),
            ),
            point,
        )
        store(written, _binary("sub", IntLoad(written), IntConstant(1)))
        self.emit(Label(point))
        self.emit(Label(kept))

        # The exponent suffix, on the exponential form only. C requires at
        # least two digits, and a double never needs more than three.
        plain = label("plain")
        unless(IntLoad(form), plain)
        put_word(b"E")  # 'E' or 'e', by the conversion's case
        above = label("above")
        signed_done = label("exponent_signed")
        unless(
            IntCompare("lt", IntLoad(decimal_exponent), IntConstant(0)), above
        )
        put(IntConstant(0x2D))  # '-'
        store(count, IntUnary("neg", IntLoad(decimal_exponent)))
        self.emit(Jump(signed_done))
        self.emit(Label(above))
        put(IntConstant(0x2B))  # '+'
        store(count, IntLoad(decimal_exponent))
        self.emit(Label(signed_done))
        two_digits = label("two_digits")
        unless(IntCompare("ge", IntLoad(count), IntConstant(100)), two_digits)
        put(
            _binary(
                "add",
                IntBinary("udiv", IntLoad(count), IntConstant(100)),
                IntConstant(48),
            )
        )
        store(count, IntBinary("umod", IntLoad(count), IntConstant(100)))
        self.emit(Label(two_digits))
        put(
            _binary(
                "add",
                IntBinary("udiv", IntLoad(count), IntConstant(10)),
                IntConstant(48),
            )
        )
        put(
            _binary(
                "add",
                IntBinary("umod", IntLoad(count), IntConstant(10)),
                IntConstant(48),
            )
        )
        self.emit(Label(plain))

        self.emit(Label(emit_text))
        self.emit_float_field(text, written)

    def emit_float_field(self, text: IntExpression, written: int) -> None:
        """Write the formatted number, with the sign and padding it was given.

        Three shapes of padding, and the zero-padded one is why this is not
        simply a run of spaces: `%08.2f` of -3.14 is `-0003.14`, so the sign
        goes first and the zeros between it and the digits. A `+` or a space
        flag puts a character in front of a number the formatter wrote none
        for, which is one more column the field has to hold.
        """

        scratch = self.float_scratch
        field, align = scratch["field"], scratch["align"]
        positive = scratch["positive"]

        # Whether the text already begins with a sign, and whether a flag
        # asks for one it does not have.
        negative = self.new_temp()
        self.emit(
            Store(negative, IntCompare("eq", HeapLoad(text, 1), IntConstant(45)))
        )
        extra = self.new_temp()
        self.emit(
            Store(
                extra,
                _binary(
                    "and",
                    IntCompare("ne", IntLoad(positive), IntConstant(0)),
                    IntCompare("eq", IntLoad(negative), IntConstant(0)),
                ),
            )
        )
        total = self.new_temp()
        self.emit(Store(total, _binary("add", IntLoad(written), IntLoad(extra))))
        wanted = _binary("sub", IntLoad(field), IntLoad(total))

        def write_flag() -> None:
            """The `+` or space, where the number carries no sign of its own."""

            skip = self.new_label("fp_no_flag")
            self.emit(JumpIfFalse(IntLoad(extra), skip))
            self.pad_runtime(IntConstant(1), IntLoad(positive))
            self.emit(Label(skip))

        plain = self.new_label("fp_no_field")
        left = self.new_label("fp_left")
        zeros = self.new_label("fp_zeros")
        done = self.new_label("fp_field_done")
        self.emit(
            JumpIfFalse(IntCompare("gt", IntLoad(field), IntLoad(total)), plain)
        )
        onward = self.new_label("fp_not_left")
        self.emit(JumpIfFalse(IntCompare("eq", IntLoad(align), IntConstant(1)), onward))
        self.emit(Jump(left))
        self.emit(Label(onward))
        after = self.new_label("fp_not_zero")
        self.emit(JumpIfFalse(IntCompare("eq", IntLoad(align), IntConstant(2)), after))
        self.emit(Jump(zeros))
        self.emit(Label(after))
        # Right-aligned with spaces, which is what a bare width means.
        self.pad_runtime(wanted, IntConstant(32))
        write_flag()
        self.put_runtime(text, IntLoad(written))
        self.emit(Jump(done))

        self.emit(Label(left))
        write_flag()
        self.put_runtime(text, IntLoad(written))
        self.pad_runtime(wanted, IntConstant(32))
        self.emit(Jump(done))

        self.emit(Label(zeros))
        write_flag()
        bare = self.new_label("fp_zero_bare")
        self.emit(JumpIfFalse(IntLoad(negative), bare))
        self.put_runtime(text, IntConstant(1))
        self.pad_runtime(wanted, IntConstant(48))
        self.put_runtime(
            _binary("add", text, IntConstant(1)),
            _binary("sub", IntLoad(written), IntConstant(1)),
        )
        self.emit(Jump(done))
        self.emit(Label(bare))
        self.pad_runtime(wanted, IntConstant(48))
        self.put_runtime(text, IntLoad(written))
        self.emit(Jump(done))

        self.emit(Label(plain))
        write_flag()
        self.put_runtime(text, IntLoad(written))
        self.emit(Label(done))

    def pad_runtime(self, count: IntExpression, character: IntExpression) -> None:
        """Write `count` copies of a character, where the count is not known."""

        base = SlotAddress(self.print_buffer())
        left = self.new_temp()
        self.emit(Store(left, count))
        self.emit(HeapStore(base, character, 1))
        top = self.new_label("pad_run")
        end = self.new_label("pad_run_end")
        self.emit(Label(top))
        self.emit(JumpIfFalse(IntCompare("gt", IntLoad(left), IntConstant(0)), end))
        self.put_runtime(base, IntConstant(1))
        self.emit(Store(left, _binary("sub", IntLoad(left), IntConstant(1))))
        self.emit(Jump(top))
        self.emit(Label(end))

    def emit_character(
        self, expression: IntExpression, flags: str = "", width: object = 0
    ) -> None:
        base = SlotAddress(self.print_buffer())
        if not isinstance(width, int):
            # The character is stored after the padding and not before it:
            # `pad_runtime` writes its own character into the same first byte
            # of this buffer, so a character put there first is padded over.
            held = self.materialize(expression)
            if "-" not in flags:
                self.pad_runtime(
                    _binary("sub", width, IntConstant(1)), IntConstant(32)
                )
            self.emit(HeapStore(base, held, 1))
            self.put_runtime(base, IntConstant(1))
            if "-" in flags:
                self.pad_runtime(
                    _binary("sub", width, IntConstant(1)), IntConstant(32)
                )
            else:
                self.pad_runtime(
                    _binary("sub", _negated(width), IntConstant(1)), IntConstant(32)
                )
            return
        self.emit(HeapStore(base, expression, 1))
        if width > 1 and "-" not in flags:
            self.put(b" " * (width - 1))
        self.put_runtime(base, IntConstant(1))
        if width > 1 and "-" in flags:
            self.put(b" " * (width - 1))

    def emit_string(
        self,
        pointer: IntExpression,
        flags: str = "",
        width: object = 0,
        read: int = 1,
    ) -> None:
        """Write a string out. `read` is how wide one of its characters is."""

        pointer_slot = self.new_temp()
        length_slot = self.new_temp()
        self.emit(Store(pointer_slot, pointer))
        self.emit(Store(length_slot, IntConstant(0)))
        top = self.new_label("strlen")
        end = self.new_label("strlen_end")
        self.emit(Label(top))
        self.emit(
            JumpIfFalse(
                HeapLoad(
                    _binary(
                        "add",
                        IntLoad(pointer_slot),
                        _binary("mul", IntLoad(length_slot), IntConstant(read)),
                    ),
                    read,
                ),
                end,
            )
        )
        self.emit(
            Store(length_slot, _binary("add", IntLoad(length_slot), IntConstant(1)))
        )
        self.emit(Jump(top))
        self.emit(Label(end))
        if width and "-" not in flags:
            self.pad_forward(width, IntLoad(length_slot))
        self.put_runtime(IntLoad(pointer_slot), IntLoad(length_slot), read)
        if width and "-" in flags:
            self.pad_forward(width, IntLoad(length_slot))
        elif not isinstance(width, int):
            self.pad_forward(_negated(width), IntLoad(length_slot))

    def emit_sign(
        self,
        buffer: IntExpression,
        index_slot: int,
        negative_slot: int,
        flags: str,
    ) -> None:
        """Put `-`, or the `+`/space a flag asked for, in front of the digits."""

        done = self.new_label("no_sign")
        if "+" in flags or " " in flags:
            # Always one character, so there is no branch about whether to
            # write one - only about which.
            self.emit(
                Store(index_slot, _binary("sub", IntLoad(index_slot), IntConstant(1)))
            )
            positive = 43 if "+" in flags else 32
            self.emit(
                HeapStore(
                    _binary("add", buffer, IntLoad(index_slot)),
                    _binary(
                        "add",
                        IntConstant(positive),
                        _binary(
                            "mul", IntLoad(negative_slot), IntConstant(45 - positive)
                        ),
                    ),
                    1,
                )
            )
            return
        self.emit(JumpIfFalse(IntLoad(negative_slot), done))
        self.emit(
            Store(index_slot, _binary("sub", IntLoad(index_slot), IntConstant(1)))
        )
        self.emit(
            HeapStore(
                _binary("add", buffer, IntLoad(index_slot)), IntConstant(45), 1
            )
        )
        self.emit(Label(done))

    def pad_back(
        self,
        buffer: IntExpression,
        index_slot: int,
        width: int,
        character: int,
        signed: bool,
        negative_slot: "int | None",
    ) -> None:
        """Fill backwards from the digits until the field is `width` wide.

        Backwards because that is where the buffer has room: the number was
        built from the end, so what goes in front of it goes in front of the
        index, which is exactly where padding belongs.
        """

        room = width
        if signed and negative_slot is not None:
            # A `-` still to be written takes one of the field's columns.
            room_slot = self.new_temp()
            self.emit(
                Store(
                    room_slot,
                    _binary("sub", IntConstant(width), IntLoad(negative_slot)),
                )
            )
            wanted: IntExpression = IntLoad(room_slot)
        else:
            wanted = IntConstant(room)
        top = self.new_label("pad")
        end = self.new_label("pad_end")
        self.emit(Label(top))
        self.emit(
            JumpIfFalse(
                IntCompare(
                    "lt",
                    _binary(
                        "sub", IntConstant(self._BUFFER_BYTES), IntLoad(index_slot)
                    ),
                    wanted,
                ),
                end,
            )
        )
        self.emit(
            Store(index_slot, _binary("sub", IntLoad(index_slot), IntConstant(1)))
        )
        self.emit(
            HeapStore(
                _binary("add", buffer, IntLoad(index_slot)),
                IntConstant(character),
                1,
            )
        )
        self.emit(Jump(top))
        self.emit(Label(end))

    def pad_forward(self, width: object, written: IntExpression) -> None:
        """Write spaces after what was emitted, for a left-aligned field."""

        left = self.new_temp()
        self.emit(Store(left, _binary("sub", _width_of(width), written)))
        top = self.new_label("pad_right")
        end = self.new_label("pad_right_end")
        self.emit(Label(top))
        self.emit(
            JumpIfFalse(IntCompare("gt", IntLoad(left), IntConstant(0)), end)
        )
        self.put(b" ")
        self.emit(Store(left, _binary("sub", IntLoad(left), IntConstant(1))))
        self.emit(Jump(top))
        self.emit(Label(end))

    def emit_number(
        self,
        expression: IntExpression,
        *,
        signed: bool,
        base: int,
        upper: bool,
        flags: str = "",
        width: int = 0,
    ) -> None:
        """Format one integer into the scratch buffer and write it to stdout.

        Digits are produced least-significant first into the end of a 32-byte
        frame buffer, which is why the index walks backwards; 20 digits plus a
        sign is the widest a 64-bit value can be.
        """

        buffer = SlotAddress(self.print_buffer())
        value_slot = self.new_temp()
        index_slot = self.new_temp()
        negative_slot = self.new_temp()
        self.emit(Store(value_slot, expression))
        if signed:
            self.emit(
                Store(
                    negative_slot,
                    IntCompare("lt", IntLoad(value_slot), IntConstant(0)),
                )
            )
            # |v| without a branch, and correct for the most negative value,
            # whose magnitude is not representable as a signed 64-bit integer:
            # (v ^ (v>>63)) - (v>>63) leaves the 0x8000... bit pattern alone,
            # which is exactly 2**63 read as unsigned.
            sign = IntBinary("rshift", IntLoad(value_slot), IntConstant(63))
            self.emit(
                Store(
                    value_slot,
                    IntBinary("sub", IntBinary("xor", IntLoad(value_slot), sign), sign),
                )
            )
        self.emit(Store(index_slot, IntConstant(self._BUFFER_BYTES)))
        digit_slot = self.new_temp()
        top = self.new_label("digits")
        self.emit(Label(top))
        self.emit(
            Store(index_slot, _binary("sub", IntLoad(index_slot), IntConstant(1)))
        )
        self.emit(
            Store(
                digit_slot,
                IntBinary("umod", IntLoad(value_slot), IntConstant(base)),
            )
        )
        character = _binary("add", IntLoad(digit_slot), IntConstant(48))
        if base == 16:
            # 'a'-'0'-10 == 39, 'A'-'0'-10 == 7; the compare yields 1 or 0.
            character = _binary(
                "add",
                character,
                _binary(
                    "mul",
                    IntCompare("gt", IntLoad(digit_slot), IntConstant(9)),
                    IntConstant(7 if upper else 39),
                ),
            )
        self.emit(
            HeapStore(
                _binary("add", buffer, IntLoad(index_slot)), character, 1
            )
        )
        self.emit(
            Store(
                value_slot,
                IntBinary("udiv", IntLoad(value_slot), IntConstant(base)),
            )
        )
        self.emit(
            JumpIfFalse(
                IntCompare("eq", IntLoad(value_slot), IntConstant(0)), top
            )
        )
        # Zero padding goes between the sign and the digits, which is why it
        # is written before the sign and space padding after it.
        given = not isinstance(width, int)
        if width and not given and "0" in flags and "-" not in flags:
            room = width - (1 if signed and ("+" in flags or " " in flags) else 0)
            self.pad_back(buffer, index_slot, room, 48, signed, negative_slot)
        if signed:
            self.emit_sign(buffer, index_slot, negative_slot, flags)
        if width and not given and ("0" not in flags or "-" in flags):
            if "-" not in flags:
                self.pad_back(buffer, index_slot, width, 32, False, None)
        written = _binary(
            "sub", IntConstant(self._BUFFER_BYTES), IntLoad(index_slot)
        )
        if given and "-" not in flags:
            # In front of the digits and into the output rather than into the
            # buffer. The buffer is a fixed frame filled backwards, so a count
            # that is not known while compiling has no bound that can be
            # checked against it - and the output has no bound to check.
            length = self.new_temp()
            self.emit(Store(length, written))
            self.pad_runtime(
                _binary("sub", _width_of(width), IntLoad(length)), IntConstant(32)
            )
            written = IntLoad(length)
            self.put_runtime(_binary("add", buffer, IntLoad(index_slot)), written)
            # And the other side, for a width that came over negative.
            self.pad_forward(_negated(width), IntLoad(length))
            return
        self.put_runtime(_binary("add", buffer, IntLoad(index_slot)), written)
        if width and "-" in flags:
            self.pad_forward(
                width,
                _binary("sub", IntConstant(self._BUFFER_BYTES), IntLoad(index_slot)),
            )

    # --- entry point ---

    def compile(self) -> Module:
        main = self.unit.functions.get("main")
        if main is None:
            raise CCompileError(
                self.filename, 1, 1, "this translation unit has no int main(void)"
            )
        if main.result != INT or main.parameters:
            self.error(
                "the entry point must have the exact form int main(void)", main.token
            )
        if main.body is None:
            self.error(
                "main() is declared but never defined in this translation unit",
                main.token,
            )
        self.scopes.append({})
        # Before main's first statement: C11 5.1.2p1 gives every object with
        # static storage duration its initial value before the program starts.
        self.declare_globals()
        context = FunctionContext(main, None, self.new_label("return_main"), True)
        self.functions.append(context)
        self.active.append("main")
        self.block(main.body)
        self.active.pop()
        self.functions.pop()
        self.check_labels(context)
        self.emit(Label(context.return_label))
        # C99: falling off the end of main returns 0.
        self.emit(Exit(0))
        # After the exit, so nothing can fall into it.
        self.emit_float_dispatch()
        self.scopes.pop()
        return Module(
            self.operations,
            self.peak_slots,
            list(self.lowered.values()),
            static_bytes=self.static_bytes,
        )


def _is_character(ctype: CType) -> bool:
    return ctype in {CHAR, SCHAR, UCHAR}


def _string_fits(literal: "StringLiteral", element: CType, target: str) -> bool:
    """Whether this literal may initialise an array of that element type.

    A plain literal goes in a character array; a wide one goes in an array of
    the type its prefix names. Anything else is a mistake C reports and this
    reports too, rather than filling the array with the wrong width.
    """

    if isinstance(literal.data, bytes):
        return _is_character(element)
    return element == literal.element_for(target)


def compile_c_to_ir(
    source: str,
    filename: str,
    target: str,
    *,
    include_dirs: tuple[str, ...] = (),
    defines: tuple[str, ...] = (),
    libraries: tuple[str, ...] = (),
) -> Module:
    """Compile C source text to a py2bin native IR module.

    ``include_dirs`` is the ``-I`` search path the preprocessor uses, and
    ``defines`` the ``-D`` macros (``NAME`` or ``NAME=value``) it starts with.

    ``libraries`` names the shared libraries a function this program calls but
    never defines may be found in - what a build with a linker would name as
    an import library. py2bin knows the library behind every function it
    vets; it cannot know the one behind a component somebody else shipped, so
    the program says.
    """

    # Imported here rather than at the top because the preprocessor is written
    # in terms of this module's tokens and lexer, and would otherwise close an
    # import cycle.
    from .c_preprocessor import preprocess

    tokens = preprocess(
        source,
        filename,
        target=target,
        include_dirs=include_dirs,
        defines=defines,
    )
    parser = Parser(tokens, filename, target)
    parser.libraries = _libraries_named(libraries)
    unit = parser.translation_unit()
    module = Lowerer(unit, filename, target).compile()
    module.symbol_libraries = dict(parser.symbol_libraries)
    return module


def _libraries_named(
    libraries: "tuple[str, ...]",
) -> "list[tuple[str, frozenset[str]]]":
    """Read `--library` into (library, the symbols it is claimed for).

    `NAME` claims every function the program declares and never defines;
    `NAME:one,two` claims exactly those. The second form is what a program
    calling into two components needs, and the first is what almost every
    program needs.
    """

    found: "list[tuple[str, frozenset[str]]]" = []
    for spelled in libraries:
        library, _colon, named = spelled.partition(":")
        found.append(
            (
                library.strip(),
                frozenset(
                    one.strip() for one in named.split(",") if one.strip()
                ),
            )
        )
    return found
