"""py2bin's C compiler: C source text straight to py2bin's native IR.

This module is a real (small) C compiler written in pure Python. It lexes and
parses C, builds a typed syntax tree of its own, applies C's integer promotions
and conversions, and emits py2bin's native IR -- which the handwritten ARM64 and
x86-64 encoders turn into machine code. No external compiler, assembler, linker,
preprocessor or toolchain is involved, and no process is ever started.

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
* ``printf`` with real runtime formatting, and the vetted ``extern`` adapter
  ABI that lets compiled C drive an embedded CPython.

What is rejected
----------------
Floating point in C, ``struct``/``union``/``enum``/``typedef`` definitions,
function pointers, variadic user functions, globals with static storage, the
preprocessor beyond a handful of ignorable ``#include``s, more than eight
arguments to a function (py2bin passes arguments only in registers), and
recursion on the targets that have no call ABI yet.
"""

from __future__ import annotations

import dataclasses
import re

from .native.frontend import _CABI_RESULT_WIDTH, _CABI_RESULTS, _CABI_SYMBOLS
from .native.compiler import CALL_CAPABLE_TARGETS
from .native.ir import (
    Call as IRCall,
    CStringConstant,
    Exit,
    ExitValue,
    ExternCall,
    Function as IRFunction,
    HeapLoad,
    HeapStore,
    IntBinary,
    IntCompare,
    IntConstant,
    IntExpression,
    IntLoad,
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


_HEADERS = frozenset(
    {
        "stdio.h",
        "stdlib.h",
        "string.h",
        "stdint.h",
        "stddef.h",
        "limits.h",
        "math.h",
        "inttypes.h",
    }
)
_INCLUDE = re.compile(r"#\s*include\s*<\s*([A-Za-z0-9_./]+)\s*>\s*\Z")

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
    def __init__(self, source: str, filename: str):
        self.source = source
        self.filename = filename
        self.index = 0
        self.line = 1
        self.column = 1
        self.line_is_blank = True

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
            self.line_is_blank = True
        else:
            self.column += 1
            if not character.isspace():
                self.line_is_blank = False
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
                if not self.line_is_blank:
                    self.error("preprocessor directives must begin a source line")
                line, column = self.line, self.column
                directive: list[str] = []
                while self.index < len(self.source) and self.source[self.index] != "\n":
                    directive.append(self.advance())
                text = "".join(directive).strip()
                match = _INCLUDE.fullmatch(text)
                if match is None or match.group(1) not in _HEADERS:
                    self.error(
                        "py2bin has no C preprocessor; the only directive it "
                        "accepts is #include of a standard header it can ignore "
                        f"({', '.join(sorted(_HEADERS))}), and every declaration "
                        "must be written out in the file",
                        line,
                        column,
                    )
                continue
            return

    def escape(self, quote: str) -> int:
        """Consume one character (or escape sequence) and return its byte value."""

        character = self.advance()
        if character != "\\":
            value = ord(character)
            if value > 0xFF:
                self.error(
                    "py2bin's C compiler handles single-byte characters only; "
                    "this literal is not representable in one byte"
                )
            return value
        if self.index >= len(self.source):
            self.error("unterminated escape sequence")
        escaped = self.advance()
        if escaped in _SIMPLE_ESCAPES:
            return _SIMPLE_ESCAPES[escaped]
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
            return value
        if escaped in "01234567":
            digits = escaped
            while len(digits) < 3 and self.index < len(self.source) and (
                self.source[self.index] in "01234567"
            ):
                digits += self.advance()
            value = int(digits, 8)
            if value > 0xFF:
                self.error("octal escape does not fit in one byte")
            return value
        self.error(f"unsupported escape sequence \\{escaped} in a {quote} literal")

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
                    self.error(
                        "floating-point literals are not implemented by py2bin's "
                        "C compiler; it compiles the integer and pointer subset",
                        line,
                        column,
                    )
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

    def character(self) -> Token:
        line, column = self.line, self.column
        self.advance()  # opening quote
        if self.index < len(self.source) and self.source[self.index] == "'":
            self.error("empty character constant", line, column)
        value = self.escape("character")
        if self.index >= len(self.source) or self.advance() != "'":
            self.error("multi-character constants are not supported", line, column)
        # A character constant has type int in C, and a plain 'char' is signed
        # in this dialect, so \xFF is -1 exactly as it is on Apple's ABI.
        if value >= 0x80:
            value -= 0x100
        return Token("integer", value, line, column, "")

    def string(self) -> Token:
        line, column = self.line, self.column
        self.advance()  # opening quote
        data = bytearray()
        while True:
            if self.index >= len(self.source):
                self.error("unterminated string literal", line, column)
            if self.source[self.index] == '"':
                self.advance()
                break
            if self.source[self.index] == "\n":
                self.error("newline in string literal", line, column)
            data.append(self.escape("string"))
        return Token("string", bytes(data), line, column)

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
                    self.error(
                        "wide and Unicode string/character literals are not "
                        "supported",
                        line,
                        column,
                    )
                result.append(Token("identifier", name, line, column))
                continue
            if character.isdigit():
                result.append(self.number())
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
class PointerType:
    target: "CType"

    def __str__(self) -> str:
        return f"{self.target} *"


@dataclasses.dataclass(frozen=True, slots=True)
class ArrayType:
    element: "CType"
    count: int | None

    def __str__(self) -> str:
        return f"{self.element}[{'' if self.count is None else self.count}]"


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


CType = VoidType | IntegerType | PointerType | ArrayType | OpaqueType

VOID = VoidType()
CHAR = IntegerType("char", 1, True, 1)
SCHAR = IntegerType("signed char", 1, True, 1)
UCHAR = IntegerType("unsigned char", 1, False, 1)
SHORT = IntegerType("short", 2, True, 2)
USHORT = IntegerType("unsigned short", 2, False, 2)
INT = IntegerType("int", 4, True, 3)
UINT = IntegerType("unsigned int", 4, False, 3)
LONG = IntegerType("long", 8, True, 4)
ULONG = IntegerType("unsigned long", 8, False, 4)
LLONG = IntegerType("long long", 8, True, 5)
ULLONG = IntegerType("unsigned long long", 8, False, 5)
BOOL = IntegerType("_Bool", 1, False, 0)

# py2bin's C is LP64 on every target it emits, including Windows. The compiler
# never shares a header or an ABI with a platform C library beyond the vetted
# adapter table (whose Py_ssize_t is 64-bit everywhere), so one model keeps the
# same source producing the same values on all six targets.
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

_UNSIGNED_COUNTERPART = {INT: UINT, LONG: ULONG, LLONG: ULLONG}

_TYPE_KEYWORDS = frozenset(
    {"void", "char", "short", "int", "long", "signed", "unsigned", "_Bool"}
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
}


def size_of(ctype: CType) -> int | None:
    """The size in bytes of ``ctype``, or None when it is incomplete."""

    if isinstance(ctype, IntegerType):
        return ctype.size
    if isinstance(ctype, PointerType):
        return 8
    if isinstance(ctype, ArrayType):
        element = size_of(ctype.element)
        if element is None or ctype.count is None:
            return None
        return element * ctype.count
    return None


def is_signed(ctype: CType) -> bool:
    return isinstance(ctype, IntegerType) and ctype.signed


def is_arithmetic(ctype: CType) -> bool:
    return isinstance(ctype, IntegerType)


def is_scalar(ctype: CType) -> bool:
    return isinstance(ctype, (IntegerType, PointerType))


def promote(ctype: CType) -> CType:
    """C's integer promotions: everything narrower than int becomes int."""

    if isinstance(ctype, IntegerType) and ctype.rank < INT.rank:
        return INT
    return ctype


def usual_conversions(left: IntegerType, right: IntegerType) -> IntegerType:
    """C11 6.3.1.8, the usual arithmetic conversions, in full."""

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
class StringLiteral(Node):
    data: bytes


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
class Index(Node):
    base: Node
    offset: Node


@dataclasses.dataclass(slots=True)
class Cast(Node):
    ctype: CType
    operand: Node


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


@dataclasses.dataclass(slots=True)
class TranslationUnit:
    functions: dict[str, Function]
    externs: dict[str, CType]  # local name -> declared C result type


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
    "struct": "struct types are not implemented",
    "union": "union types are not implemented",
    "enum": "enum types are not implemented",
    "typedef": "typedef declarations are not implemented",
    "float": "floating point is not implemented by py2bin's C compiler",
    "double": "floating point is not implemented by py2bin's C compiler",
    "static": "static storage duration is not implemented; py2bin's C has no writable data segment",
    "register": "the 'register' storage class is not accepted",
    "auto": "the 'auto' storage class is not accepted",
    "_Complex": "complex types are not implemented",
    "_Atomic": "atomic types are not implemented; py2bin emits no atomic instructions",
    "_Thread_local": "thread-local storage is not implemented",
    "inline": "the 'inline' specifier is not accepted; py2bin decides for itself "
    "whether a call is a real call or an inlined body",
}


class Parser:
    def __init__(self, tokens: list[Token], filename: str):
        self.tokens = tokens
        self.filename = filename
        self.index = 0
        self.functions: dict[str, Function] = {}
        self.externs: dict[str, CType] = {}

    # --- token helpers ---

    @property
    def token(self) -> Token:
        return self.tokens[self.index]

    def peek(self, distance: int = 1) -> Token:
        return self.tokens[min(self.index + distance, len(self.tokens) - 1)]

    def error(self, message: str, token: Token | None = None):
        location = token or self.token
        raise CCompileError(self.filename, location.line, location.column, message)

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
        if name in _TYPE_KEYWORDS or name in _QUALIFIERS or name in _TYPEDEFS:
            return True
        if name in _OPAQUE_NAMES:
            # 'PyObject x' is not something py2bin can lay out, but 'PyObject *x'
            # is a handle. Only the pointer form is a type here.
            return self.peek().value == "*"
        return False

    def type_specifier(self) -> CType:
        """Parse a declaration specifier list into one type."""

        start = self.token
        words: list[str] = []
        base: CType | None = None
        while self.token.kind == "identifier":
            name = str(self.token.value)
            if name in _UNSUPPORTED_KEYWORDS:
                self.error(_UNSUPPORTED_KEYWORDS[name])
            if name in _QUALIFIERS:
                self.index += 1
                continue
            if name in _TYPE_KEYWORDS:
                words.append(name)
                self.index += 1
                continue
            if base is None and not words and name in _TYPEDEFS:
                base = _TYPEDEFS[name]
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
            resolved = _SPECIFIER_COMBINATIONS.get(key)
            if resolved is None:
                resolved = _SPECIFIER_COMBINATIONS.get(tuple(sorted(key)))
            if resolved is None:
                self.error(f"unsupported type specifier {' '.join(words)!r}", start)
            base = resolved
        if base is None:
            self.error(f"expected a type name, found {self.describe(start)}", start)
        return base

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
        """Parse ``*name[N]...`` and return the full type plus the name.

        ``optional`` accepts either form, which is what a prototype's parameter
        list needs: ``int f(int, int *);`` names nothing, ``int f(int a);`` does.
        """

        base = self.pointer_suffix(base)
        if self.at("("):
            self.error(
                "function pointers and parenthesized declarators are not "
                "implemented; py2bin's native IR has no indirect call"
            )
        name = ""
        if not abstract and not (optional and self.token.kind != "identifier"):
            name = str(self.identifier().value)
        if self.at("("):
            self.error(
                "function pointers and function declarators are not implemented"
            )
        dimensions: list[int | None] = []
        while self.accept("["):
            if self.accept("]"):
                dimensions.append(None)
                continue
            dimensions.append(self.array_length())
            self.take("]")
        for length in reversed(dimensions):
            base = ArrayType(base, length)
        return base, name

    def array_length(self) -> int:
        token = self.token
        value = ConstantEvaluator(self.filename).value(self.assignment_expression())
        if value <= 0:
            self.error("an array needs a positive constant length", token)
        return value

    def type_name(self) -> CType:
        """A type in a cast or ``sizeof``: specifiers plus an abstract declarator."""

        base = self.type_specifier()
        base = self.pointer_suffix(base)
        while self.accept("["):
            length = self.array_length()
            self.take("]")
            base = ArrayType(base, length)
        return base

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
                self.error(
                    "struct and union member access is not implemented; py2bin's "
                    "C compiler models no aggregate layouts"
                )
            return node

    def primary_expression(self) -> Node:
        token = self.token
        if token.kind == "integer":
            self.take()
            return IntLiteral(token, int(token.value), _literal_type(token, self.filename))
        if token.kind == "string":
            self.take()
            data = bytes(token.value)  # type: ignore[arg-type]
            while self.token.kind == "string":  # adjacent literals concatenate
                data += bytes(self.take().value)  # type: ignore[arg-type]
            return StringLiteral(token, data)
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
        return Declaration(token, entries)

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

    def translation_unit(self) -> TranslationUnit:
        while self.token.kind != "eof":
            if self.accept("extern"):
                self.extern_prototype()
                continue
            self.function_definition()
        return TranslationUnit(self.functions, self.externs)

    def function_definition(self) -> None:
        start = self.token
        base = self.type_specifier()
        result = self.pointer_suffix(base)
        name_token = self.identifier()
        name = str(name_token.value)
        if not self.at("("):
            self.error(
                "py2bin's C compiler has no writable data segment, so a file-scope "
                "variable cannot be defined; only functions may appear here",
                start,
            )
        self.take("(")
        parameters: list[tuple[CType, str]] = []
        if not self.accept(")"):
            if self.at("void") and self.peek().value == ")":
                self.take("void")
                self.take(")")
            else:
                while True:
                    if self.at("..."):
                        self.error("variadic user functions are not implemented")
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
        if name in self.externs:
            self.error(f"{name!r} is already declared", name_token)
        previous = self.functions.get(name)
        if previous is not None:
            self.check_redeclaration(previous, result, parameters, name_token)
        if self.accept(";"):
            # A prototype. It carries no body, and repeating it is legal C as
            # long as the signature agrees, which check_redeclaration enforced.
            if previous is None:
                self.functions[name] = Function(
                    name, result, parameters, None, name_token
                )
            return
        if previous is not None and previous.body is not None:
            self.error(f"{name!r} is already defined", name_token)
        seen: set[str] = set()
        for parameter_type, parameter_name in parameters:
            if not parameter_name:
                self.error("every parameter needs a name", name_token)
            if parameter_name in seen:
                self.error(f"duplicate parameter {parameter_name!r}", name_token)
            seen.add(parameter_name)
            if isinstance(parameter_type, VoidType):
                self.error("a parameter cannot have type void", name_token)
        # The signature is registered before the body is parsed so that the
        # function's own name -- and any name a prototype introduced -- resolves
        # inside it. That is what makes direct and mutual recursion parseable.
        self.functions[name] = Function(name, result, parameters, None, name_token)
        body = self.compound_statement()
        self.functions[name] = Function(name, result, parameters, body, name_token)

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


def _literal_type(token: Token, filename: str) -> CType:
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
    if "u" in suffix:
        candidates = [UINT, ULONG, ULLONG][longs:]
    elif token.radix == 10:
        candidates = [INT, LONG, LLONG][longs:]
    else:
        candidates = [INT, UINT, LONG, ULONG, LLONG, ULLONG][longs * 2 :]
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
    if kind in {"cstr", "cfmt"}:
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

    def __init__(self, filename: str):
        self.filename = filename

    def error(self, message: str, token: Token):
        raise CCompileError(self.filename, token.line, token.column, message)

    def value(self, node: Node) -> int:
        result = self.evaluate(node)
        return result

    def evaluate(self, node: Node) -> int:
        if isinstance(node, IntLiteral):
            return node.value
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


def _contains_call(value: object) -> bool:
    """True when an IR expression embeds a call, so re-emitting it would repeat it.

    ``Lowerer.extern_call`` and ``Lowerer.direct_call`` pin every call in a slot
    as soon as it is lowered, so nothing this compiler builds should ever trip
    this check. It stays as the guard on the two places that reuse an expression
    -- the address of a read-modify-write target -- because the failure it
    prevents (a call happening twice because its value appeared twice in a tree)
    is the defect this backend has produced most often.
    """

    if isinstance(value, (ExternCall, IRCall)):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_call(item) for item in value)
    for name in getattr(type(value), "__slots__", ()) or ():
        if _contains_call(getattr(value, name)):
            return True
    return False


@dataclasses.dataclass(frozen=True, slots=True)
class Value:
    """A lowered C expression: its type and its canonical 64-bit IR value.

    Every integer value is kept sign-extended (signed types) or zero-extended
    (unsigned types) into 64 bits, so one representation serves both the
    register file and memory of any width. ``null`` marks the null pointer
    constant, the one integer C lets stand in for a pointer.
    """

    ctype: CType
    expr: IntExpression
    null: bool = False


@dataclasses.dataclass(slots=True)
class Local:
    ctype: CType
    slot: int


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
_LENGTHS = {
    "": {},
    "hh": {"d": SCHAR, "i": SCHAR, "u": UCHAR, "x": UCHAR, "X": UCHAR},
    "h": {"d": SHORT, "i": SHORT, "u": USHORT, "x": USHORT, "X": USHORT},
    "l": {"d": LONG, "i": LONG, "u": ULONG, "x": ULONG, "X": ULONG},
    "ll": {"d": LLONG, "i": LLONG, "u": ULLONG, "x": ULLONG, "X": ULLONG},
    "z": {"d": LONG, "i": LONG, "u": ULONG, "x": ULONG, "X": ULONG},
    "j": {"d": LLONG, "i": LLONG, "u": ULLONG, "x": ULLONG, "X": ULLONG},
}

_MAXIMUM_SLOTS = 4000

#: py2bin's call ABI passes every argument in a register. AAPCS64 has eight
#: integer parameter registers, and stack argument passing is not implemented,
#: so a longer parameter list is rejected rather than silently truncated.
_MAXIMUM_ARGUMENTS = 8


class Lowerer:
    """Lowers a parsed translation unit to py2bin's native IR."""

    def __init__(self, unit: TranslationUnit, filename: str, target: str):
        self.unit = unit
        self.filename = filename
        self.target = target
        self.operations: list[Operation] = []
        self.stack_slots = 0
        self.scopes: list[dict[str, Local]] = []
        self.counter = 0
        self.break_targets: list[str] = []
        self.continue_targets: list[str] = []
        self.switches: list[SwitchContext] = []
        self.functions: list[FunctionContext] = []
        self.active: list[str] = []
        self.buffer_slot: int | None = None
        # Real calls: every function reached from main, lowered once into its
        # own IR body, plus the set currently being lowered so a call that
        # arrives while its own body is still open (that is, recursion) emits a
        # call rather than trying to lower the body a second time.
        self.calls_are_real = target in CALL_CAPABLE_TARGETS
        self.lowered: dict[str, IRFunction] = {}
        self.lowering: set[str] = set()

    # --- bookkeeping ---

    def error(self, message: str, token: Token):
        raise CCompileError(self.filename, token.line, token.column, message)

    def emit(self, operation: Operation) -> None:
        self.operations.append(operation)

    def allocate(self, size: int) -> int:
        slots = max(1, (size + 7) // 8)
        base = self.stack_slots
        self.stack_slots += slots
        if self.stack_slots > _MAXIMUM_SLOTS:
            raise CCompileError(
                self.filename,
                1,
                1,
                f"this translation unit needs more than {_MAXIMUM_SLOTS * 8} bytes "
                "of stack frame; py2bin's native frames are a single fixed "
                "allocation, so reduce the size of the local arrays",
            )
        return base

    def new_temp(self) -> int:
        return self.allocate(8)

    def new_label(self, prefix: str) -> str:
        self.counter += 1
        return f"c_{prefix}_{self.counter}"

    def declare(self, name: str, ctype: CType, token: Token) -> Local:
        scope = self.scopes[-1]
        if name in scope:
            self.error(f"{name!r} is already declared in this scope", token)
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
        return None

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

    def assign_convert(
        self, value: Value, target: CType, token: Token, what: str
    ) -> IntExpression:
        if isinstance(target, IntegerType):
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
        size = size_of(ctype)
        if size is None:
            raise AssertionError("incomplete lvalue reached the loader")
        return Value(ctype, HeapLoad(address, size, is_signed(ctype)))

    def lvalue(self, node: Node) -> tuple[CType, IntExpression]:
        if isinstance(node, Identifier):
            local = self.lookup(node.name)
            if local is None:
                if node.name == "NULL":
                    self.error("the null pointer constant is not an lvalue", node.token)
                self.error(
                    f"{node.name!r} is not a declared local or parameter", node.token
                )
            return local.ctype, SlotAddress(local.slot)
        if isinstance(node, Unary) and node.operator == "*":
            pointer = self.rvalue(node.operand)
            if not isinstance(pointer.ctype, PointerType):
                self.error(
                    f"cannot dereference a value of type {pointer.ctype}", node.token
                )
            target = pointer.ctype.target
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
        self.error("this expression is not an lvalue", node.token)

    def stabilize(self, expression: IntExpression) -> IntExpression:
        """Pin a value in a slot when re-emitting it would repeat a call."""

        if isinstance(expression, (IntConstant, IntLoad, SlotAddress)):
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

    # --- expressions ---

    def rvalue(self, node: Node) -> Value:
        if isinstance(node, IntLiteral):
            return Value(
                node.ctype,
                _constant(node.value),
                null=node.value == 0 and node.ctype in {INT, LONG, LLONG},
            )
        if isinstance(node, StringLiteral):
            if self.target != "darwin-arm64":
                self.error(
                    "using a string literal as a pointer value needs the constant "
                    "blob the darwin-arm64 image writer emits; it is not "
                    f"implemented for {self.target!r} (printf of a literal is)",
                    node.token,
                )
            return Value(PointerType(CHAR), CStringConstant(node.data + b"\0"))
        if isinstance(node, Identifier):
            local = self.lookup(node.name)
            if local is None:
                if node.name == "NULL":
                    return Value(PointerType(VOID), IntConstant(0), null=True)
                self.error(
                    f"{node.name!r} is not a declared local or parameter", node.token
                )
            return self.load(local.ctype, SlotAddress(local.slot))
        if isinstance(node, Index):
            ctype, address = self.lvalue(node)
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
            return len(node.data) + 1
        if isinstance(node, Identifier):
            local = self.lookup(node.name)
            if local is not None:
                size = size_of(local.ctype)
                if size is None:
                    self.error(
                        f"sizeof({node.name}) needs a complete type", node.token
                    )
                return size
        saved_operations = self.operations
        saved_slots = self.stack_slots
        self.operations = []
        try:
            if isinstance(node, Index) or (
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
            ctype, address = self.lvalue(node.operand)
            return Value(PointerType(ctype), address)
        if node.operator == "*":
            ctype, address = self.lvalue(node)
            return self.load(ctype, address)
        if node.operator == "!":
            value = self.scalar(node.operand, "the operand of '!'")
            return Value(INT, _compare("eq", value.expr, IntConstant(0)))
        value = self.rvalue(node.operand)
        if not is_arithmetic(value.ctype):
            self.error(
                f"unary {node.operator!r} needs an arithmetic operand, not "
                f"{value.ctype}",
                node.token,
            )
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
        if not is_arithmetic(count.ctype):
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
            self.emit(JumpIfFalse(left.expr, end))
            right = self.scalar(node.right, "the right operand of '&&'")
            self.emit(JumpIfFalse(right.expr, end))
            self.emit(Store(slot, IntConstant(1)))
        else:
            taken = self.new_label("logic_true")
            other = self.new_label("logic_right")
            self.emit(Store(slot, IntConstant(0)))
            self.emit(JumpIfFalse(left.expr, other))
            self.emit(Jump(taken))
            self.emit(Label(other))
            right = self.scalar(node.right, "the right operand of '||'")
            self.emit(JumpIfFalse(right.expr, end))
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
        """

        test = self.scalar(node.test, "the condition of '?:'")
        otherwise = self.new_label("select_else")
        end = self.new_label("select_end")
        slot = self.new_temp()
        self.emit(JumpIfFalse(test.expr, otherwise))
        body = self.rvalue(node.body)
        self.emit(Store(slot, body.expr))
        self.emit(Jump(end))
        self.emit(Label(otherwise))
        alternative = self.rvalue(node.alternative)
        self.emit(Store(slot, alternative.expr))
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
            result: CType = usual_conversions(body.ctype, alternative.ctype)
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
        if isinstance(target, IntegerType):
            return Value(target, self.fit(value.expr, target))
        return Value(target, value.expr, null=value.null)

    def assignment(self, node: Assignment) -> Value:
        ctype, address = self.lvalue(node.target)
        if isinstance(ctype, ArrayType):
            self.error("an array is not assignable", node.token)
        address = self.stabilize(address)
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
        stored = self.materialize(stored)
        self.emit(HeapStore(address, stored, size_of(ctype)))
        return Value(ctype, stored)

    # --- calls ---

    def call(self, node: Call) -> Value:
        if node.name == "printf":
            self.error(
                "printf's return value is not implemented; call it as a statement",
                node.token,
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
                "calls has to be in this translation unit",
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

    def extern_call(self, node: Call, *, discarded: bool) -> Value:
        name = node.name
        symbol, signature = _CABI_SYMBOLS[name]
        result_kind = _CABI_RESULTS[name]
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
            if kind in {"cstr", "cfmt"}:
                if not isinstance(argument, StringLiteral):
                    self.error(
                        f"{what} must be a literal C string: py2bin materializes it "
                        "in the image, and a runtime pointer would need a lifetime "
                        "this compiler cannot verify",
                        argument.token,
                    )
                if b"\0" in argument.data:
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
            else:
                if not is_arithmetic(value.ctype):
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

    def prepare_arguments(
        self, function: Function, node: Call
    ) -> list[IntExpression]:
        """Check the argument count and convert each argument to its parameter."""

        if len(node.arguments) != len(function.parameters):
            self.error(
                f"{function.name}() takes {len(function.parameters)} argument(s), "
                f"got {len(node.arguments)}",
                node.token,
            )
        prepared: list[IntExpression] = []
        for position, (argument, (parameter_type, _name)) in enumerate(
            zip(node.arguments, function.parameters), 1
        ):
            prepared.append(
                self.assign_convert(
                    self.rvalue(argument),
                    parameter_type,
                    argument.token,
                    f"argument {position} of {function.name}()",
                )
            )
        return prepared

    # --- real calls --------------------------------------------------------
    #
    # On a target whose encoder implements the call ABI, a call is a call: the
    # callee is lowered once into its own IR Function with its own frame, and
    # the call site branches to it. Recursion then costs nothing special -- the
    # body being lowered simply refers to itself by name.

    def direct_call(self, function: Function, node: Call) -> Value:
        if len(function.parameters) > _MAXIMUM_ARGUMENTS:
            self.error(
                f"{function.name}() takes {len(function.parameters)} parameters; "
                f"py2bin's call ABI passes at most {_MAXIMUM_ARGUMENTS} arguments "
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
        return Value(function.result, self.fit(IntLoad(slot), function.result))

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
            self.scopes,
            self.buffer_slot,
            self.break_targets,
            self.continue_targets,
            self.switches,
            self.functions,
        )
        self.operations = []
        self.stack_slots = 0
        self.scopes = [{}]
        self.buffer_slot = None
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
            if self.stack_slots != len(function.parameters):
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
            body = IRFunction(
                function.name,
                len(function.parameters),
                self.stack_slots,
                self.operations,
            )
        finally:
            (
                self.operations,
                self.stack_slots,
                self.scopes,
                self.buffer_slot,
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
        for (parameter_type, name), expression in zip(function.parameters, prepared):
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
        return Value(function.result, IntLoad(result_slot))

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
            if node.name == "printf":
                self.printf(node)
                return
            if node.name in self.unit.externs:
                self.extern_call(node, discarded=True)
                return
        if isinstance(node, Comma):
            self.expression_statement(node.left)
            self.expression_statement(node.right)
            return
        self.rvalue(node)

    def declaration(self, node: Declaration) -> None:
        for ctype, name, initializer in node.entries:
            if isinstance(ctype, VoidType):
                self.error(f"{name!r} cannot have type void", node.token)
            if isinstance(ctype, ArrayType) and ctype.count is None:
                ctype = self.deduce_array(ctype, initializer, node.token)
            local = self.declare(name, ctype, node.token)
            if initializer is None:
                continue
            if isinstance(ctype, ArrayType):
                self.array_initializer(local, ctype, initializer, node.token)
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
                HeapStore(SlotAddress(local.slot), stored, size_of(ctype))
            )

    def deduce_array(
        self, ctype: ArrayType, initializer: object, token: Token
    ) -> ArrayType:
        if isinstance(initializer, tuple):
            return ArrayType(ctype.element, max(1, len(initializer[1])))
        if isinstance(initializer, StringLiteral) and _is_character(ctype.element):
            return ArrayType(ctype.element, len(initializer.data) + 1)
        self.error(
            "an array without a length needs a braced initializer (or a string "
            "literal for a character array) to deduce it from",
            token,
        )

    def array_initializer(
        self, local: Local, ctype: ArrayType, initializer: object, token: Token
    ) -> None:
        element = ctype.element
        size = size_of(element)
        if size is None or isinstance(element, ArrayType):
            self.error(
                "only a one-dimensional array of scalars can be initialized; "
                "assign the elements instead",
                token,
            )
        base = SlotAddress(local.slot)
        if isinstance(initializer, StringLiteral):
            if not _is_character(element):
                self.error(
                    "a string literal can only initialize a character array", token
                )
            data = initializer.data + b"\0"
            if len(data) > ctype.count:
                self.error(
                    f"the initializer is {len(data)} bytes but the array holds "
                    f"{ctype.count}",
                    token,
                )
            self.emit_bytes(base, 0, data + b"\0" * (ctype.count - len(data)))
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
            if isinstance(item, tuple):
                self.error("nested braced initializers are not implemented", token)
            value = self.rvalue(item)
            stored = self.assign_convert(
                value, element, token, f"initializer element {position}"
            )
            self.emit(
                HeapStore(
                    _binary("add", base, IntConstant(position * size)), stored, size
                )
            )
        # C zero-fills whatever the braces leave out.
        filled = len(items) * size
        self.emit_bytes(base, filled, b"\0" * (ctype.count * size - filled))

    def emit_bytes(self, base: IntExpression, offset: int, data: bytes) -> None:
        """Store a constant byte image, using the widest aligned store each time.

        A byte-at-a-time fill of ``char page[2048] = {0}`` would be two thousand
        instructions. Both architectures py2bin emits for are little-endian, so
        a chunk of the image packs into one store in source order.
        """

        start = offset
        end = offset + len(data)
        while offset < end:
            for width in (8, 4, 2, 1):
                if offset % width == 0 and offset + width <= end:
                    chunk = data[offset - start : offset - start + width]
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
        self.emit(JumpIfFalse(test.expr, otherwise))
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
        self.emit(JumpIfFalse(test.expr, end))
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
        self.emit(JumpIfFalse(test.expr, end))
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
            self.emit(JumpIfFalse(test.expr, end))
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
        if not is_arithmetic(control.ctype):
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
            raw = ConstantEvaluator(self.filename).value(node.value)
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
        stored = self.assign_convert(
            value, context.function.result, node.token, "this 'return'"
        )
        if context.is_main:
            self.emit(ExitValue(stored))
            return
        if context.call_body:
            self.emit(IRReturn(stored))
            return
        assert context.result_slot is not None
        self.emit(Store(context.result_slot, stored))
        self.emit(Jump(context.return_label))

    # --- printf ---

    def print_buffer(self) -> int:
        if self.buffer_slot is None:
            self.buffer_slot = self.allocate(32)
        return self.buffer_slot

    def printf(self, node: Call) -> None:
        if not node.arguments or not isinstance(node.arguments[0], StringLiteral):
            self.error(
                "printf needs a literal format string; py2bin reads the format at "
                "compile time and emits the formatting code itself",
                node.token,
            )
        segments = self.parse_format(node.arguments[0])
        arguments = node.arguments[1:]
        expected = sum(1 for kind, _ in segments if kind != "text")
        if expected != len(arguments):
            self.error(
                f"printf has {expected} conversion(s) but {len(arguments)} "
                "argument(s)",
                node.token,
            )
        if expected and self.target.startswith("windows-"):
            self.error(
                "printf with a runtime conversion needs the write syscall py2bin "
                f"only emits for POSIX targets, not {self.target!r}; a format with "
                "no conversions works everywhere",
                node.token,
            )
        # C11 6.5.2.2p10 puts a sequence point after every argument is
        # evaluated and before the call, so printf may produce no output until
        # all of its arguments have been computed. Evaluating them while
        # emitting the format text would let an argument that writes to stdout
        # interleave with the literal parts. Evaluate everything first, hold
        # each result in a slot, then emit the output.
        prepared: list[object] = []
        for position, argument in enumerate(arguments):
            style, ctype = segments[
                [i for i, (kind, _) in enumerate(segments) if kind != "text"][position]
            ][1]
            value = self.rvalue(argument)
            prepared.append((style, ctype, value, argument))
        held: list[object] = []
        for style, ctype, value, argument in prepared:
            if style == "string":
                held.append((style, value, argument))
                continue
            if not is_arithmetic(value.ctype):
                self.error(
                    f"a %{style} conversion needs an integer, not {value.ctype}",
                    argument.token,
                )
            # materialize() pins the value in a slot, so evaluating a later
            # argument cannot change what this one reads.
            held.append(
                (style, self.materialize(self.fit(value.expr, ctype)), argument)
            )

        index = 0
        for kind, payload in segments:
            if kind == "text":
                self.emit(Write(payload))
                continue
            style, value, argument = held[index]
            index += 1
            if style == "string":
                if value.null or not isinstance(value.ctype, PointerType):
                    self.error(
                        "a %s conversion needs a character pointer", argument.token
                    )
                if not _is_character(value.ctype.target):
                    self.error(
                        f"a %s conversion needs a character pointer, not "
                        f"{value.ctype}",
                        argument.token,
                    )
                self.emit_string(value.expr)
                continue
            expression = value
            if style == "char":
                self.emit_character(expression)
            elif style == "signed":
                self.emit_number(expression, signed=True, base=10, upper=False)
            elif style == "unsigned":
                self.emit_number(expression, signed=False, base=10, upper=False)
            else:
                self.emit_number(
                    expression, signed=False, base=16, upper=style == "HEX"
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
            length = ""
            while position < len(data) and chr(data[position]) in "hlzjt":
                length += chr(data[position])
                position += 1
            if position >= len(data):
                self.error("printf format ends inside a conversion", literal.token)
            specifier = chr(data[position])
            position += 1
            if specifier not in _CONVERSIONS:
                self.error(
                    f"printf conversion %{length}{specifier} is not implemented; "
                    "py2bin emits the formatting itself and supports "
                    "%d %i %u %x %X %c %s and %% with the h/hh/l/ll/z length "
                    "modifiers (no flags, width, or precision)",
                    literal.token,
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
            if text:
                segments.append(("text", bytes(text)))
                text.clear()
            segments.append(("conversion", (style, ctype)))
        if text:
            segments.append(("text", bytes(text)))
        return segments

    def emit_character(self, expression: IntExpression) -> None:
        base = SlotAddress(self.print_buffer())
        self.emit(HeapStore(base, expression, 1))
        self.emit(WriteRuntime(base, IntConstant(1)))

    def emit_string(self, pointer: IntExpression) -> None:
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
                    _binary("add", IntLoad(pointer_slot), IntLoad(length_slot)), 1
                ),
                end,
            )
        )
        self.emit(
            Store(length_slot, _binary("add", IntLoad(length_slot), IntConstant(1)))
        )
        self.emit(Jump(top))
        self.emit(Label(end))
        self.emit(WriteRuntime(IntLoad(pointer_slot), IntLoad(length_slot)))

    def emit_number(
        self, expression: IntExpression, *, signed: bool, base: int, upper: bool
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
        self.emit(Store(index_slot, IntConstant(32)))
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
        if signed:
            done = self.new_label("no_sign")
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
        self.emit(
            WriteRuntime(
                _binary("add", buffer, IntLoad(index_slot)),
                _binary("sub", IntConstant(32), IntLoad(index_slot)),
            )
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
        self.scopes.pop()
        return Module(self.operations, self.stack_slots, list(self.lowered.values()))


def _is_character(ctype: CType) -> bool:
    return ctype in {CHAR, SCHAR, UCHAR}


def compile_c_to_ir(source: str, filename: str, target: str) -> Module:
    """Compile C source text to a py2bin native IR module."""

    unit = Parser(Lexer(source, filename).tokens(), filename).translation_unit()
    return Lowerer(unit, filename, target).compile()
