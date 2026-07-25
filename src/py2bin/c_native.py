"""Handwritten canonical-C frontend for py2bin's native integer backend.

This is intentionally not a general ISO C compiler.  It accepts the small,
documented C language emitted for py2bin's signed-64-bit Python subset, rebuilds
an equivalent syntax tree, and feeds that tree to py2bin's own IR and binary
writers.  Unsupported C is rejected instead of being sent to a host compiler.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

from .csource import compile_to_c
from .native import NativeResult, compile_native_source


# Symbols the generated C may declare via an extern prototype. Kept in sync
# with py2bin.native.frontend._CABI_SYMBOLS, the compiler-side whitelist.
from .native.frontend import _CABI_RESULTS as _EXTERN_RESULTS
from .native.frontend import _CABI_SYMBOLS as _EXTERN_SYMBOLS

# How an adapter-ABI argument/result kind must be spelled in a C prototype.
# ``cstr``/``cfmt`` arguments are ``char *`` in C but compile-time constants
# here, so the prototype writes a pointer while the vetted signature keeps the
# stronger requirement.
_ABI_TO_C = {"int": ("i64", "int"), "ptr": ("ptr",), "cstr": ("ptr",), "cfmt": ("ptr",), "void": ("void",)}

_INCLUDE = re.compile(r"#\s*include\s*<\s*(stdio\.h|math\.h|string\.h)\s*>\s*\Z")
_MULTI_OPERATORS = (
    "<<=",
    ">>=",
    "&&",
    "||",
    "==",
    "!=",
    "<=",
    ">=",
    "<<",
    ">>",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "++",
    "--",
)
_SINGLE_TOKENS = set("{}();,?:=+-*/%~!<>&|^")


class CNativeCompileError(ValueError):
    """A source-located rejection from the canonical C frontend."""

    def __init__(self, filename: str, line: int, column: int, message: str):
        self.filename = filename
        self.line = line
        self.column = column
        self.message = message
        super().__init__(f"{filename}:{line}:{column}: {message}")


@dataclasses.dataclass(frozen=True, slots=True)
class CBridgeResult:
    native: NativeResult
    c_source: str
    reconstructed_python: str
    c_artifact: Path | None


@dataclasses.dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    value: str | int
    line: int
    column: int


class _Lexer:
    def __init__(self, source: str, filename: str):
        self.source = source
        self.filename = filename
        self.index = 0
        self.line = 1
        self.column = 1
        self.line_only_whitespace = True

    def error(self, message: str, line: int | None = None, column: int | None = None):
        raise CNativeCompileError(
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
            self.line_only_whitespace = True
        else:
            self.column += 1
            if not character.isspace():
                self.line_only_whitespace = False
        return character

    def startswith(self, value: str) -> bool:
        return self.source.startswith(value, self.index)

    def skip_layout(self) -> None:
        while self.index < len(self.source):
            if self.source[self.index].isspace():
                self.advance()
                continue
            if self.startswith("//"):
                while self.index < len(self.source) and self.advance() != "\n":
                    pass
                continue
            if self.startswith("/*"):
                start_line, start_column = self.line, self.column
                self.advance()
                self.advance()
                while self.index < len(self.source) and not self.startswith("*/"):
                    self.advance()
                if self.index >= len(self.source):
                    self.error("unterminated block comment", start_line, start_column)
                self.advance()
                self.advance()
                continue
            if self.source[self.index] == "#":
                if not self.line_only_whitespace:
                    self.error("preprocessor directives must begin a source line")
                start_line, start_column = self.line, self.column
                directive: list[str] = []
                while self.index < len(self.source) and self.source[self.index] != "\n":
                    directive.append(self.advance())
                text = "".join(directive).strip()
                if _INCLUDE.fullmatch(text) is None:
                    self.error(
                        "only #include <stdio.h>, <math.h>, and <string.h> are accepted",
                        start_line,
                        start_column,
                    )
                continue
            return

    def number(self) -> _Token:
        line, column = self.line, self.column
        start = self.index
        if self.startswith("0x") or self.startswith("0X"):
            self.advance()
            self.advance()
            digit_start = self.index
            while self.index < len(self.source) and (
                self.source[self.index].isdigit()
                or self.source[self.index].lower() in "abcdef"
            ):
                self.advance()
            if self.index == digit_start:
                self.error("hexadecimal integer needs at least one digit", line, column)
            value = int(self.source[start:self.index], 16)
        else:
            while self.index < len(self.source) and self.source[self.index].isdigit():
                self.advance()
            value = int(self.source[start:self.index], 10)
        if self.index < len(self.source) and (
            self.source[self.index].isalnum()
            or self.source[self.index] in {"_", "."}
        ):
            self.error(
                "only unsuffixed integer literals are accepted",
                line,
                column,
            )
        return _Token("integer", value, line, column)

    def identifier(self) -> _Token:
        line, column = self.line, self.column
        start = self.index
        while self.index < len(self.source) and (
            self.source[self.index].isalnum() or self.source[self.index] == "_"
        ):
            self.advance()
        return _Token("identifier", self.source[start:self.index], line, column)

    def string(self) -> _Token:
        line, column = self.line, self.column
        self.advance()
        result: list[str] = []
        escapes = {
            "\\": "\\",
            '"': '"',
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "0": "\0",
        }
        while self.index < len(self.source):
            character = self.advance()
            if character == '"':
                return _Token("string", "".join(result), line, column)
            if character == "\n":
                self.error("newline in string literal", line, column)
            if character != "\\":
                result.append(character)
                continue
            if self.index >= len(self.source):
                self.error("unterminated string escape", line, column)
            escaped = self.advance()
            if escaped not in escapes:
                self.error(
                    f"unsupported string escape \\{escaped}",
                    self.line,
                    max(1, self.column - 2),
                )
            result.append(escapes[escaped])
        self.error("unterminated string literal", line, column)

    def tokens(self) -> list[_Token]:
        result: list[_Token] = []
        while True:
            self.skip_layout()
            if self.index >= len(self.source):
                result.append(_Token("eof", "", self.line, self.column))
                return result
            character = self.source[self.index]
            line, column = self.line, self.column
            if character.isalpha() or character == "_":
                result.append(self.identifier())
                continue
            if character.isdigit():
                result.append(self.number())
                continue
            if character == '"':
                result.append(self.string())
                continue
            operator = next(
                (candidate for candidate in _MULTI_OPERATORS if self.startswith(candidate)),
                None,
            )
            if operator is not None:
                for _ in operator:
                    self.advance()
                result.append(_Token("symbol", operator, line, column))
                continue
            if character in _SINGLE_TOKENS:
                self.advance()
                result.append(_Token("symbol", character, line, column))
                continue
            self.error(f"unsupported character {character!r}")


class _Parser:
    _PRECEDENCE = {
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
    _BINARY = {
        "+": ast.Add,
        "-": ast.Sub,
        "*": ast.Mult,
        "<<": ast.LShift,
        ">>": ast.RShift,
        "&": ast.BitAnd,
        "|": ast.BitOr,
        "^": ast.BitXor,
    }
    _COMPARE = {
        "==": ast.Eq,
        "!=": ast.NotEq,
        "<": ast.Lt,
        "<=": ast.LtE,
        ">": ast.Gt,
        ">=": ast.GtE,
    }
    _AUGMENTED = {
        "+=": ast.Add,
        "-=": ast.Sub,
        "*=": ast.Mult,
        "<<=": ast.LShift,
        ">>=": ast.RShift,
        "&=": ast.BitAnd,
        "|=": ast.BitOr,
        "^=": ast.BitXor,
    }

    def __init__(self, tokens: list[_Token], filename: str):
        self.tokens = tokens
        self.filename = filename
        self.index = 0
        self.current_function = ""
        self.function_names: set[str] = set()
        # External symbols declared by an ``extern`` prototype in this file.
        self.externs: set[str] = set()
        # --- the canonical-C type discipline --------------------------------
        # Every expression carries one of three kinds: "i64" (a signed 64-bit
        # integer), "ptr" (an opaque pointer-sized handle such as a
        # ``PyObject *``) and "cstr" (a string literal). They lower to the same
        # machine word, but keeping them apart at the source level is what
        # stops a plain integer from reaching a parameter the callee will
        # dereference -- a silent miscompile the backend could never catch.
        # "void" marks a call that produces no value at all.
        self.locals: dict[str, str] = {}
        # name -> (parameter kinds, result kind) for helper functions.
        self.signatures: dict[str, tuple[tuple[str, ...], str]] = {}
        self.result_kind = "i64"

    # --- expression kinds ----------------------------------------------------

    @staticmethod
    def kind(node: ast.expr) -> str:
        return getattr(node, "_c_kind", "i64")

    @staticmethod
    def mark(node: ast.expr, kind: str) -> ast.expr:
        node._c_kind = kind
        return node

    @staticmethod
    def is_zero(node: ast.expr) -> bool:
        """True for the literal 0, the one integer that may spell a NULL handle."""

        return isinstance(node, ast.Constant) and node.value == 0 and node.value is not False

    def value_kind(self, node: ast.expr, token: _Token) -> str:
        kind = self.kind(node)
        if kind == "void":
            self.error("a call that returns void does not produce a value", token)
        return kind

    # An adapter-ABI argument kind, as spelled in the vetted signature table,
    # mapped onto the kind an expression in this dialect carries.
    _ABI_ARGUMENT_KIND = {"int": "i64", "ptr": "ptr", "cstr": "cstr", "cfmt": "cstr"}

    def require_kind(self, node: ast.expr, expected: str, what: str, token: _Token) -> None:
        """Reject an expression whose kind cannot stand in for ``expected``."""

        expected = self._ABI_ARGUMENT_KIND.get(expected, expected)
        if expected not in {"i64", "ptr", "cstr", "void"}:
            raise AssertionError(f"unknown canonical C kind {expected!r}")
        actual = self.value_kind(node, token)
        if expected == "i64":
            if actual == "i64":
                return
            self.error(
                f"{what} needs a 'long long' value, but this expression is a "
                f"{'pointer handle' if actual == 'ptr' else 'string literal'}",
                token,
            )
        if expected == "ptr":
            if actual == "ptr" or self.is_zero(node):
                return
            self.error(
                f"{what} needs a pointer handle; pass a handle or the null "
                "pointer constant 0",
                token,
            )
        if expected == "cstr":
            if actual == "cstr":
                return
            self.error(
                f"{what} needs a literal C string, which is the only form "
                "py2bin can materialize in the image",
                token,
            )
        if expected == "void":
            self.error(f"{what} cannot be given a value", token)

    @property
    def token(self) -> _Token:
        return self.tokens[self.index]

    def error(self, message: str, token: _Token | None = None):
        location = token or self.token
        raise CNativeCompileError(
            self.filename,
            location.line,
            location.column,
            message,
        )

    def at(self, value: str) -> bool:
        return self.token.value == value

    def take(self, value: str | None = None) -> _Token:
        token = self.token
        if value is not None and token.value != value:
            self.error(f"expected {value!r}, found {token.value!r}")
        self.index += 1
        return token

    def accept(self, value: str) -> bool:
        if not self.at(value):
            return False
        self.index += 1
        return True

    def identifier(self) -> _Token:
        if self.token.kind != "identifier":
            self.error(f"expected identifier, found {self.token.value!r}")
        return self.take()

    # Opaque handle types. Generated C-API code never dereferences a Python
    # object in C: every ``PyObject *`` is passed straight back to the runtime,
    # so one pointer-sized integer models all of them. That is what keeps this
    # compiler small enough to be handwritten while still compiling the C the
    # Python-to-C backend emits.
    _HANDLE_TYPES = frozenset({"PyObject", "PyTypeObject", "void", "char", "FILE"})
    # Identifiers that can only begin a declaration, never an expression.
    _TYPE_STARTERS = frozenset({"long", "int", "const", "Py_ssize_t"}) | _HANDLE_TYPES

    def at_declaration(self) -> bool:
        """True when the next tokens begin a local declaration, not an expression."""

        if self.token.kind != "identifier" or self.token.value not in self._TYPE_STARTERS:
            return False
        if self.token.value in {"long", "int", "const", "Py_ssize_t"}:
            return True
        # 'PyObject x' is not a type py2bin accepts, but 'PyObject *x' is, and
        # only the pointer form can be told apart from an expression here.
        return self.tokens[self.index + 1].value == "*"

    def type_name(self) -> str:
        start = self.token
        while self.accept("const"):  # accept() consumes the token it matches
            start = self.token
        if self.accept("long"):
            self.take("long")
            if self._pointer_suffix():
                # A 'long long *' would have to be dereferenced to be useful,
                # and this backend has no load/store through arbitrary
                # pointers. Only opaque handles below may be pointers.
                self.error("pointer types are not in the canonical C subset", start)
            return "i64"
        if self.accept("Py_ssize_t"):
            if self._pointer_suffix():
                self.error("pointer types are not in the canonical C subset", start)
            return "i64"
        if self.accept("int"):
            if self._pointer_suffix():
                self.error("pointer types are not in the canonical C subset", start)
            return "int"
        if self.token.kind == "identifier" and str(self.token.value) in self._HANDLE_TYPES:
            self.take()
            if self._pointer_suffix():
                return "ptr"
            if str(start.value) == "void":
                # A bare 'void' is a return type or an empty parameter list.
                return "void"
            self.error(
                f"{start.value} must be used as a pointer handle, "
                f"such as '{start.value} *'",
                start,
            )
        self.error(
            "canonical native C accepts 'long long', 'int main(void)', and "
            "pointer handles such as 'PyObject *'",
            start,
        )

    def _pointer_suffix(self) -> bool:
        """Consume any run of '*' and report whether the type is a pointer."""

        pointer = False
        while self.accept("*"):  # accept() consumes the token it matches
            pointer = True
        return pointer

    @staticmethod
    def arguments(parameters: list[str]) -> ast.arguments:
        return ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg=name, annotation=ast.Name(id="int", ctx=ast.Load()))
                for name in parameters
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )

    def module(self) -> ast.Module:
        functions: list[ast.FunctionDef] = []
        main_body: list[ast.stmt] | None = None
        while self.token.kind != "eof":
            if self.accept("extern"):
                self.extern_prototype()
                continue
            result_type = self.type_name()
            name_token = self.identifier()
            name = str(name_token.value)
            if name in self.function_names or (name == "main" and main_body is not None):
                self.error(f"duplicate function {name!r}", name_token)
            self.take("(")
            parameters: list[str] = []
            parameter_types: list[str] = []
            if self.accept("void"):
                self.take(")")
            elif self.accept(")"):
                pass
            else:
                while True:
                    parameter_types.append(self.type_name())
                    parameters.append(str(self.identifier().value))
                    if self.accept(")"):
                        break
                    self.take(",")
            if name == "main":
                if result_type != "int" or parameters:
                    self.error(
                        "entrypoint must have the exact form int main(void)",
                        name_token,
                    )
                result_kind = "i64"
            else:
                if result_type not in {"i64", "ptr", "void"} or any(
                    kind not in {"i64", "ptr"} for kind in parameter_types
                ):
                    self.error(
                        "native helper functions take 'long long' and pointer-handle "
                        "parameters and return 'long long', a pointer handle, or void",
                        name_token,
                    )
                result_kind = result_type
                # Registering the signature before the body is parsed is what
                # lets a function's own name resolve inside it; the native
                # frontend still rejects actual recursion.
                self.signatures[name] = (tuple(parameter_types), result_kind)
                self.function_names.add(name)
            previous_function = self.current_function
            previous_locals = self.locals
            previous_result = self.result_kind
            self.current_function = name
            self.locals = dict(zip(parameters, parameter_types))
            self.result_kind = result_kind
            body = self.block()
            self.current_function = previous_function
            self.locals = previous_locals
            self.result_kind = previous_result
            if not body:
                body = [ast.Pass()]
            if name == "main":
                main_body = body
            else:
                functions.append(
                    ast.FunctionDef(
                        name=name,
                        args=self.arguments(parameters),
                        body=body,
                        decorator_list=[],
                        returns=ast.Name(id="int", ctx=ast.Load()),
                        type_comment=None,
                    )
                )
        if main_body is None:
            self.error("translation unit needs exactly one int main(void)")
        prologue: list[ast.stmt] = []
        if self.externs:
            # Declared externs become the same adapter-ABI import the Python
            # frontend already lowers to real dyld-bound ExternCall operations.
            prologue.append(
                ast.ImportFrom(
                    module="py2bin.cabi",
                    names=[
                        ast.alias(name=symbol, asname=None)
                        for symbol in sorted(self.externs)
                    ],
                    level=0,
                )
            )
        tree = ast.Module(body=[*prologue, *functions, *main_body], type_ignores=[])
        self.validate_calls(tree)
        return ast.fix_missing_locations(tree)

    def validate_calls(self, tree: ast.AST) -> None:
        generated = {"SystemExit", "print", "range"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in self.function_names | generated | self.externs:
                self.error(
                    f"call to {node.func.id!r} is outside the canonical native C subset"
                )

    def block(self) -> list[ast.stmt]:
        self.take("{")
        result: list[ast.stmt] = []
        while not self.accept("}"):
            if self.token.kind == "eof":
                self.error("unterminated block")
            statement = self.statement()
            if statement is not None:
                result.append(statement)
        return result

    def statement(self) -> ast.stmt | None:
        if self.accept(";"):
            return None
        if self.at_declaration():
            return self.declaration()
        if self.accept("if"):
            self.take("(")
            token = self.token
            test = self.expression()
            self.value_kind(test, token)
            self.take(")")
            body = self.block()
            alternative = self.block() if self.accept("else") else []
            return ast.If(test=test, body=body or [ast.Pass()], orelse=alternative)
        if self.accept("while"):
            self.take("(")
            token = self.token
            test = self.expression()
            self.value_kind(test, token)
            self.take(")")
            return ast.While(test=test, body=self.block() or [ast.Pass()], orelse=[])
        if self.accept("for"):
            return self.for_statement()
        if self.accept("return"):
            return self.return_statement()
        if self.accept("break"):
            self.take(";")
            return ast.Break()
        if self.accept("continue"):
            self.take(";")
            return ast.Continue()
        if self.at("printf"):
            return self.printf_statement()
        if (
            self.token.kind == "identifier"
            and self.tokens[self.index + 1].value
            in {"=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^="}
        ):
            return self.assignment()
        expression = self.expression()
        self.take(";")
        return ast.Expr(value=expression)

    def extern_prototype(self) -> None:
        """Record ``extern TYPE name(TYPE, ...);`` as an external native symbol.

        Declaring prototypes explicitly is what removes the need for a C
        preprocessor and for parsing ``Python.h``: the generated C states the
        exact ABI it uses, so this compiler never sees a macro or a struct
        definition. Each recorded symbol is bound by the real dynamic linker.
        """

        # 'extern' has already been consumed by the caller's accept().
        result_type = self.type_name()
        name_token = self.identifier()
        name = str(name_token.value)
        self.take("(")
        declared: list[str] = []
        # accept() consumes on success, so an empty or (void) list closes here.
        if not self.accept(")"):
            while True:
                parameter_type = self.type_name()
                if parameter_type != "void":
                    declared.append(parameter_type)
                # An optional parameter name may follow the type.
                if self.token.kind == "identifier":
                    self.take()
                if self.accept(")"):
                    break
                self.take(",")
        self.take(";")
        if name not in _EXTERN_SYMBOLS:
            self.error(
                f"external symbol {name!r} is not in py2bin's vetted adapter ABI; "
                f"choose one of {', '.join(sorted(_EXTERN_SYMBOLS))}",
                name_token,
            )
        # The vetted signature -- not the prototype -- is what the call is
        # lowered against, so a prototype that disagrees with it would make the
        # C read as if it had an ABI the compiler will not emit. Reject the
        # disagreement instead of quietly preferring one of the two.
        _symbol, signature = _EXTERN_SYMBOLS[name]
        if len(declared) != len(signature):
            self.error(
                f"prototype for {name!r} declares {len(declared)} parameter(s) but "
                f"its vetted adapter ABI takes {len(signature)}",
                name_token,
            )
        for position, (declared_type, abi_kind) in enumerate(zip(declared, signature), 1):
            if declared_type not in _ABI_TO_C[abi_kind]:
                self.error(
                    f"parameter {position} of {name!r} is declared "
                    f"{declared_type!r} but its vetted adapter ABI passes "
                    f"{abi_kind!r}",
                    name_token,
                )
        if result_type not in _ABI_TO_C[_EXTERN_RESULTS[name]]:
            self.error(
                f"prototype for {name!r} returns {result_type!r} but its vetted "
                f"adapter ABI returns {_EXTERN_RESULTS[name]!r}",
                name_token,
            )
        self.externs.add(name)

    def declaration(self) -> ast.stmt | None:
        token = self.token
        kind = self.type_name()
        if kind not in {"i64", "ptr"}:
            self.error(
                "only 'long long' and pointer-handle local variables are accepted",
                token,
            )
        name_token = self.identifier()
        name = str(name_token.value)
        if name in self.locals:
            self.error(f"local {name!r} is already declared", name_token)
        self.locals[name] = kind
        if self.accept(";"):
            return None
        self.take("=")
        value_token = self.token
        value = self.expression()
        self.take(";")
        self.require_kind(value, kind, f"initializer for {name!r}", value_token)
        return ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=value,
        )

    def assignment(self) -> ast.stmt:
        name_token = self.identifier()
        name = str(name_token.value)
        operator = self.take()
        value_token = self.token
        value = self.expression()
        self.take(";")
        target = ast.Name(id=name, ctx=ast.Store())
        declared = self.locals.get(name, "i64")
        if operator.value == "=":
            self.require_kind(value, declared, f"assignment to {name!r}", value_token)
            return ast.Assign(targets=[target], value=value)
        if operator.value in {"/=", "%="}:
            self.error(
                "division and modulo are not implemented by the native C backend",
                operator,
            )
        if declared == "ptr":
            self.error(
                "pointer arithmetic is not in the canonical C subset; a handle is "
                "opaque and is never dereferenced or offset",
                operator,
            )
        self.require_kind(value, "i64", f"compound assignment to {name!r}", value_token)
        operation = self._AUGMENTED.get(str(operator.value))
        if operation is None:
            self.error(f"unsupported assignment operator {operator.value!r}", operator)
        return ast.AugAssign(target=target, op=operation(), value=value)

    def return_statement(self) -> ast.stmt:
        if self.at(";"):
            if self.result_kind != "void":
                self.error("a value-returning function needs 'return <expression>;'")
            self.take(";")
            return ast.Return(value=None)
        if self.result_kind == "void":
            self.error("a void function cannot return a value")
        value_token = self.token
        value = self.expression()
        self.take(";")
        self.require_kind(value, self.result_kind, "return", value_token)
        if self.current_function == "main":
            return ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="SystemExit", ctx=ast.Load()),
                    args=[value],
                    keywords=[],
                ),
                cause=None,
            )
        return ast.Return(value=value)

    @staticmethod
    def same_expression(left: ast.AST, right: ast.AST) -> bool:
        return ast.dump(left, include_attributes=False) == ast.dump(
            right,
            include_attributes=False,
        )

    @staticmethod
    def comparison(
        node: ast.expr,
        operator: type[ast.cmpop],
    ) -> tuple[ast.expr, ast.expr] | None:
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], operator)
            and len(node.comparators) == 1
        ):
            return node.left, node.comparators[0]
        return None

    def for_statement(self) -> ast.For:
        self.take("(")
        name_token = self.identifier()
        name = str(name_token.value)
        self.take("=")
        start_token = self.token
        start = self.expression()
        self.require_kind(start, "i64", "for initializer", start_token)
        self.locals.setdefault(name, "i64")
        if self.locals[name] != "i64":
            self.error("a for loop counter must be 'long long'", name_token)
        self.take(";")
        condition = self.expression()
        self.take(";")
        increment_name = str(self.identifier().value)
        self.take("+=")
        increment_token = self.token
        increment = self.expression()
        self.require_kind(increment, "i64", "for increment", increment_token)
        self.take(")")
        if not isinstance(condition, ast.IfExp):
            self.error(
                "for condition must use py2bin's canonical signed-step ternary",
                name_token,
            )
        step_test = self.comparison(condition.test, ast.GtE)
        positive_test = self.comparison(condition.body, ast.Lt)
        negative_test = self.comparison(condition.orelse, ast.Gt)
        if step_test is None or positive_test is None or negative_test is None:
            self.error(
                "for condition must be '(step) >= 0 ? name < stop : name > stop'",
                name_token,
            )
        test_step, zero = step_test
        positive_name, positive_stop = positive_test
        negative_name, negative_stop = negative_test
        valid_names = (
            isinstance(positive_name, ast.Name)
            and positive_name.id == name
            and isinstance(negative_name, ast.Name)
            and negative_name.id == name
            and increment_name == name
        )
        valid_values = (
            isinstance(zero, ast.Constant)
            and zero.value == 0
            and self.same_expression(test_step, increment)
            and self.same_expression(positive_stop, negative_stop)
        )
        if not valid_names or not valid_values:
            self.error(
                "for initializer, tests, and increment do not describe one canonical range",
                name_token,
            )
        iterator = ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[start, positive_stop, increment],
            keywords=[],
        )
        return ast.For(
            target=ast.Name(id=name, ctx=ast.Store()),
            iter=iterator,
            body=self.block() or [ast.Pass()],
            orelse=[],
        )

    def printf_statement(self) -> ast.Expr:
        token = self.take("printf")
        self.take("(")
        if self.token.kind != "string":
            self.error("native printf accepts one literal format string", self.token)
        raw = str(self.take().value)
        arguments: list[ast.expr] = []
        while self.accept(","):
            argument_token = self.token
            argument = self.expression()
            self.require_kind(argument, "i64", "a printf %lld argument", argument_token)
            arguments.append(argument)
        self.take(")")
        self.take(";")
        if not raw.endswith("\n"):
            self.error("native printf must end in a newline", token)
        raw = raw[:-1]
        parts: list[ast.expr] = []
        text: list[str] = []
        argument_index = 0

        def flush_text() -> None:
            if text:
                parts.append(ast.Constant(value="".join(text)))
                text.clear()

        index = 0
        while index < len(raw):
            if raw[index] != "%":
                text.append(raw[index])
                index += 1
                continue
            if index + 1 < len(raw) and raw[index + 1] == "%":
                text.append("%")
                index += 2
                continue
            if raw.startswith("%lld", index):
                if argument_index >= len(arguments):
                    self.error("printf has fewer arguments than %lld conversions", token)
                flush_text()
                parts.append(
                    ast.FormattedValue(
                        value=arguments[argument_index],
                        conversion=-1,
                        format_spec=None,
                    )
                )
                argument_index += 1
                index += 4
                continue
            self.error(
                "only %% and compile-time integer %lld printf conversions are native",
                token,
            )
        flush_text()
        if argument_index != len(arguments):
            self.error("printf has more arguments than %lld conversions", token)
        if not parts:
            value: ast.expr = ast.Constant(value="")
        elif all(isinstance(part, ast.Constant) for part in parts):
            value = ast.Constant(
                value="".join(str(part.value) for part in parts)
            )
        else:
            value = ast.JoinedStr(values=parts)
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[value],
                keywords=[],
            )
        )

    def truth_operand(self, node: ast.expr, token: _Token) -> None:
        """Validate an expression used as a condition.

        Both integers and handles are legal here: C's truth test is "not zero",
        which is exactly what a NULL check on a handle means, and it is exactly
        what the backend's ``JumpIfFalse`` lowering computes.
        """

        if self.value_kind(node, token) == "cstr":
            self.error("a string literal is not a condition", token)

    def expression(self, minimum_precedence: int = 0) -> ast.expr:
        left_token = self.token
        left = self.unary()
        while True:
            operator = str(self.token.value)
            precedence = self._PRECEDENCE.get(operator)
            if precedence is None or precedence < minimum_precedence:
                break
            token = self.take()
            right_token = self.token
            right = self.expression(precedence + 1)
            if operator in {"/", "%"}:
                self.error(
                    "division and modulo are not implemented by the native C backend",
                    token,
                )
            if operator in self._BINARY:
                self.require_kind(left, "i64", f"the left operand of {operator!r}", left_token)
                self.require_kind(right, "i64", f"the right operand of {operator!r}", right_token)
                left = ast.BinOp(
                    left=left,
                    op=self._BINARY[operator](),
                    right=right,
                )
            elif operator in self._COMPARE:
                self.compare_operands(left, right, operator, token, left_token, right_token)
                left = ast.Compare(
                    left=left,
                    ops=[self._COMPARE[operator]()],
                    comparators=[right],
                )
            elif operator == "&&":
                self.truth_operand(left, left_token)
                self.truth_operand(right, right_token)
                left = ast.IfExp(
                    test=left,
                    body=ast.IfExp(
                        test=right,
                        body=ast.Constant(value=1),
                        orelse=ast.Constant(value=0),
                    ),
                    orelse=ast.Constant(value=0),
                )
            elif operator == "||":
                self.truth_operand(left, left_token)
                self.truth_operand(right, right_token)
                left = ast.IfExp(
                    test=left,
                    body=ast.Constant(value=1),
                    orelse=ast.IfExp(
                        test=right,
                        body=ast.Constant(value=1),
                        orelse=ast.Constant(value=0),
                    ),
                )
            else:
                self.error(f"unsupported binary operator {operator!r}", token)
            self.mark(left, "i64")
        if minimum_precedence == 0 and self.accept("?"):
            self.truth_operand(left, left_token)
            body_token = self.token
            body = self.expression()
            self.take(":")
            alternative_token = self.token
            alternative = self.expression()
            body_kind = self.value_kind(body, body_token)
            alternative_kind = self.value_kind(alternative, alternative_token)
            if body_kind != alternative_kind and not (
                {body_kind, alternative_kind} == {"ptr", "i64"}
                and (self.is_zero(body) or self.is_zero(alternative))
            ):
                self.error(
                    "both arms of '?:' must have the same type", alternative_token
                )
            left = self.mark(
                ast.IfExp(test=left, body=body, orelse=alternative),
                "ptr" if "ptr" in {body_kind, alternative_kind} else body_kind,
            )
        return left

    def compare_operands(
        self,
        left: ast.expr,
        right: ast.expr,
        operator: str,
        token: _Token,
        left_token: _Token,
        right_token: _Token,
    ) -> None:
        left_kind = self.value_kind(left, left_token)
        right_kind = self.value_kind(right, right_token)
        if "cstr" in {left_kind, right_kind}:
            self.error("string literals cannot be compared", token)
        handle = "ptr" in {left_kind, right_kind}
        if not handle:
            return
        if operator not in {"==", "!="}:
            self.error(
                "handles support only '==' and '!=' comparisons; they are opaque "
                "and have no defined order",
                token,
            )
        if left_kind != right_kind and not (self.is_zero(left) or self.is_zero(right)):
            self.error(
                "a handle can only be compared with another handle or with the "
                "null pointer constant 0",
                token,
            )

    def unary(self) -> ast.expr:
        if self.token.value in {"+", "-", "!", "~"}:
            token = self.take()
            operand_token = self.token
            operand = self.unary()
            operation = {
                "+": ast.UAdd,
                "-": ast.USub,
                "!": ast.Not,
                "~": ast.Invert,
            }[str(token.value)]
            if token.value == "!":
                self.truth_operand(operand, operand_token)
            else:
                self.require_kind(
                    operand, "i64", f"the operand of unary {token.value!r}", operand_token
                )
            return self.mark(ast.UnaryOp(op=operation(), operand=operand), "i64")
        return self.primary()

    def call(self, name: str, name_token: _Token) -> ast.expr:
        """Parse an argument list and check it against the callee's signature."""

        arguments: list[ast.expr] = []
        argument_tokens: list[_Token] = []
        # accept() consumes on success, so an empty argument list closes here.
        if not self.accept(")"):
            while True:
                argument_tokens.append(self.token)
                arguments.append(self.expression())
                if self.accept(")"):
                    break
                self.take(",")
        if name in self.externs:
            _symbol, signature = _EXTERN_SYMBOLS[name]
            result = _EXTERN_RESULTS[name]
        elif name in self.signatures:
            signature, result = self.signatures[name]
        else:
            # 'printf' has its own statement form and everything else is
            # rejected by validate_calls once the whole unit is parsed.
            return self.mark(
                ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=arguments, keywords=[]),
                "i64",
            )
        if len(arguments) != len(signature):
            self.error(
                f"{name}() takes {len(signature)} argument(s), got {len(arguments)}",
                name_token,
            )
        for position, (argument, expected) in enumerate(zip(arguments, signature), 1):
            self.require_kind(
                argument, expected, f"argument {position} of {name}()", argument_tokens[position - 1]
            )
        return self.mark(
            ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=arguments, keywords=[]),
            {"int": "i64"}.get(result, result),
        )

    def primary(self) -> ast.expr:
        token = self.token
        if token.kind == "integer":
            self.take()
            return self.mark(ast.Constant(value=token.value), "i64")
        if token.kind == "string":
            # A string literal is a char* handed to an external function, which
            # is how generated C-API code passes source text and names.
            self.take()
            return self.mark(ast.Constant(value=token.value), "cstr")
        if token.kind == "identifier":
            name = str(self.take().value)
            if not self.accept("("):
                if name == "NULL" and name not in self.locals:
                    # The null pointer constant. Spelling it as a keyword of
                    # this dialect is what keeps py2bin free of a preprocessor:
                    # <stddef.h> is never read, and no macro is ever expanded.
                    return self.mark(ast.Constant(value=0), "ptr")
                if name not in self.locals:
                    self.error(f"{name!r} is not a declared local or parameter", token)
                return self.mark(ast.Name(id=name, ctx=ast.Load()), self.locals[name])
            return self.call(name, token)
        if self.accept("("):
            value = self.expression()
            self.take(")")
            return value
        self.error(f"expected integer expression, found {token.value!r}", token)


def parse_canonical_c(source: str, filename: str = "<string>") -> ast.Module:
    """Parse py2bin canonical integer C without invoking another compiler."""
    return _Parser(_Lexer(source, filename).tokens(), filename).module()


def c_to_python_source(source: str, filename: str = "<string>") -> str:
    """Return the verified intermediate source consumed by the native frontend."""
    return ast.unparse(parse_canonical_c(source, filename)) + "\n"


def compile_c_native(
    entry: Path,
    output: Path,
    *,
    target: str | None = None,
    clean: bool = False,
) -> NativeResult:
    """Compile a canonical C file with only py2bin's Python implementation."""
    entry = entry.expanduser().resolve()
    if not entry.is_file():
        raise FileNotFoundError(f"C source does not exist: {entry}")
    source = entry.read_text(encoding="utf-8")
    reconstructed = c_to_python_source(source, str(entry))
    return compile_native_source(
        entry,
        reconstructed,
        output,
        target=target,
        clean=clean,
    )


def compile_python_via_c(
    entry: Path,
    output: Path,
    *,
    target: str | None = None,
    clean: bool = False,
    c_output: Path | None = None,
) -> CBridgeResult:
    """Generate C, parse that C, and write native code with py2bin backends."""
    entry = entry.expanduser().resolve()
    output = output.expanduser().resolve()
    if not entry.is_file():
        raise FileNotFoundError(f"entry script does not exist: {entry}")
    c_artifact = c_output.expanduser().resolve() if c_output is not None else None
    if c_artifact == output:
        raise ValueError("--c-output and native --output must be different paths")
    if c_artifact is not None and c_artifact.exists() and not clean:
        raise FileExistsError(f"output already exists: {c_artifact} (use --clean)")
    c_source = compile_to_c(entry.read_text(encoding="utf-8"), str(entry))
    reconstructed = c_to_python_source(c_source, f"{entry} [generated C]")
    native = compile_native_source(
        entry,
        reconstructed,
        output,
        target=target,
        clean=clean,
    )
    if c_artifact is not None:
        c_artifact.parent.mkdir(parents=True, exist_ok=True)
        c_artifact.write_text(c_source, encoding="utf-8", newline="\n")
    return CBridgeResult(native, c_source, reconstructed, c_artifact)


__all__ = [
    "CBridgeResult",
    "CNativeCompileError",
    "c_to_python_source",
    "compile_c_native",
    "compile_python_via_c",
    "parse_canonical_c",
]
