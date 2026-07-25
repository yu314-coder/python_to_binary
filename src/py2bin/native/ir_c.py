"""Canonical C serialization for py2bin's complete native integer IR.

The output is valid, deliberately simple C.  It preserves stack slots, labels,
branches, byte writes, and process returns after local Python functions have
already been lowered and inlined.  The parser below is handwritten in Python
and reconstructs py2bin IR; no external C implementation is involved.
"""

from __future__ import annotations

import ast
import re

from .ir import (
    Exit,
    ExitValue,
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
    Store,
    Write,
)


_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_SLOT = re.compile(r"long long py2bin_slot_([0-9]+);$")
_STORE = re.compile(r"py2bin_slot_([0-9]+) = (.+);$")
_LABEL = re.compile(rf"py2bin_label_({_IDENTIFIER}):$")
_JUMP = re.compile(rf"goto py2bin_label_({_IDENTIFIER});$")
_JUMP_FALSE = re.compile(
    rf"if \(\((.+)\) == 0\) goto py2bin_label_({_IDENTIFIER});$"
)
_WRITE = re.compile(
    r'fwrite\("((?:\\x[0-9A-Fa-f]{2})+)", 1, ([0-9]+), stdout\);$'
)
_RETURN = re.compile(r"return (.+);$")
_LABEL_NAME = re.compile(rf"{_IDENTIFIER}$")


class IRCanonicalCError(ValueError):
    """A source-located rejection from the canonical IR-C parser."""

    def __init__(self, filename: str, line: int, message: str):
        self.filename = filename
        self.line = line
        self.message = message
        super().__init__(f"{filename}:{line}: {message}")


def _expression_to_c(expression: IntExpression) -> str:
    if isinstance(expression, IntConstant):
        return str(expression.value)
    if isinstance(expression, IntLoad):
        return f"py2bin_slot_{expression.slot}"
    if isinstance(expression, IntUnary):
        operand = _expression_to_c(expression.operand)
        operators = {
            "pos": "+",
            "neg": "-",
            "invert": "~",
        }
        if expression.operator == "not":
            return f"py2bin_not({operand})"
        operator = operators.get(expression.operator)
        if operator is None:
            raise ValueError(f"unknown native unary operation {expression.operator!r}")
        return f"({operator}({operand}))"
    if isinstance(expression, IntBinary):
        operators = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "and": "&",
            "or": "|",
            "xor": "^",
            "lshift": "<<",
            "rshift": ">>",
        }
        operator = operators.get(expression.operator)
        if operator is None:
            raise ValueError(f"unknown native binary operation {expression.operator!r}")
        return (
            f"(({_expression_to_c(expression.left)}) {operator} "
            f"({_expression_to_c(expression.right)}))"
        )
    if isinstance(expression, IntCompare):
        operators = {
            "eq": "==",
            "ne": "!=",
            "lt": "<",
            "le": "<=",
            "gt": ">",
            "ge": ">=",
        }
        operator = operators.get(expression.operator)
        if operator is None:
            raise ValueError(f"unknown native comparison {expression.operator!r}")
        return (
            f"(({_expression_to_c(expression.left)}) {operator} "
            f"({_expression_to_c(expression.right)}))"
        )
    raise TypeError(f"unknown native integer expression {type(expression).__name__}")


def _bytes_literal(data: bytes) -> str:
    return "".join(f"\\x{value:02x}" for value in data)


def emit_ir_c(module: Module) -> str:
    """Serialize a complete optimized native module as deterministic C."""

    lines = [
        "/* py2bin canonical native IR C v1 */",
        "#include <stdio.h>",
        "#define py2bin_not(value) ((value) == 0)",
        "",
        "int main(void) {",
    ]
    for slot in range(module.stack_slots):
        lines.append(f"    long long py2bin_slot_{slot};")
    if module.stack_slots:
        lines.append("")
    for operation in module.operations:
        if isinstance(operation, Write):
            if not operation.data:
                continue
            literal = _bytes_literal(operation.data)
            lines.append(
                f'    fwrite("{literal}", 1, {len(operation.data)}, stdout);'
            )
        elif isinstance(operation, Store):
            lines.append(
                f"    py2bin_slot_{operation.slot} = "
                f"{_expression_to_c(operation.value)};"
            )
        elif isinstance(operation, Label):
            if _LABEL_NAME.fullmatch(operation.name) is None:
                raise ValueError(f"native IR label is not a C identifier: {operation.name!r}")
            lines.append(f"py2bin_label_{operation.name}:")
        elif isinstance(operation, Jump):
            if _LABEL_NAME.fullmatch(operation.target) is None:
                raise ValueError(f"native IR label is not a C identifier: {operation.target!r}")
            lines.append(f"    goto py2bin_label_{operation.target};")
        elif isinstance(operation, JumpIfFalse):
            if _LABEL_NAME.fullmatch(operation.target) is None:
                raise ValueError(f"native IR label is not a C identifier: {operation.target!r}")
            lines.append(
                f"    if (({_expression_to_c(operation.condition)}) == 0) "
                f"goto py2bin_label_{operation.target};"
            )
        elif isinstance(operation, Exit):
            lines.append(f"    return {operation.status};")
        elif isinstance(operation, ExitValue):
            lines.append(f"    return {_expression_to_c(operation.value)};")
        else:
            raise TypeError(f"unknown native IR operation {type(operation).__name__}")
    lines.append("}")
    return "\n".join(lines) + "\n"


class _ExpressionReader:
    _BINARY = {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "mul",
        ast.BitAnd: "and",
        ast.BitOr: "or",
        ast.BitXor: "xor",
        ast.LShift: "lshift",
        ast.RShift: "rshift",
    }
    _UNARY = {
        ast.UAdd: "pos",
        ast.USub: "neg",
        ast.Invert: "invert",
        ast.Not: "not",
    }
    _COMPARE = {
        ast.Eq: "eq",
        ast.NotEq: "ne",
        ast.Lt: "lt",
        ast.LtE: "le",
        ast.Gt: "gt",
        ast.GtE: "ge",
    }

    def __init__(self, filename: str, line: int, stack_slots: int):
        self.filename = filename
        self.line = line
        self.stack_slots = stack_slots

    def error(self, message: str) -> None:
        raise IRCanonicalCError(self.filename, self.line, message)

    def read(self, source: str) -> IntExpression:
        try:
            tree = ast.parse(source, filename=self.filename, mode="eval")
        except SyntaxError as error:
            self.error(f"invalid canonical integer expression: {error.msg}")
        return self.convert(tree.body)

    def convert(self, node: ast.expr) -> IntExpression:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            return IntConstant(node.value)
        if isinstance(node, ast.Name):
            match = re.fullmatch(r"py2bin_slot_([0-9]+)", node.id)
            if match is None:
                self.error(f"unknown canonical value {node.id!r}")
            slot = int(match.group(1))
            if slot >= self.stack_slots:
                self.error(f"reference to undeclared native slot {slot}")
            return IntLoad(slot)
        if isinstance(node, ast.UnaryOp):
            operator = self._UNARY.get(type(node.op))
            if operator is None:
                self.error("unsupported canonical unary operation")
            return IntUnary(operator, self.convert(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "py2bin_not"
            and len(node.args) == 1
            and not node.keywords
        ):
            return IntUnary("not", self.convert(node.args[0]))
        if isinstance(node, ast.BinOp):
            operator = self._BINARY.get(type(node.op))
            if operator is None:
                self.error("unsupported canonical binary operation")
            right = self.convert(node.right)
            if operator in {"lshift", "rshift"} and (
                not isinstance(right, IntConstant) or not 0 <= right.value <= 63
            ):
                self.error("canonical shift count must be an integer from 0 to 63")
            return IntBinary(operator, self.convert(node.left), right)
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and len(node.comparators) == 1
        ):
            operator = self._COMPARE.get(type(node.ops[0]))
            if operator is None:
                self.error("unsupported canonical comparison")
            return IntCompare(
                operator,
                self.convert(node.left),
                self.convert(node.comparators[0]),
            )
        self.error(f"unsupported canonical expression {type(node).__name__}")


def parse_ir_c(source: str, filename: str = "<canonical-ir-c>") -> Module:
    """Parse py2bin's canonical whole-program C back into native IR."""

    lines = source.splitlines()
    if not lines or lines[0].strip() != "/* py2bin canonical native IR C v1 */":
        raise IRCanonicalCError(filename, 1, "missing canonical IR-C v1 header")

    operations = []
    declared_slots: list[int] = []
    labels: set[str] = set()
    jump_targets: list[tuple[int, str]] = []
    saw_include = False
    saw_not_macro = False
    in_main = False
    closed = False

    for line_number, raw in enumerate(lines[1:], 2):
        line = raw.strip()
        if not line:
            continue
        if not saw_include:
            if line != "#include <stdio.h>":
                raise IRCanonicalCError(
                    filename,
                    line_number,
                    "canonical IR-C must include only <stdio.h>",
                )
            saw_include = True
            continue
        if not saw_not_macro:
            if line != "#define py2bin_not(value) ((value) == 0)":
                raise IRCanonicalCError(
                    filename,
                    line_number,
                    "canonical IR-C is missing the exact py2bin_not macro",
                )
            saw_not_macro = True
            continue
        if not in_main:
            if line != "int main(void) {":
                raise IRCanonicalCError(
                    filename,
                    line_number,
                    "expected exact entrypoint int main(void)",
                )
            in_main = True
            continue
        if line == "}":
            closed = True
            in_main = False
            continue
        if closed:
            raise IRCanonicalCError(
                filename,
                line_number,
                "content after canonical main function",
            )

        match = _SLOT.fullmatch(line)
        if match is not None:
            if operations:
                raise IRCanonicalCError(
                    filename,
                    line_number,
                    "native slot declarations must precede operations",
                )
            slot = int(match.group(1))
            if slot != len(declared_slots):
                raise IRCanonicalCError(
                    filename,
                    line_number,
                    f"native slots must be contiguous; expected {len(declared_slots)}",
                )
            declared_slots.append(slot)
            continue

        reader = _ExpressionReader(filename, line_number, len(declared_slots))
        match = _STORE.fullmatch(line)
        if match is not None:
            slot = int(match.group(1))
            if slot >= len(declared_slots):
                raise IRCanonicalCError(
                    filename,
                    line_number,
                    f"store to undeclared native slot {slot}",
                )
            operations.append(Store(slot, reader.read(match.group(2))))
            continue
        match = _LABEL.fullmatch(line)
        if match is not None:
            name = match.group(1)
            if name in labels:
                raise IRCanonicalCError(
                    filename,
                    line_number,
                    f"duplicate native label {name!r}",
                )
            labels.add(name)
            operations.append(Label(name))
            continue
        match = _JUMP.fullmatch(line)
        if match is not None:
            target = match.group(1)
            jump_targets.append((line_number, target))
            operations.append(Jump(target))
            continue
        match = _JUMP_FALSE.fullmatch(line)
        if match is not None:
            target = match.group(2)
            jump_targets.append((line_number, target))
            operations.append(JumpIfFalse(reader.read(match.group(1)), target))
            continue
        match = _WRITE.fullmatch(line)
        if match is not None:
            encoded, declared_size = match.groups()
            data = bytes.fromhex(encoded.replace("\\x", ""))
            if len(data) != int(declared_size):
                raise IRCanonicalCError(
                    filename,
                    line_number,
                    "fwrite byte count does not match its canonical literal",
                )
            operations.append(Write(data))
            continue
        match = _RETURN.fullmatch(line)
        if match is not None:
            value = reader.read(match.group(1))
            if isinstance(value, IntConstant):
                operations.append(Exit(value.value))
            else:
                operations.append(ExitValue(value))
            continue
        raise IRCanonicalCError(
            filename,
            line_number,
            f"statement is outside canonical IR-C: {line!r}",
        )

    if not saw_include:
        raise IRCanonicalCError(filename, 1, "missing #include <stdio.h>")
    if not saw_not_macro:
        raise IRCanonicalCError(filename, 1, "missing py2bin_not macro")
    if in_main or not closed:
        raise IRCanonicalCError(filename, len(lines) or 1, "unterminated main function")
    for line_number, target in jump_targets:
        if target not in labels:
            raise IRCanonicalCError(
                filename,
                line_number,
                f"jump to undefined native label {target!r}",
            )
    if not operations:
        raise IRCanonicalCError(filename, len(lines) or 1, "canonical main has no operations")
    return Module(operations, len(declared_slots))


def roundtrip_ir_c(module: Module) -> tuple[str, Module]:
    """Emit and reparse canonical C, requiring exact IR preservation."""

    source = emit_ir_c(module)
    reconstructed = parse_ir_c(source)
    if reconstructed != module:
        raise ValueError("canonical IR-C round trip changed native semantics")
    return source, reconstructed


__all__ = [
    "IRCanonicalCError",
    "emit_ir_c",
    "parse_ir_c",
    "roundtrip_ir_c",
]
