from __future__ import annotations

import ast
from pathlib import Path

from .ir import Exit, Module, Write


class NativeCompileError(ValueError):
    def __init__(self, path: Path, node: ast.AST, message: str):
        line = getattr(node, "lineno", 1)
        column = getattr(node, "col_offset", 0) + 1
        super().__init__(f"{path}:{line}:{column}: {message}")


class Frontend:
    """Lower the first useful static-Python subset into portable native IR.

    The deliberately small initial subset accepts constant assignments,
    compile-time arithmetic/string expressions, print(), pass, and
    SystemExit. Unsupported dynamic semantics fail loudly instead of silently
    producing a wrong executable.
    """

    def __init__(self, path: Path):
        self.path = path
        self.values: dict[str, object] = {}
        self.operations = []

    def compile(self, source: str) -> Module:
        try:
            tree = ast.parse(source, filename=str(self.path))
        except SyntaxError as error:
            raise ValueError(f"{self.path}:{error.lineno}:{error.offset}: {error.msg}") from error
        for statement in tree.body:
            self.statement(statement)
        if not self.operations or not isinstance(self.operations[-1], Exit):
            self.operations.append(Exit(0))
        return Module(self.operations)

    def statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return  # Module docstring.
            self.expression_statement(node.value)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.values[node.targets[0].id] = self.constant(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            self.values[node.target.id] = self.constant(node.value)
        elif isinstance(node, ast.Pass):
            return
        elif isinstance(node, ast.Import) and all(alias.name == "sys" for alias in node.names):
            return  # sys.exit is lowered as a native operation below.
        elif isinstance(node, ast.Raise) and node.exc:
            self.system_exit(node.exc, node)
        else:
            raise NativeCompileError(
                self.path,
                node,
                f"{type(node).__name__} is not in the native subset yet; use bundle mode for full CPython semantics",
            )

    def expression_statement(self, node: ast.expr) -> None:
        if not isinstance(node, ast.Call):
            raise NativeCompileError(self.path, node, "only print() and SystemExit are valid expression statements")
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            if node.keywords:
                raise NativeCompileError(self.path, node, "native print() does not support keyword arguments yet")
            values = [self.constant(argument) for argument in node.args]
            text = " ".join(str(value) for value in values) + "\n"
            self.operations.append(Write(text.encode("utf-8")))
        elif self.is_exit_call(node):
            self.system_exit(node, node)
        else:
            raise NativeCompileError(self.path, node, "only print() and SystemExit are callable in the native subset")

    @staticmethod
    def is_exit_call(node: ast.Call) -> bool:
        return (
            isinstance(node.func, ast.Name)
            and node.func.id in {"exit", "SystemExit"}
        ) or (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sys"
            and node.func.attr == "exit"
        )

    def system_exit(self, expression: ast.expr, location: ast.AST) -> None:
        call = expression if isinstance(expression, ast.Call) else None
        if not call or not self.is_exit_call(call) or len(call.args) > 1 or call.keywords:
            raise NativeCompileError(self.path, location, "expected SystemExit(integer) or sys.exit(integer)")
        status = self.constant(call.args[0]) if call.args else 0
        if not isinstance(status, int):
            raise NativeCompileError(self.path, call, "exit status must be an integer constant")
        self.operations.append(Exit(status & 0xFF))

    def constant(self, node: ast.expr) -> object:
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes, int, float, bool, type(None))):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.values:
            return self.values[node.id]
        if isinstance(node, ast.UnaryOp):
            value = self.constant(node.operand)
            if isinstance(node.op, ast.USub) and isinstance(value, (int, float)):
                return -value
            if isinstance(node.op, ast.UAdd) and isinstance(value, (int, float)):
                return +value
            if isinstance(node.op, ast.Not):
                return not value
        if isinstance(node, ast.BinOp):
            left, right = self.constant(node.left), self.constant(node.right)
            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
                if isinstance(node.op, ast.Mod):
                    return left % right
                if isinstance(node.op, ast.Pow):
                    return left**right
            except (TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
                raise NativeCompileError(self.path, node, f"constant expression failed: {error}") from error
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for item in node.values:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    parts.append(item.value)
                elif isinstance(item, ast.FormattedValue):
                    parts.append(str(self.constant(item.value)))
                else:
                    raise NativeCompileError(self.path, item, "unsupported f-string component")
            return "".join(parts)
        raise NativeCompileError(self.path, node, "expression is not compile-time constant")


def lower(path: Path, source: str) -> Module:
    return Frontend(path).compile(source)
