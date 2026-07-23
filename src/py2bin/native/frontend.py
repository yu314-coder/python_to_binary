from __future__ import annotations

import ast
from pathlib import Path

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


class NativeCompileError(ValueError):
    def __init__(self, path: Path, node: ast.AST, message: str):
        line = getattr(node, "lineno", 1)
        column = getattr(node, "col_offset", 0) + 1
        super().__init__(f"{path}:{line}:{column}: {message}")


class Frontend:
    """Lower the first useful static-Python subset into portable native IR.

    The native subset accepts static string output plus a small signed 64-bit
    integer runtime: variables, arithmetic, comparisons, if/while,
    for-range, and process exit. Unsupported dynamic Python semantics fail
    loudly instead of silently producing a wrong executable.
    """

    def __init__(self, path: Path, source_roots: tuple[Path, ...] = ()):
        self.path = path
        self.source_roots = tuple(root.expanduser().resolve() for root in source_roots)
        self.values: dict[str, object] = {}
        self.operations = []
        self.slots: dict[str, int] = {}
        self.runtime_names: set[str] = set()
        self.label_number = 0
        self.break_targets: list[str] = []
        self.continue_targets: list[str] = []

    def compile(self, source: str) -> Module:
        try:
            tree = ast.parse(source, filename=str(self.path))
        except SyntaxError as error:
            raise ValueError(f"{self.path}:{error.lineno}:{error.offset}: {error.msg}") from error
        self.runtime_names.update(self.loop_mutated_names(tree))
        for statement in tree.body:
            self.statement(statement)
        if not self.operations or not isinstance(self.operations[-1], (Exit, ExitValue)):
            self.operations.append(Exit(0))
        return Module(self.operations, len(self.slots))

    @staticmethod
    def assigned_names(nodes: list[ast.stmt]) -> set[str]:
        names: set[str] = set()
        for statement in nodes:
            for node in ast.walk(statement):
                if isinstance(node, ast.Assign):
                    names.update(
                        target.id for target in node.targets if isinstance(target, ast.Name)
                    )
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names.add(node.target.id)
                elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    names.add(node.target.id)
                elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                    names.add(node.target.id)
        return names

    @classmethod
    def loop_mutated_names(cls, tree: ast.AST) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.For, ast.While)):
                names.update(cls.assigned_names(node.body))
                if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                    names.add(node.target.id)
        return names

    def new_label(self, prefix: str) -> str:
        self.label_number += 1
        return f"{prefix}_{self.label_number}"

    def slot(self, name: str) -> int:
        if name not in self.slots:
            self.slots[name] = len(self.slots)
        return self.slots[name]

    def statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return  # Module docstring.
            self.expression_statement(node.value)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.assignment(node.targets[0].id, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            self.assignment(node.target.id, node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            operators = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.LShift: "lshift",
                ast.RShift: "rshift",
                ast.BitAnd: "and",
                ast.BitOr: "or",
                ast.BitXor: "xor",
            }
            operator = operators.get(type(node.op))
            if operator is None:
                raise NativeCompileError(
                    self.path, node, "unsupported native integer augmented assignment"
                )
            name = node.target.id
            if name not in self.slots:
                raise NativeCompileError(
                    self.path, node, f"runtime integer variable {name!r} is not initialized"
                )
            self.values.pop(name, None)
            right = self.integer(node.value)
            if operator in {"lshift", "rshift"}:
                try:
                    shift = self.constant(node.value)
                except NativeCompileError as error:
                    raise NativeCompileError(
                        self.path,
                        node.value,
                        "native shift count must be an integer constant from 0 to 63",
                    ) from error
                if (
                    not isinstance(shift, int)
                    or isinstance(shift, bool)
                    or not 0 <= shift <= 63
                ):
                    raise NativeCompileError(
                        self.path,
                        node.value,
                        "native shift count must be an integer constant from 0 to 63",
                    )
                right = IntConstant(shift)
            self.operations.append(
                Store(
                    self.slots[name],
                    IntBinary(operator, IntLoad(self.slots[name]), right),
                )
            )
        elif isinstance(node, ast.If):
            self.if_statement(node)
        elif isinstance(node, ast.While):
            self.while_statement(node)
        elif isinstance(node, ast.For):
            self.for_statement(node)
        elif isinstance(node, ast.Break):
            if not self.break_targets:
                raise NativeCompileError(self.path, node, "break is outside a native loop")
            self.operations.append(Jump(self.break_targets[-1]))
        elif isinstance(node, ast.Continue):
            if not self.continue_targets:
                raise NativeCompileError(self.path, node, "continue is outside a native loop")
            self.operations.append(Jump(self.continue_targets[-1]))
        elif isinstance(node, ast.Pass):
            return
        elif isinstance(node, ast.ImportFrom):
            self.import_from(node)
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

    def source_candidate(self, module: str) -> Path | None:
        parts = module.split(".")
        for root in self.source_roots:
            module_path = root.joinpath(*parts).with_suffix(".py")
            package_path = root.joinpath(*parts, "__init__.py")
            if module_path.is_file():
                return module_path
            if package_path.is_file():
                return package_path
        return None

    def import_from(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module == "__future__":
            return
        if node.level or not node.module:
            raise NativeCompileError(
                self.path,
                node,
                "native locked-source imports must be absolute 'from MODULE import NAME'",
            )
        if any(alias.name == "*" for alias in node.names):
            raise NativeCompileError(
                self.path, node, "native locked-source imports do not support import *"
            )
        candidate = self.source_candidate(node.module)
        if candidate is None:
            raise NativeCompileError(
                self.path,
                node,
                f"locked source does not provide module {node.module!r}",
            )
        try:
            tree = ast.parse(
                candidate.read_text(encoding="utf-8"),
                filename=str(candidate),
            )
        except (OSError, SyntaxError, UnicodeDecodeError) as error:
            raise NativeCompileError(
                self.path,
                node,
                f"cannot parse locked source module {node.module!r}: {error}",
            ) from error
        provider = Frontend(candidate, self.source_roots)
        for statement in tree.body:
            try:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                ):
                    provider.values[statement.targets[0].id] = provider.constant(
                        statement.value
                    )
                elif (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.value is not None
                ):
                    provider.values[statement.target.id] = provider.constant(
                        statement.value
                    )
            except NativeCompileError:
                # Only requested, statically evaluable exports matter. Other
                # downloaded code is never executed during this inspection.
                continue
        for alias in node.names:
            if alias.name not in provider.values:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"locked source export {node.module}.{alias.name} is not a compile-time constant",
                )
            self.values[alias.asname or alias.name] = provider.values[alias.name]

    def assignment(self, name: str, expression: ast.expr) -> None:
        if name not in self.runtime_names:
            try:
                self.values[name] = self.constant(expression)
                return
            except NativeCompileError:
                self.runtime_names.add(name)
        self.values.pop(name, None)
        self.operations.append(Store(self.slot(name), self.integer(expression)))

    def materialize_runtime_names(self, names: set[str]) -> None:
        for name in sorted(names):
            if name in self.values:
                value = self.values.pop(name)
                if not isinstance(value, (int, bool)):
                    raise NativeCompileError(
                        self.path,
                        ast.Constant(value=value),
                        f"runtime variable {name!r} must be a signed 64-bit integer",
                    )
                self.operations.append(Store(self.slot(name), IntConstant(int(value))))
            self.runtime_names.add(name)

    def if_statement(self, node: ast.If) -> None:
        try:
            condition = self.constant(node.test)
        except NativeCompileError as constant_error:
            try:
                runtime_condition = self.integer(node.test)
            except NativeCompileError:
                raise constant_error
            mutated = self.assigned_names(node.body + node.orelse)
            self.materialize_runtime_names(mutated)
            false_label = self.new_label("if_false")
            end_label = self.new_label("if_end")
            self.operations.append(JumpIfFalse(runtime_condition, false_label))
            for statement in node.body:
                self.statement(statement)
            if node.orelse:
                self.operations.append(Jump(end_label))
            self.operations.append(Label(false_label))
            for statement in node.orelse:
                self.statement(statement)
            if node.orelse:
                self.operations.append(Label(end_label))
        else:
            branch = node.body if bool(condition) else node.orelse
            for statement in branch:
                self.statement(statement)

    def while_statement(self, node: ast.While) -> None:
        if node.orelse:
            raise NativeCompileError(self.path, node, "native while-else is not supported")
        start = self.new_label("while_start")
        end = self.new_label("while_end")
        self.operations.append(Label(start))
        self.operations.append(JumpIfFalse(self.integer(node.test), end))
        self.break_targets.append(end)
        self.continue_targets.append(start)
        for statement in node.body:
            self.statement(statement)
        self.continue_targets.pop()
        self.break_targets.pop()
        self.operations.append(Jump(start))
        self.operations.append(Label(end))

    def for_statement(self, node: ast.For) -> None:
        if (
            node.orelse
            or not isinstance(node.target, ast.Name)
            or not isinstance(node.iter, ast.Call)
            or not isinstance(node.iter.func, ast.Name)
            or node.iter.func.id != "range"
            or node.iter.keywords
            or not 1 <= len(node.iter.args) <= 3
        ):
            raise NativeCompileError(
                self.path, node, "native for supports only NAME in range(1-3 arguments)"
            )
        arguments = node.iter.args
        if len(arguments) == 1:
            start_expression = IntConstant(0)
            stop_expression = self.integer(arguments[0])
            step = 1
        else:
            start_expression = self.integer(arguments[0])
            stop_expression = self.integer(arguments[1])
            step_value = self.constant(arguments[2]) if len(arguments) == 3 else 1
            if not isinstance(step_value, int) or isinstance(step_value, bool) or step_value == 0:
                raise NativeCompileError(
                    self.path, node.iter, "native range step must be a nonzero integer constant"
                )
            step = step_value
        name = node.target.id
        self.values.pop(name, None)
        self.runtime_names.add(name)
        slot = self.slot(name)
        start_label = self.new_label("for_start")
        continue_label = self.new_label("for_continue")
        end_label = self.new_label("for_end")
        stop_slot = self.slot(f"<range-stop-{start_label}>")
        self.operations.append(Store(slot, start_expression))
        self.operations.append(Store(stop_slot, stop_expression))
        self.operations.append(Label(start_label))
        comparison = "lt" if step > 0 else "gt"
        self.operations.append(
            JumpIfFalse(
                IntCompare(comparison, IntLoad(slot), IntLoad(stop_slot)),
                end_label,
            )
        )
        self.break_targets.append(end_label)
        self.continue_targets.append(continue_label)
        for statement in node.body:
            self.statement(statement)
        self.continue_targets.pop()
        self.break_targets.pop()
        self.operations.append(Label(continue_label))
        self.operations.append(
            Store(slot, IntBinary("add", IntLoad(slot), IntConstant(step)))
        )
        self.operations.append(Jump(start_label))
        self.operations.append(Label(end_label))

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
        if not call.args:
            self.operations.append(Exit(0))
            return
        try:
            status = self.constant(call.args[0])
        except NativeCompileError:
            self.operations.append(ExitValue(self.integer(call.args[0])))
            return
        if not isinstance(status, int):
            raise NativeCompileError(self.path, call, "exit status must be an integer")
        self.operations.append(Exit(status & 0xFF))

    def integer(self, node: ast.expr) -> IntExpression:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            if not -(1 << 63) <= node.value < (1 << 63):
                raise NativeCompileError(
                    self.path, node, "native integer literal is outside signed 64-bit range"
                )
            return IntConstant(node.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return IntConstant(int(node.value))
        if isinstance(node, ast.Name) and node.id in self.slots:
            return IntLoad(self.slots[node.id])
        if (
            isinstance(node, ast.Name)
            and node.id in self.values
            and isinstance(self.values[node.id], (int, bool))
        ):
            return IntConstant(int(self.values[node.id]))
        if isinstance(node, ast.UnaryOp):
            operators = {
                ast.USub: "neg",
                ast.UAdd: "pos",
                ast.Not: "not",
                ast.Invert: "invert",
            }
            operator = operators.get(type(node.op))
            if operator is not None:
                return IntUnary(operator, self.integer(node.operand))
        if isinstance(node, ast.BinOp):
            operators = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.LShift: "lshift",
                ast.RShift: "rshift",
                ast.BitAnd: "and",
                ast.BitOr: "or",
                ast.BitXor: "xor",
            }
            operator = operators.get(type(node.op))
            if operator is not None:
                right = self.integer(node.right)
                if operator in {"lshift", "rshift"}:
                    try:
                        shift = self.constant(node.right)
                    except NativeCompileError as error:
                        raise NativeCompileError(
                            self.path,
                            node.right,
                            "native shift count must be an integer constant from 0 to 63",
                        ) from error
                    if (
                        not isinstance(shift, int)
                        or isinstance(shift, bool)
                        or not 0 <= shift <= 63
                    ):
                        raise NativeCompileError(
                            self.path,
                            node.right,
                            "native shift count must be an integer constant from 0 to 63",
                        )
                    right = IntConstant(shift)
                return IntBinary(operator, self.integer(node.left), right)
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            operators = {
                ast.Eq: "eq",
                ast.NotEq: "ne",
                ast.Lt: "lt",
                ast.LtE: "le",
                ast.Gt: "gt",
                ast.GtE: "ge",
            }
            operator = operators.get(type(node.ops[0]))
            if operator is not None:
                return IntCompare(
                    operator, self.integer(node.left), self.integer(node.comparators[0])
                )
        raise NativeCompileError(
            self.path,
            node,
            "expression is not in the signed 64-bit native integer subset",
        )

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
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = self.constant(node.values[0])
                for value in node.values[1:]:
                    if not result:
                        return result
                    result = self.constant(value)
                return result
            if isinstance(node.op, ast.Or):
                result = self.constant(node.values[0])
                for value in node.values[1:]:
                    if result:
                        return result
                    result = self.constant(value)
                return result
        if isinstance(node, ast.Compare):
            left = self.constant(node.left)
            for operator, comparator in zip(node.ops, node.comparators):
                right = self.constant(comparator)
                try:
                    if isinstance(operator, ast.Eq):
                        result = left == right
                    elif isinstance(operator, ast.NotEq):
                        result = left != right
                    elif isinstance(operator, ast.Lt):
                        result = left < right
                    elif isinstance(operator, ast.LtE):
                        result = left <= right
                    elif isinstance(operator, ast.Gt):
                        result = left > right
                    elif isinstance(operator, ast.GtE):
                        result = left >= right
                    elif isinstance(operator, (ast.Is, ast.IsNot)):
                        left_is_singleton = left is None or type(left) is bool
                        right_is_singleton = right is None or type(right) is bool
                        if not (left_is_singleton or right_is_singleton):
                            raise NativeCompileError(
                                self.path,
                                node,
                                "identity comparison is limited to None, True, or False",
                            )
                        result = left is right
                        if isinstance(operator, ast.IsNot):
                            result = not result
                    else:
                        raise NativeCompileError(
                            self.path, node, "comparison is not in the native subset yet"
                        )
                except (TypeError, ValueError) as error:
                    raise NativeCompileError(
                        self.path, node, f"constant comparison failed: {error}"
                    ) from error
                if not result:
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            branch = node.body if bool(self.constant(node.test)) else node.orelse
            return self.constant(branch)
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


def lower(
    path: Path,
    source: str,
    source_roots: tuple[Path, ...] = (),
) -> Module:
    return Frontend(path, source_roots).compile(source)
