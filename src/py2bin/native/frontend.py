from __future__ import annotations

import ast
import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .ir import (
    CStringConstant,
    Exit,
    ExitValue,
    ExternCall,
    FloatBinary,
    FloatCompare,
    FloatConstant,
    FloatExpression,
    FloatLoad,
    FloatStore,
    FloatToInt,
    FloatUnary,
    HeapAlloc,
    HeapInit,
    HeapLoad,
    HeapStore,
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
    Store,
    Write,
    WriteRuntime,
)
from .kernels import (
    KernelValue,
    StaticI64Tensor,
    binary as kernel_binary,
    dot as kernel_dot,
    reduce_tensor,
    relu as kernel_relu,
)

_MAX_INLINE_AST_NODES = 16_384
# Fixed runtime arena reserved once at process start. This is a bump allocator
# with no reclamation (documented, not hidden): every list/string allocation
# advances a pointer into this region and nothing is ever freed.
_HEAP_ARENA_BYTES = 16 * 1024 * 1024
_KERNEL_MODULES = frozenset({"numpy", "torch", "torch.nn.functional"})
_KERNEL_EXPORTS = {
    "numpy": frozenset(
        {
            "add",
            "arange",
            "array",
            "asarray",
            "dot",
            "maximum",
            "minimum",
            "multiply",
            "ones",
            "prod",
            "subtract",
            "sum",
            "zeros",
        }
    ),
    "torch": frozenset(
        {
            "add",
            "arange",
            "dot",
            "maximum",
            "minimum",
            "mul",
            "multiply",
            "ones",
            "prod",
            "relu",
            "sub",
            "subtract",
            "sum",
            "tensor",
            "zeros",
        }
    ),
    "torch.nn.functional": frozenset({"relu"}),
}

# --- adapter-ABI extern symbols ----------------------------------------------
#
# The ONLY honest "library" path: declare and call a genuine external native
# symbol resolved by the platform dynamic linker. ``from py2bin.cabi import
# name`` binds a vetted libSystem C symbol; the call lowers to an ``ExternCall``
# that the darwin-arm64 backend resolves through real dyld binding. This never
# translates C/C++/CUDA source. The Python shim ``py2bin/cabi.py`` mirrors these
# with the same observable behavior so the same source runs under CPython.
#
# Each entry maps the importable name -> (C symbol, tuple of argument kinds).
# Argument kinds:
#   "int"  -- a signed 64-bit integer expression, passed in an integer register.
#   "ptr"  -- an opaque pointer-sized handle (for example a ``PyObject *``).
#             Lowered exactly like "int"; the distinct name is what lets the
#             canonical-C frontend reject passing a plain integer where the
#             callee will dereference a pointer.
#   "cstr" -- a compile-time string constant, materialized as a NUL-terminated
#             blob whose pointer is passed.
#   "cfmt" -- like "cstr", but the fixed format argument of a *variadic* callee
#             that py2bin only ever calls with ZERO variadic arguments. Apple's
#             arm64 ABI passes variadic arguments on the stack rather than in
#             x0-x7, and this backend does not implement that, so a "cfmt"
#             literal containing any conversion specifier is rejected.
# ``_CABI_RESULTS`` records what each callee returns: "int" (a signed 64-bit
# value), "ptr" (an opaque handle) or "void" (nothing -- using the result of
# such a call is rejected, because the register would hold garbage natively
# while the CPython shim would hand back a defined value).
# Only symbols whose ABI is exactly one of these shapes are listed, so the
# compiler can never emit a call with a mismatched signature.
_CABI_MODULE = "py2bin.cabi"
_CABI_SYMBOLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "getpid": ("getpid", ()),
    "getppid": ("getppid", ()),
    "getuid": ("getuid", ()),
    "getgid": ("getgid", ()),
    "abs": ("abs", ("int",)),
    "labs": ("labs", ("int",)),
    "strlen": ("strlen", ("cstr",)),
    # CPython runtime entry points. These link the already-compiled interpreter
    # through dyld exactly like any other external symbol; no CPython source is
    # translated. They are what lets generated C drive an embedded interpreter,
    # which is the Nuitka-shaped tier: the application's own Python becomes
    # machine code while object semantics stay in libpython.
    "Py_Initialize": ("Py_Initialize", ()),
    "Py_Finalize": ("Py_Finalize", ()),
    "Py_IsInitialized": ("Py_IsInitialized", ()),
    "PyRun_SimpleString": ("PyRun_SimpleString", ("cstr",)),
    # --- C-API surface a code generator needs -------------------------------
    # Every one of these is a real exported function in the CPython dylib (not
    # a macro or a static inline), takes a fixed number of word-sized
    # arguments, and returns a word-sized value or nothing. Variadic entry
    # points such as PyObject_CallFunctionObjArgs are deliberately absent: the
    # arm64 variadic ABI is not implemented, so calling them would miscompile.
    "PyLong_FromLongLong": ("PyLong_FromLongLong", ("int",)),
    "PyLong_AsLongLong": ("PyLong_AsLongLong", ("ptr",)),
    "PyUnicode_FromString": ("PyUnicode_FromString", ("cstr",)),
    "PyNumber_Add": ("PyNumber_Add", ("ptr", "ptr")),
    "PyNumber_Subtract": ("PyNumber_Subtract", ("ptr", "ptr")),
    "PyNumber_Multiply": ("PyNumber_Multiply", ("ptr", "ptr")),
    "PyNumber_TrueDivide": ("PyNumber_TrueDivide", ("ptr", "ptr")),
    "PyObject_RichCompare": ("PyObject_RichCompare", ("ptr", "ptr", "int")),
    "PyObject_IsTrue": ("PyObject_IsTrue", ("ptr",)),
    "PyObject_Str": ("PyObject_Str", ("ptr",)),
    "PyObject_Repr": ("PyObject_Repr", ("ptr",)),
    "PyObject_Size": ("PyObject_Size", ("ptr",)),
    "PyObject_GetAttrString": ("PyObject_GetAttrString", ("ptr", "cstr")),
    "PyObject_CallNoArgs": ("PyObject_CallNoArgs", ("ptr",)),
    "PyObject_CallOneArg": ("PyObject_CallOneArg", ("ptr", "ptr")),
    "PyImport_ImportModule": ("PyImport_ImportModule", ("cstr",)),
    "PyList_New": ("PyList_New", ("int",)),
    "PyList_Append": ("PyList_Append", ("ptr", "ptr")),
    "PySys_GetObject": ("PySys_GetObject", ("cstr",)),
    "PySys_WriteStdout": ("PySys_WriteStdout", ("cfmt",)),
    "PyFile_WriteObject": ("PyFile_WriteObject", ("ptr", "ptr", "int")),
    "PyFile_WriteString": ("PyFile_WriteString", ("cstr", "ptr")),
    "Py_IncRef": ("Py_IncRef", ("ptr",)),
    "Py_DecRef": ("Py_DecRef", ("ptr",)),
    "PyErr_Occurred": ("PyErr_Occurred", ()),
    "PyErr_Print": ("PyErr_Print", ()),
    "PyErr_Clear": ("PyErr_Clear", ()),
}

_CABI_RESULTS: dict[str, str] = {
    "getpid": "int",
    "getppid": "int",
    "getuid": "int",
    "getgid": "int",
    "abs": "int",
    "labs": "int",
    "strlen": "int",
    "Py_Initialize": "void",
    "Py_Finalize": "void",
    "Py_IsInitialized": "int",
    "PyRun_SimpleString": "int",
    "PyLong_FromLongLong": "ptr",
    "PyLong_AsLongLong": "int",
    "PyUnicode_FromString": "ptr",
    "PyNumber_Add": "ptr",
    "PyNumber_Subtract": "ptr",
    "PyNumber_Multiply": "ptr",
    "PyNumber_TrueDivide": "ptr",
    "PyObject_RichCompare": "ptr",
    "PyObject_IsTrue": "int",
    "PyObject_Str": "ptr",
    "PyObject_Repr": "ptr",
    "PyObject_Size": "int",
    "PyObject_GetAttrString": "ptr",
    "PyObject_CallNoArgs": "ptr",
    "PyObject_CallOneArg": "ptr",
    "PyImport_ImportModule": "ptr",
    "PyList_New": "ptr",
    "PyList_Append": "int",
    "PySys_GetObject": "ptr",
    "PySys_WriteStdout": "void",
    "PyFile_WriteObject": "int",
    "PyFile_WriteString": "int",
    "Py_IncRef": "void",
    "Py_DecRef": "void",
    "PyErr_Occurred": "ptr",
    "PyErr_Print": "void",
    "PyErr_Clear": "void",
}

assert set(_CABI_RESULTS) == set(_CABI_SYMBOLS), "cabi result kinds are out of sync"

# Width and signedness of each callee's C result. AAPCS64 leaves bits 32-63 of
# the return register unspecified for a 32-bit result, so the encoder must
# extend it. Anything absent here returns a full 64-bit word (long long,
# Py_ssize_t, size_t, or a pointer) and needs no extension.
_CABI_RESULT_WIDTH: dict[str, str] = {
    # POSIX: pid_t/uid_t/gid_t and int abs(int) are 32 bits.
    "getpid": "i32",
    "getppid": "i32",
    "getuid": "u32",
    "getgid": "u32",
    "abs": "i32",
    # CPython entry points declared to return C int. Each uses -1 for failure,
    # which is exactly the case a missing sign extension destroys.
    "Py_IsInitialized": "i32",
    "PyRun_SimpleString": "i32",
    "PyObject_IsTrue": "i32",
    "PyList_Append": "i32",
    "PyFile_WriteObject": "i32",
    "PyFile_WriteString": "i32",
}


# The arm64 encoder passes every extern argument in x0-x7 (AAPCS64) and has no
# stack-argument path, so a longer signature must never reach it.
_CABI_MAX_ARGUMENTS = 8
assert all(
    len(signature) <= _CABI_MAX_ARGUMENTS for _symbol, signature in _CABI_SYMBOLS.values()
), "an adapter-ABI signature exceeds the register argument budget"


def _ir_contains_extern_call(value: object) -> bool:
    """True when a lowered IR expression performs an external native call."""

    if isinstance(value, ExternCall):
        return True
    if isinstance(value, (tuple, list)):
        return any(_ir_contains_extern_call(item) for item in value)
    slots = getattr(type(value), "__slots__", None)
    if slots:
        return any(_ir_contains_extern_call(getattr(value, name)) for name in slots)
    return False


def _repeats_extern_argument(
    function: "NativeFunction", arguments: tuple[object, ...]
) -> bool:
    """True when expression inlining would run one argument's extern call twice.

    Expression inlining binds a parameter to an already-lowered IR expression
    and splices that same expression in at every use. For a pure integer that
    only costs instructions; for an external call it would call the callee once
    per use. When that would happen, the caller falls back to the imperative
    inliner, which stores each argument in a stack slot exactly once.
    """

    if function.expression is None:
        return False
    loads = Counter(
        node.id
        for node in ast.walk(function.expression)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )
    return any(
        loads.get(parameter, 0) > 1 and _ir_contains_extern_call(argument)
        for parameter, argument in zip(function.parameters, arguments)
    )


class NativeCompileError(ValueError):
    def __init__(self, path: Path, node: ast.AST, message: str):
        line = getattr(node, "lineno", 1)
        column = getattr(node, "col_offset", 0) + 1
        super().__init__(f"{path}:{line}:{column}: {message}")


@dataclass(slots=True)
class NativeFunction:
    """One Python function eligible for expression or imperative IR inlining."""

    path: Path
    parameters: tuple[str, ...]
    positional_only: int
    defaults: tuple[int | None, ...]
    body: tuple[ast.stmt, ...]
    expression: ast.expr | None
    returns_value: bool
    values: dict[str, object]
    functions: dict[str, "NativeFunction"]
    kernel_modules: dict[str, str]
    kernel_functions: dict[str, str]
    extern_functions: dict[str, str]


@dataclass(slots=True)
class NativeClass:
    """One user-defined class with a statically known heap layout.

    Instances are plain heap blocks: field *i* lives at ``pointer + i * 8``.
    There is no object header, type pointer, or vtable, because the class of
    every instance is known at build time, so method calls resolve directly to
    one function body and are inlined like any other native call.
    """

    name: str
    path: Path
    fields: tuple[str, ...]
    initializer: NativeFunction | None
    methods: dict[str, NativeFunction]

    @property
    def size(self) -> int:
        return max(len(self.fields) * 8, 8)

    def offset(self, field: str) -> int:
        return self.fields.index(field) * 8


class _SubstituteLocals(ast.NodeTransformer):
    def __init__(self, replacements: dict[str, ast.expr]):
        self.replacements = replacements

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if isinstance(node.ctx, ast.Load) and node.id in self.replacements:
            return ast.copy_location(
                copy.deepcopy(self.replacements[node.id]),
                node,
            )
        return node


class _RenameFunctionLocals(ast.NodeTransformer):
    """Give one inlined call private variable names and stack slots."""

    def __init__(self, replacements: dict[str, str]):
        self.replacements = replacements

    def visit_Name(self, node: ast.Name) -> ast.Name:
        replacement = self.replacements.get(node.id)
        if replacement is None:
            return node
        return ast.copy_location(
            ast.Name(id=replacement, ctx=copy.deepcopy(node.ctx)),
            node,
        )


class Frontend:
    """Lower the first useful static-Python subset into portable native IR.

    The native subset accepts static string output, a signed 64-bit integer
    runtime, and an IEEE-754 binary64 (``float``) runtime: variables,
    arithmetic, comparisons, ``int``/``float`` conversion, if/while, for-range,
    and process exit. Doubles are lowered to real SSE2/NEON instructions by the
    backends. Unsupported dynamic Python semantics fail loudly instead of
    silently producing a wrong executable.
    """

    def __init__(
        self,
        path: Path,
        source_roots: tuple[Path, ...] = (),
        import_stack: tuple[Path, ...] = (),
        experimental_kernels: bool = False,
    ):
        self.path = path
        self.source_roots = tuple(root.expanduser().resolve() for root in source_roots)
        self.import_stack = (*import_stack, path.expanduser().resolve())
        self.experimental_kernels = experimental_kernels
        self.values: dict[str, object] = {}
        self.functions: dict[str, NativeFunction] = {}
        self.kernel_modules: dict[str, str] = {}
        self.kernel_functions: dict[str, str] = {}
        # Local name -> C symbol for adapter-ABI extern calls imported from
        # ``py2bin.cabi``. Resolved through real dyld binding by the backend.
        self.extern_functions: dict[str, str] = {}
        self.operations = []
        self.slots: dict[str, int] = {}
        self.runtime_names: set[str] = set()
        # Names whose slot may never have been written on some path, so
        # reading one would return whatever the stack happened to hold.
        # CPython raises UnboundLocalError there, so a read is rejected.
        self.possibly_unbound: set[str] = set()
        # Names that have definitely been assigned at this point.
        self.bound_names: set[str] = set()
        # Runtime name -> "int" | "float". A stack slot is 8 bytes and holds
        # either a signed 64-bit integer or an IEEE-754 double; this records
        # which, so the correct load/store and register file is used.
        self.value_types: dict[str, str] = {}
        # Known compile-time lengths of runtime lists built from literals, keyed
        # by variable name. Used only to reject out-of-range constant indices.
        self.list_lengths: dict[str, int] = {}
        # Lazily reserved stack slot holding the arena bump pointer, plus a flag
        # recording that the program needs the arena (so HeapInit is emitted).
        self._heap_bump_slot: int | None = None
        self._temp_number = 0
        # User-defined classes, and the class name bound to each runtime object
        # variable (including the private ``self`` of an inlined method).
        self.classes: dict[str, NativeClass] = {}
        self.object_classes: dict[str, str] = {}
        # Greater than zero while lowering a sub-expression that Python would
        # evaluate only conditionally (a conditional-expression arm or a
        # short-circuited Boolean operand). Both arms are lowered eagerly, so
        # anything that can trap at runtime must be rejected there rather than
        # trapping in a branch Python would never have taken.
        self.eager_depth = 0
        self.label_number = 0
        self.break_targets: list[str] = []
        self.continue_targets: list[str] = []
        self.return_targets: list[tuple[int | None, str]] = []
        self.active_functions: list[tuple[int, str]] = []

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
        if self._heap_bump_slot is not None:
            # Initialize the arena unconditionally at process start so that no
            # runtime path can reach an allocation before the bump pointer is
            # valid, regardless of where the first allocation appears in source.
            self.operations.insert(
                0, HeapInit(self._heap_bump_slot, _HEAP_ARENA_BYTES)
            )
        return Module(self.operations, len(self.slots))

    def ensure_heap(self) -> int:
        """Reserve the arena bump-pointer slot and return it."""

        if self._heap_bump_slot is None:
            self._heap_bump_slot = self.slot("<heap-bump>")
        return self._heap_bump_slot

    def new_temp(self) -> int:
        """Allocate a fresh, uniquely named scratch stack slot."""

        self._temp_number += 1
        return self.slot(f"<tmp-{self._temp_number}>")

    @staticmethod
    def _aligned_size(payload: IntExpression) -> IntExpression:
        """Header (8 bytes) + ``payload`` bytes rounded up to a multiple of 8."""

        rounded = IntBinary(
            "and", IntBinary("add", payload, IntConstant(7)), IntConstant(-8)
        )
        return IntBinary("add", IntConstant(8), rounded)

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
        class LoopMutationVisitor(ast.NodeVisitor):
            def __init__(self):
                self.names: set[str] = set()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

            def visit_AugAssign(self, node: ast.AugAssign) -> None:
                if isinstance(node.target, ast.Name):
                    self.names.add(node.target.id)
                self.generic_visit(node)

            def visit_For(self, node: ast.For) -> None:
                self.names.update(cls.assigned_names(node.body))
                if isinstance(node.target, ast.Name):
                    self.names.add(node.target.id)
                self.generic_visit(node)

            def visit_While(self, node: ast.While) -> None:
                self.names.update(cls.assigned_names(node.body))
                self.generic_visit(node)

        visitor = LoopMutationVisitor()
        visitor.visit(tree)
        return visitor.names

    @staticmethod
    def block_always_returns(statements: tuple[ast.stmt, ...] | list[ast.stmt]) -> bool:
        """Conservatively prove that every fall-through path returns a value."""

        for statement in statements:
            if isinstance(statement, ast.Return):
                return statement.value is not None
            if (
                isinstance(statement, ast.If)
                and statement.orelse
                and Frontend.block_always_returns(statement.body)
                and Frontend.block_always_returns(statement.orelse)
            ):
                return True
        return False

    @staticmethod
    def function_returns(body: tuple[ast.stmt, ...]) -> tuple[ast.Return, ...]:
        """Return only return statements owned by this function."""

        found: list[ast.Return] = []

        class ReturnVisitor(ast.NodeVisitor):
            def visit_Return(self, node: ast.Return) -> None:
                found.append(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

        visitor = ReturnVisitor()
        for statement in body:
            visitor.visit(statement)
        return tuple(found)

    @classmethod
    def safe_annotation(cls, node: ast.expr) -> bool:
        """Accept inert type-expression shapes without implementing their types."""

        if isinstance(node, ast.Name):
            return True
        if isinstance(node, ast.Constant):
            return node.value is None or isinstance(node.value, str)
        if isinstance(node, ast.Attribute):
            return cls.safe_annotation(node.value)
        if isinstance(node, ast.Subscript):
            return cls.safe_annotation(node.value) and cls.safe_annotation(node.slice)
        if isinstance(node, ast.Tuple):
            return all(cls.safe_annotation(item) for item in node.elts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return cls.safe_annotation(node.left) and cls.safe_annotation(node.right)
        return False

    @classmethod
    def function_local_names(
        cls,
        body: tuple[ast.stmt, ...],
        parameters: tuple[str, ...],
    ) -> set[str]:
        names: set[str] = set()
        names.update(parameters)
        names.update(cls.assigned_names(list(body)))
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
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
        ):
            self.subscript_assignment(node.targets[0], node.value)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
        ):
            self.attribute_assignment(node.targets[0], node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Attribute)
            and node.value
        ):
            self.attribute_assignment(node.target, node.value)
        elif isinstance(node, ast.ClassDef):
            self.class_definition(node)
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
        elif isinstance(node, ast.Return):
            if not self.return_targets:
                raise NativeCompileError(self.path, node, "return is outside a native function")
            result_slot, return_label = self.return_targets[-1]
            if node.value is None:
                if result_slot is not None:
                    raise NativeCompileError(
                        self.path,
                        node,
                        "native value function cannot use a bare return",
                    )
                self.operations.append(Jump(return_label))
                return
            if result_slot is None:
                raise NativeCompileError(
                    self.path,
                    node,
                    "native procedure cannot return a value",
                )
            self.operations.append(Store(result_slot, self.integer(node.value)))
            self.operations.append(Jump(return_label))
        elif isinstance(node, ast.Pass):
            return
        elif isinstance(node, ast.FunctionDef):
            self.function_definition(node)
        elif isinstance(node, ast.ImportFrom):
            self.import_from(node)
        elif isinstance(node, ast.Import):
            self.import_statement(node)
        elif isinstance(node, ast.Raise) and node.exc:
            self.system_exit(node.exc, node)
        else:
            raise NativeCompileError(
                self.path,
                node,
                f"{type(node).__name__} is not in the native subset yet; use bundle mode for full CPython semantics",
            )

    def function_definition(self, node: ast.FunctionDef) -> None:
        arguments = node.args
        if (
            node.decorator_list
            or arguments.vararg is not None
            or arguments.kwarg is not None
            or arguments.kwonlyargs
            or arguments.kw_defaults
        ):
            raise NativeCompileError(
                self.path,
                node,
                "native functions currently require undecorated positional "
                "parameters without *args, **kwargs, or keyword-only arguments",
            )
        annotations = [
            argument.annotation
            for argument in (*arguments.posonlyargs, *arguments.args)
            if argument.annotation is not None
        ]
        if node.returns is not None:
            annotations.append(node.returns)
        if any(not self.safe_annotation(annotation) for annotation in annotations):
            raise NativeCompileError(
                self.path,
                node,
                "native function annotation has runtime behavior and cannot be erased",
            )
        parameters = tuple(
            argument.arg for argument in (*arguments.posonlyargs, *arguments.args)
        )
        default_values: list[int | None] = [
            None for _ in range(len(parameters) - len(arguments.defaults))
        ]
        for default in arguments.defaults:
            try:
                value = self.constant(default)
            except NativeCompileError as error:
                raise NativeCompileError(
                    self.path,
                    default,
                    "native function defaults must be compile-time int/bool values",
                ) from error
            if not isinstance(value, (int, bool)):
                raise NativeCompileError(
                    self.path,
                    default,
                    "native function defaults must be compile-time int/bool values",
                )
            default_values.append(int(value))
        body = tuple(copy.deepcopy(node.body))
        returns = self.function_returns(body)
        value_returns = tuple(item for item in returns if item.value is not None)
        bare_returns = tuple(item for item in returns if item.value is None)
        if value_returns and bare_returns:
            raise NativeCompileError(
                self.path,
                bare_returns[0],
                "native function cannot mix value returns and bare returns",
            )
        returns_value = bool(value_returns)
        if returns_value and not self.block_always_returns(body):
            raise NativeCompileError(
                self.path,
                node,
                "native integer function must return a value on every fall-through path",
            )
        try:
            replacements = {
                name: ast.Name(id=name, ctx=ast.Load())
                for name in parameters
            }
            expression = self.function_block_expression(
                list(body),
                replacements,
                node,
            )
        except NativeCompileError:
            # Functions with loops, early returns, or mutable branch state use
            # the imperative IR inliner when called. Unsupported statements
            # still fail there with their exact source location.
            expression = None
        self.functions[node.name] = NativeFunction(
            self.path,
            parameters,
            len(arguments.posonlyargs),
            tuple(default_values),
            body,
            expression,
            returns_value,
            self.values,
            self.functions,
            self.kernel_modules,
            self.kernel_functions,
            self.extern_functions,
        )

    # --- user-defined classes ------------------------------------------------

    def class_definition(self, node: ast.ClassDef) -> None:
        """Record a class whose instances have a fixed native heap layout."""

        if node.decorator_list or node.keywords:
            raise NativeCompileError(
                self.path,
                node,
                "native classes must be undecorated and take no class keywords",
            )
        for base in node.bases:
            if not (isinstance(base, ast.Name) and base.id == "object"):
                raise NativeCompileError(
                    self.path,
                    node,
                    "native classes do not support inheritance; only a bare "
                    "class or an explicit 'object' base is supported",
                )
        previous_functions = self.functions
        self.functions = dict(previous_functions)
        methods: dict[str, NativeFunction] = {}
        try:
            for statement in node.body:
                if isinstance(statement, ast.Pass):
                    continue
                if isinstance(statement, ast.Expr) and isinstance(
                    statement.value, ast.Constant
                ):
                    continue  # Class docstring.
                if not isinstance(statement, ast.FunctionDef):
                    raise NativeCompileError(
                        self.path,
                        statement,
                        "a native class body supports only method definitions, "
                        "a docstring, and 'pass'; class attributes are not "
                        "supported",
                    )
                if statement.decorator_list:
                    raise NativeCompileError(
                        self.path,
                        statement,
                        "native methods cannot be decorated, so properties, "
                        "classmethod, and staticmethod are not supported",
                    )
                self.function_definition(statement)
                method = self.functions[statement.name]
                if not method.parameters or method.parameters[0] != "self":
                    raise NativeCompileError(
                        self.path,
                        statement,
                        f"native method {statement.name}() must take 'self' as "
                        "its first parameter",
                    )
                methods[statement.name] = method
        finally:
            self.functions = previous_functions
        initializer = methods.pop("__init__", None)
        for name in methods:
            if name.startswith("__") and name.endswith("__"):
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native classes do not implement the special method {name}()",
                )
        fields = self.discover_fields(node, initializer)
        self.classes[node.name] = NativeClass(
            node.name, self.path, fields, initializer, methods
        )

    def discover_fields(
        self, node: ast.ClassDef, initializer: NativeFunction | None
    ) -> tuple[str, ...]:
        """Derive the instance layout from ``self.NAME = ...`` in ``__init__``.

        Every attribute must be assigned in ``__init__`` so each instance has a
        complete, statically known layout. An attribute first assigned anywhere
        else would have no reserved storage, so it is rejected rather than
        silently writing outside the object.
        """

        if initializer is None:
            return ()

        def assigned_attributes(statements) -> list[str]:
            found: list[str] = []
            for statement in statements:
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr not in found
                    ):
                        found.append(target.attr)
            return found

        # Only assignments directly in the __init__ body always run, so only
        # those reserve a layout slot.
        fields = assigned_attributes(initializer.body)
        every = assigned_attributes(
            ast.walk(ast.Module(body=list(initializer.body), type_ignores=[]))
        )
        conditional = [name for name in every if name not in fields]
        if conditional:
            raise NativeCompileError(
                self.path,
                node,
                f"attribute {conditional[0]!r} is assigned only conditionally in "
                "__init__; every native attribute must be assigned "
                "unconditionally there, because Python raises AttributeError "
                "for one that was never set and the native layout cannot "
                "represent an absent attribute",
            )
        if len(fields) > 1024:
            raise NativeCompileError(
                self.path, node, "native classes support at most 1024 attributes"
            )
        return tuple(fields)

    def resolve_object_class(self, node: ast.expr) -> NativeClass | None:
        """Return the class of an object-valued expression, if it is known."""

        if isinstance(node, ast.Name):
            class_name = self.object_classes.get(node.id)
            if class_name is not None:
                return self.classes.get(class_name)
            return None
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.classes
        ):
            return self.classes[node.func.id]
        return None

    def attribute_field(self, node: ast.Attribute) -> tuple[NativeClass, int]:
        """Resolve ``obj.attr`` to its class and byte offset."""

        native_class = self.resolve_object_class(node.value)
        if native_class is None:
            raise NativeCompileError(
                self.path,
                node,
                "native attribute access requires a variable holding an "
                "instance of a class defined in this program",
            )
        if node.attr not in native_class.fields:
            raise NativeCompileError(
                self.path,
                node,
                f"{native_class.name!r} has no native attribute {node.attr!r}; "
                "every attribute must be assigned in __init__",
            )
        return native_class, native_class.offset(node.attr)

    def attribute_address(self, node: ast.Attribute) -> IntExpression:
        native_class, offset = self.attribute_field(node)
        pointer = self.integer(node.value) if not isinstance(
            node.value, ast.Name
        ) else IntLoad(self.slots[node.value.id])
        return IntBinary("add", pointer, IntConstant(offset))

    def object_assignment(self, name: str, node: ast.expr) -> None:
        """Construct an instance and bind it to ``name``."""

        native_class = self.resolve_object_class(node)
        if native_class is None or not isinstance(node, ast.Call):
            raise NativeCompileError(
                self.path,
                node,
                "a native object variable must be assigned a direct "
                "ClassName(...) construction",
            )
        previous = self.object_classes.get(name)
        if previous is not None and previous != native_class.name:
            raise NativeCompileError(
                self.path,
                node,
                f"native object variable {name!r} cannot change class from "
                f"{previous} to {native_class.name}",
            )
        bump = self.ensure_heap()
        self.runtime_names.add(name)
        pointer_slot = self.slot(name)
        self.operations.append(
            HeapAlloc(pointer_slot, IntConstant(native_class.size), bump)
        )
        # Every field starts at 0 so a layout slot is never read uninitialized.
        for index in range(len(native_class.fields)):
            self.operations.append(
                HeapStore(
                    IntBinary("add", IntLoad(pointer_slot), IntConstant(index * 8)),
                    IntConstant(0),
                    8,
                )
            )
        self.object_classes[name] = native_class.name
        if native_class.initializer is not None:
            self.inline_method(
                native_class,
                "__init__",
                native_class.initializer,
                IntLoad(pointer_slot),
                node,
                (),
            )

    def attribute_assignment(self, target: ast.Attribute, value: ast.expr) -> None:
        if self.expression_type(value) != "int":
            raise NativeCompileError(
                self.path,
                value,
                "native object attributes are signed 64-bit integers",
            )
        address = self.attribute_address(target)
        self.operations.append(HeapStore(address, self.integer(value), 8))

    def inline_method(
        self,
        native_class: NativeClass,
        method_name: str,
        method: NativeFunction,
        instance: IntExpression,
        node: ast.Call,
        call_stack: tuple[int, ...],
    ) -> IntExpression | None:
        """Inline a method body with ``self`` bound to ``instance``."""

        arguments = self.bind_native_arguments(
            f"{native_class.name}.{method_name}",
            method,
            node,
            {},
            call_stack,
            skip_parameters=1,
        )
        return self.inline_imperative_function(
            f"{native_class.name}.{method_name}",
            method,
            (instance, *arguments),
            node,
            call_stack,
            parameter_classes={"self": native_class.name},
        )

    def contains_extern_call(self, expression: ast.AST) -> bool:
        """True when ``expression`` performs an adapter-ABI external call.

        Such a call is not a pure value, so textual substitution must never
        duplicate it: two copies would run the callee twice.
        """

        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.extern_functions
            for node in ast.walk(expression)
        )

    def substitute_locals(
        self,
        expression: ast.expr,
        replacements: dict[str, ast.expr],
    ) -> ast.expr:
        # Substitution is textual, so a replacement used more than once is
        # evaluated more than once. That is invisible for a pure integer
        # expression and a miscompile for an external call. Refuse here; the
        # caller turns the refusal into "inline this function imperatively",
        # where every argument and local is materialized in a stack slot and
        # therefore evaluated exactly once.
        loads = Counter(
            node.id
            for node in ast.walk(expression)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )
        for name, replacement in replacements.items():
            if loads.get(name, 0) > 1 and self.contains_extern_call(replacement):
                raise NativeCompileError(
                    self.path,
                    expression,
                    f"{name!r} holds the result of an external native call and is "
                    "used more than once, which expression inlining would turn "
                    "into repeated calls",
                )
        substituted = _SubstituteLocals(replacements).visit(
            copy.deepcopy(expression)
        )
        self.validate_inline_size(substituted, expression)
        return substituted

    def validate_inline_size(
        self,
        expression: ast.expr,
        location: ast.AST,
    ) -> None:
        for count, _node in enumerate(ast.walk(expression), 1):
            if count > _MAX_INLINE_AST_NODES:
                raise NativeCompileError(
                    self.path,
                    location,
                    f"native inlined expression exceeds {_MAX_INLINE_AST_NODES} "
                    "AST nodes",
                )

    def function_block_expression(
        self,
        statements: list[ast.stmt],
        replacements: dict[str, ast.expr],
        location: ast.AST,
    ) -> ast.expr:
        statements = list(statements)
        if (
            statements
            and isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)
        ):
            statements.pop(0)
        index = 0
        while index < len(statements):
            statement = statements[index]
            if isinstance(statement, ast.Pass):
                index += 1
                continue
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                replacements[statement.targets[0].id] = self.substitute_locals(
                    statement.value,
                    replacements,
                )
                index += 1
                continue
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
            ):
                replacements[statement.target.id] = self.substitute_locals(
                    statement.value,
                    replacements,
                )
                index += 1
                continue
            if (
                isinstance(statement, ast.AugAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id in replacements
            ):
                if self.contains_extern_call(replacements[statement.target.id]):
                    # The old value is copied into the new expression, so an
                    # external call inside it would run twice. Fall back to the
                    # imperative inliner, which keeps the value in a slot.
                    raise NativeCompileError(
                        self.path,
                        statement,
                        f"{statement.target.id!r} holds the result of an external "
                        "native call and cannot be updated in place inside an "
                        "inlined expression function",
                    )
                replacements[statement.target.id] = ast.copy_location(
                    ast.BinOp(
                        left=copy.deepcopy(replacements[statement.target.id]),
                        op=copy.deepcopy(statement.op),
                        right=self.substitute_locals(
                            statement.value,
                            replacements,
                        ),
                    ),
                    statement,
                )
                index += 1
                continue
            if isinstance(statement, ast.Return) and statement.value is not None:
                if index != len(statements) - 1:
                    raise NativeCompileError(
                        self.path,
                        statement,
                        "native function has unreachable statements after return",
                    )
                return self.substitute_locals(
                    statement.value,
                    replacements,
                )
            if isinstance(statement, ast.If):
                condition = self.substitute_locals(
                    statement.test,
                    replacements,
                )
                if statement.orelse:
                    if index != len(statements) - 1:
                        raise NativeCompileError(
                            self.path,
                            statement,
                            "native conditional return must terminate the function",
                        )
                    body = self.function_block_expression(
                        statement.body,
                        replacements.copy(),
                        statement,
                    )
                    alternative = self.function_block_expression(
                        statement.orelse,
                        replacements.copy(),
                        statement,
                    )
                else:
                    body = self.function_block_expression(
                        statement.body,
                        replacements.copy(),
                        statement,
                    )
                    alternative = self.function_block_expression(
                        statements[index + 1 :],
                        replacements.copy(),
                        statement,
                    )
                expression = ast.copy_location(
                    ast.IfExp(
                        test=condition,
                        body=body,
                        orelse=alternative,
                    ),
                    statement,
                )
                self.validate_inline_size(expression, statement)
                return expression
            raise NativeCompileError(
                self.path,
                statement,
                f"{type(statement).__name__} is not supported inside a native "
                "function; use assignments and terminal conditional returns",
            )
        raise NativeCompileError(
            self.path,
            location,
            "native function must return a supported integer expression",
        )

    def source_candidate(self, module: str, level: int = 0) -> Path | None:
        parts = module.split(".")
        if level:
            base = self.path.parent
            for _ in range(level - 1):
                base = base.parent
            relative_base = base.joinpath(*parts)
            candidates = (
                relative_base.with_suffix(".py"),
                relative_base / "__init__.py",
            )
            for candidate in candidates:
                resolved = candidate.resolve()
                if (
                    candidate.is_file()
                    and any(
                        resolved.is_relative_to(root)
                        for root in self.source_roots
                    )
                ):
                    return candidate
            return None
        for root in self.source_roots:
            module_path = root.joinpath(*parts).with_suffix(".py")
            package_path = root.joinpath(*parts, "__init__.py")
            if module_path.is_file():
                return module_path
            if package_path.is_file():
                return package_path
        return None

    def import_statement(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "sys":
                if alias.asname not in {None, "sys"}:
                    raise NativeCompileError(
                        self.path,
                        node,
                        "the native sys.exit adapter must be imported as sys",
                    )
                continue
            if alias.name in _KERNEL_MODULES:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"the real {alias.name} package is not in the native subset; "
                    "py2bin will not silently reimplement a third-party numerical "
                    "library, because a from-scratch integer reimplementation does "
                    "not match the real package's object semantics at runtime "
                    "(e.g. a numpy/torch reduction is an np.int64 / 0-d tensor, "
                    "not a plain int, so the program's observable result differs "
                    "from CPython)",
                )
            raise NativeCompileError(
                self.path,
                node,
                f"Import is not in the native subset yet; unsupported module "
                f"{alias.name!r}",
            )

    def import_kernel_exports(self, node: ast.ImportFrom) -> bool:
        if node.level or node.module not in _KERNEL_MODULES:
            return False
        assert node.module is not None
        raise NativeCompileError(
            self.path,
            node,
            f"the real {node.module} package is not in the native subset; "
            "py2bin will not silently reimplement a third-party numerical "
            "library, because a from-scratch integer reimplementation does not "
            "match the real package's object semantics at runtime (e.g. a "
            "numpy/torch reduction is an np.int64 / 0-d tensor, not a plain int, "
            "so the program's observable result differs from CPython)",
        )

    def import_extern_symbols(self, node: ast.ImportFrom) -> bool:
        """Register adapter-ABI extern symbols from ``py2bin.cabi``.

        Returns ``True`` when handled. Each imported name must be a vetted C
        symbol; unknown names and ``import *`` are rejected so the compiler can
        never emit a call whose ABI it has not verified.
        """

        if node.level or node.module != _CABI_MODULE:
            return False
        for alias in node.names:
            if alias.name == "*":
                raise NativeCompileError(
                    self.path,
                    node,
                    f"{_CABI_MODULE} does not support 'import *'; import each "
                    "extern symbol by name",
                )
            if alias.name not in _CABI_SYMBOLS:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"{alias.name!r} is not an available adapter-ABI symbol; "
                    f"choose one of {', '.join(sorted(_CABI_SYMBOLS))}",
                )
            symbol, _signature = _CABI_SYMBOLS[alias.name]
            self.extern_functions[alias.asname or alias.name] = alias.name
        return True

    def import_from(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module == "__future__":
            return
        if self.import_kernel_exports(node):
            return
        if self.import_extern_symbols(node):
            return
        if not node.module:
            raise NativeCompileError(
                self.path,
                node,
                "native source imports require 'from MODULE import NAME'",
            )
        if any(alias.name == "*" for alias in node.names):
            raise NativeCompileError(
                self.path, node, "native locked-source imports do not support import *"
            )
        candidate = self.source_candidate(node.module, node.level)
        if candidate is None:
            raise NativeCompileError(
                self.path,
                node,
                f"native source roots do not provide module {node.module!r}",
            )
        candidate = candidate.resolve()
        if candidate in self.import_stack:
            chain = " -> ".join(path.name for path in (*self.import_stack, candidate))
            raise NativeCompileError(
                self.path,
                node,
                f"circular native source import is not supported: {chain}",
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
        provider = Frontend(
            candidate,
            self.source_roots,
            self.import_stack,
            self.experimental_kernels,
        )
        function_errors: dict[str, NativeCompileError] = {}
        unsafe_top_level: list[ast.stmt] = []
        for statement in tree.body:
            try:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                ):
                    if provider.is_kernel_expression(statement.value):
                        value = provider.kernel_value(statement.value)
                        if not isinstance(value, StaticI64Tensor):
                            raise NativeCompileError(
                                candidate,
                                statement,
                                "module-level numerical kernel must produce a "
                                "static tensor, not runtime scalar work",
                            )
                        provider.values[statement.targets[0].id] = value
                    else:
                        provider.values[statement.targets[0].id] = provider.constant(
                            statement.value
                        )
                elif (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.value is not None
                ):
                    if provider.is_kernel_expression(statement.value):
                        value = provider.kernel_value(statement.value)
                        if not isinstance(value, StaticI64Tensor):
                            raise NativeCompileError(
                                candidate,
                                statement,
                                "module-level numerical kernel must produce a "
                                "static tensor, not runtime scalar work",
                            )
                        provider.values[statement.target.id] = value
                    else:
                        provider.values[statement.target.id] = provider.constant(
                            statement.value
                        )
                elif isinstance(statement, ast.FunctionDef):
                    try:
                        provider.function_definition(statement)
                    except NativeCompileError as error:
                        function_errors[statement.name] = error
                elif (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    continue
                elif (
                    isinstance(statement, ast.ImportFrom)
                    and statement.level == 0
                    and statement.module == "__future__"
                ):
                    continue
                elif isinstance(statement, ast.ImportFrom):
                    try:
                        provider.import_from(statement)
                    except NativeCompileError:
                        unsafe_top_level.append(statement)
                elif isinstance(statement, ast.Import):
                    try:
                        provider.import_statement(statement)
                    except NativeCompileError:
                        unsafe_top_level.append(statement)
                elif isinstance(statement, ast.Pass):
                    continue
                else:
                    unsafe_top_level.append(statement)
            except NativeCompileError:
                # Only requested, statically evaluable exports matter. Other
                # downloaded code is never executed during this inspection.
                unsafe_top_level.append(statement)
                continue
        for alias in node.names:
            imported_name = alias.asname or alias.name
            if alias.name in provider.values:
                self.values[imported_name] = provider.values[alias.name]
                continue
            if alias.name in function_errors:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native source function {node.module}.{alias.name} is unsupported: "
                    f"{function_errors[alias.name]}",
                )
            if alias.name in provider.functions:
                if unsafe_top_level:
                    first = unsafe_top_level[0]
                    raise NativeCompileError(
                        self.path,
                        node,
                        f"native source module {node.module!r} has executable top-level "
                        f"{type(first).__name__}; importing its function would change Python semantics",
                    )
                self.functions[imported_name] = provider.functions[alias.name]
                continue
            if alias.name not in provider.values:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native source export {node.module}.{alias.name} is neither a "
                    "compile-time constant nor a supported pure integer function",
                )

    @staticmethod
    def dotted_name(node: ast.expr) -> str | None:
        parts: list[str] = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        parts.append(current.id)
        return ".".join(reversed(parts))

    def kernel_call_name(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return self.kernel_functions.get(node.id)
        dotted = self.dotted_name(node)
        if dotted is None:
            return None
        root, separator, suffix = dotted.partition(".")
        module = self.kernel_modules.get(root)
        if module is None or not separator:
            return None
        candidate = f"{module}.{suffix}"
        export = candidate.removeprefix(f"{module}.")
        if export in _KERNEL_EXPORTS[module]:
            return candidate
        return None

    def is_kernel_expression(
        self,
        node: ast.expr,
        bindings: dict[str, KernelValue] | None = None,
    ) -> bool:
        if not self.experimental_kernels:
            return False
        bindings = bindings or {}
        if isinstance(node, (ast.List, ast.Tuple)):
            return False
        if isinstance(node, ast.Name):
            return isinstance(
                bindings.get(node.id, self.values.get(node.id)),
                StaticI64Tensor,
            )
        if isinstance(node, ast.Call):
            if self.kernel_call_name(node.func) is not None:
                return True
            return (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"prod", "relu", "sum"}
                and self.is_kernel_expression(node.func.value, bindings)
            )
        if isinstance(node, ast.BinOp):
            return self.is_kernel_expression(
                node.left, bindings
            ) or self.is_kernel_expression(node.right, bindings)
        if isinstance(node, ast.UnaryOp):
            return self.is_kernel_expression(node.operand, bindings)
        if isinstance(node, ast.Subscript):
            return self.is_kernel_expression(node.value, bindings)
        return False

    def kernel_operand(
        self,
        node: ast.expr,
        bindings: dict[str, KernelValue] | None,
        call_stack: tuple[int, ...],
    ) -> KernelValue:
        if isinstance(node, (ast.List, ast.Tuple)) or self.is_kernel_expression(
            node, bindings
        ):
            return self.kernel_value(node, bindings, call_stack)
        return self.integer(node, bindings, call_stack)

    def require_tensor(
        self,
        value: KernelValue,
        node: ast.AST,
        operation: str,
    ) -> StaticI64Tensor:
        if not isinstance(value, StaticI64Tensor):
            raise NativeCompileError(
                self.path,
                node,
                f"{operation} requires a rank-1 static i64 tensor",
            )
        return value

    def kernel_value(
        self,
        node: ast.expr,
        bindings: dict[str, KernelValue] | None = None,
        call_stack: tuple[int, ...] = (),
    ) -> KernelValue:
        bindings = bindings or {}
        if isinstance(node, ast.Name):
            value = bindings.get(node.id, self.values.get(node.id))
            if isinstance(value, StaticI64Tensor):
                return value
        if isinstance(node, (ast.List, ast.Tuple)):
            if not node.elts:
                raise NativeCompileError(
                    self.path,
                    node,
                    "static native tensors cannot be empty",
                )
            if any(isinstance(element, (ast.List, ast.Tuple)) for element in node.elts):
                raise NativeCompileError(
                    self.path,
                    node,
                    "experimental numerical kernels currently support rank-1 tensors only",
                )
            try:
                return StaticI64Tensor(
                    tuple(
                        self.integer(element, bindings, call_stack)
                        for element in node.elts
                    ),
                    "literal",
                )
            except ValueError as error:
                raise NativeCompileError(self.path, node, str(error)) from error
        if isinstance(node, ast.BinOp):
            operators = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.BitAnd: "and",
                ast.BitOr: "or",
                ast.BitXor: "xor",
            }
            operator = operators.get(type(node.op))
            if operator is None:
                raise NativeCompileError(
                    self.path,
                    node,
                    "static tensor operators are limited to +, -, *, &, |, and ^",
                )
            try:
                return kernel_binary(
                    operator,
                    self.kernel_operand(node.left, bindings, call_stack),
                    self.kernel_operand(node.right, bindings, call_stack),
                )
            except ValueError as error:
                raise NativeCompileError(self.path, node, str(error)) from error
        if isinstance(node, ast.UnaryOp) and self.is_kernel_expression(
            node.operand, bindings
        ):
            tensor = self.require_tensor(
                self.kernel_value(node.operand, bindings, call_stack),
                node,
                "static tensor unary operation",
            )
            operator = {
                ast.USub: "neg",
                ast.UAdd: "pos",
                ast.Invert: "invert",
            }.get(type(node.op))
            if operator is None:
                raise NativeCompileError(
                    self.path,
                    node,
                    "unsupported static tensor unary operation",
                )
            return StaticI64Tensor(
                tuple(IntUnary(operator, element) for element in tensor.elements),
                tensor.origin,
            )
        if isinstance(node, ast.Subscript):
            tensor = self.require_tensor(
                self.kernel_value(node.value, bindings, call_stack),
                node,
                "static tensor indexing",
            )
            try:
                index = self.constant(node.slice)
            except NativeCompileError as error:
                raise NativeCompileError(
                    self.path,
                    node.slice,
                    "static tensor index must be a compile-time integer",
                ) from error
            if not isinstance(index, int) or isinstance(index, bool):
                raise NativeCompileError(
                    self.path,
                    node.slice,
                    "static tensor index must be a compile-time integer",
                )
            try:
                return tensor.elements[index]
            except IndexError as error:
                raise NativeCompileError(
                    self.path,
                    node.slice,
                    f"static tensor index {index} is outside shape {tensor.shape}",
                ) from error
        if isinstance(node, ast.Call):
            if node.keywords:
                raise NativeCompileError(
                    self.path,
                    node,
                    "experimental numerical kernels do not support keyword arguments",
                )
            canonical = self.kernel_call_name(node.func)
            if canonical is None and isinstance(node.func, ast.Attribute):
                method = node.func.attr
                if method in {"prod", "relu", "sum"} and self.is_kernel_expression(
                    node.func.value, bindings
                ):
                    if node.args:
                        raise NativeCompileError(
                            self.path,
                            node,
                            f"static tensor {method}() takes no arguments",
                        )
                    value = self.kernel_value(
                        node.func.value,
                        bindings,
                        call_stack,
                    )
                    if method == "relu":
                        return kernel_relu(value)
                    tensor = self.require_tensor(value, node, method)
                    return reduce_tensor(method, tensor)
            if canonical is None:
                raise NativeCompileError(
                    self.path,
                    node,
                    "expression is not an experimental numerical kernel",
                )
            module, operation = canonical.rsplit(".", 1)
            if operation in {"array", "asarray", "tensor"}:
                if len(node.args) != 1:
                    raise NativeCompileError(
                        self.path,
                        node,
                        f"{canonical} requires exactly one rank-1 input",
                    )
                return self.require_tensor(
                    self.kernel_value(node.args[0], bindings, call_stack),
                    node,
                    canonical,
                )
            if operation in {"arange", "ones", "zeros"}:
                try:
                    if operation == "arange":
                        if not 1 <= len(node.args) <= 3:
                            raise NativeCompileError(
                                self.path,
                                node,
                                f"{canonical} requires 1-3 integer arguments",
                            )
                        raw = [self.constant(argument) for argument in node.args]
                        if any(
                            not isinstance(value, int) or isinstance(value, bool)
                            for value in raw
                        ):
                            raise NativeCompileError(
                                self.path,
                                node,
                                f"{canonical} arguments must be compile-time integers",
                            )
                        start, stop, step = (
                            (0, raw[0], 1)
                            if len(raw) == 1
                            else (raw[0], raw[1], 1)
                            if len(raw) == 2
                            else (raw[0], raw[1], raw[2])
                        )
                        if step == 0:
                            raise NativeCompileError(
                                self.path,
                                node,
                                f"{canonical} step cannot be zero",
                            )
                        elements = tuple(
                            IntConstant(value) for value in range(start, stop, step)
                        )
                    else:
                        if len(node.args) != 1:
                            raise NativeCompileError(
                                self.path,
                                node,
                                f"{canonical} requires one static length",
                            )
                        length = self.constant(node.args[0])
                        if (
                            not isinstance(length, int)
                            or isinstance(length, bool)
                            or length < 1
                        ):
                            raise NativeCompileError(
                                self.path,
                                node,
                                f"{canonical} length must be a positive compile-time integer",
                            )
                        fill = 1 if operation == "ones" else 0
                        elements = tuple(IntConstant(fill) for _ in range(length))
                    return StaticI64Tensor(elements, canonical)
                except ValueError as error:
                    raise NativeCompileError(self.path, node, str(error)) from error
            if operation in {
                "add",
                "maximum",
                "minimum",
                "mul",
                "multiply",
                "sub",
                "subtract",
            }:
                if len(node.args) != 2:
                    raise NativeCompileError(
                        self.path,
                        node,
                        f"{canonical} requires exactly two inputs",
                    )
                operator = {
                    "add": "add",
                    "sub": "sub",
                    "subtract": "sub",
                    "mul": "mul",
                    "multiply": "mul",
                    "maximum": "maximum",
                    "minimum": "minimum",
                }[operation]
                try:
                    return kernel_binary(
                        operator,
                        self.kernel_operand(node.args[0], bindings, call_stack),
                        self.kernel_operand(node.args[1], bindings, call_stack),
                    )
                except ValueError as error:
                    raise NativeCompileError(self.path, node, str(error)) from error
            if operation == "relu":
                if len(node.args) != 1:
                    raise NativeCompileError(
                        self.path,
                        node,
                        f"{canonical} requires exactly one input",
                    )
                return kernel_relu(
                    self.kernel_operand(node.args[0], bindings, call_stack)
                )
            if operation in {"prod", "sum"}:
                if len(node.args) != 1:
                    raise NativeCompileError(
                        self.path,
                        node,
                        f"{canonical} requires exactly one tensor",
                    )
                tensor = self.require_tensor(
                    self.kernel_operand(node.args[0], bindings, call_stack),
                    node,
                    canonical,
                )
                return reduce_tensor(operation, tensor)
            if operation == "dot":
                if len(node.args) != 2:
                    raise NativeCompileError(
                        self.path,
                        node,
                        f"{canonical} requires exactly two tensors",
                    )
                left = self.require_tensor(
                    self.kernel_operand(node.args[0], bindings, call_stack),
                    node,
                    canonical,
                )
                right = self.require_tensor(
                    self.kernel_operand(node.args[1], bindings, call_stack),
                    node,
                    canonical,
                )
                try:
                    return kernel_dot(left, right)
                except ValueError as error:
                    raise NativeCompileError(self.path, node, str(error)) from error
        raise NativeCompileError(
            self.path,
            node,
            "expression is not in the static rank-1 i64 tensor subset",
        )

    def assignment(self, name: str, expression: ast.expr) -> None:
        self.possibly_unbound.discard(name)
        self.bound_names.add(name)
        if self.is_kernel_expression(expression):
            value = self.kernel_value(expression)
            if isinstance(value, StaticI64Tensor):
                if name in self.runtime_names:
                    raise NativeCompileError(
                        self.path,
                        expression,
                        "static tensor variables cannot change shape or be loop-mutated",
                    )
                self.values[name] = value
                return
            self.values.pop(name, None)
            self.runtime_names.add(name)
            self.operations.append(Store(self.slot(name), value))
            return
        if name not in self.runtime_names:
            try:
                self.values[name] = self.constant(expression)
                return
            except NativeCompileError:
                self.runtime_names.add(name)
        self.values.pop(name, None)
        kind = self.expression_type(expression)
        previous = self.value_types.get(name)
        if previous is not None and previous != kind:
            if {previous, kind} <= {"int", "float"}:
                message = (
                    f"native variable {name!r} cannot change between int and float"
                )
            else:
                message = (
                    f"native variable {name!r} cannot change type between "
                    f"{previous} and {kind}"
                )
            raise NativeCompileError(self.path, expression, message)
        self.value_types[name] = kind
        if kind == "object":
            self.object_assignment(name, expression)
        elif kind == "list-i64":
            self.list_assignment(name, expression)
        elif kind == "str":
            self.string_assignment(name, expression)
        elif kind == "float":
            self.operations.append(
                FloatStore(self.slot(name), self.float_expression(expression))
            )
        else:
            self.list_lengths.pop(name, None)
            self.operations.append(Store(self.slot(name), self.integer(expression)))

    # --- runtime lists ------------------------------------------------------

    def list_assignment(self, name: str, node: ast.expr) -> None:
        if not isinstance(node, ast.List):
            raise NativeCompileError(
                self.path, node, "native list variables require a list literal"
            )
        elements = node.elts
        for element in elements:
            if self.expression_type(element) != "int":
                raise NativeCompileError(
                    self.path,
                    element,
                    "native lists currently hold signed 64-bit integers only",
                )
        bump = self.ensure_heap()
        self.runtime_names.add(name)
        pointer_slot = self.slot(name)
        length = len(elements)
        # Layout: [i64 length][i64 element0][i64 element1]... (all 8-aligned).
        size = 8 + length * 8
        self.operations.append(HeapAlloc(pointer_slot, IntConstant(size), bump))
        self.operations.append(
            HeapStore(IntLoad(pointer_slot), IntConstant(length), 8)
        )
        for index, element in enumerate(elements):
            address = IntBinary(
                "add", IntLoad(pointer_slot), IntConstant(8 + index * 8)
            )
            self.operations.append(HeapStore(address, self.integer(element), 8))
        self.list_lengths[name] = length

    def list_element_address(self, node: ast.Subscript) -> IntExpression:
        target = node.value
        if not isinstance(target, ast.Name) or self.value_types.get(target.id) != "list-i64":
            raise NativeCompileError(
                self.path, node, "native indexing requires a runtime list variable"
            )
        pointer = IntLoad(self.slots[target.id])
        index_node = node.slice
        try:
            folded = self.constant(index_node)
        except NativeCompileError:
            folded = None
        if isinstance(folded, int) and not isinstance(folded, bool):
            length = self.list_lengths.get(target.id)
            resolved = folded
            if resolved < 0 and length is not None:
                # Python counts a negative index from the end.
                resolved += length
            if resolved < 0 or (length is not None and resolved >= length):
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native list index {folded} is out of range for {target.id!r}"
                    + (f" (length {length})" if length is not None else ""),
                )
            return IntBinary("add", pointer, IntConstant(8 + resolved * 8))
        # A runtime index cannot be proved in range at build time, so normalize
        # negatives the way Python does and emit a real bounds check. Without
        # this the generated code would read or write outside the list and
        # silently return a wrong answer where CPython raises IndexError.
        if self.eager_depth:
            raise NativeCompileError(
                self.path,
                node,
                "a list index that is not a compile-time constant cannot appear "
                "in a conditional expression or a short-circuited Boolean "
                "operand, because its bounds check would run even when Python "
                "would not evaluate that branch; use an if statement instead",
            )
        index = self.integer(index_node)
        bad_label = self.new_label("index_error")
        ok_label = self.new_label("index_ok")
        index_slot = self.slot(f"<index-{bad_label}>")
        length_slot = self.slot(f"<length-{bad_label}>")
        self.operations.append(Store(length_slot, HeapLoad(pointer, 8)))
        # Branchless Python negative indexing: index += length when index < 0.
        negative_mask = IntUnary("neg", IntCompare("lt", index, IntConstant(0)))
        self.operations.append(
            Store(
                index_slot,
                IntBinary(
                    "add",
                    index,
                    IntBinary("and", IntLoad(length_slot), negative_mask),
                ),
            )
        )
        in_range = IntBinary(
            "and",
            IntCompare("ge", IntLoad(index_slot), IntConstant(0)),
            IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)),
        )
        self.operations.append(JumpIfFalse(in_range, bad_label))
        self.operations.append(Jump(ok_label))
        self.operations.append(Label(bad_label))
        # CPython prints the traceback to stderr and exits 1; match both.
        self.operations.append(
            Write(b"IndexError: list index out of range\n", 2)
        )
        self.operations.append(Exit(1))
        self.operations.append(Label(ok_label))
        offset = IntBinary(
            "add", IntConstant(8), IntBinary("mul", IntLoad(index_slot), IntConstant(8))
        )
        return IntBinary("add", pointer, offset)

    def subscript_assignment(self, target: ast.Subscript, value: ast.expr) -> None:
        if self.expression_type(value) != "int":
            raise NativeCompileError(
                self.path, value, "native list elements are signed 64-bit integers"
            )
        address = self.list_element_address(target)
        self.operations.append(HeapStore(address, self.integer(value), 8))

    # --- runtime strings ----------------------------------------------------

    def string_assignment(self, name: str, node: ast.expr) -> None:
        pointer = self.string_pointer(node)
        self.runtime_names.add(name)
        self.operations.append(Store(self.slot(name), pointer))

    def string_pointer(self, node: ast.expr) -> IntExpression:
        """Emit any needed heap work and return an i64 pointer to a string block.

        A string block is ``[i64 length][raw utf-8 bytes]``. The returned
        expression is a stable load of the pointer, safe to reference twice.
        """

        if self.expression_type(node) != "str":
            raise NativeCompileError(
                self.path, node, "expression is not in the native string subset"
            )
        if isinstance(node, ast.Name) and self.value_types.get(node.id) == "str":
            return IntLoad(self.slots[node.id])
        try:
            folded = self.constant(node)
        except NativeCompileError:
            folded = None
        if isinstance(folded, str):
            if not folded.isascii():
                raise NativeCompileError(
                    self.path,
                    node,
                    "native runtime strings are limited to ASCII; a non-ASCII "
                    "literal would make len() disagree with CPython's code-point "
                    "count (its bytes still print correctly as a compile-time "
                    "constant via print())",
                )
            return self.materialize_string_constant(folded.encode("utf-8"))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.string_pointer(node.left)
            right = self.string_pointer(node.right)
            return self.emit_concat(left, right)
        raise NativeCompileError(
            self.path, node, "expression is not in the native string subset"
        )

    def materialize_string_constant(self, data: bytes) -> IntExpression:
        bump = self.ensure_heap()
        pointer_slot = self.new_temp()
        size = 8 + ((len(data) + 7) & ~7)
        self.operations.append(HeapAlloc(pointer_slot, IntConstant(size), bump))
        self.operations.append(
            HeapStore(IntLoad(pointer_slot), IntConstant(len(data)), 8)
        )
        padded = data + b"\x00" * (-len(data) % 8)
        for offset in range(0, len(padded), 8):
            word = int.from_bytes(padded[offset : offset + 8], "little")
            address = IntBinary(
                "add", IntLoad(pointer_slot), IntConstant(8 + offset)
            )
            self.operations.append(HeapStore(address, IntConstant(word), 8))
        return IntLoad(pointer_slot)

    def emit_concat(
        self, left_pointer: IntExpression, right_pointer: IntExpression
    ) -> IntExpression:
        bump = self.ensure_heap()
        left_slot = self.new_temp()
        right_slot = self.new_temp()
        left_length_slot = self.new_temp()
        right_length_slot = self.new_temp()
        total_slot = self.new_temp()
        result_slot = self.new_temp()
        self.operations.append(Store(left_slot, left_pointer))
        self.operations.append(Store(right_slot, right_pointer))
        self.operations.append(
            Store(left_length_slot, HeapLoad(IntLoad(left_slot), 8))
        )
        self.operations.append(
            Store(right_length_slot, HeapLoad(IntLoad(right_slot), 8))
        )
        self.operations.append(
            Store(
                total_slot,
                IntBinary(
                    "add", IntLoad(left_length_slot), IntLoad(right_length_slot)
                ),
            )
        )
        self.operations.append(
            HeapAlloc(result_slot, self._aligned_size(IntLoad(total_slot)), bump)
        )
        self.operations.append(
            HeapStore(IntLoad(result_slot), IntLoad(total_slot), 8)
        )
        result_data = IntBinary("add", IntLoad(result_slot), IntConstant(8))
        self.emit_byte_copy(
            result_data,
            IntBinary("add", IntLoad(left_slot), IntConstant(8)),
            IntLoad(left_length_slot),
        )
        self.emit_byte_copy(
            IntBinary("add", result_data, IntLoad(left_length_slot)),
            IntBinary("add", IntLoad(right_slot), IntConstant(8)),
            IntLoad(right_length_slot),
        )
        return IntLoad(result_slot)

    def emit_byte_copy(
        self,
        destination: IntExpression,
        source: IntExpression,
        count: IntExpression,
    ) -> None:
        index_slot = self.new_temp()
        destination_slot = self.new_temp()
        source_slot = self.new_temp()
        count_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        self.operations.append(Store(destination_slot, destination))
        self.operations.append(Store(source_slot, source))
        self.operations.append(Store(count_slot, count))
        start_label = self.new_label("copy_start")
        end_label = self.new_label("copy_end")
        self.operations.append(Label(start_label))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(count_slot)),
                end_label,
            )
        )
        byte = HeapLoad(
            IntBinary("add", IntLoad(source_slot), IntLoad(index_slot)), 1
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", IntLoad(destination_slot), IntLoad(index_slot)),
                byte,
                1,
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start_label))
        self.operations.append(Label(end_label))

    def materialize_runtime_names(self, names: set[str]) -> None:
        for name in sorted(names):
            if name in self.values:
                value = self.values.pop(name)
                if isinstance(value, (int, bool)):
                    self.operations.append(Store(self.slot(name), IntConstant(int(value))))
                    self.value_types[name] = "int"
                elif isinstance(value, float):
                    self.operations.append(
                        FloatStore(self.slot(name), FloatConstant(value))
                    )
                    self.value_types[name] = "float"
                elif isinstance(value, str):
                    if not value.isascii():
                        raise NativeCompileError(
                            self.path,
                            ast.Constant(value=value),
                            "native runtime strings are limited to ASCII; a "
                            "non-ASCII value would make len() disagree with "
                            "CPython's code-point count",
                        )
                    pointer = self.materialize_string_constant(value.encode("utf-8"))
                    self.operations.append(Store(self.slot(name), pointer))
                    self.value_types[name] = "str"
                else:
                    raise NativeCompileError(
                        self.path,
                        ast.Constant(value=value),
                        f"runtime variable {name!r} must be a signed 64-bit integer, "
                        "float, or string",
                    )
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
        # Capture binding state before the loop touches it: after a loop whose
        # range can be empty, Python leaves the name exactly as it was, which
        # means unbound if it was never assigned.
        was_bound = name in self.bound_names
        self.values.pop(name, None)
        self.runtime_names.add(name)
        slot = self.slot(name)
        start_label = self.new_label("for_start")
        continue_label = self.new_label("for_continue")
        end_label = self.new_label("for_end")
        stop_slot = self.slot(f"<range-stop-{start_label}>")
        # Iterate on a private counter and copy it into the user's variable at
        # the top of each iteration. Advancing the user's variable directly
        # would leave it holding ``stop`` after the loop, and would clobber it
        # even when the range is empty; Python does neither.
        counter_slot = self.slot(f"<range-counter-{start_label}>")
        self.operations.append(Store(counter_slot, start_expression))
        self.operations.append(Store(stop_slot, stop_expression))
        self.operations.append(Label(start_label))
        comparison = "lt" if step > 0 else "gt"
        self.operations.append(
            JumpIfFalse(
                IntCompare(comparison, IntLoad(counter_slot), IntLoad(stop_slot)),
                end_label,
            )
        )
        self.operations.append(Store(slot, IntLoad(counter_slot)))
        self.break_targets.append(end_label)
        self.continue_targets.append(continue_label)
        for statement in node.body:
            self.statement(statement)
        self.continue_targets.pop()
        self.break_targets.pop()
        self.operations.append(Label(continue_label))
        self.operations.append(
            Store(
                counter_slot,
                IntBinary("add", IntLoad(counter_slot), IntConstant(step)),
            )
        )
        self.operations.append(Jump(start_label))
        self.operations.append(Label(end_label))
        # If both bounds are compile-time constants the loop's emptiness is
        # known now, so a provably non-empty range definitely binds the name.
        definitely_runs = (
            isinstance(start_expression, IntConstant)
            and isinstance(stop_expression, IntConstant)
            and (
                start_expression.value < stop_expression.value
                if step > 0
                else start_expression.value > stop_expression.value
            )
        )
        if was_bound or definitely_runs:
            self.bound_names.add(name)
        else:
            # The body may never have run, so the name may still be unbound.
            self.possibly_unbound.add(name)

    def expression_statement(self, node: ast.expr) -> None:
        if not isinstance(node, ast.Call):
            raise NativeCompileError(self.path, node, "only print() and SystemExit are valid expression statements")
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and self.resolve_object_class(node.func.value) is not None
        ):
            native_class = self.resolve_object_class(node.func.value)
            assert native_class is not None
            method = native_class.methods.get(node.func.attr)
            if method is None:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"{native_class.name!r} has no native method {node.func.attr!r}",
                )
            self.inline_method(
                native_class,
                node.func.attr,
                method,
                IntLoad(self.slots[node.func.value.id]),
                node,
                (),
            )
            return
        if isinstance(node.func, ast.Name) and node.func.id == "print" and "print" not in self.functions:
            if node.keywords:
                raise NativeCompileError(self.path, node, "native print() does not support keyword arguments yet")
            try:
                values = [self.constant(argument) for argument in node.args]
            except NativeCompileError:
                values = None
            if values is not None:
                text = " ".join(str(value) for value in values) + "\n"
                self.operations.append(Write(text.encode("utf-8")))
                return
            if len(node.args) == 1 and self.expression_type(node.args[0]) == "str":
                # Runtime string: write its bytes straight from the heap block,
                # then the trailing newline that print() appends.
                pointer = self.string_pointer(node.args[0])
                self.operations.append(
                    WriteRuntime(
                        IntBinary("add", pointer, IntConstant(8)),
                        HeapLoad(pointer, 8),
                    )
                )
                self.operations.append(Write(b"\n"))
                return
            raise NativeCompileError(
                self.path,
                node,
                "native print() supports compile-time values or a single runtime "
                "string argument",
            )
        elif self.is_exit_call(node):
            self.system_exit(node, node)
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id in self.extern_functions
        ):
            # A bare extern call: run it for its effect and discard the result.
            self.operations.append(
                Store(self.new_temp(), self.extern_call(node, {}, (), discarded=True))
            )
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id in self.functions
        ):
            function = self.functions[node.func.id]
            if function.returns_value:
                self.integer(node)
                return
            identity = id(function)
            if identity in {item[0] for item in self.active_functions}:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"recursive native procedure call to {node.func.id}() is not supported",
                )
            arguments = self.bind_native_arguments(
                node.func.id,
                function,
                node,
                {},
                (),
            )
            if any(isinstance(argument, StaticI64Tensor) for argument in arguments):
                raise NativeCompileError(
                    self.path,
                    node,
                    "native procedures do not yet support tensor parameters",
                )
            result = self.inline_imperative_function(
                node.func.id,
                function,
                tuple(
                    argument
                    for argument in arguments
                    if not isinstance(argument, StaticI64Tensor)
                ),
                node,
                (),
            )
            assert result is None
        else:
            raise NativeCompileError(
                self.path,
                node,
                "only native functions/procedures, print(), and SystemExit are "
                "callable as native expression statements",
            )

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

    def select_integer(
        self,
        condition: IntExpression,
        body: IntExpression,
        alternative: IntExpression,
    ) -> IntExpression:
        # The mask is referenced twice below, and the backends re-emit an
        # expression tree at every occurrence. Materialising it in a slot
        # evaluates the condition exactly once; otherwise a condition
        # containing a call would perform that call twice.
        truth = IntUnary("not", IntUnary("not", condition))
        mask_slot = self.new_temp()
        self.operations.append(Store(mask_slot, IntUnary("neg", truth)))
        mask = IntLoad(mask_slot)
        return IntBinary(
            "or",
            IntBinary("and", body, mask),
            IntBinary(
                "and",
                alternative,
                IntUnary("invert", mask),
            ),
        )

    def inline_imperative_function(
        self,
        function_name: str,
        function: NativeFunction,
        arguments: tuple[IntExpression, ...],
        location: ast.AST,
        call_stack: tuple[int, ...],
        parameter_classes: dict[str, str] | None = None,
    ) -> IntExpression | None:
        """Inline a function body as labels, jumps, stores, and integer IR.

        This is the Python-only replacement for handing generated C to an
        external compiler. Each call receives private stack slots and labels;
        the existing handwritten x86-64/ARM64 encoders then turn the expanded
        IR into machine instructions.
        """

        identity = id(function)
        active_ids = {item[0] for item in self.active_functions}
        if identity in call_stack or identity in active_ids:
            raise NativeCompileError(
                self.path,
                location,
                f"recursive native function call to {function_name}() is not supported",
            )
        if len(call_stack) + len(self.active_functions) >= 64:
            raise NativeCompileError(
                self.path,
                location,
                "native function inline depth exceeds 64 calls",
            )

        call_number = self.label_number + 1
        private_names = {
            name: f"<call-{call_number}:{name}>"
            for name in self.function_local_names(function.body, function.parameters)
        }
        body = tuple(
            _RenameFunctionLocals(private_names).visit(copy.deepcopy(statement))
            for statement in function.body
        )
        ast.fix_missing_locations(ast.Module(body=list(body), type_ignores=[]))

        result_slot = (
            self.slot(f"<call-{call_number}:result>")
            if function.returns_value
            else None
        )
        return_label = self.new_label("function_return")

        for parameter, argument in zip(function.parameters, arguments):
            private_parameter = private_names[parameter]
            self.runtime_names.add(private_parameter)
            self.operations.append(Store(self.slot(private_parameter), argument))
            # An object parameter (a method's ``self``) carries its class into
            # the inlined body so attribute access there resolves statically.
            class_name = (parameter_classes or {}).get(parameter)
            if class_name is not None:
                self.object_classes[private_parameter] = class_name

        loop_mutations = self.loop_mutated_names(
            ast.Module(body=list(body), type_ignores=[])
        )
        self.runtime_names.update(loop_mutations)

        previous_path = self.path
        previous_values = self.values
        previous_functions = self.functions
        previous_kernel_modules = self.kernel_modules
        previous_kernel_functions = self.kernel_functions
        previous_extern_functions = self.extern_functions
        self.path = function.path
        self.values = dict(function.values)
        self.functions = function.functions
        self.kernel_modules = function.kernel_modules
        self.kernel_functions = function.kernel_functions
        self.extern_functions = function.extern_functions
        self.return_targets.append((result_slot, return_label))
        self.active_functions.append((identity, function_name))
        try:
            for statement in body:
                self.statement(statement)
        finally:
            self.active_functions.pop()
            self.return_targets.pop()
            self.functions = previous_functions
            self.values = previous_values
            self.kernel_modules = previous_kernel_modules
            self.kernel_functions = previous_kernel_functions
            self.extern_functions = previous_extern_functions
            self.path = previous_path
        self.operations.append(Label(return_label))
        return IntLoad(result_slot) if result_slot is not None else None

    def bind_native_arguments(
        self,
        function_name: str,
        function: NativeFunction,
        node: ast.Call,
        bindings: dict[str, KernelValue],
        call_stack: tuple[int, ...],
        skip_parameters: int = 0,
    ) -> tuple[KernelValue, ...]:
        # ``skip_parameters`` hides leading parameters the caller supplies
        # itself, which is how a method's ``self`` is bound to the instance
        # rather than to a call argument.
        parameters = function.parameters[skip_parameters:]
        defaults = function.defaults[skip_parameters:]
        if len(node.args) > len(parameters):
            raise NativeCompileError(
                self.path,
                node,
                f"native function {function_name}() accepts at most "
                f"{len(parameters)} positional arguments",
            )
        bound: list[KernelValue | None] = [None for _ in parameters]
        for index, argument in enumerate(node.args):
            bound[index] = (
                self.kernel_operand(argument, bindings, call_stack)
                if self.experimental_kernels
                else self.integer(argument, bindings, call_stack)
            )
        for keyword in node.keywords:
            if keyword.arg is None:
                raise NativeCompileError(
                    self.path,
                    keyword,
                    "native function calls do not support ** mapping expansion",
                )
            try:
                index = parameters.index(keyword.arg)
            except ValueError as error:
                raise NativeCompileError(
                    self.path,
                    keyword,
                    f"native function {function_name}() has no parameter "
                    f"{keyword.arg!r}",
                ) from error
            if index + skip_parameters < function.positional_only:
                raise NativeCompileError(
                    self.path,
                    keyword,
                    f"native function parameter {keyword.arg!r} is positional-only",
                )
            if bound[index] is not None:
                raise NativeCompileError(
                    self.path,
                    keyword,
                    f"native function {function_name}() received {keyword.arg!r} twice",
                )
            bound[index] = (
                self.kernel_operand(keyword.value, bindings, call_stack)
                if self.experimental_kernels
                else self.integer(keyword.value, bindings, call_stack)
            )
        missing: list[str] = []
        for index, value in enumerate(bound):
            if value is not None:
                continue
            default = defaults[index]
            if default is None:
                missing.append(parameters[index])
            else:
                bound[index] = IntConstant(default)
        if missing:
            raise NativeCompileError(
                self.path,
                node,
                f"native function {function_name}() is missing required "
                f"argument(s): {', '.join(missing)}",
            )
        return tuple(value for value in bound if value is not None)

    def expression_type(
        self,
        node: ast.expr,
        bindings: dict[str, KernelValue] | None = None,
    ) -> str:
        """Statically classify a runtime value as ``"int"`` or ``"float"``.

        Returns ``"float"`` only when the value is definitely a double. An
        over-optimistic ``"int"`` is safe: the integer/float lowering methods
        still reject genuinely unsupported nodes with a source location.
        """

        bindings = bindings or {}
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return "str"
            return "float" if isinstance(node.value, float) else "int"
        if isinstance(node, ast.Name):
            if node.id in bindings:
                return "int"
            if node.id in self.object_classes:
                return "object"
            if node.id in self.value_types:
                return self.value_types[node.id]
            value = self.values.get(node.id)
            if isinstance(value, str):
                return "str"
            return "float" if isinstance(value, float) else "int"
        if isinstance(node, ast.List):
            return "list-i64"
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, (ast.USub, ast.UAdd)):
                return self.expression_type(node.operand, bindings)
            return "int"
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Div):
                return "float"
            if isinstance(node.op, ast.Add):
                left = self.expression_type(node.left, bindings)
                right = self.expression_type(node.right, bindings)
                if "str" in (left, right):
                    return "str"
                return "float" if "float" in (left, right) else "int"
            if isinstance(node.op, (ast.Sub, ast.Mult)):
                left = self.expression_type(node.left, bindings)
                right = self.expression_type(node.right, bindings)
                return "float" if "float" in (left, right) else "int"
            return "int"
        if isinstance(node, ast.IfExp):
            body = self.expression_type(node.body, bindings)
            orelse = self.expression_type(node.orelse, bindings)
            return "float" if "float" in (body, orelse) else "int"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in self.classes:
                return "object"
            if node.func.id in self.extern_functions:
                return "int"
            if node.func.id == "float" and node.func.id not in self.functions:
                return "float"
            if node.func.id in {"int", "len"} and node.func.id not in self.functions:
                return "int"
        if isinstance(node, ast.Attribute) and self.resolve_object_class(node.value):
            return "int"  # Attributes are signed 64-bit integers.
        return "int"

    def float_divisor(
        self,
        node: ast.expr,
        bindings: dict[str, KernelValue],
        call_stack: tuple[int, ...],
    ) -> FloatExpression:
        """Validate and lower a float division right-hand side.

        Runtime float division is accepted only with a nonzero numeric constant
        divisor. Arbitrary divisors can be zero at runtime, and Python raises
        ``ZeroDivisionError`` there; honoring that needs the object runtime, so
        it is rejected rather than silently producing IEEE infinity/NaN.
        """

        try:
            divisor = self.constant(node)
        except NativeCompileError:
            divisor = None
        if isinstance(divisor, bool) or not isinstance(divisor, (int, float)):
            raise NativeCompileError(
                self.path,
                node,
                "runtime float division requires a nonzero numeric constant divisor",
            )
        if float(divisor) == 0.0:
            raise NativeCompileError(self.path, node, "float division by zero")
        return FloatConstant(float(divisor))

    def float_expression(
        self,
        node: ast.expr,
        bindings: dict[str, KernelValue] | None = None,
        call_stack: tuple[int, ...] = (),
    ) -> FloatExpression:
        """Lower ``node`` to an IEEE-754 double, widening integer operands."""

        bindings = bindings or {}
        if self.expression_type(node, bindings) != "float":
            return IntToFloat(self.integer(node, bindings, call_stack))
        try:
            folded = self.constant(node)
        except NativeCompileError:
            folded = None
        if not isinstance(folded, bool) and isinstance(folded, (int, float)):
            return FloatConstant(float(folded))
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            return FloatConstant(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.slots and self.value_types.get(node.id) == "float":
                return FloatLoad(self.slots[node.id])
            value = self.values.get(node.id)
            if isinstance(value, float):
                return FloatConstant(value)
            raise NativeCompileError(
                self.path, node, f"float variable {node.id!r} is not defined here"
            )
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.UAdd):
                return self.float_expression(node.operand, bindings, call_stack)
            if isinstance(node.op, ast.USub):
                return FloatUnary(
                    "neg", self.float_expression(node.operand, bindings, call_stack)
                )
            raise NativeCompileError(
                self.path, node, "unsupported native float unary operator"
            )
        if isinstance(node, ast.BinOp):
            operators = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
            operator = operators.get(type(node.op))
            if operator is None:
                raise NativeCompileError(
                    self.path, node, "native float arithmetic supports +, -, *, and /"
                )
            left = self.float_expression(node.left, bindings, call_stack)
            if operator == "div":
                return FloatBinary("div", left, self.float_divisor(node.right, bindings, call_stack))
            return FloatBinary(
                operator, left, self.float_expression(node.right, bindings, call_stack)
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and node.func.id not in self.functions
        ):
            if len(node.args) != 1 or node.keywords:
                raise NativeCompileError(
                    self.path, node, "native float() takes exactly one argument"
                )
            return self.float_expression(node.args[0], bindings, call_stack)
        raise NativeCompileError(
            self.path, node, "expression is not in the native float subset"
        )

    def extern_call(
        self,
        node: ast.Call,
        bindings: dict[str, KernelValue],
        call_stack: tuple[int, ...],
        *,
        discarded: bool = False,
    ) -> ExternCall:
        """Lower a vetted ``py2bin.cabi`` extern call to an ``ExternCall``.

        Argument shapes are checked against the symbol's declared signature so
        the emitted call always matches the callee's real ABI. ``cstr``/``cfmt``
        operands must be compile-time string constants (materialized as a
        NUL-terminated blob); ``int``/``ptr`` operands are ordinary integer
        expressions.

        ``discarded`` marks a call used as a bare statement. A callee declared
        to return ``void`` leaves nothing defined in the result register, so its
        value may only be discarded -- using it would read garbage natively
        while the CPython shim returned a defined value.
        """

        assert isinstance(node.func, ast.Name)
        local_name = node.func.id
        key = self.extern_functions[local_name]
        symbol, signature = _CABI_SYMBOLS[key]
        if self.eager_depth:
            # Conditional expressions and short-circuited Boolean operands are
            # lowered by evaluating BOTH arms and selecting between the
            # results. An external call is not a pure value: running it in the
            # arm Python would have skipped changes reference counts, writes
            # output, or sets the error indicator. Reject instead.
            raise NativeCompileError(
                self.path,
                node,
                f"external native call {local_name}() cannot appear in a "
                "conditional expression or a short-circuited Boolean operand, "
                "because this lowering evaluates both arms and the call's side "
                "effects would happen even when the branch is not taken; use an "
                "if statement instead",
            )
        if _CABI_RESULTS[key] == "void" and not discarded:
            raise NativeCompileError(
                self.path,
                node,
                f"extern call {local_name}() returns void; its result is not a "
                "value and can only be discarded",
            )
        if len(signature) > _CABI_MAX_ARGUMENTS:
            raise NativeCompileError(
                self.path,
                node,
                f"extern call {local_name}() passes {len(signature)} arguments, "
                f"but the native backend only implements {_CABI_MAX_ARGUMENTS} "
                "register arguments and has no stack-argument path",
            )
        if node.keywords:
            raise NativeCompileError(
                self.path,
                node,
                f"extern call {local_name}() does not accept keyword arguments",
            )
        if len(node.args) != len(signature):
            raise NativeCompileError(
                self.path,
                node,
                f"extern call {local_name}() expects {len(signature)} argument(s), "
                f"got {len(node.args)}",
            )
        arguments: list[IntExpression] = []
        for argument, kind in zip(node.args, signature):
            if kind in {"cstr", "cfmt"}:
                try:
                    value = self.constant(argument)
                except NativeCompileError:
                    value = None
                if isinstance(value, str):
                    data = value.encode("utf-8")
                elif isinstance(value, (bytes, bytearray)):
                    data = bytes(value)
                else:
                    raise NativeCompileError(
                        self.path,
                        argument,
                        f"extern call {local_name}() requires a compile-time "
                        "string constant for its C-string argument",
                    )
                if b"\0" in data:
                    raise NativeCompileError(
                        self.path,
                        argument,
                        f"extern call {local_name}() was handed a C-string "
                        "constant containing an embedded NUL, which the callee "
                        "would silently truncate",
                    )
                if kind == "cfmt" and b"%" in data:
                    # A variadic callee reached with a conversion specifier
                    # would read arguments py2bin never passed. Apple's arm64
                    # ABI puts variadic arguments on the stack, and this
                    # backend has no stack-argument path, so reject instead of
                    # emitting a call that reads uninitialised memory.
                    raise NativeCompileError(
                        self.path,
                        argument,
                        f"extern call {local_name}() is variadic and py2bin only "
                        "supports calling it with zero variadic arguments, so its "
                        "format string must not contain '%'",
                    )
                arguments.append(CStringConstant(data + b"\0"))
            else:
                arguments.append(self.integer(argument, bindings, call_stack))
        return ExternCall(
            symbol,
            tuple(arguments),
            _CABI_RESULT_WIDTH.get(symbol, "i64"),
        )

    def integer(
        self,
        node: ast.expr,
        bindings: dict[str, KernelValue] | None = None,
        call_stack: tuple[int, ...] = (),
    ) -> IntExpression:
        bindings = bindings or {}
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.extern_functions
        ):
            return self.extern_call(node, bindings, call_stack)
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
        if isinstance(node, ast.Name) and node.id in bindings:
            value = bindings[node.id]
            if isinstance(value, StaticI64Tensor):
                raise NativeCompileError(
                    self.path,
                    node,
                    f"static tensor {node.id!r} requires indexing or a reduction",
                )
            return value
        if isinstance(node, ast.Name) and node.id in self.possibly_unbound:
            raise NativeCompileError(
                self.path,
                node,
                f"{node.id!r} may be unbound here because its loop can run "
                "zero times; CPython raises UnboundLocalError, and the "
                "native slot would hold an unrelated value",
            )
        if isinstance(node, ast.Name) and node.id in self.slots:
            kind = self.value_types.get(node.id)
            if kind == "float":
                raise NativeCompileError(
                    self.path,
                    node,
                    f"float variable {node.id!r} needs an explicit int({node.id}) "
                    "in an integer context",
                )
            if kind == "list-i64":
                raise NativeCompileError(
                    self.path,
                    node,
                    f"list variable {node.id!r} needs indexing or len() in an "
                    "integer context",
                )
            if kind == "str":
                raise NativeCompileError(
                    self.path,
                    node,
                    f"string variable {node.id!r} needs len() in an integer context",
                )
            return IntLoad(self.slots[node.id])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "int"
            and node.func.id not in self.functions
        ):
            if len(node.args) != 1 or node.keywords:
                raise NativeCompileError(
                    self.path, node, "native int() takes exactly one argument"
                )
            argument = node.args[0]
            try:
                folded = self.constant(argument)
            except NativeCompileError:
                folded = None
            if isinstance(folded, (int, float)) and not isinstance(folded, bool):
                return IntConstant(int(folded))
            if isinstance(folded, bool):
                return IntConstant(int(folded))
            if self.expression_type(argument, bindings) == "float":
                return FloatToInt(self.float_expression(argument, bindings, call_stack))
            return self.integer(argument, bindings, call_stack)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and node.func.id not in self.functions
        ):
            if len(node.args) != 1 or node.keywords:
                raise NativeCompileError(
                    self.path, node, "native len() takes exactly one argument"
                )
            argument = node.args[0]
            if self.expression_type(argument, bindings) == "str":
                # The length header is the first i64 of the string block.
                return HeapLoad(self.string_pointer(argument), 8)
            if (
                isinstance(argument, ast.Name)
                and self.value_types.get(argument.id) == "list-i64"
            ):
                return HeapLoad(IntLoad(self.slots[argument.id]), 8)
            raise NativeCompileError(
                self.path,
                node,
                "native len() supports runtime strings and integer lists",
            )
        if isinstance(node, ast.Subscript):
            return HeapLoad(self.list_element_address(node), 8)
        if isinstance(node, ast.Attribute) and self.resolve_object_class(node.value):
            return HeapLoad(self.attribute_address(node), 8)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and self.resolve_object_class(node.func.value) is not None
        ):
            native_class = self.resolve_object_class(node.func.value)
            assert native_class is not None
            method = native_class.methods.get(node.func.attr)
            if method is None:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"{native_class.name!r} has no native method "
                    f"{node.func.attr!r}",
                )
            if not method.returns_value:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native method {node.func.attr}() does not produce a value",
                )
            if not isinstance(node.func.value, ast.Name):
                raise NativeCompileError(
                    self.path,
                    node,
                    "native method calls require an object variable receiver",
                )
            result = self.inline_method(
                native_class,
                node.func.attr,
                method,
                IntLoad(self.slots[node.func.value.id]),
                node,
                call_stack,
            )
            assert result is not None
            return result
        if (
            isinstance(node, ast.Name)
            and node.id in self.values
            and isinstance(self.values[node.id], (int, bool))
        ):
            return IntConstant(int(self.values[node.id]))
        if self.is_kernel_expression(node, bindings):
            value = self.kernel_value(node, bindings, call_stack)
            if isinstance(value, StaticI64Tensor):
                raise NativeCompileError(
                    self.path,
                    node,
                    f"static tensor with shape {value.shape} requires indexing "
                    "or a reduction before use as an integer",
                )
            return value
        if isinstance(node, ast.UnaryOp):
            operators = {
                ast.USub: "neg",
                ast.UAdd: "pos",
                ast.Not: "not",
                ast.Invert: "invert",
            }
            operator = operators.get(type(node.op))
            if operator is not None:
                return IntUnary(
                    operator,
                    self.integer(node.operand, bindings, call_stack),
                )
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
                right = self.integer(node.right, bindings, call_stack)
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
                return IntBinary(
                    operator,
                    self.integer(node.left, bindings, call_stack),
                    right,
                )
        if isinstance(node, ast.BoolOp) and node.values:
            # Python evaluates only the first operand unconditionally; the rest
            # are short-circuited, but this lowering evaluates them eagerly.
            result = self.integer(node.values[0], bindings, call_stack)
            for value in node.values[1:]:
                self.eager_depth += 1
                try:
                    right = self.integer(value, bindings, call_stack)
                finally:
                    self.eager_depth -= 1
                if isinstance(node.op, (ast.And, ast.Or)):
                    # The left operand is both the condition and one of the
                    # arms. Materialise it once, or the backends would re-emit
                    # the whole expression and evaluate it twice.
                    left_slot = self.new_temp()
                    self.operations.append(Store(left_slot, result))
                    left_value = IntLoad(left_slot)
                    result = (
                        self.select_integer(left_value, right, left_value)
                        if isinstance(node.op, ast.And)
                        else self.select_integer(left_value, left_value, right)
                    )
                else:
                    raise NativeCompileError(
                        self.path,
                        node,
                        "unsupported native Boolean operator",
                    )
            return result
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == len(node.comparators)
            and node.ops
        ):
            operators = {
                ast.Eq: "eq",
                ast.NotEq: "ne",
                ast.Lt: "lt",
                ast.LtE: "le",
                ast.Gt: "gt",
                ast.GtE: "ge",
            }
            left = node.left
            result: IntExpression | None = None
            for operator_node, right in zip(node.ops, node.comparators):
                operator = operators.get(type(operator_node))
                if operator is None:
                    raise NativeCompileError(
                        self.path,
                        node,
                        "unsupported native integer comparison",
                    )
                if (
                    self.expression_type(left, bindings) == "float"
                    or self.expression_type(right, bindings) == "float"
                ):
                    comparison = FloatCompare(
                        operator,
                        self.float_expression(left, bindings, call_stack),
                        self.float_expression(right, bindings, call_stack),
                    )
                else:
                    comparison = IntCompare(
                        operator,
                        self.integer(left, bindings, call_stack),
                        self.integer(right, bindings, call_stack),
                    )
                result = (
                    comparison
                    if result is None
                    else IntBinary("and", result, comparison)
                )
                left = right
            assert result is not None
            return result
        if isinstance(node, ast.IfExp):
            condition = self.integer(node.test, bindings, call_stack)
            # Both arms are evaluated eagerly, so neither may contain an
            # operation that can trap.
            self.eager_depth += 1
            try:
                body = self.integer(node.body, bindings, call_stack)
                alternative = self.integer(node.orelse, bindings, call_stack)
            finally:
                self.eager_depth -= 1
            return self.select_integer(condition, body, alternative)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.functions
        ):
            function = self.functions[node.func.id]
            if not function.returns_value:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native procedure {node.func.id}() does not produce a value",
                )
            identity = id(function)
            if identity in call_stack:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"recursive native function call to {node.func.id}() is not supported",
                )
            if len(call_stack) >= 64:
                raise NativeCompileError(
                    self.path,
                    node,
                    "native function inline depth exceeds 64 calls",
                )
            arguments = self.bind_native_arguments(
                node.func.id,
                function,
                node,
                bindings,
                call_stack,
            )
            if function.expression is None or _repeats_extern_argument(
                function, arguments
            ):
                if any(
                    isinstance(argument, StaticI64Tensor)
                    for argument in arguments
                ):
                    raise NativeCompileError(
                        self.path,
                        node,
                        "tensor parameters currently require an expression-only "
                        "function body; imperative tensor locals need the future "
                        "memory-layout runtime",
                    )
                return self.inline_imperative_function(
                    node.func.id,
                    function,
                    tuple(
                        argument
                        for argument in arguments
                        if not isinstance(argument, StaticI64Tensor)
                    ),
                    node,
                    call_stack,
                )
            previous_path = self.path
            previous_values = self.values
            previous_functions = self.functions
            previous_kernel_modules = self.kernel_modules
            previous_kernel_functions = self.kernel_functions
            previous_extern_functions = self.extern_functions
            self.path = function.path
            self.values = function.values
            self.functions = function.functions
            self.kernel_modules = function.kernel_modules
            self.kernel_functions = function.kernel_functions
            self.extern_functions = function.extern_functions
            try:
                return self.integer(
                    function.expression,
                    dict(zip(function.parameters, arguments)),
                    (*call_stack, identity),
                )
            finally:
                self.functions = previous_functions
                self.values = previous_values
                self.kernel_modules = previous_kernel_modules
                self.kernel_functions = previous_kernel_functions
                self.extern_functions = previous_extern_functions
                self.path = previous_path
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
    experimental_kernels: bool = False,
) -> Module:
    return Frontend(
        path,
        source_roots,
        experimental_kernels=experimental_kernels,
    ).compile(source)
