from __future__ import annotations

import ast
import copy
from collections import Counter
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

from .ir import (
    CStringConstant,
    Exit,
    ExitValue,
    ExternCall,
    BitsFloat,
    FloatBinary,
    FloatBits,
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
# The builtin exception hierarchy, as much of it as the native subset names.
# Matching an ``except`` clause against a raise is a compile-time question -
# both class names are literal in the source - so this table is all the type
# information the generated code needs. Nothing about a class survives into the
# binary; a live exception is a small integer identifying which raise produced
# it.
_EXCEPTION_BASES: dict[str, str | None] = {
    "BaseException": None,
    "Exception": "BaseException",
    "SystemExit": "BaseException",
    "KeyboardInterrupt": "BaseException",
    "ArithmeticError": "Exception",
    "ZeroDivisionError": "ArithmeticError",
    "OverflowError": "ArithmeticError",
    "AssertionError": "Exception",
    "AttributeError": "Exception",
    "BufferError": "Exception",
    "EOFError": "Exception",
    "ImportError": "Exception",
    "ModuleNotFoundError": "ImportError",
    "LookupError": "Exception",
    "IndexError": "LookupError",
    "KeyError": "LookupError",
    "MemoryError": "Exception",
    "NameError": "Exception",
    "UnboundLocalError": "NameError",
    "OSError": "Exception",
    "FileNotFoundError": "OSError",
    "PermissionError": "OSError",
    "NotImplementedError": "RuntimeError",
    "RecursionError": "RuntimeError",
    "RuntimeError": "Exception",
    "StopIteration": "Exception",
    "TypeError": "Exception",
    "ValueError": "Exception",
    "UnicodeError": "ValueError",
    "ZeroDivisionError": "ArithmeticError",
}


def exception_ancestry(name: str) -> tuple[str, ...]:
    """``name`` and every builtin exception class it inherits from."""

    chain: list[str] = []
    current: str | None = name
    while current is not None and current not in chain:
        chain.append(current)
        current = _EXCEPTION_BASES.get(current)
    return tuple(chain)


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


# The float-valued IR nodes, so an inlined argument can be recognised as one.
FLOAT_EXPRESSIONS = (
    FloatConstant,
    FloatLoad,
    FloatUnary,
    FloatBinary,
    IntToFloat,
    BitsFloat,
)


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
    # A field is an integer unless __init__ annotates it `float`. The slot is
    # eight bytes either way, so a float lives there as its bit pattern; the
    # annotation is how the layout learns which it is, since the type of the
    # value assigned there depends on the arguments at each call site.
    field_kinds: dict[str, str] = dataclass_field(default_factory=dict)

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
        # Exception state. Functions are inlined, so there are no runtime
        # frames to unwind: an active handler is a label in the same emitted
        # instruction stream, and propagation is a jump to it.
        self.handler_stack: list[str] = []
        self.exception_slot: int | None = None
        self.exception_value_slot: int | None = None
        self._dtoa_scratch_slot: int | None = None
        self._prologue: list[object] = []
        self._bool_text_slot: int | None = None
        self.boolean_names: set[str] = set()
        # A parameter substituted into a single-expression body is just a
        # value, and a string's value is a pointer - indistinguishable from an
        # integer. This records which of them are strings.
        self.string_bindings: dict[str, IntExpression] = {}
        self.exception_ids: dict[str, int] = {}
        # Each cleanup scope (a `finally`, or a `with`'s `__exit__`) records
        # how deep the jump stacks were when it opened. A jump is only a
        # problem when it would leave the scope: one inside a function or loop
        # that opened later stays within it, which is what makes a `return self`
        # in an inlined __enter__ harmless.
        self.finally_scopes: list[tuple[int, int, int]] = []
        self.continue_targets: list[str] = []
        self.return_targets: list[tuple[int | None, str]] = []
        self.active_functions: list[tuple[int, str]] = []

    def compile(self, source: str) -> Module:
        try:
            tree = ast.parse(source, filename=str(self.path))
        except SyntaxError as error:
            raise ValueError(f"{self.path}:{error.lineno}:{error.offset}: {error.msg}") from error
        self.runtime_names.update(self.loop_mutated_names(tree))
        # A name a function declares global has to live in a slot. Inlining
        # swaps the build-time constant map for the function's own, so a
        # constant written inside the body would be dropped when the module's
        # map came back.
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                self.runtime_names.update(node.names)
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
            for index, operation in enumerate(self._prologue):
                self.operations.insert(1 + index, operation)
            if self._dtoa_scratch_slot is not None:
                # Float rendering needs six big integers and two buffers. They
                # are reused, so reserve them once here rather than allocating
                # on every print - which in a loop would exhaust the arena.
                self.operations.insert(
                    1,
                    HeapAlloc(
                        self._dtoa_scratch_slot,
                        IntConstant(self.DTOA_SCRATCH_BYTES),
                        self._heap_bump_slot,
                    ),
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

    @classmethod
    def target_names(cls, target: ast.expr) -> set[str]:
        """Every name an assignment target binds, unpacking tuples."""

        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            names: set[str] = set()
            for item in target.elts:
                names.update(cls.target_names(item))
            return names
        return set()

    @classmethod
    def assigned_names(cls, nodes: list[ast.stmt]) -> set[str]:
        names: set[str] = set()
        for statement in nodes:
            for node in ast.walk(statement):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        # A tuple target assigns every name in it; missing that
                        # left `a, b = ...` inside a loop folding as constants.
                        names.update(cls.target_names(target))
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names.add(node.target.id)
                elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    names.add(node.target.id)
                elif isinstance(node, ast.For):
                    names.update(cls.target_names(node.target))
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
                self.names.update(cls.target_names(node.target))
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
        # A name declared global is the module's, so it must not be renamed
        # into a private local when the body is inlined - that renaming is what
        # makes every other assignment in a function local.
        for statement in body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Global):
                    names.difference_update(node.names)
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
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and isinstance(node.value, (ast.Tuple, ast.List))
        ):
            self.parallel_assignment(node.targets[0], node.value)
        elif isinstance(node, ast.Assign) and len(node.targets) > 1:
            # `a = b = value`: one evaluation, several names.
            first = node.targets[0]
            if not all(isinstance(item, ast.Name) for item in node.targets):
                raise NativeCompileError(
                    self.path, node, "a native chained assignment binds names"
                )
            self.assignment(first.id, node.value)
            for other in node.targets[1:]:
                self.assignment(
                    other.id, ast.copy_location(ast.Name(id=first.id, ctx=ast.Load()), node)
                )
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
            self.assignment(node.target.id, node.value, node.annotation)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if (
                self.value_types.get(node.target.id) == "float"
                or isinstance(self.values.get(node.target.id), float)
            ):
                if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                    raise NativeCompileError(
                        self.path,
                        node,
                        "native float augmented assignment supports + - * /",
                    )
                name = node.target.id
                # A constant-folded name has no slot yet; give it one, because
                # the result of this statement is a runtime value.
                self.materialize_runtime_names({name})
                # Lowering the equivalent binary operation reuses every rule
                # that applies to one, including the constant-divisor
                # restriction on runtime float division.
                combined = ast.copy_location(
                    ast.BinOp(left=node.target, op=node.op, right=node.value), node
                )
                self.operations.append(
                    FloatStore(self.slot(name), self.float_expression(combined))
                )
                self.value_types[name] = "float"
                return
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
            self.boolean_names.discard(name)
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
            if self.jump_escapes_cleanup(1, self.break_targets):
                raise NativeCompileError(
                    self.path,
                    node,
                    "native break cannot leave a try that has a finally: the "
                    "finally body is emitted on each path out, and this one has "
                    "no path to emit it on",
                )
            self.operations.append(Jump(self.break_targets[-1]))
        elif isinstance(node, ast.Continue):
            if not self.continue_targets:
                raise NativeCompileError(self.path, node, "continue is outside a native loop")
            if self.jump_escapes_cleanup(2, self.continue_targets):
                raise NativeCompileError(
                    self.path,
                    node,
                    "native continue cannot leave a try that has a finally: the "
                    "finally body is emitted on each path out, and this one has "
                    "no path to emit it on",
                )
            self.operations.append(Jump(self.continue_targets[-1]))
        elif isinstance(node, ast.Return):
            if not self.return_targets:
                raise NativeCompileError(self.path, node, "return is outside a native function")
            if self.jump_escapes_cleanup(0, self.return_targets):
                raise NativeCompileError(
                    self.path,
                    node,
                    "native return cannot leave a try that has a finally",
                )
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
            if self.expression_type(node.value) == "float":
                raise NativeCompileError(
                    self.path,
                    node,
                    "a native function cannot return a float yet: the call site "
                    "has to choose an integer or float lowering before the body "
                    "is inlined, and the returned kind is only known afterwards; "
                    "pass a float in and assign the result to an attribute or "
                    "list element instead",
                )
            self.operations.append(Store(result_slot, self.integer(node.value)))
            self.operations.append(Jump(return_label))
        elif isinstance(node, ast.Global):
            # Handled where a function's locals are chosen; nothing to emit.
            return
        elif isinstance(node, ast.Pass):
            return
        elif isinstance(node, ast.FunctionDef):
            self.function_definition(node)
        elif isinstance(node, ast.ImportFrom):
            self.import_from(node)
        elif isinstance(node, ast.Import):
            self.import_statement(node)
        elif isinstance(node, ast.Raise):
            self.raise_statement(node)
        elif isinstance(node, ast.With):
            self.with_statement(node)
        elif isinstance(node, ast.AsyncWith):
            raise NativeCompileError(
                self.path, node, "native code has no event loop, so `async with` "
                "is not in the subset"
            )
        elif isinstance(node, (ast.Try, ast.TryStar)):
            if isinstance(node, ast.TryStar):
                raise NativeCompileError(
                    self.path, node, "native try does not support exception groups"
                )
            self.try_statement(node)
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
            # __enter__/__exit__ are resolved at build time by `with`, so they
            # need no run-time protocol; the rest still have none.
            if name in {"__enter__", "__exit__"}:
                continue
            if name.startswith("__") and name.endswith("__"):
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native classes do not implement the special method {name}()",
                )
        fields, field_kinds = self.discover_fields(node, initializer)
        self.classes[node.name] = NativeClass(
            node.name, self.path, fields, initializer, methods, field_kinds
        )

    def discover_fields(
        self, node: ast.ClassDef, initializer: NativeFunction | None
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        """Derive the instance layout from ``self.NAME = ...`` in ``__init__``.

        Every attribute must be assigned in ``__init__`` so each instance has a
        complete, statically known layout. An attribute first assigned anywhere
        else would have no reserved storage, so it is rejected rather than
        silently writing outside the object.
        """

        if initializer is None:
            return (), {}

        kinds: dict[str, str] = {}

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
                        annotation = getattr(statement, "annotation", None)
                        if (
                            isinstance(annotation, ast.Name)
                            and annotation.id == "float"
                        ):
                            kinds[target.attr] = "float"
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
        return tuple(fields), kinds

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
        if self.attribute_kind(target) == "float":
            if self.expression_type(value) not in {"float", "int"}:
                raise NativeCompileError(
                    self.path, value, "this attribute holds a float"
                )
            address = self.attribute_address(target)
            self.operations.append(
                HeapStore(address, FloatBits(self.float_expression(value)), 8)
            )
            return
        if self.expression_type(value) != "int":
            raise NativeCompileError(
                self.path,
                value,
                "native object attributes are signed 64-bit integers",
            )
        address = self.attribute_address(target)
        self.operations.append(HeapStore(address, self.integer(value), 8))

    def attribute_kind(self, node: ast.Attribute) -> str:
        """``"float"`` if the class annotated this attribute so, else ``"int"``."""

        native_class = self.resolve_object_class(node.value)
        if native_class is None:
            return "int"
        return native_class.field_kinds.get(node.attr, "int")

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

        argument_kinds: list[str] = []
        arguments = self.bind_native_arguments(
            f"{native_class.name}.{method_name}",
            method,
            node,
            {},
            call_stack,
            skip_parameters=1,
            kinds=argument_kinds,
        )
        return self.inline_imperative_function(
            f"{native_class.name}.{method_name}",
            method,
            (instance, *arguments),
            node,
            call_stack,
            parameter_classes={"self": native_class.name},
            # `self` is the leading argument the caller supplied, so the kinds
            # the binding produced line up one position later.
            argument_kinds=("object", *argument_kinds),
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

    def assignment(
        self,
        name: str,
        expression: ast.expr,
        annotation: ast.expr | None = None,
    ) -> None:
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
                # The name is becoming a runtime value. If it currently holds a
                # compile-time constant, write that into its slot first: the
                # new right-hand side may read the name (``x = x + f()``), and
                # the slot has never been written.
                previous_value = self.values.get(name)
                if isinstance(previous_value, (bool, int)):
                    self.operations.append(
                        Store(self.slot(name), IntConstant(int(previous_value)))
                    )
                    self.value_types.setdefault(name, "int")
                elif isinstance(previous_value, float):
                    self.operations.append(
                        FloatStore(self.slot(name), FloatConstant(previous_value))
                    )
                    self.value_types.setdefault(name, "float")
                self.runtime_names.add(name)
        self.values.pop(name, None)
        kind = self.expression_type(expression)
        if annotation is not None and isinstance(expression, ast.List):
            declared = self.annotated_list_tag(annotation)
            if declared is not None:
                if kind != "list:int" and kind != declared:
                    raise NativeCompileError(
                        self.path,
                        expression,
                        f"this literal builds a {kind} but the annotation says "
                        f"{declared}",
                    )
                kind = declared
        if annotation is not None and isinstance(expression, ast.Dict):
            # `d: dict[str, float] = {}` says what an empty literal cannot.
            declared = self.annotated_dict_tag(annotation)
            if declared is not None:
                if kind != "dict:int:int" and kind != declared:
                    raise NativeCompileError(
                        self.path,
                        expression,
                        f"this literal builds a {kind} but the annotation says "
                        f"{declared}",
                    )
                kind = declared
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
        if kind == "int" and self.renders_as_bool(expression):
            self.boolean_names.add(name)
        else:
            self.boolean_names.discard(name)
        if kind == "object":
            self.object_assignment(name, expression)
        elif self.dict_kinds(kind) is not None:
            key_kind, value_kind = self.dict_kinds(kind)
            self.dict_assignment(name, expression, key_kind, value_kind)
        elif self.list_kind(kind) is not None:
            self.list_assignment(name, expression, self.list_kind(kind))
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

    @staticmethod
    def list_tag(element_kind: str) -> str:
        return f"list:{element_kind}"

    @staticmethod
    def list_kind(tag: str | None) -> str | None:
        """The element kind of a list tag, else None."""

        if not isinstance(tag, str) or not tag.startswith("list:"):
            return None
        return tag.split(":", 1)[1]

    def list_kind_of(self, name: str) -> str | None:
        return self.list_kind(self.value_types.get(name))

    def list_literal_tag(
        self, node: ast.List, bindings: dict[str, KernelValue] | None = None
    ) -> str:
        """The element kind a list literal builds, read off its first element.

        An empty literal has nothing to read, so it holds integers unless an
        annotation such as `xs: list[float] = []` says otherwise.
        """

        if not node.elts:
            return self.list_tag("int")
        kind = self.expression_type(node.elts[0], bindings)
        if kind not in {"int", "float"}:
            raise NativeCompileError(
                self.path,
                node.elts[0],
                "native lists hold signed 64-bit integers or floats",
            )
        return self.list_tag(kind)

    def annotated_list_tag(self, annotation: ast.expr) -> str | None:
        """The element kind `list[T]` names, or None if it is not that shape."""

        if (
            not isinstance(annotation, ast.Subscript)
            or not isinstance(annotation.value, ast.Name)
            or annotation.value.id not in {"list", "List"}
            or not isinstance(annotation.slice, ast.Name)
        ):
            return None
        if annotation.slice.id not in {"int", "float"}:
            raise NativeCompileError(
                self.path, annotation, "a native list annotation is list[int|float]"
            )
        return self.list_tag(annotation.slice.id)

    # --- slicing ------------------------------------------------------------
    #
    # Python slices never raise: a bound past either end is pulled back to it,
    # a negative one counts from the end, and a start past the stop yields
    # nothing. So the arithmetic below clamps rather than checks, which is what
    # makes indexing (which does raise) and slicing (which does not) different
    # operations here rather than one with a flag.

    def slice_bound(
        self,
        value: ast.expr | None,
        length: IntExpression,
        default: IntExpression,
    ) -> IntExpression:
        if value is None:
            return default
        bound = self.materialize_int(self.integer(value))
        adjusted = self.materialize_int(
            self.select_integer(
                IntCompare("lt", bound, IntConstant(0)),
                IntBinary("add", bound, length),
                bound,
            )
        )
        return self.materialize_int(
            self.select_integer(
                IntCompare("lt", adjusted, IntConstant(0)),
                IntConstant(0),
                self.select_integer(
                    IntCompare("gt", adjusted, length), length, adjusted
                ),
            )
        )

    def slice_bounds(
        self, node: ast.Slice, length: IntExpression
    ) -> tuple[IntExpression, IntExpression]:
        if node.step is not None:
            try:
                step = self.constant(node.step)
            except NativeCompileError:
                step = None
            if step != 1:
                raise NativeCompileError(
                    self.path,
                    node,
                    "a native slice steps by one; another step would have to "
                    "walk the source backwards or skip through it",
                )
        lower = self.slice_bound(node.lower, length, IntConstant(0))
        upper = self.slice_bound(node.upper, length, length)
        # A start past the stop is an empty slice, not a negative length.
        upper = self.materialize_int(
            self.select_integer(IntCompare("lt", upper, lower), lower, upper)
        )
        return lower, upper

    def emit_list_slice(
        self, source: IntExpression, node: ast.Slice
    ) -> IntExpression:
        bump = self.ensure_heap()
        source_slot = self.new_temp()
        self.operations.append(Store(source_slot, source))
        origin = IntLoad(source_slot)
        length = self.materialize_int(
            HeapLoad(IntBinary("add", origin, IntConstant(8)), 8)
        )
        lower, upper = self.slice_bounds(node, length)
        count_slot = self.new_temp()
        self.operations.append(
            Store(count_slot, IntBinary("sub", upper, lower))
        )
        result_slot = self.new_temp()
        self.operations.append(
            HeapAlloc(
                result_slot,
                IntBinary(
                    "add",
                    IntConstant(self.LIST_HEADER_BYTES),
                    IntBinary("mul", IntLoad(count_slot), IntConstant(8)),
                ),
                bump,
            )
        )
        result = IntLoad(result_slot)
        self.operations.append(HeapStore(result, IntLoad(count_slot), 8))
        self.operations.append(
            HeapStore(
                IntBinary("add", result, IntConstant(8)), IntLoad(count_slot), 8
            )
        )
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        start = self.new_label("slice_copy")
        end = self.new_label("slice_copy_end")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(count_slot)), end
            )
        )
        offset = IntBinary("mul", IntLoad(index_slot), IntConstant(8))
        self.operations.append(
            HeapStore(
                IntBinary(
                    "add",
                    IntBinary("add", result, IntConstant(self.LIST_HEADER_BYTES)),
                    offset,
                ),
                HeapLoad(
                    IntBinary(
                        "add",
                        IntBinary(
                            "add", origin, IntConstant(self.LIST_HEADER_BYTES)
                        ),
                        IntBinary(
                            "add",
                            IntBinary("mul", lower, IntConstant(8)),
                            offset,
                        ),
                    ),
                    8,
                ),
                8,
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
        return result

    def emit_codepoint_offset(
        self, pointer: IntExpression, count: IntExpression
    ) -> IntExpression:
        """The byte offset of the ``count``-th code point of a UTF-8 block."""

        pointer_slot = self.new_temp()
        self.operations.append(Store(pointer_slot, pointer))
        base = IntLoad(pointer_slot)
        bytes_slot = self.new_temp()
        self.operations.append(Store(bytes_slot, HeapLoad(base, 8)))
        wanted_slot = self.new_temp()
        self.operations.append(Store(wanted_slot, count))
        offset_slot = self.new_temp()
        self.operations.append(Store(offset_slot, IntConstant(0)))
        seen_slot = self.new_temp()
        self.operations.append(Store(seen_slot, IntConstant(0)))
        start = self.new_label("cp_walk")
        end = self.new_label("cp_walk_end")
        inner = self.new_label("cp_tail")
        inner_end = self.new_label("cp_tail_end")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(seen_slot), IntLoad(wanted_slot)), end
            )
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(offset_slot), IntLoad(bytes_slot)), end
            )
        )
        self.operations.append(
            Store(offset_slot, IntBinary("add", IntLoad(offset_slot), IntConstant(1)))
        )
        # Continuation bytes belong to the code point just stepped over.
        self.operations.append(Label(inner))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(offset_slot), IntLoad(bytes_slot)), inner_end
            )
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "eq",
                    IntBinary(
                        "and",
                        HeapLoad(
                            IntBinary(
                                "add",
                                IntBinary("add", base, IntConstant(8)),
                                IntLoad(offset_slot),
                            ),
                            1,
                        ),
                        IntConstant(0xC0),
                    ),
                    IntConstant(0x80),
                ),
                inner_end,
            )
        )
        self.operations.append(
            Store(offset_slot, IntBinary("add", IntLoad(offset_slot), IntConstant(1)))
        )
        self.operations.append(Jump(inner))
        self.operations.append(Label(inner_end))
        self.operations.append(
            Store(seen_slot, IntBinary("add", IntLoad(seen_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
        return IntLoad(offset_slot)

    def emit_string_slice(
        self, source: IntExpression, node: ast.Slice
    ) -> IntExpression:
        bump = self.ensure_heap()
        source_slot = self.new_temp()
        self.operations.append(Store(source_slot, source))
        origin = IntLoad(source_slot)
        # Slice bounds count code points, as CPython does, so they have to be
        # turned into byte offsets before anything is copied.
        length = self.materialize_int(self.emit_code_point_count(origin))
        lower, upper = self.slice_bounds(node, length)
        first = self.materialize_int(self.emit_codepoint_offset(origin, lower))
        last = self.materialize_int(self.emit_codepoint_offset(origin, upper))
        span_slot = self.new_temp()
        self.operations.append(Store(span_slot, IntBinary("sub", last, first)))
        result_slot = self.new_temp()
        self.operations.append(
            HeapAlloc(result_slot, self._aligned_size(IntLoad(span_slot)), bump)
        )
        result = IntLoad(result_slot)
        self.operations.append(HeapStore(result, IntLoad(span_slot), 8))
        self.emit_byte_copy(
            IntBinary("add", result, IntConstant(8)),
            IntBinary("add", IntBinary("add", origin, IntConstant(8)), first),
            IntLoad(span_slot),
        )
        return result

    def parallel_assignment(self, target, value) -> None:
        """`a, b = b, a` - every right-hand side is read before anything moves.

        Without the temporaries this would be two assignments in sequence, and
        the second would read the name the first had already overwritten.
        """

        if len(target.elts) != len(value.elts):
            raise NativeCompileError(
                self.path,
                target,
                f"native parallel assignment needs matching lengths: "
                f"{len(target.elts)} names, {len(value.elts)} values",
            )
        for item in target.elts:
            if not isinstance(item, ast.Name):
                raise NativeCompileError(
                    self.path, item, "a native parallel assignment binds names"
                )
        holders: list[str] = []
        for index, item in enumerate(value.elts):
            holder = f"<swap-{self.new_label('slot')}:{index}>"
            self.assignment(holder, item)
            holders.append(holder)
        for name, holder in zip(target.elts, holders):
            self.assignment(
                name.id,
                ast.copy_location(ast.Name(id=holder, ctx=ast.Load()), target),
            )

    _AGGREGATES = {"sum": "add", "min": "lt", "max": "gt"}

    def aggregate_call(self, node: ast.Call, bindings, call_stack):
        """`sum(xs)`, `min(xs)`, `max(xs)` over a runtime list."""

        name = node.func.id
        if len(node.args) != 1 or node.keywords:
            raise NativeCompileError(
                self.path, node, f"native {name}() takes one iterable"
            )
        source = node.args[0]
        element_kind = self.list_kind(self.expression_type(source, bindings))
        if element_kind is None:
            raise NativeCompileError(
                self.path, node, f"native {name}() takes a runtime list"
            )
        if element_kind != "int":
            raise NativeCompileError(
                self.path,
                node,
                f"native {name}() works on integer lists; a float one would "
                "need a float accumulator this call cannot return",
            )
        pointer_slot = self.new_temp()
        self.operations.append(Store(pointer_slot, self.list_pointer(source)))
        pointer = IntLoad(pointer_slot)
        length_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntBinary("add", pointer, IntConstant(8)), 8))
        )
        if name != "sum":
            # min() and max() of an empty list raise; nothing to return here.
            ok = self.new_label("aggregate_ok")
            self.operations.append(
                JumpIfFalse(
                    IntCompare("gt", IntLoad(length_slot), IntConstant(0)),
                    ok + "_empty",
                )
            )
            self.operations.append(Jump(ok))
            self.operations.append(Label(ok + "_empty"))
            self.raise_exception(
                "ValueError",
                f"ValueError: {name}() iterable argument is empty\n".encode(),
            )
            self.operations.append(Label(ok))
        result_slot = self.new_temp()
        first = IntBinary(
            "add", pointer, IntConstant(self.LIST_HEADER_BYTES)
        )
        self.operations.append(
            Store(
                result_slot,
                IntConstant(0) if name == "sum" else HeapLoad(first, 8),
            )
        )
        index_slot = self.new_temp()
        self.operations.append(
            Store(index_slot, IntConstant(0 if name == "sum" else 1))
        )
        start = self.new_label("aggregate")
        end = self.new_label("aggregate_end")
        step = self.new_label("aggregate_next")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), end
            )
        )
        item = HeapLoad(
            IntBinary(
                "add", first, IntBinary("mul", IntLoad(index_slot), IntConstant(8))
            ),
            8,
        )
        if name == "sum":
            self.operations.append(
                Store(result_slot, IntBinary("add", IntLoad(result_slot), item))
            )
        else:
            item_slot = self.new_temp()
            self.operations.append(Store(item_slot, item))
            self.operations.append(
                JumpIfFalse(
                    IntCompare(
                        self._AGGREGATES[name],
                        IntLoad(item_slot),
                        IntLoad(result_slot),
                    ),
                    step,
                )
            )
            self.operations.append(Store(result_slot, IntLoad(item_slot)))
        self.operations.append(Label(step))
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
        return IntLoad(result_slot)

    def membership_container_kind(self, node: ast.expr) -> str | None:
        """What `x in node` would search: a dict, a string, or a list's kind."""

        if isinstance(node, ast.Name) and self.dict_kinds_of(node.id):
            return "dict"
        try:
            kind = self.expression_type(node)
        except NativeCompileError:
            return None
        if kind == "str":
            return "str"
        return self.list_kind(kind)

    def emit_list_membership(
        self, node: ast.expr, container: ast.expr, element_kind: str
    ) -> int:
        """Return a 0/1 slot saying whether ``node`` is in the list."""

        pointer_slot = self.new_temp()
        self.operations.append(Store(pointer_slot, self.list_pointer(container)))
        pointer = IntLoad(pointer_slot)
        length_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntBinary("add", pointer, IntConstant(8)), 8))
        )
        found_slot = self.new_temp()
        self.operations.append(Store(found_slot, IntConstant(0)))
        wanted_slot = self.new_temp()
        if element_kind == "float":
            self.operations.append(
                FloatStore(wanted_slot, self.float_expression(node))
            )
        else:
            self.operations.append(Store(wanted_slot, self.integer(node)))
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        start = self.new_label("member")
        end = self.new_label("member_end")
        step = self.new_label("member_next")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), end
            )
        )
        item = HeapLoad(
            IntBinary(
                "add",
                IntBinary("add", pointer, IntConstant(self.LIST_HEADER_BYTES)),
                IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
            ),
            8,
        )
        if element_kind == "float":
            # Compare as floats, not as bits: 0.0 equals -0.0 and a NaN equals
            # nothing, and the bit patterns say otherwise on both counts.
            same = FloatCompare("eq", BitsFloat(item), FloatLoad(wanted_slot))
        else:
            same = IntCompare("eq", item, IntLoad(wanted_slot))
        self.operations.append(JumpIfFalse(same, step))
        self.operations.append(Store(found_slot, IntConstant(1)))
        self.operations.append(Jump(end))
        self.operations.append(Label(step))
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
        return found_slot

    def emit_substring_search(self, needle: ast.expr, haystack: ast.expr) -> int:
        """Return a 0/1 slot saying whether one string contains the other.

        A plain scan: for every starting byte, compare forward. UTF-8 makes
        that safe without decoding, because a multi-byte character can never
        match part of another one - lead and continuation bytes come from
        disjoint ranges.
        """

        outer_slot = self.new_temp()
        inner_slot = self.new_temp()
        self.operations.append(Store(outer_slot, self.string_pointer(haystack)))
        self.operations.append(Store(inner_slot, self.string_pointer(needle)))
        outer = IntBinary("add", IntLoad(outer_slot), IntConstant(8))
        inner = IntBinary("add", IntLoad(inner_slot), IntConstant(8))
        outer_length = self.new_temp()
        inner_length = self.new_temp()
        self.operations.append(
            Store(outer_length, HeapLoad(IntLoad(outer_slot), 8))
        )
        self.operations.append(
            Store(inner_length, HeapLoad(IntLoad(inner_slot), 8))
        )
        found_slot = self.new_temp()
        self.operations.append(Store(found_slot, IntConstant(0)))
        start_slot = self.new_temp()
        self.operations.append(Store(start_slot, IntConstant(0)))
        scan = self.new_label("find")
        done = self.new_label("find_done")
        next_start = self.new_label("find_next")
        self.operations.append(Label(scan))
        # Stop once the remainder is shorter than what is being looked for;
        # this also makes the empty needle match at once, as Python does.
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "le",
                    IntBinary("add", IntLoad(start_slot), IntLoad(inner_length)),
                    IntLoad(outer_length),
                ),
                done,
            )
        )
        cursor_slot = self.new_temp()
        self.operations.append(Store(cursor_slot, IntConstant(0)))
        compare = self.new_label("find_compare")
        matched = self.new_label("find_matched")
        self.operations.append(Label(compare))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(cursor_slot), IntLoad(inner_length)),
                matched,
            )
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "eq",
                    HeapLoad(
                        IntBinary(
                            "add",
                            outer,
                            IntBinary(
                                "add", IntLoad(start_slot), IntLoad(cursor_slot)
                            ),
                        ),
                        1,
                    ),
                    HeapLoad(
                        IntBinary("add", inner, IntLoad(cursor_slot)), 1
                    ),
                ),
                next_start,
            )
        )
        self.operations.append(
            Store(cursor_slot, IntBinary("add", IntLoad(cursor_slot), IntConstant(1)))
        )
        self.operations.append(Jump(compare))
        self.operations.append(Label(matched))
        self.operations.append(Store(found_slot, IntConstant(1)))
        self.operations.append(Jump(done))
        self.operations.append(Label(next_start))
        self.operations.append(
            Store(start_slot, IntBinary("add", IntLoad(start_slot), IntConstant(1)))
        )
        self.operations.append(Jump(scan))
        self.operations.append(Label(done))
        return found_slot

    def emit_list_append(
        self, pointer_slot: int, value: IntExpression
    ) -> None:
        """Append to a runtime list, moving it when it is full.

        The block cannot be extended in place - the arena hands out addresses
        in order and something else may already sit behind this one - so a full
        list is copied into a block of twice the capacity. The abandoned one
        stays in the arena, which never reclaims; that is what makes appending
        amortised rather than free.
        """

        bump = self.ensure_heap()
        value_slot = self.new_temp()
        self.operations.append(Store(value_slot, value))
        pointer = IntLoad(pointer_slot)
        length_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntBinary("add", pointer, IntConstant(8)), 8))
        )
        room = self.new_label("append_room")
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(length_slot), HeapLoad(pointer, 8)),
                room + "_full",
            )
        )
        self.operations.append(Jump(room))
        self.operations.append(Label(room + "_full"))

        grown_slot = self.new_temp()
        capacity_slot = self.new_temp()
        self.operations.append(
            Store(
                capacity_slot,
                self.select_integer(
                    IntCompare("gt", HeapLoad(pointer, 8), IntConstant(0)),
                    IntBinary("mul", HeapLoad(pointer, 8), IntConstant(2)),
                    IntConstant(4),
                ),
            )
        )
        self.operations.append(
            HeapAlloc(
                grown_slot,
                IntBinary(
                    "add",
                    IntConstant(self.LIST_HEADER_BYTES),
                    IntBinary("mul", IntLoad(capacity_slot), IntConstant(8)),
                ),
                bump,
            )
        )
        grown = IntLoad(grown_slot)
        self.operations.append(HeapStore(grown, IntLoad(capacity_slot), 8))
        self.operations.append(
            HeapStore(
                IntBinary("add", grown, IntConstant(8)), IntLoad(length_slot), 8
            )
        )
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        copy = self.new_label("append_copy")
        copied = self.new_label("append_copied")
        self.operations.append(Label(copy))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), copied
            )
        )
        offset = IntBinary(
            "add",
            IntConstant(self.LIST_HEADER_BYTES),
            IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", grown, offset),
                HeapLoad(IntBinary("add", pointer, offset), 8),
                8,
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(copy))
        self.operations.append(Label(copied))
        self.operations.append(Store(pointer_slot, IntLoad(grown_slot)))
        self.operations.append(Label(room))

        self.operations.append(
            HeapStore(
                IntBinary(
                    "add",
                    IntBinary(
                        "add", IntLoad(pointer_slot), IntConstant(self.LIST_HEADER_BYTES)
                    ),
                    IntBinary("mul", IntLoad(length_slot), IntConstant(8)),
                ),
                IntLoad(value_slot),
                8,
            )
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", IntLoad(pointer_slot), IntConstant(8)),
                IntBinary("add", IntLoad(length_slot), IntConstant(1)),
                8,
            )
        )

    def list_method_call(self, node: ast.Call) -> bool:
        """Lower `xs.append(v)`; returns whether this was one."""

        if (
            not isinstance(node.func, ast.Attribute)
            or not isinstance(node.func.value, ast.Name)
            or self.list_kind_of(node.func.value.id) is None
        ):
            return False
        name = node.func.value.id
        if node.func.attr != "append":
            raise NativeCompileError(
                self.path,
                node,
                f"native lists support append(); {node.func.attr}() is not in "
                "the subset",
            )
        if len(node.args) != 1 or node.keywords:
            raise NativeCompileError(
                self.path, node, "append() takes exactly one argument"
            )
        element_kind = self.list_kind_of(name)
        argument = node.args[0]
        if element_kind == "float":
            if self.expression_type(argument) not in {"float", "int"}:
                raise NativeCompileError(
                    self.path, argument, "this list holds floats"
                )
            value = FloatBits(self.float_expression(argument))
        else:
            if self.expression_type(argument) != "int":
                raise NativeCompileError(
                    self.path, argument, "this list holds signed 64-bit integers"
                )
            value = self.integer(argument)
        # A literal's length stops being a build-time fact once it can grow.
        self.list_lengths.pop(name, None)
        self.emit_list_append(self.slot(name), value)
        return True

    def list_pointer(self, node: ast.expr) -> IntExpression:
        """A pointer to a runtime list block, building one if needed."""

        if isinstance(node, ast.Name) and self.list_kind_of(node.id) is not None:
            return IntLoad(self.slots[node.id])
        if isinstance(node, ast.ListComp):
            return self.list_comprehension(node)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            return self.emit_list_slice(
                self.list_pointer(node.value), node.slice
            )
        raise NativeCompileError(
            self.path, node, "expression is not a native runtime list"
        )

    def list_assignment(
        self, name: str, node: ast.expr, element_kind: str
    ) -> None:
        if not isinstance(node, ast.List):
            # Not a literal, but perhaps a list-valued expression such as a
            # slice; that block is already built, so just bind the name to it.
            pointer = self.list_pointer(node)
            self.runtime_names.add(name)
            self.operations.append(Store(self.slot(name), pointer))
            return
        elements = node.elts
        for element in elements:
            if element_kind == "float":
                if self.expression_type(element) not in {"float", "int"}:
                    raise NativeCompileError(
                        self.path, element, "this list holds floats"
                    )
            elif self.expression_type(element) != "int":
                raise NativeCompileError(
                    self.path,
                    element,
                    "this list holds signed 64-bit integers",
                )
        bump = self.ensure_heap()
        self.runtime_names.add(name)
        pointer_slot = self.slot(name)
        length = len(elements)
        # Layout: [i64 capacity][i64 length][element0][element1]...
        # The capacity is what makes append() possible: without it there is
        # nowhere to put the next item and no way to know when to move.
        capacity = max(4, length)
        size = self.LIST_HEADER_BYTES + capacity * 8
        self.operations.append(HeapAlloc(pointer_slot, IntConstant(size), bump))
        self.operations.append(
            HeapStore(IntLoad(pointer_slot), IntConstant(capacity), 8)
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", IntLoad(pointer_slot), IntConstant(8)),
                IntConstant(length),
                8,
            )
        )
        for index, element in enumerate(elements):
            address = IntBinary(
                "add",
                IntLoad(pointer_slot),
                IntConstant(self.LIST_HEADER_BYTES + index * 8),
            )
            if element_kind == "float":
                stored = FloatBits(self.float_expression(element))
            else:
                stored = self.integer(element)
            self.operations.append(HeapStore(address, stored, 8))
        self.list_lengths[name] = length

    def list_element_address(self, node: ast.Subscript) -> IntExpression:
        target = node.value
        if isinstance(target, ast.Name) and self.list_kind_of(target.id) is not None:
            pointer = IntLoad(self.slots[target.id])
        elif self.list_kind(self.expression_type(target)) is not None:
            # A slice or a comprehension is a list too; index the block it
            # built rather than insisting on a name.
            pointer = self.materialize_int(self.list_pointer(target))
        else:
            raise NativeCompileError(
                self.path, node, "native indexing requires a runtime list"
            )
        index_node = node.slice
        try:
            folded = self.constant(index_node)
        except NativeCompileError:
            folded = None
        # Only a named list has a length known at build time; a slice or a
        # comprehension is measured at run time by the check further down, and
        # a negative constant index needs that length to resolve at all.
        length = (
            self.list_lengths.get(target.id) if isinstance(target, ast.Name) else None
        )
        # The constant path proves the index in range at build time, which it
        # can only do when the length is known then. A slice or a comprehension
        # is measured at run time, so even a constant index goes through the
        # run-time check below rather than skipping it.
        if (
            isinstance(folded, int)
            and not isinstance(folded, bool)
            and length is not None
        ):
            resolved = folded
            if resolved < 0:
                # Python counts a negative index from the end.
                resolved += length
            if resolved < 0 or (length is not None and resolved >= length):
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native list index {folded} is out of range"
                    + (
                        f" for {target.id!r}"
                        if isinstance(target, ast.Name)
                        else ""
                    )
                    + (f" (length {length})" if length is not None else ""),
                )
            return IntBinary(
                "add", pointer, IntConstant(self.LIST_HEADER_BYTES + resolved * 8)
            )
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
        self.operations.append(
            Store(
                length_slot,
                HeapLoad(IntBinary("add", pointer, IntConstant(8)), 8),
            )
        )
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
        # Uncaught, this prints to stderr and exits 1 as CPython does; inside a
        # try it goes to the handler, so `except IndexError` works on it.
        self.raise_exception(
            "IndexError", b"IndexError: list index out of range\n"
        )
        self.operations.append(Label(ok_label))
        offset = IntBinary(
            "add",
            IntConstant(self.LIST_HEADER_BYTES),
            IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
        )
        return IntBinary("add", pointer, offset)

    def subscript_dict_kinds(self, node: ast.Subscript) -> tuple[str, str] | None:
        if not isinstance(node.value, ast.Name):
            return None
        return self.dict_kinds_of(node.value.id)

    def dict_lookup_value_address(
        self, node: ast.Subscript, bindings: dict[str, IntExpression] | None = None
    ) -> IntExpression:
        """Probe for ``d[k]``, raise KeyError when absent, return where the
        value lives."""

        kinds = self.subscript_dict_kinds(node)
        assert kinds is not None and isinstance(node.value, ast.Name)
        key_kind, _value_kind = kinds
        if self.eager_depth:
            raise NativeCompileError(
                self.path,
                node,
                "a dict lookup can raise KeyError, so it cannot appear in a "
                "conditional expression or a short-circuited operand, whose "
                "arms are both evaluated here; use an if statement",
            )
        if self.expression_type(node.slice, bindings) != key_kind:
            raise NativeCompileError(
                self.path, node.slice, f"this dict has {key_kind} keys"
            )
        address_slot, found_slot, _key, _state = self.dict_probe(
            self.slot(node.value.id),
            self.dict_key(node.slice, key_kind),
            key_kind,
        )
        present = self.new_label("dict_present")
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(found_slot), IntConstant(0)),
                present + "_missing",
            )
        )
        self.operations.append(Jump(present))
        self.operations.append(Label(present + "_missing"))
        self.raise_exception("KeyError", b"KeyError: key not in native dict\n")
        self.operations.append(Label(present))
        return IntBinary("add", IntLoad(address_slot), IntConstant(16))

    def dict_literal_tag(
        self, node: ast.Dict, bindings: dict[str, IntExpression] | None = None
    ) -> str:
        """The kinds a dict literal builds, read off its first entry.

        An empty literal has nothing to read, so it defaults to integer keys
        and values; `d: dict[str, float] = {}` is how to say otherwise.
        """

        if not node.keys or node.keys[0] is None:
            return self.dict_tag("int", "int")
        key_kind = self.expression_type(node.keys[0], bindings)
        value_kind = self.expression_type(node.values[0], bindings)
        if key_kind not in self.DICT_KEY_KINDS:
            raise NativeCompileError(
                self.path,
                node.keys[0],
                "native dict keys are signed 64-bit integers or runtime strings",
            )
        if value_kind not in self.DICT_VALUE_KINDS:
            raise NativeCompileError(
                self.path,
                node.values[0],
                "native dict values are signed 64-bit integers or floats",
            )
        return self.dict_tag(key_kind, value_kind)

    def annotated_dict_tag(self, annotation: ast.expr) -> str | None:
        """The kinds `dict[K, V]` names, or None if it is not that shape."""

        if (
            not isinstance(annotation, ast.Subscript)
            or not isinstance(annotation.value, ast.Name)
            or annotation.value.id not in {"dict", "Dict"}
            or not isinstance(annotation.slice, ast.Tuple)
            or len(annotation.slice.elts) != 2
        ):
            return None
        names = []
        for item in annotation.slice.elts:
            if not isinstance(item, ast.Name):
                return None
            names.append(item.id)
        key_kind, value_kind = names
        if key_kind not in self.DICT_KEY_KINDS or value_kind not in self.DICT_VALUE_KINDS:
            raise NativeCompileError(
                self.path,
                annotation,
                "a native dict annotation is dict[int|str, int|float]",
            )
        return self.dict_tag(key_kind, value_kind)

    def subscript_assignment(self, target: ast.Subscript, value: ast.expr) -> None:
        if isinstance(target.value, ast.Name) and self.dict_kinds_of(
            target.value.id
        ):
            key_kind, value_kind = self.dict_kinds_of(target.value.id)
            if self.expression_type(target.slice) != key_kind:
                raise NativeCompileError(
                    self.path,
                    target.slice,
                    f"this dict has {key_kind} keys",
                )
            if value_kind == "float":
                if self.expression_type(value) not in {"float", "int"}:
                    raise NativeCompileError(
                        self.path, value, "this dict has float values"
                    )
            elif self.expression_type(value) != "int":
                raise NativeCompileError(
                    self.path, value, "this dict has signed 64-bit integer values"
                )
            self.dict_store(
                self.slot(target.value.id),
                self.dict_key(target.slice, key_kind),
                self.dict_value(value, value_kind),
                target,
                key_kind,
            )
            return
        element_kind = (
            self.list_kind_of(target.value.id)
            if isinstance(target.value, ast.Name)
            else None
        )
        if element_kind == "float":
            if self.expression_type(value) not in {"float", "int"}:
                raise NativeCompileError(
                    self.path, value, "this list holds floats"
                )
            address = self.list_element_address(target)
            # The slot is eight bytes either way, so the double goes in as its
            # bit pattern - the same trick a float dict value uses.
            self.operations.append(
                HeapStore(address, FloatBits(self.float_expression(value)), 8)
            )
            return
        if self.expression_type(value) != "int":
            raise NativeCompileError(
                self.path, value, "this list holds signed 64-bit integers"
            )
        address = self.list_element_address(target)
        self.operations.append(HeapStore(address, self.integer(value), 8))


    # --- runtime dictionaries -----------------------------------------------
    #
    # Layout: [capacity][count] then `capacity` slots of [state][key][value],
    # 24 bytes each. A state of 0 means the slot is empty; collisions are
    # resolved by linear probing, and the table doubles and rehashes once it
    # passes half full.
    #
    # Keys are either signed 64-bit integers or runtime strings, and values are
    # either signed 64-bit integers or IEEE-754 doubles held as their bit
    # pattern - the slot is eight bytes wide either way, so a float costs
    # nothing extra. Which of the four combinations a dict is, is fixed when it
    # is created and checked at build time.
    #
    # The state word does double duty. An integer-keyed entry stores 1 there
    # and finds its home slot from the key; a string-keyed entry stores its
    # hash with the low bit forced on, so the word is still never 0 for a live
    # entry, and finds its home slot from that. Keeping the hash in the entry
    # means a probe can reject a colliding key with one comparison instead of
    # walking its bytes, and it means a rehash never has to hash anything
    # again.

    LIST_HEADER_BYTES = 16
    DICT_SLOT_BYTES = 24
    DICT_HEADER_BYTES = 16
    DICT_KEY_KINDS = {"int", "str"}
    DICT_VALUE_KINDS = {"int", "float"}

    @staticmethod
    def dict_tag(key_kind: str, value_kind: str) -> str:
        return f"dict:{key_kind}:{value_kind}"

    @staticmethod
    def dict_kinds(tag: str | None) -> tuple[str, str] | None:
        """``(key kind, value kind)`` for a dict tag, else None."""

        if not isinstance(tag, str) or not tag.startswith("dict:"):
            return None
        _, key_kind, value_kind = tag.split(":")
        return key_kind, value_kind

    def dict_kinds_of(self, name: str) -> tuple[str, str] | None:
        return self.dict_kinds(self.value_types.get(name))

    def dict_capacity(self, entries: int) -> int:
        """A power-of-two capacity with room to spare, so probing terminates."""

        capacity = 8
        while capacity < (entries + 1) * 4:
            capacity *= 2
        return capacity

    def dict_key(self, node: ast.expr, key_kind: str) -> IntExpression:
        """A key as an i64: the value itself, or a pointer to a string block."""

        if key_kind == "str":
            return self.string_pointer(node)
        return self.integer(node)

    def dict_value(self, node: ast.expr, value_kind: str) -> IntExpression:
        """A value as the i64 actually stored: a float goes in as its bits."""

        if value_kind == "float":
            return FloatBits(self.float_expression(node))
        return self.integer(node)

    def dict_assignment(
        self, name: str, node: ast.expr, key_kind: str, value_kind: str
    ) -> None:
        if not isinstance(node, ast.Dict):
            raise NativeCompileError(
                self.path, node, "native dict variables require a dict literal"
            )
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise NativeCompileError(
                    self.path, node, "native dicts do not support ** unpacking"
                )
            if self.expression_type(key) != key_kind:
                raise NativeCompileError(
                    self.path,
                    key,
                    f"this dict has {key_kind} keys, so every key must be "
                    f"{key_kind}",
                )
            if value_kind == "float":
                if self.expression_type(value) not in {"float", "int"}:
                    raise NativeCompileError(
                        self.path, value, "this dict has float values"
                    )
            elif self.expression_type(value) != "int":
                raise NativeCompileError(
                    self.path, value, "this dict has signed 64-bit integer values"
                )
        capacity = self.dict_capacity(len(node.keys))
        bump = self.ensure_heap()
        self.runtime_names.add(name)
        pointer_slot = self.slot(name)
        size = self.DICT_HEADER_BYTES + capacity * self.DICT_SLOT_BYTES
        self.operations.append(HeapAlloc(pointer_slot, IntConstant(size), bump))
        pointer = IntLoad(pointer_slot)
        self.operations.append(HeapStore(pointer, IntConstant(capacity), 8))
        self.operations.append(
            HeapStore(
                IntBinary("add", pointer, IntConstant(8)), IntConstant(0), 8
            )
        )
        # HeapAlloc hands back fresh arena memory, which the kernel zero-filled,
        # so every state field already reads as empty.
        for key, value in zip(node.keys, node.values):
            self.dict_store(
                pointer_slot,
                self.dict_key(key, key_kind),
                self.dict_value(value, value_kind),
                node,
                key_kind,
            )

    # FNV-1a, which is short enough to emit inline and mixes well enough that
    # linear probing does not degrade on the keys a program actually uses. The
    # offset basis is written signed because the slots are signed 64-bit.
    _FNV_OFFSET = -3750763034362895579
    _FNV_PRIME = 1099511628211

    def emit_string_hash(self, pointer_slot: int) -> int:
        """Hash the string block in ``pointer_slot``; returns a slot holding
        the state word, which is the hash with its low bit forced on so that a
        live entry never looks empty."""

        pointer = IntLoad(pointer_slot)
        length_slot = self.new_temp()
        self.operations.append(Store(length_slot, HeapLoad(pointer, 8)))
        hash_slot = self.new_temp()
        self.operations.append(Store(hash_slot, IntConstant(self._FNV_OFFSET)))
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        start = self.new_label("hash_start")
        done = self.new_label("hash_done")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), done
            )
        )
        byte = HeapLoad(
            IntBinary(
                "add",
                IntBinary("add", pointer, IntConstant(8)),
                IntLoad(index_slot),
            ),
            1,
        )
        self.operations.append(
            Store(
                hash_slot,
                IntBinary(
                    "mul",
                    IntBinary("xor", IntLoad(hash_slot), byte),
                    IntConstant(self._FNV_PRIME),
                ),
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(done))
        state_slot = self.new_temp()
        self.operations.append(
            Store(state_slot, IntBinary("or", IntLoad(hash_slot), IntConstant(1)))
        )
        return state_slot

    def emit_string_equal(self, left: IntExpression, right: IntExpression) -> int:
        """Compare two string blocks byte for byte; returns a 0/1 slot."""

        left_slot = self.new_temp()
        right_slot = self.new_temp()
        self.operations.append(Store(left_slot, left))
        self.operations.append(Store(right_slot, right))
        result_slot = self.new_temp()
        self.operations.append(Store(result_slot, IntConstant(0)))
        done = self.new_label("streq_done")
        length_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntLoad(left_slot), 8))
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "eq",
                    IntLoad(length_slot),
                    HeapLoad(IntLoad(right_slot), 8),
                ),
                done,
            )
        )
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        start = self.new_label("streq_start")
        self.operations.append(Label(start))
        equal = self.new_label("streq_equal")
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), equal
            )
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "eq",
                    HeapLoad(
                        IntBinary(
                            "add",
                            IntBinary("add", IntLoad(left_slot), IntConstant(8)),
                            IntLoad(index_slot),
                        ),
                        1,
                    ),
                    HeapLoad(
                        IntBinary(
                            "add",
                            IntBinary("add", IntLoad(right_slot), IntConstant(8)),
                            IntLoad(index_slot),
                        ),
                        1,
                    ),
                ),
                done,
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(equal))
        self.operations.append(Store(result_slot, IntConstant(1)))
        self.operations.append(Label(done))
        return result_slot

    def dict_probe(
        self, pointer_slot: int, key: IntExpression, key_kind: str
    ) -> tuple[int, int, int, int]:
        """Find where ``key`` lives, or the empty slot it would go in.

        Returns slots holding the entry address, whether the key was found, the
        key itself, and the state word to write for it. The key is pinned in a
        slot because the caller stores it afterwards and the backends re-emit an
        expression tree at every occurrence - re-evaluating the key expression
        there could compute a different key than the one just probed for.
        """

        pointer = IntLoad(pointer_slot)
        key_slot = self.new_temp()
        self.operations.append(Store(key_slot, key))
        if key_kind == "str":
            state_slot = self.emit_string_hash(key_slot)
            home = IntLoad(state_slot)
        else:
            state_slot = self.new_temp()
            self.operations.append(Store(state_slot, IntConstant(1)))
            home = IntLoad(key_slot)
        mask_slot = self.new_temp()
        self.operations.append(
            Store(
                mask_slot,
                IntBinary("sub", HeapLoad(pointer, 8), IntConstant(1)),
            )
        )
        index_slot = self.new_temp()
        self.operations.append(
            Store(index_slot, IntBinary("and", home, IntLoad(mask_slot)))
        )
        address_slot = self.new_temp()
        found_slot = self.new_temp()
        self.operations.append(Store(found_slot, IntConstant(0)))
        start = self.new_label("probe_start")
        done = self.new_label("probe_done")
        step = self.new_label("probe_next")
        self.operations.append(Label(start))
        self.operations.append(
            Store(
                address_slot,
                IntBinary(
                    "add",
                    pointer,
                    IntBinary(
                        "add",
                        IntConstant(self.DICT_HEADER_BYTES),
                        IntBinary(
                            "mul",
                            IntLoad(index_slot),
                            IntConstant(self.DICT_SLOT_BYTES),
                        ),
                    ),
                ),
            )
        )
        state = HeapLoad(IntLoad(address_slot), 8)
        # An empty slot ends the probe: this key is not in the table, and this
        # is where it belongs.
        self.operations.append(
            JumpIfFalse(IntCompare("ne", state, IntConstant(0)), done)
        )
        self.operations.append(
            JumpIfFalse(IntCompare("eq", state, IntLoad(state_slot)), step)
        )
        if key_kind == "str":
            # Equal hashes are not equal strings; check the bytes.
            equal_slot = self.emit_string_equal(
                IntLoad(key_slot),
                HeapLoad(IntBinary("add", IntLoad(address_slot), IntConstant(8)), 8),
            )
            self.operations.append(
                JumpIfFalse(
                    IntCompare("ne", IntLoad(equal_slot), IntConstant(0)), step
                )
            )
        else:
            self.operations.append(
                JumpIfFalse(
                    IntCompare(
                        "eq",
                        HeapLoad(
                            IntBinary(
                                "add", IntLoad(address_slot), IntConstant(8)
                            ),
                            8,
                        ),
                        IntLoad(key_slot),
                    ),
                    step,
                )
            )
        self.operations.append(Store(found_slot, IntConstant(1)))
        self.operations.append(Jump(done))
        self.operations.append(Label(step))
        self.operations.append(
            Store(
                index_slot,
                IntBinary(
                    "and",
                    IntBinary("add", IntLoad(index_slot), IntConstant(1)),
                    IntLoad(mask_slot),
                ),
            )
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(done))
        return address_slot, found_slot, key_slot, state_slot

    def dict_grow(self, pointer_slot: int, key_kind: str) -> None:
        """Double the table and rehash into it.

        A hash table cannot simply be extended: every entry's home slot depends
        on the capacity, so growth means allocating a new table and probing each
        live entry into it. The old table is left in the arena, which never
        reclaims, and that is the documented cost of an arena.
        """

        bump = self.ensure_heap()
        old_slot = self.new_temp()
        self.operations.append(Store(old_slot, IntLoad(pointer_slot)))
        old = IntLoad(old_slot)
        old_capacity_slot = self.new_temp()
        self.operations.append(Store(old_capacity_slot, HeapLoad(old, 8)))
        new_capacity_slot = self.new_temp()
        self.operations.append(
            Store(
                new_capacity_slot,
                IntBinary("mul", IntLoad(old_capacity_slot), IntConstant(2)),
            )
        )
        new_slot = self.new_temp()
        self.operations.append(
            HeapAlloc(
                new_slot,
                IntBinary(
                    "add",
                    IntConstant(self.DICT_HEADER_BYTES),
                    IntBinary(
                        "mul",
                        IntLoad(new_capacity_slot),
                        IntConstant(self.DICT_SLOT_BYTES),
                    ),
                ),
                bump,
            )
        )
        new = IntLoad(new_slot)
        self.operations.append(HeapStore(new, IntLoad(new_capacity_slot), 8))
        self.operations.append(
            HeapStore(
                IntBinary("add", new, IntConstant(8)),
                HeapLoad(IntBinary("add", old, IntConstant(8)), 8),
                8,
            )
        )
        mask_slot = self.new_temp()
        self.operations.append(
            Store(
                mask_slot,
                IntBinary("sub", IntLoad(new_capacity_slot), IntConstant(1)),
            )
        )

        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        scan = self.new_label("dict_rehash")
        after = self.new_label("dict_rehash_done")
        step = self.new_label("dict_rehash_next")
        self.operations.append(Label(scan))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(old_capacity_slot)),
                after,
            )
        )
        entry_slot = self.new_temp()
        self.operations.append(
            Store(
                entry_slot,
                IntBinary(
                    "add",
                    old,
                    IntBinary(
                        "add",
                        IntConstant(self.DICT_HEADER_BYTES),
                        IntBinary(
                            "mul",
                            IntLoad(index_slot),
                            IntConstant(self.DICT_SLOT_BYTES),
                        ),
                    ),
                ),
            )
        )
        state_slot = self.new_temp()
        self.operations.append(
            Store(state_slot, HeapLoad(IntLoad(entry_slot), 8))
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(state_slot), IntConstant(0)), step
            )
        )
        key_slot = self.new_temp()
        self.operations.append(
            Store(
                key_slot,
                HeapLoad(
                    IntBinary("add", IntLoad(entry_slot), IntConstant(8)), 8
                ),
            )
        )
        # The stored state IS the string hash, so a rehash never re-hashes.
        home = IntLoad(state_slot) if key_kind == "str" else IntLoad(key_slot)
        target_slot = self.new_temp()
        self.operations.append(
            Store(target_slot, IntBinary("and", home, IntLoad(mask_slot)))
        )
        place = self.new_label("dict_place")
        placed = self.new_label("dict_placed")
        address_slot = self.new_temp()
        self.operations.append(Label(place))
        self.operations.append(
            Store(
                address_slot,
                IntBinary(
                    "add",
                    new,
                    IntBinary(
                        "add",
                        IntConstant(self.DICT_HEADER_BYTES),
                        IntBinary(
                            "mul",
                            IntLoad(target_slot),
                            IntConstant(self.DICT_SLOT_BYTES),
                        ),
                    ),
                ),
            )
        )
        # Keys were unique in the old table, so the first empty slot is ours.
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", HeapLoad(IntLoad(address_slot), 8), IntConstant(0)),
                placed,
            )
        )
        self.operations.append(
            Store(
                target_slot,
                IntBinary(
                    "and",
                    IntBinary("add", IntLoad(target_slot), IntConstant(1)),
                    IntLoad(mask_slot),
                ),
            )
        )
        self.operations.append(Jump(place))
        self.operations.append(Label(placed))
        self.operations.append(
            HeapStore(IntLoad(address_slot), IntLoad(state_slot), 8)
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", IntLoad(address_slot), IntConstant(8)),
                IntLoad(key_slot),
                8,
            )
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", IntLoad(address_slot), IntConstant(16)),
                HeapLoad(
                    IntBinary("add", IntLoad(entry_slot), IntConstant(16)), 8
                ),
                8,
            )
        )
        self.operations.append(Label(step))
        self.operations.append(
            Store(
                index_slot,
                IntBinary("add", IntLoad(index_slot), IntConstant(1)),
            )
        )
        self.operations.append(Jump(scan))
        self.operations.append(Label(after))
        self.operations.append(Store(pointer_slot, IntLoad(new_slot)))

    def dict_store(
        self,
        pointer_slot: int,
        key: IntExpression,
        value: IntExpression,
        node: ast.AST,
        key_kind: str,
    ) -> None:
        value_slot = self.new_temp()
        self.operations.append(Store(value_slot, value))
        address_slot, found_slot, key_slot, state_slot = self.dict_probe(
            pointer_slot, key, key_kind
        )
        pointer = IntLoad(pointer_slot)
        existing = self.new_label("dict_existing")
        end = self.new_label("dict_store_end")
        self.operations.append(
            JumpIfFalse(
                IntCompare("eq", IntLoad(found_slot), IntConstant(0)), existing
            )
        )
        # A new entry: grow before filling past half, so probing stays short.
        count = HeapLoad(IntBinary("add", pointer, IntConstant(8)), 8)
        capacity = HeapLoad(pointer, 8)
        room = self.new_label("dict_has_room")
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "lt",
                    IntBinary("mul", count, IntConstant(2)),
                    capacity,
                ),
                room + "_full",
            )
        )
        self.operations.append(Jump(room))
        self.operations.append(Label(room + "_full"))
        self.dict_grow(pointer_slot, key_kind)
        # Growth moved the table, so the address the first probe produced points
        # into the abandoned one. Probe again for where this key belongs now.
        regrown_address, _found, _key, _state = self.dict_probe(
            pointer_slot, IntLoad(key_slot), key_kind
        )
        self.operations.append(Store(address_slot, IntLoad(regrown_address)))
        self.operations.append(Label(room))
        self.operations.append(
            HeapStore(IntLoad(address_slot), IntLoad(state_slot), 8)
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", IntLoad(address_slot), IntConstant(8)),
                IntLoad(key_slot),
                8,
            )
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", pointer, IntConstant(8)),
                IntBinary(
                    "add",
                    HeapLoad(IntBinary("add", pointer, IntConstant(8)), 8),
                    IntConstant(1),
                ),
                8,
            )
        )
        self.operations.append(Label(existing))
        self.operations.append(
            HeapStore(
                IntBinary("add", IntLoad(address_slot), IntConstant(16)),
                IntLoad(value_slot),
                8,
            )
        )
        self.operations.append(Label(end))

    # --- runtime strings ----------------------------------------------------

    def string_assignment(self, name: str, node: ast.expr) -> None:
        pointer = self.string_pointer(node)
        self.runtime_names.add(name)
        self.operations.append(Store(self.slot(name), pointer))

    def joined_string(self, node: ast.JoinedStr) -> IntExpression:
        """Lower an f-string whose pieces are not all known at build time.

        Each piece is rendered the way `str()` would render it and concatenated
        on the spot. On the spot matters: float rendering hands back a pointer
        into scratch that the next float would overwrite, and concatenation
        copies, so the copy has to happen before the next piece is rendered.
        """

        result = self.materialize_string_constant(b"")
        for piece in node.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                result = self.emit_concat(
                    result, self.materialize_string_constant(piece.value.encode())
                )
                continue
            if not isinstance(piece, ast.FormattedValue):
                raise NativeCompileError(
                    self.path, node, "unsupported native f-string component"
                )
            if piece.conversion != -1:
                raise NativeCompileError(
                    self.path,
                    node,
                    "native f-strings do not support !r, !s, or !a conversions",
                )
            if piece.format_spec is not None:
                raise NativeCompileError(
                    self.path,
                    node,
                    "native f-strings do not support format specifiers yet; "
                    "a width or precision needs a formatter beyond str()",
                )
            result = self.emit_concat(result, self.render_as_string(piece.value))
        return result

    def render_as_string(self, node: ast.expr) -> IntExpression:
        """A pointer to the text `str()` would produce for this expression."""

        kind = self.expression_type(node)
        if kind == "str":
            return self.string_pointer(node)
        if kind == "float":
            return self.emit_float_to_string(self.float_expression(node))
        if kind == "int":
            if self.renders_as_bool(node):
                return self.emit_bool_to_string(self.integer(node))
            return self.emit_int_to_string(self.integer(node))
        raise NativeCompileError(
            self.path, node, f"a native f-string cannot render a {kind} yet"
        )

    def string_pointer(self, node: ast.expr) -> IntExpression:
        """Emit any needed heap work and return an i64 pointer to a string block.

        A string block is ``[i64 length][raw utf-8 bytes]``. The returned
        expression is a stable load of the pointer, safe to reference twice.
        """

        if self.expression_type(node) != "str":
            raise NativeCompileError(
                self.path, node, "expression is not in the native string subset"
            )
        if isinstance(node, ast.Name) and node.id in self.string_bindings:
            return self.string_bindings[node.id]
        if isinstance(node, ast.Name) and self.value_types.get(node.id) == "str":
            return IntLoad(self.slots[node.id])
        try:
            folded = self.constant(node)
        except NativeCompileError:
            folded = None
        if isinstance(folded, str):
            return self.materialize_string_constant(folded.encode("utf-8"))
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            return self.emit_string_slice(
                self.string_pointer(node.value), node.slice
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and self.expression_function_kind(node) == "str"
        ):
            function = self.functions[node.func.id]
            argument_kinds: list[str] = []
            arguments = self.bind_native_arguments(
                node.func.id, function, node, {}, (), kinds=argument_kinds
            )
            previous_path, previous_values = self.path, self.values
            previous_functions = self.functions
            previous_strings = self.string_bindings
            self.path, self.values = function.path, function.values
            self.functions = function.functions
            self.string_bindings = {
                **previous_strings,
                **{
                    parameter: value
                    for parameter, value, kind in zip(
                        function.parameters, arguments, argument_kinds
                    )
                    if kind == "str"
                },
            }
            try:
                return self.string_pointer(function.expression)
            finally:
                self.path, self.values = previous_path, previous_values
                self.functions = previous_functions
                self.string_bindings = previous_strings
        if isinstance(node, ast.JoinedStr):
            return self.joined_string(node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.string_pointer(node.left)
            right = self.string_pointer(node.right)
            return self.emit_concat(left, right)
        raise NativeCompileError(
            self.path, node, "expression is not in the native string subset"
        )

    def emit_code_point_count(self, pointer: IntExpression) -> IntExpression:
        """How many code points a UTF-8 string block holds.

        The header counts bytes, which is what a write needs, but `len()` in
        Python counts code points. In UTF-8 every byte of a continuation is
        `10xxxxxx`, so the code points are exactly the bytes that are not - one
        pass, no decoding.
        """

        pointer_slot = self.new_temp()
        self.operations.append(Store(pointer_slot, pointer))
        base = IntLoad(pointer_slot)
        length_slot = self.new_temp()
        self.operations.append(Store(length_slot, HeapLoad(base, 8)))
        count_slot = self.new_temp()
        self.operations.append(Store(count_slot, IntConstant(0)))
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        start = self.new_label("count_start")
        done = self.new_label("count_done")
        step = self.new_label("count_next")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), done
            )
        )
        byte = HeapLoad(
            IntBinary(
                "add",
                IntBinary("add", base, IntConstant(8)),
                IntLoad(index_slot),
            ),
            1,
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "ne",
                    IntBinary("and", byte, IntConstant(0xC0)),
                    IntConstant(0x80),
                ),
                step,
            )
        )
        self.operations.append(
            Store(count_slot, IntBinary("add", IntLoad(count_slot), IntConstant(1)))
        )
        self.operations.append(Label(step))
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(done))
        return IntLoad(count_slot)

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

    def iterable_element_kind(self, node: ast.expr) -> str | None:
        """The kind each item of ``node`` has, if it is something to iterate."""

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and node.func.id not in self.functions
        ):
            return "int"
        try:
            kind = self.expression_type(node)
        except NativeCompileError:
            return None
        return self.list_kind(kind)

    def emit_list_iteration(
        self, node: ast.expr, target: str
    ) -> tuple[int, int, str]:
        """Set up a walk over a runtime list.

        Returns the index slot, the pointer slot, and the element kind. The
        caller drives the loop; this only prepares what the loop reads.
        """

        element_kind = self.list_kind(self.expression_type(node))
        assert element_kind is not None
        pointer_slot = self.new_temp()
        self.operations.append(Store(pointer_slot, self.list_pointer(node)))
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        return index_slot, pointer_slot, element_kind

    def bind_list_element(
        self, target: str, index_slot: int, pointer_slot: int, element_kind: str
    ) -> None:
        address = IntBinary(
            "add",
            IntBinary(
                "add", IntLoad(pointer_slot), IntConstant(self.LIST_HEADER_BYTES)
            ),
            IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
        )
        self.values.pop(target, None)
        self.runtime_names.add(target)
        self.boolean_names.discard(target)
        if element_kind == "float":
            self.operations.append(
                FloatStore(self.slot(target), BitsFloat(HeapLoad(address, 8)))
            )
        else:
            self.operations.append(Store(self.slot(target), HeapLoad(address, 8)))
        self.value_types[target] = element_kind

    def enumerate_source(self, node: ast.expr):
        """The list and start `enumerate(...)` names, or None."""

        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "enumerate"
            or node.func.id in self.functions
            or node.keywords
            or not 1 <= len(node.args) <= 2
        ):
            return None
        if self.list_kind(self.expression_type(node.args[0])) is None:
            return None
        return node.args[0], node.args[1] if len(node.args) == 2 else None

    def for_over_enumerate(self, node: ast.For) -> None:
        """`for i, v in enumerate(xs):` - the index and the item together."""

        source, start = self.enumerate_source(node.iter)
        if (
            not isinstance(node.target, (ast.Tuple, ast.List))
            or len(node.target.elts) != 2
            or not all(isinstance(item, ast.Name) for item in node.target.elts)
        ):
            raise NativeCompileError(
                self.path,
                node,
                "a native enumerate() loop binds two names, the index and the "
                "item",
            )
        counter_name = node.target.elts[0].id
        item_name = node.target.elts[1].id
        was_bound = {
            name: name in self.bound_names for name in (counter_name, item_name)
        }
        index_slot, pointer_slot, element_kind = self.emit_list_iteration(
            source, item_name
        )
        offset_slot = self.new_temp()
        self.operations.append(
            Store(
                offset_slot,
                self.integer(start) if start is not None else IntConstant(0),
            )
        )
        length_slot = self.new_temp()
        self.operations.append(
            Store(
                length_slot,
                HeapLoad(IntBinary("add", IntLoad(pointer_slot), IntConstant(8)), 8),
            )
        )
        start_label = self.new_label("for_enum")
        continue_label = self.new_label("for_enum_continue")
        end_label = self.new_label("for_enum_end")
        self.operations.append(Label(start_label))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)),
                end_label,
            )
        )
        self.values.pop(counter_name, None)
        self.runtime_names.add(counter_name)
        self.boolean_names.discard(counter_name)
        self.operations.append(
            Store(
                self.slot(counter_name),
                IntBinary("add", IntLoad(index_slot), IntLoad(offset_slot)),
            )
        )
        self.value_types[counter_name] = "int"
        self.bind_list_element(item_name, index_slot, pointer_slot, element_kind)
        self.break_targets.append(end_label)
        self.continue_targets.append(continue_label)
        for statement in node.body:
            self.statement(statement)
        self.continue_targets.pop()
        self.break_targets.pop()
        self.operations.append(Label(continue_label))
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start_label))
        self.operations.append(Label(end_label))
        for name in (counter_name, item_name):
            if was_bound[name]:
                self.bound_names.add(name)
            else:
                # The list may be empty, and then Python binds neither name.
                self.possibly_unbound.add(name)

    def for_over_list(self, node: ast.For) -> None:
        """`for name in <list>:` - walk by index, since there is no iterator."""

        assert isinstance(node.target, ast.Name)
        name = node.target.id
        was_bound = name in self.bound_names
        index_slot, pointer_slot, element_kind = self.emit_list_iteration(
            node.iter, name
        )
        length_slot = self.new_temp()
        self.operations.append(
            Store(
                length_slot,
                HeapLoad(
                    IntBinary("add", IntLoad(pointer_slot), IntConstant(8)), 8
                ),
            )
        )
        start = self.new_label("for_list")
        continue_label = self.new_label("for_list_continue")
        end = self.new_label("for_list_end")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), end
            )
        )
        self.bind_list_element(name, index_slot, pointer_slot, element_kind)
        self.break_targets.append(end)
        self.continue_targets.append(continue_label)
        for statement in node.body:
            self.statement(statement)
        self.continue_targets.pop()
        self.break_targets.pop()
        self.operations.append(Label(continue_label))
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
        if was_bound:
            self.bound_names.add(name)
        else:
            # The list may be empty, and then Python leaves the name unbound.
            self.possibly_unbound.add(name)

    def comprehension_parts(self, node: ast.ListComp):
        """The comprehension's pieces with its target renamed out of the way.

        Python 3 gives a comprehension its own scope, so `[k for k in ...]`
        must not disturb an outer `k` - not its value and not its slot. The
        name is private and derived from this node, so asking for the element
        kind and emitting the loop agree on it.
        """

        generator = node.generators[0]
        target = generator.target.id
        private = f"<comp-{id(node):x}:{target}>"

        class _Rename(ast.NodeTransformer):
            def visit_Name(self, inner: ast.Name):
                if inner.id == target:
                    return ast.copy_location(
                        ast.Name(id=private, ctx=inner.ctx), inner
                    )
                return inner

        element = _Rename().visit(copy.deepcopy(node.elt))
        conditions = [
            _Rename().visit(copy.deepcopy(test)) for test in generator.ifs
        ]
        ast.fix_missing_locations(
            ast.Module(body=[ast.Expr(value=element)], type_ignores=[])
        )
        for test in conditions:
            ast.fix_missing_locations(
                ast.Module(body=[ast.Expr(value=test)], type_ignores=[])
            )
        return private, element, conditions, generator.iter

    def comprehension_element_kind(self, node: ast.ListComp) -> str:
        """The kind `[expr for t in it]` produces, without emitting anything."""

        if len(node.generators) != 1 or not isinstance(
            node.generators[0].target, ast.Name
        ):
            raise NativeCompileError(
                self.path,
                node,
                "a native list comprehension has one `for` binding one name",
            )
        target, element, _tests, iterable = self.comprehension_parts(node)
        item_kind = self.iterable_element_kind(iterable)
        previous = self.value_types.get(target)
        previously_bound = target in self.bound_names
        self.value_types[target] = item_kind or "int"
        self.bound_names.add(target)
        try:
            kind = self.expression_type(element)
        finally:
            if previous is None:
                self.value_types.pop(target, None)
            else:
                self.value_types[target] = previous
            if not previously_bound:
                self.bound_names.discard(target)
        if kind not in {"int", "float"}:
            raise NativeCompileError(
                self.path,
                node,
                "a native list comprehension builds integers or floats",
            )
        return kind

    def list_comprehension(self, node: ast.ListComp) -> IntExpression:
        """`[expr for t in it]`, optionally with `if`.

        The result is sized from the source, not from how many items survive
        the condition, and the real count is written into the header at the
        end. Over-reserving costs arena space that is never reclaimed anyway;
        counting first would mean running the source twice.
        """

        if len(node.generators) != 1:
            raise NativeCompileError(
                self.path, node, "a native list comprehension has one `for`"
            )
        generator = node.generators[0]
        if generator.is_async or not isinstance(generator.target, ast.Name):
            raise NativeCompileError(
                self.path,
                node,
                "a native list comprehension binds a single name and is not async",
            )
        if len(generator.ifs) > 1:
            raise NativeCompileError(
                self.path, node, "a native list comprehension has at most one `if`"
            )
        element_kind = self.comprehension_element_kind(node)
        bump = self.ensure_heap()
        target, element, conditions, _iterable = self.comprehension_parts(node)

        over_range = (
            isinstance(generator.iter, ast.Call)
            and isinstance(generator.iter.func, ast.Name)
            and generator.iter.func.id == "range"
            and generator.iter.func.id not in self.functions
        )
        index_slot = self.new_temp()
        limit_slot = self.new_temp()
        pointer_slot = None
        if over_range:
            arguments = generator.iter.args
            if not 1 <= len(arguments) <= 2 or generator.iter.keywords:
                raise NativeCompileError(
                    self.path,
                    node,
                    "a native comprehension over range takes one or two "
                    "arguments and steps by one",
                )
            if len(arguments) == 1:
                start = IntConstant(0)
                stop = self.materialize_int(self.integer(arguments[0]))
            else:
                start = self.materialize_int(self.integer(arguments[0]))
                stop = self.materialize_int(self.integer(arguments[1]))
            self.operations.append(Store(index_slot, start))
            self.operations.append(Store(limit_slot, stop))
            reserve = IntBinary("sub", IntLoad(limit_slot), IntLoad(index_slot))
        else:
            item_kind = self.list_kind(self.expression_type(generator.iter))
            if item_kind is None:
                raise NativeCompileError(
                    self.path,
                    node,
                    "a native comprehension iterates a range or a runtime list",
                )
            pointer_slot = self.new_temp()
            self.operations.append(
                Store(pointer_slot, self.list_pointer(generator.iter))
            )
            self.operations.append(Store(index_slot, IntConstant(0)))
            self.operations.append(
                Store(
                    limit_slot,
                    HeapLoad(
                        IntBinary("add", IntLoad(pointer_slot), IntConstant(8)), 8
                    ),
                )
            )
            reserve = IntLoad(limit_slot)
        # An empty range gives a negative span; reserve nothing rather than
        # asking the arena for a negative number of bytes.
        reserve_slot = self.new_temp()
        self.operations.append(
            Store(
                reserve_slot,
                self.select_integer(
                    IntCompare("gt", reserve, IntConstant(0)),
                    reserve,
                    IntConstant(0),
                ),
            )
        )
        result_slot = self.new_temp()
        self.operations.append(
            HeapAlloc(
                result_slot,
                IntBinary(
                    "add",
                    IntConstant(self.LIST_HEADER_BYTES),
                    IntBinary("mul", IntLoad(reserve_slot), IntConstant(8)),
                ),
                bump,
            )
        )
        result = IntLoad(result_slot)
        self.operations.append(HeapStore(result, IntLoad(reserve_slot), 8))
        count_slot = self.new_temp()
        self.operations.append(Store(count_slot, IntConstant(0)))

        start_label = self.new_label("comp")
        end_label = self.new_label("comp_end")
        step_label = self.new_label("comp_next")
        self.operations.append(Label(start_label))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(limit_slot)),
                end_label,
            )
        )
        if over_range:
            self.values.pop(target, None)
            self.runtime_names.add(target)
            self.boolean_names.discard(target)
            self.operations.append(Store(self.slot(target), IntLoad(index_slot)))
            self.value_types[target] = "int"
        else:
            self.bind_list_element(
                target,
                index_slot,
                pointer_slot,
                self.list_kind(self.expression_type(generator.iter)),
            )
        if conditions:
            self.operations.append(
                JumpIfFalse(self.integer(conditions[0]), step_label)
            )
        address = IntBinary(
            "add",
            IntBinary("add", result, IntConstant(self.LIST_HEADER_BYTES)),
            IntBinary("mul", IntLoad(count_slot), IntConstant(8)),
        )
        stored = (
            FloatBits(self.float_expression(element))
            if element_kind == "float"
            else self.integer(element)
        )
        self.operations.append(HeapStore(address, stored, 8))
        self.operations.append(
            Store(count_slot, IntBinary("add", IntLoad(count_slot), IntConstant(1)))
        )
        self.operations.append(Label(step_label))
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start_label))
        self.operations.append(Label(end_label))
        self.operations.append(
            HeapStore(
                IntBinary("add", result, IntConstant(8)), IntLoad(count_slot), 8
            )
        )
        return result

    def for_statement(self, node: ast.For) -> None:
        if (
            not node.orelse
            and isinstance(node.target, ast.Name)
            and self.list_kind(self.expression_type(node.iter)) is not None
        ):
            self.for_over_list(node)
            return
        if not node.orelse and self.enumerate_source(node.iter) is not None:
            self.for_over_enumerate(node)
            return
        if (
            node.orelse
            or not isinstance(node.target, ast.Name)
            or not isinstance(node.iter, ast.Call)
            or not isinstance(node.iter.func, ast.Name)
            or node.iter.func.id != "range"
            or node.iter.keywords
            or not 1 <= len(node.iter.args) <= 3
        ):
            hint = ""
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "enumerate"
            ):
                hint = "; native enumerate() walks a runtime list"
            elif isinstance(node.iter, ast.Name) and self.value_types.get(
                node.iter.id
            ) in {"str", "dict:int:int"}:
                hint = "; only ranges and runtime lists are iterable here"
            raise NativeCompileError(
                self.path,
                node,
                "native for supports NAME in range(1-3 arguments), NAME in a "
                "runtime list, or two names in enumerate(list)" + hint,
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
        if self.list_method_call(node):
            return
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
                # Everything is known now, so the whole line is one constant.
                text = " ".join(str(value) for value in values) + "\n"
                self.operations.append(Write(text.encode("utf-8")))
                return
            # Otherwise write the arguments one at a time, with the separators
            # print() would insert. Several writes rather than one buffer: the
            # process is single-threaded, so the bytes land in order.
            for index, argument in enumerate(node.args):
                if index:
                    self.operations.append(Write(b" "))
                self.emit_print_argument(argument)
            self.operations.append(Write(b"\n"))
            return
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
            argument_kinds: list[str] = []
            arguments = self.bind_native_arguments(
                node.func.id,
                function,
                node,
                {},
                (),
                kinds=argument_kinds,
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
                argument_kinds=tuple(argument_kinds),
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

    # --- exceptions ---------------------------------------------------------
    #
    # There is no runtime type object and no traceback. A raise records which
    # class it names as a small integer and jumps to the innermost enclosing
    # handler; the handler decides whether it matches by comparing that integer
    # against the ids of the classes the clause catches, a set computed at
    # build time from the static hierarchy. When no handler matches anywhere,
    # the class name goes to standard error and the process exits 1, which is
    # what CPython's exit status would be.

    def exception_slots(self) -> tuple[int, int]:
        if self.exception_slot is None:
            self.exception_slot = self.new_temp()
            self.exception_value_slot = self.new_temp()
        assert self.exception_value_slot is not None
        return self.exception_slot, self.exception_value_slot

    def exception_id(self, name: str) -> int:
        if name not in self.exception_ids:
            self.exception_ids[name] = len(self.exception_ids) + 1
        return self.exception_ids[name]

    def emit_uncaught(self) -> None:
        """Report whichever exception is live and exit, CPython's way."""

        identifier_slot, value_slot = self.exception_slots()
        for name, identifier in self.exception_ids.items():
            skip = self.new_label("not_" + name.lower())
            self.operations.append(
                JumpIfFalse(
                    IntCompare(
                        "eq", IntLoad(identifier_slot), IntConstant(identifier)
                    ),
                    skip,
                )
            )
            if name == "SystemExit":
                self.operations.append(ExitValue(IntLoad(value_slot)))
            else:
                self.operations.append(Write(name.encode("ascii") + b"\n", 2))
                self.operations.append(Exit(1))
            self.operations.append(Label(skip))
        self.operations.append(Exit(1))

    def raise_exception(
        self,
        name: str,
        message: bytes,
        value: IntExpression | None = None,
    ) -> None:
        """Raise ``name``, either into the innermost handler or out of the program."""

        if not self.handler_stack:
            # Nothing can catch it, so report it where it happens: the message
            # is known here and would be lost by going through the dispatch.
            if name == "SystemExit":
                self.operations.append(
                    ExitValue(value if value is not None else IntConstant(0))
                )
                return
            self.operations.append(Write(message, 2))
            self.operations.append(Exit(1))
            return
        identifier_slot, value_slot = self.exception_slots()
        self.operations.append(
            Store(identifier_slot, IntConstant(self.exception_id(name)))
        )
        self.operations.append(
            Store(value_slot, value if value is not None else IntConstant(0))
        )
        self.operations.append(Jump(self.handler_stack[-1]))

    def propagate(self) -> None:
        """Send the live exception outward, to a handler or out of the program."""

        if self.handler_stack:
            self.operations.append(Jump(self.handler_stack[-1]))
        else:
            self.emit_uncaught()

    def raise_statement(self, node: ast.Raise) -> None:
        if node.cause is not None:
            raise NativeCompileError(
                self.path, node, "native raise does not support 'from'"
            )
        if node.exc is None:
            if not self.exception_ids:
                raise NativeCompileError(
                    self.path, node, "bare raise is outside an except clause"
                )
            self.propagate()
            return
        exception = node.exc
        if isinstance(exception, ast.Call):
            if exception.keywords:
                raise NativeCompileError(
                    self.path, node, "native raise does not take keyword arguments"
                )
            callee, arguments = exception.func, exception.args
        elif isinstance(exception, ast.Name):
            callee, arguments = exception, []
        else:
            raise NativeCompileError(
                self.path, node, "native raise expects an exception class"
            )
        if (
            isinstance(exception, ast.Call)
            and self.is_exit_call(exception)
            and not self.handler_stack
        ):
            # Outside any try, the existing lowering folds a constant status.
            self.system_exit(exception, node)
            return
        if not isinstance(callee, ast.Name) or callee.id not in _EXCEPTION_BASES:
            name = getattr(callee, "id", None) or ast.dump(callee)
            raise NativeCompileError(
                self.path,
                node,
                f"native raise supports the builtin exception classes; {name} is "
                "not one of them",
            )
        name = callee.id
        if len(arguments) > 1:
            raise NativeCompileError(
                self.path, node, "native exceptions take at most one argument"
            )
        if name == "SystemExit":
            status = IntConstant(0)
            if arguments:
                status = self.integer(arguments[0])
            self.raise_exception(name, b"SystemExit\n", status)
            return
        detail = ""
        if arguments:
            try:
                text = self.constant(arguments[0])
            except NativeCompileError:
                text = None
            if isinstance(text, str):
                detail = ": " + text
            elif isinstance(text, (int, bool)):
                detail = ": " + str(int(text))
        message = (name + detail).encode("utf-8", "replace") + b"\n"
        self.raise_exception(name, message)

    def clause_matches(self, clause: ast.ExceptHandler) -> IntExpression | None:
        """The test for ``clause``, or None when it catches everything."""

        if clause.type is None:
            return None
        if isinstance(clause.type, ast.Tuple):
            caught = list(clause.type.elts)
        else:
            caught = [clause.type]
        names: list[str] = []
        for item in caught:
            if not isinstance(item, ast.Name) or item.id not in _EXCEPTION_BASES:
                raise NativeCompileError(
                    self.path,
                    clause,
                    "native except clauses name builtin exception classes",
                )
            names.append(item.id)
        if "BaseException" in names:
            return None
        # The clause matches a raise when the raised class inherits from one of
        # the named ones. Every raise that can reach here has been lowered
        # already, so the set of ids is complete.
        matching = [
            identifier
            for raised, identifier in self.exception_ids.items()
            if any(name in exception_ancestry(raised) for name in names)
        ]
        if not matching:
            return IntConstant(0)  # nothing reaching here can match
        identifier_slot, _value = self.exception_slots()
        test: IntExpression | None = None
        for identifier in matching:
            comparison = IntCompare(
                "eq", IntLoad(identifier_slot), IntConstant(identifier)
            )
            test = comparison if test is None else IntBinary("or", test, comparison)
        assert test is not None
        return test

    def jump_escapes_cleanup(self, index: int, targets: list) -> bool:
        """Whether a jump of this kind would leave a cleanup scope unrun."""

        return any(scope[index] >= len(targets) for scope in self.finally_scopes)

    def open_cleanup_scope(self) -> None:
        self.finally_scopes.append(
            (
                len(self.return_targets),
                len(self.break_targets),
                len(self.continue_targets),
            )
        )

    def with_statement(self, node: ast.With) -> None:
        """Run a body with a native object's `__enter__`/`__exit__` around it.

        There are no context-manager protocols to look up at run time here, so
        this resolves the two methods at build time and inlines them. `__exit__`
        runs on the way out whether the body finished or raised, which is the
        same problem `finally` solves and is emitted the same way: once per path
        out, because there is no return address to come back on.
        """

        if len(node.items) > 1:
            # `with a, b:` is `with a: with b:`; rewrite rather than duplicate.
            inner = ast.copy_location(
                ast.With(items=node.items[1:], body=node.body, type_comment=None),
                node,
            )
            ast.fix_missing_locations(inner)
            node = ast.copy_location(
                ast.With(items=node.items[:1], body=[inner], type_comment=None),
                node,
            )
        item = node.items[0]
        manager = item.context_expr
        native_class = self.resolve_object_class(manager)
        if native_class is None:
            raise NativeCompileError(
                self.path,
                node,
                "a native `with` needs a native object with __enter__ and "
                "__exit__; there is no run-time protocol lookup here",
            )
        if isinstance(manager, ast.Name):
            manager_name = manager.id
        else:
            manager_name = f"<with-{self.new_label('object')}>"
            self.assignment(manager_name, manager)
        enter = native_class.methods.get("__enter__")
        exit_method = native_class.methods.get("__exit__")
        if enter is None or exit_method is None:
            missing = "__enter__" if enter is None else "__exit__"
            raise NativeCompileError(
                self.path,
                node,
                f"{native_class.name!r} has no native {missing}()",
            )
        if exit_method.returns_value:
            raise NativeCompileError(
                self.path,
                node,
                "a native __exit__ cannot return a value: a real one returning "
                "true suppresses the exception, and there is no exception "
                "object here to decide about",
            )
        # Keep the signature Python requires, so the same source still runs
        # under CPython, but nothing truthful can be passed for the exception
        # triple - there are no exception objects here. Zeros go in, and a body
        # that reads them is refused rather than shown the wrong thing.
        if len(exit_method.parameters) != 4:
            raise NativeCompileError(
                self.path,
                node,
                f"{native_class.name}.__exit__() must take self and the three "
                "exception parameters, as CPython requires",
            )
        reported = set(exit_method.parameters[1:])
        for inner in ast.walk(
            ast.Module(body=list(exit_method.body), type_ignores=[])
        ):
            if isinstance(inner, ast.Name) and inner.id in reported:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"{native_class.name}.__exit__() reads {inner.id!r}, but "
                    "there is no exception object to pass: native exceptions "
                    "are an integer class id, not a value",
                )
        if len(enter.parameters) != 1:
            raise NativeCompileError(
                self.path, node, f"{native_class.name}.__enter__() takes only self"
            )
        instance = IntLoad(self.slot(manager_name))

        def call(name: str, method: NativeFunction, count: int = 0):
            invocation = ast.copy_location(
                ast.Call(
                    func=ast.copy_location(
                        ast.Attribute(
                            value=ast.copy_location(
                                ast.Name(id=manager_name, ctx=ast.Load()), node
                            ),
                            attr=name,
                            ctx=ast.Load(),
                        ),
                        node,
                    ),
                    args=[
                        ast.copy_location(ast.Constant(value=0), node)
                        for _ in range(count)
                    ],
                    keywords=[],
                ),
                node,
            )
            return self.inline_method(
                native_class, name, method, instance, invocation, ()
            )

        # `__exit__` is emitted on each path out, so a name it writes can no
        # longer be a build-time constant. Same for the body.
        assigned = list(node.body)
        assigned.extend(exit_method.body)
        self.materialize_runtime_names(self.assigned_names(assigned))

        entered = call("__enter__", enter)
        if item.optional_vars is not None:
            if not isinstance(item.optional_vars, ast.Name):
                raise NativeCompileError(
                    self.path, node, "a native `with ... as` binds a single name"
                )
            bound = item.optional_vars.id
            if self.enter_returns_self(enter):
                # The usual shape: the name is another way to say the manager.
                self.runtime_names.add(bound)
                self.operations.append(Store(self.slot(bound), instance))
                self.object_classes[bound] = native_class.name
                self.value_types[bound] = "object"
            elif entered is not None:
                self.runtime_names.add(bound)
                self.operations.append(Store(self.slot(bound), entered))
                self.value_types[bound] = "int"
            else:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"{native_class.name}.__enter__() returns nothing, so there "
                    "is nothing for `as` to bind",
                )

        escape = self.new_label("with_escape")
        end = self.new_label("with_end")
        self.open_cleanup_scope()
        self.handler_stack.append(escape)
        for statement in node.body:
            self.statement(statement)
        self.handler_stack.pop()
        self.finally_scopes.pop()
        call("__exit__", exit_method, 3)
        self.operations.append(Jump(end))
        self.operations.append(Label(escape))
        call("__exit__", exit_method, 3)
        self.propagate()
        self.operations.append(Label(end))

    @staticmethod
    def enter_returns_self(method: NativeFunction) -> bool:
        """Whether `__enter__` is exactly `return self`, the usual shape."""

        return (
            len(method.body) == 1
            and isinstance(method.body[0], ast.Return)
            and isinstance(method.body[0].value, ast.Name)
            and method.body[0].value.id == "self"
        )

    def try_statement(self, node: ast.Try) -> None:
        for clause in node.handlers:
            if clause.name is not None:
                raise NativeCompileError(
                    self.path,
                    clause,
                    "native except clauses cannot bind the exception to a name: "
                    "there is no exception object at runtime, only which class "
                    "was raised",
                )
        # Control reaches the end of a try by several paths, so a name written
        # on one of them can no longer be a build-time constant: pin every name
        # the statement assigns into a runtime slot first, exactly as a runtime
        # `if` does.
        assigned = list(node.body)
        for clause in node.handlers:
            assigned.extend(clause.body)
        assigned.extend(node.orelse)
        assigned.extend(node.finalbody)
        self.materialize_runtime_names(self.assigned_names(assigned))
        dispatch = self.new_label("except")
        end = self.new_label("try_end")

        def emit_finally() -> None:
            # The body is emitted on each path out rather than jumped to, since
            # there is no return address to come back on.
            for statement in node.finalbody:
                self.statement(statement)

        if node.finalbody:
            self.open_cleanup_scope()
        self.handler_stack.append(dispatch)
        for statement in node.body:
            self.statement(statement)
        self.handler_stack.pop()

        # An exception raised by an except clause, or by the else body, belongs
        # to the enclosing try - but this try's finally still has to run first.
        escape = self.new_label("finally_escape") if node.finalbody else None
        if escape is not None:
            self.handler_stack.append(escape)

        for statement in node.orelse:
            self.statement(statement)
        emit_finally()
        self.operations.append(Jump(end))

        self.operations.append(Label(dispatch))
        exhausted = True
        for clause in node.handlers:
            test = self.clause_matches(clause)
            skip = self.new_label("clause")
            if test is not None:
                self.operations.append(JumpIfFalse(test, skip))
            for statement in clause.body:
                self.statement(statement)
            emit_finally()
            self.operations.append(Jump(end))
            if test is None:
                exhausted = False  # a bare or BaseException clause catches all
                break
            self.operations.append(Label(skip))
        if exhausted:
            # No clause matched: run the finally and keep going outward.
            if escape is not None:
                self.handler_stack.pop()
            emit_finally()
            self.propagate()
            if escape is not None:
                self.handler_stack.append(escape)

        if escape is not None:
            self.handler_stack.pop()
            self.operations.append(Jump(end))
            self.operations.append(Label(escape))
            emit_finally()
            self.propagate()
        if node.finalbody:
            self.finally_scopes.pop()
        self.operations.append(Label(end))

    # The smallest signed 64-bit value has no positive counterpart, so the
    # usual "make it positive and peel digits" loop cannot handle it. It is one
    # value; spell it out rather than complicate the loop for it.
    _INT64_MIN = -9223372036854775808
    _INT64_MIN_TEXT = b"-9223372036854775808"

    def emit_int_to_string(self, value: IntExpression) -> IntExpression:
        """Render a runtime integer as decimal; returns a string-block pointer.

        Digits come out least-significant first, but they have to be written
        most-significant first, so the length is counted in one pass and the
        digits are then filled in from the end backwards.
        """

        bump = self.ensure_heap()
        pointer_slot = self.new_temp()
        # 20 digits is the widest a signed 64-bit value gets, plus a sign.
        self.operations.append(HeapAlloc(pointer_slot, IntConstant(8 + 24), bump))
        pointer = IntLoad(pointer_slot)
        payload = IntBinary("add", pointer, IntConstant(8))

        value_slot = self.new_temp()
        self.operations.append(Store(value_slot, value))
        done = self.new_label("itoa_done")

        # The one value that cannot be negated.
        general = self.new_label("itoa_general")
        self.operations.append(
            JumpIfFalse(
                IntCompare("eq", IntLoad(value_slot), IntConstant(self._INT64_MIN)),
                general,
            )
        )
        self.operations.append(
            HeapStore(pointer, IntConstant(len(self._INT64_MIN_TEXT)), 8)
        )
        for offset, byte in enumerate(self._INT64_MIN_TEXT):
            self.operations.append(
                HeapStore(
                    IntBinary("add", payload, IntConstant(offset)),
                    IntConstant(byte),
                    1,
                )
            )
        self.operations.append(Jump(done))
        self.operations.append(Label(general))

        sign_slot = self.new_temp()
        self.operations.append(Store(sign_slot, IntConstant(0)))
        positive = self.new_label("itoa_positive")
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(value_slot), IntConstant(0)), positive
            )
        )
        self.operations.append(Store(sign_slot, IntConstant(1)))
        self.operations.append(
            Store(value_slot, IntUnary("neg", IntLoad(value_slot)))
        )
        self.operations.append(
            HeapStore(payload, IntConstant(ord("-")), 1)
        )
        self.operations.append(Label(positive))

        # Pass one: how many digits. Zero has one, which no loop would produce.
        digits_slot = self.new_temp()
        self.operations.append(Store(digits_slot, IntConstant(1)))
        scratch_slot = self.new_temp()
        self.operations.append(
            Store(scratch_slot, IntBinary("sdiv", IntLoad(value_slot), IntConstant(10)))
        )
        count_start = self.new_label("itoa_count")
        count_done = self.new_label("itoa_counted")
        self.operations.append(Label(count_start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(scratch_slot), IntConstant(0)), count_done
            )
        )
        self.operations.append(
            Store(digits_slot, IntBinary("add", IntLoad(digits_slot), IntConstant(1)))
        )
        self.operations.append(
            Store(
                scratch_slot,
                IntBinary("sdiv", IntLoad(scratch_slot), IntConstant(10)),
            )
        )
        self.operations.append(Jump(count_start))
        self.operations.append(Label(count_done))

        self.operations.append(
            HeapStore(
                pointer,
                IntBinary("add", IntLoad(digits_slot), IntLoad(sign_slot)),
                8,
            )
        )
        # Pass two: fill backwards, so the most significant digit lands first.
        index_slot = self.new_temp()
        self.operations.append(
            Store(
                index_slot,
                IntBinary(
                    "sub",
                    IntBinary("add", IntLoad(digits_slot), IntLoad(sign_slot)),
                    IntConstant(1),
                ),
            )
        )
        fill_start = self.new_label("itoa_fill")
        fill_done = self.new_label("itoa_filled")
        self.operations.append(Label(fill_start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("ge", IntLoad(index_slot), IntLoad(sign_slot)), fill_done
            )
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", payload, IntLoad(index_slot)),
                IntBinary(
                    "add",
                    IntConstant(ord("0")),
                    IntBinary("smod", IntLoad(value_slot), IntConstant(10)),
                ),
                1,
            )
        )
        self.operations.append(
            Store(
                value_slot,
                IntBinary("sdiv", IntLoad(value_slot), IntConstant(10)),
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("sub", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(fill_start))
        self.operations.append(Label(fill_done))
        self.operations.append(Label(done))
        return pointer

    # --- rendering a float as the shortest decimal that reads back ----------
    #
    # CPython prints the shortest decimal string that parses back to the same
    # double. Deciding which string that is cannot be done in 64-bit
    # arithmetic: the comparison is between the value and its two neighbours,
    # scaled by a power of ten that can be 10^308 or 10^-324. So this carries
    # its own big integers - fixed-width arrays of 32-bit limbs in the arena -
    # and runs Burger and Dybvig's algorithm on them.
    #
    # Only six big integers are ever live (the value, the scale, the two
    # margins, and two temporaries), and they are reused, so the scratch is
    # reserved once at start-up rather than allocated per call. The returned
    # text lives in that scratch too and stays valid only until the next float
    # is rendered, which is enough because print() writes each argument before
    # evaluating the next.

    DTOA_LIMBS = 56  # 1792 bits: the widest intermediate needs about 1130
    DTOA_LIMB_MASK = 0xFFFFFFFF
    DTOA_BIGNUM_BYTES = 56 * 8
    DTOA_R, DTOA_S, DTOA_MP, DTOA_MM, DTOA_T1, DTOA_T2 = range(6)
    DTOA_DIGITS_OFFSET = 6 * DTOA_BIGNUM_BYTES
    DTOA_DIGITS_BYTES = 32
    DTOA_TEXT_OFFSET = DTOA_DIGITS_OFFSET + DTOA_DIGITS_BYTES
    DTOA_TEXT_BYTES = 64
    DTOA_SCRATCH_BYTES = DTOA_TEXT_OFFSET + DTOA_TEXT_BYTES

    def ensure_dtoa_scratch(self) -> int:
        """Reserve the slot holding the float-rendering scratch block."""

        if self._dtoa_scratch_slot is None:
            self.ensure_heap()
            self._dtoa_scratch_slot = self.slot("<dtoa-scratch>")
        return self._dtoa_scratch_slot

    def bignum(self, index: int) -> IntExpression:
        return IntBinary(
            "add",
            IntLoad(self.ensure_dtoa_scratch()),
            IntConstant(index * self.DTOA_BIGNUM_BYTES),
        )

    def dtoa_region(self, offset: int) -> IntExpression:
        return IntBinary(
            "add", IntLoad(self.ensure_dtoa_scratch()), IntConstant(offset)
        )

    def bn_limb(self, base: IntExpression, index: IntExpression) -> IntExpression:
        return IntBinary("add", base, IntBinary("mul", index, IntConstant(8)))

    def bn_loop(self, name: str):
        """Emit `for index in 0 .. LIMBS-1`; returns (index slot, end label)."""

        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        start = self.new_label(name)
        end = self.new_label(name + "_end")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntConstant(self.DTOA_LIMBS)),
                end,
            )
        )
        return index_slot, start, end

    def bn_loop_end(self, index_slot: int, start: str, end: str) -> None:
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))

    def bn_zero(self, base: IntExpression) -> None:
        index_slot, start, end = self.bn_loop("bn_zero")
        self.operations.append(
            HeapStore(self.bn_limb(base, IntLoad(index_slot)), IntConstant(0), 8)
        )
        self.bn_loop_end(index_slot, start, end)

    def bn_set(self, base: IntExpression, value: IntExpression) -> None:
        """Set to a non-negative value that fits in 64 bits."""

        value_slot = self.new_temp()
        self.operations.append(Store(value_slot, value))
        self.bn_zero(base)
        self.operations.append(
            HeapStore(
                base,
                IntBinary("and", IntLoad(value_slot), IntConstant(self.DTOA_LIMB_MASK)),
                8,
            )
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", base, IntConstant(8)),
                IntBinary(
                    "and",
                    IntBinary("rshift", IntLoad(value_slot), IntConstant(32)),
                    IntConstant(self.DTOA_LIMB_MASK),
                ),
                8,
            )
        )

    def bn_copy(self, destination: IntExpression, source: IntExpression) -> None:
        index_slot, start, end = self.bn_loop("bn_copy")
        self.operations.append(
            HeapStore(
                self.bn_limb(destination, IntLoad(index_slot)),
                HeapLoad(self.bn_limb(source, IntLoad(index_slot)), 8),
                8,
            )
        )
        self.bn_loop_end(index_slot, start, end)

    def bn_mul_small(self, base: IntExpression, multiplier: IntExpression) -> None:
        """Multiply in place by a small value; limb * multiplier must fit i64."""

        multiplier_slot = self.new_temp()
        self.operations.append(Store(multiplier_slot, multiplier))
        carry_slot = self.new_temp()
        self.operations.append(Store(carry_slot, IntConstant(0)))
        index_slot, start, end = self.bn_loop("bn_mul")
        product_slot = self.new_temp()
        self.operations.append(
            Store(
                product_slot,
                IntBinary(
                    "add",
                    IntBinary(
                        "mul",
                        HeapLoad(self.bn_limb(base, IntLoad(index_slot)), 8),
                        IntLoad(multiplier_slot),
                    ),
                    IntLoad(carry_slot),
                ),
            )
        )
        self.operations.append(
            HeapStore(
                self.bn_limb(base, IntLoad(index_slot)),
                IntBinary(
                    "and", IntLoad(product_slot), IntConstant(self.DTOA_LIMB_MASK)
                ),
                8,
            )
        )
        self.operations.append(
            Store(
                carry_slot,
                IntBinary("rshift", IntLoad(product_slot), IntConstant(32)),
            )
        )
        self.bn_loop_end(index_slot, start, end)

    def bn_add(self, base: IntExpression, addend: IntExpression) -> None:
        carry_slot = self.new_temp()
        self.operations.append(Store(carry_slot, IntConstant(0)))
        index_slot, start, end = self.bn_loop("bn_add")
        total_slot = self.new_temp()
        self.operations.append(
            Store(
                total_slot,
                IntBinary(
                    "add",
                    IntBinary(
                        "add",
                        HeapLoad(self.bn_limb(base, IntLoad(index_slot)), 8),
                        HeapLoad(self.bn_limb(addend, IntLoad(index_slot)), 8),
                    ),
                    IntLoad(carry_slot),
                ),
            )
        )
        self.operations.append(
            HeapStore(
                self.bn_limb(base, IntLoad(index_slot)),
                IntBinary("and", IntLoad(total_slot), IntConstant(self.DTOA_LIMB_MASK)),
                8,
            )
        )
        self.operations.append(
            Store(carry_slot, IntBinary("rshift", IntLoad(total_slot), IntConstant(32)))
        )
        self.bn_loop_end(index_slot, start, end)

    def bn_sub(self, base: IntExpression, subtrahend: IntExpression) -> None:
        """Subtract in place. The caller guarantees base >= subtrahend."""

        borrow_slot = self.new_temp()
        self.operations.append(Store(borrow_slot, IntConstant(0)))
        index_slot, start, end = self.bn_loop("bn_sub")
        difference_slot = self.new_temp()
        self.operations.append(
            Store(
                difference_slot,
                IntBinary(
                    "sub",
                    IntBinary(
                        "sub",
                        HeapLoad(self.bn_limb(base, IntLoad(index_slot)), 8),
                        HeapLoad(self.bn_limb(subtrahend, IntLoad(index_slot)), 8),
                    ),
                    IntLoad(borrow_slot),
                ),
            )
        )
        self.operations.append(
            HeapStore(
                self.bn_limb(base, IntLoad(index_slot)),
                IntBinary(
                    "and", IntLoad(difference_slot), IntConstant(self.DTOA_LIMB_MASK)
                ),
                8,
            )
        )
        # A negative difference means this limb borrowed from the next one.
        self.operations.append(
            Store(
                borrow_slot,
                IntBinary(
                    "and",
                    IntBinary("rshift", IntLoad(difference_slot), IntConstant(63)),
                    IntConstant(1),
                ),
            )
        )
        self.bn_loop_end(index_slot, start, end)

    def bn_compare(self, left: IntExpression, right: IntExpression) -> int:
        """Return a slot holding -1, 0, or 1."""

        result_slot = self.new_temp()
        self.operations.append(Store(result_slot, IntConstant(0)))
        index_slot = self.new_temp()
        self.operations.append(
            Store(index_slot, IntConstant(self.DTOA_LIMBS - 1))
        )
        start = self.new_label("bn_cmp")
        end = self.new_label("bn_cmp_end")
        step = self.new_label("bn_cmp_next")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("ge", IntLoad(index_slot), IntConstant(0)), end
            )
        )
        left_slot = self.new_temp()
        right_slot = self.new_temp()
        self.operations.append(
            Store(left_slot, HeapLoad(self.bn_limb(left, IntLoad(index_slot)), 8))
        )
        self.operations.append(
            Store(right_slot, HeapLoad(self.bn_limb(right, IntLoad(index_slot)), 8))
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(left_slot), IntLoad(right_slot)), step
            )
        )
        # The most significant differing limb decides the whole comparison.
        self.operations.append(
            Store(
                result_slot,
                self.select_integer(
                    IntCompare("gt", IntLoad(left_slot), IntLoad(right_slot)),
                    IntConstant(1),
                    IntConstant(-1),
                ),
            )
        )
        self.operations.append(Jump(end))
        self.operations.append(Label(step))
        self.operations.append(
            Store(index_slot, IntBinary("sub", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
        return result_slot

    def bn_double_times(self, base: IntExpression, count: IntExpression) -> None:
        """Multiply by 2^count, one doubling at a time."""

        remaining_slot = self.new_temp()
        self.operations.append(Store(remaining_slot, count))
        start = self.new_label("bn_shift")
        end = self.new_label("bn_shift_end")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("gt", IntLoad(remaining_slot), IntConstant(0)), end
            )
        )
        self.bn_mul_small(base, IntConstant(2))
        self.operations.append(
            Store(
                remaining_slot,
                IntBinary("sub", IntLoad(remaining_slot), IntConstant(1)),
            )
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))

    def dtoa_append(self, length_slot: int, byte: IntExpression) -> None:
        payload = self.dtoa_region(self.DTOA_TEXT_OFFSET + 8)
        self.operations.append(
            HeapStore(IntBinary("add", payload, IntLoad(length_slot)), byte, 1)
        )
        self.operations.append(
            Store(length_slot, IntBinary("add", IntLoad(length_slot), IntConstant(1)))
        )

    def dtoa_append_text(self, length_slot: int, text: bytes) -> None:
        for byte in text:
            self.dtoa_append(length_slot, IntConstant(byte))

    def dtoa_append_repeat(
        self, length_slot: int, byte: IntExpression, count: IntExpression
    ) -> None:
        remaining_slot = self.new_temp()
        self.operations.append(Store(remaining_slot, count))
        start = self.new_label("pad")
        end = self.new_label("pad_end")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(IntCompare("gt", IntLoad(remaining_slot), IntConstant(0)), end)
        )
        self.dtoa_append(length_slot, byte)
        self.operations.append(
            Store(
                remaining_slot,
                IntBinary("sub", IntLoad(remaining_slot), IntConstant(1)),
            )
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))

    def dtoa_append_digits(
        self, length_slot: int, first: IntExpression, limit: IntExpression
    ) -> None:
        digits = self.dtoa_region(self.DTOA_DIGITS_OFFSET)
        index_slot = self.new_temp()
        limit_slot = self.new_temp()
        self.operations.append(Store(index_slot, first))
        self.operations.append(Store(limit_slot, limit))
        start = self.new_label("emit_digits")
        end = self.new_label("emit_digits_end")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(limit_slot)), end
            )
        )
        self.dtoa_append(
            length_slot,
            IntBinary(
                "add",
                IntConstant(ord("0")),
                HeapLoad(
                    IntBinary("add", digits, IntLoad(index_slot)), 1
                ),
            ),
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))

    def dtoa_boundary(self, comparison: int, inclusive: int, strict: str) -> int:
        """`comparison <strict> 0`, or `== 0` too when the bound is inclusive."""

        result_slot = self.new_temp()
        self.operations.append(Store(result_slot, IntConstant(0)))
        done = self.new_label("bound_done")
        set_it = self.new_label("bound_set")
        self.operations.append(
            JumpIfFalse(
                IntCompare(strict, IntLoad(comparison), IntConstant(0)), set_it + "_no"
            )
        )
        self.operations.append(Jump(set_it))
        self.operations.append(Label(set_it + "_no"))
        # An even significand makes the neighbour exactly representable too, so
        # the boundary counts as reached rather than passed.
        self.operations.append(
            JumpIfFalse(IntCompare("ne", IntLoad(inclusive), IntConstant(0)), done)
        )
        self.operations.append(
            JumpIfFalse(IntCompare("eq", IntLoad(comparison), IntConstant(0)), done)
        )
        self.operations.append(Label(set_it))
        self.operations.append(Store(result_slot, IntConstant(1)))
        self.operations.append(Label(done))
        return result_slot

    def emit_float_to_string(self, value: FloatExpression) -> IntExpression:
        """Render a double the way CPython's repr() does.

        The shortest decimal that reads back as the same double, found by
        Burger and Dybvig's method: hold the value and its two neighbours as
        exact rationals, scale until the first digit is about to come out, then
        emit digits until what is left is unambiguously nearer this double than
        either neighbour.
        """

        scratch = self.ensure_dtoa_scratch()
        text = self.dtoa_region(self.DTOA_TEXT_OFFSET)
        digits = self.dtoa_region(self.DTOA_DIGITS_OFFSET)
        bits_slot = self.new_temp()
        self.operations.append(Store(bits_slot, FloatBits(value)))
        bits = IntLoad(bits_slot)
        length_slot = self.new_temp()
        self.operations.append(Store(length_slot, IntConstant(0)))

        biased_slot = self.new_temp()
        self.operations.append(
            Store(
                biased_slot,
                IntBinary(
                    "and",
                    IntBinary("rshift", bits, IntConstant(52)),
                    IntConstant(0x7FF),
                ),
            )
        )
        fraction_slot = self.new_temp()
        self.operations.append(
            Store(
                fraction_slot,
                IntBinary("and", bits, IntConstant((1 << 52) - 1)),
            )
        )
        negative_slot = self.new_temp()
        self.operations.append(
            Store(
                negative_slot,
                IntBinary(
                    "and", IntBinary("rshift", bits, IntConstant(63)), IntConstant(1)
                ),
            )
        )
        finish = self.new_label("dtoa_finish")

        # Infinities and NaNs first: a NaN prints unsigned, so this has to come
        # before the sign is written.
        ordinary = self.new_label("dtoa_ordinary")
        self.operations.append(
            JumpIfFalse(
                IntCompare("eq", IntLoad(biased_slot), IntConstant(0x7FF)), ordinary
            )
        )
        infinite = self.new_label("dtoa_inf")
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(fraction_slot), IntConstant(0)), infinite
            )
        )
        self.dtoa_append_text(length_slot, b"nan")
        self.operations.append(Jump(finish))
        self.operations.append(Label(infinite))
        signed_infinite = self.new_label("dtoa_inf_signed")
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(negative_slot), IntConstant(0)),
                signed_infinite,
            )
        )
        self.dtoa_append_text(length_slot, b"-")
        self.operations.append(Label(signed_infinite))
        self.dtoa_append_text(length_slot, b"inf")
        self.operations.append(Jump(finish))

        self.operations.append(Label(ordinary))
        unsigned = self.new_label("dtoa_unsigned")
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(negative_slot), IntConstant(0)), unsigned
            )
        )
        self.dtoa_append_text(length_slot, b"-")
        self.operations.append(Label(unsigned))

        # Zero has no digits to generate, and negative zero keeps its sign.
        nonzero = self.new_label("dtoa_nonzero")
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "ne",
                    IntBinary("or", IntLoad(biased_slot), IntLoad(fraction_slot)),
                    IntConstant(0),
                ),
                nonzero + "_zero",
            )
        )
        self.operations.append(Jump(nonzero))
        self.operations.append(Label(nonzero + "_zero"))
        self.dtoa_append_text(length_slot, b"0.0")
        self.operations.append(Jump(finish))
        self.operations.append(Label(nonzero))

        # v = significand * 2^exponent, with the implicit bit restored unless
        # the number is subnormal.
        significand_slot = self.new_temp()
        exponent_slot = self.new_temp()
        self.operations.append(
            Store(
                significand_slot,
                self.select_integer(
                    IntCompare("eq", IntLoad(biased_slot), IntConstant(0)),
                    IntLoad(fraction_slot),
                    IntBinary("or", IntLoad(fraction_slot), IntConstant(1 << 52)),
                ),
            )
        )
        self.operations.append(
            Store(
                exponent_slot,
                self.select_integer(
                    IntCompare("eq", IntLoad(biased_slot), IntConstant(0)),
                    IntConstant(-1074),
                    IntBinary("sub", IntLoad(biased_slot), IntConstant(1075)),
                ),
            )
        )
        # At the bottom of a binade the gap below is half the gap above - but
        # not at the bottom of the normal range, where the neighbour below is
        # subnormal and the gap is the same.
        asymmetric_slot = self.new_temp()
        self.operations.append(
            Store(
                asymmetric_slot,
                self.select_integer(
                    IntBinary(
                        "and",
                        IntCompare("eq", IntLoad(fraction_slot), IntConstant(0)),
                        IntCompare("gt", IntLoad(biased_slot), IntConstant(1)),
                    ),
                    IntConstant(1),
                    IntConstant(0),
                ),
            )
        )
        inclusive_slot = self.new_temp()
        self.operations.append(
            Store(
                inclusive_slot,
                self.select_integer(
                    IntCompare(
                        "eq",
                        IntBinary("and", IntLoad(significand_slot), IntConstant(1)),
                        IntConstant(0),
                    ),
                    IntConstant(1),
                    IntConstant(0),
                ),
            )
        )

        exponent = IntLoad(exponent_slot)
        asymmetric = IntLoad(asymmetric_slot)
        nonnegative = IntCompare("ge", exponent, IntConstant(0))
        value_shift = self.materialize_int(
            self.select_integer(
                nonnegative,
                IntBinary(
                    "add", IntBinary("add", exponent, IntConstant(1)), asymmetric
                ),
                IntBinary("add", IntConstant(1), asymmetric),
            )
        )
        scale_value = self.materialize_int(
            self.select_integer(
                nonnegative,
                IntBinary(
                    "add",
                    IntConstant(2),
                    IntBinary("mul", asymmetric, IntConstant(2)),
                ),
                IntConstant(1),
            )
        )
        scale_shift = self.materialize_int(
            self.select_integer(
                nonnegative,
                IntConstant(0),
                IntBinary(
                    "add",
                    IntBinary("sub", IntConstant(1), exponent),
                    asymmetric,
                ),
            )
        )
        plus_shift = self.materialize_int(
            self.select_integer(
                nonnegative, IntBinary("add", exponent, asymmetric), asymmetric
            )
        )
        minus_shift = self.materialize_int(
            self.select_integer(nonnegative, exponent, IntConstant(0))
        )

        r = self.bignum(self.DTOA_R)
        s = self.bignum(self.DTOA_S)
        mp = self.bignum(self.DTOA_MP)
        mm = self.bignum(self.DTOA_MM)
        t1 = self.bignum(self.DTOA_T1)
        t2 = self.bignum(self.DTOA_T2)
        self.bn_set(r, IntLoad(significand_slot))
        self.bn_double_times(r, value_shift)
        self.bn_set(s, scale_value)
        self.bn_double_times(s, scale_shift)
        self.bn_set(mp, IntConstant(1))
        self.bn_double_times(mp, plus_shift)
        self.bn_set(mm, IntConstant(1))
        self.bn_double_times(mm, minus_shift)

        # Scale until the value sits in (1/10, 1], counting the decimal point
        # position as it goes. The two tests are exact complements, so the
        # branches cannot undo each other.
        point_slot = self.new_temp()
        self.operations.append(Store(point_slot, IntConstant(0)))
        scale_start = self.new_label("dtoa_scale")
        scale_end = self.new_label("dtoa_scale_end")
        self.operations.append(Label(scale_start))
        self.bn_copy(t1, r)
        self.bn_add(t1, mp)
        upper = self.bn_compare(t1, s)
        high = self.dtoa_boundary(upper, inclusive_slot, "gt")
        smaller = self.new_label("dtoa_scale_down")
        self.operations.append(
            JumpIfFalse(IntCompare("ne", IntLoad(high), IntConstant(0)), smaller)
        )
        self.bn_mul_small(s, IntConstant(10))
        self.operations.append(
            Store(point_slot, IntBinary("add", IntLoad(point_slot), IntConstant(1)))
        )
        self.operations.append(Jump(scale_start))
        self.operations.append(Label(smaller))
        self.bn_copy(t2, t1)
        self.bn_mul_small(t2, IntConstant(10))
        scaled = self.bn_compare(t2, s)
        # Exactly the negation of the test above, applied to the scaled value,
        # which means the opposite inclusivity: not (x > s or (ok and x == s))
        # is (x < s) when ok, and (x <= s) when not. Anything else lets the two
        # branches undo each other forever.
        exclusive_slot = self.new_temp()
        self.operations.append(
            Store(
                exclusive_slot,
                IntBinary("sub", IntConstant(1), IntLoad(inclusive_slot)),
            )
        )
        too_small = self.dtoa_boundary(scaled, exclusive_slot, "lt")
        self.operations.append(
            JumpIfFalse(IntCompare("ne", IntLoad(too_small), IntConstant(0)), scale_end)
        )
        self.bn_mul_small(r, IntConstant(10))
        self.bn_mul_small(mp, IntConstant(10))
        self.bn_mul_small(mm, IntConstant(10))
        self.operations.append(
            Store(point_slot, IntBinary("sub", IntLoad(point_slot), IntConstant(1)))
        )
        self.operations.append(Jump(scale_start))
        self.operations.append(Label(scale_end))

        # Generate digits until what remains is closer to this double than to
        # either neighbour.
        count_slot = self.new_temp()
        self.operations.append(Store(count_slot, IntConstant(0)))
        digit_slot = self.new_temp()
        generate = self.new_label("dtoa_generate")
        generated = self.new_label("dtoa_generated")
        self.operations.append(Label(generate))
        self.bn_mul_small(r, IntConstant(10))
        self.operations.append(Store(digit_slot, IntConstant(0)))
        divide = self.new_label("dtoa_divide")
        divided = self.new_label("dtoa_divided")
        self.operations.append(Label(divide))
        quotient = self.bn_compare(r, s)
        self.operations.append(
            JumpIfFalse(IntCompare("ge", IntLoad(quotient), IntConstant(0)), divided)
        )
        # r < 10s on entry, so this runs at most nine times: no big division.
        self.bn_sub(r, s)
        self.operations.append(
            Store(digit_slot, IntBinary("add", IntLoad(digit_slot), IntConstant(1)))
        )
        self.operations.append(Jump(divide))
        self.operations.append(Label(divided))
        self.bn_mul_small(mp, IntConstant(10))
        self.bn_mul_small(mm, IntConstant(10))
        lower = self.bn_compare(r, mm)
        low = self.dtoa_boundary(lower, inclusive_slot, "lt")
        self.bn_copy(t1, r)
        self.bn_add(t1, mp)
        upper = self.bn_compare(t1, s)
        high = self.dtoa_boundary(upper, inclusive_slot, "gt")

        terminal = self.new_label("dtoa_terminal")
        self.operations.append(
            JumpIfFalse(IntCompare("ne", IntLoad(low), IntConstant(0)), terminal + "_a")
        )
        self.operations.append(Jump(terminal))
        self.operations.append(Label(terminal + "_a"))
        self.operations.append(
            JumpIfFalse(IntCompare("ne", IntLoad(high), IntConstant(0)), terminal + "_b")
        )
        self.operations.append(Jump(terminal))
        self.operations.append(Label(terminal + "_b"))
        # Neither bound reached: this digit is certain, keep going.
        self.operations.append(
            HeapStore(
                IntBinary("add", digits, IntLoad(count_slot)),
                IntLoad(digit_slot),
                1,
            )
        )
        self.operations.append(
            Store(count_slot, IntBinary("add", IntLoad(count_slot), IntConstant(1)))
        )
        self.operations.append(Jump(generate))

        self.operations.append(Label(terminal))
        bump_slot = self.new_temp()
        self.operations.append(Store(bump_slot, IntConstant(0)))
        settled = self.new_label("dtoa_settled")
        self.operations.append(
            JumpIfFalse(IntCompare("ne", IntLoad(high), IntConstant(0)), settled)
        )
        upper_only = self.new_label("dtoa_upper_only")
        self.operations.append(
            JumpIfFalse(IntCompare("ne", IntLoad(low), IntConstant(0)), upper_only)
        )
        self.bn_copy(t2, r)
        self.bn_add(t2, r)
        halfway = self.bn_compare(t2, s)
        self.operations.append(
            Store(
                bump_slot,
                self.select_integer(
                    IntCompare("gt", IntLoad(halfway), IntConstant(0)),
                    IntConstant(1),
                    # An exact tie: CPython breaks it toward the even digit.
                    self.select_integer(
                        IntBinary(
                            "and",
                            IntCompare("eq", IntLoad(halfway), IntConstant(0)),
                            IntBinary("and", IntLoad(digit_slot), IntConstant(1)),
                        ),
                        IntConstant(1),
                        IntConstant(0),
                    ),
                ),
            )
        )
        self.operations.append(Jump(settled))
        self.operations.append(Label(upper_only))
        # Only the upper bound was reached, so the value rounds up.
        self.operations.append(Store(bump_slot, IntConstant(1)))
        self.operations.append(Label(settled))
        self.operations.append(
            HeapStore(
                IntBinary("add", digits, IntLoad(count_slot)),
                IntBinary("add", IntLoad(digit_slot), IntLoad(bump_slot)),
                1,
            )
        )
        self.operations.append(
            Store(count_slot, IntBinary("add", IntLoad(count_slot), IntConstant(1)))
        )
        self.operations.append(Label(generated))

        # Rounding up can turn the last digit into ten. Carry it back, and if
        # it runs off the front the whole string was nines.
        carry_slot = self.new_temp()
        self.operations.append(
            Store(carry_slot, IntBinary("sub", IntLoad(count_slot), IntConstant(1)))
        )
        carry = self.new_label("dtoa_carry")
        carried = self.new_label("dtoa_carried")
        self.operations.append(Label(carry))
        self.operations.append(
            JumpIfFalse(IntCompare("ge", IntLoad(carry_slot), IntConstant(0)), carried)
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "eq",
                    HeapLoad(IntBinary("add", digits, IntLoad(carry_slot)), 1),
                    IntConstant(10),
                ),
                carried,
            )
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", digits, IntLoad(carry_slot)), IntConstant(0), 1
            )
        )
        front = self.new_label("dtoa_carry_front")
        self.operations.append(
            JumpIfFalse(
                IntCompare("eq", IntLoad(carry_slot), IntConstant(0)), front + "_no"
            )
        )
        self.operations.append(HeapStore(digits, IntConstant(1), 1))
        self.operations.append(Store(count_slot, IntConstant(1)))
        self.operations.append(
            Store(point_slot, IntBinary("add", IntLoad(point_slot), IntConstant(1)))
        )
        self.operations.append(Jump(carried))
        self.operations.append(Label(front + "_no"))
        previous = IntBinary(
            "add", digits, IntBinary("sub", IntLoad(carry_slot), IntConstant(1))
        )
        self.operations.append(
            HeapStore(
                previous, IntBinary("add", HeapLoad(previous, 1), IntConstant(1)), 1
            )
        )
        self.operations.append(
            Store(carry_slot, IntBinary("sub", IntLoad(carry_slot), IntConstant(1)))
        )
        self.operations.append(Jump(carry))
        self.operations.append(Label(carried))
        # A trailing zero is never part of the shortest representation.
        strip = self.new_label("dtoa_strip")
        stripped = self.new_label("dtoa_stripped")
        self.operations.append(Label(strip))
        self.operations.append(
            JumpIfFalse(
                IntCompare("gt", IntLoad(count_slot), IntConstant(1)), stripped
            )
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "eq",
                    HeapLoad(
                        IntBinary(
                            "add",
                            digits,
                            IntBinary("sub", IntLoad(count_slot), IntConstant(1)),
                        ),
                        1,
                    ),
                    IntConstant(0),
                ),
                stripped,
            )
        )
        self.operations.append(
            Store(count_slot, IntBinary("sub", IntLoad(count_slot), IntConstant(1)))
        )
        self.operations.append(Jump(strip))
        self.operations.append(Label(stripped))

        # Layout, following CPython: exponential when the point sits more than
        # four places left of the digits or past the sixteenth place.
        point = IntLoad(point_slot)
        count = IntLoad(count_slot)
        fixed = self.new_label("dtoa_fixed")
        self.operations.append(
            JumpIfFalse(
                IntBinary(
                    "or",
                    IntCompare("le", point, IntConstant(-4)),
                    IntCompare("gt", point, IntConstant(16)),
                ),
                fixed,
            )
        )
        self.dtoa_append_digits(length_slot, IntConstant(0), IntConstant(1))
        single = self.new_label("dtoa_single")
        self.operations.append(
            JumpIfFalse(IntCompare("gt", count, IntConstant(1)), single)
        )
        self.dtoa_append_text(length_slot, b".")
        self.dtoa_append_digits(length_slot, IntConstant(1), count)
        self.operations.append(Label(single))
        self.dtoa_append_text(length_slot, b"e")
        power_slot = self.new_temp()
        self.operations.append(
            Store(power_slot, IntBinary("sub", point, IntConstant(1)))
        )
        positive_power = self.new_label("dtoa_power_positive")
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(power_slot), IntConstant(0)), positive_power
            )
        )
        self.dtoa_append_text(length_slot, b"-")
        self.operations.append(
            Store(power_slot, IntUnary("neg", IntLoad(power_slot)))
        )
        self.operations.append(Jump(positive_power + "_done"))
        self.operations.append(Label(positive_power))
        self.dtoa_append_text(length_slot, b"+")
        self.operations.append(Label(positive_power + "_done"))
        # At least two digits, three when the exponent reaches a hundred.
        hundreds = self.new_label("dtoa_hundreds")
        self.operations.append(
            JumpIfFalse(
                IntCompare("ge", IntLoad(power_slot), IntConstant(100)), hundreds
            )
        )
        self.dtoa_append(
            length_slot,
            IntBinary(
                "add",
                IntConstant(ord("0")),
                IntBinary("sdiv", IntLoad(power_slot), IntConstant(100)),
            ),
        )
        self.operations.append(Label(hundreds))
        self.dtoa_append(
            length_slot,
            IntBinary(
                "add",
                IntConstant(ord("0")),
                IntBinary(
                    "smod",
                    IntBinary("sdiv", IntLoad(power_slot), IntConstant(10)),
                    IntConstant(10),
                ),
            ),
        )
        self.dtoa_append(
            length_slot,
            IntBinary(
                "add",
                IntConstant(ord("0")),
                IntBinary("smod", IntLoad(power_slot), IntConstant(10)),
            ),
        )
        self.operations.append(Jump(finish))

        self.operations.append(Label(fixed))
        leading = self.new_label("dtoa_leading")
        self.operations.append(
            JumpIfFalse(IntCompare("le", point, IntConstant(0)), leading + "_no")
        )
        self.operations.append(Jump(leading))
        self.operations.append(Label(leading + "_no"))
        trailing = self.new_label("dtoa_trailing")
        self.operations.append(
            JumpIfFalse(IntCompare("ge", point, count), trailing + "_no")
        )
        self.operations.append(Jump(trailing))
        self.operations.append(Label(trailing + "_no"))
        # The point falls inside the digits.
        self.dtoa_append_digits(length_slot, IntConstant(0), point)
        self.dtoa_append_text(length_slot, b".")
        self.dtoa_append_digits(length_slot, point, count)
        self.operations.append(Jump(finish))

        self.operations.append(Label(leading))
        self.dtoa_append_text(length_slot, b"0.")
        self.dtoa_append_repeat(
            length_slot, IntConstant(ord("0")), IntUnary("neg", point)
        )
        self.dtoa_append_digits(length_slot, IntConstant(0), count)
        self.operations.append(Jump(finish))

        self.operations.append(Label(trailing))
        self.dtoa_append_digits(length_slot, IntConstant(0), count)
        self.dtoa_append_repeat(
            length_slot, IntConstant(ord("0")), IntBinary("sub", point, count)
        )
        self.dtoa_append_text(length_slot, b".0")

        self.operations.append(Label(finish))
        self.operations.append(HeapStore(text, IntLoad(length_slot), 8))
        return text

    def ensure_bool_text(self) -> int:
        """A slot pointing at the string blocks for ``True`` and ``False``.

        Reserved once at start-up. Rendering them at each site would allocate
        inside loops, and they never change.
        """

        if self._bool_text_slot is None:
            bump = self.ensure_heap()
            self._bool_text_slot = self.slot("<bool-text>")
            pointer = IntLoad(self._bool_text_slot)
            self._prologue.append(
                HeapAlloc(self._bool_text_slot, IntConstant(32), bump)
            )
            for offset, text in ((0, b"True"), (16, b"False")):
                self._prologue.append(
                    HeapStore(
                        IntBinary("add", pointer, IntConstant(offset)),
                        IntConstant(len(text)),
                        8,
                    )
                )
                for index, byte in enumerate(text):
                    self._prologue.append(
                        HeapStore(
                            IntBinary(
                                "add", pointer, IntConstant(offset + 8 + index)
                            ),
                            IntConstant(byte),
                            1,
                        )
                    )
        return self._bool_text_slot

    def renders_as_bool(self, node: ast.expr) -> bool:
        """Whether this expression's Python value is a ``bool``.

        The native subset keeps a bool in an integer slot, which is right for
        arithmetic and wrong for printing: CPython writes ``True``, not ``1``.
        Nothing distinguishes the two at run time, so the question is answered
        from the source.
        """

        if isinstance(node, ast.Constant):
            return isinstance(node.value, bool)
        if isinstance(node, ast.Compare):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return True
        if isinstance(node, ast.BoolOp):
            # `a and b` yields an operand, so it is a bool only if both are.
            return all(self.renders_as_bool(value) for value in node.values)
        if isinstance(node, ast.Name):
            return node.id in self.boolean_names
        return False

    def emit_bool_to_string(self, value: IntExpression) -> IntExpression:
        base = IntLoad(self.ensure_bool_text())
        return self.select_integer(
            IntCompare("ne", value, IntConstant(0)),
            base,
            IntBinary("add", base, IntConstant(16)),
        )

    def emit_print_argument(self, node: ast.expr) -> None:
        """Write one print() argument, with no separator and no newline."""

        try:
            self.operations.append(
                Write(str(self.constant(node)).encode("utf-8"))
            )
            return
        except NativeCompileError:
            pass  # Not known at build time; render it at run time below.
        kind = self.expression_type(node)
        if kind == "str":
            pointer = self.string_pointer(node)
        elif kind == "int" and self.renders_as_bool(node):
            pointer = self.emit_bool_to_string(self.integer(node))
        elif kind == "int":
            pointer = self.emit_int_to_string(self.integer(node))
        elif kind == "float":
            pointer = self.emit_float_to_string(self.float_expression(node))
        else:
            raise NativeCompileError(
                self.path,
                node,
                f"native print() cannot render a runtime {kind} yet; integers "
                "and strings are supported",
            )
        pointer = self.materialize_int(pointer)
        self.operations.append(
            WriteRuntime(
                IntBinary("add", pointer, IntConstant(8)), HeapLoad(pointer, 8)
            )
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
        argument_kinds: tuple[str, ...] = (),
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

        for index, (parameter, argument) in enumerate(
            zip(function.parameters, arguments)
        ):
            private_parameter = private_names[parameter]
            self.runtime_names.add(private_parameter)
            kind = argument_kinds[index] if index < len(argument_kinds) else None
            # A parameter is just a local: store the argument in its slot and
            # the body reads it through the ordinary variable path. Recording
            # the kind is what makes that path pick float or integer loads.
            if isinstance(argument, FLOAT_EXPRESSIONS):
                self.operations.append(
                    FloatStore(self.slot(private_parameter), argument)
                )
                self.value_types[private_parameter] = "float"
            else:
                self.operations.append(
                    Store(self.slot(private_parameter), argument)
                )
                if kind == "str":
                    self.value_types[private_parameter] = "str"
                else:
                    self.value_types.pop(private_parameter, None)
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
        kinds: list[str] | None = None,
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
        bound_kinds: list[str] = ["int" for _ in parameters]
        for index, argument in enumerate(node.args):
            kind = self.expression_type(argument, bindings)
            if self.experimental_kernels:
                bound[index] = self.kernel_operand(argument, bindings, call_stack)
            elif kind == "float":
                bound[index] = self.float_expression(argument, bindings, call_stack)
                bound_kinds[index] = "float"
            elif kind == "str":
                # A string is its block pointer, so passing one is passing an
                # integer; only the parameter's recorded kind differs.
                bound[index] = self.string_pointer(argument)
                bound_kinds[index] = "str"
            else:
                bound[index] = self.integer(argument, bindings, call_stack)
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
        if kinds is not None:
            kinds[:] = bound_kinds
        return tuple(value for value in bound if value is not None)

    def expression_function_kind(
        self, node: ast.expr, bindings: dict[str, KernelValue] | None = None
    ) -> str | None:
        """The kind a `return <expression>` function yields, or None.

        Such a function is inlined by substituting its arguments into the one
        expression, so its result kind is that expression's kind under the
        argument kinds - no body to walk, and nothing lowered to find out.
        """

        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id in {"int", "len", "print"}
        ):
            return None
        function = self.functions.get(node.func.id)
        if function is None:
            return None
        if function.expression is None:
            # A statement body is inlined, and its result kind is only known
            # once it has been. Say integer so the caller takes the integer
            # path, where a float return is caught and reported precisely.
            return "int"
        if len(node.args) != len(function.parameters) or node.keywords:
            return None  # defaults and keywords: let the ordinary path decide
        # Stand-ins of the right kind, so nothing is emitted just to ask.
        stand_ins: dict[str, KernelValue] = {}
        strings: dict[str, IntExpression] = {}
        for parameter, argument in zip(function.parameters, node.args):
            kind = self.expression_type(argument, bindings)
            if kind == "float":
                stand_ins[parameter] = FloatConstant(0.0)
            else:
                stand_ins[parameter] = IntConstant(0)
                if kind == "str":
                    # A string's stand-in is a pointer like any other integer,
                    # so the kind has to be recorded beside it.
                    strings[parameter] = IntConstant(0)
        previous_functions, previous_values = self.functions, self.values
        previous_strings = self.string_bindings
        self.functions, self.values = function.functions, function.values
        self.string_bindings = {**previous_strings, **strings}
        try:
            return self.expression_type(function.expression, stand_ins)
        finally:
            self.functions, self.values = previous_functions, previous_values
            self.string_bindings = previous_strings

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
                if node.id in self.string_bindings:
                    return "str"
                return (
                    "float"
                    if isinstance(bindings[node.id], FLOAT_EXPRESSIONS)
                    else "int"
                )
            if node.id in self.object_classes:
                return "object"
            if node.id in self.value_types:
                return self.value_types[node.id]
            value = self.values.get(node.id)
            if isinstance(value, str):
                return "str"
            return "float" if isinstance(value, float) else "int"
        if isinstance(node, ast.ListComp):
            return self.list_tag(self.comprehension_element_kind(node))
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            return self.expression_type(node.value, bindings)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and self.dict_kinds_of(node.value.id)
        ):
            return self.dict_kinds_of(node.value.id)[1]
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and self.list_kind_of(node.value.id) is not None
        ):
            return self.list_kind_of(node.value.id)
        if isinstance(node, ast.Attribute) and self.resolve_object_class(node.value):
            return self.attribute_kind(node)
        kind = self.expression_function_kind(node, bindings)
        if kind is not None:
            return kind
        if isinstance(node, ast.JoinedStr):
            return "str"
        if isinstance(node, ast.Dict):
            return self.dict_literal_tag(node, bindings)
        if isinstance(node, ast.List):
            return self.list_literal_tag(node, bindings)
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
            if (
                node.func.id in {"int", "len", "abs", "sum", "min", "max"}
                and node.func.id not in self.functions
            ):
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

        A constant divisor is checked here, where a zero is a build error. A
        runtime divisor is checked by the program: Python raises
        ``ZeroDivisionError`` rather than producing IEEE infinity or NaN, and
        that is now something the generated code can do.
        """

        try:
            divisor = self.constant(node)
        except NativeCompileError:
            divisor = None
        if isinstance(divisor, (int, float)) and not isinstance(divisor, bool):
            if float(divisor) == 0.0:
                raise NativeCompileError(self.path, node, "float division by zero")
            return FloatConstant(float(divisor))
        value = self.float_expression(node, bindings, call_stack)
        # Pin it: the check and the division both read it, and the backends
        # re-emit an expression tree at every occurrence, so an unpinned
        # divisor containing a call would be computed twice.
        slot = self.new_temp()
        self.operations.append(FloatStore(slot, value))
        ok = self.new_label("divisor_nonzero")
        self.operations.append(
            JumpIfFalse(
                FloatCompare("eq", FloatLoad(slot), FloatConstant(0.0)), ok
            )
        )
        self.raise_exception(
            "ZeroDivisionError", b"ZeroDivisionError: float division by zero\n"
        )
        self.operations.append(Label(ok))
        return FloatLoad(slot)

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
        if isinstance(node, ast.Subscript) and self.subscript_dict_kinds(node):
            # The value sits in the entry as its bit pattern; reinterpret it.
            return BitsFloat(
                HeapLoad(self.dict_lookup_value_address(node, bindings), 8)
            )
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and self.list_kind_of(node.value.id) == "float"
        ):
            return BitsFloat(HeapLoad(self.list_element_address(node), 8))
        if (
            isinstance(node, ast.Attribute)
            and self.resolve_object_class(node.value)
            and self.attribute_kind(node) == "float"
        ):
            return BitsFloat(HeapLoad(self.attribute_address(node), 8))
        if isinstance(node, ast.Name) and node.id in bindings:
            bound = bindings[node.id]
            if isinstance(bound, FLOAT_EXPRESSIONS):
                return bound
        if self.expression_function_kind(node, bindings) == "float":
            assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            function = self.functions[node.func.id]
            arguments = self.bind_native_arguments(
                node.func.id, function, node, bindings, call_stack
            )
            previous_path, previous_values = self.path, self.values
            previous_functions = self.functions
            self.path, self.values = function.path, function.values
            self.functions = function.functions
            try:
                return self.float_expression(
                    function.expression,
                    dict(zip(function.parameters, arguments)),
                    (*call_stack, id(function)),
                )
            finally:
                self.path, self.values = previous_path, previous_values
                self.functions = previous_functions
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


    def floor_divide(
        self,
        node: ast.BinOp,
        bindings,
        call_stack,
        remainder: bool,
    ) -> IntExpression:
        """Lower Python's ``//`` or ``%`` for runtime integers.

        The hardware divide truncates toward zero, but Python floors toward
        negative infinity: -7 // 2 is -4, not -3, and -7 % 2 is 1, not -1. The
        two agree unless the remainder is non-zero and its sign differs from
        the divisor's, so the correction is computed branchlessly from exactly
        that condition. Division by zero raises in Python, so the emitted code
        reports it and exits 1 rather than trapping or returning nonsense.
        """

        left = self.materialize_int(self.integer(node.left, bindings, call_stack))
        right = self.materialize_int(self.integer(node.right, bindings, call_stack))

        ok_label = self.new_label("divide_ok")
        bad_label = self.new_label("divide_by_zero")
        self.operations.append(
            JumpIfFalse(IntCompare("eq", right, IntConstant(0)), ok_label)
        )
        self.operations.append(Label(bad_label))
        self.operations.append(
            Write(
                b"ZeroDivisionError: integer division or modulo by zero\n", 2
            )
        )
        self.operations.append(Exit(1))
        self.operations.append(Label(ok_label))

        truncated = self.materialize_int(IntBinary("sdiv", left, right))
        rest = self.materialize_int(IntBinary("smod", left, right))
        # The quotient is one too high exactly when the remainder is non-zero
        # and its sign differs from the divisor's; -(cond) is 0 or -1.
        differs = IntCompare("lt", IntBinary("xor", rest, right), IntConstant(0))
        nonzero = IntCompare("ne", rest, IntConstant(0))
        correction = self.materialize_int(
            IntBinary("and", differs, nonzero)
        )
        if remainder:
            # Add the divisor back on exactly those cases.
            return IntBinary(
                "add",
                rest,
                IntBinary("and", right, IntUnary("neg", correction)),
            )
        return IntBinary("sub", truncated, correction)

    def materialize_int(self, expression: IntExpression) -> IntExpression:
        """Pin a value in a slot so re-reading it cannot re-evaluate it."""

        if isinstance(expression, (IntConstant, IntLoad)):
            return expression
        slot = self.new_temp()
        self.operations.append(Store(slot, expression))
        return IntLoad(slot)

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
            if node.id in self.string_bindings:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"{node.id!r} was passed a string, so it cannot be used "
                    "where an integer is required",
                )
            if isinstance(value, FLOAT_EXPRESSIONS):
                raise NativeCompileError(
                    self.path,
                    node,
                    f"{node.id!r} was passed a float, so it cannot be used where "
                    "an integer is required; wrap it in int() to truncate",
                )
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
            if self.list_kind(kind) is not None:
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
            and node.func.id == "abs"
            and node.func.id not in self.functions
        ):
            if len(node.args) != 1 or node.keywords:
                raise NativeCompileError(
                    self.path, node, "native abs() takes exactly one argument"
                )
            value = self.materialize_int(
                self.integer(node.args[0], bindings, call_stack)
            )
            return self.select_integer(
                IntCompare("lt", value, IntConstant(0)),
                IntUnary("neg", value),
                value,
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self._AGGREGATES
            and node.func.id not in self.functions
        ):
            return self.aggregate_call(node, bindings, call_stack)
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
                return self.emit_code_point_count(self.string_pointer(argument))
            if (
                isinstance(argument, ast.Name)
                and self.list_kind_of(argument.id) is not None
            ):
                return HeapLoad(
                    IntBinary(
                        "add", IntLoad(self.slots[argument.id]), IntConstant(8)
                    ),
                    8,
                )
            if isinstance(argument, ast.Subscript) and isinstance(
                argument.slice, ast.Slice
            ):
                return HeapLoad(
                    IntBinary("add", self.list_pointer(argument), IntConstant(8)), 8
                )
            if isinstance(argument, ast.Name) and self.dict_kinds_of(argument.id):
                # The live count is the second i64 of the table header.
                return HeapLoad(
                    IntBinary("add", IntLoad(self.slots[argument.id]), IntConstant(8)),
                    8,
                )
            raise NativeCompileError(
                self.path,
                node,
                "native len() supports runtime strings and integer lists",
            )
        if isinstance(node, ast.Subscript) and self.subscript_dict_kinds(node):
            _key_kind, value_kind = self.subscript_dict_kinds(node)
            if value_kind != "int":
                raise NativeCompileError(
                    self.path, node, "this dict has float values"
                )
            return HeapLoad(self.dict_lookup_value_address(node, bindings), 8)
        if isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Name)
                and self.list_kind_of(node.value.id) == "float"
            ):
                raise NativeCompileError(
                    self.path, node, "this list holds floats"
                )
            return HeapLoad(self.list_element_address(node), 8)
        if isinstance(node, ast.Attribute) and self.resolve_object_class(node.value):
            if self.attribute_kind(node) == "float":
                raise NativeCompileError(
                    self.path, node, "this attribute holds a float"
                )
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
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                if self.eager_depth:
                    raise NativeCompileError(
                        self.path,
                        node,
                        "// and % can raise ZeroDivisionError, so they cannot "
                        "appear in a conditional expression or a "
                        "short-circuited operand, whose arms are both "
                        "evaluated here; use an if statement",
                    )
                return self.floor_divide(
                    node, bindings, call_stack, isinstance(node.op, ast.Mod)
                )
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
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
            and self.expression_type(node.left, bindings) == "str"
            and self.expression_type(node.comparators[0], bindings) == "str"
        ):
            # Equal text is equal bytes, so this is the same comparison a
            # string-keyed dict already makes when it probes.
            same = self.emit_string_equal(
                self.string_pointer(node.left),
                self.string_pointer(node.comparators[0]),
            )
            return IntCompare(
                "ne" if isinstance(node.ops[0], ast.Eq) else "eq",
                IntLoad(same),
                IntConstant(0),
            )
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and self.membership_container_kind(node.comparators[0]) is not None
        ):
            # Membership only probes, so unlike a lookup it cannot raise and is
            # safe in an eagerly evaluated arm.
            container = node.comparators[0]
            shape = self.membership_container_kind(container)
            present = isinstance(node.ops[0], ast.In)
            if shape != "dict":
                if shape == "str":
                    found = self.emit_substring_search(node.left, container)
                else:
                    found = self.emit_list_membership(node.left, container, shape)
                return IntCompare(
                    "ne" if present else "eq",
                    IntLoad(found),
                    IntConstant(0),
                )
            key_kind, _value_kind = self.dict_kinds_of(container.id)
            if self.expression_type(node.left, bindings) != key_kind:
                raise NativeCompileError(
                    self.path, node.left, f"this dict has {key_kind} keys"
                )
            _address, found_slot, _key, _state = self.dict_probe(
                self.slot(container.id),
                self.dict_key(node.left, key_kind),
                key_kind,
            )
            return IntCompare(
                "ne" if present else "eq", IntLoad(found_slot), IntConstant(0)
            )
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
            # A chain such as `a < b < c` names b once but compares it twice.
            # Lower every operand exactly once and, when a value is reused as
            # the next comparison's left side, hold it in a slot: the backends
            # re-emit an expression tree at each occurrence, so reusing the
            # tree would evaluate the operand twice.
            def lower_operand(
                item: ast.expr, as_float: bool
            ) -> IntExpression | FloatExpression:
                if as_float:
                    return self.float_expression(item, bindings, call_stack)
                return self.integer(item, bindings, call_stack)

            operand_is_float = [
                self.expression_type(item, bindings) == "float"
                for item in (node.left, *node.comparators)
            ]
            # A comparison is a float comparison when either side is a float.
            pair_is_float = [
                operand_is_float[index] or operand_is_float[index + 1]
                for index in range(len(node.ops))
            ]
            result: IntExpression | None = None
            carried: IntExpression | FloatExpression | None = None
            carried_is_float = False
            for index, (operator_node, right) in enumerate(
                zip(node.ops, node.comparators)
            ):
                operator = operators.get(type(operator_node))
                if operator is None:
                    raise NativeCompileError(
                        self.path,
                        node,
                        "unsupported native integer comparison",
                    )
                as_float = pair_is_float[index]
                if index == 0:
                    left_value = lower_operand(node.left, as_float)
                elif as_float and not carried_is_float:
                    left_value = IntToFloat(carried)
                else:
                    left_value = carried
                # Python evaluates the rest of a chain only if the comparisons
                # so far were true. This lowering is branchless, so anything
                # that could trap or have an effect must be rejected there.
                if index:
                    self.eager_depth += 1
                try:
                    right_value = lower_operand(right, as_float)
                finally:
                    if index:
                        self.eager_depth -= 1
                if index + 1 < len(node.ops):
                    # Reused as the next left operand: evaluate it once.
                    slot = self.new_temp()
                    if as_float:
                        self.operations.append(FloatStore(slot, right_value))
                        right_value = FloatLoad(slot)
                    else:
                        self.operations.append(Store(slot, right_value))
                        right_value = IntLoad(slot)
                    carried, carried_is_float = right_value, as_float
                comparison = (
                    FloatCompare(operator, left_value, right_value)
                    if as_float
                    else IntCompare(operator, left_value, right_value)
                )
                result = (
                    comparison
                    if result is None
                    else IntBinary("and", result, comparison)
                )
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
            argument_kinds: list[str] = []
            arguments = self.bind_native_arguments(
                node.func.id,
                function,
                node,
                bindings,
                call_stack,
                kinds=argument_kinds,
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
                    argument_kinds=tuple(argument_kinds),
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
            previous_strings = self.string_bindings
            self.string_bindings = {
                **previous_strings,
                **{
                    parameter: value
                    for parameter, value, kind in zip(
                        function.parameters, arguments, argument_kinds
                    )
                    if kind == "str"
                },
            }
            try:
                return self.integer(
                    function.expression,
                    dict(zip(function.parameters, arguments)),
                    (*call_stack, identity),
                )
            finally:
                self.string_bindings = previous_strings
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
