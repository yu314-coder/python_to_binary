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
    Function,
    FunctionAddress,
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
    is_float_expression,
    Jump,
    JumpIfFalse,
    Label,
    Module,
    Return,
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
#   "f64"  -- a C ``double``, passed in a SIMD&FP argument register. AAPCS64
#             numbers those from their own counter, so a double never consumes
#             an integer register and vice versa. Only fixed-arity callees may
#             take one: Apple's arm64 variadic ABI puts every variadic
#             argument on the stack, which is the opposite rule, so "f64" must
#             never appear in a "cfmt" signature.
#   "imp"  -- an Objective-C method implementation: the ENTRY ADDRESS of a
#             module-level Python ``def`` in this image, which the runtime will
#             later call on its own stack. The argument must be a bare name
#             referring to such a def, and the def is lowered into a real
#             ``Function`` with its own frame rather than inlined. It occupies
#             an integer register like "ptr". A signature carrying one must
#             place a "cstr" immediately after it: that string is the method
#             type encoding, and it is the only statement of what the runtime
#             will put in the callee's registers.
#   "bool" -- a C ``BOOL``, which is ONE BYTE. Handing such a callee an
#             out-of-range word is undefined -- it may test the whole register,
#             mask the low bit, or read only the low byte, and those disagree
#             for 256 -- so the argument is lowered as ``value != 0`` and the
#             callee only ever receives 0 or 1. It occupies an integer register
#             like "int".
# ``_CABI_RESULTS`` records what each callee returns: "int" (a signed 64-bit
# value), "ptr" (an opaque handle), "float" (a C double) or "void" (nothing --
# using the result of such a call is rejected, because the register would hold
# garbage natively while the CPython shim would hand back a defined value).
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
    # The libm double entry points. These are the smallest honest exercise of
    # the floating-point half of AAPCS64: pow/fmod/hypot/atan2/copysign take two
    # doubles in d0-d1, and ldexp takes a double in d0 AND an int in x0, which
    # is what proves the two register files are numbered independently.
    "pow": ("pow", ("f64", "f64")),
    "fmod": ("fmod", ("f64", "f64")),
    "hypot": ("hypot", ("f64", "f64")),
    "atan2": ("atan2", ("f64", "f64")),
    "copysign": ("copysign", ("f64", "f64")),
    "ldexp": ("ldexp", ("f64", "int")),
    # The Objective-C runtime. Cocoa itself is compiled Objective-C shipped
    # inside macOS with no source to translate, but the runtime that dispatches
    # to it is a plain C API, and these are the whole of it. objc_msgSend is
    # declared variadic and is not one - it reads its arguments from the
    # ordinary registers - so a fixed-arity binding per arity is the same cast
    # an Objective-C compiler makes before every call.
    "objc_getClass": ("objc_getClass", ("cstr",)),
    "sel_registerName": ("sel_registerName", ("cstr",)),
    "objc_msgSend": ("objc_msgSend", ("ptr", "ptr")),
    "objc_msgSend2": ("objc_msgSend", ("ptr", "ptr", "ptr")),
    "objc_msgSend_str": ("objc_msgSend", ("ptr", "ptr", "cstr")),
    # The message shapes a window and a web view need. There is one entry per
    # argument SHAPE, not per arity: a cast to the callee's prototype is a
    # claim about that prototype, and (id, SEL, NSInteger) is a different claim
    # from (id, SEL, id) even where arm64 places the two identically.
    #
    # An NSRect is four CGFloats, which AAPCS64 classifies as a homogeneous
    # floating-point aggregate and passes in four consecutive SIMD&FP
    # registers. Because the rectangle is the first floating-point argument in
    # each of these prototypes, that placement is exactly what four "f64"
    # entries produce, and the shims declare the real aggregate so the two
    # paths cannot drift.
    "objc_msgSend_id_id": ("objc_msgSend", ("ptr", "ptr", "ptr", "ptr")),
    "objc_msgSend_long": ("objc_msgSend", ("ptr", "ptr", "int")),
    "objc_msgSend_bool_void": ("objc_msgSend", ("ptr", "ptr", "bool")),
    "objc_msgSend_rect": ("objc_msgSend", ("ptr", "ptr", "f64", "f64", "f64", "f64")),
    "objc_msgSend_rect_id": (
        "objc_msgSend",
        ("ptr", "ptr", "f64", "f64", "f64", "f64", "ptr"),
    ),
    "objc_msgSend_rect_uint_uint_bool": (
        "objc_msgSend",
        ("ptr", "ptr", "f64", "f64", "f64", "f64", "int", "int", "bool"),
    ),
    # Building a class at run time. Cocoa is not driven by sending messages
    # alone: an application delegate, a window delegate and a navigation
    # delegate are all objects the framework calls BACK into, so a program that
    # cannot hand the runtime a method implementation cannot close its own
    # window. These three are the whole of that API. A class has to be
    # registered before it can be messaged, and allocating one whose name is
    # already taken answers nil -- which is silent, because every message to nil
    # returns zero.
    "objc_allocateClassPair": ("objc_allocateClassPair", ("ptr", "cstr", "int")),
    "class_addMethod": ("class_addMethod", ("ptr", "ptr", "imp", "cstr")),
    "objc_registerClassPair": ("objc_registerClassPair", ("ptr",)),
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
    "pow": "float",
    "fmod": "float",
    "hypot": "float",
    "atan2": "float",
    "copysign": "float",
    "ldexp": "float",
    "objc_getClass": "ptr",
    "sel_registerName": "ptr",
    "objc_msgSend": "ptr",
    "objc_msgSend2": "ptr",
    "objc_msgSend_str": "ptr",
    "objc_msgSend_id_id": "ptr",
    "objc_msgSend_long": "ptr",
    "objc_msgSend_bool_void": "void",
    "objc_msgSend_rect": "ptr",
    "objc_msgSend_rect_id": "ptr",
    "objc_msgSend_rect_uint_uint_bool": "ptr",
    "objc_allocateClassPair": "ptr",
    "class_addMethod": "int",
    "objc_registerClassPair": "void",
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

# Width and signedness of each callee's C result, keyed by IMPORT NAME so an
# aliased binding (objc_msgSend2 and friends all share one C symbol) can carry
# its own result shape. AAPCS64 leaves bits 32-63 of the return register
# unspecified for a 32-bit result, so the encoder must extend it. Anything
# absent here returns a full 64-bit word (long long, Py_ssize_t, size_t, or a
# pointer) and needs no extension.
_CABI_RESULT_WIDTH: dict[str, str] = {
    # POSIX: pid_t/uid_t/gid_t and int abs(int) are 32 bits.
    "getpid": "i32",
    "getppid": "i32",
    "getuid": "u32",
    "getgid": "u32",
    "abs": "i32",
    # class_addMethod returns a C ``BOOL``, which on arm64 macOS is C99 _Bool
    # and therefore ONE BYTE: AAPCS64 leaves bits 8-63 of the result register
    # unspecified, so the byte has to be isolated before the value is compared
    # against anything. The CPython shim declares c_bool and hands back 0 or 1,
    # so without this the two runs disagree exactly when the register happens
    # to carry dirt from the call.
    "class_addMethod": "u8",
    # The double-returning libm entries. "f64" is not a width but a different
    # register file: the value comes back in d0, not x0.
    "pow": "f64",
    "fmod": "f64",
    "hypot": "f64",
    "atan2": "f64",
    "copysign": "f64",
    "ldexp": "f64",
    # CPython entry points declared to return C int. Each uses -1 for failure,
    # which is exactly the case a missing sign extension destroys.
    "Py_IsInitialized": "i32",
    "PyRun_SimpleString": "i32",
    "PyObject_IsTrue": "i32",
    "PyList_Append": "i32",
    "PyFile_WriteObject": "i32",
    "PyFile_WriteString": "i32",
}


# The arm64 encoder passes extern arguments in x0-x7 and d0-d7 (AAPCS64) and has
# no stack-argument path, so a signature that overflows either file must never
# reach it. The two counters are independent, which is why the budget is checked
# per file rather than against the total argument count: a nine-argument call is
# perfectly legal when four of those arguments are doubles.
_CABI_MAX_ARGUMENTS = 8


def _register_demand(signature: tuple[str, ...]) -> tuple[int, int]:
    """The (integer, floating-point) argument registers ``signature`` consumes."""

    floats = sum(1 for kind in signature if kind == "f64")
    return len(signature) - floats, floats


assert all(
    max(_register_demand(signature)) <= _CABI_MAX_ARGUMENTS
    for _symbol, signature in _CABI_SYMBOLS.values()
), "an adapter-ABI signature exceeds the register argument budget"
assert not any(
    "f64" in signature and "cfmt" in signature
    for _symbol, signature in _CABI_SYMBOLS.values()
), "a variadic callee cannot take a double: Apple's arm64 ABI stacks those"
assert all(
    (_CABI_RESULT_WIDTH.get(name) == "f64") == (kind == "float")
    for name, kind in _CABI_RESULTS.items()
), "a float-returning extern must declare the f64 result width"
assert all(
    signature[position + 1 :][:1] == ("cstr",)
    for _symbol, signature in _CABI_SYMBOLS.values()
    for position, kind in enumerate(signature)
    if kind == "imp"
), "an 'imp' argument must be followed by the 'cstr' that encodes its method type"


# --- Objective-C method implementations --------------------------------------
#
# A method type encoding is a string whose first character encodes the result
# and whose remaining characters encode the arguments, the first two of which
# are always the receiver (``@``) and the selector (``:``). It is not
# decoration: it is the only statement anywhere of what the runtime will put in
# the callee's registers, and AppKit reads it back through
# methodSignatureForSelector: whenever it forwards or observes a message.
#
# Only these codes are accepted, and the reason is the calling convention
# rather than taste. Everything here arrives in an ordinary integer register,
# which is the only place a compiled ``Function`` prologue looks. A "d" or "f"
# argument arrives in d0-d7 and a "{...}" struct may arrive in the SIMD&FP
# registers or through the x8 indirect-result pointer; a body that read those
# positions as words would get whatever the caller last left in x2, and it
# would do it silently. Those are rejected rather than approximated.
_IMP_RESULT_CODES: dict[str, str] = {
    "v": "void",   # no result; the result register is left undefined
    "q": "int",    # long long, returned whole in x0
    "@": "ptr",    # an object pointer, likewise whole
    "B": "bool",   # a one-byte BOOL, so the value is normalised to 0 or 1
}
#: Argument codes an implementation may take. ``:`` is only ever the second one.
_IMP_ARGUMENT_CODES = frozenset("@q:")


def parse_method_encoding(encoding: str) -> tuple[str, tuple[str, ...]]:
    """Split a vetted method type encoding into ``(result, arguments)``.

    Raises :class:`ValueError` describing the first code that py2bin cannot
    deliver through the integer registers a compiled ``Function`` reads.
    """

    if not encoding:
        raise ValueError("a method type encoding cannot be empty")
    result, arguments = encoding[0], tuple(encoding[1:])
    if result not in _IMP_RESULT_CODES:
        raise ValueError(
            f"method type encoding {encoding!r} returns {result!r}, and py2bin "
            f"can only return {', '.join(sorted(_IMP_RESULT_CODES))} from a "
            "callback: a floating-point result comes back in d0 and a struct "
            "result through the x8 indirect-result register, neither of which "
            "a compiled function body writes"
        )
    if arguments[:2] != ("@", ":"):
        raise ValueError(
            f"method type encoding {encoding!r} must begin its arguments with "
            "'@:', the receiver and the selector every Objective-C method is "
            "called with"
        )
    for code in arguments[2:]:
        if code not in _IMP_ARGUMENT_CODES or code == ":":
            raise ValueError(
                f"method type encoding {encoding!r} takes an argument of type "
                f"{code!r}, which py2bin cannot receive: only '@' (an object) "
                "and 'q' (a long long) arrive in the integer registers a "
                "compiled function reads. A 'd'/'f' argument arrives in d0-d7, "
                "a '{...}' struct may arrive there or through x8, and a 'B' is "
                "one byte whose register's upper bits are undefined"
            )
    return _IMP_RESULT_CODES[result], arguments


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


class NotConstant(NativeCompileError):
    """The catch-all "this simply is not a constant" rejection.

    Every runtime expression raises it, so it says nothing about what is wrong
    and must never be what a caller reports after also trying a runtime path.
    A subclass rather than a flag, so that every existing
    ``except NativeCompileError`` still catches it and no control flow moves.
    """


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
# Prefer ``is_float_expression`` over a bare isinstance against this tuple: an
# ExternCall is float-valued or not depending on its declared result, and a
# class test alone would silently route a returned double through the integer
# register file.
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
    # A subclass repeats its base's fields first, in the base's own order, so
    # one inherited method body reads the same offsets on either class.
    fields: tuple[str, ...]
    initializer: NativeFunction | None
    # Already merged with the base's methods at definition time, the subclass
    # winning, so no lookup anywhere else has to walk a chain.
    methods: dict[str, NativeFunction]
    # A field is an integer unless __init__ annotates it `float`. The slot is
    # eight bytes either way, so a float lives there as its bit pattern; the
    # annotation is how the layout learns which it is, since the type of the
    # value assigned there depends on the arguments at each call site.
    field_kinds: dict[str, str] = dataclass_field(default_factory=dict)
    base: str | None = None

    @property
    def size(self) -> int:
        return max(len(self.fields) * 8, 8)

    def offset(self, field: str) -> int:
        return self.fields.index(field) * 8


def method_display_name(owner: str, method: str) -> str:
    """How a method should be named in a message the user reads.

    The rewritten `super().__init__` has a private key that appears nowhere in
    the source, so reporting it verbatim would name something the user never
    wrote. Show it as what they did write.
    """

    if method.startswith("<") and method.endswith(".__init__>"):
        return "super().__init__"
    return f"{owner}.{method}"


def super_init_key(base: str) -> str:
    """The private method name a rewritten ``super().__init__`` calls.

    The angle brackets keep it out of the identifier space a program can
    write, so it can never collide with a real method or be called directly.
    """

    return f"<{base}.__init__>"


def super_init_call(statement: ast.stmt) -> ast.Call | None:
    """``super().__init__(...)`` used as a bare statement, or None."""

    if not isinstance(statement, ast.Expr):
        return None
    call = statement.value
    if not isinstance(call, ast.Call):
        return None
    attribute = call.func
    if not (isinstance(attribute, ast.Attribute) and attribute.attr == "__init__"):
        return None
    inner = attribute.value
    if not (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == "super"
        and not inner.args
        and not inner.keywords
    ):
        return None
    return call


_FORMAT_ALIGNMENTS = "<>=^"
_FORMAT_DIGITS = "0123456789"
# Fixed-point digits are generated into a fixed scratch buffer that is written
# without a bound check, and the padding of one field is allocated out of the
# arena, so both are capped here rather than trusted.
_FORMAT_MAX_PRECISION = 100
_FORMAT_MAX_WIDTH = 1000
_FORMAT_SUPPORTED = (
    "a native f-string format specifier is "
    "[[fill]align][sign][0][width][,][.precision][type] with align one of "
    "<>=^, type one of d, f, s or omitted, width up to "
    f"{_FORMAT_MAX_WIDTH} and precision up to {_FORMAT_MAX_PRECISION}"
)


@dataclass(frozen=True, slots=True)
class FormatSpec:
    """A parsed and kind-checked ``format()`` mini-language specifier."""

    fill: bytes
    align: str
    sign: str
    width: int
    grouping: bool
    precision: int | None
    type: str


def parse_format_spec(text: str, kind: str) -> FormatSpec:
    """Parse a literal format specifier, or raise ValueError explaining why not.

    Only the part of the mini-language the native renderers reproduce exactly
    is accepted. Everything else is refused here rather than approximated,
    because a format that is close is indistinguishable from one that is right
    until someone reads the output.
    """

    index = 0
    fill = ""
    align = ""
    if len(text) >= 2 and text[1] in _FORMAT_ALIGNMENTS:
        fill, align, index = text[0], text[1], 2
    elif text and text[0] in _FORMAT_ALIGNMENTS:
        align, index = text[0], 1
    sign = "-"
    signed = text[index : index + 1] in ("+", "-", " ")
    if signed:
        sign, index = text[index], index + 1
    if text[index : index + 1] in ("z", "#"):
        raise ValueError(
            f"the {text[index]!r} flag is not supported; {_FORMAT_SUPPORTED}"
        )
    zero = text[index : index + 1] == "0"
    if zero:
        index += 1
    start = index
    while index < len(text) and text[index] in _FORMAT_DIGITS:
        index += 1
    width = int(text[start:index]) if index > start else 0
    grouping = False
    if text[index : index + 1] == "_":
        raise ValueError(f"the '_' separator is not supported; {_FORMAT_SUPPORTED}")
    if text[index : index + 1] == ",":
        grouping, index = True, index + 1
    precision: int | None = None
    if text[index : index + 1] == ".":
        index += 1
        start = index
        while index < len(text) and text[index] in _FORMAT_DIGITS:
            index += 1
        if index == start:
            raise ValueError(f"the precision has no digits; {_FORMAT_SUPPORTED}")
        precision = int(text[start:index])
    type_code = text[index:]
    if len(type_code) > 1 or (type_code and type_code not in "dfs"):
        raise ValueError(
            f"format type {type_code!r} is not supported; {_FORMAT_SUPPORTED}"
        )

    allowed = {"str": ("", "s"), "int": ("", "d", "f"), "float": ("", "f")}[kind]
    if type_code not in allowed:
        raise ValueError(
            f"format type {type_code!r} is not supported for "
            f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}; "
            f"{_FORMAT_SUPPORTED}"
        )
    if kind == "str":
        if signed:
            raise ValueError("a sign is not allowed in a string format specifier")
        if align == "=" or zero:
            raise ValueError(
                "'=' alignment, and the zero flag that implies it, are not "
                "allowed in a string format specifier"
            )
        if grouping:
            raise ValueError(
                "a thousands separator is not allowed in a string format specifier"
            )
        if precision is not None:
            raise ValueError(
                "a precision truncates a string, which is not supported; "
                f"{_FORMAT_SUPPORTED}"
            )
    else:
        if precision is not None and type_code != "f":
            raise ValueError(
                f"a precision is only supported with format type 'f'; "
                f"{_FORMAT_SUPPORTED}"
            )
        if grouping and (kind != "int" or type_code == "f"):
            raise ValueError(
                "a thousands separator is only supported for an integer "
                f"rendered as an integer; {_FORMAT_SUPPORTED}"
            )
        if grouping and (zero or align == "="):
            # The padding zeros themselves get separators, which needs a
            # digit-position calculation the renderer does not do.
            raise ValueError(
                "a thousands separator combined with zero or '=' padding is "
                f"not supported; {_FORMAT_SUPPORTED}"
            )
    if type_code == "f" and precision is None:
        precision = 6
    if precision is not None and precision > _FORMAT_MAX_PRECISION:
        raise ValueError(
            f"a precision above {_FORMAT_MAX_PRECISION} is not supported"
        )
    if width > _FORMAT_MAX_WIDTH:
        raise ValueError(f"a width above {_FORMAT_MAX_WIDTH} is not supported")

    if not fill:
        # A bare zero flag pads with zeros between the sign and the digits, but
        # an explicit fill or alignment keeps its own meaning and only the
        # width survives.
        fill = "0" if zero else " "
        if not align and zero:
            align = "="
    if not align:
        align = "<" if kind == "str" else ">"
    return FormatSpec(
        fill=fill.encode("utf-8"),
        align=align,
        sign=sign,
        width=width,
        grouping=grouping,
        precision=precision,
        type=type_code,
    )


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


@dataclass(slots=True)
class ComprehensionSource:
    """One `for` clause of a comprehension, already measured.

    Every clause's source is evaluated once, before any loop runs, so `span`
    is a fact by the time the reserve is computed and stays one while the
    loops run.
    """

    target: str
    conditions: list[ast.expr]
    index_slot: int
    start: IntExpression
    limit_slot: int
    span: IntExpression
    element_kind: str
    pointer_slot: int | None
    holds_bool: bool | None


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
        # How many runtime `if` arms enclose the statement being lowered. A
        # `def` inside one cannot be chosen at run time, because a call is
        # inlined at its site from a single body.
        self.runtime_branch_depth = 0
        # Names the print() lowering binds to hold an argument it has already
        # evaluated. Numbered so that two prints never share one.
        self.print_argument_count = 0
        # Names that have definitely been assigned at this point.
        self.bound_names: set[str] = set()
        # Runtime name -> "int" | "float". A stack slot is 8 bytes and holds
        # either a signed 64-bit integer or an IEEE-754 double; this records
        # which, so the correct load/store and register file is used.
        self.value_types: dict[str, str] = {}
        # Known compile-time lengths of runtime lists built from literals, keyed
        # by variable name. Used only to reject out-of-range constant indices.
        self.list_lengths: dict[str, int] = {}
        # Names bound to a lambda, and the source position of every lambda that
        # is such a binding. Positions rather than identities because a function
        # body is deep-copied at definition, so the lambda lowered at a call
        # site is not the node the whole-tree pass looked at.
        self.lambda_names: set[str] = set()
        self._lambda_bindings: set[tuple[int, int]] = set()
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
        # Whether a container's elements are bools. A container has no place to
        # keep that at run time - the slot holds a number either way - so it is
        # a property of the variable, decided by everything stored into it.
        # None means nothing has been stored yet.
        self.container_bool: dict[str, bool | None] = {}
        # An empty list literal with no annotation has nothing to read its
        # element kind from, so it waits for the first thing stored in it
        # rather than guessing integers and refusing anything else.
        self.undecided_lists: set[str] = set()
        self._bool_query: set[int] = set()
        self._kind_query: set[int] = set()
        # Per tuple name, the bytes print() must write for each element whose
        # repr is settled at build time, and None where it is not. CPython
        # prints a tuple with the repr of every element, and the quotes and
        # backslash escapes repr picks for a string built at run time are not
        # something the string machinery here can reproduce.
        self.tuple_texts: dict[str, tuple[bytes | None, ...]] = {}
        # Names whose list block is also reachable through a container: one
        # that was stored into a list anywhere in the module, and one that was
        # read back out of one. Appending moves a block, and only the name it
        # moves through learns the new address.
        self.escaped_list_names: set[str] = set()
        self.shared_list_names: set[str] = set()
        # A parameter substituted into a single-expression body is just a
        # value, and a string's value is a pointer - indistinguishable from an
        # integer. This records which of them are strings.
        self.string_bindings: dict[str, IntExpression] = {}
        # The numeric parameters of the function being inlined, held the same
        # way its string parameters are. Without these a rendering call such as
        # str(n) inside a string-returning function saw n as nothing at all.
        self.value_bindings: dict[str, KernelValue] = {}
        self.exception_ids: dict[str, int] = {}
        # Each cleanup scope (a `finally`, or a `with`'s `__exit__`) records
        # how deep the jump stacks were when it opened. A jump is only a
        # problem when it would leave the scope: one inside a function or loop
        # that opened later stays within it, which is what makes a `return self`
        # in an inlined __enter__ harmless.
        self.finally_scopes: list[tuple[int, int, int]] = []
        self.continue_targets: list[str] = []
        # The list names a `for` is walking right now, innermost last. A walk
        # takes the length once and counts up, so a body that shortens the list
        # would run off the end of it.
        self.iterated_lists: list[str] = []
        self.return_targets: list[tuple[int | None, str]] = []
        # Whether the innermost inlined function's result slot holds the
        # address of a string block rather than a number. Decided by the call
        # site before the body runs, because it is the call site that has to
        # read the slot back.
        self.returns_string = False
        self.active_functions: list[tuple[int, str]] = []
        # Python defs handed to the Objective-C runtime as method
        # implementations, keyed by (def name, method type encoding). Each is a
        # real ``Function`` with its own frame, because the runtime calls it and
        # there is no call site here to inline it into.
        self._callback_functions: dict[tuple[str, str], Function] = {}
        # How many places in the whole module give each name a value. A
        # callback is lowered where it is registered but runs at a time nobody
        # here can name, so a module-level value it reads is only safe to bake
        # into it when nothing can ever rebind that name.
        self._binding_counts: dict[str, int] = {}

    def compile(self, source: str) -> Module:
        try:
            tree = ast.parse(source, filename=str(self.path))
        except SyntaxError as error:
            raise ValueError(f"{self.path}:{error.lineno}:{error.offset}: {error.msg}") from error
        self._binding_counts = {
            name: len(sites)
            for name, sites in self.name_binding_sites(tree).items()
        }
        self.runtime_names.update(self.loop_mutated_names(tree))
        self.note_escaping_list_names(tree)
        # A name a function declares global has to live in a slot. Inlining
        # swaps the build-time constant map for the function's own, so a
        # constant written inside the body would be dropped when the module's
        # map came back.
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                self.runtime_names.update(node.names)
        self.note_lambda_bindings(tree)
        for statement in tree.body:
            self.statement(statement)
        if not self.operations or not isinstance(self.operations[-1], (Exit, ExitValue)):
            self.operations.append(Exit(0))
        if self._heap_bump_slot is not None:
            self.guard_arena_limit()
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
        return Module(
            self.operations,
            len(self.slots),
            functions=list(self._callback_functions.values()),
        )

    def guard_arena_limit(self) -> None:
        """Check every allocation against the end of the arena.

        The arena is one fixed reservation and the bump pointer only moves
        forward, so running past the end is not a failed allocation - it is a
        write to memory the process never asked for, which is a segmentation
        fault at best and silent corruption at worst. The check is added here,
        once, rather than at the sixteen places that allocate.
        """

        assert self._heap_bump_slot is not None
        end_slot = self.slot("<heap-end>")
        guarded: list[object] = []
        checks = 0
        for operation in self.operations:
            guarded.append(operation)
            if not isinstance(operation, HeapAlloc):
                continue
            checks += 1
            ok = f"arena_ok_{checks}"
            guarded.append(
                JumpIfFalse(
                    IntCompare(
                        "gt",
                        IntLoad(self._heap_bump_slot),
                        IntLoad(end_slot),
                    ),
                    ok,
                )
            )
            guarded.append(
                Write(b"MemoryError: native arena exhausted\n", 2)
            )
            guarded.append(Exit(1))
            guarded.append(Label(ok))
        self.operations[:] = guarded
        self.operations.insert(
            0,
            Store(
                end_slot,
                IntBinary(
                    "add",
                    IntLoad(self._heap_bump_slot),
                    IntConstant(_HEAP_ARENA_BYTES),
                ),
            ),
        )

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
    def name_binding_sites(cls, tree: ast.AST) -> dict[str, list[ast.AST]]:
        """Every place each name is given a value, with the node that does it.

        Broader than `assigned_names`, which only cares about what a block
        rewrites: this has to see every way a name could come to mean something
        else, so a name may be shown to mean one thing everywhere.
        """

        sites: dict[str, list[ast.AST]] = {}

        def note(name: str, node: ast.AST) -> None:
            sites.setdefault(name, []).append(node)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in cls.target_names(target):
                        note(name, node)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(node.target, ast.Name):
                    note(node.target.id, node)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                for name in cls.target_names(node.target):
                    note(name, node)
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                note(node.name, node)
            elif isinstance(node, ast.arg):
                # A parameter shadows the outer name for the whole body.
                note(node.arg, node)
            elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                note(node.target.id, node)
            elif isinstance(node, ast.comprehension):
                for name in cls.target_names(node.target):
                    note(name, node.target)
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                for name in cls.target_names(node.optional_vars):
                    note(name, node.optional_vars)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                note(node.name, node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    note(alias.asname or alias.name.split(".")[0], node)
        return sites

    _LAMBDA_REFUSAL = (
        "a native lambda is compiled as the one-expression function it would "
        "otherwise be written as, under the name it is bound to, and a call is "
        "resolved to that function at build time. No value here carries code, "
        "so the only lambda with a representation is `name = lambda ...:` "
        "written directly in a module or function body - not one passed as an "
        "argument, returned, stored in a container, or called where it stands"
    )

    def note_lambda_bindings(self, tree: ast.Module) -> None:
        """Accept the lambdas that are plain function definitions in disguise."""

        bodies: list[list[ast.stmt]] = [tree.body]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bodies.append(node.body)
        bindings: dict[str, ast.Assign] = {}
        for body in bodies:
            for statement in body:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and isinstance(statement.value, ast.Lambda)
                ):
                    bound = statement.value
                    self._lambda_bindings.add((bound.lineno, bound.col_offset))
                    # Keep the first, so a rebinding is reported as the second
                    # thing that happened rather than the first.
                    bindings.setdefault(statement.targets[0].id, statement)
        # A lambda spelled as a keyword argument is left alone here: whatever
        # rejects that keyword knows why its own callee cannot take a callable,
        # which is a better answer than this general one. `integer()` refuses
        # it by name if nothing more specific speaks first.
        keyword_values = {
            id(node.value) for node in ast.walk(tree) if isinstance(node, ast.keyword)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Lambda)
                and (node.lineno, node.col_offset) not in self._lambda_bindings
                and id(node) not in keyword_values
            ):
                raise NativeCompileError(self.path, node, self._LAMBDA_REFUSAL)
        if not bindings:
            return
        sites = self.name_binding_sites(tree)
        for name, statement in bindings.items():
            others = sorted(
                (site for site in sites.get(name, ()) if site is not statement),
                key=lambda site: (site.lineno, site.col_offset),
            )
            if others:
                raise NativeCompileError(
                    self.path,
                    others[0],
                    f"{name!r} is bound to a lambda on line {statement.lineno} and "
                    f"bound again on line {others[0].lineno}; a call to it is "
                    "resolved to one compiled function at build time, so which "
                    "code the name means cannot depend on what ran. Give the "
                    "other binding a name of its own",
                )
        self.lambda_names.update(bindings)
        called = {
            id(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and node.id in self.lambda_names
                and isinstance(node.ctx, ast.Load)
                and id(node) not in called
            ):
                raise NativeCompileError(
                    self.path,
                    node,
                    f"{node.id!r} names a lambda compiled as a native function, "
                    "so it can only be called; there is no run-time value here "
                    "for a name to hold that carries code",
                )

    @staticmethod
    def block_breaks(body: list[ast.stmt]) -> bool:
        """Whether a `break` in this block belongs to the loop that owns it.

        A break inside a nested loop is that loop's, and a break inside a
        nested function is not a break out of anything here, so neither says
        the owning loop can skip its `else`.
        """

        found = False

        class BreakVisitor(ast.NodeVisitor):
            def visit_Break(self, node: ast.Break) -> None:
                nonlocal found
                found = True

            def visit_For(self, node: ast.For) -> None:
                # The nested loop owns its own breaks, but its `else` body is
                # still part of this block.
                for statement in node.orelse:
                    self.visit(statement)

            def visit_AsyncFor(self, node) -> None:
                return

            def visit_While(self, node: ast.While) -> None:
                for statement in node.orelse:
                    self.visit(statement)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, node) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

        visitor = BreakVisitor()
        for statement in body:
            visitor.visit(statement)
        return found

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

    def container_length(
        self, node: ast.expr, bindings: dict[str, KernelValue] | None = None
    ) -> IntExpression | None:
        """How many items ``node`` holds, or None when it is not a container.

        A string answers with its byte count rather than its code-point count.
        Both are zero together, and the truth of a string is the only thing
        this is asked for, so the cheaper one is enough.
        """

        try:
            kind = self.expression_type(node, bindings)
        except NativeCompileError:
            return None
        if kind == "str":
            return HeapLoad(self.string_pointer(node), 8)
        if self.list_kind(kind) is not None:
            return HeapLoad(
                IntBinary("add", self.list_pointer(node), IntConstant(8)), 8
            )
        if isinstance(node, ast.Name) and (
            self.dict_kinds_of(node.id) is not None
            or self.set_kind_of(node.id) is not None
        ):
            # The live count is the second word of the table header.
            return HeapLoad(
                IntBinary("add", IntLoad(self.slots[node.id]), IntConstant(8)), 8
            )
        tuple_kinds = self.tuple_kinds(kind)
        if tuple_kinds is not None:
            # A tuple's length is fixed when it is written.
            return IntConstant(len(tuple_kinds))
        return None

    def truth_value(
        self, node: ast.expr, bindings: dict[str, KernelValue] | None = None
    ) -> IntExpression:
        """``node`` as a condition: a container is true when it is not empty.

        Only where a truth value is what is wanted - a test, a `not`, a
        `bool()`. Everywhere else a container is still refused in an integer
        context, because a block's address is not a number and reading it as
        one is the mistake this compiler exists to prevent.
        """

        length = self.container_length(node, bindings)
        if length is not None:
            return IntCompare("ne", length, IntConstant(0))
        if isinstance(node, ast.BoolOp) and node.values:
            # As a condition, `a and b` is true when both are - which is a
            # question about each operand's truth and not about its value, so
            # the operands that could not be lowered as numbers still work
            # here. The value form is still refused: `xs and ys` answers with
            # one of the two, and one slot cannot hold either kind.
            parts = [self.truth_value(node.values[0], bindings)]
            self.eager_depth += 1
            try:
                parts.extend(
                    self.truth_value(value, bindings) for value in node.values[1:]
                )
            finally:
                self.eager_depth -= 1
            # Each part is reduced to 0 or 1 before they are combined. The
            # operators here are bitwise, and an operand that answered with its
            # own value rather than a truth - an integer does - would make
            # `s and n` with n == 2 come out as 1 & 2, which is false.
            combined = IntCompare("ne", parts[0], IntConstant(0))
            for part in parts[1:]:
                combined = IntBinary(
                    "and" if isinstance(node.op, ast.And) else "or",
                    combined,
                    IntCompare("ne", part, IntConstant(0)),
                )
            return combined
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return IntCompare(
                "eq", self.truth_value(node.operand, bindings), IntConstant(0)
            )
        try:
            if self.expression_type(node, bindings) == "object":
                # An instance with no __bool__ and no __len__ is always true,
                # and neither is in the subset. Said outright rather than left
                # to fall through to the integer path, where the answer would
                # be right only because a live block's address is never zero.
                return IntConstant(1)
        except NativeCompileError:
            pass
        try:
            if self.expression_type(node, bindings) == "float":
                # Every float but zero is true, and NaN with it, which is what
                # comparing against zero says.
                return FloatCompare(
                    "ne", self.float_expression(node, bindings), FloatConstant(0.0)
                )
        except NativeCompileError:
            pass
        return self.integer(node, bindings or {})

    def refuse_unbound(self, name: str, node: ast.AST | None = None) -> None:
        """Refuse a read of a name that may not have been bound.

        CPython raises NameError here. There is no run-time bit saying whether
        a slot was written, so this is decided at build time: the slot holds
        whatever preceded it, which for an integer is a stale number and for
        anything on the heap is an address that is not a block - a dict read
        that way probed for a key forever instead of answering.
        """

        if name in self.possibly_unbound:
            raise NativeCompileError(
                self.path,
                node,
                f"{name!r} may be unbound here: a path reaching this point "
                "does not bind it - a loop that runs zero times, a break that "
                "skips the else body, or a branch that binds it while another "
                "does not; CPython raises NameError, and the native slot would "
                "hold an unrelated value",
            )

    def slot(self, name: str) -> int:
        if name not in self.slots:
            self.slots[name] = len(self.slots)
        return self.slots[name]

    def statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return  # Module docstring.
            self.expression_statement(node.value)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Lambda)
        ):
            self.lambda_definition(node.targets[0].id, node.value)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.assignment(node.targets[0].id, node.value)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and isinstance(node.value, (ast.Tuple, ast.List))
        ):
            self.parallel_assignment(node.targets[0], node.value)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and self.is_divmod_call(node.value)
        ):
            self.divmod_assignment(node.targets[0], node.value)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and self.unpacked_tuple_kinds(node.value) is not None
        ):
            self.tuple_unpacking(node.targets[0], node.value)
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
            if self.set_kind_of(node.target.id) is not None or (
                isinstance(node.value, ast.Name)
                and self.set_kind_of(node.value.id) is not None
            ):
                # Checked before the integer operator table below, which would
                # otherwise read `s |= t` as bitwise-or over the two slots and
                # write the or of two table addresses into the variable.
                if type(node.op) not in self._SET_OPERATORS:
                    raise NativeCompileError(
                        self.path,
                        node,
                        "native set augmented assignment supports |= &= -=",
                    )
                combined = ast.copy_location(
                    ast.BinOp(left=node.target, op=node.op, right=node.value),
                    node,
                )
                kind = self.expression_type(combined)
                self.values.pop(node.target.id, None)
                self.set_assignment(
                    node.target.id, combined, self.set_kind(kind)
                )
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
            self.forget_conditional_list_lengths(node.body + node.orelse)
        elif isinstance(node, ast.For):
            self.for_statement(node)
            self.forget_conditional_list_lengths(node.body + node.orelse)
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
                    "a native function with a loop or an early return cannot "
                    "return a float: this body is inlined statement by "
                    "statement, and the call site had to choose an integer or "
                    "float lowering before that started. A body that is one "
                    "expression, or a chain of ifs that all end in a return, "
                    "is folded into a conditional expression instead and can "
                    "return a float.",
                )
            if self.expression_type(node.value) == "str":
                # The slot holds the block's address. What makes this safe is
                # that the call site already asked what this body returns and
                # got the same answer, so it is reading the slot as a string.
                if not self.returns_string:
                    raise NativeCompileError(
                        self.path,
                        node,
                        "this native function returns a string on one path and "
                        "a number on another, so the call site cannot know "
                        "which it is holding; return the same kind everywhere",
                    )
                self.operations.append(
                    Store(result_slot, self.string_pointer(node.value))
                )
                self.operations.append(Jump(return_label))
                return
            if self.returns_string:
                raise NativeCompileError(
                    self.path,
                    node,
                    "this native function returns a string on one path and a "
                    "number on another, so the call site cannot know which it "
                    "is holding; return the same kind everywhere",
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
        elif isinstance(node, ast.Delete):
            self.delete_statement(node)
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
            conditional: list[ast.stmt] = list(node.body)
            for clause in node.handlers:
                conditional.extend(clause.body)
            conditional.extend(node.orelse)
            self.forget_conditional_list_lengths(conditional)
        else:
            raise NativeCompileError(
                self.path,
                node,
                f"{type(node).__name__} is not in the native subset yet; use bundle mode for full CPython semantics",
            )

    def lambda_definition(self, name: str, node: ast.Lambda) -> None:
        """Define `name = lambda ...:` as the function it would otherwise be.

        Calls here are resolved by name and inlined at build time, so a lambda
        bound to a name is a one-expression function under that name and
        nothing else. Rewriting it into one gets the parameter, default and
        return-kind handling of `def` rather than a second copy of it.
        """

        if (node.lineno, node.col_offset) not in self._lambda_bindings:
            # A binding the whole-tree pass did not accept: inside an `if`, a
            # loop, or any block that may not run, where which lambda the name
            # means would depend on what happened.
            raise NativeCompileError(self.path, node, self._LAMBDA_REFUSAL)
        synthetic = ast.FunctionDef(
            name=name,
            args=node.args,
            body=[ast.copy_location(ast.Return(value=node.body), node)],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        if "type_params" in ast.FunctionDef._fields:
            synthetic.type_params = []
        ast.copy_location(synthetic, node)
        ast.fix_missing_locations(synthetic)
        self.function_definition(synthetic)

    def function_definition(self, node: ast.FunctionDef) -> None:
        if self.runtime_branch_depth:
            raise NativeCompileError(
                self.path,
                node,
                f"{node.name!r} is defined under a condition that is only "
                "known at run time; a call is inlined from one body chosen at "
                "build time, so the branch that ran could not decide which "
                "body that is - define the function once and branch inside it",
            )
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
            # Kept as a bool where it is one. It is an int either way for
            # arithmetic, and the difference shows only when the value is
            # printed - where True must not come out as 1.
            default_values.append(value)
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
        base = self.class_base(node)
        previous_functions = self.functions
        self.functions = dict(previous_functions)
        methods: dict[str, NativeFunction] = {}
        super_called = False
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
                prepared, called_super = self.prepare_method(statement, base)
                if statement.name == "__init__":
                    # A redefined __init__ replaces the earlier one, so the
                    # last definition is the one whose super() call counts.
                    super_called = called_super
                self.function_definition(prepared)
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
        own_initializer = methods.pop("__init__", None)
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
        fields, field_kinds = self.discover_fields(
            node, own_initializer, base, super_called
        )
        initializer = own_initializer
        if base is not None:
            if initializer is None:
                initializer = base.initializer
            if super_called and base.initializer is not None:
                methods[super_init_key(base.name)] = base.initializer
            # The subclass's own entries overwrite the base's, which is what
            # makes an override win everywhere the base's name would resolve.
            methods = {**base.methods, **methods}
        # An attribute and a method are looked up by two different paths, each
        # of which finds its own name first, so a name that is both would
        # resolve to whichever path the expression happens to take. CPython has
        # one namespace and the attribute shadows the method, making the call a
        # TypeError; refuse rather than answer from the wrong path. Inheritance
        # is what makes this an ordinary accident instead of an obvious one.
        collisions = sorted(set(fields) & {
            name for name in methods if not name.startswith("<")
        })
        if collisions:
            raise NativeCompileError(
                self.path,
                node,
                f"{node.name!r} has {collisions[0]!r} as both an attribute and "
                "a method; they are looked up by different paths here, so one "
                "name cannot be both",
            )
        self.classes[node.name] = NativeClass(
            node.name,
            self.path,
            fields,
            initializer,
            methods,
            field_kinds,
            base.name if base is not None else None,
        )

    def class_base(self, node: ast.ClassDef) -> NativeClass | None:
        """Resolve the single base class of ``node``, or None for a root class."""

        if len(node.bases) > 1:
            raise NativeCompileError(
                self.path,
                node,
                "native classes support only single inheritance",
            )
        if not node.bases:
            return None
        base = node.bases[0]
        if isinstance(base, ast.Name) and base.id == "object":
            return None
        if isinstance(base, ast.Name) and base.id in self.classes:
            return self.classes[base.id]
        name = base.id if isinstance(base, ast.Name) else ast.unparse(base)
        raise NativeCompileError(
            self.path,
            node,
            f"native base class {name!r} must be a class defined earlier in "
            "this module",
        )

    def prepare_method(
        self, statement: ast.FunctionDef, base: NativeClass | None
    ) -> tuple[ast.FunctionDef, bool]:
        """Rewrite ``super().__init__(...)`` and reject every other ``super()``.

        The base initializer is registered as an ordinary method under a
        private name, so the existing method-inlining path does all the work:
        ``self`` binds to the subclass instance, and because the base's fields
        occupy the same leading slots there, its assignments land correctly.
        Nothing else needs a notion of a base class at a call site.
        """

        prepared = copy.deepcopy(statement)
        called_super = False
        if statement.name == "__init__" and base is not None:
            body: list[ast.stmt] = []
            for inner in prepared.body:
                call = super_init_call(inner)
                if call is None:
                    body.append(inner)
                    continue
                if called_super:
                    raise NativeCompileError(
                        self.path,
                        inner,
                        "native __init__ may call super().__init__() only once",
                    )
                called_super = True
                if base.initializer is None:
                    if call.args or call.keywords:
                        raise NativeCompileError(
                            self.path,
                            inner,
                            f"{base.name} defines no __init__, so "
                            "super().__init__() accepts no arguments",
                        )
                    continue  # object.__init__(self) does nothing.
                body.append(
                    ast.copy_location(
                        ast.Expr(
                            ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="self", ctx=ast.Load()),
                                    attr=super_init_key(base.name),
                                    ctx=ast.Load(),
                                ),
                                args=call.args,
                                keywords=call.keywords,
                            )
                        ),
                        inner,
                    )
                )
            prepared.body = body or [ast.copy_location(ast.Pass(), prepared)]
            ast.fix_missing_locations(prepared)
        for inner_node in ast.walk(prepared):
            if isinstance(inner_node, ast.Name) and inner_node.id == "super":
                raise NativeCompileError(
                    self.path,
                    inner_node,
                    "native super() is supported only as a bare "
                    "super().__init__(...) statement written directly in the "
                    "__init__ of a class that has a base class",
                )
        return prepared, called_super

    def discover_fields(
        self,
        node: ast.ClassDef,
        initializer: NativeFunction | None,
        base: NativeClass | None = None,
        super_called: bool = False,
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        """Derive the instance layout from ``self.NAME = ...`` in ``__init__``.

        Every attribute must be assigned in ``__init__`` so each instance has a
        complete, statically known layout. An attribute first assigned anywhere
        else would have no reserved storage, so it is rejected rather than
        silently writing outside the object.

        ``initializer`` is the class's *own* ``__init__``. Inherited fields keep
        the base's order at the front of the layout, ahead of anything the
        subclass adds, so a base method reads the same slot on either class.
        """

        inherited = base.fields if base is not None else ()
        inherited_kinds = dict(base.field_kinds) if base is not None else {}
        if initializer is None:
            return inherited, inherited_kinds

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
        own = assigned_attributes(initializer.body)
        every = assigned_attributes(
            ast.walk(ast.Module(body=list(initializer.body), type_ignores=[]))
        )
        conditional = [
            name for name in every if name not in own and name not in inherited
        ]
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
        if not super_called:
            # Without super().__init__(), the base's body never runs, so
            # Python would leave these attributes absent; a zero-filled slot
            # would answer 0 where CPython raises AttributeError.
            missing = [name for name in inherited if name not in own]
            if missing:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"{node.name}.__init__() does not assign inherited "
                    f"attribute {missing[0]!r}; call super().__init__(...) "
                    "there or assign it directly, because a native attribute "
                    "must be assigned unconditionally",
                )
        wording = {"int": "an integer", "float": "a float"}
        for name in inherited:
            base_kind = inherited_kinds.get(name, "int")
            own_kind = kinds.get(name, base_kind)
            if own_kind != base_kind:
                assert base is not None
                raise NativeCompileError(
                    self.path,
                    node,
                    f"inherited attribute {name!r} is {wording[base_kind]} in "
                    f"{base.name} but {wording[own_kind]} in {node.name}; one "
                    "slot cannot be both",
                )
        kinds.update(inherited_kinds)
        fields = (*inherited, *(name for name in own if name not in inherited))
        if len(fields) > 1024:
            raise NativeCompileError(
                self.path, node, "native classes support at most 1024 attributes"
            )
        return fields, kinds

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

    def classes_are_related(self, first: str, second: str) -> bool:
        """True when one of the two class names inherits from the other."""

        def ancestry(name: str) -> set[str]:
            seen: set[str] = set()
            current: str | None = name
            while current is not None and current not in seen:
                seen.add(current)
                found = self.classes.get(current)
                current = found.base if found is not None else None
            return seen

        return second in ancestry(first) or first in ancestry(second)

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
            detail = ""
            if self.classes_are_related(previous, native_class.name):
                # Assigning a subclass to a name already holding its base is
                # the one case Python allows, and it is exactly the case a
                # vtable would be needed for: there is none, so a method call
                # would resolve to whichever class the name was declared with.
                detail = (
                    "; a base and its subclass are still different classes "
                    "here, because native method calls resolve at build time "
                    "from the class of the variable"
                )
            raise NativeCompileError(
                self.path,
                node,
                f"native object variable {name!r} cannot change class from "
                f"{previous} to {native_class.name}{detail}",
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
        native_class = self.resolve_object_class(target.value)
        if native_class is not None and self.attribute_kind(target) != "float":
            self.note_stored_bool(
                f"{native_class.name}.{target.attr}",
                value,
                f"{native_class.name}.{target.attr}",
            )
        if self.attribute_kind(target) == "float":
            stored = self.expression_type(value)
            if stored == "int":
                # CPython keeps the int and prints 3; this slot can only hold a
                # double and would print 3.0. Widening is the caller's decision.
                raise NativeCompileError(
                    self.path,
                    value,
                    "this attribute holds a float, and storing an integer here "
                    "would print as 3.0 where CPython prints 3; write float(x) "
                    "if widening is what you want",
                )
            if stored != "float":
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
            method_display_name(native_class.name, method_name),
            method,
            node,
            {},
            call_stack,
            skip_parameters=1,
            kinds=argument_kinds,
        )
        return self.inline_imperative_function(
            method_display_name(native_class.name, method_name),
            method,
            (instance, *arguments),
            node,
            call_stack,
            parameter_classes={"self": native_class.name},
            # `self` is the leading argument the caller supplied, so the kinds
            # the binding produced line up one position later.
            argument_kinds=("object", *argument_kinds),
            returns_string=self.method_returns_string(native_class, method, node),
        )

    def method_call_kind(self, node: ast.Call) -> str | None:
        """``"str"`` when this is a method call whose body answers a string.

        None when it is not a method call on a native class at all, and "int"
        when it is one that answers with a number - the caller only has to
        tell an address apart from a number.
        """

        if not isinstance(node.func, ast.Attribute):
            return None
        try:
            native_class = self.resolve_object_class(node.func.value)
        except NativeCompileError:
            return None
        if native_class is None:
            return None
        method = native_class.methods.get(node.func.attr)
        if method is None or not method.returns_value:
            return None
        if id(method) in self._kind_query:
            return "int"  # Recursive; the ordinary path reports it properly.
        return "str" if self.method_returns_string(native_class, method, node) else "int"

    def method_returns_string(
        self, native_class: NativeClass, method: NativeFunction, node: ast.Call
    ) -> bool:
        """Whether this method answers with a string block's address."""

        previous_objects = self.object_classes
        # `self` inside the body is an instance of this class, which is what
        # makes an attribute read there resolve to a field rather than fail.
        self.object_classes = {**previous_objects, "self": native_class.name}
        try:
            return self.statement_body_returns_string(
                method, node, None, skip_parameters=1
            )
        finally:
            self.object_classes = previous_objects

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
                if isinstance(statement.value, ast.Lambda):
                    # A lambda binding is a definition, not a value to copy
                    # into the uses of its name: substituting it would leave a
                    # call whose callee is the lambda itself, which resolves to
                    # no function. The imperative inliner runs the binding as
                    # the definition it is.
                    raise NativeCompileError(
                        self.path,
                        statement,
                        "a lambda binding defines a function, so this body is "
                        "inlined statement by statement rather than folded",
                    )
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
                if not expression.elts:
                    # Only the empty literal has nothing to read its kind from,
                    # which is the whole reason to write the annotation.
                    kind = declared
                elif kind != declared and {kind, declared} != {
                    "list:int",
                    "list:bool",
                }:
                    # A non-empty literal says what it holds. Believing the
                    # annotation over it would store 1 as 1.0 and print it back
                    # as 1.0, where CPython keeps the int. `list[int] = [True]`
                    # is the exception: CPython keeps the bools too.
                    raise NativeCompileError(
                        self.path,
                        expression,
                        f"this literal builds a {kind} but the annotation says "
                        f"{declared}; an annotation names what an EMPTY literal "
                        "holds, it does not convert the elements of one that is "
                        "not empty",
                    )
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
        if annotation is not None and (
            isinstance(expression, ast.Set) or self.empty_set_call(expression)
        ):
            # `s: set[str] = set()` says what set() cannot.
            declared = self.annotated_set_tag(annotation)
            if declared is not None:
                # A non-empty literal says what it holds, and believing the
                # annotation over it would hash an integer as a string pointer.
                if not self.empty_set_call(expression) and kind != declared:
                    raise NativeCompileError(
                        self.path,
                        expression,
                        f"this literal builds a {kind} but the annotation says "
                        f"{declared}; an annotation names what set() holds, it "
                        "does not convert the elements of a literal",
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
        elif self.set_kind(kind) is not None:
            self.set_assignment(name, expression, self.set_kind(kind))
        elif self.list_kind(kind) is not None:
            if isinstance(expression, ast.List) and not expression.elts:
                if annotation is None or self.annotated_list_tag(annotation) is None:
                    self.undecided_lists.add(name)
                else:
                    self.undecided_lists.discard(name)
            else:
                self.undecided_lists.discard(name)
            if (
                isinstance(expression, ast.Name)
                and self.list_kind_of(expression.id) is not None
            ):
                raise NativeCompileError(
                    self.path,
                    expression,
                    f"a native list variable holds the block itself, not a "
                    f"reference to it, so {name!r} cannot be another name for "
                    f"{expression.id!r}: appending moves the block and only one "
                    "of them would follow it. Copy it with a slice "
                    f"({expression.id}[:]) if a second list is what you want",
                )
            self.list_assignment(name, expression, self.list_kind(kind))
        elif self.tuple_kinds(kind) is not None:
            self.tuple_assignment(name, expression, self.tuple_kinds(kind))
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
        kind = self.list_kind(self.value_types.get(name))
        if kind is not None:
            self.refuse_unbound(name)
        return kind

    # An element is eight bytes whatever it holds, so a string or a nested list
    # travels as its block pointer and only the kind has to be carried along.
    # `bool` is an element kind of its own rather than an integer, because
    # nothing at run time tells True from 1 and an element read back out of an
    # unnamed list - `xs[0][1]` - has no name whose bookkeeping could say.
    _LIST_LEAF_KINDS = frozenset({"int", "float", "str", "bool"})

    @staticmethod
    def element_value_type(element_kind: str) -> str:
        """The `expression_type` reading an element of this kind gives back."""

        return "int" if element_kind == "bool" else element_kind

    def kind_noun(self, kind: str) -> str:
        """How a diagnostic names an element kind."""

        nested = self.list_kind(kind)
        if nested is not None:
            return f"lists of {self.kind_noun(nested)}"
        return {
            "int": "signed 64-bit integers",
            "bool": "bools",
            "float": "floats",
            "str": "strings",
        }.get(kind, kind)

    def element_kind_of(
        self, node: ast.expr, bindings: dict[str, KernelValue] | None = None
    ) -> str:
        """The element kind storing ``node`` in a native list would need."""

        kind = self.expression_type(node, bindings)
        if kind == "int" and self.renders_as_bool(node):
            return "bool"
        return kind

    def storable_element_kind(self, kind: str) -> bool:
        return kind in self._LIST_LEAF_KINDS or self.list_kind(kind) is not None

    def settle_element_kind(self, name: str, node: ast.expr) -> str:
        """The element kind of ``name``, deciding it now if it is still open.

        `xs = []` has nothing to read a kind from, so the first thing stored
        chooses one. Guessing integers at the literal and refusing everything
        else would reject `flags = []` followed by `flags.append(a > b)`, which
        is ordinary Python.
        """

        element_kind = self.list_kind_of(name)
        if name not in self.undecided_lists:
            return element_kind
        found = self.element_kind_of(node)
        if not self.storable_element_kind(found):
            return element_kind
        self.undecided_lists.discard(name)
        self.value_types[name] = self.list_tag(found)
        return found

    def check_element(self, node: ast.expr, element_kind: str, where: str) -> None:
        """Refuse an expression that cannot be stored as this element kind."""

        found = self.element_kind_of(node)
        if found != element_kind:
            self.refuse_element_mismatch(node, element_kind, found, where)

    def refuse_element_mismatch(
        self, node: ast.expr, element_kind: str, found: str, where: str
    ) -> None:
        """Say why an element of one kind cannot join a list of another.

        A list element keeps whatever was put in it - CPython does not convert
        one to the list's declared type - so widening an integer into a float
        list would read back as 2.0 where CPython reads 2.
        """

        if {found, element_kind} == {"bool", "int"}:
            self.refuse_bool_mix(node, element_kind == "bool", where)
        hint = (
            "; write 1.0 rather than 1, because an element keeps whatever it "
            "was given and CPython would read the integer back out"
            if element_kind == "float" and found == "int"
            else "; one list holds one kind, because nothing at run time tells "
            "the eight bytes of an element apart. An empty inner literal "
            "counts as a list of integers, so annotate the name if that is "
            "what disagrees"
        )
        raise NativeCompileError(
            self.path,
            node,
            f"{where} holds {self.kind_noun(element_kind)} and this is "
            f"{self.kind_noun(found)}{hint}",
        )

    def element_word(self, node: ast.expr, element_kind: str) -> IntExpression:
        """The eight bytes an element of this kind stores for ``node``.

        A float goes in as its bit pattern, and a string or a nested list as
        the address of its block; an integer and a bool are already words.
        """

        if element_kind == "float":
            return FloatBits(self.float_expression(node))
        if element_kind == "str":
            return self.string_pointer(node)
        if self.list_kind(element_kind) is not None:
            return self.list_pointer(node)
        return self.integer(node)

    def list_literal_tag(
        self, node: ast.List, bindings: dict[str, KernelValue] | None = None
    ) -> str:
        """The element kind a list literal builds, read off its first element.

        An empty literal has nothing to read, so its kind is settled by the
        first thing stored in it, or by an annotation such as
        `xs: list[float] = []`. The rest of the elements have to agree: one
        list is one kind, because the eight bytes of an element say nothing
        about what they hold.
        """

        if not node.elts:
            return self.list_tag("int")
        kind = self.element_kind_of(node.elts[0], bindings)
        if not self.storable_element_kind(kind):
            raise NativeCompileError(
                self.path,
                node.elts[0],
                "a native list element is a signed 64-bit integer, a float, a "
                "string, a bool, or another list",
            )
        for element in node.elts[1:]:
            other = self.element_kind_of(element, bindings)
            if other == kind:
                continue
            self.refuse_element_mismatch(element, kind, other, "this list")
        return self.list_tag(kind)

    def annotated_list_tag(self, annotation: ast.expr) -> str | None:
        """The element kind `list[T]` names, or None if it is not that shape."""

        if (
            not isinstance(annotation, ast.Subscript)
            or not isinstance(annotation.value, ast.Name)
            or annotation.value.id not in {"list", "List"}
        ):
            return None
        return self.list_tag(self.annotated_element_kind(annotation.slice))

    def annotated_element_kind(self, node: ast.expr) -> str:
        """The element kind the `T` of a `list[T]` annotation names."""

        if isinstance(node, ast.Name) and node.id in self._LIST_LEAF_KINDS:
            return node.id
        nested = (
            self.annotated_list_tag(node) if isinstance(node, ast.Subscript) else None
        )
        if nested is None:
            raise NativeCompileError(
                self.path,
                node,
                "a native list annotation is list[T] where T is int, float, "
                "str, bool, or another list[...]",
            )
        return nested

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

    def slice_step(self, node: ast.Slice) -> int:
        """The slice's step, which has to be known at build time.

        A step decides the direction of the walk, and the direction decides
        what the defaults for the two bounds are and which way they clamp. A
        step only known at run time would mean emitting both walks and choosing
        between them, for a shape nothing writes.
        """

        if node.step is None:
            return 1
        try:
            step = self.constant(node.step)
        except NativeCompileError:
            step = None
        if not isinstance(step, int) or isinstance(step, bool) or step == 0:
            raise NativeCompileError(
                self.path,
                node,
                "a native slice steps by a non-zero integer constant"
                + ("; a step of 0 is an error in Python too" if step == 0 else ""),
            )
        return step

    def strided_slice_extent(
        self, node: ast.Slice, length: IntExpression
    ) -> tuple[int, int, int]:
        """Slots holding the first index and how many the slice takes.

        Python's own rules, which differ by direction: going forwards the
        bounds default to 0 and the length and clamp into `[0, length]`; going
        backwards they default to the last index and to just before the first,
        and clamp into `[-1, length - 1]`.
        """

        step = self.slice_step(node)
        length_slot = self.materialize_int(length)
        start_slot = self.new_temp()
        stop_slot = self.new_temp()
        for slot, bound, forward_default, backward_default in (
            (start_slot, node.lower, IntConstant(0), IntBinary("sub", length_slot, IntConstant(1))),
            (stop_slot, node.upper, length_slot, IntConstant(-1)),
        ):
            if bound is None:
                self.operations.append(
                    Store(slot, forward_default if step > 0 else backward_default)
                )
                continue
            self.operations.append(Store(slot, self.integer(bound)))
            self.operations.append(
                Store(
                    slot,
                    self.select_integer(
                        IntCompare("lt", IntLoad(slot), IntConstant(0)),
                        IntBinary("add", IntLoad(slot), length_slot),
                        IntLoad(slot),
                    ),
                )
            )
            low, high = (
                (IntConstant(0), length_slot)
                if step > 0
                else (IntConstant(-1), IntBinary("sub", length_slot, IntConstant(1)))
            )
            self.operations.append(
                Store(
                    slot,
                    self.select_integer(
                        IntCompare("lt", IntLoad(slot), low), low, IntLoad(slot)
                    ),
                )
            )
            self.operations.append(
                Store(
                    slot,
                    self.select_integer(
                        IntCompare("gt", IntLoad(slot), high), high, IntLoad(slot)
                    ),
                )
            )
        # ceil((stop - start) / step) without a division: the span rounded up
        # by adding one less than the step's size before dividing.
        span = (
            IntBinary("sub", IntLoad(stop_slot), IntLoad(start_slot))
            if step > 0
            else IntBinary("sub", IntLoad(start_slot), IntLoad(stop_slot))
        )
        size = abs(step)
        count_slot = self.new_temp()
        self.operations.append(
            Store(
                count_slot,
                IntBinary(
                    "sdiv",
                    IntBinary("add", span, IntConstant(size - 1)),
                    IntConstant(size),
                ),
            )
        )
        self.operations.append(
            Store(
                count_slot,
                self.select_integer(
                    IntCompare("lt", IntLoad(count_slot), IntConstant(0)),
                    IntConstant(0),
                    IntLoad(count_slot),
                ),
            )
        )
        return start_slot, count_slot, step

    def slice_bounds(
        self, node: ast.Slice, length: IntExpression
    ) -> tuple[IntExpression, IntExpression]:
        if self.slice_step(node) != 1:
            raise NativeCompileError(
                self.path,
                node,
                "this slice steps by one only",
            )
        lower = self.slice_bound(node.lower, length, IntConstant(0))
        upper = self.slice_bound(node.upper, length, length)
        # A start past the stop is an empty slice, not a negative length.
        upper = self.materialize_int(
            self.select_integer(IntCompare("lt", upper, lower), lower, upper)
        )
        return lower, upper

    def emit_strided_list_slice(
        self, source: IntExpression, node: ast.Slice
    ) -> IntExpression:
        """`xs[a:b:step]` - the selected elements, copied one at a time.

        One at a time because the source words are no longer contiguous; a
        step of one still goes through the block copy above.
        """

        bump = self.ensure_heap()
        source_slot = self.new_temp()
        self.operations.append(Store(source_slot, source))
        origin = IntLoad(source_slot)
        start_slot, count_slot, step = self.strided_slice_extent(
            node, HeapLoad(IntBinary("add", origin, IntConstant(8)), 8)
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
        walk = self.new_label("stride")
        done = self.new_label("stride_done")
        self.operations.append(Label(walk))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(count_slot)), done
            )
        )
        taken = IntBinary(
            "add",
            IntLoad(start_slot),
            IntBinary("mul", IntLoad(index_slot), IntConstant(step)),
        )
        self.operations.append(
            HeapStore(
                IntBinary(
                    "add",
                    IntBinary("add", result, IntConstant(self.LIST_HEADER_BYTES)),
                    IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
                ),
                HeapLoad(
                    IntBinary(
                        "add",
                        IntBinary(
                            "add", origin, IntConstant(self.LIST_HEADER_BYTES)
                        ),
                        IntBinary("mul", taken, IntConstant(8)),
                    ),
                    8,
                ),
                8,
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(walk))
        self.operations.append(Label(done))
        return result

    def emit_list_slice(
        self, source: IntExpression, node: ast.Slice
    ) -> IntExpression:
        if self.slice_step(node) != 1:
            return self.emit_strided_list_slice(source, node)
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

    def emit_list_copy(self, source: IntExpression) -> IntExpression:
        """A fresh block holding the same elements as ``source``.

        The elements move as raw 64-bit words, so a float's bit pattern
        arrives unchanged and -0.0 stays -0.0.
        """

        bump = self.ensure_heap()
        source_slot = self.new_temp()
        self.operations.append(Store(source_slot, source))
        origin = IntLoad(source_slot)
        count_slot = self.new_temp()
        self.operations.append(
            Store(count_slot, HeapLoad(IntBinary("add", origin, IntConstant(8)), 8))
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
        start = self.new_label("list_copy")
        end = self.new_label("list_copy_end")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(count_slot)), end
            )
        )
        offset = IntBinary(
            "add",
            IntConstant(self.LIST_HEADER_BYTES),
            IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", result, offset),
                HeapLoad(IntBinary("add", origin, offset), 8),
                8,
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
        return result

    def list_word_address(
        self, pointer_slot: int, index: IntExpression
    ) -> IntExpression:
        """The address of element ``index`` of the list in ``pointer_slot``."""

        return IntBinary(
            "add",
            IntBinary(
                "add", IntLoad(pointer_slot), IntConstant(self.LIST_HEADER_BYTES)
            ),
            IntBinary("mul", index, IntConstant(8)),
        )

    def emit_refuse_nan(self, pointer_slot: int) -> None:
        """Refuse to sort a float list that holds a NaN.

        CPython does not raise on one - it returns whatever order timsort's
        particular sequence of comparisons happens to leave behind, and an
        insertion sort makes a different sequence. Matching would mean
        reimplementing timsort, so a NaN is refused rather than answered with a
        different order. A NaN is the one value that is not equal to itself,
        and both backends make that comparison unordered-correct.
        """

        length_slot = self.new_temp()
        self.operations.append(
            Store(
                length_slot,
                HeapLoad(IntBinary("add", IntLoad(pointer_slot), IntConstant(8)), 8),
            )
        )
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        item_slot = self.new_temp()
        start = self.new_label("nan_scan")
        found = self.new_label("nan_scan_found")
        end = self.new_label("nan_scan_end")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), end
            )
        )
        self.operations.append(
            Store(
                item_slot,
                HeapLoad(self.list_word_address(pointer_slot, IntLoad(index_slot)), 8),
            )
        )
        self.operations.append(
            JumpIfFalse(
                FloatCompare(
                    "eq",
                    BitsFloat(IntLoad(item_slot)),
                    BitsFloat(IntLoad(item_slot)),
                ),
                found,
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(found))
        self.raise_exception(
            "ValueError",
            b"ValueError: py2bin cannot sort a list containing nan; CPython's "
            b"order for one is whatever its own comparisons leave behind, and "
            b"this sort would produce a different one\n",
        )
        self.operations.append(Label(end))

    def emit_insertion_sort(
        self, pointer_slot: int, element_kind: str, descending: bool
    ) -> None:
        """Sort a list block in place, stably, allocating nothing.

        Insertion sort rather than something faster because the arena never
        reclaims: a merge sort's scratch buffer would be abandoned once per
        call, and a sort inside a loop would exhaust the arena. Only a strict
        comparison moves an element, which is what keeps equal elements in the
        order they arrived - observable when -0.0 sits beside 0.0.
        """

        length_slot = self.new_temp()
        self.operations.append(
            Store(
                length_slot,
                HeapLoad(IntBinary("add", IntLoad(pointer_slot), IntConstant(8)), 8),
            )
        )
        outer_slot = self.new_temp()
        self.operations.append(Store(outer_slot, IntConstant(1)))
        value_slot = self.new_temp()
        inner_slot = self.new_temp()
        outer = self.new_label("sort_outer")
        outer_end = self.new_label("sort_outer_end")
        inner = self.new_label("sort_inner")
        inner_end = self.new_label("sort_inner_end")
        self.operations.append(Label(outer))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(outer_slot), IntLoad(length_slot)), outer_end
            )
        )
        # The element being placed has to be held in a slot: the shifting below
        # overwrites the position it came from.
        self.operations.append(
            Store(
                value_slot,
                HeapLoad(self.list_word_address(pointer_slot, IntLoad(outer_slot)), 8),
            )
        )
        self.operations.append(
            Store(inner_slot, IntBinary("sub", IntLoad(outer_slot), IntConstant(1)))
        )
        self.operations.append(Label(inner))
        # Two jumps rather than one `j >= 0 and a[j] > v`: the backends evaluate
        # an expression tree eagerly, so the fused form would load a[-1] - the
        # word in front of the block - on every insertion at the front.
        self.operations.append(
            JumpIfFalse(
                IntCompare("ge", IntLoad(inner_slot), IntConstant(0)), inner_end
            )
        )
        held = HeapLoad(self.list_word_address(pointer_slot, IntLoad(inner_slot)), 8)
        if element_kind == "float":
            # Compare the numbers, not the words holding them: read as signed
            # integers, -1.0 sorts above -2.0, the reverse of the float order.
            comparison = FloatCompare(
                "lt" if descending else "gt",
                BitsFloat(held),
                BitsFloat(IntLoad(value_slot)),
            )
        elif element_kind == "str":
            # The words are block addresses, so they are compared through the
            # text they point at. The byte walk emits here, inside the inner
            # loop, which is where it has to run.
            comparison = IntCompare(
                "lt" if descending else "gt",
                IntLoad(self.emit_string_order(held, IntLoad(value_slot))),
                IntConstant(0),
            )
        else:
            comparison = IntCompare(
                "lt" if descending else "gt", held, IntLoad(value_slot)
            )
        self.operations.append(JumpIfFalse(comparison, inner_end))
        self.operations.append(
            HeapStore(
                self.list_word_address(
                    pointer_slot, IntBinary("add", IntLoad(inner_slot), IntConstant(1))
                ),
                HeapLoad(self.list_word_address(pointer_slot, IntLoad(inner_slot)), 8),
                8,
            )
        )
        self.operations.append(
            Store(inner_slot, IntBinary("sub", IntLoad(inner_slot), IntConstant(1)))
        )
        self.operations.append(Jump(inner))
        self.operations.append(Label(inner_end))
        self.operations.append(
            HeapStore(
                self.list_word_address(
                    pointer_slot, IntBinary("add", IntLoad(inner_slot), IntConstant(1))
                ),
                IntLoad(value_slot),
                8,
            )
        )
        self.operations.append(
            Store(outer_slot, IntBinary("add", IntLoad(outer_slot), IntConstant(1)))
        )
        self.operations.append(Jump(outer))
        self.operations.append(Label(outer_end))

    def emit_round(self, node: ast.Call, bindings, call_stack) -> IntExpression:
        """`round(x)` - to the nearest integer, ties to the even one.

        Python rounds a tie to even rather than away from zero, so round(2.5)
        is 2 and round(3.5) is 4. The fractional part is computed by
        subtracting the floor, which is exact for every double: below 2**52 the
        subtraction of two nearby values loses nothing, and above it there is no
        fractional part left to lose.
        """

        if len(node.args) != 1 or node.keywords:
            raise NativeCompileError(
                self.path,
                node,
                "native round() takes one argument. round(x, n) rounds in "
                "decimal, which means converting to decimal and back, and "
                "answers a float rather than an int; use int(x * 100 + 0.5) / "
                "100.0 if approximate decimal rounding will do",
            )
        argument = node.args[0]
        if self.expression_type(argument, bindings) != "float":
            # round(5) is 5. An integer is already round.
            return self.integer(argument, bindings, call_stack)
        bits = self.new_temp()
        self.operations.append(
            Store(bits, FloatBits(self.float_expression(argument, bindings, call_stack)))
        )
        value = BitsFloat(IntLoad(bits))
        truncated = self.new_temp()
        self.operations.append(Store(truncated, FloatToInt(value)))
        # Truncation goes toward zero, so for a negative value with a fraction
        # it lands one above the floor.
        floor_slot = self.new_temp()
        self.operations.append(
            Store(
                floor_slot,
                IntBinary(
                    "sub",
                    IntLoad(truncated),
                    FloatCompare("lt", value, IntToFloat(IntLoad(truncated))),
                ),
            )
        )
        fraction = FloatBinary("sub", value, IntToFloat(IntLoad(floor_slot)))
        fraction_slot = self.new_temp()
        self.operations.append(Store(fraction_slot, FloatBits(fraction)))
        part = BitsFloat(IntLoad(fraction_slot))
        half = FloatConstant(0.5)
        return IntBinary(
            "add",
            IntLoad(floor_slot),
            IntBinary(
                "or",
                FloatCompare("gt", part, half),
                IntBinary(
                    "and",
                    FloatCompare("eq", part, half),
                    IntBinary("and", IntLoad(floor_slot), IntConstant(1)),
                ),
            ),
        )

    def is_divmod_call(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "divmod"
            and node.func.id not in self.functions
        )

    def divmod_assignment(self, target: ast.Tuple, node: ast.expr) -> None:
        """`q, r = divmod(a, b)` - the quotient and remainder in one go.

        Only this shape. divmod() answers a tuple, and a tuple here is a block
        built from a literal; taking one apart again to get at the two numbers
        would allocate for a pair that is always unpacked immediately.

        Each operand is bound to a hidden name first, so that `divmod(f(), g())`
        calls each once rather than once per half of the answer.
        """

        assert isinstance(node, ast.Call)
        if len(node.args) != 2 or node.keywords:
            raise NativeCompileError(
                self.path, node, "native divmod() takes two integers"
            )
        if len(target.elts) != 2 or not all(
            isinstance(element, ast.Name) for element in target.elts
        ):
            raise NativeCompileError(
                self.path,
                target,
                "native divmod() answers two values, so it needs two names to "
                "put them in",
            )
        for argument in node.args:
            if self.expression_type(argument) == "float":
                raise NativeCompileError(
                    self.path,
                    argument,
                    "native divmod() takes integers; for floats the two halves "
                    "are x // y and x % y, which are both in the subset",
                )
        operands = []
        for argument in node.args:
            name = f"__divmod_operand_{self.print_argument_count}"
            self.print_argument_count += 1
            self.assignment(name, argument)
            operands.append(ast.copy_location(ast.Name(id=name, ctx=ast.Load()), argument))
        for element, operator in zip(target.elts, (ast.FloorDiv(), ast.Mod())):
            assert isinstance(element, ast.Name)
            self.assignment(
                element.id,
                ast.copy_location(
                    ast.BinOp(left=operands[0], op=operator, right=operands[1]),
                    node,
                ),
            )

    def emit_ord(self, node: ast.Call) -> IntExpression:
        """`ord(s)` - the code point of a one-character string."""

        if len(node.args) != 1 or node.keywords:
            raise NativeCompileError(
                self.path, node, "native ord() takes exactly one argument"
            )
        if self.expression_type(node.args[0]) != "str":
            raise NativeCompileError(
                self.path,
                node.args[0],
                "native ord() takes a string; a bytes object is not in the "
                "subset",
            )
        pointer = self.materialize_int(self.string_pointer(node.args[0]))
        single = self.new_label("ord_single")
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "eq", self.emit_code_point_count(pointer), IntConstant(1)
                ),
                single + "_no",
            )
        )
        self.operations.append(Jump(single))
        self.operations.append(Label(single + "_no"))
        self.raise_exception(
            "TypeError", b"TypeError: ord() expected a character\n"
        )
        self.operations.append(Label(single))
        return self.emit_decode_codepoint(pointer, IntConstant(0))

    def emit_chr(self, node: ast.Call) -> IntExpression:
        """`chr(n)` - the one-code-point string block for ``n``."""

        if len(node.args) != 1 or node.keywords:
            raise NativeCompileError(
                self.path, node, "native chr() takes exactly one argument"
            )
        return self.emit_encode_codepoint(self.integer(node.args[0]))

    def emit_decode_codepoint(
        self, pointer: IntExpression, offset: IntExpression
    ) -> IntExpression:
        """The code point encoded at byte ``offset`` of a UTF-8 block.

        Branching rather than reading all four bytes and masking: a one-byte
        code point at the very end of a string has no second byte, and the
        block is only as long as its contents, so the read would be outside it.

        Nothing validates the encoding. Every string here was either written by
        the compiler from source text or built by concatenating ones that were,
        so a malformed sequence cannot arrive.
        """

        pointer_slot = self.new_temp()
        self.operations.append(Store(pointer_slot, pointer))
        offset_slot = self.new_temp()
        self.operations.append(Store(offset_slot, offset))
        first = IntBinary(
            "add",
            IntBinary("add", IntLoad(pointer_slot), IntConstant(8)),
            IntLoad(offset_slot),
        )

        def byte(index: int) -> IntExpression:
            return HeapLoad(IntBinary("add", first, IntConstant(index)), 1)

        def tail(index: int) -> IntExpression:
            return IntBinary("and", byte(index), IntConstant(0x3F))

        lead_slot = self.new_temp()
        self.operations.append(Store(lead_slot, byte(0)))
        lead = IntLoad(lead_slot)
        result = self.new_temp()
        done = self.new_label("cp_decoded")
        for limit, width, mask in ((0x80, 1, 0x7F), (0xE0, 2, 0x1F), (0xF0, 3, 0x0F)):
            skip = self.new_label("cp_wider")
            self.operations.append(
                JumpIfFalse(IntCompare("lt", lead, IntConstant(limit)), skip)
            )
            value: IntExpression = IntBinary("and", lead, IntConstant(mask))
            for index in range(1, width):
                value = IntBinary(
                    "or", IntBinary("lshift", value, IntConstant(6)), tail(index)
                )
            self.operations.append(Store(result, value))
            self.operations.append(Jump(done))
            self.operations.append(Label(skip))
        four: IntExpression = IntBinary("and", lead, IntConstant(0x07))
        for index in range(1, 4):
            four = IntBinary(
                "or", IntBinary("lshift", four, IntConstant(6)), tail(index)
            )
        self.operations.append(Store(result, four))
        self.operations.append(Label(done))
        return IntLoad(result)

    def emit_encode_codepoint(self, value: IntExpression) -> IntExpression:
        """A one-code-point string block holding ``value``, UTF-8 encoded."""

        value_slot = self.new_temp()
        self.operations.append(Store(value_slot, value))
        code = IntLoad(value_slot)
        in_range = self.new_label("chr_in_range")
        self.operations.append(
            JumpIfFalse(
                IntBinary(
                    "and",
                    IntCompare("ge", code, IntConstant(0)),
                    IntCompare("le", code, IntConstant(0x10FFFF)),
                ),
                in_range + "_bad",
            )
        )
        self.operations.append(Jump(in_range))
        self.operations.append(Label(in_range + "_bad"))
        self.raise_exception(
            "ValueError", b"ValueError: chr() arg not in range(0x110000)\n"
        )
        self.operations.append(Label(in_range))
        # A lone surrogate has no UTF-8 form. CPython's chr() hands one back and
        # only fails when it is written out; a string here is its UTF-8 bytes
        # and has nowhere to keep one, so this is where it has to be reported.
        not_surrogate = self.new_label("chr_not_surrogate")
        self.operations.append(
            JumpIfFalse(
                IntBinary(
                    "and",
                    IntCompare("ge", code, IntConstant(0xD800)),
                    IntCompare("le", code, IntConstant(0xDFFF)),
                ),
                not_surrogate,
            )
        )
        self.raise_exception(
            "ValueError",
            b"ValueError: chr() of a lone surrogate has no UTF-8 form, and a "
            b"native string is its UTF-8 bytes\n",
        )
        self.operations.append(Label(not_surrogate))
        width = self.new_temp()
        self.operations.append(Store(width, IntConstant(1)))
        for limit, size in ((0x80, 2), (0x800, 3), (0x10000, 4)):
            narrower = self.new_label("chr_width")
            self.operations.append(
                JumpIfFalse(IntCompare("ge", code, IntConstant(limit)), narrower)
            )
            self.operations.append(Store(width, IntConstant(size)))
            self.operations.append(Label(narrower))
        bump = self.ensure_heap()
        block = self.new_temp()
        self.operations.append(
            HeapAlloc(block, self._aligned_size(IntLoad(width)), bump)
        )
        self.operations.append(HeapStore(IntLoad(block), IntLoad(width), 8))
        text = IntBinary("add", IntLoad(block), IntConstant(8))

        def put(index: int, byte: IntExpression) -> None:
            self.operations.append(
                HeapStore(IntBinary("add", text, IntConstant(index)), byte, 1)
            )

        def low(shift: int) -> IntExpression:
            return IntBinary(
                "or",
                IntConstant(0x80),
                IntBinary("and", IntBinary("rshift", code, IntConstant(shift)), IntConstant(0x3F)),
            )

        written = self.new_label("chr_written")
        for size, lead_mask, lead_shift in (
            (1, 0x00, 0),
            (2, 0xC0, 6),
            (3, 0xE0, 12),
            (4, 0xF0, 18),
        ):
            other = self.new_label("chr_size")
            self.operations.append(
                JumpIfFalse(
                    IntCompare("eq", IntLoad(width), IntConstant(size)), other
                )
            )
            put(
                0,
                IntBinary(
                    "or",
                    IntConstant(lead_mask),
                    IntBinary("rshift", code, IntConstant(lead_shift)),
                )
                if size > 1
                else code,
            )
            for index in range(1, size):
                put(index, low(6 * (size - 1 - index)))
            self.operations.append(Jump(written))
            self.operations.append(Label(other))
        self.operations.append(Label(written))
        return IntLoad(block)

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

    def emit_string_index(
        self,
        source: IntExpression,
        index: ast.expr | None,
        position: IntExpression | None = None,
    ) -> IntExpression:
        """`s[i]` - the one-code-point string at position ``i``.

        Indexing is not slicing with a narrower window: a slice clamps and
        `s[99:100]` is empty, while `s[99]` raises IndexError. So the bound is
        checked rather than clamped, and the check runs on the code-point count
        because that is what CPython indexes by.
        """

        bump = self.ensure_heap()
        source_slot = self.new_temp()
        self.operations.append(Store(source_slot, source))
        origin = IntLoad(source_slot)
        length = self.materialize_int(self.emit_code_point_count(origin))
        wanted = self.materialize_int(
            position if position is not None else self.integer(index)
        )
        position_slot = self.new_temp()
        self.operations.append(
            Store(
                position_slot,
                self.select_integer(
                    IntCompare("lt", wanted, IntConstant(0)),
                    IntBinary("add", wanted, length),
                    wanted,
                ),
            )
        )
        ok = self.new_label("string_index_ok")
        self.operations.append(
            JumpIfFalse(
                IntBinary(
                    "and",
                    IntCompare("ge", IntLoad(position_slot), IntConstant(0)),
                    IntCompare("lt", IntLoad(position_slot), length),
                ),
                ok + "_bad",
            )
        )
        self.operations.append(Jump(ok))
        self.operations.append(Label(ok + "_bad"))
        self.raise_exception(
            "IndexError", b"IndexError: string index out of range\n"
        )
        self.operations.append(Label(ok))
        first = self.materialize_int(
            self.emit_codepoint_offset(origin, IntLoad(position_slot))
        )
        last = self.materialize_int(
            self.emit_codepoint_offset(
                origin, IntBinary("add", IntLoad(position_slot), IntConstant(1))
            )
        )
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

    def emit_reversed_string(self, source: IntExpression) -> IntExpression:
        """`s[::-1]` - the code points in the opposite order.

        Walking forwards and writing backwards, rather than walking backwards:
        a UTF-8 sequence can only be measured from its lead byte, so the width
        of each code point is known going forwards and would have to be found
        by scanning back over continuation bytes going the other way. The
        result is exactly as long in bytes as the source.
        """

        bump = self.ensure_heap()
        source_slot = self.new_temp()
        self.operations.append(Store(source_slot, source))
        origin = IntLoad(source_slot)
        total = self.materialize_int(HeapLoad(origin, 8))
        result_slot = self.new_temp()
        self.operations.append(HeapAlloc(result_slot, self._aligned_size(total), bump))
        result = IntLoad(result_slot)
        self.operations.append(HeapStore(result, total, 8))
        read = self.new_temp()
        write = self.new_temp()
        self.operations.append(Store(read, IntConstant(0)))
        self.operations.append(Store(write, total))
        walk = self.new_label("reverse")
        done = self.new_label("reverse_done")
        self.operations.append(Label(walk))
        self.operations.append(
            JumpIfFalse(IntCompare("lt", IntLoad(read), total), done)
        )
        lead = HeapLoad(
            IntBinary(
                "add", IntBinary("add", origin, IntConstant(8)), IntLoad(read)
            ),
            1,
        )
        width = self.new_temp()
        self.operations.append(Store(width, IntConstant(4)))
        for limit, size in ((0xF0, 3), (0xE0, 2), (0x80, 1)):
            wider = self.new_label("reverse_width")
            self.operations.append(
                JumpIfFalse(IntCompare("lt", lead, IntConstant(limit)), wider)
            )
            self.operations.append(Store(width, IntConstant(size)))
            self.operations.append(Label(wider))
        self.operations.append(
            Store(write, IntBinary("sub", IntLoad(write), IntLoad(width)))
        )
        self.emit_byte_copy(
            IntBinary(
                "add", IntBinary("add", result, IntConstant(8)), IntLoad(write)
            ),
            IntBinary(
                "add", IntBinary("add", origin, IntConstant(8)), IntLoad(read)
            ),
            IntLoad(width),
        )
        self.operations.append(
            Store(read, IntBinary("add", IntLoad(read), IntLoad(width)))
        )
        self.operations.append(Jump(walk))
        self.operations.append(Label(done))
        return result

    def emit_string_slice(
        self, source: IntExpression, node: ast.Slice
    ) -> IntExpression:
        if self.slice_step(node) != 1:
            if (
                self.slice_step(node) == -1
                and node.lower is None
                and node.upper is None
            ):
                return self.emit_reversed_string(source)
            raise NativeCompileError(
                self.path,
                node,
                "a native string slice steps by one, or is s[::-1]. A wider "
                "step would have to find the byte offset of every code point "
                "it lands on, and a UTF-8 sequence can only be measured from "
                "its lead byte - so it would rescan from the start each time",
            )
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
    _AGGREGATE_CALLS = frozenset({"sum", "min", "max", "any", "all"})

    def aggregate_call(self, node: ast.Call, bindings, call_stack):
        """`sum/min/max/any/all(xs)` over a runtime list or a generator."""

        name = node.func.id
        if name in {"min", "max"} and len(node.args) == 2 and not node.keywords:
            bools = [self.renders_as_bool(item) for item in node.args]
            if any(bools) and not all(bools):
                # The result is whichever argument wins, so with a bool on one
                # side and a number on the other its kind is decided at run
                # time - min(True, 3) is True and min(True, 0) is 0. One slot
                # cannot print both ways, so this is refused the way a mixed
                # container and a mixed conditional already are.
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native {name}() is given a bool and a number, and returns "
                    "whichever wins, so whether the answer prints as True or as "
                    "1 would be decided at run time; wrap the bool in int() to "
                    "compare numbers",
                )
            # min(a, b) is a comparison, not a walk. CPython returns the FIRST
            # argument when they are equal, which is why the test is strict.
            left = self.materialize_int(
                self.integer(node.args[0], bindings, call_stack)
            )
            right = self.materialize_int(
                self.integer(node.args[1], bindings, call_stack)
            )
            keeps_left = IntCompare(
                "lt" if name == "min" else "gt", right, left
            )
            return self.select_integer(keeps_left, right, left)
        if name == "sum" and len(node.args) == 2 and not node.keywords:
            # sum(xs, start) is the walk plus the start value, and the start
            # decides the kind: sum([], 0.0) is 0.0 in CPython.
            total = self.aggregate_call(
                ast.copy_location(
                    ast.Call(func=node.func, args=node.args[:1], keywords=[]), node
                ),
                bindings,
                call_stack,
            )
            if self.expression_type(node.args[1], bindings) == "float":
                raise NativeCompileError(
                    self.path,
                    node.args[1],
                    "native sum() adds up integers, so its start must be one "
                    "too; a float start would make the result a float, and the "
                    "walk has already added the elements as integers",
                )
            return IntBinary(
                "add", total, self.integer(node.args[1], bindings, call_stack)
            )
        if len(node.args) != 1 or node.keywords:
            raise NativeCompileError(
                self.path,
                node,
                f"native {name}() takes one iterable"
                + (", or two values" if name in {"min", "max"} else "")
                + (", or an iterable and a start" if name == "sum" else ""),
            )
        source = node.args[0]
        if isinstance(source, ast.GeneratorExp):
            return self.aggregate_over_generator(
                node, name, source, bindings, call_stack
            )
        element_kind = self.list_kind(self.expression_type(source, bindings))
        if element_kind is None:
            raise NativeCompileError(
                self.path,
                node,
                f"native {name}() takes a runtime list or a generator "
                "expression over one",
            )
        if element_kind not in {"int", "bool"}:
            raise NativeCompileError(
                self.path,
                node,
                f"native {name}() works on integer lists, and this one holds "
                f"{self.kind_noun(element_kind)}; a float one would need a "
                "float accumulator this call cannot return, and a string or a "
                "list one would compare block addresses",
            )
        pointer_slot = self.new_temp()
        self.operations.append(Store(pointer_slot, self.list_pointer(source)))
        pointer = IntLoad(pointer_slot)
        length_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntBinary("add", pointer, IntConstant(8)), 8))
        )
        first = IntBinary("add", pointer, IntConstant(self.LIST_HEADER_BYTES))

        def walk(step) -> None:
            index_slot = self.new_temp()
            self.operations.append(Store(index_slot, IntConstant(0)))
            start = self.new_label("aggregate")
            end = self.new_label("aggregate_end")
            self.operations.append(Label(start))
            self.operations.append(
                JumpIfFalse(
                    IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), end
                )
            )
            step(
                HeapLoad(
                    IntBinary(
                        "add",
                        first,
                        IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
                    ),
                    8,
                )
            )
            self.operations.append(
                Store(
                    index_slot,
                    IntBinary("add", IntLoad(index_slot), IntConstant(1)),
                )
            )
            self.operations.append(Jump(start))
            self.operations.append(Label(end))

        return self.emit_aggregate(name, walk)

    def aggregate_over_generator(
        self,
        node: ast.Call,
        name: str,
        generator: ast.GeneratorExp,
        bindings=None,
        call_stack: tuple[int, ...] = (),
    ):
        """`sum(v * 2 for v in xs)` lowered as the loop it describes.

        A generator expression is consumed here and nowhere else, so nothing
        has to represent the lazy object: the loops run and the accumulator is
        the answer. No list is built, which matters because the arena never
        gives anything back.
        """

        element_kind = self.comprehension_element_kind(generator, bindings)
        if element_kind not in {"int", "bool"}:
            raise NativeCompileError(
                self.path,
                node,
                f"native {name}() works on integer elements, and these are "
                f"{self.kind_noun(element_kind)}; a float one would need a "
                "float accumulator this call cannot return, and a string or a "
                "list one would compare block addresses",
            )
        sources, element = self.comprehension_clause_sources(
            generator, bindings, call_stack
        )

        def walk(step) -> None:
            self.emit_comprehension_loops(
                sources,
                lambda: step(self.integer(element, bindings, call_stack)),
                bindings,
                call_stack,
            )

        return self.emit_aggregate(name, walk)

    def emit_aggregate(self, name: str, emit_loop):
        """Fold every value ``emit_loop`` hands to the ``step`` it is given.

        Only the accumulator lives here, so walking a list and running a
        generator expression's loops share one definition of what each of the
        five names means.
        """

        result_slot = self.new_temp()
        seeded_slot = self.new_temp() if name in {"min", "max"} else None
        self.operations.append(
            Store(result_slot, IntConstant(1 if name == "all" else 0))
        )
        if seeded_slot is not None:
            self.operations.append(Store(seeded_slot, IntConstant(0)))
        done = self.new_label("aggregate_done")

        def step(value: IntExpression) -> None:
            value_slot = self.new_temp()
            self.operations.append(Store(value_slot, value))
            item = IntLoad(value_slot)
            if name == "sum":
                self.operations.append(
                    Store(result_slot, IntBinary("add", IntLoad(result_slot), item))
                )
                return
            if name in {"any", "all"}:
                keep_going = self.new_label("aggregate_keep")
                decisive = "ne" if name == "any" else "eq"
                self.operations.append(
                    JumpIfFalse(
                        IntCompare(decisive, item, IntConstant(0)), keep_going
                    )
                )
                self.operations.append(
                    Store(result_slot, IntConstant(1 if name == "any" else 0))
                )
                # CPython stops at the first decisive element. Nothing here has
                # an effect worth observing, but leaving early is still what
                # the loop means.
                self.operations.append(Jump(done))
                self.operations.append(Label(keep_going))
                return
            take = self.new_label("aggregate_take")
            kept = self.new_label("aggregate_kept")
            # The first item seeds the accumulator: no value a 64-bit slot can
            # hold is unavailable to an element, so emptiness needs its own flag.
            self.operations.append(
                JumpIfFalse(
                    IntCompare("ne", IntLoad(seeded_slot), IntConstant(0)), take
                )
            )
            self.operations.append(
                JumpIfFalse(
                    IntCompare(
                        self._AGGREGATES[name], item, IntLoad(result_slot)
                    ),
                    kept,
                )
            )
            self.operations.append(Label(take))
            self.operations.append(Store(result_slot, item))
            self.operations.append(Store(seeded_slot, IntConstant(1)))
            self.operations.append(Label(kept))

        emit_loop(step)
        if seeded_slot is not None:
            # min() and max() of an empty iterable raise; with a generator the
            # emptiness is only known once the loops are over.
            ok = self.new_label("aggregate_ok")
            self.operations.append(
                JumpIfFalse(
                    IntCompare("ne", IntLoad(seeded_slot), IntConstant(0)),
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
        if name in {"any", "all"}:
            self.operations.append(Label(done))
        return IntLoad(result_slot)

    def sort_direction(self, node: ast.expr, keywords: list[ast.keyword]) -> bool:
        """Read `reverse=` off a sort, refusing everything else by name."""

        descending = False
        for keyword in keywords:
            if keyword.arg is None:
                raise NativeCompileError(
                    self.path,
                    node,
                    "native sorting does not take **kwargs; the only keyword it "
                    "accepts is reverse=True or reverse=False",
                )
            if keyword.arg != "reverse":
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native sorting does not support {keyword.arg}=; there is "
                    "no way to hold a comparison callable in the subset, so "
                    "only reverse=True and reverse=False are accepted",
                )
            try:
                value = self.constant(keyword.value)
            except NativeCompileError:
                value = None
            if not isinstance(value, bool):
                raise NativeCompileError(
                    self.path,
                    keyword.value,
                    "native sorting needs reverse= to be the constant True or "
                    "False; a direction only known at run time would need both "
                    "comparison directions compiled in and chosen between",
                )
            descending = value
        return descending

    def sorted_call_shape(self, node: ast.expr):
        """The list `sorted(...)` sorts, its element kind and its direction.

        Returns None only when this is not a `sorted()` call at all. A call
        that is one but is spelled in a way the subset cannot do is refused
        here, so the keyword or the argument names itself instead of reaching
        a generic "not an integer" message further down.
        """

        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "sorted"
            or node.func.id in self.functions
        ):
            return None
        if len(node.args) != 1:
            raise NativeCompileError(
                self.path,
                node,
                "native sorted() takes one runtime list, optionally with "
                "reverse=True or reverse=False",
            )
        source = node.args[0]
        if isinstance(source, ast.Name) and self.set_kind_of(source.id) is not None:
            # Sorting is the one thing that can be read out of a set without
            # ever observing its order, because the answer does not depend on
            # the order the elements came out in.
            element_kind = self.set_kind_of(source.id)
            if element_kind != "int":
                raise NativeCompileError(
                    self.path,
                    node,
                    "native sorted() takes a set of integers; sorting a set of "
                    "strings would compare block addresses rather than text, "
                    "which is allocation order and not the order Python gives",
                )
            if self.container_bool.get(source.id):
                raise NativeCompileError(
                    self.path,
                    node,
                    "native sorted() works on sets of integers, and this one "
                    "holds bools; nothing at run time tells True from 1, so "
                    "the result would print as 1 and 0",
                )
            return source, element_kind, self.sort_direction(node, node.keywords)
        element_kind = self.list_kind(self.expression_type(source))
        # A bool list reaches `refuse_bool_list`, which says why in its own
        # words; everything else that is not a number is refused here, because
        # the comparison would be between block addresses - allocation order,
        # which is neither lexicographic nor structural.
        if element_kind is None or element_kind not in {"int", "float", "bool", "str"}:
            raise NativeCompileError(
                self.path,
                node,
                "native sorted() takes a runtime list of integers, floats or "
                "strings; sorting a dict, or a list of lists, would compare "
                "block addresses rather than values, which is allocation "
                "order and not the order Python gives",
            )
        return source, element_kind, self.sort_direction(node, node.keywords)

    def list_source_name(self, node: ast.expr) -> str | None:
        """The name whose bookkeeping answers for this list expression."""

        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            return self.list_source_name(node.value)
        shape = self.sorted_call_shape(node)
        if shape is not None:
            return self.list_source_name(shape[0])
        return None

    def list_holds_bool(self, node: ast.expr) -> bool | None:
        """Whether every element of this list expression is a bool.

        The list's own tag answers this, which is what lets an element of a
        list nobody named - `xs[0]`, a slice of a slice, a comprehension used
        in place - still print True rather than 1.
        """

        try:
            kind = self.list_kind(self.expression_type(node))
        except NativeCompileError:
            return None
        if kind is None:
            return None
        return kind == "bool"

    def refuse_bool_list(self, node: ast.expr, source: ast.expr, what: str) -> None:
        """Refuse to reorder a list of bools.

        `sorted([True, False])` is `[False, True]` and prints as bools, but
        nothing at run time tells True from 1, and the name a sorted copy is
        bound to does not inherit the source's entry. Reordering one would
        print 1 and 0.
        """

        if self.list_holds_bool(source) is not True:
            return
        raise NativeCompileError(
            self.path,
            node,
            f"native {what} works on lists of integers or floats, and this one "
            "holds bools; nothing at run time tells True from 1, so the result "
            "would print as 1 and 0. Store int(b) instead if the numbers are "
            "what you want",
        )

    def sorted_call(self, node: ast.Call) -> IntExpression:
        """`sorted(xs)` - a new sorted list, leaving `xs` in its own order."""

        source, element_kind, descending = self.sorted_call_shape(node)
        if isinstance(source, ast.Name) and self.set_kind_of(source.id) is not None:
            pointer_slot = self.emit_set_to_list(source.id)
            self.emit_insertion_sort(pointer_slot, element_kind, descending)
            return IntLoad(pointer_slot)
        self.refuse_bool_list(node, source, "sorted()")
        pointer_slot = self.new_temp()
        self.operations.append(
            Store(pointer_slot, self.emit_list_copy(self.list_pointer(source)))
        )
        if element_kind == "float":
            self.emit_refuse_nan(pointer_slot)
        self.emit_insertion_sort(pointer_slot, element_kind, descending)
        return IntLoad(pointer_slot)

    def membership_container_kind(self, node: ast.expr) -> str | None:
        """What `x in node` searches: "dict", "str", or the list's own tag.

        The tag rather than the element kind, because a list of strings and a
        string are two different searches and both would answer "str".
        """

        if isinstance(node, ast.Name) and self.dict_kinds_of(node.id):
            return "dict"
        if isinstance(node, ast.Name) and self.set_kind_of(node.id) is not None:
            return "set"
        if isinstance(node, ast.Set):
            # Probing needs a table, and building one here would allocate on
            # every evaluation - once per iteration inside a loop.
            raise NativeCompileError(
                self.path,
                node,
                "`in` over a set literal is not supported: searching one means "
                "building its table first, and doing that here would allocate "
                "on every evaluation. Bind the set to a name and search that",
            )
        try:
            kind = self.expression_type(node)
        except NativeCompileError:
            return None
        if kind == "str":
            return "str"
        if self.tuple_kinds(kind) is not None:
            # A tuple search compares each element with a different comparison
            # per position, and nothing here does that.
            raise NativeCompileError(
                self.path,
                node,
                "`in` over a native tuple is not supported: searching one "
                "means comparing the left side with each element, and a tuple "
                "can hold a different kind at every position. Search a runtime "
                "list, a runtime dict or a string instead",
            )
        return kind if self.list_kind(kind) is not None else None

    def emit_list_membership(
        self, node: ast.expr, container: ast.expr, element_kind: str
    ) -> int:
        """Return a 0/1 slot saying whether ``node`` is in the list."""

        if self.list_kind(element_kind) is not None:
            raise NativeCompileError(
                self.path,
                container,
                "native `in` over a list of lists is not in the subset: CPython "
                "compares the elements of the two lists, and the eight bytes "
                "here are a block address, so two equal lists that were "
                "allocated separately would answer False",
            )
        wanted = self.element_kind_of(node)
        agrees = (
            wanted in {"int", "float", "bool"}
            if element_kind == "float"
            else self.element_value_type(wanted)
            == self.element_value_type(element_kind)
        )
        if not agrees:
            raise NativeCompileError(
                self.path,
                node,
                f"this list holds {self.kind_noun(element_kind)} and this is "
                f"{self.kind_noun(wanted)}",
            )
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
        elif element_kind == "str":
            self.operations.append(Store(wanted_slot, self.string_pointer(node)))
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
        elif element_kind == "str":
            # The words are block addresses, and two equal strings are two
            # allocations; compare the bytes rather than where they live.
            item_slot = self.new_temp()
            self.operations.append(Store(item_slot, item))
            equal_slot = self.emit_string_equal(
                IntLoad(item_slot), IntLoad(wanted_slot)
            )
            same = IntCompare("ne", IntLoad(equal_slot), IntConstant(0))
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

    def emit_substring_scan(
        self, haystack_slot: int, needle_slot: int, from_slot: int
    ) -> tuple[int, int]:
        """Look for a needle at or after a byte offset into a haystack.

        Returns a 0/1 slot and the byte offset the match starts at. The three
        arguments are slot numbers holding two string-block pointers and a
        starting offset; nothing here re-evaluates them, so a caller may drive
        this from inside a loop to walk every occurrence.

        A plain scan: for every starting byte, compare forward. UTF-8 makes
        that safe without decoding, because a multi-byte character can never
        match part of another one - lead and continuation bytes come from
        disjoint ranges.

        Only code-point boundaries are offered as starting positions. For a
        non-empty needle that is already implied, since its first byte can
        never equal a continuation byte; the empty needle is what needs it,
        because CPython finds an empty string between characters and not
        between the bytes of one.
        """

        outer = IntBinary("add", IntLoad(haystack_slot), IntConstant(8))
        inner = IntBinary("add", IntLoad(needle_slot), IntConstant(8))
        outer_length = self.new_temp()
        inner_length = self.new_temp()
        self.operations.append(
            Store(outer_length, HeapLoad(IntLoad(haystack_slot), 8))
        )
        self.operations.append(
            Store(inner_length, HeapLoad(IntLoad(needle_slot), 8))
        )
        found_slot = self.new_temp()
        self.operations.append(Store(found_slot, IntConstant(0)))
        start_slot = self.new_temp()
        self.operations.append(Store(start_slot, IntLoad(from_slot)))
        scan = self.new_label("find")
        done = self.new_label("find_done")
        next_start = self.new_label("find_next")
        aligned = self.new_label("find_aligned")
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
        # The end of the string is a boundary, and reading a byte there would
        # be a read past the block.
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(start_slot), IntLoad(outer_length)),
                aligned,
            )
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "ne",
                    IntBinary(
                        "and",
                        HeapLoad(
                            IntBinary("add", outer, IntLoad(start_slot)), 1
                        ),
                        IntConstant(0xC0),
                    ),
                    IntConstant(0x80),
                ),
                next_start,
            )
        )
        self.operations.append(Label(aligned))
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
        return found_slot, start_slot

    def emit_substring_search(self, needle: ast.expr, haystack: ast.expr) -> int:
        """Return a 0/1 slot saying whether one string contains the other."""

        haystack_slot = self.new_temp()
        needle_slot = self.new_temp()
        from_slot = self.new_temp()
        self.operations.append(
            Store(haystack_slot, self.string_pointer(haystack))
        )
        self.operations.append(Store(needle_slot, self.string_pointer(needle)))
        self.operations.append(Store(from_slot, IntConstant(0)))
        found_slot, _ = self.emit_substring_scan(
            haystack_slot, needle_slot, from_slot
        )
        return found_slot

    def note_stored_bool(self, name: str, node: ast.expr, where: str) -> None:
        """Record whether ``name``'s elements are bools, and refuse a mix.

        A container keeps one answer for all of its elements, because there is
        nowhere at run time to keep a different one per slot. Storing a bool
        beside a number would make one of them print wrongly, so a mix is
        refused rather than guessed at.
        """

        stored = self.renders_as_bool(node)
        current = self.container_bool.get(name)
        if current is None:
            self.container_bool[name] = stored
            return
        self.refuse_bool_mix(node, current, where)

    def refuse_bool_mix(self, node: ast.expr, current: bool, where: str) -> None:
        """Refuse a bool beside a number, or the reverse, in one container."""

        stored = self.renders_as_bool(node)
        if current == stored:
            return
        raise NativeCompileError(
            self.path,
            node,
            f"{where} holds {'bools' if current else 'numbers'} already, and "
            f"this is a {'bool' if stored else 'number'}. One slot cannot "
            "print both ways, so a mixed container is refused; wrap the "
            "bool in int() to store the number instead",
        )

    def refuse_stored_bool(self, node: ast.expr, where: str) -> None:
        """Refuse to store a bool where its kind would be forgotten.

        A bool lives in an integer slot, and which slots hold one is tracked
        from the source rather than at run time. Storing one in a list or a
        dict loses that tracking, and reading it back would print 1 instead of
        True - a wrong answer rather than a refusal. Until the kind travels
        with the value, say so.

        A bool passed as an argument has the same problem, but is not refused
        here: a procedure that never renders its parameter is unaffected, and
        refusing every call would reject working programs to catch the few that
        print. That case is still a divergence, and is documented as one.
        """

        if self.renders_as_bool(node):
            raise NativeCompileError(
                self.path,
                node,
                f"a bool cannot be stored in {where} yet: nothing distinguishes "
                "True from 1 at run time, so it would print as 1; wrap it in "
                "int() if the number is what you want",
            )

    def note_escaping_list_names(self, tree: ast.Module) -> None:
        """Record every name whose block is stored inside a container.

        A native list variable holds its block rather than a reference to it,
        and appending to a full one copies the block and writes the new address
        back into that variable's slot. An element that was handed the old
        address is not updated, so ``xs.append(inner)`` followed later by
        ``inner.append(v)`` would leave ``xs[0]`` on the abandoned copy - a
        stale length and stale contents, where CPython's element is the same
        object and sees the growth.

        Collected over the whole module rather than in source order, because a
        loop's back edge puts the append textually before the store that shared
        the block.
        """

        for node in ast.walk(tree):
            stored: list[ast.expr] = []
            if isinstance(node, ast.List):
                stored = list(node.elts)
            elif isinstance(node, ast.ListComp):
                stored = [node.elt]
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and len(node.args) == 1
            ):
                stored = list(node.args)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Subscript) for target in node.targets
            ):
                stored = [node.value]
            for item in stored:
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load):
                    self.escaped_list_names.add(item.id)

    def refuse_appending_to_a_shared_block(
        self, node: ast.expr, name: str
    ) -> None:
        """Refuse append() on a list whose block another container also holds."""

        if name in self.escaped_list_names:
            raise NativeCompileError(
                self.path,
                node,
                f"{name!r} is stored inside another container somewhere in this "
                "module, and a native list variable holds its block rather than "
                "a reference to it: appending moves the block, and the element "
                "holding the old address would be left on the abandoned copy "
                f"while CPython sees the growth. Store a copy ({name}[:]) so "
                "the two are separate lists, or build the elements before "
                "storing them",
            )
        if name in self.shared_list_names:
            raise NativeCompileError(
                self.path,
                node,
                f"{name!r} names a list that is an element of another one, and "
                "a native list variable holds its block rather than a reference "
                "to it: appending moves the block and the element it came from "
                "would be left on the abandoned copy while CPython sees the "
                f"growth. Copy it ({name} = {name}[:]) if a separate list is "
                "what you want",
            )

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

    def list_sort_call(self, node: ast.Call, name: str) -> None:
        """`xs.sort()` - reorder the block the name already points at."""

        if node.args:
            raise NativeCompileError(
                self.path,
                node,
                "native sort() takes no positional argument; there is no way "
                "to hold a comparison callable in the subset, so only "
                "reverse=True and reverse=False are accepted",
            )
        descending = self.sort_direction(node, node.keywords)
        self.refuse_bool_list(node, node.func.value, "sort()")
        element_kind = self.list_kind_of(name)
        if element_kind not in {"int", "float", "str"}:
            raise NativeCompileError(
                self.path,
                node,
                f"native sort() works on lists of integers, floats or strings, "
                f"and this one holds {self.kind_noun(element_kind)}; comparing "
                "those would compare block addresses, which is allocation "
                "order and not the order Python gives",
            )
        pointer_slot = self.slot(name)
        if element_kind == "float":
            self.emit_refuse_nan(pointer_slot)
        self.emit_insertion_sort(pointer_slot, element_kind, descending)

    def list_method_call(self, node: ast.Call) -> bool:
        """Lower `xs.append(v)` and `xs.sort()`; returns whether this was one."""

        if not isinstance(node.func, ast.Attribute):
            return False
        if not isinstance(node.func.value, ast.Name):
            if node.func.attr in {"append", "sort"} and self.list_kind(
                self.expression_type(node.func.value)
            ) is not None:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native {node.func.attr}() is only called on a list held "
                    "by a name: appending writes the moved block back into the "
                    "variable's slot, and an element inside another list has "
                    "no slot to write back to. Bind it to a name first, and "
                    "note that doing so makes it share the block",
                )
            return False
        if self.list_kind_of(node.func.value.id) is None:
            return False
        name = node.func.value.id
        if node.func.attr == "sort":
            self.list_sort_call(node, name)
            return True
        if node.func.attr == "insert":
            self.emit_list_insert(node, name)
            return True
        if node.func.attr == "remove":
            self.emit_list_remove(node, name)
            return True
        if node.func.attr in {"pop", "index", "count"}:
            # Each answers with a value. As a statement the answer is dropped,
            # which for pop() is the ordinary way to shorten a list.
            self.discard_expression(node)
            return True
        if node.func.attr != "append":
            raise NativeCompileError(
                self.path,
                node,
                "native lists support append(), sort(), insert(), remove(), "
                f"pop(), index() and count(); {node.func.attr}() is not in "
                "the subset",
            )
        if len(node.args) != 1 or node.keywords:
            raise NativeCompileError(
                self.path, node, "append() takes exactly one argument"
            )
        argument = node.args[0]
        self.refuse_appending_to_a_shared_block(node, name)
        element_kind = self.settle_element_kind(name, argument)
        self.check_element(argument, element_kind, "this list")
        value = self.element_word(argument, element_kind)
        # A literal's length stops being a build-time fact once it can grow.
        self.list_lengths.pop(name, None)
        self.emit_list_append(self.slot(name), value)
        return True

    def list_pointer(self, node: ast.expr) -> IntExpression:
        """A pointer to a runtime list block, building one if needed."""

        if isinstance(node, ast.Name) and self.list_kind_of(node.id) is not None:
            return IntLoad(self.slots[node.id])
        if self.list_kind(self.list_method_shape(node, "pop") or "") is not None:
            # A nested list is stored as its block address, so the word pop()
            # answers with already is the pointer.
            assert isinstance(node, ast.Call)
            return self.emit_list_pop(node)
        if isinstance(node, ast.ListComp):
            return self.list_comprehension(node)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            return self.emit_list_slice(
                self.list_pointer(node.value), node.slice
            )
        if (
            isinstance(node, ast.Subscript)
            and self.list_kind(self.expression_type(node)) is not None
        ):
            # A nested list is stored as the address of its own block, so the
            # element word is already the pointer.
            return HeapLoad(self.list_element_address(node), 8)
        if self.sorted_call_shape(node) is not None:
            return self.sorted_call(node)
        if self.list_kind(self.string_method_kind(node) or "") is not None:
            assert isinstance(node, ast.Call)
            return self.string_method_list(node)
        self.refuse_lazy_comprehension(node)
        if isinstance(node, ast.List):
            return self.emit_list_block(node, self.list_kind(self.expression_type(node)))
        raise NativeCompileError(
            self.path, node, "expression is not a native runtime list"
        )

    def refuse_lazy_comprehension(self, node: ast.expr) -> None:
        """Name the comprehension forms that have no native representation."""

        if isinstance(node, ast.GeneratorExp):
            raise NativeCompileError(
                self.path,
                node,
                "a generator expression is a lazy object and nothing here can "
                "hold one; it is supported only as the sole argument of sum(), "
                "min(), max(), any() or all(). Write [ ... ] for a list "
                "comprehension if you need the values kept",
            )
        if isinstance(node, ast.SetComp):
            raise NativeCompileError(
                self.path,
                node,
                "a native comprehension builds a list, and a set comprehension "
                "would build a set nothing may iterate; a set's order here "
                "could not match CPython's. Write [ ... ] for a list, or build "
                "the set with add()",
            )
        if isinstance(node, ast.DictComp):
            raise NativeCompileError(
                self.path,
                node,
                "a native comprehension builds a list; there is no runtime "
                "dict for a dict comprehension to build",
            )

    def refuse_set_iteration(self, node: ast.expr) -> None:
        """Refuse to walk a set's elements.

        CPython's set iteration order is unspecified, is not insertion order,
        and for strings is randomized per process by PYTHONHASHSEED, so there
        is no order this compiler could produce that matches it for every
        input. Refusing only `print` inside the loop would not be honest:
        appending to a list, breaking after the first element, or building a
        string all carry the order out of the loop just as well, and each would
        become a silent wrong answer.
        """

        try:
            kind = self.expression_type(node)
        except NativeCompileError:
            return
        if self.set_kind(kind) is None:
            return
        element_kind = self.set_kind(kind)
        hint = (
            "; sorted(s) gives a runtime list in a defined order"
            if element_kind == "int"
            else ""
        )
        raise NativeCompileError(
            self.path,
            node,
            "a native set cannot be iterated: CPython's set order is not "
            "insertion order and is not specified, so any order produced here "
            "would differ from it for some input and print the wrong answer. "
            "Sets support len(s), x in s, x not in s, s.add(), s.discard(), "
            "s.remove() and the | & - operators" + hint,
        )

    def list_assignment(
        self, name: str, node: ast.expr, element_kind: str
    ) -> None:
        if not isinstance(node, ast.List):
            # Not a literal, but perhaps a list-valued expression such as a
            # slice; that block is already built, so just bind the name to it.
            # Ask before emitting: the answer is about the source expression,
            # and lowering it renames a comprehension's targets.
            holds_bool = self.list_holds_bool(node)
            pointer = self.list_pointer(node)
            self.container_bool[name] = holds_bool
            if isinstance(node, ast.Subscript) and not isinstance(
                node.slice, ast.Slice
            ):
                # This name and the element it came from are the same block.
                self.shared_list_names.add(name)
            else:
                self.shared_list_names.discard(name)
            self.runtime_names.add(name)
            # Whatever length a literal left recorded under this name is no
            # longer this list's length, and an index checked against the old
            # one would be refused for being out of a range it is inside.
            self.list_lengths.pop(name, None)
            self.operations.append(Store(self.slot(name), pointer))
            return
        self.container_bool[name] = element_kind == "bool"
        self.shared_list_names.discard(name)
        self.runtime_names.add(name)
        # Build into a slot of its own and move it over afterwards. A literal
        # that reads the name it is being assigned to - `xs = [xs[0] + 1]` -
        # would otherwise take the new block's address out of the name and read
        # its uninitialised elements.
        pointer_slot = self.new_temp()
        self.emit_list_literal(pointer_slot, node, element_kind)
        self.operations.append(Store(self.slot(name), IntLoad(pointer_slot)))
        self.list_lengths[name] = len(node.elts)

    def emit_list_block(self, node: ast.List, element_kind: str) -> IntExpression:
        """Build a list literal that no name owns - a nested one, or an
        argument to append() - and hand back a pointer to it."""

        pointer_slot = self.new_temp()
        self.emit_list_literal(pointer_slot, node, element_kind)
        return IntLoad(pointer_slot)

    def emit_list_literal(
        self, pointer_slot: int, node: ast.List, element_kind: str
    ) -> None:
        elements = node.elts
        for element in elements:
            self.check_element(element, element_kind, "this list")
        bump = self.ensure_heap()
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
            # An inner literal or a concatenation allocates while it is being
            # lowered, so the word is built before the store that takes its
            # address - the arena only moves forward, so this block stays put.
            stored = self.element_word(element, element_kind)
            address = IntBinary(
                "add",
                IntLoad(pointer_slot),
                IntConstant(self.LIST_HEADER_BYTES + index * 8),
            )
            self.operations.append(HeapStore(address, stored, 8))

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
                "a list index whose bounds check has to run at run time cannot "
                "appear in a conditional expression or a short-circuited "
                "Boolean operand, because the check would run even when Python "
                "would not evaluate that branch; use an if statement instead. "
                "An index is checked at run time when it is not a constant, or "
                "when the list's length is not known at build time - which is "
                "the case for an element of another list, a slice, and a "
                "comprehension, even under a constant index",
            )
        index_slot, _length_slot = self.resolve_list_index(
            pointer,
            self.integer(index_node),
            # CPython words the two cases differently, and matching it is the
            # difference between the same message and merely the same class.
            assigning=isinstance(node.ctx, ast.Store),
        )
        offset = IntBinary(
            "add",
            IntConstant(self.LIST_HEADER_BYTES),
            IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
        )
        return IntBinary("add", pointer, offset)

    def resolve_list_index(
        self,
        pointer: IntExpression,
        index: IntExpression,
        *,
        assigning: bool = False,
        message: bytes | None = None,
    ) -> tuple[int, int]:
        """Normalize an index against a list's length and check it is in range.

        Returns the slots holding the resolved index and the length. A negative
        index counts back from the end the way Python's does, and one outside
        the list raises IndexError rather than addressing outside the block.
        ``assigning`` picks the wording CPython uses for a store or a del,
        and ``message`` overrides it where CPython has wording of its own -
        pop() does.
        """

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
            "IndexError",
            message
            or (
                b"IndexError: list assignment index out of range\n"
                if assigning
                else b"IndexError: list index out of range\n"
            ),
        )
        self.operations.append(Label(ok_label))
        return index_slot, length_slot

    def delete_statement(self, node: ast.Delete) -> None:
        """`del xs[i]` on a runtime list and `del d[k]` on a runtime dict;
        everything else is refused by name."""

        # Python evaluates the targets left to right, and each one can raise.
        for target in node.targets:
            self.delete_list_element(target)

    def delete_list_element(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            raise NativeCompileError(
                self.path,
                target,
                f"del {target.id} is not supported: a native variable is a "
                "stack slot holding a value, with nothing in it that could "
                "record being unbound, so a later read cannot raise "
                "NameError the way CPython's would; native del takes one "
                "element of a runtime list or one key of a runtime dict",
            )
        if isinstance(target, ast.Attribute):
            raise NativeCompileError(
                self.path,
                target,
                "del on an attribute is not supported: a native object's "
                "fields are fixed slots laid out at build time, so a later "
                "read of a deleted one cannot raise AttributeError; native "
                "del takes one element of a runtime list or one key of a "
                "runtime dict",
            )
        if not isinstance(target, ast.Subscript):
            raise NativeCompileError(
                self.path,
                target,
                "native del takes one element of a runtime list, or one key "
                "of a runtime dict, held by a name",
            )
        on_a_dict = (
            isinstance(target.value, ast.Name)
            and self.dict_kinds_of(target.value.id) is not None
        )
        if isinstance(target.slice, ast.Slice):
            if on_a_dict:
                raise NativeCompileError(
                    self.path,
                    target,
                    "del on a dict slice is not supported, as CPython's is not "
                    "either: a slice is unhashable, so it can never be a key; "
                    "del one key at a time",
                )
            raise NativeCompileError(
                self.path,
                target,
                "del on a list slice is not supported; del one element at a "
                "time",
            )
        if (
            isinstance(target.value, ast.Name)
            and self.tuple_kinds_of(target.value.id) is not None
        ):
            raise NativeCompileError(
                self.path,
                target,
                "del on a native tuple is not supported, as CPython's is not "
                "either: a tuple is immutable and its length is fixed at build "
                "time; native del takes one element of a runtime list or "
                "one key of a runtime dict",
            )
        if on_a_dict:
            assert isinstance(target.value, ast.Name)
            self.delete_dict_entry(target, target.value.id)
            return
        if (
            not isinstance(target.value, ast.Name)
            or self.list_kind_of(target.value.id) is None
        ):
            raise NativeCompileError(
                self.path,
                target,
                "native del takes one element of a runtime list, or one key "
                "of a runtime dict, held by a name",
            )
        name = target.value.id
        if name in self.iterated_lists:
            raise NativeCompileError(
                self.path,
                target,
                f"del cannot shorten {name!r} while a for loop is walking it: "
                "the walk took its length once and counts up, so it would run "
                "past the new end and yield an element CPython's iterator "
                "skips; collect the indexes to remove and del them after the "
                "loop",
            )
        pointer = IntLoad(self.slot(name))
        length = self.list_lengths.get(name)
        try:
            folded = self.constant(target.slice)
        except NativeCompileError:
            folded = None
        # A length known at build time settles a constant index now, the same
        # way indexing does, rather than leaving it to the run-time check.
        if (
            isinstance(folded, int)
            and not isinstance(folded, bool)
            and length is not None
        ):
            resolved = folded + length if folded < 0 else folded
            if not 0 <= resolved < length:
                raise NativeCompileError(
                    self.path,
                    target,
                    f"native list index {folded} is out of range for {name!r} "
                    f"(length {length})",
                )
        index_slot, length_slot = self.resolve_list_index(
            pointer, self.integer(target.slice), assigning=True
        )
        self.emit_list_remove_at(pointer, index_slot, length_slot)
        # Another name can hold this same block, and its recorded length is now
        # one too many - which would let a constant index past the new end pass
        # the build-time check and skip the run-time one. Nothing here tracks
        # which names alias which block, so every recorded length goes.
        self.list_lengths.clear()

    def discard_expression(self, node: ast.expr) -> None:
        """Lower an expression for its effect and drop the value it answers.

        The work is already in the operation list by the time the expression
        comes back, so dropping it drops only the value.
        """

        kind = self.expression_type(node)
        if kind == "float":
            self.float_expression(node)
        elif kind == "str":
            self.string_pointer(node)
        elif self.list_kind(kind) is not None:
            self.list_pointer(node)
        else:
            self.integer(node)

    def emit_list_insert(self, node: ast.Call, name: str) -> None:
        """`xs.insert(i, v)` - append, then rotate the tail up by one.

        Appending first is what makes the block big enough, and it is the only
        path that knows how to grow one and write the moved address back.
        """

        if len(node.args) != 2 or node.keywords:
            raise NativeCompileError(
                self.path, node, "native insert() takes an index and a value"
            )
        if name in self.iterated_lists:
            raise NativeCompileError(
                self.path,
                node,
                f"insert() cannot lengthen {name!r} while a for loop is "
                "walking it: the walk would see the shifted elements twice",
            )
        self.refuse_appending_to_a_shared_block(node, name)
        element_kind = self.settle_element_kind(name, node.args[1])
        self.check_element(node.args[1], element_kind, "this list")
        value = self.new_temp()
        self.operations.append(
            Store(value, self.element_word(node.args[1], element_kind))
        )
        pointer_slot = self.slot(name)
        length = HeapLoad(IntBinary("add", IntLoad(pointer_slot), IntConstant(8)), 8)
        # insert() clamps rather than raising: xs.insert(99, v) appends and
        # xs.insert(-99, v) prepends. The length it clamps against is the one
        # before the insert, so it is read here and not after the append.
        target = self.new_temp()
        self.operations.append(Store(target, self.materialize_int(self.integer(node.args[0]))))
        self.operations.append(
            Store(
                target,
                self.select_integer(
                    IntCompare("lt", IntLoad(target), IntConstant(0)),
                    IntBinary("add", IntLoad(target), length),
                    IntLoad(target),
                ),
            )
        )
        self.operations.append(
            Store(
                target,
                self.select_integer(
                    IntCompare("lt", IntLoad(target), IntConstant(0)),
                    IntConstant(0),
                    IntLoad(target),
                ),
            )
        )
        self.operations.append(
            Store(
                target,
                self.select_integer(
                    IntCompare("gt", IntLoad(target), length),
                    length,
                    IntLoad(target),
                ),
            )
        )
        self.list_lengths.pop(name, None)
        self.emit_list_append(pointer_slot, IntLoad(value))
        elements = IntBinary(
            "add", IntLoad(pointer_slot), IntConstant(self.LIST_HEADER_BYTES)
        )
        cursor = self.new_temp()
        self.operations.append(Store(cursor, IntBinary("sub", length, IntConstant(1))))
        shift = self.new_label("insert_shift")
        shifted = self.new_label("insert_shifted")
        self.operations.append(Label(shift))
        self.operations.append(
            JumpIfFalse(
                IntCompare("ge", IntLoad(cursor), IntLoad(target)), shifted
            )
        )
        at = IntBinary("add", elements, IntBinary("mul", IntLoad(cursor), IntConstant(8)))
        self.operations.append(
            HeapStore(IntBinary("add", at, IntConstant(8)), HeapLoad(at, 8), 8)
        )
        self.operations.append(
            Store(cursor, IntBinary("sub", IntLoad(cursor), IntConstant(1)))
        )
        self.operations.append(Jump(shift))
        self.operations.append(Label(shifted))
        self.operations.append(
            HeapStore(
                IntBinary(
                    "add", elements, IntBinary("mul", IntLoad(target), IntConstant(8))
                ),
                IntLoad(value),
                8,
            )
        )

    def emit_list_remove(self, node: ast.Call, name: str) -> None:
        """`xs.remove(v)` - drop the first element equal to ``v``."""

        element_kind = self.list_kind_of(name)
        assert element_kind is not None
        if name in self.iterated_lists:
            raise NativeCompileError(
                self.path,
                node,
                f"remove() cannot shorten {name!r} while a for loop is "
                "walking it: the walk counts up against the length, so it "
                "would run past the new end",
            )
        wanted = self.list_search_argument(node, element_kind, "remove")
        pointer = IntLoad(self.slot(name))
        index_slot, found_slot = self.emit_list_find(pointer, wanted, element_kind)
        present = self.new_label("remove_present")
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(found_slot), IntConstant(0)),
                present + "_missing",
            )
        )
        self.operations.append(Jump(present))
        self.operations.append(Label(present + "_missing"))
        self.raise_exception(
            "ValueError", b"ValueError: list.remove(x): x not in list\n"
        )
        self.operations.append(Label(present))
        length_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntBinary("add", pointer, IntConstant(8)), 8))
        )
        self.emit_list_remove_at(pointer, index_slot, length_slot)
        self.list_lengths.clear()

    def emit_list_index(self, node: ast.Call) -> IntExpression:
        """`xs.index(v)` - where ``v`` first appears, or ValueError."""

        assert isinstance(node.func, ast.Attribute)
        assert isinstance(node.func.value, ast.Name)
        element_kind = self.list_kind_of(node.func.value.id)
        assert element_kind is not None
        wanted = self.list_search_argument(node, element_kind, "index")
        index_slot, found_slot = self.emit_list_find(
            IntLoad(self.slot(node.func.value.id)), wanted, element_kind
        )
        present = self.new_label("index_present")
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(found_slot), IntConstant(0)),
                present + "_missing",
            )
        )
        self.operations.append(Jump(present))
        self.operations.append(Label(present + "_missing"))
        self.raise_exception(
            "ValueError", b"ValueError: list.index(x): x not in list\n"
        )
        self.operations.append(Label(present))
        return IntLoad(index_slot)

    def emit_list_count(self, node: ast.Call) -> IntExpression:
        """`xs.count(v)` - how many elements equal ``v``."""

        assert isinstance(node.func, ast.Attribute)
        assert isinstance(node.func.value, ast.Name)
        element_kind = self.list_kind_of(node.func.value.id)
        assert element_kind is not None
        wanted_slot = self.new_temp()
        self.operations.append(
            Store(wanted_slot, self.list_search_argument(node, element_kind, "count"))
        )
        pointer_slot = self.new_temp()
        self.operations.append(
            Store(pointer_slot, IntLoad(self.slot(node.func.value.id)))
        )
        total = self.new_temp()
        index_slot = self.new_temp()
        self.operations.append(Store(total, IntConstant(0)))
        self.operations.append(Store(index_slot, IntConstant(0)))
        scan = self.new_label("count_scan")
        done = self.new_label("count_done")
        step = self.new_label("count_step")
        self.operations.append(Label(scan))
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "lt",
                    IntLoad(index_slot),
                    HeapLoad(
                        IntBinary("add", IntLoad(pointer_slot), IntConstant(8)), 8
                    ),
                ),
                done,
            )
        )
        element = HeapLoad(
            IntBinary(
                "add",
                IntBinary(
                    "add",
                    IntLoad(pointer_slot),
                    IntConstant(self.LIST_HEADER_BYTES),
                ),
                IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
            ),
            8,
        )
        if element_kind == "str":
            same: IntExpression = IntLoad(
                self.emit_string_equal(element, IntLoad(wanted_slot))
            )
        elif element_kind == "float":
            same = FloatCompare(
                "eq", BitsFloat(element), BitsFloat(IntLoad(wanted_slot))
            )
        else:
            same = IntCompare("eq", element, IntLoad(wanted_slot))
        self.operations.append(JumpIfFalse(same, step))
        self.operations.append(
            Store(total, IntBinary("add", IntLoad(total), IntConstant(1)))
        )
        self.operations.append(Label(step))
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(scan))
        self.operations.append(Label(done))
        return IntLoad(total)

    def list_method_shape(self, node: ast.expr, attribute: str) -> str | None:
        """The element kind of `xs.<attribute>(...)`, or None if it is not that."""

        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or node.func.attr != attribute
            or not isinstance(node.func.value, ast.Name)
        ):
            return None
        return self.list_kind_of(node.func.value.id)

    def emit_list_pop(self, node: ast.Call) -> IntExpression:
        """`xs.pop()` or `xs.pop(i)` - the element word, with the list closed up.

        The word rather than the value, because every element is eight bytes
        whatever it holds: a float is its bit pattern and a string its block
        address, so the caller reinterprets it for the kind it asked for.
        """

        assert isinstance(node.func, ast.Attribute)
        assert isinstance(node.func.value, ast.Name)
        name = node.func.value.id
        if len(node.args) > 1 or node.keywords:
            raise NativeCompileError(
                self.path, node, "native pop() takes an optional index"
            )
        if name in self.iterated_lists:
            raise NativeCompileError(
                self.path,
                node,
                f"pop() cannot shorten {name!r} while a for loop is walking "
                "it: the walk counts up against the length, so it would skip "
                "the element that moved into the gap",
            )
        pointer = IntLoad(self.slot(name))
        if node.args:
            index = self.integer(node.args[0])
        else:
            # CPython has its own wording for an empty list, so the emptiness
            # is checked here rather than left to the index check below.
            nonempty = self.new_label("pop_nonempty")
            self.operations.append(
                JumpIfFalse(
                    IntCompare(
                        "eq",
                        HeapLoad(IntBinary("add", pointer, IntConstant(8)), 8),
                        IntConstant(0),
                    ),
                    nonempty,
                )
            )
            self.raise_exception(
                "IndexError", b"IndexError: pop from empty list\n"
            )
            self.operations.append(Label(nonempty))
            index = IntBinary(
                "sub",
                HeapLoad(IntBinary("add", pointer, IntConstant(8)), 8),
                IntConstant(1),
            )
        index_slot, length_slot = self.resolve_list_index(
            pointer,
            index,
            assigning=True,
            message=b"IndexError: pop index out of range\n",
        )
        taken = self.new_temp()
        self.operations.append(
            Store(
                taken,
                HeapLoad(
                    IntBinary(
                        "add",
                        IntBinary(
                            "add", pointer, IntConstant(self.LIST_HEADER_BYTES)
                        ),
                        IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
                    ),
                    8,
                ),
            )
        )
        self.emit_list_remove_at(pointer, index_slot, length_slot)
        # Another name may hold this block, and every recorded length for it is
        # now one too many - the same reason `del xs[i]` clears them.
        self.list_lengths.clear()
        return IntLoad(taken)

    def emit_list_find(
        self, pointer: IntExpression, wanted: IntExpression, element_kind: str
    ) -> tuple[int, int]:
        """Scan for ``wanted``; returns slots holding its index and 0/1 found.

        Stops at the first match, as `index()` and `remove()` both do. A float
        is compared as a number rather than as its bits, so that -0.0 finds
        0.0 the way CPython's == does.
        """

        pointer_slot = self.new_temp()
        self.operations.append(Store(pointer_slot, pointer))
        wanted_slot = self.new_temp()
        self.operations.append(Store(wanted_slot, wanted))
        index_slot = self.new_temp()
        found_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        self.operations.append(Store(found_slot, IntConstant(0)))
        scan = self.new_label("list_find")
        done = self.new_label("list_find_done")
        miss = self.new_label("list_find_miss")
        self.operations.append(Label(scan))
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "lt",
                    IntLoad(index_slot),
                    HeapLoad(
                        IntBinary("add", IntLoad(pointer_slot), IntConstant(8)), 8
                    ),
                ),
                done,
            )
        )
        element = HeapLoad(
            IntBinary(
                "add",
                IntBinary(
                    "add",
                    IntLoad(pointer_slot),
                    IntConstant(self.LIST_HEADER_BYTES),
                ),
                IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
            ),
            8,
        )
        if element_kind == "str":
            same: IntExpression = IntLoad(
                self.emit_string_equal(element, IntLoad(wanted_slot))
            )
        elif element_kind == "float":
            same = FloatCompare(
                "eq", BitsFloat(element), BitsFloat(IntLoad(wanted_slot))
            )
        else:
            same = IntCompare("eq", element, IntLoad(wanted_slot))
        self.operations.append(JumpIfFalse(same, miss))
        self.operations.append(Store(found_slot, IntConstant(1)))
        self.operations.append(Jump(done))
        self.operations.append(Label(miss))
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(scan))
        self.operations.append(Label(done))
        return index_slot, found_slot

    def list_search_argument(
        self, node: ast.Call, element_kind: str, what: str
    ) -> IntExpression:
        """The word to search a list for, checked against its element kind."""

        if len(node.args) != 1 or node.keywords:
            raise NativeCompileError(
                self.path, node, f"native {what}() takes exactly one argument"
            )
        argument = node.args[0]
        kind = self.expression_type(argument)
        if element_kind == "str":
            if kind != "str":
                raise NativeCompileError(
                    self.path, argument, "this list holds strings"
                )
            return self.string_pointer(argument)
        if element_kind == "float":
            return FloatBits(self.float_expression(argument))
        if kind == "float":
            raise NativeCompileError(
                self.path, argument, "this list holds integers"
            )
        return self.integer(argument)

    def emit_list_remove_at(
        self, pointer: IntExpression, index_slot: int, length_slot: int
    ) -> None:
        """Drop element ``index_slot`` from a list block, closing the gap.

        ``length_slot`` holds the length the block has now; the caller has
        already settled that the index is in range.
        """

        elements = IntBinary("add", pointer, IntConstant(self.LIST_HEADER_BYTES))
        cursor_slot = self.new_temp()
        self.operations.append(Store(cursor_slot, IntLoad(index_slot)))
        shift = self.new_label("del_shift")
        shifted = self.new_label("del_shifted")
        self.operations.append(Label(shift))
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "lt",
                    IntBinary("add", IntLoad(cursor_slot), IntConstant(1)),
                    IntLoad(length_slot),
                ),
                shifted,
            )
        )
        # One 8-byte word per element whatever the list holds: a float lives in
        # it as its bit pattern, so moving the word moves the value.
        element = IntBinary(
            "add",
            elements,
            IntBinary("mul", IntLoad(cursor_slot), IntConstant(8)),
        )
        self.operations.append(
            HeapStore(
                element,
                HeapLoad(IntBinary("add", element, IntConstant(8)), 8),
                8,
            )
        )
        self.operations.append(
            Store(cursor_slot, IntBinary("add", IntLoad(cursor_slot), IntConstant(1)))
        )
        self.operations.append(Jump(shift))
        self.operations.append(Label(shifted))
        self.operations.append(
            HeapStore(
                IntBinary("add", pointer, IntConstant(8)),
                IntBinary("sub", IntLoad(length_slot), IntConstant(1)),
                8,
            )
        )

    def delete_dict_entry(self, target: ast.Subscript, name: str) -> None:
        """`del d[k]`: tombstone the entry and unlist its key.

        The state word becomes a tombstone rather than empty so that a probe
        which walked past this slot to reach a later key still walks past it.
        The count drops, which is also what makes a del inside a walk of the
        same dict raise RuntimeError the way CPython's does. The used word does
        not drop: the slot is still not somewhere a probe can stop.
        """

        key_kind, _value_kind = self.dict_kinds_of(name)
        if self.expression_type(target.slice) != key_kind:
            raise NativeCompileError(
                self.path, target.slice, f"this dict has {key_kind} keys"
            )
        pointer_slot = self.slot(name)
        address_slot, found_slot, _key, _state = self.dict_probe(
            pointer_slot, self.dict_key(target.slice, key_kind), key_kind
        )
        present = self.new_label("dict_del_present")
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
        # The key word out of the entry, not out of the source expression: this
        # is the very word the order list holds, so a str key needs no byte
        # comparison to be found there.
        key_slot = self.new_temp()
        self.operations.append(
            Store(
                key_slot,
                HeapLoad(IntBinary("add", IntLoad(address_slot), IntConstant(8)), 8),
            )
        )
        self.operations.append(
            HeapStore(IntLoad(address_slot), IntConstant(self.DICT_TOMBSTONE), 8)
        )
        pointer = IntLoad(pointer_slot)
        self.operations.append(
            HeapStore(
                IntBinary("add", pointer, IntConstant(8)),
                IntBinary(
                    "sub",
                    HeapLoad(IntBinary("add", pointer, IntConstant(8)), 8),
                    IntConstant(1),
                ),
                8,
            )
        )
        keys_slot = self.new_temp()
        self.operations.append(
            Store(
                keys_slot,
                HeapLoad(
                    IntBinary("add", pointer, IntConstant(self.DICT_KEYS_OFFSET)), 8
                ),
            )
        )
        keys = IntLoad(keys_slot)
        length_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntBinary("add", keys, IntConstant(8)), 8))
        )
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        scan = self.new_label("dict_unlist")
        found = self.new_label("dict_unlisted")
        done = self.new_label("dict_unlist_done")
        self.operations.append(Label(scan))
        # The key is in the list, so the length bound is never what ends this
        # scan; it is there so that a bug would leave the order untouched
        # instead of running off the end of the block.
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), done
            )
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "ne",
                    HeapLoad(
                        IntBinary(
                            "add",
                            IntBinary(
                                "add", keys, IntConstant(self.LIST_HEADER_BYTES)
                            ),
                            IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
                        ),
                        8,
                    ),
                    IntLoad(key_slot),
                ),
                found,
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(scan))
        self.operations.append(Label(found))
        self.emit_list_remove_at(keys, index_slot, length_slot)
        self.operations.append(Label(done))

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
        if self.tuple_kinds(self.expression_type(target.value)) is not None:
            raise NativeCompileError(
                self.path,
                target,
                "a native tuple is immutable, as CPython's is, so an element "
                "cannot be assigned; build a new tuple, or use a runtime list, "
                "whose elements do assign",
            )
        if isinstance(target.value, ast.Name) and self.dict_kinds_of(
            target.value.id
        ):
            key_kind, value_kind = self.dict_kinds_of(target.value.id)
            if self.tuple_kinds(self.expression_type(target.slice)) is not None:
                raise NativeCompileError(
                    self.path,
                    target.slice,
                    "a native dict key is a signed 64-bit integer or a runtime "
                    "string; a tuple key would need an element-wise hash and "
                    "an element-wise comparison that the table does not have",
                )
            if self.expression_type(target.slice) != key_kind:
                raise NativeCompileError(
                    self.path,
                    target.slice,
                    f"this dict has {key_kind} keys",
                )
            if key_kind == "int":
                self.note_stored_bool(
                    self.dict_keys_name(target.value.id),
                    target.slice,
                    "this dict's keys",
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
            else:
                self.note_stored_bool(target.value.id, value, "this dict")
            self.dict_store(
                self.slot(target.value.id),
                self.dict_key(target.slice, key_kind),
                self.dict_value(value, value_kind),
                target,
                key_kind,
            )
            return
        element_kind = self.list_kind(self.expression_type(target.value))
        if element_kind is None:
            raise NativeCompileError(
                self.path, target, "native indexing requires a runtime list"
            )
        self.check_element(value, element_kind, "this list")
        # The eight bytes go in as they are: a double as its bit pattern, a
        # string or a nested list as the address of its block.
        stored = self.element_word(value, element_kind)
        address = self.list_element_address(target)
        self.operations.append(HeapStore(address, stored, 8))

    # --- runtime tuples -----------------------------------------------------
    #
    # Layout: [i64 length][element0][element1]... A tuple is immutable and its
    # length is fixed when it is written, so unlike a list it needs no capacity
    # word and no room to grow, and the block it is given is the block it keeps
    # for the life of the program.
    #
    # The length is also in the tag, so nothing reads that first word; it is
    # there so an empty tuple is still an eight-byte block with an address of
    # its own rather than a zero-byte one sharing whatever is allocated next.
    #
    # The tag carries one kind PER ELEMENT - `tuple:int,float,str` - which is
    # the whole reason a tuple is worth having beside a list. A list is one
    # kind throughout because nothing at run time tells the eight bytes of one
    # element from another's; a tuple gets away with mixing them because every
    # index that reads one is a constant, so the kind is settled at build time.

    TUPLE_HEADER_BYTES = 8
    _TUPLE_LEAF_KINDS = frozenset({"int", "float", "str", "bool"})

    @staticmethod
    def tuple_tag(kinds: tuple[str, ...]) -> str:
        return "tuple:" + ",".join(kinds)

    @staticmethod
    def tuple_kinds(tag: str | None) -> tuple[str, ...] | None:
        """The per-element kinds of a tuple tag, else None."""

        if not isinstance(tag, str) or not tag.startswith("tuple:"):
            return None
        rest = tag[len("tuple:") :]
        return tuple(rest.split(",")) if rest else ()

    def tuple_kinds_of(self, name: str) -> tuple[str, ...] | None:
        kind = self.tuple_kinds(self.value_types.get(name))
        if kind is not None:
            self.refuse_unbound(name)
        return kind

    def tuple_literal_kinds(
        self, node: ast.Tuple, bindings: dict[str, KernelValue] | None = None
    ) -> tuple[str, ...]:
        """The kind of every element of a tuple literal, in order."""

        kinds: list[str] = []
        for element in node.elts:
            kind = self.element_kind_of(element, bindings)
            if kind not in self._TUPLE_LEAF_KINDS:
                raise NativeCompileError(
                    self.path,
                    element,
                    "a native tuple element is a signed 64-bit integer, a "
                    "float, a string or a bool; a nested tuple, a list, a dict "
                    "or an object cannot be one, because every element is read "
                    "back by a load whose kind is fixed at build time",
                )
            kinds.append(kind)
        return tuple(kinds)

    def constant_index(self, node: ast.expr) -> int | None:
        """A subscript's index when it is a build-time integer, else None."""

        try:
            folded = self.constant(node)
        except NativeCompileError:
            return None
        if isinstance(folded, bool) or not isinstance(folded, int):
            return None
        return folded

    def tuple_subscript_kind(
        self, node: ast.expr, bindings: dict[str, KernelValue] | None = None
    ) -> str | None:
        """The element kind ``t[i]`` reads, or None when that is not settled.

        None also covers a runtime index into a tuple of mixed kinds and an
        out-of-range constant one; both are refused where the read is lowered,
        which is where there is something to say about them.
        """

        if not isinstance(node, ast.Subscript) or isinstance(node.slice, ast.Slice):
            return None
        try:
            kinds = self.tuple_kinds(self.expression_type(node.value, bindings))
        except NativeCompileError:
            return None
        if kinds is None:
            return None
        index = self.constant_index(node.slice)
        if index is not None:
            resolved = index + len(kinds) if index < 0 else index
            return kinds[resolved] if 0 <= resolved < len(kinds) else None
        return kinds[0] if kinds and len(set(kinds)) == 1 else None

    def tuple_pointer(
        self, node: ast.expr, bindings: dict[str, KernelValue] | None = None
    ) -> IntExpression:
        """An i64 pointer to a tuple block, building a literal if that is what
        this is."""

        if isinstance(node, ast.Name) and self.tuple_kinds_of(node.id) is not None:
            return IntLoad(self.slots[node.id])
        if isinstance(node, ast.Tuple):
            pointer_slot = self.new_temp()
            self.emit_tuple_literal(
                pointer_slot, node, self.tuple_literal_kinds(node, bindings)
            )
            return IntLoad(pointer_slot)
        raise NativeCompileError(
            self.path,
            node,
            "expression is not a native runtime tuple: a tuple comes from a "
            "literal or from a name bound to one. A native function returns a "
            "signed 64-bit integer, a float or a string, never a tuple, and "
            "there is no tuple slicing or concatenation",
        )

    def emit_tuple_literal(
        self, pointer_slot: int, node: ast.Tuple, kinds: tuple[str, ...]
    ) -> None:
        bump = self.ensure_heap()
        size = self.TUPLE_HEADER_BYTES + len(kinds) * 8
        self.operations.append(HeapAlloc(pointer_slot, IntConstant(size), bump))
        self.operations.append(
            HeapStore(IntLoad(pointer_slot), IntConstant(len(kinds)), 8)
        )
        for index, (element, kind) in enumerate(zip(node.elts, kinds)):
            # A string element allocates its own block while it is lowered, so
            # the word is built before the store that takes its address; the
            # arena only moves forward, so this block stays where it is.
            stored = self.element_word(element, kind)
            address = IntBinary(
                "add",
                IntLoad(pointer_slot),
                IntConstant(self.TUPLE_HEADER_BYTES + index * 8),
            )
            self.operations.append(HeapStore(address, stored, 8))

    def tuple_literal_texts(
        self, node: ast.Tuple, kinds: tuple[str, ...]
    ) -> tuple[bytes | None, ...]:
        """What print() must write for each element whose repr is settled now.

        Only a string needs this: an int, a float and a bool are rendered at
        run time by code that already matches CPython. A string's repr is
        computed here by the host, on the very same `str` object the target
        will hold, so it is exact by construction.
        """

        texts: list[bytes | None] = []
        for element, kind in zip(node.elts, kinds):
            if kind != "str":
                texts.append(None)
                continue
            try:
                folded = self.constant(element)
            except NativeCompileError:
                folded = None
            texts.append(
                repr(folded).encode("utf-8") if isinstance(folded, str) else None
            )
        return tuple(texts)

    def tuple_element_texts(
        self, node: ast.expr, kinds: tuple[str, ...]
    ) -> tuple[bytes | None, ...]:
        if isinstance(node, ast.Tuple):
            return self.tuple_literal_texts(node, kinds)
        if isinstance(node, ast.Name):
            recorded = self.tuple_texts.get(node.id)
            if recorded is not None and len(recorded) == len(kinds):
                return recorded
        return (None,) * len(kinds)

    def tuple_assignment(
        self, name: str, node: ast.expr, kinds: tuple[str, ...]
    ) -> None:
        self.runtime_names.add(name)
        self.boolean_names.discard(name)
        self.shared_list_names.discard(name)
        self.container_bool.pop(name, None)
        self.list_lengths.pop(name, None)
        if isinstance(node, ast.Tuple):
            self.tuple_texts[name] = self.tuple_literal_texts(node, kinds)
            # Build into a slot of its own and move it over afterwards. A
            # literal that reads the name it is being assigned to - `t = (t[0]
            # - 1, t[1])` - would otherwise take the new block's address out of
            # the name and read its uninitialised elements.
            pointer_slot = self.new_temp()
            self.emit_tuple_literal(pointer_slot, node, kinds)
            self.operations.append(Store(self.slot(name), IntLoad(pointer_slot)))
            return
        # A second name for the same block, which a list refuses: appending
        # moves a list and only the name it moves through learns the new
        # address, but a tuple has no append and never moves.
        self.tuple_texts[name] = self.tuple_element_texts(node, kinds)
        self.operations.append(Store(self.slot(name), self.tuple_pointer(node)))

    def tuple_element_address(self, node: ast.Subscript) -> IntExpression:
        target = node.value
        kinds = self.tuple_kinds(self.expression_type(target))
        assert kinds is not None
        pointer = self.materialize_int(self.tuple_pointer(target))
        index = self.constant_index(node.slice)
        if index is not None:
            resolved = index + len(kinds) if index < 0 else index
            if not 0 <= resolved < len(kinds):
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native tuple index {index} is out of range"
                    + (f" for {target.id!r}" if isinstance(target, ast.Name) else "")
                    + f" (length {len(kinds)})",
                )
            return IntBinary(
                "add",
                pointer,
                IntConstant(self.TUPLE_HEADER_BYTES + resolved * 8),
            )
        if len(set(kinds)) > 1:
            # The point of a tuple is that each element can be a different
            # kind, and the kind is what decides whether this reads an integer,
            # the bits of a double, or the address of a string block. A runtime
            # index has no kind, so there is no load to emit.
            raise NativeCompileError(
                self.path,
                node,
                "indexing a tuple whose elements are not all the same kind "
                f"needs a constant index (this one holds {', '.join(kinds)}); "
                "a runtime index is supported when every element is one kind",
            )
        if self.eager_depth:
            raise NativeCompileError(
                self.path,
                node,
                "a tuple index whose bounds check has to run at run time "
                "cannot appear in a conditional expression or a short-circuited "
                "Boolean operand, because the check would run even when Python "
                "would not evaluate that branch; use an if statement instead",
            )
        index_slot = self.resolve_tuple_index(self.integer(node.slice), len(kinds))
        offset = IntBinary(
            "add",
            IntConstant(self.TUPLE_HEADER_BYTES),
            IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
        )
        return IntBinary("add", pointer, offset)

    def resolve_tuple_index(self, index: IntExpression, length: int) -> int:
        """Normalize a runtime index against a build-time length and check it.

        Returns the slot holding the resolved index. The length never has to be
        read from the block: a tuple cannot grow or shrink, so how many
        elements it has is settled where it was written.
        """

        index = self.materialize_int(index)
        bad_label = self.new_label("tuple_index_error")
        ok_label = self.new_label("tuple_index_ok")
        index_slot = self.slot(f"<index-{bad_label}>")
        # Branchless Python negative indexing: index += length when index < 0.
        negative_mask = IntUnary("neg", IntCompare("lt", index, IntConstant(0)))
        self.operations.append(
            Store(
                index_slot,
                IntBinary(
                    "add",
                    index,
                    IntBinary("and", IntConstant(length), negative_mask),
                ),
            )
        )
        in_range = IntBinary(
            "and",
            IntCompare("ge", IntLoad(index_slot), IntConstant(0)),
            IntCompare("lt", IntLoad(index_slot), IntConstant(length)),
        )
        self.operations.append(JumpIfFalse(in_range, bad_label))
        self.operations.append(Jump(ok_label))
        self.operations.append(Label(bad_label))
        self.raise_exception(
            "IndexError", b"IndexError: tuple index out of range\n"
        )
        self.operations.append(Label(ok_label))
        return index_slot

    def unpacked_tuple_kinds(self, value: ast.expr) -> tuple[str, ...] | None:
        """The kinds `a, b = value` would bind, when value is a tuple."""

        try:
            return self.tuple_kinds(self.expression_type(value))
        except NativeCompileError:
            return None

    def tuple_unpacking(self, target: ast.expr, value: ast.expr) -> None:
        """`a, b = t` - bind one name per element, each with its own kind."""

        kinds = self.tuple_kinds(self.expression_type(value))
        assert kinds is not None
        for item in target.elts:
            if not isinstance(item, ast.Name):
                raise NativeCompileError(
                    self.path, item, "a native tuple unpacking binds names"
                )
        if len(target.elts) != len(kinds):
            raise NativeCompileError(
                self.path,
                target,
                f"native tuple unpacking needs matching lengths: "
                f"{len(target.elts)} names, {len(kinds)} values",
            )
        # Bind the right-hand side once. Reading it again per name would build
        # a literal once per element, and CPython evaluates it exactly once.
        holder = f"<unpack-{self.new_label('slot')}>"
        self.assignment(holder, value)
        for index, item in enumerate(target.elts):
            element = ast.copy_location(
                ast.Subscript(
                    value=ast.copy_location(
                        ast.Name(id=holder, ctx=ast.Load()), target
                    ),
                    slice=ast.copy_location(ast.Constant(value=index), target),
                    ctx=ast.Load(),
                ),
                target,
            )
            self.assignment(item.id, element)

    def bind_tuple_element(
        self, target: str, index_slot: int, pointer_slot: int, element_kind: str
    ) -> None:
        address = IntBinary(
            "add",
            IntBinary(
                "add", IntLoad(pointer_slot), IntConstant(self.TUPLE_HEADER_BYTES)
            ),
            IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
        )
        self.values.pop(target, None)
        self.runtime_names.add(target)
        self.possibly_unbound.discard(target)
        self.shared_list_names.discard(target)
        self.list_lengths.pop(target, None)
        self.tuple_texts.pop(target, None)
        if element_kind == "bool":
            self.boolean_names.add(target)
        else:
            self.boolean_names.discard(target)
        if element_kind == "float":
            self.operations.append(
                FloatStore(self.slot(target), BitsFloat(HeapLoad(address, 8)))
            )
        else:
            self.operations.append(Store(self.slot(target), HeapLoad(address, 8)))
        self.value_types[target] = self.element_value_type(element_kind)

    def for_over_tuple(self, node: ast.For) -> None:
        """`for name in <tuple>:` - a walk of a length fixed at build time."""

        assert isinstance(node.target, ast.Name)
        kinds = self.tuple_kinds(self.expression_type(node.iter))
        assert kinds is not None
        if len(set(kinds)) > 1:
            raise NativeCompileError(
                self.path,
                node.iter,
                "iterating a tuple needs every element to be the same kind, "
                "because the loop name is one stack slot and nothing at run "
                f"time says what a slot holds (this one holds {', '.join(kinds)}); "
                "read the elements by constant index instead",
            )
        broke = self.open_loop_else(node)
        name = node.target.id
        was_bound = name in self.bound_names
        pointer_slot = self.new_temp()
        self.operations.append(Store(pointer_slot, self.tuple_pointer(node.iter)))
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        element_kind = kinds[0] if kinds else "int"
        start = self.new_label("for_tuple")
        continue_label = self.new_label("for_tuple_continue")
        end = self.new_label("for_tuple_end")
        self.operations.append(Label(start))
        # A list iterator re-reads the list's length every step so that an
        # append inside the loop extends the walk. A tuple has no append, so
        # its length is a constant here and there is nothing to re-read.
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "lt", IntLoad(index_slot), IntConstant(len(kinds))
                ),
                end,
            )
        )
        self.bind_tuple_element(name, index_slot, pointer_slot, element_kind)
        self.break_targets.append(end if broke is None else broke)
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
            # An empty tuple runs the body zero times, and then Python leaves
            # the name unbound.
            self.possibly_unbound.add(name)
        self.close_loop_else(node, broke)

    def emit_word_to_string(self, word: IntExpression, kind: str) -> IntExpression:
        """The text CPython's repr gives one stored word of this kind."""

        if kind == "bool":
            return self.emit_bool_to_string(word)
        if kind == "float":
            return self.emit_float_to_string(BitsFloat(word))
        return self.emit_int_to_string(word)

    def emit_dict_to_string(self, node: ast.expr) -> IntExpression:
        """Build the text CPython's repr gives a dict; returns a string block.

        The walk is the insertion-order key list, the same one `for k in d`
        uses, which is what makes the printed order CPython's. The value is
        found by probing rather than by remembering an address, because growth
        moves entries and nothing here pins them.
        """

        assert isinstance(node, ast.Name)
        name = node.id
        key_kind, value_kind = self.dict_kinds_of(name)
        if self.container_bool.get(self.dict_keys_name(name)) is True:
            key_kind = "bool"
        if self.container_bool.get(name) is True:
            value_kind = "bool"
        dict_slot = self.slot(name)
        pointer = IntLoad(dict_slot)
        keys_slot = self.new_temp()
        self.operations.append(
            Store(
                keys_slot,
                HeapLoad(
                    IntBinary("add", pointer, IntConstant(self.DICT_KEYS_OFFSET)), 8
                ),
            )
        )
        result_slot = self.new_temp()
        self.operations.append(
            Store(result_slot, self.materialize_string_constant(b"{"))
        )
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        start = self.new_label("print_dict")
        end = self.new_label("print_dict_end")
        first = self.new_label("print_dict_first")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "lt",
                    IntLoad(index_slot),
                    HeapLoad(
                        IntBinary("add", IntLoad(keys_slot), IntConstant(8)), 8
                    ),
                ),
                end,
            )
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare("gt", IntLoad(index_slot), IntConstant(0)), first
            )
        )
        self.operations.append(
            Store(
                result_slot,
                self.emit_concat(
                    IntLoad(result_slot), self.materialize_string_constant(b", ")
                ),
            )
        )
        self.operations.append(Label(first))
        key_slot = self.new_temp()
        self.operations.append(
            Store(
                key_slot,
                HeapLoad(
                    IntBinary(
                        "add",
                        IntBinary(
                            "add",
                            IntLoad(keys_slot),
                            IntConstant(self.LIST_HEADER_BYTES),
                        ),
                        IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
                    ),
                    8,
                ),
            )
        )
        self.operations.append(
            Store(
                result_slot,
                self.emit_concat(
                    IntLoad(result_slot),
                    self.emit_word_to_string(IntLoad(key_slot), key_kind),
                ),
            )
        )
        self.operations.append(
            Store(
                result_slot,
                self.emit_concat(
                    IntLoad(result_slot), self.materialize_string_constant(b": ")
                ),
            )
        )
        address_slot, _found, _key, _state = self.dict_probe(
            dict_slot, IntLoad(key_slot), self.dict_kinds_of(name)[0]
        )
        value = HeapLoad(
            IntBinary("add", IntLoad(address_slot), IntConstant(16)), 8
        )
        self.operations.append(
            Store(
                result_slot,
                self.emit_concat(
                    IntLoad(result_slot),
                    self.emit_word_to_string(value, value_kind),
                ),
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
        self.operations.append(
            Store(
                result_slot,
                self.emit_concat(
                    IntLoad(result_slot), self.materialize_string_constant(b"}")
                ),
            )
        )
        return IntLoad(result_slot)

    def emit_print_list(self, node: ast.expr, element_kind: str) -> None:
        """Write a whole list the way CPython's repr does."""

        text = self.materialize_int(self.emit_list_to_string(node, element_kind))
        self.operations.append(
            WriteRuntime(IntBinary("add", text, IntConstant(8)), HeapLoad(text, 8))
        )

    def emit_list_to_string(
        self, node: ast.expr, element_kind: str
    ) -> IntExpression:
        """Build the text CPython's repr gives a list; returns a string block.

        Unlike a tuple, the length is only known at run time, so the elements
        are walked rather than unrolled and the separator needs a branch: a
        comma goes before every element except the first.

        print() goes through this too rather than writing each piece straight
        out. Writing would save the allocations, but then an f-string and a
        print would be two implementations of one format, free to drift apart -
        and the cost of that is a wrong answer, while the cost of this is arena
        space the guard already reports honestly.
        """

        result_slot = self.new_temp()
        self.operations.append(
            Store(result_slot, self.materialize_string_constant(b"["))
        )
        pointer = self.materialize_int(self.list_pointer(node))
        length = self.materialize_int(
            HeapLoad(IntBinary("add", pointer, IntConstant(8)), 8)
        )
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        start = self.new_label("print_list")
        end = self.new_label("print_list_end")
        first = self.new_label("print_list_first")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(IntCompare("lt", IntLoad(index_slot), length), end)
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare("gt", IntLoad(index_slot), IntConstant(0)), first
            )
        )
        self.operations.append(
            Store(
                result_slot,
                self.emit_concat(
                    IntLoad(result_slot), self.materialize_string_constant(b", ")
                ),
            )
        )
        self.operations.append(Label(first))
        word = HeapLoad(
            IntBinary(
                "add",
                IntBinary("add", pointer, IntConstant(self.LIST_HEADER_BYTES)),
                IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
            ),
            8,
        )
        if element_kind == "bool":
            text = self.emit_bool_to_string(word)
        elif element_kind == "float":
            text = self.emit_float_to_string(BitsFloat(word))
        else:
            text = self.emit_int_to_string(word)
        self.operations.append(
            Store(result_slot, self.emit_concat(IntLoad(result_slot), text))
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
        self.operations.append(
            Store(
                result_slot,
                self.emit_concat(
                    IntLoad(result_slot), self.materialize_string_constant(b"]")
                ),
            )
        )
        return IntLoad(result_slot)

    def emit_print_tuple(self, node: ast.expr, kinds: tuple[str, ...]) -> None:
        """Write a whole tuple the way CPython's repr does."""

        texts = self.tuple_element_texts(node, kinds)
        for index, kind in enumerate(kinds):
            if kind == "str" and texts[index] is None:
                raise NativeCompileError(
                    self.path,
                    node,
                    "native print() renders a tuple whose string elements are "
                    "known at build time, because CPython prints the repr of "
                    "every element and picking the quote character and the "
                    "backslash escapes for a string built at run time is not "
                    "implemented here; print that element on its own, where it "
                    "is written as itself. Integers, floats and bools in a "
                    "tuple are supported",
                )
        # Nothing is loaded when every element is a build-time string, so there
        # is no reason to build the block for one.
        pointer = (
            self.materialize_int(self.tuple_pointer(node))
            if any(kind != "str" for kind in kinds)
            else None
        )
        self.operations.append(Write(b"("))
        for index, kind in enumerate(kinds):
            if index:
                self.operations.append(Write(b", "))
            if kind == "str":
                self.operations.append(Write(texts[index]))
                continue
            word = HeapLoad(
                IntBinary(
                    "add",
                    pointer,
                    IntConstant(self.TUPLE_HEADER_BYTES + index * 8),
                ),
                8,
            )
            if kind == "bool":
                text = self.emit_bool_to_string(word)
            elif kind == "float":
                text = self.emit_float_to_string(BitsFloat(word))
            else:
                text = self.emit_int_to_string(word)
            text = self.materialize_int(text)
            self.operations.append(
                WriteRuntime(
                    IntBinary("add", text, IntConstant(8)), HeapLoad(text, 8)
                )
            )
        # `(1,)` is a tuple and `(1)` is an integer, so CPython's repr keeps the
        # comma that says which.
        self.operations.append(Write(b",)" if len(kinds) == 1 else b")"))

    # --- runtime dictionaries -----------------------------------------------
    #
    # Layout: [capacity][count][keys][used] then `capacity` slots of
    # [state][key][value], 24 bytes each. A state of 0 means the slot is empty
    # and a state of 2 means it held a key that was deleted; collisions are
    # resolved by linear probing, and the table rehashes once it passes half
    # full - into twice the capacity, or into the same capacity when what
    # filled it was tombstones rather than keys.
    #
    # The fourth header word is live keys plus tombstones, which is what "half
    # full" has to mean: a probe stops at an empty slot, and a table with none
    # left would never stop however few of its keys were live.
    #
    # The third header word points at a runtime list of the keys in the order
    # they were first stored. The table itself cannot answer that question -
    # walking it yields hash order, and CPython iterates in insertion order -
    # so the order is recorded as it happens and iteration walks the list,
    # looking each key back up in the table.
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
    DICT_HEADER_BYTES = 32
    DICT_KEYS_OFFSET = 16
    # Live entries plus tombstones. Growth has to watch this rather than the
    # count, because a table that is all tombstones has nowhere left for a
    # probe to stop and would spin forever.
    DICT_USED_OFFSET = 24
    # A third state word, meaning "empty now, but a probe must keep walking".
    # A live state is 1 for an int key and an odd hash for a str key, so no
    # even nonzero word can be mistaken for one.
    DICT_TOMBSTONE = 2
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
        kinds = self.dict_kinds(self.value_types.get(name))
        if kinds is not None:
            # Asked wherever a name is used as a dict, which is the one place
            # every read of one passes through.
            self.refuse_unbound(name)
        return kinds

    @staticmethod
    def dict_keys_name(name: str) -> str:
        """The bookkeeping name under which a dict's KEYS are bools or not.

        The values already answer to ``name``, and keys need their own answer:
        `{True: 1}` iterates to True, not to 1.
        """

        return f"{name}<keys>"

    def emit_dict_key_order_block(
        self, pointer_slot: int, entries: int, bump: int
    ) -> None:
        """Give a fresh dict its insertion-order list, empty."""

        keys_slot = self.new_temp()
        capacity = max(4, entries)
        self.operations.append(
            HeapAlloc(
                keys_slot,
                IntConstant(self.LIST_HEADER_BYTES + capacity * 8),
                bump,
            )
        )
        self.operations.append(
            HeapStore(IntLoad(keys_slot), IntConstant(capacity), 8)
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", IntLoad(keys_slot), IntConstant(8)),
                IntConstant(0),
                8,
            )
        )
        self.operations.append(
            HeapStore(
                IntBinary(
                    "add", IntLoad(pointer_slot), IntConstant(self.DICT_KEYS_OFFSET)
                ),
                IntLoad(keys_slot),
                8,
            )
        )

    def dict_capacity(self, entries: int) -> int:
        """A power-of-two capacity with room to spare, so probing terminates."""

        capacity = 8
        while capacity < (entries + 1) * 4:
            capacity *= 2
        return capacity

    def dict_get_shape(self, node: ast.expr) -> tuple[str, str, str] | None:
        """`d.get(...)` as (name, key kind, value kind), or None."""

        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or node.func.attr != "get"
            or not isinstance(node.func.value, ast.Name)
        ):
            return None
        kinds = self.dict_kinds_of(node.func.value.id)
        if kinds is None:
            return None
        return (node.func.value.id, kinds[0], kinds[1])

    def emit_dict_get(
        self, node: ast.Call, bindings: dict[str, IntExpression] | None = None
    ) -> IntExpression:
        """`d.get(k, default)` - the stored word, or the default word.

        The two-argument form only. `d.get(k)` answers None when the key is
        absent, and there is no None here to answer with; a default that is
        never used is better than a value that cannot be represented.
        """

        shape = self.dict_get_shape(node)
        assert shape is not None
        name, key_kind, value_kind = shape
        if len(node.args) != 2 or node.keywords:
            raise NativeCompileError(
                self.path,
                node,
                "native dict.get() takes a key and a default; the one-argument "
                "form answers None when the key is absent, and None is not in "
                "the subset",
            )
        if self.expression_type(node.args[0], bindings) != key_kind:
            raise NativeCompileError(
                self.path, node.args[0], f"this dict has {key_kind} keys"
            )
        if self.expression_type(node.args[1], bindings) != value_kind:
            raise NativeCompileError(
                self.path,
                node.args[1],
                f"this dict has {value_kind} values, so the default must be "
                f"a {value_kind}",
            )
        result = self.new_temp()
        # Before the probe: Python evaluates both arguments whether or not the
        # key turns out to be there.
        self.operations.append(
            Store(result, self.dict_value(node.args[1], value_kind))
        )
        address_slot, found_slot, _key, _state = self.dict_probe(
            self.slot(name), self.dict_key(node.args[0], key_kind), key_kind
        )
        end = self.new_label("dict_get_end")
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(found_slot), IntConstant(0)), end
            )
        )
        self.operations.append(
            Store(
                result,
                HeapLoad(
                    IntBinary("add", IntLoad(address_slot), IntConstant(16)), 8
                ),
            )
        )
        self.operations.append(Label(end))
        return IntLoad(result)

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
            if key_kind == "int":
                self.note_stored_bool(
                    self.dict_keys_name(name), key, "this dict's keys"
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
            else:
                self.note_stored_bool(name, value, "this dict")
        capacity = self.dict_capacity(len(node.keys))
        bump = self.ensure_heap()
        self.runtime_names.add(name)
        # Build into a slot of its own and move it over once it is filled. A
        # literal that reads the name it is being assigned to - `d = {0: d[0] +
        # 1}` - would otherwise look the old key up in the new, empty table.
        pointer_slot = self.new_temp()
        size = self.DICT_HEADER_BYTES + capacity * self.DICT_SLOT_BYTES
        self.operations.append(HeapAlloc(pointer_slot, IntConstant(size), bump))
        pointer = IntLoad(pointer_slot)
        self.operations.append(HeapStore(pointer, IntConstant(capacity), 8))
        self.operations.append(
            HeapStore(
                IntBinary("add", pointer, IntConstant(8)), IntConstant(0), 8
            )
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", pointer, IntConstant(self.DICT_USED_OFFSET)),
                IntConstant(0),
                8,
            )
        )
        self.emit_dict_key_order_block(pointer_slot, len(node.keys), bump)
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
        # dict_store rewrites pointer_slot when the table grows, so the address
        # the name gets is read after the last store rather than before it.
        self.operations.append(Store(self.slot(name), IntLoad(pointer_slot)))

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

    def emit_string_order(self, left: IntExpression, right: IntExpression) -> int:
        """Compare two string blocks; returns a slot holding -1, 0 or 1.

        Byte by byte, which is also code point by code point: UTF-8 was built
        so that comparing the bytes of two sequences puts them in the same
        order as comparing the code points, which is the order CPython uses.
        Bytes are unsigned here, so a continuation byte does not read as
        negative and sort before ASCII.
        """

        left_slot = self.new_temp()
        right_slot = self.new_temp()
        self.operations.append(Store(left_slot, left))
        self.operations.append(Store(right_slot, right))
        left_length = HeapLoad(IntLoad(left_slot), 8)
        right_length = HeapLoad(IntLoad(right_slot), 8)
        shortest = self.new_temp()
        self.operations.append(
            Store(
                shortest,
                self.select_integer(
                    IntCompare("lt", left_length, right_length),
                    left_length,
                    right_length,
                ),
            )
        )
        result = self.new_temp()
        index = self.new_temp()
        self.operations.append(Store(result, IntConstant(0)))
        self.operations.append(Store(index, IntConstant(0)))
        scan = self.new_label("strcmp")
        done = self.new_label("strcmp_done")
        tail = self.new_label("strcmp_tail")
        self.operations.append(Label(scan))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index), IntLoad(shortest)), tail
            )
        )
        def byte_at(slot: int) -> IntExpression:
            return HeapLoad(
                IntBinary(
                    "add",
                    IntBinary("add", IntLoad(slot), IntConstant(8)),
                    IntLoad(index),
                ),
                1,
            )

        differs = self.new_label("strcmp_differs")
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", byte_at(left_slot), byte_at(right_slot)), differs
            )
        )
        self.operations.append(
            Store(
                result,
                self.select_integer(
                    IntCompare("lt", byte_at(left_slot), byte_at(right_slot)),
                    IntConstant(-1),
                    IntConstant(1),
                ),
            )
        )
        self.operations.append(Jump(done))
        self.operations.append(Label(differs))
        self.operations.append(
            Store(index, IntBinary("add", IntLoad(index), IntConstant(1)))
        )
        self.operations.append(Jump(scan))
        # One is a prefix of the other, or they are equal: the shorter sorts
        # first, which is what comparing the lengths says.
        self.operations.append(Label(tail))
        self.operations.append(
            Store(
                result,
                self.select_integer(
                    IntCompare("lt", left_length, right_length),
                    IntConstant(-1),
                    self.select_integer(
                        IntCompare("gt", left_length, right_length),
                        IntConstant(1),
                        IntConstant(0),
                    ),
                ),
            )
        )
        self.operations.append(Label(done))
        return result

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

        When the key is absent the address is the first tombstone the probe
        walked over, or the empty slot it stopped at if there was none, so that
        an insert refills a deleted slot rather than pushing the table towards
        having nowhere to stop.
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
        # No entry address is ever 0, so 0 means "no tombstone seen yet".
        reuse_slot = self.new_temp()
        self.operations.append(Store(reuse_slot, IntConstant(0)))
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
        # A tombstone holds no key to compare, and does not end the probe; it
        # is only remembered as somewhere an insert may go.
        live = self.new_label("probe_live")
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", state, IntConstant(self.DICT_TOMBSTONE)), live
            )
        )
        self.operations.append(Jump(live + "_alive"))
        self.operations.append(Label(live))
        self.operations.append(
            Store(
                reuse_slot,
                self.select_integer(
                    IntCompare("eq", IntLoad(reuse_slot), IntConstant(0)),
                    IntLoad(address_slot),
                    IntLoad(reuse_slot),
                ),
            )
        )
        self.operations.append(Jump(step))
        self.operations.append(Label(live + "_alive"))
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
        # An absent key belongs in the first tombstone walked over rather than
        # in the empty slot past it. A found key keeps its own address.
        reuse = self.new_label("probe_reuse")
        self.operations.append(
            JumpIfFalse(IntCompare("eq", IntLoad(found_slot), IntConstant(0)), reuse)
        )
        self.operations.append(
            Store(
                address_slot,
                self.select_integer(
                    IntCompare("ne", IntLoad(reuse_slot), IntConstant(0)),
                    IntLoad(reuse_slot),
                    IntLoad(address_slot),
                ),
            )
        )
        self.operations.append(Label(reuse))
        return address_slot, found_slot, key_slot, state_slot

    def dict_grow(self, pointer_slot: int, key_kind: str) -> None:
        """Rehash into a fresh table, twice the size or the same size.

        A hash table cannot simply be extended: every entry's home slot depends
        on the capacity, so growth means allocating a new table and probing each
        live entry into it. The old table is left in the arena, which never
        reclaims, and that is the documented cost of an arena.

        Tombstones are what makes the same size an option. A table filled with
        them has run out of room for a probe to stop even though few keys are
        live, and rehashing at the same capacity drops them all. That is also
        what makes a delete inside a loop allocate: one table per capacity/2
        deletions, the same shape of cost as sorting inside a loop, and it ends
        in MemoryError rather than in a wrong answer.
        """

        bump = self.ensure_heap()
        old_slot = self.new_temp()
        self.operations.append(Store(old_slot, IntLoad(pointer_slot)))
        old = IntLoad(old_slot)
        old_capacity_slot = self.new_temp()
        self.operations.append(Store(old_capacity_slot, HeapLoad(old, 8)))
        # Room for one more live entry in a quarter of the table means the
        # tombstones were the problem, not the keys. Without a delete the count
        # equals the used word, this test is always false, and the table
        # doubles exactly as it did before tombstones existed.
        new_capacity_slot = self.new_temp()
        self.operations.append(
            Store(
                new_capacity_slot,
                self.select_integer(
                    IntCompare(
                        "le",
                        IntBinary(
                            "mul",
                            IntBinary(
                                "add",
                                HeapLoad(IntBinary("add", old, IntConstant(8)), 8),
                                IntConstant(1),
                            ),
                            IntConstant(4),
                        ),
                        IntLoad(old_capacity_slot),
                    ),
                    IntLoad(old_capacity_slot),
                    IntBinary("mul", IntLoad(old_capacity_slot), IntConstant(2)),
                ),
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
        # The rehash below scatters the entries into new home slots, so the
        # insertion order lives only in this list; carry it over or it is lost.
        self.operations.append(
            HeapStore(
                IntBinary("add", new, IntConstant(self.DICT_KEYS_OFFSET)),
                HeapLoad(
                    IntBinary("add", old, IntConstant(self.DICT_KEYS_OFFSET)), 8
                ),
                8,
            )
        )
        # Only live entries make the move, so the new table starts with no
        # tombstones and its used word is its count.
        self.operations.append(
            HeapStore(
                IntBinary("add", new, IntConstant(self.DICT_USED_OFFSET)),
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
        # A tombstone's key word is whatever the deleted entry left behind, so
        # carrying one over would resurrect it as an entry with a stale key.
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "ne", IntLoad(state_slot), IntConstant(self.DICT_TOMBSTONE)
                ),
                step,
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
        order: bool = True,
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
        # Tombstones count towards full as much as live keys do, because what
        # keeps a probe terminating is a slot it can stop at, not a free key.
        used = HeapLoad(IntBinary("add", pointer, IntConstant(self.DICT_USED_OFFSET)), 8)
        capacity = HeapLoad(pointer, 8)
        room = self.new_label("dict_has_room")
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "lt",
                    IntBinary("mul", used, IntConstant(2)),
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
        # Refilling a tombstone spends no new slot: one fewer tombstone, one
        # more live key. Only a slot that was empty raises the used word.
        fresh = self.new_label("dict_fresh_slot")
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "eq", HeapLoad(IntLoad(address_slot), 8), IntConstant(0)
                ),
                fresh,
            )
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", pointer, IntConstant(self.DICT_USED_OFFSET)),
                IntBinary("add", used, IntConstant(1)),
                8,
            )
        )
        self.operations.append(Label(fresh))
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
        # Only a key that was not already here joins the order list, which is
        # why this sits in the new-entry arm: replacing a value must not move a
        # key, and must not list it twice. A set passes order=False: nothing
        # may read a set's order, so keeping one would only be a list that
        # grows with every add and that discard would leave stale.
        if order:
            keys_slot = self.new_temp()
            keys_field = IntBinary(
                "add", pointer, IntConstant(self.DICT_KEYS_OFFSET)
            )
            self.operations.append(Store(keys_slot, HeapLoad(keys_field, 8)))
            self.emit_list_append(keys_slot, IntLoad(key_slot))
            self.operations.append(HeapStore(keys_field, IntLoad(keys_slot), 8))
        self.operations.append(Label(existing))
        self.operations.append(
            HeapStore(
                IntBinary("add", IntLoad(address_slot), IntConstant(16)),
                IntLoad(value_slot),
                8,
            )
        )
        self.operations.append(Label(end))

    # --- runtime sets -------------------------------------------------------

    # A set is a dict block with the value word never written and never read,
    # and with the insertion-order field left 0 because nothing may read a
    # set's order. Sharing the block means sharing dict_probe, dict_grow and
    # dict_store, so a set's probing, growth and rehashing are the same code
    # the dicts have already been tested through rather than a second table
    # written from scratch. The price is eight wasted bytes per slot in an
    # arena that never reclaims anyway.

    SET_ELEMENT_KINDS = {"int", "str"}

    @staticmethod
    def set_tag(element_kind: str) -> str:
        return f"set:{element_kind}"

    @staticmethod
    def set_kind(tag: str | None) -> str | None:
        """The element kind of a set tag, else None."""

        if not isinstance(tag, str) or not tag.startswith("set:"):
            return None
        return tag.split(":", 1)[1]

    def set_kind_of(self, name: str) -> str | None:
        kind = self.set_kind(self.value_types.get(name))
        if kind is not None:
            self.refuse_unbound(name)
        return kind

    def set_literal_tag(
        self, node: ast.Set, bindings: dict[str, IntExpression] | None = None
    ) -> str:
        """The kind a set literal builds, read off its first element."""

        element_kind = self.expression_type(node.elts[0], bindings)
        if element_kind not in self.SET_ELEMENT_KINDS:
            raise NativeCompileError(
                self.path,
                node.elts[0],
                "native set elements are signed 64-bit integers or runtime "
                "strings",
            )
        return self.set_tag(element_kind)

    def annotated_set_tag(self, annotation: ast.expr) -> str | None:
        """The kind `set[T]` names, or None if it is not that shape."""

        if (
            not isinstance(annotation, ast.Subscript)
            or not isinstance(annotation.value, ast.Name)
            or annotation.value.id not in {"set", "Set"}
            or not isinstance(annotation.slice, ast.Name)
        ):
            return None
        element_kind = annotation.slice.id
        if element_kind not in self.SET_ELEMENT_KINDS:
            raise NativeCompileError(
                self.path, annotation, "a native set annotation is set[int|str]"
            )
        return self.set_tag(element_kind)

    def empty_set_call(self, node: ast.expr) -> bool:
        """Whether this is `set()`, the only way to write an empty set."""

        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set"
            and node.func.id not in self.functions
            and node.func.id not in self.slots
        )

    _SET_OPERATORS = {
        ast.BitOr: "union",
        ast.BitAnd: "intersection",
        ast.Sub: "difference",
    }

    def set_binop_operands(self, node: ast.expr) -> tuple[str, str, str] | None:
        """`(left name, right name, element kind)` for `a | b`, `a & b`, `a - b`.

        None when this is not a set operation at all; a build error when one
        side is a set and the other is not, because `s - 1` has no meaning and
        falling through would lower the set's slot as a number.
        """

        if not isinstance(node, ast.BinOp) or type(node.op) not in self._SET_OPERATORS:
            return None
        left_kind = (
            self.set_kind_of(node.left.id)
            if isinstance(node.left, ast.Name)
            else None
        )
        right_kind = (
            self.set_kind_of(node.right.id)
            if isinstance(node.right, ast.Name)
            else None
        )
        if left_kind is None and right_kind is None:
            return None
        if left_kind is None or right_kind is None:
            raise NativeCompileError(
                self.path,
                node,
                "a native set operation has a set on both sides; | & and - "
                "combine two set variables of the same element kind",
            )
        if left_kind != right_kind:
            raise NativeCompileError(
                self.path,
                node,
                f"these sets hold different kinds ({left_kind} and "
                f"{right_kind}), so they cannot be combined",
            )
        assert isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name)
        return node.left.id, node.right.id, left_kind

    def emit_set_block(self, capacity: IntExpression, bump: int) -> int:
        """Allocate a zeroed set table and return the slot holding it."""

        pointer_slot = self.new_temp()
        self.operations.append(
            HeapAlloc(
                pointer_slot,
                IntBinary(
                    "add",
                    IntConstant(self.DICT_HEADER_BYTES),
                    IntBinary("mul", capacity, IntConstant(self.DICT_SLOT_BYTES)),
                ),
                bump,
            )
        )
        pointer = IntLoad(pointer_slot)
        self.operations.append(HeapStore(pointer, capacity, 8))
        self.operations.append(
            HeapStore(IntBinary("add", pointer, IntConstant(8)), IntConstant(0), 8)
        )
        # The order field stays 0 for a set. dict_grow copies it across, and 0
        # copies to 0, so growth never looks for a list that is not there.
        self.operations.append(
            HeapStore(
                IntBinary("add", pointer, IntConstant(self.DICT_KEYS_OFFSET)),
                IntConstant(0),
                8,
            )
        )
        # A set is removed from by rebuilding, so it never holds a tombstone
        # and its used word only ever tracks its count. dict_store maintains it
        # either way.
        self.operations.append(
            HeapStore(
                IntBinary("add", pointer, IntConstant(self.DICT_USED_OFFSET)),
                IntConstant(0),
                8,
            )
        )
        # HeapAlloc hands back fresh arena memory, which the kernel zero-filled,
        # so every state field already reads as empty.
        return pointer_slot

    def emit_set_scan(self, source_slot: int, emit_body) -> None:
        """Walk every live entry of the set in ``source_slot``.

        ``emit_body`` is called with the slot holding the entry address and the
        slot holding its element. The source is read from its slot on every
        iteration, so nothing the body does may move it - the destinations here
        are always a different block.
        """

        capacity_slot = self.new_temp()
        self.operations.append(
            Store(capacity_slot, HeapLoad(IntLoad(source_slot), 8))
        )
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        scan = self.new_label("set_scan")
        after = self.new_label("set_scan_done")
        step = self.new_label("set_scan_next")
        self.operations.append(Label(scan))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(capacity_slot)),
                after,
            )
        )
        entry_slot = self.new_temp()
        self.operations.append(
            Store(
                entry_slot,
                IntBinary(
                    "add",
                    IntLoad(source_slot),
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
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", HeapLoad(IntLoad(entry_slot), 8), IntConstant(0)),
                step,
            )
        )
        element_slot = self.new_temp()
        self.operations.append(
            Store(
                element_slot,
                HeapLoad(IntBinary("add", IntLoad(entry_slot), IntConstant(8)), 8),
            )
        )
        emit_body(entry_slot, element_slot)
        self.operations.append(Label(step))
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(scan))
        self.operations.append(Label(after))

    def emit_set_scan_add(
        self,
        destination_slot: int,
        source_slot: int,
        key_kind: str,
        node: ast.AST,
        filter_slot: int | None = None,
        want_found: int = 1,
        exclude_slot: int | None = None,
    ) -> None:
        """Copy the elements of one set into another, optionally filtered.

        With ``filter_slot`` an element joins only when probing that set
        answers ``want_found``, which is intersection and difference. With
        ``exclude_slot`` the one entry at that address is left behind, which is
        discard. The result depends on which elements are copied and never on
        the order they are copied in, so nothing here observes a set's order.
        """

        def body(entry_slot: int, element_slot: int) -> None:
            skip = self.new_label("set_copy_skip")
            if exclude_slot is not None:
                self.operations.append(
                    JumpIfFalse(
                        IntCompare(
                            "ne", IntLoad(entry_slot), IntLoad(exclude_slot)
                        ),
                        skip,
                    )
                )
            if filter_slot is not None:
                _address, found_slot, _key, _state = self.dict_probe(
                    filter_slot, IntLoad(element_slot), key_kind
                )
                self.operations.append(
                    JumpIfFalse(
                        IntCompare(
                            "eq", IntLoad(found_slot), IntConstant(want_found)
                        ),
                        skip,
                    )
                )
            self.dict_store(
                destination_slot,
                IntLoad(element_slot),
                IntConstant(0),
                node,
                key_kind,
                order=False,
            )
            self.operations.append(Label(skip))

        self.emit_set_scan(source_slot, body)

    def set_binop(self, node: ast.BinOp) -> int:
        """Build the set `a | b`, `a & b` or `a - b` into a fresh block."""

        operands = self.set_binop_operands(node)
        assert operands is not None
        left, right, key_kind = operands
        bump = self.ensure_heap()
        # A fresh block every time, and never one of the operands: dict_store
        # relocates on growth, and scanning a block that moves underneath the
        # scan would read the abandoned one. This is also what makes `a |= b`
        # and `a |= a` safe.
        destination_slot = self.emit_set_block(IntConstant(8), bump)
        left_slot = self.new_temp()
        self.operations.append(Store(left_slot, IntLoad(self.slots[left])))
        right_slot = self.new_temp()
        self.operations.append(Store(right_slot, IntLoad(self.slots[right])))
        operation = self._SET_OPERATORS[type(node.op)]
        if operation == "union":
            self.emit_set_scan_add(destination_slot, left_slot, key_kind, node)
            self.emit_set_scan_add(destination_slot, right_slot, key_kind, node)
        else:
            self.emit_set_scan_add(
                destination_slot,
                left_slot,
                key_kind,
                node,
                filter_slot=right_slot,
                want_found=1 if operation == "intersection" else 0,
            )
        return destination_slot

    def set_result_holds_bool(self, node: ast.expr, name: str) -> bool | None:
        """Whether the set this expression builds holds bools.

        A set built from two others inherits their answer, and two operands
        that disagree have no single answer, so they are refused rather than
        guessed at.
        """

        operands = self.set_binop_operands(node)
        if operands is None:
            return None
        left, right, _kind = operands
        left_bool = self.container_bool.get(left)
        right_bool = self.container_bool.get(right)
        if left_bool is None:
            return right_bool
        if right_bool is None:
            return left_bool
        if left_bool != right_bool:
            raise NativeCompileError(
                self.path,
                node,
                f"one of these sets holds bools and the other holds numbers, "
                f"so {name!r} could not print either way; nothing at run time "
                "tells True from 1",
            )
        return left_bool

    def set_assignment(self, name: str, node: ast.expr, element_kind: str) -> None:
        if isinstance(node, ast.Name) and self.set_kind_of(node.id) is not None:
            raise NativeCompileError(
                self.path,
                node,
                f"a native set variable holds the block itself, not a "
                f"reference to it, so {name!r} cannot be another name for "
                f"{node.id!r}: adding to one moves the block and only one of "
                f"them would follow it. Write {name} = {node.id} | {node.id} "
                "if a second set is what you want",
            )
        if self.set_binop_operands(node) is not None:
            assert isinstance(node, ast.BinOp)
            holds_bool = self.set_result_holds_bool(node, name)
            pointer_slot = self.set_binop(node)
            self.runtime_names.add(name)
            if holds_bool is None:
                self.container_bool.pop(name, None)
            else:
                self.container_bool[name] = holds_bool
            self.operations.append(Store(self.slot(name), IntLoad(pointer_slot)))
            return
        if self.empty_set_call(node):
            assert isinstance(node, ast.Call)
            if node.args or node.keywords:
                raise NativeCompileError(
                    self.path,
                    node,
                    "native set() takes no arguments; build a set from a "
                    "literal such as {1, 2} or add() to an empty one",
                )
            self.container_bool.pop(name, None)
            bump = self.ensure_heap()
            self.runtime_names.add(name)
            pointer_slot = self.emit_set_block(IntConstant(8), bump)
            self.operations.append(Store(self.slot(name), IntLoad(pointer_slot)))
            return
        if not isinstance(node, ast.Set):
            raise NativeCompileError(
                self.path,
                node,
                "a native set variable is built from a set literal, from "
                "set(), or from | & - over two set variables",
            )
        for element in node.elts:
            if self.expression_type(element) != element_kind:
                raise NativeCompileError(
                    self.path,
                    element,
                    f"this set holds {element_kind} elements, so every element "
                    f"must be {element_kind}",
                )
            if element_kind == "int":
                # `{True, 1}` is one element in CPython because True == 1, and
                # two here, because the probe cannot tell them apart either -
                # but then one of them has to print wrongly. Refuse the mix.
                # The answer belongs to the NAME, not to this literal: rebinding
                # the name to a set of the other kind is refused too, because a
                # rebinding inside an `if` would leave the build-time answer
                # disagreeing with what the slot holds at run time.
                self.note_stored_bool(name, element, f"the set {name!r}")
        capacity = self.dict_capacity(len(node.elts))
        bump = self.ensure_heap()
        self.runtime_names.add(name)
        # Build into a slot of its own and move it over once it is filled, so
        # that a literal reading the name it is assigned to reads the old set.
        pointer_slot = self.emit_set_block(IntConstant(capacity), bump)
        for element in node.elts:
            self.dict_store(
                pointer_slot,
                self.dict_key(element, element_kind),
                IntConstant(0),
                node,
                element_kind,
                order=False,
            )
        self.operations.append(Store(self.slot(name), IntLoad(pointer_slot)))

    def set_method_call(self, node: ast.Call) -> bool:
        """Lower `s.add(v)`, `s.discard(v)` and `s.remove(v)`."""

        if not isinstance(node.func, ast.Attribute):
            return False
        if not isinstance(node.func.value, ast.Name):
            return False
        name = node.func.value.id
        element_kind = self.set_kind_of(name)
        if element_kind is None:
            return False
        method = node.func.attr
        if method not in {"add", "discard", "remove"}:
            raise NativeCompileError(
                self.path,
                node,
                f"native sets support add(), discard() and remove(); "
                f"{method}() is not in the subset",
            )
        if len(node.args) != 1 or node.keywords:
            raise NativeCompileError(
                self.path, node, f"{method}() takes exactly one argument"
            )
        argument = node.args[0]
        if self.expression_type(argument) != element_kind:
            raise NativeCompileError(
                self.path,
                argument,
                f"this set holds {element_kind} elements",
            )
        if element_kind == "int":
            self.note_stored_bool(name, argument, f"the set {name!r}")
        if method == "add":
            self.dict_store(
                self.slot(name),
                self.dict_key(argument, element_kind),
                IntConstant(0),
                node,
                element_kind,
                order=False,
            )
            return True
        self.emit_set_remove(name, argument, element_kind, node, method == "remove")
        return True

    def emit_set_remove(
        self,
        name: str,
        argument: ast.expr,
        element_kind: str,
        node: ast.AST,
        raising: bool,
    ) -> None:
        """Remove one element by rebuilding the table without it.

        Linear probing cannot simply blank a slot: a probe that walks past the
        hole would stop there and report a live key as missing. `del d[k]`
        answers that with a tombstone, which this could use too; rebuilding is
        what was already written and tested, and it costs a whole table per
        call, so discarding inside a loop is quadratic in arena bytes. It fails
        loudly with MemoryError rather than wrongly.
        """

        pointer_slot = self.slot(name)
        address_slot, found_slot, _key, _state = self.dict_probe(
            pointer_slot, self.dict_key(argument, element_kind), element_kind
        )
        if raising:
            present = self.new_label("set_remove_present")
            self.operations.append(
                JumpIfFalse(
                    IntCompare("ne", IntLoad(found_slot), IntConstant(0)),
                    present + "_missing",
                )
            )
            self.operations.append(Jump(present))
            self.operations.append(Label(present + "_missing"))
            self.raise_exception(
                "KeyError", b"KeyError: element not in native set\n"
            )
            self.operations.append(Label(present))
        done = self.new_label("set_discard_done")
        # Absent is a no-op, and skipping the rebuild also skips its allocation.
        self.operations.append(
            JumpIfFalse(IntCompare("ne", IntLoad(found_slot), IntConstant(0)), done)
        )
        bump = self.ensure_heap()
        old_slot = self.new_temp()
        self.operations.append(Store(old_slot, IntLoad(pointer_slot)))
        destination_slot = self.emit_set_block(
            HeapLoad(IntLoad(old_slot), 8), bump
        )
        self.emit_set_scan_add(
            destination_slot,
            old_slot,
            element_kind,
            node,
            exclude_slot=address_slot,
        )
        self.operations.append(Store(pointer_slot, IntLoad(destination_slot)))
        self.operations.append(Label(done))

    def emit_set_to_list(self, name: str) -> int:
        """Copy a set's elements into a fresh list block, in table order.

        Table order is not an order any program may see, so this is only ever
        handed straight to a sort.
        """

        bump = self.ensure_heap()
        source_slot = self.new_temp()
        self.operations.append(Store(source_slot, IntLoad(self.slots[name])))
        list_slot = self.new_temp()
        length_slot = self.new_temp()
        self.operations.append(
            Store(
                length_slot,
                HeapLoad(
                    IntBinary("add", IntLoad(source_slot), IntConstant(8)), 8
                ),
            )
        )
        # Sized to the live count, so the appends below never grow it. An
        # empty set gives a capacity of 0, which emit_list_append handles.
        self.operations.append(
            HeapAlloc(
                list_slot,
                IntBinary(
                    "add",
                    IntConstant(self.LIST_HEADER_BYTES),
                    IntBinary("mul", IntLoad(length_slot), IntConstant(8)),
                ),
                bump,
            )
        )
        self.operations.append(
            HeapStore(IntLoad(list_slot), IntLoad(length_slot), 8)
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", IntLoad(list_slot), IntConstant(8)),
                IntConstant(0),
                8,
            )
        )

        def body(_entry_slot: int, element_slot: int) -> None:
            self.emit_list_append(list_slot, IntLoad(element_slot))

        self.emit_set_scan(source_slot, body)
        return list_slot

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
            result = self.emit_concat(result, self.render_formatted(piece))
        return result

    def format_spec_text(self, piece: ast.FormattedValue) -> str:
        """The literal text of a field's format specifier, or the empty string."""

        spec = piece.format_spec
        if spec is None:
            return ""
        parts: list[str] = []
        if not isinstance(spec, ast.JoinedStr):
            raise NativeCompileError(
                self.path,
                spec,
                "native f-strings need a format specifier written out in the "
                "source; this one is built at run time",
            )
        for item in spec.values:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                parts.append(item.value)
            else:
                raise NativeCompileError(
                    self.path,
                    piece,
                    "a native f-string format specifier has to be literal text; "
                    f"{_FORMAT_SUPPORTED}",
                )
        return "".join(parts)

    def render_formatted(self, piece: ast.FormattedValue) -> IntExpression:
        """A pointer to the text one f-string field produces."""

        node = piece.value
        spec_text = self.format_spec_text(piece)
        if piece.conversion == -1 and not spec_text:
            return self.render_as_string(node)

        kind = self.expression_type(node)
        if kind not in ("int", "float", "str"):
            raise NativeCompileError(
                self.path, node, f"a native f-string cannot render a {kind} yet"
            )
        body: IntExpression | None = None
        if piece.conversion != -1:
            if kind == "str" and piece.conversion != ord("s"):
                raise NativeCompileError(
                    self.path,
                    node,
                    "!r and !a on a string add quotes and backslash escapes, "
                    "which a native f-string does not reproduce; only !s is "
                    "supported on a string",
                )
            if piece.conversion not in (ord("s"), ord("r"), ord("a")):
                raise NativeCompileError(
                    self.path, node, "unsupported native f-string conversion"
                )
            # On an int, a float, or a bool all three conversions produce the
            # same text str() does, and the field is then a string - which
            # changes the default alignment, so the kind changes with it.
            body = self.render_as_string(node)
            kind = "str"
        try:
            spec = parse_format_spec(spec_text, kind)
        except ValueError as error:
            raise NativeCompileError(self.path, node, str(error)) from error

        if body is not None:
            return self.emit_padded_field(None, body, spec)
        if kind == "str":
            return self.emit_padded_field(None, self.string_pointer(node), spec)
        if spec.type == "f":
            # An int formatted as a fixed-point number goes through the double,
            # which is what CPython does too, so the whole i64 range matches.
            magnitude, negative = self.emit_float_fixed(
                self.float_expression(node), spec.precision or 0
            )
        elif kind == "float":
            magnitude, negative = self.emit_split_sign(
                self.emit_float_to_string(self.float_expression(node))
            )
        else:
            magnitude, negative = self.emit_split_sign(
                self.emit_int_to_string(self.integer(node))
            )
            if spec.grouping:
                magnitude = self.emit_group_digits(magnitude)
        return self.emit_padded_field((negative, spec.sign), magnitude, spec)

    def emit_padded_field(
        self,
        sign: tuple[int, str] | None,
        body: IntExpression,
        spec: FormatSpec,
    ) -> IntExpression:
        """Put the sign in front of a rendered field and pad it to the width.

        The sign is kept apart from the digits because '=' alignment - which a
        bare zero flag selects - puts the fill between them.
        """

        body_slot = self.new_temp()
        self.operations.append(Store(body_slot, body))
        body = IntLoad(body_slot)
        prefix: IntExpression | None = None
        if sign is not None:
            negative_slot, flag = sign
            unsigned = {"+": b"+", " ": b" ", "-": b""}[flag]
            prefix = self.materialize_int(
                self.select_integer(
                    IntCompare("ne", IntLoad(negative_slot), IntConstant(0)),
                    self.materialize_string_constant(b"-"),
                    self.materialize_string_constant(unsigned),
                )
            )
        if spec.width == 0:
            return body if prefix is None else self.emit_concat(prefix, body)

        # A width counts code points, not bytes, and the fill may be a
        # multi-byte character of its own.
        width_used = self.emit_code_point_count(body)
        if prefix is not None:
            width_used = IntBinary("add", HeapLoad(prefix, 8), width_used)
        pad_slot = self.new_temp()
        self.operations.append(
            Store(pad_slot, IntBinary("sub", IntConstant(spec.width), width_used))
        )
        self.operations.append(
            Store(
                pad_slot,
                self.select_integer(
                    IntCompare("gt", IntLoad(pad_slot), IntConstant(0)),
                    IntLoad(pad_slot),
                    IntConstant(0),
                ),
            )
        )
        pad = IntLoad(pad_slot)
        if spec.align == "=":
            filled = self.emit_fill_run(spec.fill, pad)
            assert prefix is not None
            return self.emit_concat(self.emit_concat(prefix, filled), body)
        if spec.align == "^":
            left_slot = self.new_temp()
            self.operations.append(
                Store(left_slot, IntBinary("sdiv", pad, IntConstant(2)))
            )
            # The odd character goes on the right, as CPython puts it.
            head = self.emit_fill_run(spec.fill, IntLoad(left_slot))
            tail = self.emit_fill_run(
                spec.fill, IntBinary("sub", pad, IntLoad(left_slot))
            )
            middle = body if prefix is None else self.emit_concat(prefix, body)
            return self.emit_concat(self.emit_concat(head, middle), tail)
        filled = self.emit_fill_run(spec.fill, pad)
        middle = body if prefix is None else self.emit_concat(prefix, body)
        if spec.align == "<":
            return self.emit_concat(middle, filled)
        return self.emit_concat(filled, middle)

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
        if self.list_kind(kind) in {"int", "float", "bool"}:
            return self.emit_list_to_string(node, self.list_kind(kind))
        if (
            self.dict_kinds(kind) is not None
            and isinstance(node, ast.Name)
            and self.dict_kinds(kind)[0] == "int"
            and self.dict_kinds(kind)[1] in {"int", "float"}
        ):
            return self.emit_dict_to_string(node)
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
        if isinstance(node, ast.Name):
            self.refuse_unbound(node.id, node)
        if isinstance(node, ast.Call) and self.method_call_kind(node) == "str":
            # Inlined the ordinary way; the difference is only that the result
            # slot is read as an address.
            return self.integer(node)
        if isinstance(node, ast.IfExp):
            # `"neg" if n < 0 else "pos"` - and also what a body of ifs that
            # all end in a return is folded into, which is how a branching
            # function comes to answer with a string.
            condition = self.truth_value(node.test)
            self.eager_depth += 1
            try:
                chosen = self.string_pointer(node.body)
                other = self.string_pointer(node.orelse)
            finally:
                self.eager_depth -= 1
            return self.select_integer(condition, chosen, other)
        if self.list_method_shape(node, "pop") == "str":
            # The element word is the string's block address.
            assert isinstance(node, ast.Call)
            return self.emit_list_pop(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "chr"
            and node.func.id not in self.functions
        ):
            return self.emit_chr(node)
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
        if (
            isinstance(node, ast.Subscript)
            and not isinstance(node.slice, ast.Slice)
            and self.expression_type(node.value) == "str"
        ):
            return self.emit_string_index(
                self.string_pointer(node.value), node.slice
            )
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            return self.emit_string_slice(
                self.string_pointer(node.value), node.slice
            )
        if (
            isinstance(node, ast.Subscript)
            and self.list_kind(self.expression_type(node.value)) == "str"
        ):
            # A string element is the address of its block, already a pointer.
            return HeapLoad(self.list_element_address(node), 8)
        if self.tuple_subscript_kind(node) == "str":
            assert isinstance(node, ast.Subscript)
            return HeapLoad(self.tuple_element_address(node), 8)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and self.expression_function_kind(node) == "str"
            and self.functions[node.func.id].expression is None
        ):
            # A statement body is inlined the ordinary way; what is different
            # is only that its result slot is read as an address.
            result = self.integer(node)
            return result
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
            # Asked before anything is swapped, in the caller, where the
            # argument expressions still mean what they say.
            # bind_native_arguments reports only int, float or str, and a bool
            # has to be told apart from an int by its source.
            bool_parameters = {
                parameter
                for parameter, argument in zip(function.parameters, node.args)
                if self.renders_as_bool(argument)
            } | {
                # A parameter the call left out takes its default, and
                # `flag=True` is as much a bool as an argument that says so.
                parameter
                for parameter, default in zip(
                    function.parameters[len(node.args) :],
                    function.defaults[len(node.args) :],
                )
                if isinstance(default, bool)
            }
            previous_path, previous_values = self.path, self.values
            previous_functions = self.functions
            previous_strings = self.string_bindings
            self.path = function.path
            # A parameter shadows a name of its own from outside, so any value
            # folded for that name has to go; otherwise f"{v}" inside the
            # function rendered the outer v rather than the argument.
            self.values = {
                name: value
                for name, value in function.values.items()
                if name not in function.parameters
            }
            self.functions = function.functions
            previous_values_bound = self.value_bindings
            previous_booleans = set(self.boolean_names)
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
            # The numbers too, so that str(n) and f"{n}" inside the function
            # can see them. Only the string ones were carried before, which is
            # why rendering a parameter was refused.
            self.value_bindings = {
                **previous_values_bound,
                **{
                    parameter: value
                    for parameter, value, kind in zip(
                        function.parameters, arguments, argument_kinds
                    )
                    if kind != "str"
                },
            }
            # A bool argument keeps its identity across the call, so that
            # f"{flag}" inside the function writes True and not 1. Nothing
            # tells the two apart at run time, so it has to be carried.
            self.boolean_names = {
                name
                for name in self.boolean_names
                if name not in function.parameters
            } | bool_parameters
            try:
                return self.string_pointer(function.expression)
            finally:
                self.path, self.values = previous_path, previous_values
                self.functions = previous_functions
                self.string_bindings = previous_strings
                self.value_bindings = previous_values_bound
                self.boolean_names = previous_booleans
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "str"
            and node.func.id not in self.functions
        ):
            if len(node.args) != 1 or node.keywords:
                raise NativeCompileError(
                    self.path, node, "native str() takes exactly one argument"
                )
            # str(x) is what an f-string field already renders, so it is the
            # same call rather than a second implementation of the same text.
            return self.render_as_string(node.args[0])
        if isinstance(node, ast.JoinedStr):
            return self.joined_string(node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.string_pointer(node.left)
            right = self.string_pointer(node.right)
            return self.emit_concat(left, right)
        if self.string_method_kind(node) == "str":
            assert isinstance(node, ast.Call)
            return self.string_method_pointer(node)
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
        return self.emit_code_point_count_upto(
            IntLoad(pointer_slot), HeapLoad(IntLoad(pointer_slot), 8)
        )

    def emit_code_point_count_upto(
        self, pointer: IntExpression, limit: IntExpression
    ) -> IntExpression:
        """How many code points begin in the first ``limit`` bytes of a block.

        This is what turns the byte offset of a match into the character index
        CPython reports from `.find()`. ``limit`` is clamped to the block's own
        length, so an offset left behind by a search that found nothing cannot
        walk past the end.
        """

        pointer_slot = self.new_temp()
        self.operations.append(Store(pointer_slot, pointer))
        base = IntLoad(pointer_slot)
        length_slot = self.new_temp()
        self.operations.append(Store(length_slot, limit))
        self.operations.append(
            Store(
                length_slot,
                self.select_integer(
                    IntCompare("lt", IntLoad(length_slot), HeapLoad(base, 8)),
                    IntLoad(length_slot),
                    HeapLoad(base, 8),
                ),
            )
        )
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

    # --- runtime string methods ---------------------------------------------

    _STRING_METHOD_KINDS = {
        "split": "list:str",
        "join": "str",
        "startswith": "bool",
        "endswith": "bool",
        "isdigit": "bool",
        "isalpha": "bool",
        "find": "int",
        "index": "int",
        "count": "int",
        "replace": "str",
        "strip": "str",
        "lstrip": "str",
        "rstrip": "str",
        "upper": "str",
        "lower": "str",
        "capitalize": "str",
        "title": "str",
        "zfill": "str",
        "center": "str",
        "ljust": "str",
        "rjust": "str",
    }
    # split() and join() are not in this table: split() takes no argument or
    # one, and join() takes a list rather than a string or a width, neither of
    # which the fixed-arity check below can express.
    _STRING_METHOD_ARITY = {
        "startswith": 1,
        "endswith": 1,
        "isdigit": 0,
        "isalpha": 0,
        "find": 1,
        "index": 1,
        "count": 1,
        "replace": 2,
        "strip": 0,
        "lstrip": 0,
        "rstrip": 0,
        "upper": 0,
        "lower": 0,
        "capitalize": 0,
        "title": 0,
        "zfill": 1,
        "center": 1,
        "ljust": 1,
        "rjust": 1,
    }
    # These four take a character count, not another string.
    _STRING_METHOD_WIDTHS = frozenset({"zfill", "center", "ljust", "rjust"})
    # Case and character-class answers come from Unicode tables that are not in
    # the binary, so these run only on text proven ASCII at run time.
    _ASCII_ONLY_STRING_METHODS = frozenset(
        {"upper", "lower", "capitalize", "title", "isdigit", "isalpha"}
    )

    def string_method_kind(
        self,
        node: ast.expr,
        bindings: dict[str, KernelValue] | None = None,
    ) -> str | None:
        """``"str"``, ``"int"`` or ``"bool"`` if this is a runtime str method."""

        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return None
        kind = self._STRING_METHOD_KINDS.get(node.func.attr)
        if kind is None:
            return None
        if self.resolve_object_class(node.func.value) is not None:
            # A user class may well define its own find(); that one wins.
            return None
        try:
            receiver = self.expression_type(node.func.value, bindings)
        except NativeCompileError:
            return None
        return kind if receiver == "str" else None

    def check_string_method(self, node: ast.Call) -> str:
        """Validate the shape of a runtime string method call, and name it."""

        assert isinstance(node.func, ast.Attribute)
        name = node.func.attr
        if name in {"split", "join"}:
            return self.check_split_or_join(node, name)
        arity = self._STRING_METHOD_ARITY[name]
        if node.keywords or len(node.args) != arity:
            expected = "no arguments" if arity == 0 else f"{arity} argument"
            if arity > 1:
                expected += "s"
            raise NativeCompileError(
                self.path,
                node,
                f"native str.{name}() takes {expected}; the optional start, "
                "end, count and fill-character forms are not in the subset",
            )
        for argument in node.args:
            wanted = "int" if name in self._STRING_METHOD_WIDTHS else "str"
            if self.expression_type(argument) != wanted:
                noun = "an integer width" if wanted == "int" else "a string"
                raise NativeCompileError(
                    self.path, argument, f"native str.{name}() takes {noun}"
                )
        if self.eager_depth and (
            name in self._ASCII_ONLY_STRING_METHODS or name == "index"
        ):
            # Conditional expressions and short-circuited Boolean operands are
            # lowered by evaluating both arms and selecting between them. These
            # methods can stop the program, and doing so on a branch CPython
            # would never have taken is a divergence, not a refusal.
            raise NativeCompileError(
                self.path,
                node,
                f"native str.{name}() cannot appear in a conditional expression "
                "or a short-circuited Boolean operand, because this lowering "
                "evaluates both arms and the call can stop the program from the "
                "arm Python skips; use an if statement instead",
            )
        if name in self._ASCII_ONLY_STRING_METHODS:
            try:
                receiver = self.constant(node.func.value)
            except NativeCompileError:
                receiver = None
            if isinstance(receiver, str) and not receiver.isascii():
                raise NativeCompileError(
                    self.path,
                    node,
                    f"native str.{name}() is limited to ASCII text, and this "
                    f"receiver holds {receiver!r}; the Unicode tables it would "
                    "need are not in the binary",
                )
        return name

    def check_split_or_join(self, node: ast.Call, name: str) -> str:
        """Validate `s.split(...)` and `sep.join(...)`, which have their own shapes."""

        if name == "join":
            if node.keywords or len(node.args) != 1:
                raise NativeCompileError(
                    self.path,
                    node,
                    "native str.join() takes one runtime list of strings",
                )
            if self.list_kind(self.expression_type(node.args[0])) != "str":
                raise NativeCompileError(
                    self.path,
                    node.args[0],
                    "native str.join() takes a list of strings; there is no "
                    "run-time str() to render other elements with",
                )
            return name
        if node.keywords or len(node.args) > 1:
            raise NativeCompileError(
                self.path,
                node,
                "native str.split() takes no argument or one separator string; "
                "the maxsplit form is not in the subset",
            )
        if not node.args:
            return name
        if self.expression_type(node.args[0]) != "str":
            raise NativeCompileError(
                self.path, node.args[0], "native str.split() takes a separator string"
            )
        try:
            folded = self.constant(node.args[0])
        except NativeCompileError:
            folded = None
        if folded == "":
            raise NativeCompileError(
                self.path,
                node,
                "str.split('') raises ValueError: empty separator, so this call "
                "can only fail; split() with no argument is the form that "
                "splits on whitespace",
            )
        if folded is None and self.eager_depth:
            # The empty separator raises, and raising from an arm Python never
            # evaluates is a divergence rather than a refusal.
            raise NativeCompileError(
                self.path,
                node,
                "native str.split() with a separator that is not a literal "
                "cannot appear in a conditional expression or a short-circuited "
                "Boolean operand, because this lowering evaluates both arms and "
                "an empty separator raises ValueError from the arm Python "
                "skips; use an if statement instead",
            )
        return name

    def emit_empty_list_block(self, capacity: int = 4) -> int:
        """A slot holding a fresh, empty list block."""

        bump = self.ensure_heap()
        slot = self.new_temp()
        self.operations.append(
            HeapAlloc(
                slot,
                IntConstant(self.LIST_HEADER_BYTES + capacity * 8),
                bump,
            )
        )
        self.operations.append(HeapStore(IntLoad(slot), IntConstant(capacity), 8))
        self.operations.append(
            HeapStore(
                IntBinary("add", IntLoad(slot), IntConstant(8)), IntConstant(0), 8
            )
        )
        return slot

    def string_method_list(self, node: ast.Call) -> IntExpression:
        """Lower a string method whose result is a runtime list of strings."""

        assert isinstance(node.func, ast.Attribute)
        self.check_string_method(node)
        receiver_slot = self.new_temp()
        self.operations.append(
            Store(receiver_slot, self.string_pointer(node.func.value))
        )
        if not node.args:
            return self.emit_split_whitespace(receiver_slot)
        separator_slot = self.new_temp()
        self.operations.append(
            Store(separator_slot, self.string_pointer(node.args[0]))
        )
        return self.emit_split_separator(receiver_slot, separator_slot)

    def emit_split_separator(
        self, receiver_slot: int, separator_slot: int
    ) -> IntExpression:
        """`s.split(sep)` - the pieces between non-overlapping separators.

        This form keeps the empty pieces, which is what makes it differ from
        the whitespace one: `",a,".split(",")` is three pieces and `"".split(",")`
        is one empty piece rather than none. There is always one more piece
        than there were separators, so the tail is appended unconditionally.
        """

        separator_length = HeapLoad(IntLoad(separator_slot), 8)
        usable = self.new_label("split_separator_ok")
        self.operations.append(
            JumpIfFalse(
                IntCompare("eq", separator_length, IntConstant(0)), usable
            )
        )
        self.raise_exception("ValueError", b"ValueError: empty separator\n")
        self.operations.append(Label(usable))

        result_slot = self.emit_empty_list_block()
        from_slot = self.new_temp()
        self.operations.append(Store(from_slot, IntConstant(0)))
        scan = self.new_label("split_scan")
        done = self.new_label("split_scanned")
        self.operations.append(Label(scan))
        found_slot, at_slot = self.emit_substring_scan(
            receiver_slot, separator_slot, from_slot
        )
        self.operations.append(
            JumpIfFalse(IntCompare("ne", IntLoad(found_slot), IntConstant(0)), done)
        )
        self.emit_list_append(
            result_slot,
            self.emit_string_span(
                receiver_slot, IntLoad(from_slot), IntLoad(at_slot)
            ),
        )
        self.operations.append(
            Store(
                from_slot,
                IntBinary("add", IntLoad(at_slot), separator_length),
            )
        )
        self.operations.append(Jump(scan))
        self.operations.append(Label(done))
        self.emit_list_append(
            result_slot,
            self.emit_string_span(
                receiver_slot,
                IntLoad(from_slot),
                HeapLoad(IntLoad(receiver_slot), 8),
            ),
        )
        return IntLoad(result_slot)

    def emit_split_whitespace(self, receiver_slot: int) -> IntExpression:
        """`s.split()` - the runs of non-whitespace, dropping every empty piece.

        The predicate is the 29 Unicode whitespace code points, the same set
        `.strip()` uses, so `"\\u00a0"` separates here as it does in CPython.
        """

        result_slot = self.emit_empty_list_block()
        length_slot = self.new_temp()
        base_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntLoad(receiver_slot), 8))
        )
        self.operations.append(
            Store(
                base_slot, IntBinary("add", IntLoad(receiver_slot), IntConstant(8))
            )
        )
        cursor_slot = self.new_temp()
        self.operations.append(Store(cursor_slot, IntConstant(0)))
        outer = self.new_label("wsplit")
        outer_end = self.new_label("wsplit_end")
        self.operations.append(Label(outer))

        skip = self.new_label("wsplit_skip")
        skipped = self.new_label("wsplit_skipped")
        self.operations.append(Label(skip))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(cursor_slot), IntLoad(length_slot)),
                skipped,
            )
        )
        width_slot = self.emit_whitespace_width(base_slot, cursor_slot, length_slot)
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(width_slot), IntConstant(0)), skipped
            )
        )
        self.operations.append(
            Store(
                cursor_slot,
                IntBinary("add", IntLoad(cursor_slot), IntLoad(width_slot)),
            )
        )
        self.operations.append(Jump(skip))
        self.operations.append(Label(skipped))
        # Trailing whitespace ends the walk without producing a piece, which is
        # what makes `"  ".split()` empty where `"  ".split(" ")` is not.
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(cursor_slot), IntLoad(length_slot)),
                outer_end,
            )
        )
        start_slot = self.new_temp()
        self.operations.append(Store(start_slot, IntLoad(cursor_slot)))

        run = self.new_label("wsplit_run")
        ran = self.new_label("wsplit_ran")
        self.operations.append(Label(run))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(cursor_slot), IntLoad(length_slot)), ran
            )
        )
        run_width_slot = self.emit_whitespace_width(
            base_slot, cursor_slot, length_slot
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare("eq", IntLoad(run_width_slot), IntConstant(0)), ran
            )
        )
        # One byte at a time: a continuation byte is never whitespace, so the
        # run cannot stop inside a character even though the step is not a
        # whole one.
        self.operations.append(
            Store(
                cursor_slot, IntBinary("add", IntLoad(cursor_slot), IntConstant(1))
            )
        )
        self.operations.append(Jump(run))
        self.operations.append(Label(ran))
        self.emit_list_append(
            result_slot,
            self.emit_string_span(
                receiver_slot, IntLoad(start_slot), IntLoad(cursor_slot)
            ),
        )
        self.operations.append(Jump(outer))
        self.operations.append(Label(outer_end))
        return IntLoad(result_slot)

    def emit_string_join(
        self, separator_slot: int, pieces: IntExpression
    ) -> IntExpression:
        """`sep.join(xs)` - one allocation, sized by a first pass.

        Concatenating in a loop would allocate an intermediate per element and
        abandon every one of them, which is quadratic in an arena that never
        gives anything back.
        """

        bump = self.ensure_heap()
        pieces_slot = self.new_temp()
        self.operations.append(Store(pieces_slot, pieces))
        count_slot = self.new_temp()
        self.operations.append(
            Store(
                count_slot,
                HeapLoad(IntBinary("add", IntLoad(pieces_slot), IntConstant(8)), 8),
            )
        )
        index_slot = self.new_temp()
        piece_slot = self.new_temp()
        separator_length = HeapLoad(IntLoad(separator_slot), 8)

        def load_piece() -> None:
            self.operations.append(
                Store(
                    piece_slot,
                    HeapLoad(
                        IntBinary(
                            "add",
                            IntBinary(
                                "add",
                                IntLoad(pieces_slot),
                                IntConstant(self.LIST_HEADER_BYTES),
                            ),
                            IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
                        ),
                        8,
                    ),
                )
            )

        total_slot = self.new_temp()
        self.operations.append(Store(total_slot, IntConstant(0)))
        self.operations.append(Store(index_slot, IntConstant(0)))
        measure = self.new_label("join_measure")
        measured = self.new_label("join_measured")
        self.operations.append(Label(measure))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(count_slot)), measured
            )
        )
        load_piece()
        self.operations.append(
            Store(
                total_slot,
                IntBinary(
                    "add", IntLoad(total_slot), HeapLoad(IntLoad(piece_slot), 8)
                ),
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(measure))
        self.operations.append(Label(measured))
        # One separator fewer than there are pieces, and none at all for an
        # empty list, where the answer is the empty string.
        self.operations.append(
            Store(
                total_slot,
                IntBinary(
                    "add",
                    IntLoad(total_slot),
                    IntBinary(
                        "mul",
                        separator_length,
                        self.select_integer(
                            IntCompare("gt", IntLoad(count_slot), IntConstant(0)),
                            IntBinary("sub", IntLoad(count_slot), IntConstant(1)),
                            IntConstant(0),
                        ),
                    ),
                ),
            )
        )
        result_slot = self.new_temp()
        self.operations.append(
            HeapAlloc(result_slot, self._aligned_size(IntLoad(total_slot)), bump)
        )
        self.operations.append(
            HeapStore(IntLoad(result_slot), IntLoad(total_slot), 8)
        )
        write_slot = self.new_temp()
        self.operations.append(Store(write_slot, IntConstant(0)))
        self.operations.append(Store(index_slot, IntConstant(0)))
        copy = self.new_label("join_copy")
        copied = self.new_label("join_copied")
        plain = self.new_label("join_no_separator")
        self.operations.append(Label(copy))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(count_slot)), copied
            )
        )
        self.operations.append(
            JumpIfFalse(IntCompare("gt", IntLoad(index_slot), IntConstant(0)), plain)
        )
        self.emit_byte_copy(
            IntBinary(
                "add",
                IntBinary("add", IntLoad(result_slot), IntConstant(8)),
                IntLoad(write_slot),
            ),
            IntBinary("add", IntLoad(separator_slot), IntConstant(8)),
            separator_length,
        )
        self.operations.append(
            Store(
                write_slot, IntBinary("add", IntLoad(write_slot), separator_length)
            )
        )
        self.operations.append(Label(plain))
        load_piece()
        self.emit_byte_copy(
            IntBinary(
                "add",
                IntBinary("add", IntLoad(result_slot), IntConstant(8)),
                IntLoad(write_slot),
            ),
            IntBinary("add", IntLoad(piece_slot), IntConstant(8)),
            HeapLoad(IntLoad(piece_slot), 8),
        )
        self.operations.append(
            Store(
                write_slot,
                IntBinary(
                    "add", IntLoad(write_slot), HeapLoad(IntLoad(piece_slot), 8)
                ),
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(copy))
        self.operations.append(Label(copied))
        return IntLoad(result_slot)

    def string_method_pointer(self, node: ast.Call) -> IntExpression:
        """Lower a string method whose result is a string."""

        assert isinstance(node.func, ast.Attribute)
        name = self.check_string_method(node)
        receiver_slot = self.new_temp()
        self.operations.append(
            Store(receiver_slot, self.string_pointer(node.func.value))
        )
        if name in self._ASCII_ONLY_STRING_METHODS:
            self.emit_require_ascii(receiver_slot, name)
        if name in {"upper", "lower", "capitalize", "title"}:
            return self.emit_ascii_case(receiver_slot, name)
        if name in {"strip", "lstrip", "rstrip"}:
            return self.emit_string_strip(receiver_slot, name)
        if name == "join":
            return self.emit_string_join(
                receiver_slot, self.list_pointer(node.args[0])
            )
        if name == "replace":
            old_slot = self.new_temp()
            new_slot = self.new_temp()
            self.operations.append(
                Store(old_slot, self.string_pointer(node.args[0]))
            )
            self.operations.append(
                Store(new_slot, self.string_pointer(node.args[1]))
            )
            return self.emit_string_replace(receiver_slot, old_slot, new_slot)
        return self.emit_string_pad(receiver_slot, name, self.integer(node.args[0]))

    def string_method_integer(self, node: ast.Call) -> IntExpression:
        """Lower a string method whose result is an integer or a bool."""

        assert isinstance(node.func, ast.Attribute)
        name = self.check_string_method(node)
        receiver_slot = self.new_temp()
        self.operations.append(
            Store(receiver_slot, self.string_pointer(node.func.value))
        )
        if name in {"isdigit", "isalpha"}:
            self.emit_require_ascii(receiver_slot, name)
            return self.emit_ascii_class(receiver_slot, name)
        needle_slot = self.new_temp()
        self.operations.append(
            Store(needle_slot, self.string_pointer(node.args[0]))
        )
        if name == "startswith":
            return self.emit_string_prefix(
                receiver_slot, needle_slot, IntConstant(0)
            )
        if name == "endswith":
            return self.emit_string_prefix(
                receiver_slot,
                needle_slot,
                IntBinary(
                    "sub",
                    HeapLoad(IntLoad(receiver_slot), 8),
                    HeapLoad(IntLoad(needle_slot), 8),
                ),
            )
        if name == "count":
            return self.emit_string_count(receiver_slot, needle_slot)
        return self.emit_string_find(
            receiver_slot, needle_slot, raising=name == "index"
        )

    def emit_require_ascii(self, pointer_slot: int, name: str) -> None:
        """Stop the program unless the string is ASCII.

        `.upper()` and its neighbours are Unicode mappings, not byte tricks:
        'e' with an acute uppercases to its accented capital, and the German
        sharp s uppercases to the two characters "SS", which makes the string
        longer. Being exactly right needs the full case and category tables in
        the image. Flipping bit 5 of every letter byte instead would leave
        every non-ASCII character untouched and print a confidently wrong
        answer, so text that is not ASCII stops here.

        This is a limit of the compiler rather than a Python-level error, so it
        is a write and an exit like the arena guard, not a raise: an
        `except Exception:` must not be able to swallow it and carry on down a
        path CPython never took.
        """

        base = IntBinary("add", IntLoad(pointer_slot), IntConstant(8))
        length_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntLoad(pointer_slot), 8))
        )
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        start = self.new_label("ascii_check")
        step = self.new_label("ascii_next")
        done = self.new_label("ascii_done")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), done
            )
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "ne",
                    IntBinary(
                        "and",
                        HeapLoad(IntBinary("add", base, IntLoad(index_slot)), 1),
                        IntConstant(0x80),
                    ),
                    IntConstant(0),
                ),
                step,
            )
        )
        self.operations.append(
            Write(
                f"py2bin: str.{name}() is limited to ASCII text; this string is "
                "not ASCII, and the Unicode tables it would need are not in "
                "this binary\n".encode("utf-8"),
                2,
            )
        )
        self.operations.append(Exit(1))
        self.operations.append(Label(step))
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(done))

    def emit_byte_fill(
        self,
        destination: IntExpression,
        value: IntExpression,
        count: IntExpression,
    ) -> None:
        """Write ``count`` copies of one byte, for the padding methods."""

        index_slot = self.new_temp()
        destination_slot = self.new_temp()
        value_slot = self.new_temp()
        count_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        self.operations.append(Store(destination_slot, destination))
        self.operations.append(Store(value_slot, value))
        self.operations.append(Store(count_slot, count))
        start_label = self.new_label("fill_start")
        end_label = self.new_label("fill_end")
        self.operations.append(Label(start_label))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(count_slot)),
                end_label,
            )
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", IntLoad(destination_slot), IntLoad(index_slot)),
                IntLoad(value_slot),
                1,
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start_label))
        self.operations.append(Label(end_label))

    def emit_string_span(
        self, pointer_slot: int, start: IntExpression, end: IntExpression
    ) -> IntExpression:
        """A fresh string block holding the bytes ``[start, end)`` of another."""

        bump = self.ensure_heap()
        start_slot = self.new_temp()
        span_slot = self.new_temp()
        result_slot = self.new_temp()
        self.operations.append(Store(start_slot, start))
        self.operations.append(
            Store(span_slot, IntBinary("sub", end, IntLoad(start_slot)))
        )
        self.operations.append(
            HeapAlloc(result_slot, self._aligned_size(IntLoad(span_slot)), bump)
        )
        self.operations.append(
            HeapStore(IntLoad(result_slot), IntLoad(span_slot), 8)
        )
        self.emit_byte_copy(
            IntBinary("add", IntLoad(result_slot), IntConstant(8)),
            IntBinary(
                "add",
                IntBinary("add", IntLoad(pointer_slot), IntConstant(8)),
                IntLoad(start_slot),
            ),
            IntLoad(span_slot),
        )
        return IntLoad(result_slot)

    def emit_next_boundary(
        self, base: IntExpression, position: IntExpression, limit: IntExpression
    ) -> IntExpression:
        """The byte offset one code point past ``position``.

        ``position`` is itself a boundary. At the very end of the string this
        answers ``limit + 1``, which is what stops a loop stepping over the
        empty match CPython finds there.
        """

        base_slot = self.new_temp()
        limit_slot = self.new_temp()
        cursor_slot = self.new_temp()
        self.operations.append(Store(base_slot, base))
        self.operations.append(Store(limit_slot, limit))
        self.operations.append(
            Store(cursor_slot, IntBinary("add", position, IntConstant(1)))
        )
        start = self.new_label("boundary")
        done = self.new_label("boundary_done")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(cursor_slot), IntLoad(limit_slot)), done
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
                                "add", IntLoad(base_slot), IntLoad(cursor_slot)
                            ),
                            1,
                        ),
                        IntConstant(0xC0),
                    ),
                    IntConstant(0x80),
                ),
                done,
            )
        )
        self.operations.append(
            Store(cursor_slot, IntBinary("add", IntLoad(cursor_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(done))
        return IntLoad(cursor_slot)

    def emit_character_width(
        self, base_slot: int, offset_slot: int
    ) -> IntExpression:
        """How many bytes the UTF-8 character at a boundary offset occupies."""

        byte_slot = self.new_temp()
        self.operations.append(
            Store(
                byte_slot,
                HeapLoad(
                    IntBinary("add", IntLoad(base_slot), IntLoad(offset_slot)), 1
                ),
            )
        )
        byte = IntLoad(byte_slot)
        return IntBinary(
            "add",
            IntConstant(1),
            IntBinary(
                "add",
                IntCompare("ge", byte, IntConstant(0xC0)),
                IntBinary(
                    "add",
                    IntCompare("ge", byte, IntConstant(0xE0)),
                    IntCompare("ge", byte, IntConstant(0xF0)),
                ),
            ),
        )

    def emit_whitespace_width(
        self, base_slot: int, offset_slot: int, limit_slot: int
    ) -> int:
        """A slot holding the byte width of the whitespace at an offset, or 0.

        `str.strip()` with no argument strips Unicode whitespace, which is 29
        code points and not the ASCII five. Unlike case mapping that is a small
        closed set which has not moved in twenty years, so it is matched here
        as byte sequences rather than refused.
        """

        def between(value: IntExpression, low: int, high: int) -> IntExpression:
            return IntBinary(
                "and",
                IntCompare("ge", value, IntConstant(low)),
                IntCompare("le", value, IntConstant(high)),
            )

        def equals(value: IntExpression, other: int) -> IntExpression:
            return IntCompare("eq", value, IntConstant(other))

        def room(bytes_needed: int, label: str) -> None:
            self.operations.append(
                JumpIfFalse(
                    IntCompare(
                        "le",
                        IntBinary(
                            "add", IntLoad(offset_slot), IntConstant(bytes_needed)
                        ),
                        IntLoad(limit_slot),
                    ),
                    label,
                )
            )

        def byte_at(distance: int) -> IntExpression:
            address = IntBinary("add", IntLoad(base_slot), IntLoad(offset_slot))
            if distance:
                address = IntBinary("add", address, IntConstant(distance))
            return HeapLoad(address, 1)

        width_slot = self.new_temp()
        self.operations.append(Store(width_slot, IntConstant(0)))
        done = self.new_label("ws_done")
        wide = self.new_label("ws_wide")
        first_slot = self.new_temp()
        self.operations.append(Store(first_slot, byte_at(0)))
        first = IntLoad(first_slot)
        # Tab through carriage return, then the four separators and the space.
        self.operations.append(
            JumpIfFalse(
                IntBinary("or", between(first, 0x09, 0x0D), between(first, 0x1C, 0x20)),
                wide,
            )
        )
        self.operations.append(Store(width_slot, IntConstant(1)))
        self.operations.append(Jump(done))
        self.operations.append(Label(wide))
        # Nothing else starts below C2, and reading a second or third byte
        # needs the room to be there.
        self.operations.append(
            JumpIfFalse(IntCompare("ge", first, IntConstant(0xC2)), done)
        )
        room(2, done)
        second_slot = self.new_temp()
        self.operations.append(Store(second_slot, byte_at(1)))
        second = IntLoad(second_slot)
        three_byte = self.new_label("ws_three")
        self.operations.append(
            JumpIfFalse(equals(first, 0xC2), three_byte)
        )
        # U+0085 NEXT LINE and U+00A0 NO-BREAK SPACE.
        self.operations.append(
            JumpIfFalse(
                IntBinary("or", equals(second, 0x85), equals(second, 0xA0)), done
            )
        )
        self.operations.append(Store(width_slot, IntConstant(2)))
        self.operations.append(Jump(done))
        self.operations.append(Label(three_byte))
        room(3, done)
        third_slot = self.new_temp()
        self.operations.append(Store(third_slot, byte_at(2)))
        third = IntLoad(third_slot)
        ogham = IntBinary(
            "and",
            equals(first, 0xE1),
            IntBinary("and", equals(second, 0x9A), equals(third, 0x80)),
        )
        # U+2000-200A, U+2028, U+2029 and U+202F.
        punctuation = IntBinary(
            "and",
            IntBinary("and", equals(first, 0xE2), equals(second, 0x80)),
            IntBinary(
                "or",
                between(third, 0x80, 0x8A),
                IntBinary(
                    "or",
                    IntBinary("or", equals(third, 0xA8), equals(third, 0xA9)),
                    equals(third, 0xAF),
                ),
            ),
        )
        # U+205F MEDIUM MATHEMATICAL SPACE.
        mathematical = IntBinary(
            "and",
            IntBinary("and", equals(first, 0xE2), equals(second, 0x81)),
            equals(third, 0x9F),
        )
        # U+3000 IDEOGRAPHIC SPACE.
        ideographic = IntBinary(
            "and",
            IntBinary("and", equals(first, 0xE3), equals(second, 0x80)),
            equals(third, 0x80),
        )
        self.operations.append(
            JumpIfFalse(
                IntBinary(
                    "or",
                    IntBinary("or", ogham, punctuation),
                    IntBinary("or", mathematical, ideographic),
                ),
                done,
            )
        )
        self.operations.append(Store(width_slot, IntConstant(3)))
        self.operations.append(Label(done))
        return width_slot

    def emit_string_strip(self, pointer_slot: int, name: str) -> IntExpression:
        """`.strip()`, `.lstrip()` and `.rstrip()` over Unicode whitespace."""

        length_slot = self.new_temp()
        base_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntLoad(pointer_slot), 8))
        )
        self.operations.append(
            Store(
                base_slot,
                IntBinary("add", IntLoad(pointer_slot), IntConstant(8)),
            )
        )
        start_slot = self.new_temp()
        self.operations.append(Store(start_slot, IntConstant(0)))
        if name in {"strip", "lstrip"}:
            loop = self.new_label("lstrip")
            stop = self.new_label("lstrip_done")
            self.operations.append(Label(loop))
            self.operations.append(
                JumpIfFalse(
                    IntCompare("lt", IntLoad(start_slot), IntLoad(length_slot)),
                    stop,
                )
            )
            width_slot = self.emit_whitespace_width(
                base_slot, start_slot, length_slot
            )
            self.operations.append(
                JumpIfFalse(
                    IntCompare("ne", IntLoad(width_slot), IntConstant(0)), stop
                )
            )
            self.operations.append(
                Store(
                    start_slot,
                    IntBinary("add", IntLoad(start_slot), IntLoad(width_slot)),
                )
            )
            self.operations.append(Jump(loop))
            self.operations.append(Label(stop))
        end_slot = self.new_temp()
        self.operations.append(Store(end_slot, IntLoad(length_slot)))
        if name in {"strip", "rstrip"}:
            # One forward pass remembering where the last non-whitespace
            # character ended. Walking backwards would have to step over
            # continuation bytes to find each character's lead byte first.
            cursor_slot = self.new_temp()
            self.operations.append(Store(cursor_slot, IntConstant(0)))
            self.operations.append(Store(end_slot, IntConstant(0)))
            loop = self.new_label("rstrip")
            stop = self.new_label("rstrip_done")
            skip = self.new_label("rstrip_skip")
            self.operations.append(Label(loop))
            self.operations.append(
                JumpIfFalse(
                    IntCompare("lt", IntLoad(cursor_slot), IntLoad(length_slot)),
                    stop,
                )
            )
            width_slot = self.emit_whitespace_width(
                base_slot, cursor_slot, length_slot
            )
            step_slot = self.new_temp()
            self.operations.append(
                Store(
                    step_slot,
                    self.select_integer(
                        IntCompare("ne", IntLoad(width_slot), IntConstant(0)),
                        IntLoad(width_slot),
                        self.emit_character_width(base_slot, cursor_slot),
                    ),
                )
            )
            self.operations.append(
                Store(
                    cursor_slot,
                    IntBinary("add", IntLoad(cursor_slot), IntLoad(step_slot)),
                )
            )
            self.operations.append(
                JumpIfFalse(
                    IntCompare("eq", IntLoad(width_slot), IntConstant(0)), skip
                )
            )
            self.operations.append(Store(end_slot, IntLoad(cursor_slot)))
            self.operations.append(Label(skip))
            self.operations.append(Jump(loop))
            self.operations.append(Label(stop))
        # An all-whitespace string leaves the end behind the start.
        self.operations.append(
            Store(
                end_slot,
                self.select_integer(
                    IntCompare("lt", IntLoad(end_slot), IntLoad(start_slot)),
                    IntLoad(start_slot),
                    IntLoad(end_slot),
                ),
            )
        )
        return self.emit_string_span(
            pointer_slot, IntLoad(start_slot), IntLoad(end_slot)
        )

    def emit_ascii_case(self, pointer_slot: int, name: str) -> IntExpression:
        """`.upper()`, `.lower()`, `.capitalize()` and `.title()` over ASCII.

        The receiver has already been proven ASCII, so one byte is one
        character and the whole mapping is a per-byte choice.
        """

        bump = self.ensure_heap()
        length_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntLoad(pointer_slot), 8))
        )
        result_slot = self.new_temp()
        self.operations.append(
            HeapAlloc(result_slot, self._aligned_size(IntLoad(length_slot)), bump)
        )
        self.operations.append(
            HeapStore(IntLoad(result_slot), IntLoad(length_slot), 8)
        )
        source = IntBinary("add", IntLoad(pointer_slot), IntConstant(8))
        destination = IntBinary("add", IntLoad(result_slot), IntConstant(8))
        index_slot = self.new_temp()
        upper_slot = self.new_temp()
        cased_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        self.operations.append(Store(cased_slot, IntConstant(0)))
        loop = self.new_label("case")
        stop = self.new_label("case_done")
        self.operations.append(Label(loop))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), stop
            )
        )
        byte_slot = self.new_temp()
        self.operations.append(
            Store(
                byte_slot,
                HeapLoad(IntBinary("add", source, IntLoad(index_slot)), 1),
            )
        )
        byte = IntLoad(byte_slot)
        is_lower = IntBinary(
            "and",
            IntCompare("ge", byte, IntConstant(0x61)),
            IntCompare("le", byte, IntConstant(0x7A)),
        )
        is_upper = IntBinary(
            "and",
            IntCompare("ge", byte, IntConstant(0x41)),
            IntCompare("le", byte, IntConstant(0x5A)),
        )
        if name == "upper":
            self.operations.append(Store(upper_slot, IntConstant(1)))
        elif name == "lower":
            self.operations.append(Store(upper_slot, IntConstant(0)))
        elif name == "capitalize":
            self.operations.append(
                Store(
                    upper_slot,
                    IntCompare("eq", IntLoad(index_slot), IntConstant(0)),
                )
            )
        else:
            # CPython titlecases a letter whenever the character before it was
            # not cased, which over ASCII means not a letter: "they're" becomes
            # "They'Re", and splitting on spaces would not.
            self.operations.append(
                Store(upper_slot, IntUnary("not", IntLoad(cased_slot)))
            )
        raised = self.select_integer(
            is_lower, IntBinary("sub", byte, IntConstant(32)), byte
        )
        lowered = self.select_integer(
            is_upper, IntBinary("add", byte, IntConstant(32)), byte
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", destination, IntLoad(index_slot)),
                self.select_integer(
                    IntCompare("ne", IntLoad(upper_slot), IntConstant(0)),
                    raised,
                    lowered,
                ),
                1,
            )
        )
        if name == "title":
            self.operations.append(
                Store(cased_slot, IntBinary("or", is_lower, is_upper))
            )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(loop))
        self.operations.append(Label(stop))
        return IntLoad(result_slot)

    def emit_ascii_class(self, pointer_slot: int, name: str) -> IntExpression:
        """`.isdigit()` and `.isalpha()` over text already proven ASCII."""

        length_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntLoad(pointer_slot), 8))
        )
        result_slot = self.new_temp()
        # CPython answers False for the empty string.
        self.operations.append(
            Store(
                result_slot,
                IntCompare("gt", IntLoad(length_slot), IntConstant(0)),
            )
        )
        base = IntBinary("add", IntLoad(pointer_slot), IntConstant(8))
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        loop = self.new_label("class")
        miss = self.new_label("class_miss")
        stop = self.new_label("class_done")
        self.operations.append(Label(loop))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(length_slot)), stop
            )
        )
        byte_slot = self.new_temp()
        self.operations.append(
            Store(
                byte_slot,
                HeapLoad(IntBinary("add", base, IntLoad(index_slot)), 1),
            )
        )
        byte = IntLoad(byte_slot)
        if name == "isdigit":
            member = IntBinary(
                "and",
                IntCompare("ge", byte, IntConstant(0x30)),
                IntCompare("le", byte, IntConstant(0x39)),
            )
        else:
            member = IntBinary(
                "or",
                IntBinary(
                    "and",
                    IntCompare("ge", byte, IntConstant(0x41)),
                    IntCompare("le", byte, IntConstant(0x5A)),
                ),
                IntBinary(
                    "and",
                    IntCompare("ge", byte, IntConstant(0x61)),
                    IntCompare("le", byte, IntConstant(0x7A)),
                ),
            )
        self.operations.append(JumpIfFalse(member, miss))
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(loop))
        self.operations.append(Label(miss))
        self.operations.append(Store(result_slot, IntConstant(0)))
        self.operations.append(Label(stop))
        return IntLoad(result_slot)

    def emit_string_prefix(
        self, pointer_slot: int, needle_slot: int, offset: IntExpression
    ) -> IntExpression:
        """Whether the needle's bytes sit at a byte offset of the haystack.

        Both callers pass a code-point boundary, and a valid UTF-8 needle
        cannot match anywhere else, so this compares bytes.
        """

        offset_slot = self.new_temp()
        haystack_length = self.new_temp()
        needle_length = self.new_temp()
        result_slot = self.new_temp()
        self.operations.append(Store(offset_slot, offset))
        self.operations.append(
            Store(haystack_length, HeapLoad(IntLoad(pointer_slot), 8))
        )
        self.operations.append(
            Store(needle_length, HeapLoad(IntLoad(needle_slot), 8))
        )
        self.operations.append(Store(result_slot, IntConstant(1)))
        miss = self.new_label("prefix_miss")
        done = self.new_label("prefix_done")
        self.operations.append(
            JumpIfFalse(
                IntCompare("ge", IntLoad(offset_slot), IntConstant(0)), miss
            )
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "le",
                    IntBinary(
                        "add", IntLoad(offset_slot), IntLoad(needle_length)
                    ),
                    IntLoad(haystack_length),
                ),
                miss,
            )
        )
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        loop = self.new_label("prefix")
        self.operations.append(Label(loop))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(index_slot), IntLoad(needle_length)),
                done,
            )
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "eq",
                    HeapLoad(
                        IntBinary(
                            "add",
                            IntBinary(
                                "add",
                                IntBinary(
                                    "add",
                                    IntLoad(pointer_slot),
                                    IntConstant(8),
                                ),
                                IntLoad(offset_slot),
                            ),
                            IntLoad(index_slot),
                        ),
                        1,
                    ),
                    HeapLoad(
                        IntBinary(
                            "add",
                            IntBinary(
                                "add", IntLoad(needle_slot), IntConstant(8)
                            ),
                            IntLoad(index_slot),
                        ),
                        1,
                    ),
                ),
                miss,
            )
        )
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(loop))
        self.operations.append(Label(miss))
        self.operations.append(Store(result_slot, IntConstant(0)))
        self.operations.append(Label(done))
        return IntLoad(result_slot)

    def emit_match_advance(
        self,
        pointer_slot: int,
        position_slot: int,
        needle_length: int,
        length_slot: int,
    ) -> IntExpression:
        """Where to resume scanning after a match.

        Past the needle, so overlapping occurrences are not counted twice -
        `"aaa".count("aa")` is 1, not 2. An empty needle matches everywhere, so
        it steps one whole character instead, which is what makes
        `"abc".count("")` four and a two-byte character count once.
        """

        boundary_slot = self.new_temp()
        self.operations.append(
            Store(
                boundary_slot,
                self.emit_next_boundary(
                    IntBinary("add", IntLoad(pointer_slot), IntConstant(8)),
                    IntLoad(position_slot),
                    IntLoad(length_slot),
                ),
            )
        )
        return self.select_integer(
            IntCompare("eq", IntLoad(needle_length), IntConstant(0)),
            IntLoad(boundary_slot),
            IntBinary("add", IntLoad(position_slot), IntLoad(needle_length)),
        )

    def emit_string_count(
        self, pointer_slot: int, needle_slot: int
    ) -> IntExpression:
        """`s.count(t)`, counting non-overlapping occurrences."""

        length_slot = self.new_temp()
        needle_length = self.new_temp()
        count_slot = self.new_temp()
        cursor_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntLoad(pointer_slot), 8))
        )
        self.operations.append(
            Store(needle_length, HeapLoad(IntLoad(needle_slot), 8))
        )
        self.operations.append(Store(count_slot, IntConstant(0)))
        self.operations.append(Store(cursor_slot, IntConstant(0)))
        loop = self.new_label("occurrences")
        done = self.new_label("occurrences_done")
        self.operations.append(Label(loop))
        found_slot, position_slot = self.emit_substring_scan(
            pointer_slot, needle_slot, cursor_slot
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(found_slot), IntConstant(0)), done
            )
        )
        self.operations.append(
            Store(count_slot, IntBinary("add", IntLoad(count_slot), IntConstant(1)))
        )
        self.operations.append(
            Store(
                cursor_slot,
                self.emit_match_advance(
                    pointer_slot, position_slot, needle_length, length_slot
                ),
            )
        )
        self.operations.append(Jump(loop))
        self.operations.append(Label(done))
        return IntLoad(count_slot)

    def emit_string_find(
        self, pointer_slot: int, needle_slot: int, raising: bool
    ) -> IntExpression:
        """`s.find(t)`, or `s.index(t)` when a miss must raise."""

        cursor_slot = self.new_temp()
        self.operations.append(Store(cursor_slot, IntConstant(0)))
        found_slot, position_slot = self.emit_substring_scan(
            pointer_slot, needle_slot, cursor_slot
        )
        if raising:
            present = self.new_label("index_present")
            self.operations.append(
                JumpIfFalse(
                    IntCompare("eq", IntLoad(found_slot), IntConstant(0)), present
                )
            )
            self.raise_exception(
                "ValueError", b"ValueError: substring not found\n"
            )
            self.operations.append(Label(present))
            return self.emit_code_point_count_upto(
                IntLoad(pointer_slot), IntLoad(position_slot)
            )
        # CPython reports a character index, not a byte offset.
        return self.select_integer(
            IntCompare("ne", IntLoad(found_slot), IntConstant(0)),
            self.emit_code_point_count_upto(
                IntLoad(pointer_slot), IntLoad(position_slot)
            ),
            IntConstant(-1),
        )

    def emit_string_replace(
        self, pointer_slot: int, old_slot: int, new_slot: int
    ) -> IntExpression:
        """`s.replace(old, new)`, in two passes over the haystack.

        The first pass only counts, so the size of the answer is known before
        anything is written and a single allocation covers it. Concatenating
        once per occurrence would allocate inside the loop, and the arena never
        reclaims.
        """

        bump = self.ensure_heap()
        length_slot = self.new_temp()
        old_length = self.new_temp()
        new_length = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntLoad(pointer_slot), 8))
        )
        self.operations.append(
            Store(old_length, HeapLoad(IntLoad(old_slot), 8))
        )
        self.operations.append(
            Store(new_length, HeapLoad(IntLoad(new_slot), 8))
        )
        matches_slot = self.new_temp()
        self.operations.append(
            Store(matches_slot, self.emit_string_count(pointer_slot, old_slot))
        )
        total_slot = self.new_temp()
        self.operations.append(
            Store(
                total_slot,
                IntBinary(
                    "add",
                    IntLoad(length_slot),
                    IntBinary(
                        "mul",
                        IntLoad(matches_slot),
                        IntBinary(
                            "sub", IntLoad(new_length), IntLoad(old_length)
                        ),
                    ),
                ),
            )
        )
        result_slot = self.new_temp()
        self.operations.append(
            HeapAlloc(result_slot, self._aligned_size(IntLoad(total_slot)), bump)
        )
        self.operations.append(
            HeapStore(IntLoad(result_slot), IntLoad(total_slot), 8)
        )
        source = IntBinary("add", IntLoad(pointer_slot), IntConstant(8))
        destination = IntBinary("add", IntLoad(result_slot), IntConstant(8))
        write_slot = self.new_temp()
        cursor_slot = self.new_temp()
        self.operations.append(Store(write_slot, IntConstant(0)))
        self.operations.append(Store(cursor_slot, IntConstant(0)))
        loop = self.new_label("replace")
        done = self.new_label("replace_done")
        self.operations.append(Label(loop))
        found_slot, position_slot = self.emit_substring_scan(
            pointer_slot, old_slot, cursor_slot
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(found_slot), IntConstant(0)), done
            )
        )
        chunk_slot = self.new_temp()
        self.operations.append(
            Store(
                chunk_slot,
                IntBinary("sub", IntLoad(position_slot), IntLoad(cursor_slot)),
            )
        )
        self.emit_byte_copy(
            IntBinary("add", destination, IntLoad(write_slot)),
            IntBinary("add", source, IntLoad(cursor_slot)),
            IntLoad(chunk_slot),
        )
        self.operations.append(
            Store(
                write_slot, IntBinary("add", IntLoad(write_slot), IntLoad(chunk_slot))
            )
        )
        self.emit_byte_copy(
            IntBinary("add", destination, IntLoad(write_slot)),
            IntBinary("add", IntLoad(new_slot), IntConstant(8)),
            IntLoad(new_length),
        )
        self.operations.append(
            Store(
                write_slot, IntBinary("add", IntLoad(write_slot), IntLoad(new_length))
            )
        )
        boundary_slot = self.new_temp()
        self.operations.append(
            Store(
                boundary_slot,
                self.emit_next_boundary(
                    source, IntLoad(position_slot), IntLoad(length_slot)
                ),
            )
        )
        stop_slot = self.new_temp()
        self.operations.append(
            Store(
                stop_slot,
                self.select_integer(
                    IntCompare(
                        "lt", IntLoad(boundary_slot), IntLoad(length_slot)
                    ),
                    IntLoad(boundary_slot),
                    IntLoad(length_slot),
                ),
            )
        )
        # An empty needle matches in front of a character rather than instead
        # of one, so that character has to be carried across before moving on.
        carry_slot = self.new_temp()
        self.operations.append(
            Store(
                carry_slot,
                self.select_integer(
                    IntCompare("eq", IntLoad(old_length), IntConstant(0)),
                    IntBinary("sub", IntLoad(stop_slot), IntLoad(position_slot)),
                    IntConstant(0),
                ),
            )
        )
        self.emit_byte_copy(
            IntBinary("add", destination, IntLoad(write_slot)),
            IntBinary("add", source, IntLoad(position_slot)),
            IntLoad(carry_slot),
        )
        self.operations.append(
            Store(
                write_slot, IntBinary("add", IntLoad(write_slot), IntLoad(carry_slot))
            )
        )
        self.operations.append(
            Store(
                cursor_slot,
                self.select_integer(
                    IntCompare("eq", IntLoad(old_length), IntConstant(0)),
                    IntLoad(boundary_slot),
                    IntBinary(
                        "add", IntLoad(position_slot), IntLoad(old_length)
                    ),
                ),
            )
        )
        self.operations.append(Jump(loop))
        self.operations.append(Label(done))
        tail_slot = self.new_temp()
        self.operations.append(
            Store(
                tail_slot,
                IntBinary("sub", IntLoad(length_slot), IntLoad(cursor_slot)),
            )
        )
        # The last empty match steps past the end, which leaves no tail.
        self.operations.append(
            Store(
                tail_slot,
                self.select_integer(
                    IntCompare("gt", IntLoad(tail_slot), IntConstant(0)),
                    IntLoad(tail_slot),
                    IntConstant(0),
                ),
            )
        )
        self.emit_byte_copy(
            IntBinary("add", destination, IntLoad(write_slot)),
            IntBinary("add", source, IntLoad(cursor_slot)),
            IntLoad(tail_slot),
        )
        return IntLoad(result_slot)

    def emit_string_pad(
        self, pointer_slot: int, name: str, width: IntExpression
    ) -> IntExpression:
        """`.zfill()`, `.center()`, `.ljust()` and `.rjust()`.

        The width counts characters, as CPython's does. Padding to a byte
        length would under-pad every string holding a non-ASCII character.
        """

        bump = self.ensure_heap()
        length_slot = self.new_temp()
        width_slot = self.new_temp()
        self.operations.append(
            Store(length_slot, HeapLoad(IntLoad(pointer_slot), 8))
        )
        self.operations.append(Store(width_slot, width))
        characters = self.materialize_int(
            self.emit_code_point_count(IntLoad(pointer_slot))
        )
        pad_slot = self.new_temp()
        self.operations.append(
            Store(pad_slot, IntBinary("sub", IntLoad(width_slot), characters))
        )
        self.operations.append(
            Store(
                pad_slot,
                self.select_integer(
                    IntCompare("gt", IntLoad(pad_slot), IntConstant(0)),
                    IntLoad(pad_slot),
                    IntConstant(0),
                ),
            )
        )
        total_slot = self.new_temp()
        self.operations.append(
            Store(
                total_slot,
                IntBinary("add", IntLoad(length_slot), IntLoad(pad_slot)),
            )
        )
        result_slot = self.new_temp()
        self.operations.append(
            HeapAlloc(result_slot, self._aligned_size(IntLoad(total_slot)), bump)
        )
        self.operations.append(
            HeapStore(IntLoad(result_slot), IntLoad(total_slot), 8)
        )
        source = IntBinary("add", IntLoad(pointer_slot), IntConstant(8))
        destination = IntBinary("add", IntLoad(result_slot), IntConstant(8))
        if name == "zfill":
            # A leading sign stays in front of the zeros: "-5".zfill(4) is
            # "-005". Both signs are ASCII, so the first byte answers this;
            # an empty string has no first byte to read.
            first_slot = self.new_temp()
            self.operations.append(Store(first_slot, IntConstant(0)))
            empty = self.new_label("zfill_empty")
            self.operations.append(
                JumpIfFalse(
                    IntCompare("gt", IntLoad(length_slot), IntConstant(0)), empty
                )
            )
            self.operations.append(Store(first_slot, HeapLoad(source, 1)))
            self.operations.append(Label(empty))
            sign_slot = self.new_temp()
            self.operations.append(
                Store(
                    sign_slot,
                    IntBinary(
                        "or",
                        IntCompare("eq", IntLoad(first_slot), IntConstant(0x2D)),
                        IntCompare("eq", IntLoad(first_slot), IntConstant(0x2B)),
                    ),
                )
            )
            self.emit_byte_copy(destination, source, IntLoad(sign_slot))
            self.emit_byte_fill(
                IntBinary("add", destination, IntLoad(sign_slot)),
                IntConstant(0x30),
                IntLoad(pad_slot),
            )
            self.emit_byte_copy(
                IntBinary(
                    "add",
                    IntBinary("add", destination, IntLoad(sign_slot)),
                    IntLoad(pad_slot),
                ),
                IntBinary("add", source, IntLoad(sign_slot)),
                IntBinary("sub", IntLoad(length_slot), IntLoad(sign_slot)),
            )
            return IntLoad(result_slot)
        left_slot = self.new_temp()
        if name == "ljust":
            self.operations.append(Store(left_slot, IntConstant(0)))
        elif name == "rjust":
            self.operations.append(Store(left_slot, IntLoad(pad_slot)))
        else:
            # CPython's split of an odd margin: the extra space goes on the
            # left when both the margin and the width are odd, which is why
            # "ab".center(5) is "  ab " and not " ab  ".
            self.operations.append(
                Store(
                    left_slot,
                    IntBinary(
                        "add",
                        IntBinary("rshift", IntLoad(pad_slot), IntConstant(1)),
                        IntBinary(
                            "and",
                            IntBinary(
                                "and", IntLoad(pad_slot), IntLoad(width_slot)
                            ),
                            IntConstant(1),
                        ),
                    ),
                )
            )
        self.emit_byte_fill(destination, IntConstant(0x20), IntLoad(left_slot))
        self.emit_byte_copy(
            IntBinary("add", destination, IntLoad(left_slot)),
            source,
            IntLoad(length_slot),
        )
        self.emit_byte_fill(
            IntBinary(
                "add",
                IntBinary("add", destination, IntLoad(left_slot)),
                IntLoad(length_slot),
            ),
            IntConstant(0x20),
            IntBinary("sub", IntLoad(pad_slot), IntLoad(left_slot)),
        )
        return IntLoad(result_slot)

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

    def emit_string_tail(
        self, pointer: IntExpression, skip: IntExpression
    ) -> IntExpression:
        """A copy of a string block with its first ``skip`` bytes dropped."""

        bump = self.ensure_heap()
        source_slot = self.new_temp()
        skip_slot = self.new_temp()
        length_slot = self.new_temp()
        result_slot = self.new_temp()
        self.operations.append(Store(source_slot, pointer))
        self.operations.append(Store(skip_slot, skip))
        self.operations.append(
            Store(
                length_slot,
                IntBinary(
                    "sub", HeapLoad(IntLoad(source_slot), 8), IntLoad(skip_slot)
                ),
            )
        )
        self.operations.append(
            HeapAlloc(result_slot, self._aligned_size(IntLoad(length_slot)), bump)
        )
        self.operations.append(
            HeapStore(IntLoad(result_slot), IntLoad(length_slot), 8)
        )
        self.emit_byte_copy(
            IntBinary("add", IntLoad(result_slot), IntConstant(8)),
            IntBinary(
                "add",
                IntBinary("add", IntLoad(source_slot), IntConstant(8)),
                IntLoad(skip_slot),
            ),
            IntLoad(length_slot),
        )
        return IntLoad(result_slot)

    def emit_split_sign(
        self, pointer: IntExpression
    ) -> tuple[IntExpression, int]:
        """Split a rendered number into its magnitude and a negative flag.

        Both number renderers write a leading minus and nothing else in front
        of the digits, so the sign can be taken off the text rather than
        threaded through the renderer. Padding needs them apart: zero fill and
        '=' alignment go between the sign and the first digit.
        """

        source_slot = self.new_temp()
        self.operations.append(Store(source_slot, pointer))
        source = IntLoad(source_slot)
        negative_slot = self.new_temp()
        self.operations.append(
            Store(
                negative_slot,
                self.select_integer(
                    IntCompare(
                        "eq",
                        HeapLoad(IntBinary("add", source, IntConstant(8)), 1),
                        IntConstant(ord("-")),
                    ),
                    IntConstant(1),
                    IntConstant(0),
                ),
            )
        )
        return self.emit_string_tail(source, IntLoad(negative_slot)), negative_slot

    def emit_fill_run(self, fill: bytes, count: IntExpression) -> IntExpression:
        """A string block holding ``count`` copies of one literal character."""

        bump = self.ensure_heap()
        count_slot = self.new_temp()
        self.operations.append(Store(count_slot, count))
        self.operations.append(
            Store(
                count_slot,
                self.select_integer(
                    IntCompare("gt", IntLoad(count_slot), IntConstant(0)),
                    IntLoad(count_slot),
                    IntConstant(0),
                ),
            )
        )
        length_slot = self.new_temp()
        self.operations.append(
            Store(
                length_slot,
                IntBinary("mul", IntLoad(count_slot), IntConstant(len(fill))),
            )
        )
        result_slot = self.new_temp()
        self.operations.append(
            HeapAlloc(result_slot, self._aligned_size(IntLoad(length_slot)), bump)
        )
        self.operations.append(
            HeapStore(IntLoad(result_slot), IntLoad(length_slot), 8)
        )
        cursor_slot = self.new_temp()
        self.operations.append(
            Store(
                cursor_slot,
                IntBinary("add", IntLoad(result_slot), IntConstant(8)),
            )
        )
        start = self.new_label("fill_start")
        end = self.new_label("fill_end")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(IntCompare("gt", IntLoad(count_slot), IntConstant(0)), end)
        )
        for offset, byte in enumerate(fill):
            self.operations.append(
                HeapStore(
                    IntBinary("add", IntLoad(cursor_slot), IntConstant(offset)),
                    IntConstant(byte),
                    1,
                )
            )
        self.operations.append(
            Store(
                cursor_slot,
                IntBinary("add", IntLoad(cursor_slot), IntConstant(len(fill))),
            )
        )
        self.operations.append(
            Store(count_slot, IntBinary("sub", IntLoad(count_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
        return IntLoad(result_slot)

    def emit_group_digits(self, pointer: IntExpression) -> IntExpression:
        """Insert a comma every three digits, counting from the right."""

        bump = self.ensure_heap()
        source_slot = self.new_temp()
        self.operations.append(Store(source_slot, pointer))
        digits_slot = self.new_temp()
        self.operations.append(
            Store(digits_slot, HeapLoad(IntLoad(source_slot), 8))
        )
        total_slot = self.new_temp()
        self.operations.append(
            Store(
                total_slot,
                IntBinary(
                    "add",
                    IntLoad(digits_slot),
                    IntBinary(
                        "sdiv",
                        IntBinary("sub", IntLoad(digits_slot), IntConstant(1)),
                        IntConstant(3),
                    ),
                ),
            )
        )
        result_slot = self.new_temp()
        self.operations.append(
            HeapAlloc(result_slot, self._aligned_size(IntLoad(total_slot)), bump)
        )
        self.operations.append(HeapStore(IntLoad(result_slot), IntLoad(total_slot), 8))
        read_slot = self.new_temp()
        write_slot = self.new_temp()
        run_slot = self.new_temp()
        self.operations.append(
            Store(read_slot, IntBinary("sub", IntLoad(digits_slot), IntConstant(1)))
        )
        self.operations.append(
            Store(write_slot, IntBinary("sub", IntLoad(total_slot), IntConstant(1)))
        )
        self.operations.append(Store(run_slot, IntConstant(0)))
        source_data = IntBinary("add", IntLoad(source_slot), IntConstant(8))
        result_data = IntBinary("add", IntLoad(result_slot), IntConstant(8))
        start = self.new_label("group_start")
        end = self.new_label("group_end")
        plain = self.new_label("group_plain")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(IntCompare("ge", IntLoad(read_slot), IntConstant(0)), end)
        )
        self.operations.append(
            JumpIfFalse(IntCompare("eq", IntLoad(run_slot), IntConstant(3)), plain)
        )
        self.operations.append(
            HeapStore(
                IntBinary("add", result_data, IntLoad(write_slot)),
                IntConstant(ord(",")),
                1,
            )
        )
        self.operations.append(
            Store(write_slot, IntBinary("sub", IntLoad(write_slot), IntConstant(1)))
        )
        self.operations.append(Store(run_slot, IntConstant(0)))
        self.operations.append(Label(plain))
        self.operations.append(
            HeapStore(
                IntBinary("add", result_data, IntLoad(write_slot)),
                HeapLoad(IntBinary("add", source_data, IntLoad(read_slot)), 1),
                1,
            )
        )
        self.operations.append(
            Store(write_slot, IntBinary("sub", IntLoad(write_slot), IntConstant(1)))
        )
        self.operations.append(
            Store(read_slot, IntBinary("sub", IntLoad(read_slot), IntConstant(1)))
        )
        self.operations.append(
            Store(run_slot, IntBinary("add", IntLoad(run_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
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

    def forget_conditional_list_lengths(self, bodies: list[ast.stmt]) -> None:
        """Drop the recorded length of every list a maybe-skipped block assigns.

        A list literal records its length under the name it is bound to, and a
        constant index is proved in range against that. Lowering a branch or a
        loop body records the length whether or not the block runs, so on the
        path that skipped it the recorded length belongs to a list that was
        never built - and an index past the real end was proved in range and
        read uninitialised memory instead of raising IndexError.
        """

        for name in self.assigned_names(bodies):
            self.list_lengths.pop(name, None)

    def forget_looped_list_lengths(self, body: list[ast.stmt]) -> None:
        """Drop the length of every list a loop body can resize, before it runs.

        A loop body is lowered once but runs many times, so a length recorded
        before the loop only describes the first pass. `xs.append(...)` after a
        `f(*xs)` in the same body leaves the star standing for one fewer
        argument than the second iteration actually has, which CPython answers
        with a TypeError.

        Deliberately over-broad: a name listed here only loses a build-time
        length, which costs an honest refusal, while one missed costs a wrong
        answer.
        """

        names = self.assigned_names(body)
        for statement in body:
            for node in ast.walk(statement):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                ):
                    names.add(node.func.value.id)
                elif isinstance(node, ast.Delete):
                    for target in node.targets:
                        if isinstance(target, ast.Subscript) and isinstance(
                            target.value, ast.Name
                        ):
                            names.add(target.value.id)
        for name in names:
            self.list_lengths.pop(name, None)

    def open_loop_else(self, node: ast.For | ast.While) -> str | None:
        """The label a `break` out of this loop jumps to, if it needs its own.

        A loop with an `else` gets two exit labels: the one it falls through to
        when the test fails, and this one past the `else` body. `break` targets
        this one, which is how the `else` is skipped without a run-time flag.
        A loop whose own body never breaks always runs its `else`, so it needs
        only the ordinary exit.
        """

        if not node.orelse or not self.block_breaks(node.body):
            return None
        # This label is reached both by a break and by falling off the end of
        # the `else` body, so a name the `else` body assigns holds different
        # values on the two paths and cannot stay a build-time constant.
        self.materialize_runtime_names(self.assigned_names(node.orelse))
        return self.new_label("loop_else_end")

    def close_loop_else(
        self, node: ast.For | ast.While, broke: str | None
    ) -> None:
        """Emit the loop's `else` body, and the label a `break` skipped it by.

        Called after the loop's own binding bookkeeping so that reading the
        loop variable in the `else` body of a possibly empty loop is refused
        the same way reading it after the loop is.
        """

        if not node.orelse:
            return
        bound_before = set(self.bound_names)
        for statement in node.orelse:
            self.statement(statement)
        if broke is not None:
            # The break path jumped over the `else` body, so a name first bound
            # there is not bound on every path that reaches here.
            self.possibly_unbound.update(self.bound_names - bound_before)
            self.operations.append(Label(broke))

    def if_statement(self, node: ast.If) -> None:
        try:
            condition = self.constant(node.test)
        except NativeCompileError as constant_error:
            try:
                runtime_condition = self.truth_value(node.test)
            except NativeCompileError as runtime_error:
                # Every runtime condition fails the fold, so "not a constant"
                # names nothing that is wrong here; the lowering failure does.
                # A fold that failed for a specific reason is the exception:
                # there the fold found the real fault and the integer path only
                # restates that the expression is out of the subset.
                if isinstance(constant_error, NotConstant):
                    raise runtime_error from constant_error
                raise constant_error from runtime_error
            mutated = self.assigned_names(node.body + node.orelse)
            self.materialize_runtime_names(mutated)
            bound_before = set(self.bound_names)
            false_label = self.new_label("if_false")
            end_label = self.new_label("if_end")
            self.operations.append(JumpIfFalse(runtime_condition, false_label))
            self.runtime_branch_depth += 1
            try:
                for statement in node.body:
                    self.statement(statement)
                if node.orelse:
                    self.operations.append(Jump(end_label))
                self.operations.append(Label(false_label))
                for statement in node.orelse:
                    self.statement(statement)
            finally:
                self.runtime_branch_depth -= 1
            if node.orelse:
                self.operations.append(Label(end_label))
            self.forget_conditional_list_lengths(node.body + node.orelse)
            self.mark_conditionally_bound(node, bound_before)
        else:
            branch = node.body if bool(condition) else node.orelse
            for statement in branch:
                self.statement(statement)

    def mark_conditionally_bound(
        self, node: ast.If, bound_before: set[str]
    ) -> None:
        """Refuse names one arm binds and the other does not.

        A name is bound after an `if` only if every path reaching that point
        binds it. Where that does not hold, CPython raises NameError while the
        slot holds whatever was there before - a stale constant for an integer,
        and an address that is not a block for anything on the heap, which is
        why a dict read this way probed forever instead of answering.

        There is no run-time "is it bound" bit to test, so this is refused at
        build time, the same way a loop that can run zero times already is.
        """

        paths = []
        if self.block_falls_through(node.body):
            paths.append(self.names_every_path_binds(node.body))
        if self.block_falls_through(node.orelse):
            paths.append(self.names_every_path_binds(node.orelse))
        if not paths:
            # Both arms leave by a raise or a return, so nothing follows.
            return
        certain = set.intersection(*paths)
        possible = self.assigned_names(node.body + node.orelse)
        possible.update(self.defined_function_names(node.body + node.orelse))
        for name in (possible - certain) - bound_before:
            self.bound_names.discard(name)
            # Drop any value the taken arm folded, so the read is refused
            # rather than answered from a branch that may not have run.
            self.values.pop(name, None)
            self.possibly_unbound.add(name)

    @classmethod
    def defined_function_names(cls, body: list[ast.stmt]) -> set[str]:
        return {
            statement.name
            for statement in body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    @classmethod
    def block_falls_through(cls, body: list[ast.stmt]) -> bool:
        """Whether control can reach the statement after ``body``."""

        for statement in body:
            if isinstance(
                statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)
            ):
                return False
            if isinstance(statement, ast.If) and statement.orelse:
                if not cls.block_falls_through(
                    statement.body
                ) and not cls.block_falls_through(statement.orelse):
                    return False
        return True

    @classmethod
    def names_every_path_binds(cls, body: list[ast.stmt]) -> set[str]:
        """The names ``body`` binds on every path through it.

        Deliberately narrow: a loop may run zero times and a `try` may jump out
        part-way, so neither counts, and only what is reached unconditionally
        does. Being wrong in this direction refuses a working program; being
        wrong in the other direction reads an unbound slot.
        """

        names: set[str] = set()
        for statement in body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    names.update(cls.target_names(target))
            elif isinstance(statement, ast.AnnAssign):
                if statement.value is not None and isinstance(
                    statement.target, ast.Name
                ):
                    names.add(statement.target.id)
            elif isinstance(statement, ast.AugAssign):
                names.update(cls.target_names(statement.target))
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(statement.name)
            elif isinstance(statement, ast.With):
                # The body of a `with` always runs, and its items bind too.
                for item in statement.items:
                    if item.optional_vars is not None:
                        names.update(cls.target_names(item.optional_vars))
                names.update(cls.names_every_path_binds(statement.body))
            elif isinstance(statement, ast.If):
                # An elif chain is a nested If, so without this an
                # if/elif/else that binds a name in every arm would be refused.
                arms = []
                if cls.block_falls_through(statement.body):
                    arms.append(cls.names_every_path_binds(statement.body))
                if cls.block_falls_through(statement.orelse):
                    arms.append(cls.names_every_path_binds(statement.orelse))
                if arms:
                    names.update(set.intersection(*arms))
            elif isinstance(
                statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)
            ):
                break
        return names

    def while_statement(self, node: ast.While) -> None:
        broke = self.open_loop_else(node)
        start = self.new_label("while_start")
        end = self.new_label("while_end")
        self.operations.append(Label(start))
        self.operations.append(JumpIfFalse(self.truth_value(node.test), end))
        self.break_targets.append(end if broke is None else broke)
        self.continue_targets.append(start)
        for statement in node.body:
            self.statement(statement)
        self.continue_targets.pop()
        self.break_targets.pop()
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
        self.close_loop_else(node, broke)

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
        if isinstance(node, ast.Name) and self.list_kind_of(node.id) is not None:
            # Read through the variable's own slot rather than a snapshot of
            # it. Appending moves the block and writes the new address back to
            # that slot, so a copy taken here would leave the walk on the
            # abandoned one.
            pointer_slot = self.slot(node.id)
        else:
            # A slice or a comprehension built a block nothing else can reach,
            # so there is nothing to keep up with.
            pointer_slot = self.new_temp()
            self.operations.append(Store(pointer_slot, self.list_pointer(node)))
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        return index_slot, pointer_slot, element_kind

    def iterated_list_name(self, node: ast.expr) -> str | None:
        """The name of the list a `for` walks, when it walks one by name."""

        if isinstance(node, ast.Name) and self.list_kind_of(node.id) is not None:
            return node.id
        return None

    def bind_list_element(
        self,
        target: str,
        index_slot: int,
        pointer_slot: int,
        element_kind: str,
        holds_bool: bool | None = None,
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
        # Inside the body the name is bound, whatever an earlier loop that
        # could run zero times left it as.
        self.possibly_unbound.discard(target)
        # The element carries the list's bool-ness: a name bound to one out of
        # a list of bools has to print True, not 1.
        if element_kind == "bool" or (element_kind == "int" and holds_bool is True):
            self.boolean_names.add(target)
        else:
            self.boolean_names.discard(target)
        if self.list_kind(element_kind) is not None:
            # The name and the element are the same block, so growing one
            # through the name would leave the other on the abandoned copy.
            self.shared_list_names.add(target)
        else:
            self.shared_list_names.discard(target)
        # Whatever length a literal recorded under this name is not this
        # value's length, and an index checked against it would be refused for
        # being outside a range it is inside.
        self.list_lengths.pop(target, None)
        self.container_bool[target] = element_kind == "bool"
        if element_kind == "float":
            self.operations.append(
                FloatStore(self.slot(target), BitsFloat(HeapLoad(address, 8)))
            )
        else:
            self.operations.append(Store(self.slot(target), HeapLoad(address, 8)))
        self.value_types[target] = self.element_value_type(element_kind)

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
        broke = self.open_loop_else(node)
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
        self.bind_list_element(
            item_name,
            index_slot,
            pointer_slot,
            element_kind,
            self.list_holds_bool(source),
        )
        self.break_targets.append(end_label if broke is None else broke)
        self.continue_targets.append(continue_label)
        walked = self.iterated_list_name(source)
        if walked is not None:
            self.iterated_lists.append(walked)
        for statement in node.body:
            self.statement(statement)
        if walked is not None:
            self.iterated_lists.pop()
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
        self.close_loop_else(node, broke)

    def zip_sources(self, node: ast.expr) -> list[ast.expr] | None:
        """The lists `zip(...)` walks together, or None."""

        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "zip"
            or node.func.id in self.functions
            or node.keywords
            or len(node.args) < 2
        ):
            return None
        if any(
            self.list_kind(self.expression_type(argument)) is None
            for argument in node.args
        ):
            return None
        return list(node.args)

    def for_over_zip(self, node: ast.For) -> None:
        """`for a, b in zip(xs, ys):` - one step along each list at a time.

        The walk stops with the shortest, which is checked at every step rather
        than once: an append inside the body lengthens the walk here exactly as
        it does for a single list, because each length is read from its own
        header each time round.
        """

        sources = self.zip_sources(node.iter)
        assert sources is not None
        if (
            not isinstance(node.target, (ast.Tuple, ast.List))
            or len(node.target.elts) != len(sources)
            or not all(isinstance(item, ast.Name) for item in node.target.elts)
        ):
            raise NativeCompileError(
                self.path,
                node,
                f"this native zip() loop walks {len(sources)} lists, so it "
                f"binds {len(sources)} names",
            )
        names = [item.id for item in node.target.elts]
        if len(set(names)) != len(names):
            raise NativeCompileError(
                self.path,
                node.target,
                "a native zip() loop binds one name per list, and these repeat",
            )
        broke = self.open_loop_else(node)
        was_bound = {name: name in self.bound_names for name in names}
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        walks = []
        for source, name in zip(sources, names):
            if isinstance(source, ast.Name) and self.list_kind_of(source.id) is not None:
                # Through the variable's own slot, so that an append inside the
                # body - which moves the block - is followed rather than lost.
                pointer_slot = self.slot(source.id)
            else:
                pointer_slot = self.new_temp()
                self.operations.append(
                    Store(pointer_slot, self.list_pointer(source))
                )
            walks.append(
                (pointer_slot, self.list_kind(self.expression_type(source)), source)
            )
        start_label = self.new_label("for_zip")
        continue_label = self.new_label("for_zip_continue")
        end_label = self.new_label("for_zip_end")
        self.operations.append(Label(start_label))
        for pointer_slot, _kind, _source in walks:
            self.operations.append(
                JumpIfFalse(
                    IntCompare(
                        "lt",
                        IntLoad(index_slot),
                        HeapLoad(
                            IntBinary("add", IntLoad(pointer_slot), IntConstant(8)),
                            8,
                        ),
                    ),
                    end_label,
                )
            )
        for name, (pointer_slot, element_kind, source) in zip(names, walks):
            self.bind_list_element(
                name,
                index_slot,
                pointer_slot,
                element_kind,
                self.list_holds_bool(source),
            )
        self.break_targets.append(end_label if broke is None else broke)
        self.continue_targets.append(continue_label)
        walked = [name for name in (self.iterated_list_name(s) for s in sources) if name]
        self.iterated_lists.extend(walked)
        for statement in node.body:
            self.statement(statement)
        for _ in walked:
            self.iterated_lists.pop()
        self.continue_targets.pop()
        self.break_targets.pop()
        self.operations.append(Label(continue_label))
        self.operations.append(
            Store(index_slot, IntBinary("add", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start_label))
        self.operations.append(Label(end_label))
        for name in names:
            if was_bound[name]:
                self.bound_names.add(name)
            else:
                # Either list may be empty, and then nothing is bound.
                self.possibly_unbound.add(name)
        self.close_loop_else(node, broke)

    def reversed_source(self, node: ast.expr):
        """The list `reversed(...)` walks backwards, or None."""

        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "reversed"
            or node.func.id in self.functions
        ):
            return None
        if len(node.args) != 1 or node.keywords:
            raise NativeCompileError(
                self.path, node, "native reversed() takes one runtime list"
            )
        if self.list_kind(self.expression_type(node.args[0])) is None:
            raise NativeCompileError(
                self.path,
                node,
                "native reversed() takes a runtime list; reversing a range or "
                "a string is not in the subset",
            )
        return node.args[0]

    def refuse_rebinding_the_iterated_list(self, node: ast.For, name: str) -> None:
        """Refuse a body that assigns to the name being walked backwards.

        The loop reads the block through the name's slot on every iteration, so
        that growth moving the block never leaves it on a stale copy. Rebinding
        the name would point it at a different list mid-loop, where CPython's
        iterator holds on to the original object - a different answer, not just
        a stale read.
        """

        # The target counts too: `for xs in reversed(xs)` would write each
        # element over the very pointer the walk reads.
        for statement in [node.target, *node.body]:
            for inner in ast.walk(statement):
                if (
                    isinstance(inner, ast.Name)
                    and isinstance(inner.ctx, ast.Store)
                    and inner.id == name
                ):
                    raise NativeCompileError(
                        self.path,
                        inner,
                        f"'{name}' is the list this loop is walking, and "
                        "rebinding it would leave the loop on a different list "
                        "than Python stays on; iterate a copy of the name "
                        "instead",
                    )

    def for_over_reversed(self, node: ast.For) -> None:
        """`for name in reversed(xs):` - the same block, walked from the back.

        Nothing is allocated: the indices run down from the last one, which is
        what CPython's reverse iterator does rather than building a copy. That
        iterator re-reads the list on every step and stops as soon as the index
        is past the current length, so the pointer and the length are read
        again each time round rather than snapshotted - a body that appends
        moves the block, and a snapshot would go on reading the abandoned one.
        """

        assert isinstance(node.target, ast.Name)
        source = self.reversed_source(node.iter)
        assert source is not None
        self.refuse_bool_list(node.iter, source, "reversed()")
        element_kind = self.list_kind(self.expression_type(source))
        broke = self.open_loop_else(node)
        name = node.target.id
        was_bound = name in self.bound_names
        if isinstance(source, ast.Name) and self.list_kind_of(source.id) is not None:
            # Walk through the name's own slot, which append() keeps current.
            self.refuse_rebinding_the_iterated_list(node, source.id)
            pointer_slot = self.slot(source.id)
        else:
            # A slice or a sorted copy is a block nothing else can reach, so
            # there is nothing for it to be stale with respect to.
            pointer_slot = self.new_temp()
            self.operations.append(Store(pointer_slot, self.list_pointer(source)))
        index_slot = self.new_temp()
        self.operations.append(
            Store(
                index_slot,
                IntBinary(
                    "sub",
                    HeapLoad(
                        IntBinary("add", IntLoad(pointer_slot), IntConstant(8)), 8
                    ),
                    IntConstant(1),
                ),
            )
        )
        start = self.new_label("for_reversed")
        continue_label = self.new_label("for_reversed_continue")
        end = self.new_label("for_reversed_end")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(IntCompare("ge", IntLoad(index_slot), IntConstant(0)), end)
        )
        # An index past the end exhausts CPython's reverse iterator rather than
        # skipping one element, which is what a list shortened mid-loop does.
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "lt",
                    IntLoad(index_slot),
                    HeapLoad(
                        IntBinary("add", IntLoad(pointer_slot), IntConstant(8)), 8
                    ),
                ),
                end,
            )
        )
        self.bind_list_element(name, index_slot, pointer_slot, element_kind)
        self.break_targets.append(end if broke is None else broke)
        self.continue_targets.append(continue_label)
        walked = self.iterated_list_name(source)
        if walked is not None:
            self.iterated_lists.append(walked)
        for statement in node.body:
            self.statement(statement)
        if walked is not None:
            self.iterated_lists.pop()
        self.continue_targets.pop()
        self.break_targets.pop()
        self.operations.append(Label(continue_label))
        self.operations.append(
            Store(index_slot, IntBinary("sub", IntLoad(index_slot), IntConstant(1)))
        )
        self.operations.append(Jump(start))
        self.operations.append(Label(end))
        if was_bound:
            self.bound_names.add(name)
        else:
            # The list may be empty, and then Python leaves the name unbound.
            self.possibly_unbound.add(name)
        self.close_loop_else(node, broke)

    def for_over_string(self, node: ast.For) -> None:
        """`for ch in s:` - one code point at a time, as Python iterates.

        A string never moves and never changes length, so unlike a list the
        block and the count are read once rather than at every step.
        """

        assert isinstance(node.target, ast.Name)
        name = node.target.id
        was_bound = name in self.bound_names
        broke = self.open_loop_else(node)
        pointer = self.materialize_int(self.string_pointer(node.iter))
        length = self.materialize_int(self.emit_code_point_count(pointer))
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        start = self.new_label("for_str")
        continue_label = self.new_label("for_str_continue")
        end = self.new_label("for_str_end")
        self.operations.append(Label(start))
        self.operations.append(
            JumpIfFalse(IntCompare("lt", IntLoad(index_slot), length), end)
        )
        self.values.pop(name, None)
        self.runtime_names.add(name)
        self.boolean_names.discard(name)
        self.possibly_unbound.discard(name)
        self.operations.append(
            Store(
                self.slot(name),
                self.emit_string_index(pointer, None, IntLoad(index_slot)),
            )
        )
        self.value_types[name] = "str"
        # A `break` skips the `else` body, so it leaves past it, not here.
        self.break_targets.append(end if broke is None else broke)
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
            # An empty string binds nothing, exactly as an empty list does.
            self.possibly_unbound.add(name)
        # After the bookkeeping, so that the `else` body is held to the same
        # rule as the code after the loop.
        self.close_loop_else(node, broke)

    def for_over_list(self, node: ast.For) -> None:
        """`for name in <list>:` - walk by index, since there is no iterator."""

        assert isinstance(node.target, ast.Name)
        broke = self.open_loop_else(node)
        name = node.target.id
        was_bound = name in self.bound_names
        index_slot, pointer_slot, element_kind = self.emit_list_iteration(
            node.iter, name
        )
        start = self.new_label("for_list")
        continue_label = self.new_label("for_list_continue")
        end = self.new_label("for_list_end")
        self.operations.append(Label(start))
        # CPython's list iterator holds an index and compares it against the
        # list's CURRENT length at every step, which is what makes an append
        # inside the loop extend the walk and a del shorten it. Reading the
        # length once would stop early or run past the end instead.
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "lt",
                    IntLoad(index_slot),
                    HeapLoad(
                        IntBinary("add", IntLoad(pointer_slot), IntConstant(8)),
                        8,
                    ),
                ),
                end,
            )
        )
        self.bind_list_element(
            name,
            index_slot,
            pointer_slot,
            element_kind,
            self.list_holds_bool(node.iter),
        )
        self.break_targets.append(end if broke is None else broke)
        self.continue_targets.append(continue_label)
        walked = self.iterated_list_name(node.iter)
        if walked is not None:
            self.iterated_lists.append(walked)
        for statement in node.body:
            self.statement(statement)
        if walked is not None:
            self.iterated_lists.pop()
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
        self.close_loop_else(node, broke)

    def dict_iteration_source(self, node: ast.expr) -> tuple[str, str] | None:
        """The dict name and what walking ``node`` yields, or None.

        `d` and `d.keys()` yield keys, `d.values()` values, `d.items()` both.
        The match is on the shape of the expression rather than on its type,
        because `d.keys()` has no type of its own here - anything looser would
        let a stray `d.keys()` elsewhere be taken for something it is not.
        """

        if isinstance(node, ast.Name):
            if self.dict_kinds_of(node.id) is None:
                return None
            return node.id, "keys"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"keys", "values", "items"}
            and not node.args
            and not node.keywords
            and isinstance(node.func.value, ast.Name)
            and self.dict_kinds_of(node.func.value.id) is not None
        ):
            return node.func.value.id, node.func.attr
        return None

    def dict_iteration_targets(self, node: ast.For, mode: str) -> list[str]:
        """The names a dict loop binds, refusing any target it cannot bind."""

        if mode == "items":
            if (
                not isinstance(node.target, (ast.Tuple, ast.List))
                or len(node.target.elts) != 2
                or not all(isinstance(item, ast.Name) for item in node.target.elts)
            ):
                raise NativeCompileError(
                    self.path,
                    node,
                    "a native items() loop binds two names, the key and the "
                    "value",
                )
            return [item.id for item in node.target.elts]
        if not isinstance(node.target, ast.Name):
            raise NativeCompileError(
                self.path,
                node,
                "a native loop over a dict binds one name; use items() to bind "
                "a key and a value",
            )
        return [node.target.id]

    def refuse_deleting_from_the_iterated_dict(
        self, node: ast.For, name: str
    ) -> None:
        """Refuse a body that deletes a key from the dict being walked.

        The walk counts along the insertion-order list, and deleting shifts
        every later key down one place, so the index then skips a key or
        re-reads a shifted one. The size check that guards this loop cannot see
        it: a body that deletes and inserts in the same pass leaves the count
        where it was, so the guard never fires and the loop simply visits the
        wrong keys.

        CPython decides this by key identity rather than size and raises
        RuntimeError for most of these, though not all - `del d[k]` followed by
        `d[k] = ...` is legal there and prints. Refusing the whole shape rejects
        that one legal program, which is the cost of not carrying a version
        stamp on the dict. A refusal is the acceptable half of that trade; a
        loop that silently visits the wrong keys is not.
        """

        for statement in node.body:
            for inner in ast.walk(statement):
                if not isinstance(inner, ast.Delete):
                    continue
                for target in inner.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == name
                    ):
                        raise NativeCompileError(
                            self.path,
                            inner,
                            f"a native `for` over {name!r} cannot delete from "
                            f"{name!r}: the walk counts along the insertion "
                            "order, and deleting shifts every later key, so the "
                            "loop would visit the wrong ones. Collect the keys "
                            "to remove and delete them after the loop",
                        )

    def refuse_rebinding_the_iterated_dict(self, node: ast.For, name: str) -> None:
        """Refuse a body that assigns to the name being walked.

        The loop reads the table through the name's slot on every iteration, so
        that it never touches a block that growth has moved. Rebinding the name
        would point it at a different dict mid-loop, where CPython keeps
        walking the original one - a different answer, not just a stale read.
        """

        # The target counts too. `for d, v in d.items()` would write the key
        # over the very slot the walk reads the table through, and the next
        # iteration would dereference a key as a pointer.
        for statement in [node.target, *node.body]:
            for inner in ast.walk(statement):
                if (
                    isinstance(inner, ast.Name)
                    and isinstance(inner.ctx, ast.Store)
                    and inner.id == name
                ):
                    raise NativeCompileError(
                        self.path,
                        inner,
                        f"'{name}' is the dict this loop is walking, and "
                        "rebinding it would leave the loop on a different "
                        "dict than Python stays on; iterate a copy of the "
                        "name instead",
                    )

    def bind_dict_key(
        self, target: str, key_slot: int, name: str, key_kind: str
    ) -> None:
        self.values.pop(target, None)
        self.runtime_names.add(target)
        # Inside the body the name is bound, whatever an earlier loop that
        # could run zero times left it as.
        self.possibly_unbound.discard(target)
        self.operations.append(Store(self.slot(target), IntLoad(key_slot)))
        self.value_types[target] = key_kind
        if key_kind == "int" and self.container_bool.get(
            self.dict_keys_name(name)
        ) is True:
            self.boolean_names.add(target)
        else:
            self.boolean_names.discard(target)

    def bind_dict_value(
        self, target: str, address: IntExpression, name: str, value_kind: str
    ) -> None:
        self.values.pop(target, None)
        self.runtime_names.add(target)
        self.possibly_unbound.discard(target)
        if value_kind == "float":
            self.boolean_names.discard(target)
            self.operations.append(
                FloatStore(self.slot(target), BitsFloat(HeapLoad(address, 8)))
            )
        else:
            self.operations.append(Store(self.slot(target), HeapLoad(address, 8)))
            if self.container_bool.get(name) is True:
                self.boolean_names.add(target)
            else:
                self.boolean_names.discard(target)
        self.value_types[target] = value_kind

    def for_over_dict(self, node: ast.For) -> None:
        """`for k in d:` and the keys()/values()/items() spellings of it.

        A dict iterates in insertion order, which the table cannot supply, so
        the walk is over the order list in the dict header and each key is
        looked back up to reach its value.
        """

        source = self.dict_iteration_source(node.iter)
        assert source is not None
        name, mode = source
        key_kind, value_kind = self.dict_kinds_of(name)
        targets = self.dict_iteration_targets(node, mode)
        self.refuse_rebinding_the_iterated_dict(node, name)
        self.refuse_deleting_from_the_iterated_dict(node, name)
        broke = self.open_loop_else(node)
        was_bound = {target: target in self.bound_names for target in targets}
        dict_slot = self.slot(name)
        count_slot = self.new_temp()
        self.operations.append(
            Store(
                count_slot,
                HeapLoad(IntBinary("add", IntLoad(dict_slot), IntConstant(8)), 8),
            )
        )
        index_slot = self.new_temp()
        self.operations.append(Store(index_slot, IntConstant(0)))
        start = self.new_label("for_dict")
        continue_label = self.new_label("for_dict_continue")
        end = self.new_label("for_dict_end")
        unchanged = self.new_label("for_dict_unchanged")
        self.operations.append(Label(start))
        # The size check comes before the bounds test because CPython makes it
        # first too: a dict grown inside the last iteration raises even though
        # there was nothing left to yield.
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "eq",
                    HeapLoad(
                        IntBinary("add", IntLoad(dict_slot), IntConstant(8)), 8
                    ),
                    IntLoad(count_slot),
                ),
                unchanged + "_changed",
            )
        )
        self.operations.append(Jump(unchanged))
        self.operations.append(Label(unchanged + "_changed"))
        self.raise_exception(
            "RuntimeError",
            b"RuntimeError: dictionary changed size during iteration\n",
        )
        self.operations.append(Label(unchanged))
        # Re-read the header every time: a store inside the body can have grown
        # the table, which moves it, and the order list moves on its own too.
        keys_slot = self.new_temp()
        self.operations.append(
            Store(
                keys_slot,
                HeapLoad(
                    IntBinary(
                        "add", IntLoad(dict_slot), IntConstant(self.DICT_KEYS_OFFSET)
                    ),
                    8,
                ),
            )
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "lt",
                    IntLoad(index_slot),
                    HeapLoad(
                        IntBinary("add", IntLoad(keys_slot), IntConstant(8)), 8
                    ),
                ),
                end,
            )
        )
        key_slot = self.new_temp()
        self.operations.append(
            Store(
                key_slot,
                HeapLoad(
                    IntBinary(
                        "add",
                        IntBinary(
                            "add",
                            IntLoad(keys_slot),
                            IntConstant(self.LIST_HEADER_BYTES),
                        ),
                        IntBinary("mul", IntLoad(index_slot), IntConstant(8)),
                    ),
                    8,
                ),
            )
        )
        if mode in {"keys", "items"}:
            self.bind_dict_key(targets[0], key_slot, name, key_kind)
        if mode in {"values", "items"}:
            # This key came out of the order list, so it is in the table; the
            # probe's found flag has nothing left to say and is dropped. A
            # string key is hashed again here, once per iteration, which is the
            # price of not caching entry addresses that growth would move.
            address_slot, _found, _key, _state = self.dict_probe(
                dict_slot, IntLoad(key_slot), key_kind
            )
            self.bind_dict_value(
                targets[-1],
                IntBinary("add", IntLoad(address_slot), IntConstant(16)),
                name,
                value_kind,
            )
        self.break_targets.append(end if broke is None else broke)
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
        for target in targets:
            if was_bound[target]:
                self.bound_names.add(target)
            else:
                # The dict may be empty, and then Python binds nothing.
                self.possibly_unbound.add(target)
        self.close_loop_else(node, broke)

    def comprehension_shape(self, node) -> str:
        """Check what the clauses are, and name the construct for messages."""

        what = (
            "list comprehension"
            if isinstance(node, ast.ListComp)
            else "generator expression"
        )
        for generator in node.generators:
            if generator.is_async or not isinstance(generator.target, ast.Name):
                raise NativeCompileError(
                    self.path,
                    node,
                    f"a native {what} binds a single name per `for` and is "
                    "not async",
                )
        return what

    def comprehension_parts(self, node):
        """The clauses with every target renamed out of the way, and the element.

        Python 3 gives a comprehension its own scope, so `[k for k in ...]`
        must not disturb an outer `k` - not its value and not its slot. The
        names are private and derived from this node, so asking for the
        element kind and emitting the loops agree on them. Renaming is
        sequential: a clause's iterable sees only the clauses before it, which
        is what separates `[b for a in xs for b in ys]` from the `ys` an outer
        scope might also call `a`. The clause index is part of the private
        name because `[a for a in xs for a in ys]` reuses one.
        """

        renames: dict[str, str] = {}

        def rewrite(tree: ast.expr) -> ast.expr:
            class _Rename(ast.NodeTransformer):
                def visit_Name(self, inner: ast.Name):
                    replacement = renames.get(inner.id)
                    if replacement is None:
                        return inner
                    return ast.copy_location(
                        ast.Name(id=replacement, ctx=inner.ctx), inner
                    )

            renamed = _Rename().visit(copy.deepcopy(tree))
            ast.fix_missing_locations(
                ast.Module(body=[ast.Expr(value=renamed)], type_ignores=[])
            )
            return renamed

        clauses = []
        for index, generator in enumerate(node.generators):
            iterable = rewrite(generator.iter)
            private = f"<comp-{id(node):x}-{index}:{generator.target.id}>"
            renames[generator.target.id] = private
            clauses.append(
                (private, iterable, [rewrite(test) for test in generator.ifs])
            )
        return clauses, rewrite(node.elt)

    def comprehension_answer(self, node, ask):
        """Ask ``ask`` about the element with every clause target bound.

        The targets are private names that exist only while the loops run, so
        whatever the enclosing scope had under them is put back afterwards.
        """

        clauses, element = self.comprehension_parts(node)
        restore: list[tuple[str, str | None, bool, bool]] = []
        try:
            for target, iterable, _conditions in clauses:
                item_kind = self.iterable_element_kind(iterable)
                restore.append(
                    (
                        target,
                        self.value_types.get(target),
                        target in self.bound_names,
                        target in self.boolean_names,
                    )
                )
                self.value_types[target] = self.element_value_type(item_kind or "int")
                self.bound_names.add(target)
                if item_kind == "bool":
                    self.boolean_names.add(target)
                else:
                    self.boolean_names.discard(target)
            return ask(element)
        finally:
            for target, kind, was_bound, was_bool in reversed(restore):
                if kind is None:
                    self.value_types.pop(target, None)
                else:
                    self.value_types[target] = kind
                if was_bound:
                    self.bound_names.add(target)
                else:
                    self.bound_names.discard(target)
                if was_bool:
                    self.boolean_names.add(target)
                else:
                    self.boolean_names.discard(target)

    def comprehension_element_kind(self, node, bindings=None) -> str:
        """The kind `[expr for t in it]` produces, without emitting anything."""

        what = self.comprehension_shape(node)
        kind = self.comprehension_answer(
            node, lambda element: self.element_kind_of(element, bindings)
        )
        if not self.storable_element_kind(kind):
            raise NativeCompileError(
                self.path,
                node,
                f"a native {what} builds integers, floats, strings, bools or "
                "lists",
            )
        return kind

    def comprehension_element_bool(self, node) -> bool:
        """Whether the element this comprehension produces is a bool."""

        self.comprehension_shape(node)
        return self.comprehension_answer(node, self.renders_as_bool)

    def comprehension_source(
        self,
        node,
        outermost: bool,
        target: str,
        iterable: ast.expr,
        conditions: list[ast.expr],
        bindings=None,
        call_stack: tuple[int, ...] = (),
    ) -> ComprehensionSource:
        """Evaluate one clause's source and describe the loop over it.

        Every source is measured here, before any loop is emitted, which is
        what makes the product of the spans an upper bound on the result and
        keeps an inner source from being rebuilt once per outer iteration.
        Hoisting is only sound if evaluating the source cannot raise where
        CPython would never have evaluated it at all, so an inner clause is
        restricted to sources that cannot raise.
        """

        over_range = (
            isinstance(iterable, ast.Call)
            and isinstance(iterable.func, ast.Name)
            and iterable.func.id == "range"
            and iterable.func.id not in self.functions
        )
        if not outermost and not self.hoistable_source(iterable, over_range):
            raise NativeCompileError(
                self.path,
                node,
                "a native comprehension evaluates every `for` source once, "
                "before the loops; a second or later source must therefore be "
                "a name holding a list, or range() over names and constants, "
                "so that hoisting it cannot raise where Python would not have "
                "reached it",
            )
        index_slot = self.new_temp()
        limit_slot = self.new_temp()
        if over_range:
            if not 1 <= len(iterable.args) <= 2 or iterable.keywords:
                raise NativeCompileError(
                    self.path,
                    node,
                    "a native comprehension over range takes one or two "
                    "arguments and steps by one",
                )
            start_slot = self.new_temp()
            if len(iterable.args) == 1:
                start = IntConstant(0)
                stop = self.materialize_int(
                    self.integer(iterable.args[0], bindings, call_stack)
                )
            else:
                start = self.materialize_int(
                    self.integer(iterable.args[0], bindings, call_stack)
                )
                stop = self.materialize_int(
                    self.integer(iterable.args[1], bindings, call_stack)
                )
            self.operations.append(Store(start_slot, start))
            self.operations.append(Store(limit_slot, stop))
            return ComprehensionSource(
                target=target,
                conditions=conditions,
                index_slot=index_slot,
                start=IntLoad(start_slot),
                limit_slot=limit_slot,
                span=IntBinary(
                    "sub", IntLoad(limit_slot), IntLoad(start_slot)
                ),
                element_kind="int",
                pointer_slot=None,
                holds_bool=False,
            )
        element_kind = self.list_kind(self.expression_type(iterable))
        if element_kind is None:
            raise NativeCompileError(
                self.path,
                node,
                "a native comprehension iterates a range or a runtime list",
            )
        holds_bool = self.list_holds_bool(iterable)
        pointer_slot = self.new_temp()
        self.operations.append(Store(pointer_slot, self.list_pointer(iterable)))
        self.operations.append(
            Store(
                limit_slot,
                HeapLoad(IntBinary("add", IntLoad(pointer_slot), IntConstant(8)), 8),
            )
        )
        return ComprehensionSource(
            target=target,
            conditions=conditions,
            index_slot=index_slot,
            start=IntConstant(0),
            limit_slot=limit_slot,
            span=IntLoad(limit_slot),
            element_kind=element_kind,
            pointer_slot=pointer_slot,
            holds_bool=holds_bool,
        )

    def hoistable_source(self, iterable: ast.expr, over_range: bool) -> bool:
        """Whether evaluating this source early can be told from not at all.

        `[1 for a in xs for b in range(0, p // q)]` with an empty `xs` and a
        zero `q` never evaluates the inner source in CPython. Hoisting it
        would raise ZeroDivisionError instead. Only sources that cannot raise
        are allowed there, which closes the difference by construction rather
        than by hoping the expression is harmless.
        """

        if over_range:
            return all(
                isinstance(argument, ast.Constant)
                or (
                    isinstance(argument, ast.Name)
                    and self.expression_type(argument) in {"int", "float"}
                )
                for argument in iterable.args
            )
        return (
            isinstance(iterable, ast.Name)
            and self.list_kind_of(iterable.id) is not None
        )

    def comprehension_reserve(
        self, sources: list[ComprehensionSource]
    ) -> IntExpression:
        """How many elements to reserve: the product of the sources.

        `mul` wraps, and a product that came out negative would be clamped to
        zero here, allocated as a bare header, and then written far past its
        end - which the arena guard cannot see, because the bump pointer never
        moved. Each factor is clamped to just past what the arena could hold
        before it is multiplied, so the running product never approaches the
        wrap, and the product itself is checked after every step.
        """

        limit = _HEAP_ARENA_BYTES // 8
        product_slot = self.new_temp()
        self.operations.append(Store(product_slot, IntConstant(1)))
        too_big = self.new_label("comp_too_big")
        sized = self.new_label("comp_sized")
        for source in sources:
            span_slot = self.new_temp()
            # An empty range gives a negative span; reserve nothing rather
            # than asking the arena for a negative number of bytes.
            self.operations.append(
                Store(
                    span_slot,
                    self.select_integer(
                        IntCompare("gt", source.span, IntConstant(0)),
                        source.span,
                        IntConstant(0),
                    ),
                )
            )
            self.operations.append(
                Store(
                    span_slot,
                    self.select_integer(
                        IntCompare("gt", IntLoad(span_slot), IntConstant(limit)),
                        IntConstant(limit + 1),
                        IntLoad(span_slot),
                    ),
                )
            )
            self.operations.append(
                Store(
                    product_slot,
                    IntBinary("mul", IntLoad(product_slot), IntLoad(span_slot)),
                )
            )
            self.operations.append(
                JumpIfFalse(
                    IntCompare(
                        "lt", IntLoad(product_slot), IntConstant(limit + 1)
                    ),
                    too_big,
                )
            )
        self.operations.append(Jump(sized))
        self.operations.append(Label(too_big))
        # Same report as guard_arena_limit: this is the arena running out, not
        # a Python-level exception.
        self.operations.append(Write(b"MemoryError: native arena exhausted\n", 2))
        self.operations.append(Exit(1))
        self.operations.append(Label(sized))
        return IntLoad(product_slot)

    def bind_comprehension_target(self, source: ComprehensionSource) -> None:
        """Bind one clause's private name to the item the loop is on."""

        if source.pointer_slot is None:
            self.values.pop(source.target, None)
            self.runtime_names.add(source.target)
            self.boolean_names.discard(source.target)
            self.operations.append(
                Store(self.slot(source.target), IntLoad(source.index_slot))
            )
            self.value_types[source.target] = "int"
            return
        self.bind_list_element(
            source.target,
            source.index_slot,
            source.pointer_slot,
            source.element_kind,
            source.holds_bool,
        )

    def emit_comprehension_loops(
        self,
        sources: list[ComprehensionSource],
        body,
        bindings=None,
        call_stack: tuple[int, ...] = (),
    ) -> None:
        """Emit one loop per clause, calling ``body`` at the innermost point."""

        def level(index: int) -> None:
            if index == len(sources):
                body()
                return
            source = sources[index]
            # This runs again on every iteration of the loop outside it, so
            # the index is reset here rather than where the source was
            # measured.
            self.operations.append(Store(source.index_slot, source.start))
            start_label = self.new_label("comp")
            step_label = self.new_label("comp_next")
            end_label = self.new_label("comp_end")
            self.operations.append(Label(start_label))
            self.operations.append(
                JumpIfFalse(
                    IntCompare(
                        "lt",
                        IntLoad(source.index_slot),
                        IntLoad(source.limit_slot),
                    ),
                    end_label,
                )
            )
            self.bind_comprehension_target(source)
            for condition in source.conditions:
                self.operations.append(
                    JumpIfFalse(
                        self.integer(condition, bindings, call_stack), step_label
                    )
                )
            level(index + 1)
            self.operations.append(Label(step_label))
            self.operations.append(
                Store(
                    source.index_slot,
                    IntBinary("add", IntLoad(source.index_slot), IntConstant(1)),
                )
            )
            self.operations.append(Jump(start_label))
            self.operations.append(Label(end_label))

        level(0)

    def comprehension_clause_sources(
        self, node, bindings=None, call_stack: tuple[int, ...] = ()
    ):
        """Measure every clause of ``node``, outermost first, and the element."""

        clauses, element = self.comprehension_parts(node)
        bound: set[str] = set()
        sources = []
        for index, (target, iterable, conditions) in enumerate(clauses):
            if any(
                isinstance(inner, ast.Name) and inner.id in bound
                for inner in ast.walk(iterable)
            ):
                raise NativeCompileError(
                    self.path,
                    node,
                    "a native comprehension measures every `for` source once, "
                    "before the loops run, so a later source cannot depend on "
                    "an earlier target: the size reserved for the result would "
                    "stop being an upper bound on it",
                )
            sources.append(
                self.comprehension_source(
                    node,
                    index == 0,
                    target,
                    iterable,
                    conditions,
                    bindings,
                    call_stack,
                )
            )
            bound.add(target)
        return sources, element

    def list_comprehension(self, node: ast.ListComp) -> IntExpression:
        """`[expr for t in it ...]`, with any number of `for` and `if` clauses.

        The result is sized from the sources, not from how many items survive
        the conditions, and the real count is written into the header at the
        end. Over-reserving costs arena space that is never reclaimed anyway;
        counting first would mean running the sources twice.
        """

        element_kind = self.comprehension_element_kind(node)
        bump = self.ensure_heap()
        sources, element = self.comprehension_clause_sources(node)
        reserve = self.comprehension_reserve(sources)
        result_slot = self.new_temp()
        self.operations.append(
            HeapAlloc(
                result_slot,
                IntBinary(
                    "add",
                    IntConstant(self.LIST_HEADER_BYTES),
                    IntBinary("mul", reserve, IntConstant(8)),
                ),
                bump,
            )
        )
        result = IntLoad(result_slot)
        self.operations.append(HeapStore(result, reserve, 8))
        count_slot = self.new_temp()
        self.operations.append(Store(count_slot, IntConstant(0)))

        def store_element() -> None:
            address = IntBinary(
                "add",
                IntBinary("add", result, IntConstant(self.LIST_HEADER_BYTES)),
                IntBinary("mul", IntLoad(count_slot), IntConstant(8)),
            )
            stored = self.element_word(element, element_kind)
            self.operations.append(HeapStore(address, stored, 8))
            self.operations.append(
                Store(
                    count_slot,
                    IntBinary("add", IntLoad(count_slot), IntConstant(1)),
                )
            )

        self.emit_comprehension_loops(sources, store_element)
        self.operations.append(
            HeapStore(
                IntBinary("add", result, IntConstant(8)), IntLoad(count_slot), 8
            )
        )
        return result

    def for_statement(self, node: ast.For) -> None:
        if self.dict_iteration_source(node.iter) is not None:
            self.for_over_dict(node)
            return
        self.refuse_set_iteration(node.iter)
        if self.zip_sources(node.iter) is not None:
            self.for_over_zip(node)
            return
        if (
            isinstance(node.target, ast.Name)
            and self.expression_type(node.iter) == "str"
        ):
            self.for_over_string(node)
            return
        if (
            isinstance(node.target, ast.Name)
            and self.reversed_source(node.iter) is not None
        ):
            self.for_over_reversed(node)
            return
        if (
            isinstance(node.target, ast.Name)
            and self.list_kind(self.expression_type(node.iter)) is not None
        ):
            self.for_over_list(node)
            return
        if (
            isinstance(node.target, ast.Name)
            and self.tuple_kinds(self.expression_type(node.iter)) is not None
        ):
            self.for_over_tuple(node)
            return
        if self.enumerate_source(node.iter) is not None:
            self.for_over_enumerate(node)
            return
        if (
            not isinstance(node.target, ast.Name)
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
            ) == "str":
                hint = "; only ranges and runtime lists are iterable here"
            raise NativeCompileError(
                self.path,
                node,
                "native for supports NAME in range(1-3 arguments), NAME in a "
                "runtime list, NAME in a tuple whose elements are all one "
                "kind, NAME in reversed(list), two names in enumerate(list), "
                "or a dict with keys(), values() or items()"
                + hint,
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
        broke = self.open_loop_else(node)
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
        self.break_targets.append(end_label if broke is None else broke)
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
        self.close_loop_else(node, broke)

    def expression_statement(self, node: ast.expr) -> None:
        if not isinstance(node, ast.Call):
            raise NativeCompileError(self.path, node, "only print() and SystemExit are valid expression statements")
        if self.list_method_call(node):
            return
        if self.set_method_call(node):
            return
        if self.string_method_kind(node) is not None:
            # Strings are immutable, so this changed nothing. CPython would
            # discard the result too, but silently, and a discarded
            # `s.replace(...)` is a bug every time it is written.
            assert isinstance(node.func, ast.Attribute)
            raise NativeCompileError(
                self.path,
                node,
                f"str.{node.func.attr}() returns a new string and changes "
                "nothing; assign the result or print it",
            )
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
            arguments = node.args
            if len(arguments) > 1 and all(
                self.can_pre_evaluate_print_argument(argument)
                for argument in arguments
            ):
                # print() evaluates every argument before it writes any of
                # them, so an argument that raises must leave no output at
                # all. Writing while walking printed the earlier ones first.
                #
                # All or nothing: only scalars can be bound to a hidden name,
                # and pre-evaluating some while leaving others to the write
                # loop would run them out of order - `print(xs.pop(), len(xs))`
                # measured the list before it was shortened. A print with a
                # container argument keeps the old order-by-construction walk
                # and with it the chance of a partial line before a raise.
                arguments = [
                    self.pre_evaluate_print_argument(argument)
                    for argument in arguments
                ]
            for index, argument in enumerate(arguments):
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
            # The store has to name the right register file even though nothing
            # reads the temp, because the encoder rejects a node handed to the
            # wrong one rather than guessing.
            call = self.extern_call(node, {}, (), discarded=True)
            self.operations.append(
                FloatStore(self.new_temp(), call)
                if call.result == "f64"
                else Store(self.new_temp(), call)
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
            self.refuse_starred_call(node)
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
    # repr() never needs more than seventeen digits, but a fixed-point field
    # does: 1e308 with a hundred decimals is 309 integer digits and a hundred
    # more, and the buffers are written without a bound check.
    DTOA_DIGITS_BYTES = 512
    DTOA_TEXT_OFFSET = DTOA_DIGITS_OFFSET + DTOA_DIGITS_BYTES
    DTOA_TEXT_BYTES = 8 + 512
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

    def emit_float_fixed(
        self, value: FloatExpression, precision: int
    ) -> tuple[IntExpression, int]:
        """Render ``abs(value)`` with exactly ``precision`` decimals.

        This is not repr() with the tail cut off. CPython rounds the exact
        binary value, so ``f"{2.675:.2f}"`` is ``2.67``: the double really is
        2.674999999999999822..., and any shortcut that works from the shortest
        decimal gets it wrong. The value is therefore held exactly as R/S over
        the same big integers repr() uses, scaled by a power of ten, and
        rounded half to even against the remainder.

        Returns the magnitude text and a slot holding one when the sign is
        negative. The sign is left to the caller because zero padding and '='
        alignment both put the pad between the sign and the digits.
        """

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
            Store(fraction_slot, IntBinary("and", bits, IntConstant((1 << 52) - 1)))
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
        finish = self.new_label("fixed_finish")

        ordinary = self.new_label("fixed_ordinary")
        self.operations.append(
            JumpIfFalse(
                IntCompare("eq", IntLoad(biased_slot), IntConstant(0x7FF)), ordinary
            )
        )
        infinite = self.new_label("fixed_inf")
        self.operations.append(
            JumpIfFalse(
                IntCompare("ne", IntLoad(fraction_slot), IntConstant(0)), infinite
            )
        )
        # A NaN prints unsigned even when its sign bit is set.
        self.operations.append(Store(negative_slot, IntConstant(0)))
        self.dtoa_append_text(length_slot, b"nan")
        self.operations.append(Jump(finish))
        self.operations.append(Label(infinite))
        self.dtoa_append_text(length_slot, b"inf")
        self.operations.append(Jump(finish))
        self.operations.append(Label(ordinary))

        # Zero has no digits to scale, and the scaling loop below would spin
        # forever on it, so it is laid out directly. Negative zero keeps its
        # sign, which is already in negative_slot.
        nonzero = self.new_label("fixed_nonzero")
        self.operations.append(
            JumpIfFalse(
                IntCompare(
                    "eq",
                    IntBinary("or", IntLoad(biased_slot), IntLoad(fraction_slot)),
                    IntConstant(0),
                ),
                nonzero,
            )
        )
        self.dtoa_append_text(length_slot, b"0")
        if precision > 0:
            self.dtoa_append_text(length_slot, b"." + b"0" * precision)
        self.operations.append(Jump(finish))
        self.operations.append(Label(nonzero))

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
        exponent = IntLoad(exponent_slot)
        value_shift = self.materialize_int(
            self.select_integer(
                IntCompare("gt", exponent, IntConstant(0)), exponent, IntConstant(0)
            )
        )
        scale_shift = self.materialize_int(
            self.select_integer(
                IntCompare("lt", exponent, IntConstant(0)),
                IntUnary("neg", exponent),
                IntConstant(0),
            )
        )

        r = self.bignum(self.DTOA_R)
        s = self.bignum(self.DTOA_S)
        t1 = self.bignum(self.DTOA_T1)
        self.bn_set(r, IntLoad(significand_slot))
        self.bn_double_times(r, value_shift)
        self.bn_set(s, IntConstant(1))
        self.bn_double_times(s, scale_shift)

        # Scale into 1/10 <= R/S < 1, counting where the decimal point lands.
        # Unlike repr() there are no neighbour margins here: the target is a
        # fixed number of places, not the shortest string that reads back.
        point_slot = self.new_temp()
        self.operations.append(Store(point_slot, IntConstant(0)))
        up_start = self.new_label("fixed_scale_up")
        up_end = self.new_label("fixed_scale_up_end")
        self.operations.append(Label(up_start))
        high = self.bn_compare(r, s)
        self.operations.append(
            JumpIfFalse(IntCompare("ge", IntLoad(high), IntConstant(0)), up_end)
        )
        self.bn_mul_small(s, IntConstant(10))
        self.operations.append(
            Store(point_slot, IntBinary("add", IntLoad(point_slot), IntConstant(1)))
        )
        self.operations.append(Jump(up_start))
        self.operations.append(Label(up_end))

        down_start = self.new_label("fixed_scale_down")
        down_end = self.new_label("fixed_scale_down_end")
        self.operations.append(Label(down_start))
        self.bn_copy(t1, r)
        self.bn_mul_small(t1, IntConstant(10))
        low = self.bn_compare(t1, s)
        self.operations.append(
            JumpIfFalse(IntCompare("lt", IntLoad(low), IntConstant(0)), down_end)
        )
        self.bn_copy(r, t1)
        self.operations.append(
            Store(point_slot, IntBinary("sub", IntLoad(point_slot), IntConstant(1)))
        )
        self.operations.append(Jump(down_start))
        self.operations.append(Label(down_end))

        count_slot = self.new_temp()
        self.operations.append(Store(count_slot, IntConstant(0)))
        last_slot = self.new_temp()
        self.operations.append(Store(last_slot, IntConstant(0)))
        wanted_slot = self.new_temp()
        self.operations.append(
            Store(
                wanted_slot,
                IntBinary("add", IntLoad(point_slot), IntConstant(precision)),
            )
        )
        laid_out = self.new_label("fixed_layout")
        # Below half of the last kept place there is nothing to generate and
        # nothing to round; the answer is all zeros, and the digit loop would
        # otherwise round up on a remainder that is not a remainder yet.
        representable = self.new_label("fixed_representable")
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(wanted_slot), IntConstant(0)), representable
            )
        )
        self.operations.append(Store(point_slot, IntConstant(-precision)))
        self.operations.append(Jump(laid_out))
        self.operations.append(Label(representable))

        generate = self.new_label("fixed_generate")
        generated = self.new_label("fixed_generated")
        self.operations.append(Label(generate))
        self.operations.append(
            JumpIfFalse(
                IntCompare("lt", IntLoad(count_slot), IntLoad(wanted_slot)), generated
            )
        )
        self.bn_mul_small(r, IntConstant(10))
        digit_slot = self.new_temp()
        self.operations.append(Store(digit_slot, IntConstant(0)))
        divide = self.new_label("fixed_divide")
        divided = self.new_label("fixed_divided")
        self.operations.append(Label(divide))
        quotient = self.bn_compare(r, s)
        self.operations.append(
            JumpIfFalse(IntCompare("ge", IntLoad(quotient), IntConstant(0)), divided)
        )
        # R < 10S on entry, so this runs at most nine times: no big division.
        self.bn_sub(r, s)
        self.operations.append(
            Store(digit_slot, IntBinary("add", IntLoad(digit_slot), IntConstant(1)))
        )
        self.operations.append(Jump(divide))
        self.operations.append(Label(divided))
        self.operations.append(
            HeapStore(
                IntBinary("add", digits, IntLoad(count_slot)), IntLoad(digit_slot), 1
            )
        )
        self.operations.append(Store(last_slot, IntLoad(digit_slot)))
        self.operations.append(
            Store(count_slot, IntBinary("add", IntLoad(count_slot), IntConstant(1)))
        )
        self.operations.append(Jump(generate))
        self.operations.append(Label(generated))

        # Round half to even against the exact remainder. With no digits at all
        # the notional last digit is zero, which is even, so a tie rounds down.
        t2 = self.bignum(self.DTOA_T2)
        self.bn_copy(t2, r)
        self.bn_add(t2, r)
        halfway = self.bn_compare(t2, s)
        bump_slot = self.new_temp()
        self.operations.append(
            Store(
                bump_slot,
                self.select_integer(
                    IntCompare("gt", IntLoad(halfway), IntConstant(0)),
                    IntConstant(1),
                    self.select_integer(
                        IntBinary(
                            "and",
                            IntCompare("eq", IntLoad(halfway), IntConstant(0)),
                            IntBinary("and", IntLoad(last_slot), IntConstant(1)),
                        ),
                        IntConstant(1),
                        IntConstant(0),
                    ),
                ),
            )
        )
        self.operations.append(
            JumpIfFalse(IntCompare("ne", IntLoad(bump_slot), IntConstant(0)), laid_out)
        )
        carry_slot = self.new_temp()
        self.operations.append(
            Store(carry_slot, IntBinary("sub", IntLoad(count_slot), IntConstant(1)))
        )
        carry = self.new_label("fixed_carry")
        overflow = self.new_label("fixed_carry_overflow")
        self.operations.append(Label(carry))
        self.operations.append(
            JumpIfFalse(IntCompare("ge", IntLoad(carry_slot), IntConstant(0)), overflow)
        )
        place = IntBinary("add", digits, IntLoad(carry_slot))
        raised_slot = self.new_temp()
        self.operations.append(
            Store(raised_slot, IntBinary("add", HeapLoad(place, 1), IntConstant(1)))
        )
        self.operations.append(
            JumpIfFalse(
                IntCompare("eq", IntLoad(raised_slot), IntConstant(10)),
                carry + "_settled",
            )
        )
        self.operations.append(HeapStore(place, IntConstant(0), 1))
        self.operations.append(
            Store(carry_slot, IntBinary("sub", IntLoad(carry_slot), IntConstant(1)))
        )
        self.operations.append(Jump(carry))
        self.operations.append(Label(carry + "_settled"))
        self.operations.append(HeapStore(place, IntLoad(raised_slot), 1))
        self.operations.append(Jump(laid_out))
        self.operations.append(Label(overflow))
        # Every digit was a nine, so the string grows a leading one. The rest
        # were zeroed on the way out; only the new last place is fresh.
        self.operations.append(
            HeapStore(
                IntBinary("add", digits, IntLoad(count_slot)), IntConstant(0), 1
            )
        )
        self.operations.append(
            Store(count_slot, IntBinary("add", IntLoad(count_slot), IntConstant(1)))
        )
        self.operations.append(
            Store(point_slot, IntBinary("add", IntLoad(point_slot), IntConstant(1)))
        )
        self.operations.append(HeapStore(digits, IntConstant(1), 1))

        self.operations.append(Label(laid_out))
        # count - point is the precision, always, which is what makes the
        # layout a single branch on where the point sits.
        point = IntLoad(point_slot)
        count = IntLoad(count_slot)
        leading = self.new_label("fixed_leading")
        joined = self.new_label("fixed_joined")
        self.operations.append(
            JumpIfFalse(IntCompare("le", point, IntConstant(0)), leading + "_no")
        )
        self.operations.append(Jump(leading))
        self.operations.append(Label(leading + "_no"))
        self.dtoa_append_digits(length_slot, IntConstant(0), point)
        if precision > 0:
            self.dtoa_append_text(length_slot, b".")
            self.dtoa_append_digits(length_slot, point, count)
        self.operations.append(Jump(joined))
        self.operations.append(Label(leading))
        self.dtoa_append_text(length_slot, b"0")
        if precision > 0:
            self.dtoa_append_text(length_slot, b".")
            self.dtoa_append_repeat(
                length_slot, IntConstant(ord("0")), IntUnary("neg", point)
            )
            self.dtoa_append_digits(length_slot, IntConstant(0), count)
        self.operations.append(Label(joined))

        self.operations.append(Label(finish))
        self.operations.append(HeapStore(text, IntLoad(length_slot), 8))
        return text, negative_slot

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
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "bool"
            and node.func.id not in self.functions
        ):
            return True
        if isinstance(node, ast.Compare):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return True
        if isinstance(node, ast.BoolOp):
            # `a and b` yields an operand, so it is a bool only if both are.
            return all(self.renders_as_bool(value) for value in node.values)
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)
        ):
            # `True & True` is True, but `True & 1` is 1: bool has its own
            # implementation of these three and it applies only between bools.
            return self.renders_as_bool(node.left) and self.renders_as_bool(
                node.right
            )
        if isinstance(node, ast.Name):
            return node.id in self.boolean_names
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
            # A tuple answers per element, so `(True, 1)[0]` is a bool and
            # `(True, 1)[1]` is not - which is the whole point of it.
            element = self.tuple_subscript_kind(node)
            if element is not None:
                return element == "bool"
            # A list says what its elements are in its own tag, which is what
            # answers for one nobody named - `xs[0][1]`, a slice of a slice, a
            # comprehension used in place. A dict keeps the answer under a name.
            holds = self.list_holds_bool(node.value)
            if holds is not None:
                return holds
            if isinstance(node.value, ast.Name):
                return self.container_bool.get(node.value.id) is True
            return False
        if isinstance(node, ast.Attribute):
            native_class = self.resolve_object_class(node.value)
            if native_class is not None:
                # A field is one slot shared by every instance of the class, so
                # the answer belongs to the class and the field, not to a name.
                return (
                    self.container_bool.get(f"{native_class.name}.{node.attr}")
                    is True
                )
        if isinstance(node, ast.IfExp):
            # Both arms land in the same slot, so they have to agree. A
            # branching function body is normalised into one of these, which is
            # where a mixed return shows up.
            taken = self.renders_as_bool(node.body)
            other = self.renders_as_bool(node.orelse)
            if taken != other:
                raise NativeCompileError(
                    self.path,
                    node,
                    "one arm of this is a bool and the other is a number; they "
                    "share a slot, and one slot cannot print both ways, so wrap "
                    "the bool in int() or make both arms the same kind",
                )
            return taken
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self._AGGREGATE_CALLS
            and node.func.id not in self.functions
            and len(node.args) == 1
        ):
            if node.func.id in {"any", "all"}:
                return True
            if node.func.id in {"min", "max"}:
                # The answer is one of the elements, so it prints the way they
                # do. sum() of bools is an int in CPython, so it is not here.
                argument = node.args[0]
                if isinstance(argument, ast.GeneratorExp):
                    try:
                        return self.comprehension_element_bool(argument)
                    except NativeCompileError:
                        return False
                return self.list_holds_bool(argument) is True
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if (
                node.func.id in {"min", "max"}
                and node.func.id not in self.functions
                and len(node.args) == 2
                and not node.keywords
            ):
                # min(a, b) returns one of its arguments, so it is a bool when
                # both of them are - the same rule `a and b` follows.
                return all(self.renders_as_bool(item) for item in node.args)
            return self.call_returns_bool(node)
        if self.string_method_kind(node) == "bool":
            return True
        return False

    def call_returns_bool(self, node: ast.Call) -> bool:
        """Whether a call to a native function yields a bool.

        The answer depends on the arguments - `def same(a, b): return a == b`
        always yields one, `def pick(a, b): return a` yields one only when `a`
        does - so the parameters are given the bool-ness of the arguments and
        the body is asked under that.
        """

        function = self.functions.get(node.func.id)
        if function is None or id(function) in self._bool_query:
            return False
        # Ask about the arguments the call really passes. An unexpanded star
        # looks like the wrong number of them, and the "not a bool" that would
        # come back prints 1 where CPython prints True.
        node = self.call_with_expanded_stars(node)
        if node.keywords or len(node.args) != len(function.parameters):
            return False
        previous = set(self.boolean_names)
        self._bool_query.add(id(function))
        try:
            self.boolean_names.difference_update(function.parameters)
            self.boolean_names.update(
                parameter
                for parameter, argument in zip(function.parameters, node.args)
                if self.renders_as_bool(argument)
            )
            if function.expression is not None:
                return self.renders_as_bool(function.expression)
            return self.body_returns_bool(function.body, node)
        finally:
            self._bool_query.discard(id(function))
            self.boolean_names.clear()
            self.boolean_names.update(previous)

    def body_returns_bool(self, body, location: ast.AST) -> bool:
        """Whether every value a statement body returns is a bool.

        A local assigned from a bool carries it forward, so the statements are
        walked in order. A body that returns a bool on one path and a number on
        another cannot be represented - one slot cannot print both ways - so it
        is refused rather than resolved arbitrarily.
        """

        answers: list[bool] = []

        def walk(statements) -> None:
            for statement in statements:
                if isinstance(statement, ast.Assign) and len(
                    statement.targets
                ) == 1 and isinstance(statement.targets[0], ast.Name):
                    name = statement.targets[0].id
                    if self.renders_as_bool(statement.value):
                        self.boolean_names.add(name)
                    else:
                        self.boolean_names.discard(name)
                elif isinstance(statement, ast.AugAssign) and isinstance(
                    statement.target, ast.Name
                ):
                    self.boolean_names.discard(statement.target.id)
                elif isinstance(statement, ast.Return) and statement.value:
                    answers.append(self.renders_as_bool(statement.value))
                for field in ("body", "orelse", "finalbody"):
                    nested = getattr(statement, field, None)
                    if nested:
                        walk(nested)
                for handler in getattr(statement, "handlers", ()):
                    walk(handler.body)

        walk(list(body))
        if answers and all(answers):
            return True
        if any(answers) and not all(answers):
            raise NativeCompileError(
                self.path,
                location,
                "this function returns a bool on one path and a number on "
                "another; one slot cannot print both ways, so wrap the bool in "
                "int() or return one kind throughout",
            )
        return False

    def emit_bool_to_string(self, value: IntExpression) -> IntExpression:
        base = IntLoad(self.ensure_bool_text())
        return self.select_integer(
            IntCompare("ne", value, IntConstant(0)),
            base,
            IntBinary("add", base, IntConstant(16)),
        )

    def can_pre_evaluate_print_argument(self, node: ast.expr) -> bool:
        """Whether this argument can be evaluated ahead of the writing.

        A name or a constant needs no evaluating, and a scalar can be held in
        a hidden name. Anything else - a list, a dict, a string expression -
        has nowhere to be held: binding a container to a second name is an
        alias, which is refused because growth moves the block.
        """

        if isinstance(node, (ast.Name, ast.Constant)):
            return True
        try:
            return self.expression_type(node) in {"int", "bool", "float"}
        except NativeCompileError:
            return False

    def pre_evaluate_print_argument(self, node: ast.expr) -> ast.expr:
        """Bind a scalar print() argument to a hidden name, and return the name.

        Only scalars. Binding a list or a dict to a second name is an alias,
        which is refused because growth would move the block and update one
        name and not the other - and neither can raise while being rendered,
        so there is nothing here to gain by it.
        """

        if isinstance(node, (ast.Name, ast.Constant)):
            return node  # Already a value; evaluating it cannot raise.
        try:
            kind = self.expression_type(node)
        except NativeCompileError:
            return node  # Let the write path report what is wrong with it.
        if kind not in ("int", "bool", "float"):
            return node
        name = f"__print_argument_{self.print_argument_count}"
        self.print_argument_count += 1
        self.assignment(name, node)
        return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)

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
        elif self.tuple_kinds(kind) is not None:
            self.emit_print_tuple(node, self.tuple_kinds(kind))
            return
        elif self.list_kind(kind) in {"int", "float", "bool"}:
            self.emit_print_list(node, self.list_kind(kind))
            return
        elif (
            self.dict_kinds(kind) is not None
            and isinstance(node, ast.Name)
            and self.dict_kinds(kind)[0] == "int"
            and self.dict_kinds(kind)[1] in {"int", "float"}
        ):
            text = self.materialize_int(self.emit_dict_to_string(node))
            self.operations.append(
                WriteRuntime(
                    IntBinary("add", text, IntConstant(8)), HeapLoad(text, 8)
                )
            )
            return
        elif self.list_kind(kind) is not None:
            # A string or a nested list would need the repr of each element,
            # and choosing the quote character and the backslash escapes for a
            # string built at run time is not implemented. The same limit
            # applies to a tuple's string elements.
            raise NativeCompileError(
                self.path,
                node,
                f"native print() renders a list of integers, floats or bools; "
                f"this one holds {self.kind_noun(self.list_kind(kind))}, and "
                "CPython prints the repr of every element, which for a runtime "
                "string means picking its quotes and escapes. Print the "
                "elements one at a time instead",
            )
        elif self.set_kind(kind) is not None:
            # Rendering a set means choosing an order to render it in, and no
            # order here matches CPython's.
            raise NativeCompileError(
                self.path,
                node,
                f"native print() cannot render a runtime {kind}: writing one "
                "means choosing the order of its elements, and CPython's set "
                "order is unspecified, so any choice would print the wrong "
                "answer for some input. Print len(s), or `x in s`"
                + (
                    ", or print sorted(s) one element at a time"
                    if self.set_kind(kind) == "int"
                    else ""
                ),
            )
        else:
            raise NativeCompileError(
                self.path,
                node,
                f"native print() cannot render a runtime {kind} yet: nothing "
                "here writes the brackets and commas CPython would, so print "
                "the elements one at a time. Integers, floats, bools and "
                "strings are supported",
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
        returns_string: bool = False,
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
            if is_float_expression(argument):
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
        previous_returns_string = self.returns_string
        self.returns_string = returns_string
        self.active_functions.append((identity, function_name))
        try:
            for statement in body:
                self.statement(statement)
        finally:
            self.returns_string = previous_returns_string
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

    def starred_refusal(self, value: ast.expr) -> str:
        accepted = (
            "a list or tuple literal, or a name holding a list or tuple whose "
            "length is still known here"
        )
        if isinstance(value, ast.Name) and self.list_kind_of(value.id) is not None:
            return (
                f"the length of {value.id!r} is no longer a build-time fact, so "
                "* cannot say how many arguments it stands for. A list loses its "
                "known length once it is appended to, del'd from, assigned from "
                "anything but a literal, or assigned inside a block that may not "
                "run. The callee is inlined against a parameter count fixed at "
                f"build time and needs a known number of arguments: * accepts "
                f"{accepted}"
            )
        return (
            "a native call is inlined against a parameter count fixed at build "
            "time, so the callee needs a number of arguments known then; * "
            f"accepts {accepted}"
        )

    def starred_elements(self, node: ast.Starred) -> list[ast.expr]:
        """The arguments one `*` stands for, as expressions to splice in."""

        value = node.value
        if isinstance(value, (ast.List, ast.Tuple)):
            return list(value.elts)
        length: int | None = None
        if isinstance(value, ast.Name):
            if self.list_kind_of(value.id) is not None:
                length = self.list_lengths.get(value.id)
            else:
                kinds = self.tuple_kinds_of(value.id)
                if kinds is not None:
                    length = len(kinds)
        if length is None:
            raise NativeCompileError(self.path, node, self.starred_refusal(value))
        assert isinstance(value, ast.Name)
        elements: list[ast.expr] = []
        for index in range(length):
            element = ast.Subscript(
                value=ast.Name(id=value.id, ctx=ast.Load()),
                slice=ast.Constant(value=index),
                ctx=ast.Load(),
            )
            ast.copy_location(element, node)
            ast.fix_missing_locations(element)
            elements.append(element)
        return elements

    def call_with_expanded_stars(self, node: ast.Call) -> ast.Call:
        """Spell `f(*xs)` out as `f(xs[0], xs[1], ...)`, or say why it cannot be.

        Indexing already knows an element's kind, whether it renders as a bool,
        and how to prove a constant index in range, so the expansion is written
        as subscripts and handed back to the ordinary argument path rather than
        lowered here. A new node every time: a function body is deep-copied once
        and re-lowered per call site, so a rewrite kept on the shared body would
        freeze one call site's known lengths into another's.
        """

        if not any(isinstance(argument, ast.Starred) for argument in node.args):
            return node
        companions = [
            argument for argument in node.args if not isinstance(argument, ast.Starred)
        ]
        companions.extend(keyword.value for keyword in node.keywords)
        for other in companions:
            if any(isinstance(inner, ast.Call) for inner in ast.walk(other)):
                raise NativeCompileError(
                    self.path,
                    other,
                    "a native starred call cannot share its argument list with "
                    "a call: CPython unpacks the list after that call has run, "
                    "so a call that appended to it would change how many "
                    "arguments the star stands for, and this build already "
                    "fixed that number. Evaluate the call into a name first",
                )
        expanded: list[ast.expr] = []
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                expanded.extend(self.starred_elements(argument))
            else:
                expanded.append(argument)
        replacement = ast.Call(func=node.func, args=expanded, keywords=node.keywords)
        return ast.copy_location(replacement, node)

    def refuse_starred_call(self, node: ast.expr) -> None:
        """Name the callees that cannot take a `*` argument at all."""

        if not isinstance(node, ast.Call) or not any(
            isinstance(argument, ast.Starred) for argument in node.args
        ):
            return
        raise NativeCompileError(
            self.path,
            node,
            "* argument expansion is supported only when calling a function, "
            "lambda or procedure defined in this file, because the expansion "
            "is spelled out against that callee's parameter count at build "
            "time. This callee takes its arguments another way, so write them "
            "out",
        )

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
        node = self.call_with_expanded_stars(node)
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
        if id(function) in self._kind_query:
            # Asking this function's kind is what led here, so the program is
            # recursive. Answering "integer" sends the caller down the ordinary
            # inlining path, which refuses recursion with a located message -
            # far better than this walk running out of Python stack and killing
            # the compiler with a traceback.
            return "int"
        node = self.call_with_expanded_stars(node)
        if function.expression is None:
            # A statement body is inlined, and what it answers with is only
            # settled once it has been - except for the one thing the call site
            # must decide first, which is whether it is holding a number or the
            # address of a string block.
            if self.statement_body_returns_string(function, node, bindings):
                return "str"
            # Say integer so the caller takes the integer path, where a float
            # return is caught and reported precisely.
            return "int"
        if node.keywords or len(node.args) > len(function.parameters):
            return None  # keywords: let the ordinary path decide
        # A parameter the call left out takes its default, whose kind is what
        # the body will see. Treating that as unanswerable sent a call such as
        # tag(1) to a path that cannot render a string at all.
        supplied: list[ast.expr] = list(node.args)
        for default in function.defaults[len(node.args) :]:
            if default is None:
                return None  # A missing argument with no default: not our call.
            supplied.append(ast.Constant(value=default))
        # Stand-ins of the right kind, so nothing is emitted just to ask.
        stand_ins: dict[str, KernelValue] = {}
        strings: dict[str, IntExpression] = {}
        for parameter, argument in zip(function.parameters, supplied):
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
        self._kind_query.add(id(function))
        try:
            return self.expression_type(function.expression, stand_ins)
        finally:
            self._kind_query.discard(id(function))
            self.functions, self.values = previous_functions, previous_values
            self.string_bindings = previous_strings

    def statement_body_returns_string(
        self,
        function: NativeFunction,
        node: ast.Call,
        bindings: dict[str, KernelValue] | None,
        skip_parameters: int = 0,
    ) -> bool:
        """Whether every `return` in this body answers with a string.

        The call site has to know before the body is inlined, because a string
        is an address and a number is a number and the two are read out of the
        result slot differently. So the body is read rather than run: the
        parameters get stand-ins of the right kind, the locals are typed by a
        walk over the assignments in source order, and each return expression
        is asked what it is under those.

        Deliberately all-or-nothing. A body that answers a string on one path
        and a number on another has no single kind for the slot, and saying so
        here is what lets the return statement report it precisely.
        """

        if id(function) in self._kind_query:
            # Asking this body's kind is what led here, so the program is
            # recursive. Answering "not a string" sends the caller down the
            # ordinary path, which refuses recursion with a located message
            # rather than running the compiler out of Python stack.
            return False
        returns = self.function_returns(function.body)
        if not returns or any(item.value is None for item in returns):
            return False
        parameters = function.parameters[skip_parameters:]
        if node.keywords or len(node.args) > len(parameters):
            return False
        supplied: list[ast.expr] = list(node.args)
        for default in function.defaults[skip_parameters + len(node.args) :]:
            if default is None:
                return False
            supplied.append(ast.Constant(value=default))
        stand_ins: dict[str, KernelValue] = {}
        strings: dict[str, IntExpression] = {}
        # A method's `self` is an object; it holds an address like any other
        # integer here, and only the attribute path cares what it points at.
        for parameter in function.parameters[:skip_parameters]:
            stand_ins[parameter] = IntConstant(0)
        for parameter, argument in zip(parameters, supplied):
            try:
                kind = self.expression_type(argument, bindings)
            except NativeCompileError:
                return False
            self.note_stand_in(stand_ins, strings, parameter, kind)
        previous_functions, previous_values = self.functions, self.values
        previous_strings = self.string_bindings
        self.functions = function.functions
        self.values = {
            name: value
            for name, value in function.values.items()
            if name not in function.parameters
        }
        self.string_bindings = {**previous_strings, **strings}
        self._kind_query.add(id(function))
        try:
            for statement in function.body:
                for inner in ast.walk(statement):
                    if not isinstance(inner, ast.Assign) or len(inner.targets) != 1:
                        continue
                    target = inner.targets[0]
                    if not isinstance(target, ast.Name):
                        continue
                    try:
                        kind = self.expression_type(inner.value, stand_ins)
                    except NativeCompileError:
                        continue
                    self.note_stand_in(stand_ins, strings, target.id, kind)
                    self.string_bindings = {**previous_strings, **strings}
            kinds = set()
            for item in returns:
                assert item.value is not None
                try:
                    kinds.add(self.expression_type(item.value, stand_ins))
                except NativeCompileError:
                    return False
            return kinds == {"str"}
        finally:
            self._kind_query.discard(id(function))
            self.functions, self.values = previous_functions, previous_values
            self.string_bindings = previous_strings

    @staticmethod
    def note_stand_in(
        stand_ins: dict[str, KernelValue],
        strings: dict[str, IntExpression],
        name: str,
        kind: str,
    ) -> None:
        """Record a value of ``kind`` under ``name``, without emitting one."""

        if kind == "float":
            stand_ins[name] = FloatConstant(0.0)
            strings.pop(name, None)
            return
        stand_ins[name] = IntConstant(0)
        if kind == "str":
            # A string's stand-in is a pointer like any other integer, so the
            # kind has to be recorded beside it.
            strings[name] = IntConstant(0)
        else:
            strings.pop(name, None)

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
        if self.value_bindings:
            # The call's own substitutions win: a name bound here shadows the
            # parameter of a function further out that happens to share it.
            bindings = {**self.value_bindings, **bindings}
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return "str"
            return "float" if isinstance(node.value, float) else "int"
        shape = self.dict_get_shape(node)
        if shape is not None:
            return shape[2]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "chr"
            and node.func.id not in self.functions
        ):
            return "str"
        popped = self.list_method_shape(node, "pop")
        if popped is not None:
            return popped
        if any(
            self.list_method_shape(node, attribute) is not None
            for attribute in ("index", "count")
        ):
            return "int"
        if isinstance(node, ast.Name):
            # Checked before `bindings`, because several callers ask for a
            # type without threading the bindings through - string_pointer
            # among them, which is exactly where a string parameter is needed.
            if node.id in self.string_bindings:
                return "str"
            if node.id in bindings:
                return (
                    "float"
                    if is_float_expression(bindings[node.id])
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
        if (
            isinstance(node, ast.Subscript)
            and not isinstance(node.slice, ast.Slice)
            and self.expression_type(node.value, bindings) == "str"
        ):
            # One code point of a string is a string, as it is in Python; there
            # is no character type for it to be instead. Asked of the value
            # rather than of value_types, because a string that folded at build
            # time is not recorded there.
            return "str"
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            sliced = self.expression_type(node.value, bindings)
            if self.tuple_kinds(sliced) is not None:
                # A slice of a tuple is a shorter tuple, and answering with the
                # whole one would tell len() the wrong number.
                raise NativeCompileError(
                    self.path,
                    node,
                    "native tuple slicing is not supported: read one element "
                    "at a time by index, or use a runtime list, which slices",
                )
            return sliced
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and self.dict_kinds_of(node.value.id)
        ):
            return self.dict_kinds_of(node.value.id)[1]
        element_kind = self.tuple_subscript_kind(node, bindings)
        if element_kind is not None:
            return self.element_value_type(element_kind)
        if isinstance(node, ast.Subscript):
            # Any list expression, not just a name: this is what lets
            # `xs[0][1]` know what it has at whatever depth it sits.
            try:
                container = self.expression_type(node.value, bindings)
            except NativeCompileError:
                container = None
            element = self.list_kind(container)
            if element is not None:
                return self.element_value_type(element)
        if isinstance(node, ast.Attribute) and self.resolve_object_class(node.value):
            return self.attribute_kind(node)
        method = self.string_method_kind(node, bindings)
        if method is not None:
            if method == "str" or self.list_kind(method) is not None:
                return method
            return "int"
        kind = self.expression_function_kind(node, bindings)
        if kind is not None:
            return kind
        if isinstance(node, ast.JoinedStr):
            return "str"
        if isinstance(node, ast.Dict):
            return self.dict_literal_tag(node, bindings)
        if isinstance(node, ast.Set):
            return self.set_literal_tag(node, bindings)
        if self.empty_set_call(node):
            # `set()` alone cannot say what it will hold; int unless an
            # annotation says otherwise, the same rule `{}` follows.
            return self.set_tag("int")
        if isinstance(node, ast.List):
            return self.list_literal_tag(node, bindings)
        if isinstance(node, ast.Tuple):
            return self.tuple_tag(self.tuple_literal_kinds(node, bindings))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, (ast.USub, ast.UAdd)):
                return self.expression_type(node.operand, bindings)
            return "int"
        if isinstance(node, ast.BinOp):
            operands = self.set_binop_operands(node)
            if operands is not None:
                return self.set_tag(operands[2])
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
            if body == "str" and orelse == "str":
                # Both arms have to agree. One slot holds either an address or
                # a number, and which one it is cannot be settled at run time.
                return "str"
            return "float" if "float" in (body, orelse) else "int"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and self.method_call_kind(node) == "str"
        ):
            return "str"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in self.classes:
                return "object"
            if node.func.id in self.extern_functions:
                key = self.extern_functions[node.func.id]
                return "float" if _CABI_RESULTS[key] == "float" else "int"
            if node.func.id == "float" and node.func.id not in self.functions:
                return "float"
            if (
                node.func.id == "sorted"
                and node.func.id not in self.functions
                and len(node.args) == 1
            ):
                # A sorted copy holds what the source held.
                source_kind = self.expression_type(node.args[0], bindings)
                element = self.list_kind(source_kind)
                if element is not None:
                    return self.list_tag(element)
                if self.set_kind(source_kind) is not None:
                    return self.list_tag(self.set_kind(source_kind))
            if node.func.id == "str" and node.func.id not in self.functions:
                return "str"
            if (
                node.func.id in {"int", "len", "abs"} | self._AGGREGATE_CALLS
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
        if self.eager_depth:
            raise NativeCompileError(
                self.path,
                node,
                "a float divisor that is not a compile-time constant cannot "
                "appear in a conditional expression or a short-circuited "
                "Boolean operand, because its zero check would run even when "
                "Python would not evaluate that branch; use an if statement "
                "instead",
            )
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

    def pin_extern_arguments(self, arguments: tuple[object, ...]) -> tuple[object, ...]:
        """Store every extern-calling argument in a slot, so each runs once.

        An argument that is spliced in at more than one use of its parameter
        would otherwise perform its external call once per use, which is a
        different sequence of calls than CPython makes even when the callee is
        pure enough that the arithmetic still comes out right.
        """

        pinned: list[object] = []
        for argument in arguments:
            if not _ir_contains_extern_call(argument):
                pinned.append(argument)
                continue
            slot = self.new_temp()
            if is_float_expression(argument):
                self.operations.append(FloatStore(slot, argument))
                pinned.append(FloatLoad(slot))
            else:
                self.operations.append(Store(slot, argument))
                pinned.append(IntLoad(slot))
        return tuple(pinned)

    def float_expression(
        self,
        node: ast.expr,
        bindings: dict[str, KernelValue] | None = None,
        call_stack: tuple[int, ...] = (),
    ) -> FloatExpression:
        """Lower ``node`` to an IEEE-754 double, widening integer operands."""

        bindings = bindings or {}
        if self.value_bindings:
            # The call's own substitutions win: a name bound here shadows the
            # parameter of a function further out that happens to share it.
            bindings = {**self.value_bindings, **bindings}
        if isinstance(node, ast.Name) and node.id not in bindings:
            # A float has no kind registry to hang this off, so it is asked
            # here. Not for a name the inliner has substituted: that one is a
            # parameter standing in for an argument, and it is bound.
            self.refuse_unbound(node.id, node)
        if self.expression_type(node, bindings) != "float":
            return IntToFloat(self.integer(node, bindings, call_stack))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.extern_functions
        ):
            # Ahead of the constant fold: an extern call has side effects and a
            # result the compiler cannot know, so it is never a constant.
            return self.extern_call(node, bindings, call_stack)
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
        if self.dict_get_shape(node) is not None:
            return BitsFloat(self.emit_dict_get(node, bindings))
        if self.list_method_shape(node, "pop") == "float":
            assert isinstance(node, ast.Call)
            return BitsFloat(self.emit_list_pop(node))
        if (
            isinstance(node, ast.Subscript)
            and not isinstance(node.slice, ast.Slice)
            and self.list_kind(self.expression_type(node.value, bindings)) == "float"
        ):
            return BitsFloat(HeapLoad(self.list_element_address(node), 8))
        if self.tuple_subscript_kind(node, bindings) == "float":
            assert isinstance(node, ast.Subscript)
            return BitsFloat(HeapLoad(self.tuple_element_address(node), 8))
        if (
            isinstance(node, ast.Attribute)
            and self.resolve_object_class(node.value)
            and self.attribute_kind(node) == "float"
        ):
            return BitsFloat(HeapLoad(self.attribute_address(node), 8))
        if isinstance(node, ast.Name) and node.id in bindings:
            bound = bindings[node.id]
            if is_float_expression(bound):
                return bound
        if self.expression_function_kind(node, bindings) == "float":
            assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            function = self.functions[node.func.id]
            if id(function) in call_stack:
                # This branch substitutes the body at the call site, so a
                # function that calls itself would substitute forever. The
                # integer path reports this with a location; get there rather
                # than running out of Python stack.
                raise NativeCompileError(
                    self.path,
                    node,
                    f"recursive native function call to {node.func.id}() is "
                    "not supported",
                )
            arguments = self.bind_native_arguments(
                node.func.id, function, node, bindings, call_stack
            )
            if _repeats_extern_argument(function, arguments):
                # This branch splices the argument expression in at every use of
                # the parameter, which for an external call would run the callee
                # once per use. CPython runs it once. Pin it in a slot first.
                arguments = self.pin_extern_arguments(arguments)
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
        if isinstance(node, ast.IfExp):
            # Both arms land in one slot, so they have to agree on kind. A
            # branching function body is normalised into one of these, which is
            # where a function returning a float from one arm and an int from
            # the other shows up.
            taken_kind = self.expression_type(node.body, bindings)
            other_kind = self.expression_type(node.orelse, bindings)
            if taken_kind != other_kind:
                raise NativeCompileError(
                    self.path,
                    node,
                    f"the two arms of this have different kinds, {taken_kind} "
                    f"and {other_kind}; they share a slot, and widening the "
                    "integer arm to a double would print 1.0 where Python "
                    "prints 1, so write float() on the integer arm",
                )
            condition = self.truth_value(node.test, bindings)
            # A real branch, not the integer path's evaluate-both-and-select: a
            # float arm can carry a division whose zero check must not run on
            # the arm Python never evaluates.
            result_slot = self.new_temp()
            alternative_label = self.new_label("float_if_else")
            join_label = self.new_label("float_if_end")
            self.operations.append(JumpIfFalse(condition, alternative_label))
            self.operations.append(
                FloatStore(
                    result_slot,
                    self.float_expression(node.body, bindings, call_stack),
                )
            )
            self.operations.append(Jump(join_label))
            self.operations.append(Label(alternative_label))
            self.operations.append(
                FloatStore(
                    result_slot,
                    self.float_expression(node.orelse, bindings, call_stack),
                )
            )
            self.operations.append(Label(join_label))
            return FloatLoad(result_slot)
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

    def callback_address(
        self,
        call: ast.Call,
        target: ast.expr,
        encoding_node: ast.expr,
        local_name: str,
    ) -> FunctionAddress:
        """Lower a Python ``def`` into a callable ``Function`` and take its address.

        Every other Python function in this compiler is inlined, so it has no
        address to take. An Objective-C method implementation cannot be: the
        runtime holds the pointer and calls it later, from its own stack, with
        the receiver in x0 and the selector in x1. So the def is lowered a
        second, separate time into a real ``Function`` with a real frame, and
        the value handed to ``class_addMethod`` is that body's entry address.

        The body is lowered into a FRESH front end, because a callback's frame
        is not the entry point's. Everything the module's lowering keeps in a
        stack slot -- the heap arena's bump pointer above all -- is addressed
        off the entry point's frame pointer and is simply not there when the
        runtime calls in. Rather than let a callback read a slot that belongs to
        another frame, the separate lowering is checked afterwards for anything
        that would have needed one, and the build is refused if it finds any.
        """

        if not isinstance(target, ast.Name):
            raise NativeCompileError(
                self.path,
                target,
                f"extern call {local_name}() needs the NAME of a function "
                "defined in this module for its implementation argument; a "
                "method implementation is an address in this image, and only a "
                "def has one",
            )
        name = target.id
        function = self.functions.get(name)
        if function is None:
            raise NativeCompileError(
                self.path,
                target,
                f"{name!r} is not a function defined in this module, so it has "
                "no address to give the Objective-C runtime",
            )
        try:
            encoding = self.constant(encoding_node)
        except NativeCompileError:
            encoding = None
        if isinstance(encoding, (bytes, bytearray)):
            encoding = bytes(encoding).decode("utf-8", "replace")
        if not isinstance(encoding, str):
            raise NativeCompileError(
                self.path,
                encoding_node,
                f"extern call {local_name}() requires a compile-time string "
                "constant for the method type encoding; it decides how the "
                "runtime passes the arguments, so it cannot be discovered at "
                "run time",
            )
        try:
            result_kind, argument_codes = parse_method_encoding(encoding)
        except ValueError as error:
            raise NativeCompileError(self.path, encoding_node, str(error)) from error
        if len(function.parameters) != len(argument_codes):
            raise NativeCompileError(
                self.path,
                target,
                f"method type encoding {encoding!r} describes "
                f"{len(argument_codes)} argument(s) but {name}() takes "
                f"{len(function.parameters)}; an Objective-C method is called "
                "with the receiver and the selector before its own arguments, "
                f"so {name}() must begin with two parameters for those",
            )
        if result_kind == "void" and function.returns_value:
            raise NativeCompileError(
                self.path,
                target,
                f"method type encoding {encoding!r} says the method returns "
                f"nothing, but {name}() returns a value; the caller would never "
                "read it",
            )
        if result_kind != "void" and not function.returns_value:
            raise NativeCompileError(
                self.path,
                target,
                f"method type encoding {encoding!r} says the method returns a "
                f"value, but {name}() returns none; the caller would read "
                "whatever happened to be in the result register",
            )
        self.check_callback_names(target, name, function)
        key = (name, encoding)
        existing = self._callback_functions.get(key)
        if existing is not None:
            return FunctionAddress(existing.name)
        # A name no C identifier and no Python identifier can have, so a
        # callback body can never be confused with a ``Call`` target.
        ir_name = f"objc method {name} {encoding}"
        body = self.lower_callback(call, target, name, function, result_kind)
        self._callback_functions[key] = Function(
            ir_name, len(function.parameters), body[1], body[0]
        )
        return FunctionAddress(ir_name)

    def callback_free_names(
        self, function: NativeFunction, seen: set[int] | None = None
    ) -> set[str]:
        """Names ``function`` and everything it calls read without binding first.

        Deliberately over-approximate: a name bound by a lambda parameter or a
        comprehension target is reported as free too. Naming one name too many
        costs a refusal, and naming one too few costs a wrong answer.
        """

        seen = set() if seen is None else seen
        if id(function) in seen:
            return set()
        seen.add(id(function))
        bound = self.function_local_names(function.body, function.parameters)
        free: set[str] = set()
        for statement in function.body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in bound:
                        free.add(node.id)
                elif isinstance(node, ast.Global):
                    free.update(node.names)
        for callee in list(free):
            nested = function.functions.get(callee)
            if nested is not None:
                free |= self.callback_free_names(nested, seen)
        return free

    def check_callback_names(
        self, target: ast.expr, name: str, function: NativeFunction
    ) -> None:
        """Refuse a callback that depends on a module value that can change.

        A callback is lowered here, at the point it is registered, but it runs
        whenever the runtime decides to call it. Anything it reads from the
        module is therefore frozen at the value the module happened to hold at
        registration, while the same source under CPython would read whatever
        the value is by then. The two only agree when the name can never be
        rebound, which is what the binding count decides. A name the module
        keeps in a stack slot is worse still and is refused outright: the
        callback's frame is not the entry point's, so its copy is not even the
        same storage.
        """

        unstable: list[str] = []
        for free in sorted(self.callback_free_names(function)):
            if (
                free in function.functions
                or free in function.extern_functions
                or free in function.kernel_functions
                or free in function.kernel_modules
                or free in self.classes
            ):
                continue
            if free not in function.values and free not in self.runtime_names:
                # A builtin, or a name that is not the module's at all. Neither
                # carries a value that can go stale.
                continue
            if free in self.runtime_names or self._binding_counts.get(free, 0) != 1:
                unstable.append(free)
        if unstable:
            raise NativeCompileError(
                self.path,
                target,
                f"{name}() reads the module-level name(s) "
                f"{', '.join(repr(item) for item in unstable)}, which is not a "
                "value a method implementation can be given. The runtime calls "
                "it at a time this compiler cannot name, so the only module "
                "value it could carry is the one that was current when the "
                "class was built, while the same source under CPython would "
                "read whatever the name means by then. Only a name bound "
                "exactly once in the whole module is stable enough; pass "
                "anything else through the message",
            )

    def lower_callback(
        self,
        call: ast.Call,
        target: ast.expr,
        name: str,
        function: NativeFunction,
        result_kind: str,
    ) -> tuple[list[object], int]:
        """Lower ``function`` as a standalone frame; returns (operations, slots)."""

        provider = Frontend(
            function.path,
            self.source_roots,
            self.import_stack[:-1],
            self.experimental_kernels,
        )
        provider.classes = self.classes
        # Slots 0 .. n-1 of a Function's frame are where its prologue spills the
        # incoming argument registers, so those have to be claimed first and in
        # order. The names are unspellable so no body local can land on one.
        for index in range(len(function.parameters)):
            provider.slot(f"<imp-argument:{index}>")
        arguments = tuple(
            IntLoad(index) for index in range(len(function.parameters))
        )
        result = provider.inline_imperative_function(
            name, function, arguments, target, ()
        )
        if result is not None:
            if result_kind == "bool":
                # A BOOL is one byte and its caller may read only that byte, so
                # returning 2 from a callback would be true natively and could
                # be anything under a shim that converts through Python's bool.
                # Normalising here is the same thing an Objective-C compiler
                # emits for `return (BOOL)(x != 0)`, and it is what makes the
                # two runs agree for every value the body can produce.
                result = IntCompare("ne", result, IntConstant(0))
            provider.operations.append(Return(result))
        if provider._heap_bump_slot is not None:
            raise NativeCompileError(
                self.path,
                target,
                f"{name}() allocates, so it cannot be a method implementation. "
                "The heap arena's bump pointer lives in a stack slot of the "
                "entry point's frame, and the runtime calls a method on a frame "
                "of its own where that slot does not exist, so the allocation "
                "would write through whatever the address happened to hold. A "
                "str, list, tuple, dict, set, f-string, exception, or a print "
                "of a float all allocate",
            )
        if provider._callback_functions:
            raise NativeCompileError(
                self.path,
                target,
                f"{name}() registers a method implementation of its own, which "
                "py2bin does not collect from inside a callback; register every "
                "class from the module body",
            )
        leaked = sorted(
            slot
            for slot in provider.slots
            if not slot.startswith("<")
        )
        if leaked:
            # Every slot this lowering creates for its own bookkeeping, and
            # every local of an inlined body, is given a name in angle brackets
            # that no Python identifier can have. A bare name surviving here is
            # therefore a free variable of the callback -- a module-level
            # runtime variable, or a name declared `global` -- and it has been
            # given a slot in the CALLBACK's frame, which is not the frame the
            # module's copy of that variable lives in. Reading it would return
            # stack dirt and writing it would be invisible to the rest of the
            # program.
            raise NativeCompileError(
                self.path,
                target,
                f"{name}() uses the module-level runtime variable(s) "
                f"{', '.join(repr(item) for item in leaked)}, which a method "
                "implementation cannot reach: it runs on a frame of the "
                "runtime's making and the module's variables live in the entry "
                "point's frame. Pass what it needs through the message instead",
            )
        return provider.operations, len(provider.slots)

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
        expressions; ``bool`` is an integer expression normalised to 0 or 1;
        ``f64`` is a double.

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
        if max(_register_demand(signature)) > _CABI_MAX_ARGUMENTS:
            words, doubles = _register_demand(signature)
            raise NativeCompileError(
                self.path,
                node,
                f"extern call {local_name}() passes {words} integer and "
                f"{doubles} floating-point arguments, but the native backend "
                f"only implements {_CABI_MAX_ARGUMENTS} registers in each file "
                "and has no stack-argument path",
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
        arguments: list[IntExpression | FloatExpression] = []
        for position, (argument, kind) in enumerate(zip(node.args, signature)):
            if kind == "imp":
                # The encoding sits in the next argument, which the signature
                # table is asserted to declare as a "cstr". It has to be read
                # here rather than later because it decides what the runtime
                # puts in the callee's registers, and therefore whether the
                # callee can be compiled at all.
                arguments.append(
                    self.callback_address(
                        node, argument, node.args[position + 1], local_name
                    )
                )
                continue
            if kind == "f64":
                arguments.append(
                    self.float_expression(argument, bindings, call_stack)
                )
                continue
            if kind == "bool":
                # A C BOOL is one byte, so the callee's reading of an
                # out-of-range word is its own business: 256 is true if it
                # tests the register and false if it reads the low byte. The
                # comparison is the same normalisation an Objective-C compiler
                # emits for (BOOL)(x != 0), and the CPython shim performs it
                # too, so neither run can depend on that choice.
                arguments.append(
                    IntCompare(
                        "ne",
                        self.integer(argument, bindings, call_stack),
                        IntConstant(0),
                    )
                )
                continue
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
            # Keyed by the import name, not the C symbol: several bindings share
            # one symbol (the objc_msgSend arities) and each declares its own
            # result shape. Keying by symbol would silently hand back "i64".
            _CABI_RESULT_WIDTH.get(key, "i64"),
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
        self.raise_exception(
            "ZeroDivisionError",
            b"ZeroDivisionError: integer division or modulo by zero\n",
        )
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
        if self.value_bindings:
            # The call's own substitutions win: a name bound here shadows the
            # parameter of a function further out that happens to share it.
            bindings = {**self.value_bindings, **bindings}
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.extern_functions
        ):
            call = self.extern_call(node, bindings, call_stack)
            if call.result == "f64":
                raise NativeCompileError(
                    self.path,
                    node,
                    f"extern call {node.func.id}() returns a C double, so its "
                    "result is not an integer; wrap it in int() to truncate",
                )
            return call
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
            if is_float_expression(value):
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
                f"{node.id!r} may be unbound here because a loop reaching this "
                "point can be left without binding it - by running zero times, "
                "or by a break skipping the else body; CPython raises "
                "UnboundLocalError, and the native slot would hold an "
                "unrelated value",
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
            if self.tuple_kinds(kind) is not None:
                # Without this the compare ladder below would lower each side
                # to its slot and compare BLOCK ADDRESSES, so two equal tuples
                # at different addresses would answer False where CPython
                # answers True.
                raise NativeCompileError(
                    self.path,
                    node,
                    f"tuple variable {node.id!r} needs indexing or len() in an "
                    "integer context: a whole tuple cannot be compared, added, "
                    "or passed to a native function, because the slot holds "
                    "the address of its block and not the elements",
                )
            if kind == "str":
                raise NativeCompileError(
                    self.path,
                    node,
                    f"string variable {node.id!r} needs len() in an integer context",
                )
            if self.dict_kinds(kind) is not None or self.set_kind(kind) is not None:
                # Without this the slot lowers to the table's ADDRESS, so
                # `d + 1` answered with an arena offset and `if d:` was true
                # for an empty table. Both are wrong answers, not refusals.
                what = "dict" if self.dict_kinds(kind) is not None else "set"
                extra = "a lookup, " if what == "dict" else ""
                raise NativeCompileError(
                    self.path,
                    node,
                    f"{what} variable {node.id!r} needs len(), {extra}or `in` "
                    f"in an integer context: the slot holds the address of its "
                    f"table, so arithmetic on it, comparing it, or testing it "
                    f"for truth would use that address. Write len({node.id}) "
                    f"> 0 for the truth test",
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
            and node.func.id == "ord"
            and node.func.id not in self.functions
        ):
            return self.emit_ord(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "round"
            and node.func.id not in self.functions
        ):
            return self.emit_round(node, bindings, call_stack)
        if self.is_divmod_call(node):
            raise NativeCompileError(
                self.path,
                node,
                "native divmod() answers two values at once and is written "
                "`q, r = divmod(a, b)`; there is no tuple here for it to be "
                "held in otherwise",
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "bool"
            and node.func.id not in self.functions
        ):
            if len(node.args) != 1 or node.keywords:
                raise NativeCompileError(
                    self.path, node, "native bool() takes exactly one argument"
                )
            return self.truth_value(node.args[0], bindings)
        if self.dict_get_shape(node) is not None:
            return self.emit_dict_get(node, bindings)
        if self.list_method_shape(node, "pop") in {"int", "bool"}:
            assert isinstance(node, ast.Call)
            return self.emit_list_pop(node)
        if self.list_method_shape(node, "index") is not None:
            assert isinstance(node, ast.Call)
            return self.emit_list_index(node)
        if self.list_method_shape(node, "count") is not None:
            assert isinstance(node, ast.Call)
            return self.emit_list_count(node)
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
            and node.func.id in {"sorted", "reversed"}
            and node.func.id not in self.functions
        ):
            # Check the call before refusing it, so a bad keyword or an
            # argument that is not a list names itself rather than hiding
            # behind "this is not a number".
            if node.func.id == "sorted":
                self.sorted_call_shape(node)
            else:
                self.reversed_source(node)
            raise NativeCompileError(
                self.path,
                node,
                f"native {node.func.id}() produces a sequence, not a number; "
                "sorted(list) can be assigned to a name, indexed, or iterated, "
                "and reversed(list) is only a for-loop header",
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self._AGGREGATE_CALLS
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
            if isinstance(argument, ast.Name) and (
                self.dict_kinds_of(argument.id)
                or self.set_kind_of(argument.id) is not None
            ):
                # The live count is the second i64 of the table header, and a
                # set is the same table.
                return HeapLoad(
                    IntBinary("add", IntLoad(self.slots[argument.id]), IntConstant(8)),
                    8,
                )
            tuple_kinds = self.tuple_kinds(
                self.expression_type(argument, bindings)
            )
            if tuple_kinds is not None:
                # A tuple cannot grow, so how long it is was settled where it
                # was written and nothing has to be read from the block.
                return IntConstant(len(tuple_kinds))
            if self.list_kind(self.expression_type(argument, bindings)) is not None:
                # A slice, a comprehension, an inner list: the length is the
                # second word of whatever block it built or pointed at.
                return HeapLoad(
                    IntBinary(
                        "add",
                        self.materialize_int(self.list_pointer(argument)),
                        IntConstant(8),
                    ),
                    8,
                )
            raise NativeCompileError(
                self.path,
                node,
                "native len() supports runtime strings, runtime lists, runtime "
                "dicts, runtime sets and tuples",
            )
        if isinstance(node, ast.Subscript) and self.subscript_dict_kinds(node):
            _key_kind, value_kind = self.subscript_dict_kinds(node)
            if value_kind != "int":
                raise NativeCompileError(
                    self.path, node, "this dict has float values"
                )
            return HeapLoad(self.dict_lookup_value_address(node, bindings), 8)
        if (
            isinstance(node, ast.Subscript)
            and not isinstance(node.slice, ast.Slice)
            and self.tuple_kinds(self.expression_type(node.value, bindings))
            is not None
        ):
            element_kind = self.tuple_subscript_kind(node, bindings)
            if element_kind == "float":
                raise NativeCompileError(
                    self.path, node, "this tuple element is a float"
                )
            if element_kind == "str":
                raise NativeCompileError(
                    self.path,
                    node,
                    "this tuple element is a string, so it needs len() in an "
                    "integer context",
                )
            return HeapLoad(self.tuple_element_address(node), 8)
        if isinstance(node, ast.Subscript):
            element_kind = self.list_kind(self.expression_type(node.value, bindings))
            if element_kind == "float":
                raise NativeCompileError(
                    self.path, node, "this list holds floats"
                )
            if element_kind is not None and element_kind not in {"int", "bool"}:
                # The word is a block address, and an address is not the value
                # CPython would put here.
                raise NativeCompileError(
                    self.path,
                    node,
                    f"this list holds {self.kind_noun(element_kind)}, so this "
                    "element needs len() or indexing in an integer context",
                )
            return HeapLoad(self.list_element_address(node), 8)
        if isinstance(node, ast.Attribute) and self.resolve_object_class(node.value):
            if self.attribute_kind(node) == "float":
                raise NativeCompileError(
                    self.path, node, "this attribute holds a float"
                )
            return HeapLoad(self.attribute_address(node), 8)
        if self.string_method_kind(node, bindings) in {"int", "bool"}:
            assert isinstance(node, ast.Call)
            return self.string_method_integer(node)
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
                if isinstance(node.op, ast.Not):
                    # Through the truth question rather than the value, so
                    # that `not xs` and `not x` work for a container and a
                    # float as well as for an integer.
                    return IntCompare(
                        "eq",
                        self.truth_value(node.operand, bindings),
                        IntConstant(0),
                    )
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
            and node.ops
            and all(
                isinstance(
                    operator,
                    (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq),
                )
                for operator in node.ops
            )
            and any(
                isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                for operator in node.ops
            )
            and all(
                self.expression_type(item, bindings) == "str"
                for item in (node.left, *node.comparators)
            )
        ):
            # `"0" <= ch <= "9"` is one chain and two comparisons, so every
            # operand is lowered once into a slot and reused, the way the
            # numeric chain below does it.
            pointers = [
                self.materialize_int(self.string_pointer(item))
                for item in (node.left, *node.comparators)
            ]
            answer: IntExpression | None = None
            for position, operator in enumerate(node.ops):
                left_pointer, right_pointer = pointers[position], pointers[position + 1]
                if isinstance(operator, (ast.Eq, ast.NotEq)):
                    same = IntLoad(
                        self.emit_string_equal(left_pointer, right_pointer)
                    )
                    step: IntExpression = IntCompare(
                        "ne" if isinstance(operator, ast.Eq) else "eq",
                        same,
                        IntConstant(0),
                    )
                else:
                    order = IntLoad(
                        self.emit_string_order(left_pointer, right_pointer)
                    )
                    step = IntCompare(
                        {
                            ast.Lt: "lt",
                            ast.LtE: "le",
                            ast.Gt: "gt",
                            ast.GtE: "ge",
                        }[type(operator)],
                        order,
                        IntConstant(0),
                    )
                answer = step if answer is None else IntBinary("and", answer, step)
            assert answer is not None
            return answer
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
            if shape == "set":
                element_kind = self.set_kind_of(container.id)
                if self.expression_type(node.left, bindings) != element_kind:
                    raise NativeCompileError(
                        self.path,
                        node.left,
                        f"this set holds {element_kind} elements",
                    )
                _address, found_slot, _key, _state = self.dict_probe(
                    self.slot(container.id),
                    self.dict_key(node.left, element_kind),
                    element_kind,
                )
                return IntCompare(
                    "ne" if present else "eq",
                    IntLoad(found_slot),
                    IntConstant(0),
                )
            if shape != "dict":
                if shape == "str":
                    found = self.emit_substring_search(node.left, container)
                else:
                    found = self.emit_list_membership(
                        node.left, container, self.list_kind(shape)
                    )
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
                    if isinstance(operator_node, (ast.Is, ast.IsNot)):
                        # A runtime slot holds a number or a pointer, never a
                        # singleton, so there is no object whose identity this
                        # could ask about. Answering it would mean guessing.
                        raise NativeCompileError(
                            self.path,
                            node,
                            "native code has no runtime 'is': compare with == "
                            "or restructure so the identity test is constant",
                        )
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
            arms = (
                self.expression_type(node.body, bindings),
                self.expression_type(node.orelse, bindings),
            )
            if "str" in arms:
                # One slot holds either an address or a number. Which it is
                # cannot be settled at run time, so it has to be settled here.
                raise NativeCompileError(
                    self.path,
                    node,
                    "one arm of this conditional is a string and the other "
                    f"is {'an ' if (arms[0] if arms[1] == 'str' else arms[1]) == 'int' else 'a '}{arms[0] if arms[1] == 'str' else arms[1]}; the two "
                    "are held the same way and told apart only at build time, "
                    "so both arms have to be the same kind. A function whose "
                    "returns disagree reads as this, because a body of ifs "
                    "that all end in a return is folded into one conditional",
                )
            condition = self.truth_value(node.test, bindings)
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
                    returns_string=self.statement_body_returns_string(
                        function, node, bindings
                    ),
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
        if isinstance(node, ast.Lambda):
            raise NativeCompileError(self.path, node, self._LAMBDA_REFUSAL)
        if isinstance(node, ast.Starred):
            # Reached one argument at a time, so the call around it is gone by
            # here; only the callees that lower their arguments this way get
            # this far, and none of them has a parameter list to expand against.
            raise NativeCompileError(
                self.path,
                node,
                "* expansion is accepted only when calling a function, lambda "
                "or procedure defined in this file, where the expansion is "
                "spelled out against that callee's parameter count at build "
                "time; print() and the builtins take their arguments written "
                "out",
            )
        self.refuse_starred_call(node)
        self.refuse_lazy_comprehension(node)
        if isinstance(node, ast.Tuple):
            raise NativeCompileError(
                self.path,
                node,
                "a native tuple cannot be used where a signed 64-bit integer "
                "is required: it is a block of elements, not a number. Index "
                "it, take its len(), unpack it, or bind it to a name; a whole "
                "tuple cannot be compared, added or concatenated",
            )
        raise NativeCompileError(
            self.path,
            node,
            "expression is not in the signed 64-bit native integer subset",
        )

    def constant_field(self, piece: ast.FormattedValue) -> str:
        """Fold one f-string field at build time.

        The compiler runs on CPython, so the reference implementation of
        format() is right here and is used directly. The specifier is still
        checked against what the runtime renderers can do, because a program
        must not compile only because its values happened to be foldable.
        """

        value = self.constant(piece.value)
        spec_text = self.format_spec_text(piece)
        if piece.conversion == -1 and not spec_text:
            return str(value)
        kind = {bool: "int", int: "int", float: "float", str: "str"}.get(type(value))
        if kind is None:
            raise NativeCompileError(
                self.path,
                piece,
                f"a native f-string cannot render a {type(value).__name__} yet",
            )
        if piece.conversion != -1:
            if kind == "str" and piece.conversion != ord("s"):
                raise NativeCompileError(
                    self.path,
                    piece,
                    "!r and !a on a string add quotes and backslash escapes, "
                    "which a native f-string does not reproduce; only !s is "
                    "supported on a string",
                )
            if piece.conversion not in (ord("s"), ord("r"), ord("a")):
                raise NativeCompileError(
                    self.path, piece, "unsupported native f-string conversion"
                )
            value = str(value)
            kind = "str"
        try:
            parse_format_spec(spec_text, kind)
            return format(value, spec_text)
        except ValueError as error:
            raise NativeCompileError(self.path, piece, str(error)) from error

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
                except NativeCompileError:
                    # Our own rejections above are already located and worded;
                    # NativeCompileError is a ValueError, so without this the
                    # handler below would wrap one inside a second prefix.
                    raise
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
                    parts.append(self.constant_field(item))
                else:
                    raise NativeCompileError(self.path, item, "unsupported f-string component")
            return "".join(parts)
        raise NotConstant(self.path, node, "expression is not compile-time constant")


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
