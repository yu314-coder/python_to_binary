"""Adapter-ABI extern symbols: the only honest py2bin "library" path.

Importing a name from this module and calling it declares a call to a genuine
external native symbol. When py2bin compiles the program for ``darwin-arm64``,
each call is lowered to a real ``ExternCall`` that is bound to the symbol in
``/usr/lib/libSystem.B.dylib`` through actual dyld binding (LC_LOAD_DYLIB plus
classic bind opcodes writing a ``__got`` slot) -- no C/C++/CUDA source is ever
translated.

Under CPython, these shims call the *same* libc symbols through the standard
library's :mod:`ctypes`, so running the source with ``python3`` and running the
compiled native binary invoke identical machine code and produce identical
observable results. That is what makes the compiler's output verifiable against
CPython.

Only symbols with a simple, fixed integer/pointer/double ABI are exposed; see
``py2bin.native.frontend._CABI_SYMBOLS`` for the compiler-side whitelist, which
must stay in sync with the callables defined here.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import operator

__all__ = [
    "getpid", "getppid", "getuid", "getgid", "abs", "labs", "strlen",
    "exit",
    "pow", "fmod", "hypot", "atan2", "copysign", "ldexp",
    "objc_getClass", "sel_registerName", "objc_msgSend", "objc_msgSend2",
    "objc_msgSend_str", "objc_msgSend_id_id", "objc_msgSend_long",
    "objc_msgSend_bool_void", "objc_msgSend_rect", "objc_msgSend_rect_id",
    "objc_msgSend_rect_uint_uint_bool",
    "objc_allocateClassPair", "class_addMethod", "objc_registerClassPair",
    "Py_Initialize", "Py_Finalize", "Py_IsInitialized", "PyRun_SimpleString",
    "PyLong_FromLongLong", "PyLong_AsLongLong", "PyUnicode_FromString",
    "PyNumber_Add", "PyNumber_Subtract", "PyNumber_Multiply",
    "PyNumber_TrueDivide", "PyObject_RichCompare", "PyObject_IsTrue",
    "PyObject_Str", "PyObject_Repr", "PyObject_Size", "PyObject_GetAttrString",
    "PyObject_CallNoArgs", "PyObject_CallOneArg", "PyImport_ImportModule",
    # Calls of any arity: the arguments go in a tuple, which is what
    # PyObject_Call takes. PyTuple_SetItem steals its reference, which the
    # callers here account for.
    "PyObject_Call", "PyTuple_New", "PyTuple_SetItem",
    # Iteration. PyIter_Next answers NULL both when the sequence is exhausted
    # and when it fails, which PyErr_Occurred tells apart.
    "PyObject_GetIter", "PyIter_Next",
    "PyFloat_FromDouble", "PyFloat_AsDouble",
    "PyObject_GetItem", "PyObject_SetItem",
    "PyNumber_Remainder", "PyNumber_FloorDivide", "PyNumber_Power",
    "PyDict_New", "PyDict_SetItem", "PyTuple_Pack", "PySequence_Contains",
    "PyErr_ExceptionMatches", "PyErr_SetObject", "PySlice_New",
    "PyNumber_Or", "PyNumber_And", "PyNumber_Xor",
    "PyNumber_Lshift", "PyNumber_Rshift", "PyObject_DelItem",
    "PyErr_GetRaisedException", "PyCFunction_New", "PyTuple_GetItem",
    "PyObject_SetAttrString", "PyErr_SetRaisedException",
    "PyBytes_FromStringAndSize", "PyNumber_Negative", "PyNumber_Positive",
    "PyNumber_Invert", "Py_EnterRecursiveCall", "Py_LeaveRecursiveCall",
    "PyLong_FromString", "PyUnicode_DecodeUTF8", "PyImport_AddModule",
    "PyList_New", "PyList_Append", "PySys_GetObject", "PySys_WriteStdout",
    "PyFile_WriteObject", "PyFile_WriteString", "Py_IncRef", "Py_DecRef",
    "PyErr_Occurred", "PyErr_Print", "PyErr_Clear",
]


def _load_libc() -> ctypes.CDLL:
    name = ctypes.util.find_library("c") or "libSystem.B.dylib"
    return ctypes.CDLL(name, use_errno=False)


_libc = _load_libc()

_libc.getpid.restype = ctypes.c_int
_libc.getpid.argtypes = ()
_libc.getppid.restype = ctypes.c_int
_libc.getppid.argtypes = ()
_libc.getuid.restype = ctypes.c_uint
_libc.getuid.argtypes = ()
_libc.getgid.restype = ctypes.c_uint
_libc.getgid.argtypes = ()
_libc.abs.restype = ctypes.c_int
_libc.abs.argtypes = (ctypes.c_int,)
_libc.labs.restype = ctypes.c_long
_libc.labs.argtypes = (ctypes.c_long,)
_libc.strlen.restype = ctypes.c_size_t
_libc.strlen.argtypes = (ctypes.c_char_p,)

# The libm doubles. Declaring the prototypes is what makes ctypes marshal these
# through the SIMD&FP registers, which is the same ABI the compiled call uses,
# so the two runs execute the identical library code on identical bits.
for _name in ("pow", "fmod", "hypot", "atan2", "copysign"):
    _function = getattr(_libc, _name)
    _function.restype = ctypes.c_double
    _function.argtypes = (ctypes.c_double, ctypes.c_double)
del _name, _function
_libc.ldexp.restype = ctypes.c_double
_libc.ldexp.argtypes = (ctypes.c_double, ctypes.c_int)


def getpid() -> int:
    """Return the current process id (POSIX ``getpid``)."""

    return int(_libc.getpid())


def getppid() -> int:
    """Return the parent process id (POSIX ``getppid``)."""

    return int(_libc.getppid())


def getuid() -> int:
    """Return the real user id (POSIX ``getuid``)."""

    return int(_libc.getuid())


def getgid() -> int:
    """Return the real group id (POSIX ``getgid``)."""

    return int(_libc.getgid())


def abs(value: int) -> int:
    """C ``abs``: absolute value of a 32-bit ``int`` argument."""

    return int(_libc.abs(int(value)))


def labs(value: int) -> int:
    """C ``labs``: absolute value of a ``long`` argument."""

    return int(_libc.labs(int(value)))


def strlen(text: str | bytes) -> int:
    """C ``strlen``: number of bytes before the first NUL.

    A ``str`` is encoded as UTF-8, matching the NUL-terminated bytes py2bin
    materializes in the native image.
    """

    data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    return int(_libc.strlen(data))


def exit(status: int) -> int:  # noqa: A001 - the C name is the point
    """C ``exit``: end the process with ``status``.

    Under a compiled binary this ends the process. Called from Python here it
    would end the *interpreter*, which is never what a caller of this module
    means, so it raises SystemExit instead - the same thing, in the terms this
    side of the boundary uses.
    """

    raise SystemExit(int(status))


# These are C ``pow``/``fmod``/... and NOT Python's builtins: C pow(-8.0, 1/3)
# is NaN where Python's ** raises, and C fmod keeps the dividend's sign where
# Python's % keeps the divisor's. The name in this module always means the C
# function, which is the whole point of the module.


def pow(base: float, exponent: float) -> float:
    """C ``pow``: ``base`` raised to ``exponent``, both C doubles."""

    return float(_libc.pow(float(base), float(exponent)))


def fmod(numerator: float, denominator: float) -> float:
    """C ``fmod``: the remainder with the sign of ``numerator``."""

    return float(_libc.fmod(float(numerator), float(denominator)))


def hypot(x: float, y: float) -> float:
    """C ``hypot``: sqrt(x*x + y*y) without intermediate overflow."""

    return float(_libc.hypot(float(x), float(y)))


def atan2(y: float, x: float) -> float:
    """C ``atan2``: the angle of the point (x, y), in radians."""

    return float(_libc.atan2(float(y), float(x)))


def copysign(magnitude: float, sign: float) -> float:
    """C ``copysign``: ``magnitude`` with the sign bit of ``sign``."""

    return float(_libc.copysign(float(magnitude), float(sign)))


def ldexp(fraction: float, exponent: int) -> float:
    """C ``ldexp``: ``fraction * 2**exponent``.

    The exponent is a C ``int``, so only its low 32 bits reach the callee. A
    compiled call puts the whole word in x0 and the callee reads w0; masking to
    the same 32 bits here is what keeps the two runs identical rather than
    letting ctypes raise on a value the native side would have accepted.

    ``operator.index`` rather than ``int()`` because the compiler rejects a
    float in this position: silently truncating one here would make the
    interpreted run answer where the compiled run refuses to build.
    """

    truncated = operator.index(exponent) & 0xFFFFFFFF
    if truncated >= 0x80000000:
        truncated -= 0x100000000
    return float(_libc.ldexp(float(fraction), truncated))


# --- library routing ---------------------------------------------------------
#
# Every extern symbol names the native library that provides it. libSystem is
# the default; the CPython runtime is a second library so a generated binary can
# drive an embedded interpreter through real dyld binding. Adding a library here
# does NOT translate its source -- it links an already-compiled component.

LIBSYSTEM = "/usr/lib/libSystem.B.dylib"


def _cpython_library() -> str:
    """Absolute path of the running CPython's shared library."""

    import sysconfig
    from pathlib import Path

    framework = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX")
    name = sysconfig.get_config_var("PYTHONFRAMEWORK")
    version = sysconfig.get_config_var("py_version_short")
    if framework and name:
        candidate = Path(framework) / f"{name}.framework" / "Versions" / str(version) / str(name)
        if candidate.is_file():
            return str(candidate)
    library_directory = sysconfig.get_config_var("LIBDIR") or ""
    library_name = sysconfig.get_config_var("LDLIBRARY") or ""
    candidate = Path(library_directory) / library_name
    return str(candidate)


_LIBC_SYMBOLS = frozenset(
    {
        "getpid", "getppid", "getuid", "getgid", "abs", "labs", "strlen",
        "exit",
        "pow", "fmod", "hypot", "atan2", "copysign", "ldexp",
    }
)
# Everything else exported here is a CPython runtime entry point, so the two
# sets cannot drift apart as the vetted ABI grows.
_OBJC_SYMBOLS = frozenset(
    {
        "objc_getClass",
        "sel_registerName",
        "objc_msgSend",
        "objc_msgSend2",
        "objc_msgSend_str",
        "objc_msgSend_id_id",
        "objc_msgSend_long",
        "objc_msgSend_bool_void",
        "objc_msgSend_rect",
        "objc_msgSend_rect_id",
        "objc_msgSend_rect_uint_uint_bool",
        "objc_allocateClassPair",
        "class_addMethod",
        "objc_registerClassPair",
    }
)

# A class exists only once the framework that defines it is in the process:
# libobjc holds the runtime, not the classes. Loading Foundation is what makes
# objc_getClass("NSProcessInfo") answer rather than return nil, and it is what
# a linked Objective-C program gets from -framework Foundation.
# AppKit defines NSWindow and NSApplication, WebKit defines WKWebView and
# WKWebViewConfiguration, and neither answers objc_getClass until its framework
# is in the process, so the window path needs all three loaded.
OBJC_FRAMEWORKS = (
    "/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation",
    "/System/Library/Frameworks/AppKit.framework/Versions/C/AppKit",
    "/System/Library/Frameworks/WebKit.framework/Versions/A/WebKit",
)

# Whatever is left over is CPython's, so a new binding elsewhere has to be
# named in one of the two sets above or it will be looked for in libpython.
_CPYTHON_SYMBOLS = frozenset(
    name
    for name in __all__
    if name not in _LIBC_SYMBOLS and name not in _OBJC_SYMBOLS
)


def symbol_library(symbol: str) -> str | None:
    """Return the library that provides ``symbol``, or None for the default."""

    if symbol in _CPYTHON_SYMBOLS:
        return _cpython_library()
    if symbol in _OBJC_SYMBOLS:
        return LIBOBJC
    return LIBSYSTEM


# --- the Objective-C runtime ------------------------------------------------
#
# Cocoa is not a library with source to compile; it is compiled Objective-C
# that ships inside macOS. But the runtime that dispatches to it is a plain C
# API, and these three functions are the whole of it: look up a class by name,
# look up a selector by name, and send a message. Anything Cocoa can do is
# reachable through them, which is how a C program drives AppKit without a line
# of Objective-C.
#
# objc_msgSend is declared variadic in the header and is not one: it is a
# trampoline that reads its arguments from the ordinary registers, which is why
# an Objective-C compiler casts it to the callee's real prototype before every
# call. Each binding below is one such cast, so there is a binding per argument
# SHAPE rather than per arity: (id, SEL, NSInteger) and (id, SEL, id) happen to
# agree on arm64, but a cast is a claim about the callee's prototype and the
# two claims are not the same one.

LIBOBJC = "/usr/lib/libobjc.A.dylib"


def _bytes(text: bytes | str) -> bytes:
    return text.encode("utf-8") if isinstance(text, str) else bytes(text)


_objc = ctypes.CDLL(LIBOBJC)

# A compiled image lists these in its load commands, so they are in the process
# before its entry point runs. An interpreted run has to load them at the same
# point or the two disagree at the very first step: objc_getClass("NSWindow")
# answers nil under CPython and a real class in the binary, and every message
# after that goes to nil and silently returns zero.
_frameworks = tuple(ctypes.CDLL(path) for path in OBJC_FRAMEWORKS)


def objc_getClass(name: bytes | str) -> int:
    """The Class object registered under ``name``, or 0."""

    lookup = _objc.objc_getClass
    lookup.restype = ctypes.c_void_p
    lookup.argtypes = (ctypes.c_char_p,)
    return _returned(lookup(_bytes(name)))


def sel_registerName(name: bytes | str) -> int:
    """The selector registered under ``name``."""

    lookup = _objc.sel_registerName
    lookup.restype = ctypes.c_void_p
    lookup.argtypes = (ctypes.c_char_p,)
    return _returned(lookup(_bytes(name)))


def objc_msgSend(receiver: int, selector: int) -> int:
    """Send a no-argument message; returns the result as a word."""

    send = _objc.objc_msgSend
    send.restype = ctypes.c_void_p
    send.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    return _returned(send(receiver, selector))


def objc_msgSend_str(receiver: int, selector: int, text: bytes | str) -> int:
    """Send a message whose one argument is a C string."""

    send = _objc.objc_msgSend
    send.restype = ctypes.c_void_p
    send.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)
    return _returned(send(receiver, selector, _bytes(text)))


def objc_msgSend2(receiver: int, selector: int, argument: int) -> int:
    """Send a one-argument message; returns the result as a word."""

    send = _objc.objc_msgSend
    send.restype = ctypes.c_void_p
    send.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    return _returned(send(receiver, selector, argument))


# The shapes below carry values the compiled call site truncates to a machine
# word or normalises to a C ``BOOL``. Doing the same arithmetic here is what
# keeps the two runs identical: ctypes would otherwise raise on an argument the
# native side would have accepted, and ``operator.index`` refuses a float in a
# position where the compiler also refuses one.


def _word(value: int) -> int:
    """``value`` as the 64-bit word a compiled call would put in the register."""

    return operator.index(value) & 0xFFFFFFFFFFFFFFFF


def _returned(value: int | None) -> int:
    """A returned machine word, read the way the compiled code reads it.

    ctypes hands a c_void_p result back unsigned, but the compiled call leaves
    the callee's word in x0 and treats it as signed - so a method returning a
    negative NSInteger printed 18446744073709551615 interpreted and -1
    compiled. A pointer on arm64 never sets the top bit, so reading the word as
    signed costs nothing and is what makes the two runs agree.
    """

    word = value or 0
    return word - (1 << 64) if word >= (1 << 63) else word


def _flag(value: int) -> bool:
    """``value`` as the 0-or-1 ``BOOL`` a compiled call passes.

    A C ``BOOL`` is one byte, so handing the callee an out-of-range word is
    undefined: it may test the whole register, mask the low bit, or read only
    the low byte, and those disagree for 256. The compiler therefore lowers a
    ``bool`` argument as ``value != 0`` and this does the same, so no call site
    can ever pass a value whose meaning depends on the callee's codegen.
    """

    return operator.index(value) != 0


class _NSPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class _NSSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class _NSRect(ctypes.Structure):
    """Cocoa's ``NSRect``, spelled exactly as Foundation declares it.

    This has to be a ctypes Structure and not four separate ``c_double``
    arguments. AAPCS64 classifies a struct of four doubles as a homogeneous
    floating-point aggregate and hands it to the callee in four CONSECUTIVE
    SIMD&FP registers, which is the same placement four loose doubles get only
    because the rectangle is the first floating-point argument in each of these
    prototypes. Declaring the aggregate keeps that an ABI fact rather than a
    coincidence, and it is what the compiled call site reproduces.
    """

    _fields_ = [("origin", _NSPoint), ("size", _NSSize)]


def _rect(x: float, y: float, width: float, height: float) -> _NSRect:
    return _NSRect(_NSPoint(float(x), float(y)), _NSSize(float(width), float(height)))


def objc_msgSend_id_id(
    receiver: int, selector: int, first: int, second: int
) -> int:
    """Send a two-object message, as in ``loadHTMLString:baseURL:``."""

    send = _objc.objc_msgSend
    send.restype = ctypes.c_void_p
    send.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    )
    return _returned(send(receiver, selector, _word(first), _word(second)))


def objc_msgSend_long(receiver: int, selector: int, value: int) -> int:
    """Send a message whose one argument is an ``NSInteger``."""

    send = _objc.objc_msgSend
    send.restype = ctypes.c_void_p
    send.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
    word = _word(value)
    if word >= 0x8000000000000000:
        word -= 0x10000000000000000
    return _returned(send(receiver, selector, word))


def objc_msgSend_bool_void(receiver: int, selector: int, flag: int) -> None:
    """Send a ``BOOL`` setter that returns nothing.

    The callee leaves the result register undefined, so this returns None and
    the compiler refuses to use the value of such a call.
    """

    send = _objc.objc_msgSend
    send.restype = None
    send.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool)
    send(receiver, selector, _flag(flag))


def objc_msgSend_rect(
    receiver: int, selector: int, x: float, y: float, width: float, height: float
) -> int:
    """Send a message whose one argument is an ``NSRect``."""

    send = _objc.objc_msgSend
    send.restype = ctypes.c_void_p
    send.argtypes = (ctypes.c_void_p, ctypes.c_void_p, _NSRect)
    return _returned(send(receiver, selector, _rect(x, y, width, height)))


def objc_msgSend_rect_id(
    receiver: int,
    selector: int,
    x: float,
    y: float,
    width: float,
    height: float,
    argument: int,
) -> int:
    """Send ``initWithFrame:configuration:`` and anything shaped like it."""

    send = _objc.objc_msgSend
    send.restype = ctypes.c_void_p
    send.argtypes = (ctypes.c_void_p, ctypes.c_void_p, _NSRect, ctypes.c_void_p)
    return _returned(send(receiver, selector, _rect(x, y, width, height), _word(argument)))


def objc_msgSend_rect_uint_uint_bool(
    receiver: int,
    selector: int,
    x: float,
    y: float,
    width: float,
    height: float,
    style_mask: int,
    backing: int,
    defer: int,
) -> int:
    """Send ``initWithContentRect:styleMask:backing:defer:``.

    Nine arguments, and every one of them fits in a register because AAPCS64
    counts the two register files apart: the rectangle takes d0-d3 and the
    remaining five words take x0-x4.
    """

    send = _objc.objc_msgSend
    send.restype = ctypes.c_void_p
    send.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        _NSRect,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_bool,
    )
    return send(
        receiver,
        selector,
        _rect(x, y, width, height),
        _word(style_mask),
        _word(backing),
        _flag(defer),
    ) or 0


# --- building a class at run time -------------------------------------------
#
# Cocoa is not driven by sending messages alone. An application delegate, a
# window delegate and a navigation delegate are objects the framework calls
# back into, so a program that cannot hand the runtime a method implementation
# cannot be told that its own window closed. These three functions are that
# whole API: allocate a class pair, give it methods, publish it.
#
# Compiled, the implementation is the entry address of a real function in the
# image and the runtime branches straight to it. Here it has to be a C-callable
# trampoline around the Python function, and the two must agree about what
# arrives in which register -- which is what the method type encoding says.


def _signed(value: int) -> int:
    """``value`` as the SIGNED 64-bit integer a register holds it as.

    A compiled callback reads its arguments out of stack slots that hold raw
    machine words interpreted as signed 64-bit integers, and returns one the
    same way. Passing the Python function a ctypes pointer object instead would
    hand it ``None`` for a nil receiver where the compiled body sees 0.
    """

    word = operator.index(value) & 0xFFFFFFFFFFFFFFFF
    return word - 0x10000000000000000 if word >= 0x8000000000000000 else word


# The C types each method type encoding code maps to. This table is the CPython
# half of ``py2bin.native.frontend.parse_method_encoding``; the two are checked
# against each other by the test suite, because a disagreement is precisely a
# case where the interpreted run and the compiled run put a value in different
# registers.
_IMP_RESULT_TYPES = {
    "v": None,
    "q": ctypes.c_int64,
    "@": ctypes.c_int64,
    # A BOOL is one byte and the caller may read only that byte, so the value
    # is normalised to 0 or 1. The compiled body does the same with an explicit
    # `!= 0`, so a callback that returns 2 is true in both runs rather than
    # true in one and 2 in the other.
    "B": ctypes.c_bool,
}
_IMP_ARGUMENT_TYPES = {"@": ctypes.c_int64, ":": ctypes.c_int64, "q": ctypes.c_int64}

# Every trampoline ever built, kept forever. ctypes frees the thunk when the
# CFUNCTYPE object is collected, and the runtime keeps the pointer for the life
# of the class: dropping it would leave Cocoa branching into freed memory,
# which is a crash where the compiled binary is fine.
_registered_implementations: list[object] = []


def _implementation(function, encoding: str):
    """A C-callable trampoline for ``function`` under the given encoding."""

    result, arguments = encoding[0], encoding[1:]
    if result not in _IMP_RESULT_TYPES or not set(arguments) <= set(_IMP_ARGUMENT_TYPES):
        raise ValueError(
            f"py2bin cannot build a method implementation for the type encoding "
            f"{encoding!r}; the compiler refuses it too"
        )
    prototype = ctypes.CFUNCTYPE(
        _IMP_RESULT_TYPES[result], *(_IMP_ARGUMENT_TYPES[code] for code in arguments)
    )

    def entry(*registers):
        value = function(*(_signed(register) for register in registers))
        if result == "v":
            return None
        if result == "B":
            return bool(value)
        # The compiled body leaves a 64-bit register, so an out-of-range result
        # wraps rather than raising, and this has to wrap identically.
        return _signed(value)

    thunk = prototype(entry)
    _registered_implementations.append(thunk)
    return thunk


def objc_allocateClassPair(superclass: int, name: bytes | str, extra: int) -> int:
    """A new, unregistered subclass of ``superclass``, or 0.

    Returns 0 when a class called ``name`` is already registered, which is the
    one failure a caller has to check: every message to the nil it leaves
    behind silently returns zero.
    """

    allocate = _objc.objc_allocateClassPair
    allocate.restype = ctypes.c_void_p
    allocate.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t)
    return allocate(superclass, _bytes(name), _word(extra)) or 0


def class_addMethod(
    cls: int, selector: int, implementation, types: bytes | str
) -> int:
    """Give ``cls`` a method; returns 1 on success and 0 if it already had one.

    ``implementation`` is a Python function here and the address of a compiled
    function in the binary. It is called with the receiver, the selector, and
    then the method's own arguments, so it needs two parameters before its own.

    Only an implementation on ``cls`` ITSELF blocks this; an inherited one does
    not, which is how a subclass overrides a method. Changing an implementation
    the class already has is ``class_replaceMethod``, not this. Registration
    order does not matter here -- both sides of
    :func:`objc_registerClassPair` behave the same -- but a class must be
    registered before anything can be messaged.
    """

    encoding = _bytes(types).decode("utf-8")
    add = _objc.class_addMethod
    add.restype = ctypes.c_bool
    add.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
    )
    thunk = _implementation(implementation, encoding)
    # int(), not the bool: the compiled call site widens the callee's one-byte
    # BOOL into a machine word, so it prints 1 where an unconverted True would
    # print True.
    return int(add(cls, selector, ctypes.cast(thunk, ctypes.c_void_p), _bytes(types)))


def objc_registerClassPair(cls: int) -> None:
    """Publish ``cls`` so it can be messaged. Returns nothing."""

    register = _objc.objc_registerClassPair
    register.restype = None
    register.argtypes = (ctypes.c_void_p,)
    register(cls)


# --- CPython runtime entry points -------------------------------------------
#
# Compiled, these bind to the CPython dylib through dyld. Under CPython itself
# the interpreter is already running, so the shims below reproduce the same
# observable behaviour without re-initialising it. That keeps a program's
# native exit status and stdout identical to `python3 program.py`.


def Py_Initialize() -> int:
    return 0


def Py_Finalize() -> int:
    return 0


def Py_IsInitialized() -> int:
    return 1


def PyRun_SimpleString(source: bytes | str) -> int:
    text = source.decode("utf-8") if isinstance(source, bytes) else source
    exec(compile(text.rstrip("\x00"), "<PyRun_SimpleString>", "exec"), {"__name__": "__main__"})
    return 0


# --- CPython C-API surface ----------------------------------------------------
#
# These shims call the SAME exported C functions the compiled binary binds to,
# through ``ctypes.pythonapi`` (which keeps the GIL held for the duration of the
# call, as the C-API requires). A ``PyObject *`` is modelled on both sides as an
# opaque pointer-sized integer, so a program written against this ABI performs
# the identical sequence of C-API calls whether it is interpreted or compiled,
# including reference-count ownership: a shim that wraps a function returning a
# NEW reference returns that reference's raw address and the caller still owns
# it, exactly as in C.
#
# ONE DOCUMENTED DIVERGENCE. ``ctypes.pythonapi`` inspects CPython's error
# indicator after every call and turns a set indicator into a raised Python
# exception. A compiled binary sees no such thing: the callee returns NULL and
# the error stays pending until the program checks it. So the interpreted and
# compiled runs agree only while no C-API call fails. On the failure path the
# interpreted run raises and unwinds, while the compiled run continues with a
# NULL handle; ``PyErr_Occurred`` correspondingly can never report a pending
# error under the shim, because ctypes has already consumed it. Programs that
# must behave identically in both modes have to stay on the success path.

_python = ctypes.pythonapi


def _bind(name: str, restype, *argtypes):
    """Return a callable that binds ``name`` in the CPython dylib on first use.

    Binding lazily keeps importing this module free of any dependence on which
    C-API symbols a particular interpreter build exports: a program only pays
    for -- and only fails on -- the entry points it actually calls.
    """

    bound: list = []

    def call(*arguments):
        if not bound:
            function = getattr(_python, name)
            function.restype = restype
            function.argtypes = argtypes
            bound.append(function)
        return bound[0](*arguments)

    call.__name__ = name
    return call


_HANDLE = ctypes.c_void_p
_SSIZE = ctypes.c_ssize_t

_PyLong_FromLongLong = _bind("PyLong_FromLongLong", _HANDLE, ctypes.c_longlong)
_PyLong_AsLongLong = _bind("PyLong_AsLongLong", ctypes.c_longlong, _HANDLE)
_PyUnicode_FromString = _bind("PyUnicode_FromString", _HANDLE, ctypes.c_char_p)
_PyNumber_Add = _bind("PyNumber_Add", _HANDLE, _HANDLE, _HANDLE)
_PyNumber_Subtract = _bind("PyNumber_Subtract", _HANDLE, _HANDLE, _HANDLE)
_PyNumber_Multiply = _bind("PyNumber_Multiply", _HANDLE, _HANDLE, _HANDLE)
_PyNumber_TrueDivide = _bind("PyNumber_TrueDivide", _HANDLE, _HANDLE, _HANDLE)
_PyObject_RichCompare = _bind(
    "PyObject_RichCompare", _HANDLE, _HANDLE, _HANDLE, ctypes.c_int
)
_PyObject_IsTrue = _bind("PyObject_IsTrue", ctypes.c_int, _HANDLE)
_PyObject_Str = _bind("PyObject_Str", _HANDLE, _HANDLE)
_PyObject_Repr = _bind("PyObject_Repr", _HANDLE, _HANDLE)
_PyObject_Size = _bind("PyObject_Size", _SSIZE, _HANDLE)
_PyObject_GetAttrString = _bind(
    "PyObject_GetAttrString", _HANDLE, _HANDLE, ctypes.c_char_p
)
_PyObject_CallNoArgs = _bind("PyObject_CallNoArgs", _HANDLE, _HANDLE)
_PyObject_CallOneArg = _bind("PyObject_CallOneArg", _HANDLE, _HANDLE, _HANDLE)
_PyImport_ImportModule = _bind("PyImport_ImportModule", _HANDLE, ctypes.c_char_p)
_PyObject_Call = _bind("PyObject_Call", _HANDLE, _HANDLE, _HANDLE, _HANDLE)
_PyTuple_New = _bind("PyTuple_New", _HANDLE, _SSIZE)
_PyTuple_SetItem = _bind("PyTuple_SetItem", ctypes.c_int, _HANDLE, _SSIZE, _HANDLE)
_PyObject_GetIter = _bind("PyObject_GetIter", _HANDLE, _HANDLE)
_PyIter_Next = _bind("PyIter_Next", _HANDLE, _HANDLE)
_PyFloat_FromDouble = _bind("PyFloat_FromDouble", _HANDLE, ctypes.c_double)
_PyFloat_AsDouble = _bind("PyFloat_AsDouble", ctypes.c_double, _HANDLE)
_PyObject_GetItem = _bind("PyObject_GetItem", _HANDLE, _HANDLE, _HANDLE)
_PyObject_SetItem = _bind("PyObject_SetItem", ctypes.c_int, _HANDLE, _HANDLE, _HANDLE)
_PyNumber_Remainder = _bind("PyNumber_Remainder", _HANDLE, _HANDLE, _HANDLE)
_PyNumber_FloorDivide = _bind("PyNumber_FloorDivide", _HANDLE, _HANDLE, _HANDLE)
_PyNumber_Power = _bind("PyNumber_Power", _HANDLE, _HANDLE, _HANDLE, _HANDLE)
_PyDict_New = _bind("PyDict_New", _HANDLE)
_PyDict_SetItem = _bind("PyDict_SetItem", ctypes.c_int, _HANDLE, _HANDLE, _HANDLE)
_PyTuple_Pack = _bind("PyTuple_Pack", _HANDLE, _SSIZE, _HANDLE, _HANDLE)
_PySequence_Contains = _bind("PySequence_Contains", ctypes.c_int, _HANDLE, _HANDLE)
_PyErr_ExceptionMatches = _bind("PyErr_ExceptionMatches", ctypes.c_int, _HANDLE)
_PyErr_SetObject = _bind("PyErr_SetObject", None, _HANDLE, _HANDLE)
_PySlice_New = _bind("PySlice_New", _HANDLE, _HANDLE, _HANDLE, _HANDLE)
_PyNumber_Or = _bind("PyNumber_Or", _HANDLE, _HANDLE, _HANDLE)
_PyNumber_And = _bind("PyNumber_And", _HANDLE, _HANDLE, _HANDLE)
_PyNumber_Xor = _bind("PyNumber_Xor", _HANDLE, _HANDLE, _HANDLE)
_PyNumber_Lshift = _bind("PyNumber_Lshift", _HANDLE, _HANDLE, _HANDLE)
_PyNumber_Rshift = _bind("PyNumber_Rshift", _HANDLE, _HANDLE, _HANDLE)
_PyObject_DelItem = _bind("PyObject_DelItem", ctypes.c_int, _HANDLE, _HANDLE)
_PyErr_GetRaisedException = _bind("PyErr_GetRaisedException", _HANDLE)
_PyCFunction_New = _bind("PyCFunction_New", _HANDLE, _HANDLE, _HANDLE)
_PyTuple_GetItem = _bind("PyTuple_GetItem", _HANDLE, _HANDLE, _SSIZE)
_PyObject_SetAttrString = _bind(
    "PyObject_SetAttrString", ctypes.c_int, _HANDLE, ctypes.c_char_p, _HANDLE
)
_PyErr_SetRaisedException = _bind("PyErr_SetRaisedException", None, _HANDLE)
_PyBytes_FromStringAndSize = _bind(
    "PyBytes_FromStringAndSize", _HANDLE, ctypes.c_char_p, _SSIZE
)
_PyNumber_Negative = _bind("PyNumber_Negative", _HANDLE, _HANDLE)
_PyNumber_Positive = _bind("PyNumber_Positive", _HANDLE, _HANDLE)
_PyNumber_Invert = _bind("PyNumber_Invert", _HANDLE, _HANDLE)
_Py_EnterRecursiveCall = _bind(
    "Py_EnterRecursiveCall", ctypes.c_int, ctypes.c_char_p
)
_Py_LeaveRecursiveCall = _bind("Py_LeaveRecursiveCall", None)
_PyLong_FromString = _bind(
    "PyLong_FromString", _HANDLE, ctypes.c_char_p, _HANDLE, ctypes.c_int
)
_PyUnicode_DecodeUTF8 = _bind(
    "PyUnicode_DecodeUTF8", _HANDLE, ctypes.c_char_p, _SSIZE, _HANDLE
)
_PyImport_AddModule = _bind("PyImport_AddModule", _HANDLE, ctypes.c_char_p)
_PyList_New = _bind("PyList_New", _HANDLE, _SSIZE)
_PyList_Append = _bind("PyList_Append", ctypes.c_int, _HANDLE, _HANDLE)
_PySys_GetObject = _bind("PySys_GetObject", _HANDLE, ctypes.c_char_p)
_PySys_WriteStdout = _bind("PySys_WriteStdout", None, ctypes.c_char_p)
_PyFile_WriteObject = _bind(
    "PyFile_WriteObject", ctypes.c_int, _HANDLE, _HANDLE, ctypes.c_int
)
_PyFile_WriteString = _bind(
    "PyFile_WriteString", ctypes.c_int, ctypes.c_char_p, _HANDLE
)
_Py_IncRef = _bind("Py_IncRef", None, _HANDLE)
_Py_DecRef = _bind("Py_DecRef", None, _HANDLE)
_PyErr_Occurred = _bind("PyErr_Occurred", _HANDLE)
_PyErr_Print = _bind("PyErr_Print", None)
_PyErr_Clear = _bind("PyErr_Clear", None)


def _cstring(text: bytes | str) -> bytes:
    """Encode a py2bin C-string constant exactly as the native image embeds it."""

    data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    return data.split(b"\0", 1)[0]


def _handle(result: int | None) -> int:
    """Normalise a ``c_void_p`` result: NULL is the integer 0, as in C."""

    return 0 if result is None else int(result)


def PyLong_FromLongLong(value: int) -> int:
    """New reference to a Python ``int`` built from a signed 64-bit value."""

    return _handle(_PyLong_FromLongLong(int(value)))


def PyLong_AsLongLong(handle: int) -> int:
    """Signed 64-bit value of a Python ``int`` handle."""

    return int(_PyLong_AsLongLong(handle))


def PyUnicode_FromString(text: bytes | str) -> int:
    """New reference to a ``str`` decoded from a UTF-8 C string."""

    return _handle(_PyUnicode_FromString(_cstring(text)))


def PyNumber_Add(left: int, right: int) -> int:
    return _handle(_PyNumber_Add(left, right))


def PyNumber_Subtract(left: int, right: int) -> int:
    return _handle(_PyNumber_Subtract(left, right))


def PyNumber_Multiply(left: int, right: int) -> int:
    return _handle(_PyNumber_Multiply(left, right))


def PyNumber_TrueDivide(left: int, right: int) -> int:
    return _handle(_PyNumber_TrueDivide(left, right))


def PyObject_RichCompare(left: int, right: int, operation: int) -> int:
    """New reference to the result of ``left <op> right`` (Py_LT..Py_GE)."""

    return _handle(_PyObject_RichCompare(left, right, int(operation)))


def PyObject_IsTrue(handle: int) -> int:
    return int(_PyObject_IsTrue(handle))


def PyObject_Str(handle: int) -> int:
    return _handle(_PyObject_Str(handle))


def PyObject_Repr(handle: int) -> int:
    return _handle(_PyObject_Repr(handle))


def PyObject_Size(handle: int) -> int:
    return int(_PyObject_Size(handle))


def PyObject_GetAttrString(handle: int, name: bytes | str) -> int:
    return _handle(_PyObject_GetAttrString(handle, _cstring(name)))


def PyObject_CallNoArgs(callable_handle: int) -> int:
    return _handle(_PyObject_CallNoArgs(callable_handle))


def PyObject_CallOneArg(callable_handle: int, argument: int) -> int:
    return _handle(_PyObject_CallOneArg(callable_handle, argument))


def PyImport_ImportModule(name: bytes | str) -> int:
    return _handle(_PyImport_ImportModule(_cstring(name)))


def PyObject_Call(callable_handle: int, arguments: int, keywords: int) -> int:
    return _handle(_PyObject_Call(callable_handle, arguments, keywords))


def PyTuple_New(length: int) -> int:
    return _handle(_PyTuple_New(length))


def PyFloat_FromDouble(value: float) -> int:
    return _handle(_PyFloat_FromDouble(float(value)))


def PyFloat_AsDouble(value: int) -> float:
    return float(_PyFloat_AsDouble(value))


def PyObject_GetItem(container: int, key: int) -> int:
    return _handle(_PyObject_GetItem(container, key))


def PyObject_SetItem(container: int, key: int, value: int) -> int:
    return _returned(_PyObject_SetItem(container, key, value))


def PyNumber_Remainder(left: int, right: int) -> int:
    return _handle(_PyNumber_Remainder(left, right))


def PyNumber_FloorDivide(left: int, right: int) -> int:
    return _handle(_PyNumber_FloorDivide(left, right))


def PyNumber_Power(base: int, exponent: int, modulus: int) -> int:
    return _handle(_PyNumber_Power(base, exponent, modulus))


def PyDict_New() -> int:
    return _handle(_PyDict_New())


def PyDict_SetItem(mapping: int, key: int, value: int) -> int:
    return _returned(_PyDict_SetItem(mapping, key, value))


def PyTuple_Pack(length: int, first: int, second: int) -> int:
    """A two-element tuple. The vetted arity is fixed, as it must be: this is
    a variadic C function and py2bin passes no variadic arguments."""

    return _handle(_PyTuple_Pack(length, first, second))


def PyErr_SetObject(exception: int, value: int) -> None:
    """Set the exception that is being raised. Nothing unwinds here; the
    caller is expected to return to its own error path immediately."""

    _PyErr_SetObject(exception, value)


def PySlice_New(start: int, stop: int, step: int) -> int:
    return _handle(_PySlice_New(start, stop, step))

def PyNumber_Or(left: int, right: int) -> int:
    return _handle(_PyNumber_Or(left, right))

def PyNumber_And(left: int, right: int) -> int:
    return _handle(_PyNumber_And(left, right))

def PyNumber_Xor(left: int, right: int) -> int:
    return _handle(_PyNumber_Xor(left, right))

def PyNumber_Lshift(left: int, right: int) -> int:
    return _handle(_PyNumber_Lshift(left, right))

def PyNumber_Rshift(left: int, right: int) -> int:
    return _handle(_PyNumber_Rshift(left, right))

def PyObject_DelItem(container: int, key: int) -> int:
    return _returned(_PyObject_DelItem(container, key))


def PyErr_GetRaisedException() -> int:
    """Take the exception being handled, which also clears it."""

    return _handle(_PyErr_GetRaisedException())


def PyCFunction_New(method_table: int, closure: int) -> int:
    """A callable Python object backed by a C function.

    ``closure`` becomes the ``self`` the C function receives, which is how a
    compiled nested function carries the values it captured.
    """

    return _handle(_PyCFunction_New(method_table, closure))


def PyNumber_Negative(value: int) -> int:
    """`-x`. Not `0 - x`: those differ for a float, because `0 - 0.0` is
    positive zero where `-0.0` is negative zero."""

    return _handle(_PyNumber_Negative(value))


def PyNumber_Positive(value: int) -> int:
    """`+x`."""

    return _handle(_PyNumber_Positive(value))


def PyImport_AddModule(name: bytes) -> int:
    """The module of this name, created and registered if it is not there.

    It *borrows* the reference - `sys.modules` owns it. Registering is the
    point: an `import` of that name afterwards finds this object rather than
    going to look for a file.
    """

    return _handle(_PyImport_AddModule(name))


def PyUnicode_DecodeUTF8(text: bytes, length: int, errors: int) -> int:
    """A str decoded from exactly ``length`` bytes of UTF-8.

    The length is what lets the text carry a NUL, which
    ``PyUnicode_FromString`` would read as the end of it.
    """

    return _handle(_PyUnicode_DecodeUTF8(text, length, errors))


def PyLong_FromString(text: bytes, end: int, base: int) -> int:
    """An integer read from its decimal text.

    Python's integers have no width, so a literal like 2**63 has no C type to
    arrive in. Its digits do.
    """

    return _handle(_PyLong_FromString(text, end, base))


def Py_EnterRecursiveCall(where: bytes) -> int:
    """Count one level deeper, or answer non-zero with RecursionError set.

    This is what keeps a runaway recursion a Python exception rather than a
    stack the operating system takes away.
    """

    return int(_Py_EnterRecursiveCall(where))


def Py_LeaveRecursiveCall() -> None:
    """Count back out again. Every entered level must leave exactly once."""

    _Py_LeaveRecursiveCall()


def PyNumber_Invert(value: int) -> int:
    """`~x`."""

    return _handle(_PyNumber_Invert(value))


def PyBytes_FromStringAndSize(text: bytes, length: int) -> int:
    """A bytes object of exactly ``length`` bytes, copied from ``text``."""

    return _handle(_PyBytes_FromStringAndSize(text, length))


def PyErr_SetRaisedException(exception: int) -> None:
    """Set an exception again, exactly as it was - traceback included.

    It *steals* the reference, which is what makes it the right partner for
    ``PyErr_GetRaisedException``: a `finally` takes the exception so that the
    clause can run Python at all, then gives the same object back.
    """

    _PyErr_SetRaisedException(exception)


def PyObject_SetAttrString(object_handle: int, name: bytes, value: int) -> int:
    """Set an attribute by name. Zero on success, -1 with an exception set."""

    return int(_PyObject_SetAttrString(object_handle, name, value))


def PyTuple_GetItem(tuple_handle: int, index: int) -> int:
    """The item at ``index``. This *borrows* the reference - the tuple still
    owns it, so a caller keeping it must increment first."""

    return _handle(_PyTuple_GetItem(tuple_handle, index))


def PyErr_ExceptionMatches(exception: int) -> int:
    """Whether the exception being handled is this class or a subclass."""

    return _returned(_PyErr_ExceptionMatches(exception))


def PySequence_Contains(container: int, value: int) -> int:
    """1, 0, or -1 on failure - which is why the caller must test for -1 and
    not merely for truth."""

    return _returned(_PySequence_Contains(container, value))


def PyObject_GetIter(object_handle: int) -> int:
    return _handle(_PyObject_GetIter(object_handle))


def PyIter_Next(iterator: int) -> int:
    """The next item, or 0 when the sequence is exhausted *or* it failed.

    The two are told apart by asking ``PyErr_Occurred`` afterwards, which is
    what the generated C does.
    """

    return _handle(_PyIter_Next(iterator))


def PyTuple_SetItem(tuple_handle: int, index: int, value: int) -> int:
    """Put ``value`` in the tuple. This *steals* the reference to it."""

    return _returned(_PyTuple_SetItem(tuple_handle, index, value))


def PyList_New(length: int) -> int:
    return _handle(_PyList_New(int(length)))


def PyList_Append(list_handle: int, item: int) -> int:
    return int(_PyList_Append(list_handle, item))


def PySys_GetObject(name: bytes | str) -> int:
    """Borrowed reference to ``sys.<name>`` (0 when the attribute is absent)."""

    return _handle(_PySys_GetObject(_cstring(name)))


def PySys_WriteStdout(text: bytes | str) -> None:
    """Write a literal string to ``sys.stdout``.

    ``PySys_WriteStdout`` is variadic in C. py2bin only ever calls it with the
    fixed format argument and no variadic arguments, and the compiler rejects a
    format string containing ``%``, so this shim sees a plain literal too.
    """

    data = _cstring(text)
    if b"%" in data:
        raise ValueError("py2bin calls PySys_WriteStdout without conversions")
    _PySys_WriteStdout(data)


def PyFile_WriteObject(handle: int, file_handle: int, flags: int) -> int:
    """Write an object to a file object (flags=1 is ``Py_PRINT_RAW``)."""

    return int(_PyFile_WriteObject(handle, file_handle, int(flags)))


def PyFile_WriteString(text: bytes | str, file_handle: int) -> int:
    return int(_PyFile_WriteString(_cstring(text), file_handle))


def Py_IncRef(handle: int) -> None:
    _Py_IncRef(handle)


def Py_DecRef(handle: int) -> None:
    _Py_DecRef(handle)


def PyErr_Occurred() -> int:
    """Handle of the pending exception type, or 0.

    See the divergence note above: under ctypes a failing C-API call has
    already been turned into a raised Python exception, so this shim reports 0
    in every situation the interpreted program can actually reach.
    """

    return _handle(_PyErr_Occurred())


def PyErr_Print() -> None:
    _PyErr_Print()


def PyErr_Clear() -> None:
    _PyErr_Clear()
