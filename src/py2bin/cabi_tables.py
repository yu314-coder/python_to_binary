"""Which native library provides which vetted symbol.

Split out of `cabi` so that compiling needs nothing but Python. `cabi` is the
half that *calls* these symbols, and calling them means `ctypes`, which in turn
pulls in `ctypes.util` and, through it, `subprocess`. The compiler never calls
them: it only has to know which library each one lives in, so that it can write
the load commands. That is a table, and a table needs no FFI.

The distinction matters wherever Python runs but a subprocess does not - on a
phone, that is everywhere. A build should ask for an interpreter and nothing
else, and with the tables here `py2bin compile-capi` runs from end to end with
`ctypes` never imported.

The vetted list itself lives here for the same reason, and `cabi.__all__` is
taken from it, so there is still exactly one place that says what the adapter
ABI is.
"""

from __future__ import annotations

#: Every symbol the adapter ABI vets. A binding in `cabi` that is not named
#: here is unreachable from compiled code, and a name here without a binding is
#: a symbol nothing can call.
VETTED = [
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
    "PyNumber_TrueDivide", "PyObject_RichCompare", "PyObject_IsTrue", "PyObject_IsInstance",
    "PyObject_Str", "PyObject_Repr", "PyObject_Format", "PyObject_Size",
    "PyObject_GetAttrString",
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
    # What `sys.exc_info()` reads inside an `except`, and what a new
    # exception raised there takes its `__context__` from.
    "PyErr_GetHandledException", "PyErr_SetHandledException",
    # The interned-name forms. `...String` builds a fresh str from the
    # char* on every call and hashes it; these take a str built once,
    # which is what makes an attribute in a loop cost a lookup rather
    # than an allocation.
    "PyObject_GetAttr", "PyObject_SetAttr", "PyUnicode_InternFromString",
    # `functools.partialmethod` was what made a compiled method bind, and
    # its `__get__` is written in Python: every `obj.method` built a
    # `functools.partial` through the interpreter. `instancemethod` is the
    # same idea in C - its `__get__` is `PyMethod_New` and nothing else -
    # and `PyObject_VectorcallMethod` skips even that, calling the function
    # found on the type without binding an object to hold the pair.
    "PyInstanceMethod_New", "PyObject_VectorcallMethod",
    # The verdict without the object. `PyObject_RichCompare` answers `True`
    # or `False` and every condition then asked `PyObject_IsTrue` what that
    # meant - two calls where the interpreter makes one.
    "PyObject_RichCompareBool",
    # One pass over the pieces instead of a chain of concatenations. An
    # f-string built with `+` copies the whole accumulated string at every
    # piece, which is quadratic in its length.
    "PyUnicode_Join",
    # A sequence index without an integer object to hold it. `xs[i]` had to
    # build a PyLong for the index and then go through the mapping protocol,
    # which handles slices; these two go to the sequence protocol with a
    # machine integer. `PySequence_Check` is what keeps a dict out of it -
    # `d[0]` is a mapping lookup and must stay one.
    "PySequence_Check", "PySequence_GetItem",
    # Fill a list made at its final length. Steals the reference, like its
    # tuple sibling, and never grows the storage - which is what a counted
    # comprehension wants, knowing its item count before the first item.
    "PyList_SetItem",
    # Two strings, concatenated the way an f-string joins - without asking
    # either operand's type for `__add__`, which is also what makes it
    # right for the join: `BUILD_STRING` never dispatches an override.
    "PyUnicode_Concat",
    "PyBytes_FromStringAndSize", "PyNumber_Negative", "PyNumber_Positive",
    "PyNumber_Invert", "Py_EnterRecursiveCall", "Py_LeaveRecursiveCall",
    "PyLong_FromString", "PyUnicode_DecodeUTF8", "PyImport_AddModule",
    "PyObject_Vectorcall",
    "PyList_New", "PyList_Append", "PySys_GetObject", "PySys_WriteStdout",
    "PyFile_WriteObject", "PyFile_WriteString", "Py_IncRef", "Py_DecRef",
    "PyErr_Occurred", "PyErr_Print", "PyErr_Clear",
]

LIBOBJC = "/usr/lib/libobjc.A.dylib"

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
    for name in VETTED
    if name not in _LIBC_SYMBOLS and name not in _OBJC_SYMBOLS
)


def symbol_library(symbol: str) -> str | None:
    """Return the library that provides ``symbol``, or None for the default."""

    if symbol in _CPYTHON_SYMBOLS:
        return _cpython_library()
    if symbol in _OBJC_SYMBOLS:
        return LIBOBJC
    return LIBSYSTEM




#: Where a vetted symbol lives on Windows. The kernel provides the process
#: services; `msvcrt.dll` provides the C library functions, and is the one
#: C runtime present on every Windows without shipping a redistributable; the
#: interpreter provides the rest.
WINDOWS_KERNEL = "KERNEL32.dll"
WINDOWS_C_RUNTIME = "msvcrt.dll"


def windows_symbol_library(symbol: str, python_dll: str) -> str:
    """Which DLL a vetted symbol is imported from.

    `python_dll` is the interpreter's own, `python313.dll` and so on - the name
    is version-specific because CPython's ABI is, and importing the wrong one
    would resolve names that mean something different.

    The Objective-C runtime has no Windows equivalent, so a program that
    reached it is refused rather than given an import that cannot resolve.
    """

    if symbol in _OBJC_SYMBOLS:
        raise ValueError(
            f"{symbol!r} is part of the Objective-C runtime, which exists on "
            "darwin only; there is no Windows import for it"
        )
    if symbol in _LIBC_SYMBOLS:
        return WINDOWS_C_RUNTIME
    return python_dll
