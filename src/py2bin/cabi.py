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

Only symbols with a simple, fixed integer/pointer ABI are exposed; see
``py2bin.native.frontend._CABI_SYMBOLS`` for the compiler-side whitelist, which
must stay in sync with the callables defined here.
"""

from __future__ import annotations

import ctypes
import ctypes.util

__all__ = [
    "getpid", "getppid", "getuid", "getgid", "abs", "labs", "strlen",
    "Py_Initialize", "Py_Finalize", "Py_IsInitialized", "PyRun_SimpleString",
    "PyLong_FromLongLong", "PyLong_AsLongLong", "PyUnicode_FromString",
    "PyNumber_Add", "PyNumber_Subtract", "PyNumber_Multiply",
    "PyNumber_TrueDivide", "PyObject_RichCompare", "PyObject_IsTrue",
    "PyObject_Str", "PyObject_Repr", "PyObject_Size", "PyObject_GetAttrString",
    "PyObject_CallNoArgs", "PyObject_CallOneArg", "PyImport_ImportModule",
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
    {"getpid", "getppid", "getuid", "getgid", "abs", "labs", "strlen"}
)
# Everything else exported here is a CPython runtime entry point, so the two
# sets cannot drift apart as the vetted ABI grows.
_CPYTHON_SYMBOLS = frozenset(name for name in __all__ if name not in _LIBC_SYMBOLS)


def symbol_library(symbol: str) -> str | None:
    """Return the library that provides ``symbol``, or None for the default."""

    if symbol in _CPYTHON_SYMBOLS:
        return _cpython_library()
    return LIBSYSTEM


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
