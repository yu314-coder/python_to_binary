"""The CPython C-API tier, as Python: the exact program in capi_embedding.c.

py2bin's C frontend parses that .c file into a Python AST, and this file is the
result of printing that AST back out. Running it under `python3` performs the
SAME sequence of C-API calls (`py2bin.cabi` binds them through
`ctypes.pythonapi`), which is how the compiled binary is verified: build, run
natively, run this with CPython, require identical stdout and exit status.

    PYTHONPATH=src python3 -m py2bin compile-c examples/capi_embedding.c \
        --target darwin-arm64 -o capi.bin --clean
    ./capi.bin; echo "exit=$?"                        # -> 7
    PYTHONPATH=src python3 examples/capi_embedding.py; echo "exit=$?"  # -> 7

`compile` accepts this file directly too, lowering it to the same IR:

    PYTHONPATH=src python3 -m py2bin compile examples/capi_embedding.py \
        --target darwin-arm64 -o capi.bin --clean

One divergence, documented in py2bin/cabi.py: under ctypes a failing C-API call
raises immediately, while the compiled binary just receives NULL. The two runs
agree only while every call succeeds, as they do here.
"""

from py2bin.cabi import PyFile_WriteObject, PyFile_WriteString, PyImport_ImportModule, PyLong_AsLongLong, PyLong_FromLongLong, PyNumber_Multiply, PyObject_CallOneArg, PyObject_GetAttrString, PyObject_Str, PySys_GetObject, Py_DecRef, Py_Finalize, Py_Initialize

def square(value: int) -> int:
    return PyNumber_Multiply(value, value)

def show(value: int, stream: int) -> int:
    text = PyObject_Str(value)
    PyFile_WriteObject(text, stream, 1)
    PyFile_WriteString('\n', stream)
    Py_DecRef(text)
Py_Initialize()
stream = PySys_GetObject('stdout')
if stream == 0:
    raise SystemExit(9)
index = 1
total = 0
while index < 6:
    item = PyLong_FromLongLong(index)
    squared = square(item)
    show(squared, stream)
    total += PyLong_AsLongLong(squared)
    Py_DecRef(item)
    Py_DecRef(squared)
    index += 1
module = PyImport_ImportModule('math')
if module == 0:
    raise SystemExit(10)
function = PyObject_GetAttrString(module, 'isqrt')
item = PyLong_FromLongLong(total)
result = PyObject_CallOneArg(function, item)
PyFile_WriteString('isqrt(total) = ', stream)
show(result, stream)
total = PyLong_AsLongLong(result)
Py_DecRef(result)
Py_DecRef(item)
Py_DecRef(function)
Py_DecRef(module)
Py_Finalize()
raise SystemExit(total)
