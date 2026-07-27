/* The CPython C-API tier: canonical py2bin C that drives a real interpreter.
 *
 * This is the Nuitka-shaped path with Nuitka's essential dependency removed.
 * Nuitka hands generated C to gcc; py2bin compiles this file with its OWN
 * handwritten C parser and ARM64 encoder, and emits a Mach-O whose calls are
 * bound to the CPython shared library through real dyld binding.
 *
 * Note what is NOT here: no #include <Python.h>. py2bin never reads a system
 * header, so every entry point states its own ABI as an `extern` prototype and
 * every PyObject * is an OPAQUE 64-bit handle -- passed, returned, and compared
 * against NULL, never dereferenced. That restriction is what makes a
 * handwritten C compiler feasible at all.
 *
 * Reference counting is explicit and unmanaged: py2bin emits exactly the
 * Py_IncRef/Py_DecRef calls written below and checks nothing. Likewise, a
 * failing C-API call would return NULL and leave the error pending; this
 * program stays on the success path.
 *
 * Build, run, and diff against CPython (darwin-arm64 only):
 *
 *     PYTHONPATH=src python3 -m py2bin compile-c examples/capi_embedding.c \
 *         --target darwin-arm64 -o capi.bin --clean
 *     ./capi.bin; echo "exit=$?"                        # -> 7
 *     PYTHONPATH=src python3 examples/capi_embedding.py; echo "exit=$?"  # -> 7
 *
 * The emitted binary is not standalone: `otool -L` shows an LC_LOAD_DYLIB for
 * the build host's CPython at an absolute path, alongside libSystem.
 */

extern void Py_Initialize(void);
extern void Py_Finalize(void);
extern PyObject *PyLong_FromLongLong(long long value);
extern long long PyLong_AsLongLong(PyObject *value);
extern PyObject *PyNumber_Multiply(PyObject *left, PyObject *right);
extern PyObject *PyObject_Str(PyObject *value);
extern PyObject *PySys_GetObject(const char *name);
extern int PyFile_WriteObject(PyObject *value, PyObject *stream, int flags);
extern int PyFile_WriteString(const char *text, PyObject *stream);
extern PyObject *PyImport_ImportModule(const char *name);
extern PyObject *PyObject_GetAttrString(PyObject *value, const char *name);
extern PyObject *PyObject_CallOneArg(PyObject *callable, PyObject *argument);
extern void Py_DecRef(PyObject *value);

/* Helper returning a NEW reference. Handle-typed helpers are inlined by py2bin
 * exactly like integer ones, and an argument expression is evaluated once. */
PyObject *square(PyObject *value)
{
    return PyNumber_Multiply(value, value);
}

/* str(value) written to a Python file object, then the temporary released. */
void show(PyObject *value, PyObject *stream)
{
    PyObject *text;
    text = PyObject_Str(value);
    PyFile_WriteObject(text, stream, 1);
    PyFile_WriteString("\n", stream);
    Py_DecRef(text);
}

int main(void)
{
    PyObject *stream;
    PyObject *item;
    PyObject *squared;
    PyObject *module;
    PyObject *function;
    PyObject *result;
    long long index;
    long long total;

    Py_Initialize();
    stream = PySys_GetObject("stdout");
    if (stream == NULL) {
        return 9;
    }

    /* Real Python int objects: CPython does the arithmetic and the repr. */
    index = 1;
    total = 0;
    while (index < 6) {
        item = PyLong_FromLongLong(index);
        squared = square(item);
        show(squared, stream);
        total += PyLong_AsLongLong(squared);
        Py_DecRef(item);
        Py_DecRef(squared);
        index += 1;
    }

    /* Import a stdlib module and call one of its functions. */
    module = PyImport_ImportModule("math");
    if (module == NULL) {
        return 10;
    }
    function = PyObject_GetAttrString(module, "isqrt");
    item = PyLong_FromLongLong(total);
    result = PyObject_CallOneArg(function, item);
    PyFile_WriteString("isqrt(total) = ", stream);
    show(result, stream);
    total = PyLong_AsLongLong(result);

    Py_DecRef(result);
    Py_DecRef(item);
    Py_DecRef(function);
    Py_DecRef(module);
    Py_Finalize();
    return total;
}
