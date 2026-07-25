# The adapter-ABI extern boundary (slice 4)

py2bin's identity is absolute honesty: it never ships a binary that runs
incorrectly, and it refuses Python it cannot compile truthfully. That principle
governs how py2bin treats "libraries".

## What this is, and what it is not

A third-party native package (NumPy, PyTorch, Blender's `bpy`, ...) is
precompiled C/C++/CUDA machine code. There is **no Python source to translate**,
so py2bin does not, and will never, pretend to compile such a package from
source. Importing one is rejected with a clear message.

The one honest "library" path is the **adapter ABI**: declare an already-compiled
external native symbol and *call* it, resolving the address through the
platform's real dynamic linker. py2bin generates the call site and the linker
metadata; the callee's machine code is whatever the platform already ships.

This slice implements that path for a vetted subset of **libSystem** symbols on
**darwin-arm64**, verified by running the emitted binary natively.

## What is real today (darwin-arm64)

Source declares extern calls by importing from `py2bin.cabi`:

```python
from py2bin.cabi import strlen
raise SystemExit(strlen("hello, native world"))   # exit code 19
```

Compiling this for `darwin-arm64` produces a Mach-O that:

- has a **`__DATA,__got`** section with one 8-byte pointer slot per symbol;
- carries an **`LC_LOAD_DYLIB`** for `/usr/lib/libSystem.B.dylib`, a
  `LC_LOAD_DYLINKER` for `/usr/lib/dyld`, and a `LC_SYMTAB` / `LC_DYSYMTAB`
  describing the undefined imports and the indirect symbol table;
- carries **`LC_DYLD_INFO_ONLY`** with classic **bind opcodes** that instruct
  dyld to store the resolved address of `_strlen` into the GOT slot before the
  entry point runs (eager, non-lazy binding);
- calls through the slot with `adrp x16 / ldr x16,[x16,#off] / blr x16`,
  following the AArch64 calling convention (integer/pointer args in `x0..x7`,
  result in `x0`);
- is ad-hoc code-signed so Apple's loader accepts it.

`nm -mu` shows `_strlen (undefined) (from libSystem)` and `dyld_info -fixups`
shows the bind `__DATA __got ... bind libSystem/_strlen`. The binary runs
natively and its exit code equals the same source run under CPython, whose
`py2bin.cabi` shim calls the identical libc symbols through `ctypes`.

### Verified symbols

`py2bin.native.frontend._CABI_SYMBOLS` is the whitelist. Only symbols with a
simple fixed integer/pointer ABI are listed, so the compiler can never emit a
call with a mismatched signature:

| import      | C symbol   | signature            | verification                    |
|-------------|------------|----------------------|---------------------------------|
| `getpid`    | `getpid`   | `() -> int`          | native pid > 0 (nondeterministic value) |
| `getppid`   | `getppid`  | `() -> int`          | native ppid > 0                 |
| `getuid`    | `getuid`   | `() -> int`          | returns real uid                |
| `getgid`    | `getgid`   | `() -> int`          | returns real gid                |
| `abs`       | `abs`      | `(int) -> int`       | exact match vs CPython          |
| `labs`      | `labs`     | `(long) -> int`      | exact match vs CPython          |
| `strlen`    | `strlen`   | `(cstr) -> int`      | exact match vs CPython          |

`cstr` arguments must be **compile-time string constants**; py2bin materializes
them as NUL-terminated blobs in `__TEXT` and passes the pointer. A non-constant
`cstr` argument is rejected with a source location, because a runtime py2bin
string is a length-prefixed heap block, not a C string.

Tests: `tests/test_native_extern.py` (native execution gated on
`platform.system()=="Darwin" and platform.machine()=="arm64"`).

### The CPython C-API surface

The same mechanism binds the already-compiled **CPython interpreter**, which is
what makes the Nuitka-shaped tier possible: the application's own logic becomes
machine code while object semantics stay in `libpython`. The interpreter's
dylib is a second `LC_LOAD_DYLIB` with its own two-level-namespace ordinal
(`py2bin.cabi.symbol_library` routes each symbol to its library).

Argument kinds in the whitelist:

| kind   | meaning                                                                |
|--------|------------------------------------------------------------------------|
| `int`  | signed 64-bit integer expression, passed in an integer register         |
| `ptr`  | opaque pointer-sized handle (a `PyObject *`), never dereferenced by py2bin |
| `cstr` | compile-time string constant, materialized as a NUL-terminated blob     |
| `cfmt` | a variadic callee's fixed format argument, called with **zero** variadic arguments |

Vetted C-API symbols (all verified present in the interpreter dylib by
`tests/test_native_capi.py`):

| import                    | signature                    | result |
|---------------------------|------------------------------|--------|
| `Py_Initialize`           | `()`                         | void   |
| `Py_Finalize`             | `()`                         | void   |
| `Py_IsInitialized`        | `()`                         | int    |
| `PyRun_SimpleString`      | `(cstr)`                     | int    |
| `PyLong_FromLongLong`     | `(int)`                      | ptr    |
| `PyLong_AsLongLong`       | `(ptr)`                      | int    |
| `PyUnicode_FromString`    | `(cstr)`                     | ptr    |
| `PyNumber_Add`            | `(ptr, ptr)`                 | ptr    |
| `PyNumber_Subtract`       | `(ptr, ptr)`                 | ptr    |
| `PyNumber_Multiply`       | `(ptr, ptr)`                 | ptr    |
| `PyNumber_TrueDivide`     | `(ptr, ptr)`                 | ptr    |
| `PyObject_RichCompare`    | `(ptr, ptr, int)`            | ptr    |
| `PyObject_IsTrue`         | `(ptr)`                      | int    |
| `PyObject_Str`            | `(ptr)`                      | ptr    |
| `PyObject_Repr`           | `(ptr)`                      | ptr    |
| `PyObject_Size`           | `(ptr)`                      | int    |
| `PyObject_GetAttrString`  | `(ptr, cstr)`                | ptr    |
| `PyObject_CallNoArgs`     | `(ptr)`                      | ptr    |
| `PyObject_CallOneArg`     | `(ptr, ptr)`                 | ptr    |
| `PyImport_ImportModule`   | `(cstr)`                     | ptr    |
| `PyList_New`              | `(int)`                      | ptr    |
| `PyList_Append`           | `(ptr, ptr)`                 | int    |
| `PySys_GetObject`         | `(cstr)`                     | ptr    |
| `PySys_WriteStdout`       | `(cfmt)`                     | void   |
| `PyFile_WriteObject`      | `(ptr, ptr, int)`            | int    |
| `PyFile_WriteString`      | `(cstr, ptr)`                | int    |
| `Py_IncRef` / `Py_DecRef` | `(ptr)`                      | void   |
| `PyErr_Occurred`          | `()`                         | ptr    |
| `PyErr_Print` / `PyErr_Clear` | `()`                     | void   |

Deliberately **absent**: `PyObject_CallFunctionObjArgs` and every other
variadic entry point. Apple's arm64 ABI passes variadic arguments on the stack
rather than in `x0..x7`, and the backend has no stack-argument path, so calling
one would read arguments py2bin never wrote. `PySys_WriteStdout` is listed only
with the `cfmt` kind, which rejects a format string containing `%`.

A `void` result is a non-value: the register holds nothing defined after the
call, so using the result of `Py_DecRef` (or any other `void` symbol) is
rejected instead of reading garbage.

### The canonical-C handle dialect

`py2bin compile-c` accepts opaque handle types (`PyObject *`, `void *`,
`char *`, `FILE *`) as locals, parameters, and return types, plus `NULL`,
`const`, and `Py_ssize_t`. It tracks three expression kinds -- `i64`, `ptr` and
`cstr` -- and refuses to mix them, because they lower to the same machine word
and a confusion there is a silent miscompile:

- passing an integer where the callee will dereference a `PyObject *`;
- assigning a handle to a `long long` local, or an integer to a handle;
- pointer arithmetic or ordered (`<`, `>`) comparison of handles;
- an `extern` prototype whose arity or types disagree with the vetted ABI;
- using a `void` result as a value;
- `long long *` and any other dereferenceable pointer (unchanged).

Handles may be compared with `==`/`!=` against another handle or the null
pointer constant `0`/`NULL`, and used directly as a truth test (`if (obj)`),
which is exactly C's "not zero".

### Extern calls are not pure values

Two rules follow from an external call having side effects (reference counts,
output, the error indicator):

1. An extern call is **rejected inside a conditional expression or a
   short-circuited Boolean operand**. py2bin lowers `A ? B : C` and `x && y` by
   evaluating *both* arms and selecting, so a call in the untaken arm would
   still run. Use an `if` statement.
2. Expression inlining substitutes a parameter's value at every use. When an
   argument contains an extern call and the parameter is used more than once,
   the compiler **falls back to the imperative inliner**, which materializes
   each argument in a stack slot and therefore evaluates it exactly once. The
   analogous case for a local inside an inlined function body raises, which
   selects the same imperative path.

## Pure-Python helper libraries (part a)

Separately from extern calls, a **pure-Python** helper module still compiles
natively through the existing function-inlining path, now including the `float`
value type. `tests/test_native_extern.py::NativeLibraryHelperTests` builds a
package whose helper mixes a float computation and an integer loop, imports it
via a source root, compiles for darwin-arm64, and confirms the native exit code
matches CPython. The static whole-library audit
(`py2bin.native.audit_native_library`) reports, without executing, which
top-level functions lower to native IR and which are blocked.

## What is design-only (honest gaps)

The following are **not** implemented and are rejected rather than faked:

1. **Non-darwin targets.** linux (ELF/`DT_NEEDED` + PLT/GOT relocations),
   windows (PE import table + IAT), and the x86-64 backends do not yet emit
   extern call sites or import metadata. Any program that calls `py2bin.cabi`
   symbols for a non-`darwin-arm64` target is rejected in
   `compiler._emit_native_module` with an exact message. The GOT-call encoding
   would differ per backend (x86-64 `call [rip+got]`; PE IAT thunks), and the
   import metadata differs per format; each needs its own verified slice.

2. **Lazy binding / stubs.** This slice binds **eagerly** (all symbols resolved
   at load). Real toolchains also emit `__stubs` / `__stub_helper` /
   `__la_symbol_ptr` for lazy binding. Eager binding is simpler and fully
   correct; lazy binding is an optimization left undone.

3. **Richer ABIs.** Multiple dylibs are wired (each `LC_LOAD_DYLIB` gets its own
   two-level ordinal), but only integer/pointer arguments with an
   integer/pointer/void return, and at most **8** of them -- the AAPCS64
   register budget. A ninth argument raises rather than being truncated; there
   is no stack-argument path. Floating-point arguments/returns (`v0..v7`,
   `d0`), struct-by-value, variadic calls, `errno` inspection, and callbacks
   are not supported. Adding a symbol whose ABI is not exactly
   "integer/pointer args in, one word out" requires extending both the
   whitelist and the backend argument marshaling, with new native-run
   verification.

5. **The CPython error path.** The compiled binary and the `py2bin.cabi` shim
   agree only while no C-API call fails. `ctypes.pythonapi` turns a set error
   indicator into a raised Python exception after every call, so the
   interpreted run unwinds where the compiled run would continue with a NULL
   handle, and `PyErr_Occurred` can never report a pending error under the
   shim. Differential verification against CPython is therefore only valid for
   programs that stay on the success path.

4. **Chained fixups.** Modern linkers emit `LC_DYLD_CHAINED_FIXUPS`; py2bin uses
   the classic `LC_DYLD_INFO_ONLY` bind opcodes, which the current dyld still
   loads. If a future macOS drops classic-bind support, the writer would need a
   chained-fixups path.

Each gap is a place where py2bin refuses to overclaim. Partial-but-correct beats
broad-but-broken.
