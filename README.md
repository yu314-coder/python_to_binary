# python-to-binary

`py2bin` is a self-contained compiler and application builder implemented with
the Python standard library. The project itself has no runtime or build
dependencies: no Cython, Nuitka, mypyc, Rust, C, C++, PyInstaller, PPCI, native
bootloader, assembler, linker, or SDK.

It has seven deliberately separate execution paths. They must not be confused:

1. **Strict whole-application AOT:** `aot-plan` walks the closed local source
   graph; `aot-build` emits an artifact only if every reached operation has a
   CPython-free route. It never falls back, embeds source/bytecode, or creates a
   self-extracting Python payload.
2. **Native compile:** Python AST → py2bin IR → py2bin optimizer → handwritten
   x86-64/ARM64 instructions → ELF, PE, or Mach-O. This invokes no external
   toolchain, and the generated program needs no Python runtime.
3. **Runtime freeze:** arbitrary CPython projects and target-compatible packages
   are collected with an embedded CPython runtime. The default output is one
   self-extracting `.exe` or `.bin`; this is compatibility packaging, not
   native translation of the application.
4. **Lightweight bundle:** `.pyz`, executable `.bin`, and directory formats
   package project code and dependencies but use a compatible target Python.
5. **Portable-C frontend:** a useful typed subset of Python becomes readable C
   source, or a checksummed `.py2cbin` C-source container. Imports automatically
   plan for the compatible CPython bundle instead of pretending native packages
   can be translated from Python source.
6. **C compiler:** `compile-c` is py2bin's own C compiler, written in Python.
   It lexes and parses C, applies C's integer promotions and conversions, and
   emits py2bin IR that the handwritten encoders turn into machine code. It
   implements the integer and pointer language: the whole integer type zoo with
   exact truncation and extension, local arrays, `&x`, `*p`, pointer
   arithmetic, casts, `sizeof`, `++`/`--`, `?:`, the comma operator, division
   and remainder, `if`/`while`/`do`/`for`/`switch`/`goto`, functions called
   through a real machine call ABI on ARM64 (**recursion works** — direct,
   deep, and mutual), function pointers called through a genuine indirect
   branch, file-scope variables in real static storage, and `printf`. It is not
   a general C, Cython-C, C++, NumPy C-API, ATen, or CUDA compiler: `long
   double`, variadic user functions, and more than eight arguments are rejected
   with a file:line:column error rather than approximated. A real preprocessor
   runs first — macros with `#` and `##`, conditionals, and `#include` of files
   it can find — but a system header it cannot parse is refused, not guessed
   at. `compile-via-c` is the older, narrower bridge that
   round-trips py2bin's *own* generated C.
7. **CPython C-API embedding (`darwin-arm64` only):** the same handwritten C
   frontend also accepts calls to a fixed, vetted list of exported CPython
   entry points through opaque `PyObject *` handles, and the emitted Mach-O
   binds the interpreter's shared library with real dyld binding. The program's
   own logic becomes machine code; Python *object* semantics stay in libpython.
   py2bin does **not** translate ordinary Python into those calls — you write
   them, either as canonical C or as Python importing `py2bin.cabi`.

### Reaching a platform framework

Path 7 binds a fixed list of exported C entry points through real dyld. Since
those may come from any library, they include the Objective-C runtime -
`objc_getClass`, `sel_registerName`, `objc_msgSend` from
`/usr/lib/libobjc.A.dylib` - and the image loads Foundation so classes exist,
because libobjc holds the runtime and not the classes.

That is worth stating carefully, because it is easy to read as more than it is.
Cocoa is compiled Objective-C shipped inside macOS with no source to translate,
and that has not changed. What has changed is that a compiled py2bin binary can
*call* it: the runtime that dispatches to Cocoa is a plain C API, and three
functions are the whole of it. A native binary already builds an `NSString`,
uppercases it, and reads `NSProcessInfo` - linking `libSystem`, `libobjc` and
`Foundation`, with no CPython anywhere in the file.

`objc_msgSend` is declared variadic and is not one: it reads its arguments from
the ordinary registers, which is why an Objective-C compiler casts it to the
callee's real prototype before every call. Each binding here is one such cast,
vetted with an exact signature.

What this does **not** do is make `import webview`, `import AppKit`, or any
PyObjC-based module compile. Those reach Cocoa through a C extension with no
Python source, and nothing here changes that. An application written against
this bridge is written against it directly, in the native subset - it is not a
translation of an existing PyObjC program.

## The three tiers, stated plainly

Three different things in this repository all end in "a file you can run". They
give very different guarantees, and the difference matters more than the
similarity.

| | (a) `freeze` / `bundle` | (b) CPython C-API path | (c) `compile` native |
|---|---|---|---|
| Shape of the idea | PyInstaller-shaped | Nuitka-shaped | a small AOT compiler |
| What executes your logic | CPython, interpreting your source/bytecode | machine code py2bin wrote | machine code py2bin wrote |
| Is CPython present? | Yes, bundled inside the artifact | Yes, linked as an external shared library | No |
| Accepts arbitrary Python? | Yes, in practice | No — a small explicit subset | No — a small explicit subset |
| Third-party packages | Yes, carried as payload | Only whatever the linked interpreter can already import | No |
| Targets | all implemented targets, given a runtime pack and target wheels | `darwin-arm64` only | all implemented targets |
| Artifact runs standalone? | Yes | **No** — needs that exact CPython installed | Yes |

**(a) Freeze** is compatibility packaging. Your program is not translated; it
is shipped next to an interpreter that runs it. This is the tier that handles
real applications with NumPy, Torch, or a GUI.

**(b) The CPython C-API path** is the Nuitka-shaped tier, with Nuitka's
essential dependency removed: Nuitka hands its generated C to gcc, and py2bin
compiles the C itself. What survives that constraint is genuinely narrower than
Nuitka. See [What the C-API path supports](#what-the-c-api-path-supports).

**(c) Native compile** removes CPython entirely. Nothing is interpreted and
nothing is linked; the ELF/PE/Mach-O contains only instructions py2bin encoded.
The price is the smallest subset of the three.

None of these tiers invokes gcc, clang, `as`, `ld`, Xcode, or an SDK. py2bin
cannot: the library does not import `subprocess` and never starts a process, and
the test suite fails if that changes.

## Compiling C

py2bin includes a C compiler written in Python. It lexes and parses C, applies
C's own type rules, and writes machine code directly -- no assembler, no
linker, no gcc or clang anywhere in the path.

```sh
py2bin cc hello.c        # writes ./hello for this machine
./hello
```

That is the whole common case. `--output/-o` names the executable, `-I` and
`-D` reach the preprocessor, and `--target` (or `--os`/`--arch`) cross-compiles
to any of the six supported targets from any host:

```sh
py2bin cc hello.c --os windows --arch x64 -o hello.exe
py2bin cc hello.c --target linux-arm64 -o hello-arm64
```

**What compiles.** The integer types with C11 promotion and conversion, `float`
and `double`, pointers, arrays, `struct`, `union`, `enum`, `typedef`, function
pointers, file-scope variables, recursion, the full statement set including
`switch` and `goto`, a real preprocessor (`#define` with function-like and
variadic macros, `#if`, `#include`, `#` and `##`), and `printf` with runtime
conversions. `<math.h>` supplies `sqrt`, `fabs`, `floor`, `ceil` and `trunc` as
single hardware instructions, and `exp`, `log`, `sin`, `cos`, `tan` and `pow`
as C that py2bin compiles -- accurate to within one unit in the last place
against the platform's own libm.

**What does not.** Variadic *functions* you define yourself, `long double`,
`_Complex`, `_Atomic`, variable-length arrays, bitfields, inline assembly,
compiler builtins, and real system headers, which use extensions this front end
does not implement. There is no linker, so a program is one translation unit
and cannot call into a separate object file or library. Each of these is
refused with a `file:line:col` message rather than mis-compiled.

**This is not a C++ compiler**, and C++ is not a small step from here.
Templates, virtual dispatch, exceptions, destructors, overload resolution and
name mangling are each larger than everything above.

## What the C-API path supports

This is the honest boundary of tier (b). It is deliberately unflattering.

**How you reach it.** Two entry points, both producing the same IR:

```sh
# 1. Python that imports vetted C-API names from py2bin.cabi.
PYTHONPATH=src python3 -m py2bin compile program.py \
  --target darwin-arm64 -o program.bin

# 2. Canonical C that declares the same functions with `extern` prototypes.
PYTHONPATH=src python3 -m py2bin compile-c program.c \
  --target darwin-arm64 -o program.bin
```

The two are interconvertible: py2bin's C frontend parses the C into a Python
AST, so the same program can be run under `python3` (where `py2bin.cabi` makes
the identical calls through `ctypes.pythonapi`) and diffed against the compiled
binary. Every C-API feature below is verified that way — build, run natively,
run the same program under CPython, require identical stdout and exit status.

**What goes through.**

- A fixed table of 31 exported CPython entry points: `Py_Initialize`,
  `Py_Finalize`, `Py_IsInitialized`, `PyRun_SimpleString`,
  `PyLong_FromLongLong`, `PyLong_AsLongLong`, `PyUnicode_FromString`,
  `PyNumber_Add`/`Subtract`/`Multiply`/`TrueDivide`, `PyObject_RichCompare`,
  `PyObject_IsTrue`, `PyObject_Str`, `PyObject_Repr`, `PyObject_Size`,
  `PyObject_GetAttrString`, `PyObject_CallNoArgs`, `PyObject_CallOneArg`,
  `PyImport_ImportModule`, `PyList_New`, `PyList_Append`, `PySys_GetObject`,
  `PySys_WriteStdout`, `PyFile_WriteObject`, `PyFile_WriteString`, `Py_IncRef`,
  `Py_DecRef`, `PyErr_Occurred`, `PyErr_Print`, `PyErr_Clear`. Every one is a
  real exported function — not a macro or a `static inline` — with a fixed
  count of word-sized arguments. A test asserts each is exported by the running
  interpreter's dylib.
- Opaque `PyObject *` handles in locals, parameters, and return values.
- `long long` locals, arithmetic, comparisons, `if`, `while`, and calls to
  your own functions, which are inlined.
- `NULL` checks: `handle == NULL` and `handle != NULL`.
- Compile-time string literals as `const char *` arguments.
- Because the interpreter is real, so is everything it does: importing `math`,
  calling `math.isqrt`, building a `list` and printing `[2, 4, 6, 8]`, and
  `str()` of a Python `int` all work, because CPython performs them.
- `PyImport_ImportModule` reaches anything that interpreter can already import,
  including third-party packages on its `sys.path`. Note what that does and
  does not mean: py2bin neither translates nor packages those modules, and the
  binary does not carry them. The same applies to `PyRun_SimpleString`, which
  really does execute arbitrary Python — *interpreted from a string at runtime*.
  Wrapping a program in `PyRun_SimpleString` would produce a launcher for
  interpreted source, not compiled code, and this project does not count that
  as compiling anything.

**What is rejected, with a `file:line:col` error.**

- Every target except `darwin-arm64`. There is no C-API path on Linux, on
  Windows, or on x86-64 macOS.
- Dereferencing a handle, pointer arithmetic on it, ordering comparisons
  (`<`, `>`) between handles, or mixing a handle with a `long long`. Handles
  are opaque 64-bit values and nothing else. This is why py2bin never needs to
  read `Python.h`: its preprocessor could include the file, but what is inside
  is macros, `static inline` functions and struct layouts that this compiler
  does not implement, so the vetted table of entry points is written out
  instead.
- Any prototype that disagrees with the vetted ABI (wrong argument count, or a
  non-`void` return for a `void` function).
- Variadic C-API entry points such as `PyObject_CallFunctionObjArgs`. Apple's
  arm64 ABI passes variadic arguments on the stack and this backend has no
  stack-argument path, so they are absent from the table rather than
  miscompiled. `PySys_WriteStdout` is allowed only with a literal containing
  no `%`.
- More than eight arguments — the AAPCS64 register budget. Refused, not
  truncated.
- An extern call inside `A ? B : C` or a short-circuited `&&`/`||`. Both arms
  are lowered eagerly, so the call in the untaken arm would still run.
- Using the result of a `void` function as a value.

**What py2bin does not do for you, and will not pretend to.**

- **No automatic translation of Python into C-API calls.** This is the largest
  gap versus Nuitka. A list comprehension, a `dict`, a `for` over a list, and
  `import json` are all rejected outright by the native frontend's own subset,
  which still applies here. Constructs that *are* in that subset — `print("hi")`,
  integer and float arithmetic, `while`, a `class` — compile the way tier (c)
  compiles them, straight to instructions; `print` becomes a `write` syscall,
  not `PyObject_Print`. Either way nothing becomes a `PyObject` operation on
  your behalf. If you want a Python object built, you call
  `PyLong_FromLongLong` yourself.
- **No automatic reference counting.** You call `Py_IncRef`/`Py_DecRef`. py2bin
  emits exactly the calls you wrote and does not verify ownership, so a leak or
  a double-free in your program stays a leak or a double-free.
- **No exception propagation.** A failing C-API call returns `NULL` and the
  error stays pending. py2bin inserts no checks and generates no unwinding;
  checking `PyErr_Occurred` is your program's job.
- **The artifact is not standalone.** The Mach-O carries an `LC_LOAD_DYLIB` for
  the *build host's* CPython at its absolute path (`otool -L` shows it next to
  `libSystem`). Move the binary to a machine without that exact interpreter at
  that exact path and dyld will refuse to start it. Tier (a) is what produces a
  distributable artifact.
- **Two divergences between compiled and interpreted runs.** Both are inherent
  to embedding rather than bugs py2bin can fix, and both are silent, so they are
  stated rather than hidden.
  1. *A failing call.* Under CPython the shims go through `ctypes.pythonapi`,
     which converts a set error indicator into a raised exception; a compiled
     binary just receives `NULL`. The two runs therefore agree only while no
     C-API call fails. `py2bin/cabi.py` states this in the source.
  2. *Unflushed output when `Py_Finalize` is not called.* `sys.stdout` is
     buffered inside the interpreter. Running the twin `.py` under `python3`
     flushes it at interpreter shutdown, but a compiled binary that returns from
     `main` without calling `Py_Finalize` exits before anything is flushed and
     prints **nothing at all**. This is ordinary C-embedding behaviour — a
     gcc-built program does the same — but it means a program that looks correct
     under `python3` can produce empty output once compiled. Call `Py_Finalize`
     before returning. Every shipped example and test does.

## What “pure Python” means

The **py2bin implementation** imports only Python standard-library modules.
Both `[build-system].requires` and runtime dependencies are empty. The test
suite checks these invariants.

This guarantee applies to py2bin itself. A user who asks py2bin to package
Torch, `bpy`, Manim, NumPy, or another third-party project is explicitly asking
to carry that project's code and native files as application payload. Those
projects may have been implemented using C, C++, Rust, Cython, CUDA, or other
tools; py2bin neither claims ownership of that code nor recompiles it.

`py2bin compile` does not import or invoke Cython, Nuitka, mypyc, Rust, a C/C++
compiler, PyInstaller, Docker, or a native bootloader. Supplying binary wheels
to `freeze` also avoids local native builds, but it does not turn those wheels
into py2bin-authored machine code.

Native binaries are always target-specific—there is no single machine-code file
that runs on every OS and CPU. `py2bin` can manufacture PE/ELF/Mach-O files in
pure Python as target backends are implemented. Third-party native packages
remain tied to their operating system, CPU architecture, Python ABI, drivers,
and system libraries, so full-library bundle mode uses a compatible Python
runtime while native mode progressively replaces Python semantics with its own
runtime and library adapters.

## Quick start

No installation is required in the repository:

```sh
# Produce Windows, Linux, and macOS x86-64/arm64 artifacts in one command.
# This does not invoke an assembler, linker, compiler, SDK, or target runtime.
PYTHONPATH=src python3 -m py2bin compile-all examples/native_hello.py \
  --output-dir dist/all --mac-app --clean

PYTHONPATH=src python3 -m py2bin compile examples/native_hello.py \
  --output dist/native-hello --clean
./dist/native-hello

# Prove the entire reachable app can use only the direct-native backend.
PYTHONPATH=src python3 -m py2bin aot-plan \
  examples/native_library/main.py \
  --source-root examples/native_library --json --strict

# Build or fail before writing anything. The proof records the target, raw
# artifact SHA-256, byte count, and no-CPython/no-payload/no-extraction result.
PYTHONPATH=src python3 -m py2bin aot-build \
  examples/native_library/main.py \
  --source-root examples/native_library \
  --via-c --c-output dist/NativeLibrary.c \
  --target windows-x86_64 --output dist/NativeLibrary.exe \
  --attestation dist/NativeLibrary.aot.json --clean

# Inline supported pure-Python library functions into every native target.
PYTHONPATH=src python3 -m py2bin compile-all \
  examples/native_library/main.py \
  --source-root examples/native_library \
  --strict-library-root examples/native_library/native_math \
  --output-dir dist/native-library --clean

# Audit every top-level function in a library without importing or running it.
PYTHONPATH=src python3 -m py2bin audit-library app/purelib \
  --source-root app --json --strict

# Make a native build fail if any function in the selected library needs
# CPython. Supported unused functions are validated and dead-code removed.
PYTHONPATH=src python3 -m py2bin compile app/main.py \
  --source-root app --strict-library-root app/purelib \
  --target windows-x86_64 --output dist/App.exe --clean

# Strict whole-source gate: fail instead of falling back to bundled CPython.
PYTHONPATH=src python3 -m py2bin assemble \
  examples/native_library/main.py \
  --source-root examples/native_library --mode native \
  --output dist/NativeLibrary --clean

# Translate supported Python to portable C without invoking a C compiler.
PYTHONPATH=src python3 -m py2bin emit-c examples/c_program.py \
  --output dist/c_program.c --clean

# For the smaller integer intersection, retain and then really parse the C
# before py2bin writes target machine code. No external C toolchain is used.
PYTHONPATH=src python3 -m py2bin compile-via-c app/integer_program.py \
  --c-output dist/integer_program.c \
  --target windows-x86_64 --output dist/integer_program.exe --clean

# Compile a canonical py2bin C file through the same handwritten frontend.
PYTHONPATH=src python3 -m py2bin compile-c dist/integer_program.c \
  --target linux-arm64 --output dist/integer_program-arm64 --clean

# Nuitka-shaped tier, without Nuitka's gcc: C that calls the CPython C-API,
# compiled to machine code by py2bin itself. darwin-arm64 only, and the result
# dyld-links this machine's CPython rather than being standalone.
PYTHONPATH=src python3 -m py2bin compile-c examples/capi_embedding.c \
  --target darwin-arm64 --output dist/capi_embedding --clean
./dist/capi_embedding
# The same program as Python, for a stdout/exit-status diff against the binary.
PYTHONPATH=src python3 examples/capi_embedding.py

# Explain whether a program can use the C subset or needs CPython bundling.
PYTHONPATH=src python3 -m py2bin plan-c app/main.py

# List common-library support, or inspect one source file without importing it.
PYTHONPATH=src python3 -m py2bin capabilities
PYTHONPATH=src python3 -m py2bin capabilities app/main.py --json

# Turn an installed/staged package tree into a standards-structured wheel.
PYTHONPATH=src python3 -m py2bin wheel build/package-root \
  --output-dir dist/wheels --name my-package --version 1.0

# Fetch pinned imported source and attempt the real native compiler only.
PYTHONPATH=src python3 -m py2bin compile app.py -o dist/app \
  --source-lock py2bin-sources.lock.json \
  --source-cache /Volumes/D/py2bin-source-cache

# A native Apple Silicon application bundle:
PYTHONPATH=src python3 -m py2bin compile examples/native_hello.py \
  --target darwin-arm64 --app --output dist/NativeHello --clean

# Automatically choose native code when possible and otherwise emit one
# self-extracting embedded-CPython file.
PYTHONPATH=src python3 -m py2bin assemble examples/generic_app/main.py \
  --source-root examples/generic_app --dependency-mode none \
  --output dist/GenericApp --compact --clean
./dist/GenericApp.bin hello

# Short compatible-bundle spelling. `bundle` is an alias for `freeze`, and
# `--optimize-size` is an alias for the explicit `--compact` policy.
py2bin bundle app.py -o dist/App --optimize-size --clean

PYTHONPATH=src python3 -m py2bin analyze examples/hello/main.py
PYTHONPATH=src python3 -m py2bin build examples/hello/main.py \
  --source-root examples/hello --format bin --output dist/hello --clean
./dist/hello
```

Or install it into a virtual environment:

```sh
python3 -m pip install python-to-binary
py2bin build app/main.py --source-root app --output dist/my-app --format bin
```

Install the current GitHub version:

```sh
python3 -m pip install \
  "git+https://github.com/yu314-coder/python_to_binary.git"
```

See the [CPython-free whole-application architecture](docs/NATIVE_AOT_ARCHITECTURE.md)
and the [detailed compiler, bundling, target, and release guide](docs/DETAILED_GUIDE.md).

## Formats

| Command/mode | Output | Application logic | Installed Python needed on target |
|---|---|---|---:|
| `aot-build` | ELF, PE, Mach-O, macOS `.app` | closed-world, attested py2bin-generated machine code or no artifact | No |
| `aot-build --via-c` | canonical whole-program `.c`, then ELF/PE/Mach-O | local functions are lowered/inlined, emitted as C, reparsed to identical IR, then encoded | No |
| `compile` | ELF, PE, Mach-O, macOS `.app` | py2bin-generated machine code | No |
| `compile-via-c` | ELF, PE, or Mach-O | canonical C parsed by py2bin, then py2bin-generated machine code | No |
| `compile-c` | ELF, PE, or Mach-O | C compiled by py2bin's own C compiler into py2bin-generated machine code | No |
| `emit-c` | `.c` or `.py2cbin` | C source, not yet an executable | N/A |
| `build --format pyz` | Python zip application | CPython bytecode/source | Yes |
| `build --format bin` | Executable Python zip application | CPython bytecode/source | Yes |
| `build --format dir` | Project, packages, and launcher | CPython bytecode/source | Yes |
| `freeze` / `bundle` (default) | One self-extracting PE/ELF/Mach-O file, or macOS `.app` with one embedded payload | CPython bytecode/source | No |
| `freeze --onedir` / `bundle --onedir` | Unpacked embedded-runtime directory | CPython bytecode/source | No |

Every executable above is a valid OS binary or executable launcher, but only
`aot-build`, `compile`, `compile-via-c`, and `compile-c` translate the
supported application logic into py2bin-generated CPU instructions.
`aot-build` is the strongest contract because it audits the reachable local
source graph and writes a proof record without permitting another backend.
With `--via-c`, the optimized whole-program IR—including supported local
library functions—is serialized as deterministic C containing explicit native
slots, labels, branches, byte writes, and returns. py2bin's handwritten parser
must reconstruct the exact IR before the binary writers run. The retained C is
not passed to GCC, Clang, or another compiler.
`freeze` produces a real native launcher around embedded CPython; calling the
contained Python application “natively compiled” would be incorrect.

## Why NumPy/Torch imports are rejected, not reimplemented

`import numpy` and `import torch` are rejected by `compile` with a
source-located error. py2bin does not silently reimplement a third-party
numerical library, because a from-scratch integer reimplementation does not
match the real package's runtime object semantics, and shipping a binary whose
observable result differs from CPython would violate the compiler's honesty
contract.

Concretely, a NumPy/Torch reduction is not a plain `int`: `numpy.sum(...)`
returns an `numpy.int64` and `torch.sum(...)` returns a 0-dimensional
`Tensor`. Under CPython, `raise SystemExit(numpy.sum(...))` therefore prints
the value's repr and exits `1`, and mixing a NumPy result with a Torch tensor,
or calling `torch.relu` on a NumPy array, raises. A native integer kernel that
returned `int` would diverge from every one of these behaviors, so the import
is refused rather than approximated. (An earlier `--experimental-kernels`
option attempted this substitution and was removed for exactly this reason.)

If a program genuinely needs the real NumPy or PyTorch, use `freeze`/`bundle`,
which carries the real packages and the embedded CPython runtime that executes
them. That is compatibility packaging, not native translation.

## Claims audit

The following wording is intentionally strict. “No installed Python on the
target” is not the same claim as “the application does not use CPython.”

| Claim | Accurate? | Exact meaning |
|---|---:|---|
| No GCC, Clang, assembler, or linker is required | Yes | `compile`, `compile-via-c`, and `compile-c` write ELF, PE, and Mach-O bytes directly. `freeze` copies a compatible CPython runtime and installed package files. |
| The target computer does not need Python installed | Yes, for native compile modes and `freeze` | A native compile artifact has no Python runtime. A `freeze` artifact carries its own CPython runtime. The lighter `build` formats still need compatible target Python. |
| A complete frozen application does not use CPython | No | `freeze` embeds and starts CPython; only the documented native subsets replace Python execution with generated machine code. |
| Third-party packages do not need a Python runtime | No | NumPy, Torch, `bpy`, Manim, and similar packages are imported by the embedded CPython runtime in `freeze` mode. |
| Arbitrary Python is translated completely to machine code | No | `compile` accepts the documented static subset and rejects everything else with a source location. Third-party numerical packages such as NumPy/Torch are rejected outright, never reimplemented. |
| py2bin compiles C that calls the CPython C-API, without gcc | Yes, for a vetted subset on `darwin-arm64` | py2bin's own C parser and ARM64 encoder produce a Mach-O that dyld-binds the interpreter and calls 31 vetted entry points. Verified by running the binary and diffing stdout and exit status against the same program under CPython. |
| py2bin is a Nuitka replacement | No | Nuitka translates ordinary Python into C-API calls. py2bin does not generate those calls at all — you write them. Tier (b) is an embedding surface, not an automatic compiler for the Python language. |
| A C-API artifact runs on a machine without Python | No | It records an `LC_LOAD_DYLIB` for the build host's CPython at an absolute path. Only tier (c) `compile` output and `freeze` output are standalone. |
| py2bin itself has no third-party Python dependency | Yes | The package imports only Python standard-library modules, and its build/runtime dependency lists are empty. Tests enforce both properties. |
| The build computer does not need Python | No | Building requires Python 3.10+ to run py2bin. No native compiler toolchain is required for the supported direct-binary path. |

## Source-only builds: exact boundary

“Use only the application source and py2bin” has a precise, limited meaning:

- For programs inside the documented static `compile` subset, it is enough to
  have the source, Python 3.10+, and py2bin on the build computer. py2bin can
  write Windows PE, Linux ELF, and macOS Mach-O files without Wine, Rosetta,
  an assembler, a linker, a target SDK, or a target Python runtime.
- For arbitrary Python, source files alone are not enough. Full Python
  semantics require CPython or a future complete py2bin runtime.
- An imported third-party package must also exist as target-compatible package
  data. An import name does not contain that package's implementation.
- `freeze` embeds the current compatible CPython runtime and copies compatible
  installed packages or supplied wheels. It does not manufacture a missing
  Windows runtime on macOS, translate a macOS native wheel into a Windows
  wheel, or recreate a package whose files were never supplied.

For example, the `manim_app` repository imports `webview`, but does not contain
pywebview's implementation. It also installs Manim and related tools into a
separate environment. A working Windows build therefore needs Windows CPython,
pywebview and its Windows backend, plus every other required package and native
file. With only the `manim_app` source and py2bin, the current compiler must
reject the program. Producing a PE header that cannot start the application
would be a broken artifact, not successful assembly.

`manim_app` also imports `winpty`. Its full terminal mode therefore requires a
matching `pywinpty` wheel; for CPython 3.12 on Windows x86-64 that wheel must
carry the `cp312` and `win_amd64` tags. py2bin extracts its `.pyd`, DLL, agent,
and console helper files as wheel data. Excluding `winpty` intentionally selects
the application's simpler subprocess fallback instead.

This limitation is not specific to pywebview or Manim. “Any Python program and
every third-party package from source alone” would require py2bin to implement
the complete Python language, standard library, extension ABI, GUI frameworks,
and each missing third-party implementation. That work is not complete and is
not claimed.

## Pinned source download and native attempt

`compile --source-lock` detects statically imported non-stdlib modules,
downloads their locked source archives through Python's HTTPS client (no pip),
verifies SHA-256, extracts them without running project code, and attempts the
handwritten native frontend. A local archive path can be locked for offline
builds.

Example lock:

```json
{
  "schema": 1,
  "sources": {
    "demo": {
      "url": "https://github.com/owner/demo/archive/COMMIT.tar.gz",
      "revision": "FULL_COMMIT_ID",
      "sha256": "64_LOWERCASE_HEX_DIGITS",
      "subdirectory": "src"
    }
  }
}
```

Build:

```sh
py2bin compile app.py -o dist/app \
  --source-root . \
  --source-lock py2bin-sources.lock.json \
  --source-cache /Volumes/D/py2bin-source-cache \
  --target darwin-arm64
```

After the lock is supplied, source discovery and fetching are automatic.
py2bin intentionally does not guess a Git repository from an import name:
import names are not globally unique, repositories can be renamed or
compromised, and an unpinned `main` branch is not a reproducible input.

The fetcher:

- accepts credential-free HTTPS URLs or explicit local archive paths;
- requires an immutable revision label and exact SHA-256;
- accepts ZIP or tar archives;
- rejects traversal paths, links, special files, duplicate/case-colliding
  members, excessive member counts, and configured download/expanded-size
  limits;
- records and rechecks a hash of the extracted source tree;
- never imports the downloaded package or runs `setup.py`, build backends, or
  shell commands.

The current successful cross-package native forms are deliberately narrow:

- `from MODULE import CONSTANT`, when the export is statically evaluable; and
- `from MODULE import FUNCTION`, when the function has only positional
  parameters with optional static integer defaults, no decorators/variadics,
  and consists of supported
  integer assignments, Boolean/chained-comparison logic, native `if`/`while`/
  `for range` control flow, and integer returns on every fall-through path.
  Procedures returning `None` may use the same native control flow, bare
  returns, constant output, and calls to other supported procedures. Static
  annotation expressions are erased when the supported closed program does not
  otherwise request annotation reflection; they do not add runtime type
  implementations.

Supported functions can call other supported non-recursive functions. Calls
use expression inlining for simple functions and imperative IR inlining with
private stack slots/labels for loops, early returns, and mutable branches. The
expanded IR is encoded by the handwritten x86-64/ARM64 backends. There is no
Python call, bytecode, CPython runtime, or hidden fallback in the result.
Absolute and relative local modules—including nested local imports—are
confined to `--source-root`; pinned modules use verified source-lock roots.
The regression suite compares simple imported helpers with manually inlined
equivalents and separately executes an imperative function-loop sample.

When a function is imported, its provider module may contain constants,
function definitions, statically resolvable local `from` imports, docstrings,
`__future__` imports, and `pass`, but not executable top-level initialization
whose omission would change Python semantics. Recursion, classes, mutable
containers, dynamic calls/imports, Cython-generated C, C/C++/Rust/Fortran/CUDA
sources, and CPython extensions still fail with a source location. A failed
native conversion cannot be mistaken for a native artifact.

`audit-library` walks a source tree without importing it, lowers every
top-level function through the real native frontend, and separately lists
prebuilt native payloads and browser assets. `compile`/`compile-all`
`--strict-library-root` applies that audit as a build gate. Passing the gate
means every inspected function is accepted by the current AOT semantics; it
does not claim that `.so`/`.pyd` ABI bindings or HTML/CSS/JavaScript became
py2bin-authored CPU instructions.

`fetch-sources` performs only the verified download/extraction phase:

```sh
py2bin fetch-sources app.py --source-root . \
  --source-lock py2bin-sources.lock.json \
  --source-cache /Volumes/D/py2bin-source-cache --json
```

Native compile targets currently implemented are `linux-x86_64` (ELF),
`linux-arm64` (ELF), `darwin-x86_64` and `darwin-arm64` (Mach-O), and
`windows-x86_64` and `windows-arm64` (PE `.exe`).
Run `py2bin targets` to list them. The native
frontend supports module constants, static strings/f-strings, a signed 64-bit
runtime for variables and arithmetic, an IEEE-754 binary64 (`float`) runtime
for variables, arithmetic, comparisons, and `int`/`float` conversion,
comparisons, `if`, `while`, `for NAME in range(...)`, `break`, `continue`,
loop `else` bodies,
`print()` of compile-time values, pure integer function inlining across
local/pinned modules, and integer-expression exit status. On POSIX targets it
additionally supports runtime lists, dicts, sets, tuples, runtime UTF-8
strings, exceptions, and user-defined classes over a bump-arena (anonymous
`mmap`); on `darwin-arm64` it
can call vetted libc symbols and vetted CPython C-API entry points through the
`py2bin.cabi` adapter ABI. It rejects everything else with a source location
rather than producing a subtly incorrect executable.

A class instance is a plain heap block whose layout is fixed at build time:
attribute *i* lives at `pointer + i * 8`. Because the class of every instance is
statically known, there is no object header, type pointer, or vtable, and a
method call resolves directly to one body that is inlined like any other native
call. That is what makes classes compile without an interpreter — and it is also
why dynamic attributes and duck-typed dispatch are rejected rather than
approximated. Single inheritance fits inside that rule: a subclass repeats its
base's attributes at the front of its own layout, so one inherited body reads
the same offsets on either class, and its methods are merged at build time with
the subclass winning. What stays rejected is the part that would need a vtable
— binding one variable to a base and later to its subclass — because dispatch
resolves from the variable's class, not from the object.

Verification status is deliberately explicit. The compiler host for this
project is `darwin-arm64`, so every native feature above is verified by
building a `darwin-arm64` Mach-O, running it, and comparing its exit code and
stdout against the same source under CPython. The other targets are verified by
building a structurally valid image (correct ELF/PE/Mach-O header) and, for
x86-64, disassembling the emitted `.text` with `otool -tvV`; they are not
executed on this host, and no claim is made beyond "builds and disassembles to
the intended instructions." The heap and adapter-ABI features are intentionally
narrower than the integer/float core for this reason.

Runtime doubles are lowered to real hardware floating point: the x86-64 backend
emits SSE2 (`movsd`, `addsd`/`subsd`/`mulsd`/`divsd`, `ucomisd`,
`cvtsi2sd`/`cvttsd2si`) and the ARM64 backend emits scalar NEON/FP
(`fadd`/`fsub`/`fmul`/`fdiv`, `fcmp`, `scvtf`/`fcvtzs`). No soft-float library,
external compiler, or Python runtime is involved. Runtime `float` division is
accepted only with a nonzero numeric constant divisor, because a runtime
divisor can be zero and Python raises `ZeroDivisionError` there; honoring that
exception needs the object runtime, so it is rejected rather than silently
emitting IEEE infinity/NaN. Formatting a runtime double back to text is supported and is the compiler's
own work: shortest round-trip conversion (Burger-Dybvig over fixed-width
bignums in the arena) that reproduces CPython's `repr` exactly, including its
tie-breaking toward the even digit.

The word “supports” is intentionally narrow:

| Python feature | `compile` now | Exact behavior |
|---|---:|---|
| Literal `str`, `bytes`, `int`, `float`, `bool`, `None` | Yes | Represented while lowering the static program |
| Single-name assignment and annotation | Yes | Static values are folded; runtime integer values use native stack slots |
| Integer `+`, `-`, `*`, bitwise operations, shifts | Yes | Runtime signed 64-bit instructions; overflow wraps to 64 bits rather than creating Python big integers |
| Runtime `float` `+`, `-`, `*`, unary `-` | Yes | IEEE-754 binary64 in hardware SSE2/NEON registers; mixed `int`/`float` promotes the integer operand |
| Runtime `float` `/` | Yes | A constant zero divisor is a build error. A runtime divisor is pinned in a slot, compared against zero, and raises a catchable `ZeroDivisionError` rather than producing IEEE infinity or NaN |
| `float`/`int` conversion and `float` comparisons | Yes | `float(x)` widens with `cvtsi2sd`/`scvtf`; `int(x)` truncates toward zero with `cvttsd2si`/`fcvtzs`; comparisons use `ucomisd`/`fcmp` |
| Integer comparisons and dynamic `if` | Yes | Runtime signed comparisons and native branches |
| Constant arithmetic, Boolean and conditional expressions | Yes | Evaluated at build time when no runtime value is involved |
| Constant `if` | Yes | Only the selected branch is emitted |
| `while`, `for NAME in range(...)`, `break`, `continue` | Yes | Native branches; `range` step must be a nonzero integer constant |
| `while ... else`, `for ... else` | Yes | The else body sits between the loop's fall-through label and a second exit label, which is where a `break` jumps; no run-time flag, and a break in a nested loop does not skip the outer loop's else |
| `del xs[i]` on a runtime list | Restricted, POSIX only | The tail shifts down and the header length drops by one, in place, so an alias sees the deletion. Negative indices count from the end; an out-of-range index reports `IndexError` and exits 1. `del name`, `del obj.attr`, `del xs[a:b]`, and a `del` shortening a list a `for` is walking are each rejected by name |
| `del d[k]` on a runtime dict | Restricted, POSIX only | The entry's state word becomes a tombstone, which a probe walks over instead of stopping at, and the key leaves the insertion-order list. A missing key reports `KeyError` and exits 1; deleting while a `for` walks the same dict reports `RuntimeError: dictionary changed size during iteration`, as CPython does, even for the last key. Tombstones count towards the table being full, so the table is rebuilt at the same capacity every capacity/2 deletions and a delete-heavy loop spends arena: around 200000 insert/delete pairs on one dict end in `MemoryError` where CPython keeps going |
| f-string | Restricted | Fields may be runtime `int`, `float`, `bool`, or `str`. A literal format specifier `[[fill]align][sign][0][width][,][.precision][type]` is supported with `type` one of `d`, `f`, `s` or omitted; `.Nf` rounds the exact binary value half to even, as CPython does. `!r`/`!s`/`!a` are accepted on numbers and `!s` on a string. `e`, `g`, `n`, `%`, `b`, `o`, `x`, `#`, `z`, `_`, a precision on a string, a separator beside zero padding, and a non-literal specifier are rejected at build time |
| Integer functions and procedures | Restricted | Positional and undecorated; static integer defaults and named calls; assignments, Boolean/chained comparisons, integer control flow, loops, value or bare returns, constant side effects, and acyclic calls are inlined into native IR |
| Runtime `list` | Restricted, POSIX only | Literal build, constant/runtime index load and store, and `len()` over a bump-arena (anonymous `mmap`) of signed-64-bit elements. Negative indices count from the end as in Python. A constant index is range-checked at build time; a runtime index is range-checked by emitted instructions that report `IndexError` on stderr and exit 1, exactly like CPython, instead of reading or writing outside the list. An element may be an `int`, `float`, `bool`, `str`, or another list, and one list holds one kind: a list element keeps whatever object was put in it, so widening an integer into a float list would read back as `2.0` where CPython reads `2`, and a mix is refused. An empty literal takes its kind from the first thing stored in it, or from a `list[T]` annotation. A second name for the same list is refused, because appending moves the block and only one name would follow it; `xs[:]` copies. Bare list use is rejected |
| Runtime list index inside `A if C else B` or `and`/`or` | No | Both arms are lowered eagerly, so a bounds check would run even when Python would not evaluate that branch; rejected with a source location. Use an `if` statement, which is supported |
| Float `A if C else B`, and a function returning a float from a branching body | Restricted | Lowered as a real branch, so an arm that can trap is only evaluated on the path Python takes. Both arms must have the same kind: one `int` arm and one `float` arm share a slot and one slot cannot print both `1` and `2.5`, so the mix is rejected. A body whose branches all end in a return is folded into one conditional expression before the call site picks a lowering, so the returned kind is known in time; a body with a loop is inlined statement by statement instead |
| Runtime float divisor inside `A if C else B` or `and`/`or` | No | The integer conditional lowers both arms eagerly, so the divisor's zero check would raise on the branch Python never takes; rejected with a source location. Use an `if` statement |
| Runtime `str` | Restricted | `""` seed, `+` concatenation, slicing, `s[i]`, `for ch in s`, `==`/`!=`, `<`/`<=`/`>`/`>=`, `in`, `len()`, `ord()`/`chr()`, `print()`, sorting a list of them, and use as a dict key, via the same arena. Holds any UTF-8 text: the header counts bytes, which is what a write needs, and both `len()` and indexing count code points, which is what CPython reports. Ordering walks the bytes, which UTF-8 makes the same order as walking the code points |
| Runtime `dict`, `set`, `tuple` | Restricted, POSIX only | Open-addressed table in the arena with linear probing, doubling past half full, and tombstones for `del`; iteration follows insertion order. Keys are `int` or `str`, values `int` or `float`. `d.get(k, default)` requires the default, because the one-argument form answers `None` and there is no `None` here. A set cannot be iterated: CPython's order is unspecified and would differ |
| List methods | Restricted, POSIX only | `append()`, `sort()`, `pop()`, `pop(i)`, `insert()`, `remove()`, `index()`, `count()`. `insert()` clamps its index as CPython's does; the others raise CPython's own messages. Sorting is in-place insertion sort over integers, floats or strings |
| File reading and writing | Restricted, POSIX only | `open(path).read()` answers the whole file as a string; `with open(path, "w") as f` accepts `f.write(...)` inside the block. Modes `r`, `w`, `a`. Straight to the open/read/write/close system calls - no libc. A failed open raises a catchable `FileNotFoundError` or `OSError`. Windows is refused by name, since it would need CreateFile and its handles |
| `assert` | Yes | Raises a catchable `AssertionError` with CPython's message. Always emitted - CPython drops asserts under -O and there is no -O here. The message must be known at build time, since it is written into the image |
| `[v] * n` | Restricted, POSIX only | Repeats the literal, with the count read at run time; zero or less gives an empty list. Repeating `[]` is refused, since nothing in it states an element kind |
| Runtime `float` `//` and `%` | Yes | Through a remainder found by repeated subtraction of a scaled divisor, because `x - trunc(x / y) * y` is wrong once the quotient is large enough to round. Python's sign rule, so the answer follows the divisor; a zero divisor raises `ZeroDivisionError` |
| Truth of a container | Yes | `if xs:`, `while queue:`, `not s`, `bool(d)` - true when it is not empty, answered from the count. The slot holds a block address, which is never zero, so reading it as a number made an empty container true; that is why this was once refused. A runtime float is true when it is not zero. `and`/`or` work in a condition; the value form (`xs and ys`) is still refused, since it answers with one of the two and one slot cannot hold either kind |
| Stepped slice | Restricted | `xs[a:b:step]` on a list takes any non-zero constant step, following Python's rules for the direction. `s[::-1]` reverses a string by code point; a wider step on a string is rejected, since it would have to rescan from the start for every code point it lands on |
| `zip()`, `round()`, `divmod()` | Restricted | `for a, b in zip(xs, ys)` over any number of lists, stopping with the shortest. `round(x)` breaks ties toward the even number; `round(x, n)` is refused because it rounds in decimal and answers a float. `divmod()` is only `q, r = divmod(a, b)`, since a tuple here is a block built from a literal |
| A name bound on only some paths | No | CPython raises `NameError`; there is no run-time bit recording whether a slot was written, so reading one is refused at build time rather than answering with whatever preceded it. An arm that leaves by `raise` or `return` does not count against this |
| User-defined `class` | Restricted | Construction, integer attributes, attribute load/store, and methods (including a method calling another method) over the same bump arena. Every attribute must be assigned unconditionally in `__init__`, which fixes the layout and guarantees no read hits a slot Python would treat as unset. Dispatch is static, so calls inline. A `float` attribute is declared by annotating it in `__init__` (`self.x: float = ...`); an integer stored into one is refused rather than widened, because CPython keeps the integer. Class attributes, decorated methods (`property`, `staticmethod`, `classmethod`), special methods other than `__enter__`/`__exit__`, recursion, attributes created outside `__init__`, a name that is both an attribute and a method, and rebinding a variable to another class are rejected. Single inheritance is supported: the subclass's layout is the base's attributes followed by its own, methods are inherited unless overridden, and `super().__init__(...)` is accepted as a bare statement in the subclass's `__init__`. Multiple bases, a base that is not a class defined earlier in the same module, `super()` anywhere else, a subclass `__init__` that leaves an inherited attribute unassigned, and an inherited attribute whose `float`/integer kind disagrees between the two classes are rejected. |
| Adapter-ABI extern call | Restricted, `darwin-arm64` only | `from py2bin.cabi import NAME` binds a vetted libc symbol (e.g. `abs`, `strlen`, `getpid`) through real dyld; integer and compile-time-constant C-string arguments only. Rejected for every non-`darwin-arm64` target and for unknown symbols |
| CPython C-API call | Restricted, `darwin-arm64` only | The same adapter ABI also exposes 31 vetted CPython entry points, so a compiled binary can drive an embedded interpreter through opaque `PyObject *` handles. You write the calls; py2bin never generates them from ordinary Python, never manages reference counts, and never propagates a C-API error. The artifact links the build host's CPython by absolute path and is not standalone. See [What the C-API path supports](#what-the-c-api-path-supports) |
| `print(...)` | Yes | Constant UTF-8 bytes, or (POSIX only) a runtime ASCII string, emitted through an OS write API/syscall |
| `SystemExit(integer)` / `sys.exit(integer)` | Yes | Constant or runtime integer expression becomes the OS process-exit value |
| Runtime input and command-line arguments, multiple inheritance, dynamic attribute access, `set` iteration, and a `def` under a run-time condition | No | Rejected by `compile` with a source location; compatible mode needs CPython |
| Imports | Restricted local/pinned source plus `sys` | Static constants and supported functions can cross nested local module boundaries; native extensions and dynamic modules are not translated |

The bundle-format `bin` uses Python; `py2bin compile` produces actual machine
code. These writers encode executable headers, import tables, system calls, and
instructions directly rather than shelling out to an assembler.

## Self-written optimizer

The native compiler has its own target-independent optimizer. It currently:

- propagates assignment constants;
- folds arithmetic, Boolean, comparison, conditional-expression, and f-string
  constants;
- selects only the reachable arm of a constant `if`;
- removes assignments that have no runtime representation;
- removes empty writes;
- merges adjacent writes to reduce system calls;
- removes operations after the first process exit;
- inserts one canonical successful exit when required.

The compiler also lowers runtime integer variables and structured control flow
to target-independent IR. The x86-64 and ARM64 backends encode stack loads and
stores, arithmetic, comparisons, conditional/unconditional branches, and
process exit directly. That runtime path is not described as constant folding.

These transformations are deterministic and covered by equivalence tests. No
optimizer can truthfully be “fully optimal” for every Python program. py2bin
therefore documents each safe optimization instead of claiming universal
optimality, and unsupported dynamic semantics remain compilation errors.

Select the OS and architecture independently:

```sh
py2bin compile program.py -o dist/program.exe --os windows --arch arm64
py2bin compile program.py -o dist/program --os linux --arch x86_64
py2bin compile program.py -o dist/program --os macos --arch arm64
```

Accepted architecture aliases include `x64`, `amd64`, and `aarch64`. Exact
canonical targets remain available through `--target`.

## Python-to-C path

`emit-c` lowers a deterministic Python subset to ISO-style C source using only
the Python standard library. The current subset includes numeric and string
constants, local variables, arithmetic, comparisons, Boolean expressions,
`if`, `while`, `for range(...)`, functions, returns, `break`, `continue`,
`print`, and simple f-strings.

```sh
py2bin emit-c program.py -o program.c
py2bin emit-c program.py -o program.py2cbin --container
py2bin plan-c program.py

# Smaller integer subset: the generated C is parsed again, then lowered to
# py2bin IR and handwritten target instructions.
py2bin compile-via-c program.py --c-output program.c \
  --target windows-x86_64 -o program.exe

# Direct C input goes through py2bin's own C compiler, preprocessor first.
py2bin compile-c program.c --target linux-arm64 -o program-arm64

# -I adds a directory the preprocessor searches for #include; -D defines a
# macro before the file is read, exactly as any other C compiler does.
py2bin compile-c program.c -I include -D LEVEL=2 -D NDEBUG \
  --target darwin-arm64 -o program
```

A `.py2cbin` file is a versioned, checksummed container holding generated C; it
is not an executable. `compile-via-c` is a literal zero-toolchain bridge for
the signed-64-bit intersection: it emits the C text, lexes and parses that text
with py2bin's standard-library-only implementation, reconstructs verified
integer semantics, and then uses the same IR and PE/ELF/Mach-O writers as
`compile`. Neither command calls a system assembler, linker, or C compiler.

The canonical C that `compile-via-c` round-trips is deliberately small: it is
exactly the shape py2bin's own generator emits — `long long` functions and
locals, `int main(void)`, integer `+`, `-`, `*`, bitwise operations,
constant-count shifts, comparisons, conditional expressions, assignments,
structured `if`/`while`, py2bin's canonical `for range` form, function calls,
returns, and newline-terminated `printf` with literal text or compile-time
integer `%lld` values, plus `extern` prototypes for the vetted adapter ABI and
*opaque* pointer handles that are never dereferenced.

`compile-c` is the real C compiler and accepts considerably more (see
[`src/py2bin/c_frontend.py`](src/py2bin/c_frontend.py)):

- every integer type — `char`, `short`, `int`, `long`, `long long`, their
  `unsigned` forms and the `<stdint.h>` fixed-width names — with C's integer
  promotions, usual arithmetic conversions, and exact truncation and
  sign/zero extension on assignment, cast, and narrow-typed arithmetic;
- local arrays (including multi-dimensional), `&x`, `*p`, `a[i]`, pointer
  arithmetic and pointer difference, with loads and stores at the real widths;
- `float` and `double`: decimal and hexadecimal floating constants,
  arithmetic, IEEE comparisons (every ordering is false when an operand is a
  NaN), the usual arithmetic conversions across the integer/floating boundary,
  and conversions and casts in both directions. A `float` object really
  occupies four bytes, so `sizeof`, array striding, and struct offsets are what
  C requires. Every floating expression is **evaluated** in double precision
  and rounded to `float` only where C says the extra precision must go —
  assignment, cast, argument passing, and return — which is
  `FLT_EVAL_METHOD == 1` (C11 6.3.1.8p2). A floating argument or result crosses
  a call as its IEEE bit pattern in an integer register; that calling
  convention is py2bin's own and never meets a platform C function.
  `long double` is rejected rather than quietly aliased to `double`;
- casts between integer types, between pointer types, and across the two, and
  `sizeof` for every complete type;
- `++`/`--` (prefix and postfix), `?:`, short-circuit `&&`/`||`, the comma
  operator, compound assignment, and signed/unsigned `/` and `%`;
- `if`/`else`, `while`, `do`/`while`, `for`, `switch`/`case`/`default` with
  fallthrough, `break`, `continue`, `goto` with labels, and `return`;
- functions. On `darwin-arm64` and `linux-arm64` a call is a **real machine
  call**: an AAPCS64 frame with a saved link register, arguments in `x0`-`x7`,
  and a direct `bl`, so **recursion works** — direct, deep (thousands of
  frames), and mutual. On the targets whose encoder has no call ABI yet
  (both x86-64 targets and `windows-arm64`) a call is still **inlined** at its
  site and recursion is rejected there with a file:line:column error;
- `printf` with real runtime formatting (`%d %i %u %x %X %c %s %f %F %e %E
  %g %G %%` with `h`/`hh`/`l`/`ll`/`z`, and a precision up to 120 on the
  floating conversions). The floating conversions are **exact**, not
  approximate: every finite double is a finite decimal, and the emitted code
  builds that decimal digit by digit and then rounds the digits half to even,
  so `printf("%.0f", 1e300)` produces the whole 301-digit integer that double
  equals. Flags and field widths are rejected by name rather than ignored;
- a **real preprocessor** ([`src/py2bin/c_preprocessor.py`](src/py2bin/c_preprocessor.py)):
  line splicing, comments, `#define` of object-like, function-like and
  variadic macros with `#` and `##`, `#undef`, `#include` of files it finds on
  the `-I` search path or beside the file that included them, `#if`/`#ifdef`/
  `#ifndef`/`#elif`/`#else`/`#endif` with a real 64-bit constant-expression
  evaluator and `defined`, `#error`, `#pragma once`, `-D` on the command line,
  and the predefined `__FILE__`, `__LINE__`, `__STDC__`, `__STDC_VERSION__`,
  `__STDC_HOSTED__` and `__py2bin*__` macros. `__DATE__` and `__TIME__` are the
  fixed `"Jan  1 1970"` and `"00:00:00"` that C11 6.10.8.1 permits when the date
  of translation is unavailable, so a build stays reproducible. Expansion follows the standard's
  algorithm with hide sets, and reproduces the expansions C11 6.10.3.3 and
  6.10.3.5 print, token for token. Arguments are substituted **textually**, so
  an argument written once in a replacement list is evaluated once at run time
  and one written twice is evaluated twice, exactly as C requires.
  `<stdio.h>`, `<stdlib.h>`, `<string.h>`, `<stdint.h>`, `<stddef.h>`,
  `<stdbool.h>`, `<limits.h>`, `<math.h>` and `<inttypes.h>` are served from
  py2bin's own built-in copies, which carry the macros of those headers
  (`INT_MAX`, `NULL`, `bool`, `PRId64`, ...) because the types and `printf`
  they declare are built into the compiler.

py2bin's C is LP64 on all six targets and gives plain `char` the signed
meaning Apple and x86 give it, so one source file produces the same values
everywhere. It rejects — with a file:line:column error, never an
approximation — `long double`, variadic user functions, a function type with
C's unspecified `()` parameter list, `static` inside a block, `extern` objects
(one translation unit, no linker), functions with more than eight parameters
(py2bin passes arguments only in registers), recursion, function pointers or
file-scope variables on the targets with no call ABI or static block, `#line`,
`#warning`, any `#pragma` other than `once`, the GNU `, ## __VA_ARGS__`
extension, a header it cannot find (there is no system include path, because a
real system header uses extensions this compiler does not have), arbitrary
libc, Cython-generated C, the NumPy C-API, C++, ATen, and CUDA.
The CPython C-API is accepted only as the fixed vetted symbol table
described under [What the C-API path supports](#what-the-c-api-path-supports),
not as a general C-API compiler: py2bin never reads `Python.h`, so anything
that is a macro, a `static inline`, a struct field, or variadic is out of
reach. The broader `emit-c` frontend can still emit constructs that require a
conventional C implementation; acceptance by `emit-c` alone does not promise
acceptance by `compile-via-c`. `emit-c`, `compile-via-c`, and `plan-c` are the
CPython-free portable-C route and reject any program that imports
`py2bin.cabi`; the C-API path is reached through `compile` or `compile-c`.

`plan-c` returns `c-source` when direct C translation is safe and
`cpython-bundle` when imports or unsupported Python semantics require the
compatibility path. Programs importing `manim`, `torch`, `transformers`, `bpy`,
or `webview` therefore keep their real CPython and native-extension behavior
rather than receiving incomplete generated C.

## Zero-toolchain contract

Native compilation requires only Python 3.10+ and this repository/package on
the build machine. It does not inspect or invoke `as`, `ld`, `clang`, `gcc`,
Visual Studio, Xcode, a target SDK, Docker, a virtual environment, or a target
Python. `compile-all` cross-compiles every implemented target from the same
process. Its generated native artifacts require only the matching operating
system and CPU.

This guarantee does not create missing inputs. If an application imports a
third-party library, that library's source, wheel, or adapter payload must be
supplied to py2bin. Native libraries such as Torch and `bpy` contain
target-specific compiled code; future standalone framework mode will consume
provided wheels/runtime archives directly without installing them into the
build environment.

## Fetching a target runtime and target wheels

A cross-target compatible bundle needs two things a foreign build machine does
not have: the target's CPython runtime and target-compatible wheels. Both are
published, so `freeze --auto-fetch` retrieves them instead of failing:

```sh
# Build a Windows x86-64 app from macOS or Linux. No Wine, no Windows machine,
# no pip: py2bin downloads python.org's embeddable CPython and the PyPI wheels.
py2bin freeze app/main.py --source-root app \
  --target windows-x86_64 --auto-fetch \
  --fetch-map webview=pywebview --fetch-map PIL=pillow \
  --fetch-lock py2bin-fetch.lock.json \
  --app --name MyApp --icon icon.ico -o dist/MyApp --clean
```

Downloads use `urllib.request` and `zipfile` from the standard library. pip,
setuptools, and virtualenv are never invoked, so the dependency-free guarantee
is unchanged. Every download is credential-free HTTPS, may not redirect off
HTTPS, is size-capped, is verified against the SHA-256 that PyPI or python.org
publishes for it, and is cached content-addressed so a rebuild re-uses bytes it
already verified. Archives are extracted with the traversal, link, and
special-file rejection the source fetcher already uses.

`--fetch-lock` records the URL and digest of every fetched file. When the lock
exists it becomes authoritative: a changed artifact is an error, not a silent
upgrade. State the integrity model exactly — the first fetch trusts the digest
the index serves over HTTPS, and the lock makes every later build reproducible
and auditable.

py2bin never guesses a PyPI project from an import name. `import webview` comes
from `pywebview` and `import PIL` from `pillow`; import names and project names
are different namespaces, and a wrong guess would install an unrelated or
hostile package. A bare import with no wheel therefore stops the build until
`--fetch-map IMPORT=PROJECT` names the project explicitly.

Two cases still require a supplied wheel, and both say so precisely:

- a project that publishes only a source distribution, because py2bin does not
  execute `setup.py` or a build backend; and
- a project with no wheel for the requested interpreter, ABI, or platform.

`--auto-fetch` currently retrieves Windows runtimes, because python.org
publishes the embeddable distribution for `windows-x86_64` and
`windows-arm64`. Other targets still need `--runtime-pack`; wheel fetching
works for every target.

## Dependency collection

`py2bin` parses imports without executing the application, maps top-level
packages to installed distributions, and copies every file declared by those
distributions. Dynamic imports can be declared explicitly:

```sh
py2bin build render.py -o dist/render --include manim --include torch
```

Dependency modes are:

- `closure` (default): imported distributions plus their installed dependency
  closure.
- `imported`: only directly imported distributions.
- `none`: project source only.

Use `--exclude MODULE` for optional backends you do not ship. `analyze` returns
exit status 1 when it sees an unresolved import.

### Wheel and prebuilt-Cython pipeline

`py2bin wheel` creates a wheel using only the standard library. The input is an
already-staged tree in the layout that should be installed into
`site-packages`. Python files, package data, and already-built Cython/native
extensions are stored byte-for-byte. `METADATA`, `WHEEL`, `top_level.txt`, and
the SHA-256 `RECORD` are generated by py2bin.

Pure Python example:

```sh
py2bin wheel build/package-root -o dist/wheels \
  --name example-package --version 1.0
```

Prebuilt Windows CPython 3.11 x86-64 Cython extension:

```sh
py2bin wheel build/windows-cp311 -o dist/wheels \
  --name example-native --version 1.0 \
  --python-tag cp311 --abi-tag cp311 --platform-tag win_amd64
```

Native payloads cannot use `py3-none-any`; exact interpreter, ABI, and platform
tags are mandatory. A `.pyx` or `.pxd` file may be included as source data, but
the command reports that it was not compiled. Supply the corresponding
target-built `.pyd` or `.so` when runtime importability is required.

Feed the created wheel directly into the compatible bundle:

```sh
py2bin freeze app/main.py --source-root app -o dist/App \
  --runtime-pack runtimes/windows-cp311-amd64 \
  --target windows-x86_64 \
  --wheel dist/wheels/example_native-1.0-cp311-cp311-win_amd64.whl \
  --icon icon.ico

# Output: dist/App.exe
```

This pipeline does not invoke Cython, a C compiler, or a linker. If Cython is
used, its output must be built before `py2bin wheel`, normally once per target.
Implementing a machine-code backend for arbitrary Cython-generated C would
require a complete C preprocessor/compiler, target ABI, object linker, CPython
C API, and dynamic loader; the current handwritten backend does not claim
those components.

For arbitrary CPython packages, freeze the interpreter and complete package
trees into a target-side bundle:

```sh
PYTHONPATH=src python3 -m py2bin freeze app/main.py \
  --source-root app --output dist/MyApp \
  --include torch --include transformers --clean

./dist/MyApp.bin
```

`freeze` carries the current compatible CPython runtime, standard library,
native extension modules, distribution metadata, and package data. It can also
consume wheels directly without pip or installation:

```sh
py2bin freeze app.py -o dist/App --wheel wheels/custom_backend.whl
```

Frozen bundles are specific to the build runtime's OS, CPU, Python ABI, and
accelerator variant. By default, the unpacked runtime tree is compressed behind
a handwritten native launcher in one `.bin` or `.exe`. `--onedir` keeps that
tree unpacked for inspection and debugging. Cross-target compatibility builds
require an explicit matching runtime pack and complete target-wheel closure.
Dynamic imports still need `--include`.

Windows one-file outputs accept an `.ico` directly. py2bin writes the PE
resources itself; no resource compiler is invoked. For `--app`, it replaces
the inherited Python icon and version information on both the outer one-file
launcher and the embedded app host. `--name` supplies the app/product
identity, and `--icon` supplies the executable icon:

```sh
py2bin freeze app.py -o dist/App --target windows-x86_64 \
  --runtime-pack runtimes/windows-cp311-amd64 \
  --wheel-dir wheels/windows-cp311 --icon icon.ico --app --clean
```

### Windows GUI identity and startup

For Windows, `--app` makes both executable layers GUI-subsystem PE files and
uses the runtime pack's `pythonw.exe` as the inner app host. It suppresses
console windows for the extractor and compatible CPython process. Omit `--app`
for a console program.

| What Windows sees | Implementation | Identity |
|---|---|---|
| Distributed `NAME.exe` | Handwritten py2bin PE launcher carrying the ZIP payload | `NAME`, app version resource, and `--icon` |
| Cached/running `NAME.exe` | Renamed target `pythonw.exe` that loads the bundled CPython DLL | The same `NAME`, version resource, and `--icon`; inherited Python branding is removed |
| Loaded `python3XY.dll` and native packages | Bundled CPython, `.pyd`, and dependency DLLs | Still visible to diagnostic tools because this is the compatibility path |

Before application code creates a window, the bootstrap assigns an explicit
`PythonToBinary.NAME` Windows AppUserModelID using the exact wide-string API
signature. Microsoft documents that this ID identifies the process to the
taskbar and should be set during initial startup before UI is shown:
[AppUserModelIDs](https://learn.microsoft.com/en-us/windows/win32/shell/appids)
and
[`SetCurrentProcessExplicitAppUserModelID`](https://learn.microsoft.com/en-us/windows/win32/api/shobjidl_core/nf-shobjidl_core-setcurrentprocessexplicitappusermodelid).

A GUI toolkit can deliberately override its window icon. A previously pinned
`.lnk` shortcut can also retain its own old icon or AppUserModelID; py2bin does
not rewrite existing Windows shortcuts, so unpin and re-pin the rebuilt
executable when checking a new identity build.

These changes fix Windows presentation and grouping, not compilation mode. A
frozen Manim/Torch/pywebview application still executes bundled CPython. Seeing
`python3XY.dll` in a module list is therefore expected and is not equivalent to
the taskbar incorrectly identifying the application as Python.

The generated isolated-runtime path includes the matching embedded
standard-library archive, such as `python312.zip`; target-compatible runtime
packs must supply that archive or an equivalent `Lib` tree. py2bin rewrites
both the executable-specific and versioned-DLL `_pth` files because Windows
CPython can give the latter priority.

For the fastest first launch of a large Windows GUI, use the unpacked app
layout. It starts `pythonw.exe` directly and performs no one-file extraction:

```sh
py2bin freeze app.py -o dist/App --target windows-x86_64 \
  --runtime-pack runtimes/windows-cp311-amd64 \
  --wheel-dir wheels/windows-cp311 --app --onedir --compact --clean

# Start dist/App/App.exe and distribute the complete dist/App directory.
```

This is a distribution tradeoff, not a different compilation mode. A one-file
build must materialize native `.pyd` and `.dll` files before Windows can load
them; packages such as Torch and `bpy` can therefore make the first launch
substantially slower. The content-addressed one-file cache makes later launches
reuse the extracted tree. The one-file launcher passes its path and original
command line directly to its extraction process instead of querying WMI/CIM.
On a cached Windows launch, the extractor performs one direct
`System.IO.File.Exists` marker check before creating any mutex; the named mutex
and its second race-prevention check are used only on a cache miss. Path,
delete, move, and process-start operations use direct .NET APIs instead of
PowerShell filesystem-provider cmdlets.
The generated bootstrap embeds its entrypoint and AppUserModelID at build time,
avoiding per-launch JSON and `pathlib` work; traceback support is imported only
after an application error. ZIP level 6 is used as the default size/extraction
balance.

On macOS, `freeze --app` wraps the one-file payload in the directory structure
required by the Apple `.app` format. The runtime archive is embedded in
`Contents/MacOS/NAME`; there is no unpacked `Contents/Resources/bundle`.
Both `darwin-arm64` and `darwin-x86_64` are supported, and an Intel app can be
cross-built from Apple Silicon (or the reverse) with a matching
`--runtime-pack` and target wheels — py2bin writes the launcher bytes itself
and never runs Rosetta, Wine, or an emulator. The two differ in signing:
arm64 macOS refuses to load an unsigned executable, so the arm64 launcher
embeds an ad-hoc signature sealing `Info.plist` and the resource hashes.
Intel macOS still loads unsigned executables, so the x86-64 launcher is emitted
unsigned. It runs locally, but it is neither signed nor notarized, so Gatekeeper
will quarantine it if it is downloaded rather than built on the machine.
`--icon` accepts ICNS, a square PNG at a standard icon size, or a
PNG-backed multi-resolution Windows ICO. ICO-to-ICNS conversion is implemented
in pure Python and skips only ICO sizes, such as 48×48, that have no matching
modern ICNS record:

```sh
py2bin freeze app.py --source-root . -o dist/MyApp \
  --app --name MyApp --icon icon.ico --include webview --compact --clean

dist/MyApp.app/Contents/MacOS/MyApp
```

A dependency-free generic resource/argument example is included for smoke
testing independently of large frameworks:

```sh
py2bin freeze examples/generic_app/main.py \
  --source-root examples/generic_app \
  -o dist/GenericApp --app --compact --clean

dist/GenericApp.app/Contents/MacOS/GenericApp hello
```

The generated `Info.plist` declares `AppIcon.icns`, and the icon is stored in
`Contents/Resources`. A frozen app does not require an installed target Python,
but third-party extensions still have to match the build OS, CPU, and Python
ABI.

The macOS app entry in `Contents/MacOS` is a directly executable Mach-O, not a
shell script. py2bin writes its instructions, ad-hoc code signature, resource
seal, and app metadata itself. The launcher starts the embedded CPython runtime
for full-library compatibility; it does not claim that a dynamic pywebview or
Manim application has been translated into the narrow native Python subset.

`--compact` and its descriptive alias `--optimize-size` omit distribution
tests, loose bytecode caches, CPython build/debug support, and documented
GUI/demo modules that are not used by a typical pywebview app. The policy now
applies consistently to installed distributions, supplied wheels, supplied
runtime-pack trees/ZIPs, and the Windows embedded standard-library ZIP. Runtime
`.pyc` members inside that standard-library ZIP are preserved.

On the retained CPython 3.11.9 Windows x86-64 runtime pack, this reduced
21,665,688 bytes to 20,588,888 bytes: 1,076,800 bytes, or 4.97%. Across the
72-wheel heavy-library reference closure, 2,412 test/cache files represented
38,198,995 uncompressed bytes and 9,547,608 bytes in the original compressed
wheels. A minimal real-runtime Windows GUI one-file comparison decreased from
11,539,423 to 10,899,205 bytes, saving 640,218 bytes or 5.55%. Final savings
vary with the application because py2bin recompresses the complete payload.

Leave size optimization off when the packaged program imports package test
suites, `tkinter`, `unittest`, `lib2to3`, or CPython build configuration files
at runtime.

### Whole-library AOT boundary

The intended native pipeline classifies dependency content instead of
pretending every file is the same language:

- supported pure-Python functions become py2bin IR and handwritten target
  instructions;
- existing `.pyd`, `.so`, `.dll`, `.dylib`, C/C++/Rust/CUDA engines remain
  target-native binaries and need a future non-CPython adapter ABI when they
  currently expose only the CPython extension API;
- HTML, CSS, JavaScript, shaders, fonts, models, templates, and other data stay
  assets because converting data to CPU instructions would be meaningless;
- unsupported dynamic Python is rejected by strict native mode. Compatible
  `bundle` mode still carries CPython and must not be described as complete
  AOT.

C source is an optional readable intermediate, not machine code. `emit-c`
stops at C text. `compile-via-c` proves and compiles only the documented
canonical integer intersection by parsing that text with py2bin itself;
`compile-c` runs py2bin's own C compiler — preprocessor included — over the
wider integer-and-pointer language described above. C that needs a system
header, or a language feature this compiler does not implement, still needs an
external C implementation, which py2bin never invokes silently. Strict native
compilation always ends in py2bin IR and handwritten target instructions.

HTML and CSS remain declarative data, and browser JavaScript remains code for a
JavaScript/browser engine. Compatible one-file bundles copy, compress, and
embed `.html`, `.css`, `.js`, `.mjs`, `.cjs`, and `.wasm` files inside the
executable payload. `py2bin-freeze.json` records their relative paths, kinds,
sizes, and SHA-256 hashes. This makes them integrity-checked binary payload
data; it does not mislabel HTML/CSS/JS as CPU instructions.

Consequently there is not yet a truthful switch that converts an arbitrary
Torch/Transformers/Manim/`bpy`/pywebview application and its entire dependency
closure into CPython-free py2bin machine code. The restricted function
inlining above is real progress toward compiling pure-Python glue, while the
adapter ABI and a substantially broader Python object runtime remain future
work.

Use `assemble --mode native --source-root PROJECT` as the strict gate when an
artifact must contain only py2bin-generated application instructions. It
fails on the first unsupported dependency construct and never falls back to
the CPython-compatible bundle.

## Heavy-library compatibility

Run `py2bin capabilities` for the catalog below, or
`py2bin capabilities APP.py --json` to audit the imports and native-subset
result of one entry file without importing or executing it.

| Import/project | Fully translated by `compile`? | Can `freeze` carry it? | What is still required |
|---|---:|---:|---|
| NumPy / SciPy / pandas / scikit-learn | No | Conditional | Matching CPython ABI and complete target wheels/native libraries |
| PyTorch / TorchVision | No | Conditional | Target wheels, C++ libraries, accelerator variant, and target GPU driver when used |
| TensorFlow / JAX | No | Conditional | Supported target runtime wheels; JAX also needs matching `jaxlib` |
| Transformers | No | Conditional | CPython, backend such as Torch, dependency closure, model/config/tokenizer files |
| `tokenizers` | No | Conditional | Its Rust extension compiled for the target CPython ABI |
| Manim | No | Conditional | CPython, target wheels, fonts/assets, and media tools such as FFmpeg; LaTeX when used |
| Matplotlib | No | Conditional | Target wheels, NumPy, rendering backend, fonts, and package data |
| Pillow / OpenCV | No | Conditional | Target extension wheels and their native image/media/GUI libraries |
| Blender `bpy` | No | Conditional | Exactly compatible Blender/`bpy`, CPython ABI, resources, OS, and CPU |
| pywebview (`webview`) | No | Conditional | CPython, target dependencies, and the OS webview framework/runtime |
| Gradio / Streamlit | No | Conditional | CPython server packages and frontend assets plus a browser/webview |
| Numba / llvmlite | No | Conditional | Mutually compatible target wheels, including the LLVM components |
| Requests / Flask / Django / FastAPI | No | Conditional | CPython, dependencies, and application templates/static/configuration data |
| Unknown third-party or local import | No by default | Conditional | Its real implementation and complete target-compatible dependency/data closure |

The NumPy and PyTorch rows describe the real packages. `compile` rejects those
imports outright rather than reimplementing them, so neither row is ever “Yes”
for native compilation; they are collected only through `freeze`/`bundle`.

“Conditional” means the necessary files can be collected when they are
actually supplied and compatible. It does not mean every version exists for
every OS, architecture, or CPython ABI, and it does not mean py2bin has
translated the package into its own machine code.

“Supports all libraries” means the collector is generic and does not maintain a
hardcoded allowlist. It cannot guarantee that every third-party binary, driver,
external executable, license, network model, or platform service is portable.
It also cannot make mutually incompatible third-party requirements coexist in
one Python environment. For example, some current Manim and `bpy` releases
require incompatible NumPy major versions; use compatible releases or separate
runtime sidecars rather than bypassing package constraints.

Native compilation of those libraries is a different problem: Torch contains
millions of lines of precompiled C++/CUDA code, `bpy` is coupled to Blender, and
Manim invokes external tools. They can be made self-contained per target by
shipping their native components and a compatible embedded runtime, but they
cannot truthfully become one CPU-independent executable. The long-term native
API is an adapter ABI: pure-Python modules compile through py2bin IR, while
large native libraries link as target-specific prebuilt components.
Until that adapter ABI is complete, `freeze` is the full-compatibility engine.

## Comparison with dante-biase/py2bin

This project is unrelated to
[dante-biase/py2bin](https://github.com/dante-biase/py2bin). That repository
describes itself as a streamlined PyInstaller interface, declares
PyInstaller 3.6, Click 7.1.1, and py2x 1.0 as dependencies, and invokes
PyInstaller's `--onefile` mode in a subprocess. Its README lists macOS as its
compatibility target.

| Capability | This project | dante-biase/py2bin |
|---|---|---|
| Implementation dependencies | Python standard library only | PyInstaller, Click, and py2x |
| Actual Python-to-machine-code path | Yes, for the documented small static subset | No; it delegates packaging to PyInstaller |
| Arbitrary-package compatibility path | Embedded-CPython `freeze` bundle | PyInstaller one-file bundle |
| Direct binary writers | ELF, PE, and Mach-O; x86-64 and ARM64 | None in that wrapper |
| Installed Python required on target | No for `compile` or `freeze` | No for the produced PyInstaller bundle |
| Arbitrary Python becomes native machine code | No | No |

The fair comparison is therefore `freeze` versus its PyInstaller wrapper for
application compatibility, and `compile` versus a real compiler for native
translation. Calling either project's compatibility bundle a complete native
translation would be inaccurate.

## PPCI relationship

The pipeline is inspired by [windelbouwman/ppci](https://github.com/windelbouwman/ppci):
keep parsing, IR, instruction selection, linking, and file formats separate.
Unlike simply wrapping PPCI, this repository starts with its own narrow Python
frontend and direct executable writers so every emitted byte is controlled and
the dependency-free guarantee remains testable. PPCI's own documentation calls
its Python-to-IR support preliminary; this project therefore treats broad
Python compatibility as staged compiler work, not an already-solved claim.

## Runtime behavior

Single-file artifacts extract into a content-addressed cache before execution.
Set `PY2BIN_CACHE_DIR` to control its location. The app can read
`PY2BIN_BUNDLE_ROOT` to locate bundled resources. Rebuilding changes the cache
fingerprint; deleting the cache is safe when no bundled program is running.

“One file” describes distribution, not execution without extraction. The first
launch atomically expands the embedded runtime; later launches reuse it.
Windows launchers use the Windows PowerShell/.NET ZIP facilities included with
normal Windows 10/11 installations. The handwritten PE passes its own path and
original Unicode command line through the child environment, avoiding WMI/CIM
process queries. Cached Windows launches check the completion marker before
allocating the extraction mutex; first-run filesystem work uses direct
`System.IO` methods. Linux and macOS launchers use `/bin/sh` plus `tail`,
`head`, and `tar` from the base operating system. Extremely minimal Windows or
Linux images that remove those OS facilities need `--onedir`.

A macOS `.app` can never literally be one filesystem file because Apple defines
it as a directory bundle. py2bin minimizes it to the native executable carrying
the compressed payload, `Info.plist`, the code-resource seal, and optional
icon.

## Development

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The implementation intentionally depends only on the standard library. The
unit suite performs no package downloads and writes only inside its selected
temporary directory:

```sh
TMPDIR=/path/on/your/data/disk \
  PYTHONPATH=src python3 -m unittest discover -s tests -v
```

GitHub Actions is disabled for this repository at two levels: active workflow
files are absent from `.github/workflows/`, and the repository Actions
permission is disabled. Reference definitions remain inert under
`.github/workflows-disabled/`. Do not move them or re-enable repository Actions
unless the owner explicitly changes this policy.

This repository is standalone. It does not modify or depend on CodeBench or
`python-ios-lib`; those projects can consume a future release as an ordinary
package or copied source dependency when their platform integration is ready.
