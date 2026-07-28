# python-to-binary

`py2bin` turns Python into machine code using nothing but the Python standard
library. No Cython, Nuitka, mypyc, Rust, C, C++, PyInstaller, PPCI, bootloader,
assembler, linker or SDK - and no `gcc` or `clang` at any point. The only thing
a build needs is an interpreter.

```sh
pip install python-to-binary
py2bin compile-capi app.py --target darwin-arm64 -o app
```

## The paths through it

Three ways to turn a program into an artifact. They trade the same three things
against each other, and which one you want depends on which you care about.

| | `compile` | `compile-capi` | `freeze` |
|---|---|---|---|
| **speed** on a 30M-iteration loop | **0.062 s** | 1.27 s | 0.66 s |
| **artifact** | **48 KB** | 66 KB | tens of MB |
| **needs Python on the machine?** | **no** | yes, or bundle it | no, it carries one |
| **how much Python works** | a small subset | most of it: 878 of an 889-program corpus | **everything** |
| **third-party packages** | none | any the interpreter can import | **carried inside** |
| what actually runs your logic | machine code | machine code | CPython, interpreting |

**`compile` is the fastest and the smallest.** Python AST → py2bin IR →
optimizer → handwritten x86-64/ARM64 → ELF, PE or Mach-O. There is no
interpreter in the artifact and none on the machine: 11× faster than CPython on
that loop, in 48 KB that runs on a bare system. You pay for it in what it will
accept - integers, floats, strings, control flow, your own functions - and it
will not import a package at all.

**`freeze` is the most complete.** It ships your program beside an interpreter
that runs it, so NumPy, Torch and a GUI toolkit all work exactly as they do
now. Nothing is translated, so nothing is faster; the artifact is the largest
of the three because an interpreter and every dependency are inside it.

**`compile-capi` is the middle, and the one under active work.** It translates
ordinary Python into C that drives the CPython C API, then compiles that C with
py2bin's own C compiler - the tier Nuitka occupies, with Nuitka's dependency on
clang removed. Almost the whole language goes through, and anything the linked
interpreter can import still works, so a real application with pywebview and
Pillow compiles. Integer loops beat CPython because their locals are held in
registers; most other operations are slower, because each one is a real C-API
call where the interpreter has specialised bytecode. See
[what it supports](#what-compile-capi-supports) for the per-feature table.

The loop above is deliberately unkind to `compile-capi`: its accumulator is
compared against a parameter, which the register analysis cannot claim, so the
fast path is off. On a loop it can claim, the same tier is 1.67× faster than
CPython.

## Platforms

What `compile-capi` - the tier that turns your program into machine code that
drives CPython - can target today.

| | x86-64 | arm64 |
|---|---|---|
| **macOS** | ✅ works | ✅ works |
| **Windows** | ✅ works | ⬜ future work |
| **Linux** | ⬜ future work | ⬜ future work |

Each working target is held to the same standard: an 889-program corpus is
compiled for it and every program's output and exit code compared against
CPython's. macOS agrees on 878 and differs on 7; a 100-program slice through
Wine agrees on 93 and differs on 5. The differences are the same ones on every
platform and are inherent rather than open: CPython's "Did you mean" needs a
Python frame to suggest from, the repr of a compiled function really is a
builtin function's, and `"v" is "v"` depends on an interning the compiler does
not reproduce.

The two remaining cells are the same shape of job as the three that are done.
Windows arm64 needs an encoder for the import-table call; Linux needs an ELF
`.got.plt` and its relocations.

The **native** tier (`py2bin compile`, no CPython at all) targets all six, and
**freeze** targets whatever it has a runtime pack for. This grid is about
`compile-capi` because that is the tier with the interesting constraint: it has
to bind an external interpreter through the platform's own dynamic linker.

## Using it

Installed:

```sh
pip install python-to-binary
py2bin compile-capi app.py --target darwin-arm64 -o app
```

From a checkout, which needs nothing installed at all:

```sh
git clone https://github.com/yu314-coder/python_to_binary.git
cd python_to_binary
PYTHONPATH=src python3 -m py2bin compile-capi app.py --target darwin-arm64 -o app
```

The two are the same program; `py2bin` is a console script and
`python3 -m py2bin` is the module. Everything below works either way.

| command | what it does |
|---|---|
| `compile-capi` | Python → C driving the CPython C API → machine code |
| `compile` | Python → machine code, no CPython anywhere |
| `compile-c` | py2bin's own C compiler, on your C |
| `freeze` / `bundle` | ship the program beside an interpreter |
| `aot-plan` / `aot-build` | refuse to build unless every operation is CPython-free |
| `targets` | list the targets this build knows |

## How it is put together

Nothing here wraps a toolchain; each stage is a module you can read.

```
src/py2bin/
  capi_emit.py        Python AST  ->  C that calls the CPython C API
  capi_ints.py          which locals may live in a machine register
  c_preprocessor.py   #include, macros, conditionals
  c_frontend.py       C  ->  py2bin IR (the integer and pointer language)
  native/
    ir.py             the IR itself
    optimizer.py      constant folding, dead code, write merging
    arm64.py          IR  ->  ARM64 instructions
    x86_64.py         IR  ->  x86-64 instructions, System V and Microsoft x64
    formats/
      macho.py        Mach-O, static and dyld-binding
      pe.py           PE32+, with a multi-DLL import table
      elf.py          ELF
  freezer.py          bundling: interpreter, packages, pruning, archives
  cabi.py             the vetted CPython entry points, callable from Python
  cabi_tables.py        which library each one lives in - no ctypes, so a
                        build never imports it
```

So `compile-capi` on a program is five stages, all of them here:
`capi_emit` → `c_preprocessor` → `c_frontend` → `native.x86_64`/`native.arm64`
→ `native.formats.macho`/`pe`.

## Bundling an application

One command turns a Python program into a macOS `.app` that carries its own
interpreter and its own packages:

```sh
py2bin compile-capi app.py --target darwin-arm64 \
  --app --name "My App" --icon icon.icns \
  --embed-python --site ../Resources/site-packages \
  --bundle-site /path/to/venv/lib/python3.14/site-packages \
  --prune-unused --zip-stdlib \
  -o dist/MyApp.app --clean
```

What each part does:

| flag | effect |
|---|---|
| `--app` | write a `.app` bundle rather than a bare executable |
| `--embed-python` | carry the interpreter, so the result runs on a Mac without Python |
| `--bundle-site DIR` | copy a virtualenv's packages into the bundle |
| `--site DIR` | where the program looks for them, relative to the executable |
| `--prune-unused` | drop modules the program cannot import, plus `.dSYM` debug companions |
| `--zip-stdlib` | pack the carried library into the `pythonXY.zip` the interpreter already reads |
| `--exclude MODULE` | drop something the static walk had to keep - see below |

`--exclude` is for what the walk cannot work out. Pillow is the case that
matters: `Image.init()` imports whatever plugin sits beside it, so a static
walk keeps them all, and each optional codec holds its native library alive.
Naming both halves drops the plugin, the extension, and the library behind it:

```sh
  --exclude PIL.AvifImagePlugin --exclude PIL._avif \
  --exclude PIL.ImageFont --exclude PIL._imagingft
```

That took 7 MB off the bundle below. What the program can then no longer do is
the caller's to judge - dropping `_avif` means an AVIF file stops opening.

### Measured on a real application

manim_app: 10,100 lines, pywebview + Pillow + pyobjc, built both ways on the
same machine and interleaved so that drift falls on both equally.

| | py2bin | Nuitka |
|---|---|---|
| whole `.app` | **61.2 MB** | 72.6 MB |
| main binary | **9.2 MB** | 29.6 MB |
| bare interpreter start | **9.5 ms** | 16.2 ms |
| start with the app's imports | 52.1 ms | **44.8 ms** |
| compile time | **16.7 s** | minutes |

Verified from a copy moved elsewhere on disk: every module the program imports
resolves, a pty opens and echoes, Pillow still round-trips PNG/JPEG/GIF/BMP/
WEBP/TIFF, and the app starts with no traceback.

### How the targets are reached

Every target binds its interpreter through the platform's own dynamic linker.
On macOS each architecture's encoder emits GOT and static reference sites and
one Mach-O writer lays out `__got`, the bind opcodes and `__DATA`; the two
architectures differ in four things and no more - the header's CPU, the page
segments align to, how `__text` states its alignment, and how a reference is
spelled. arm64 reaches an address in two instructions, a page and then an
offset into it; x86-64 uses one rip-relative displacement.

On Windows the import directory names several DLLs rather than one - the kernel
for process services, `msvcrt` for the C library half of the vetted ABI, and
`pythonXY.dll` for the interpreter. The Microsoft x64 call is not the System V
one with the registers renamed: System V allocates from two independent
counters, so the first pointer goes in `rdi` however many doubles precede it,
where Microsoft allocates by *position*. A float argument is copied into the
integer register as well, because a variadic callee reads it there, and the
caller reserves 32 bytes of shadow space regardless.

On all three, static storage lives in the image and is addressed PC-relatively,
never through a callee-saved register. A compiled closure handed to
`sorted(key=...)` is entered from inside CPython's own frames, and while those
frames are live that register holds CPython's value, not the program's. This
was found the hard way on arm64 (`x28`) and applies unchanged to `r15`.

Windows argument passing is the one that caught a real bug, and it is worth
naming because it is the kind that survives a structural check. Microsoft x64
does not take a prefix of System V's register order, it uses a different order:
the first argument is `rcx`, not `rdi`. The table here was System V's and the
Windows path took the first four entries of it, so every argument arrived two
registers out - which showed up as a `PyObject *` in `rdx` and an instruction
pointer somewhere in CPython's heap. It survived because nothing ran a Windows
image; the tests built one and read its structure, which is exactly what the
bug leaves intact.

An exe needs `pythonXY.dll` beside it or on the path. That is what you ship,
not something the compiler can settle.

## What `compile-capi` supports

Every row is checked by compiling it, running it, running the same source under
CPython, and requiring identical stdout and exit status.

| feature | |
|---|---|
| int, float, str, bytes, bool, None | ✅ |
| unbounded integers (`2 ** 200` exact) | ✅ |
| f-strings, format specs, `!r`/`!s`/`!a` | ✅ |
| list, tuple, dict, set, slicing, subscripts | ✅ |
| comprehensions and generator expressions | ✅ |
| `if` / `while` / `for` / `else`, `break`, `continue` | ✅ |
| chained comparison, ternary, `and` / `or` | ✅ |
| functions: defaults, `*args`, `**kwargs` | ✅ |
| lambdas and closures | ✅ |
| classes, `__init__`, methods, inheritance, `super()` | ✅ |
| dunder methods (`__repr__`, `__eq__`, …) | ✅ |
| decorators | ✅ |
| `try` / `except` / `finally`, `with` | ✅ |
| `import`, `from … import`, relative imports | ✅ |
| `global` / `nonlocal`, tuple unpacking | ✅ |
| the whole program: every `.py` beside the entry is compiled in | ✅ |
| `__name__`, `__file__`, `inspect.signature` on compiled functions | ✅ |
| `raise … from …` | ❌ |
| starred unpacking (`a, *b = …`) | ❌ |
| walrus (`:=`) | ❌ |
| generators (`yield`) | ❌ |
| `async` / `await` | ❌ |
| `match` statement | ❌ |

A refusal is a `file:line:col` error, never a silent approximation. On an
889-program corpus, 878 programs produce byte-identical output to CPython; the
7 that differ do so inherently (CPython's "Did you mean" needs a Python frame,
`"v" is "v"` depends on interning) and 4 are refused outright.

### The interpreter surface it may use

- A fixed table of 71 exported CPython entry points: `PyBytes_FromStringAndSize`, `PyCFunction_New`, `PyDict_New`,
  `PyDict_SetItem`, `PyErr_Clear`, `PyErr_ExceptionMatches`,
  `PyErr_GetRaisedException`, `PyErr_Occurred`, `PyErr_Print`,
  `PyErr_SetObject`, `PyErr_SetRaisedException`, `PyFile_WriteObject`,
  `PyFile_WriteString`, `PyFloat_AsDouble`, `PyFloat_FromDouble`,
  `PyImport_AddModule`, `PyImport_ImportModule`, `PyIter_Next`,
  `PyList_Append`, `PyList_New`, `PyLong_AsLongLong`, `PyLong_FromLongLong`,
  `PyLong_FromString`, `PyNumber_Add`, `PyNumber_And`,
  `PyNumber_FloorDivide`, `PyNumber_Invert`, `PyNumber_Lshift`,
  `PyNumber_Multiply`, `PyNumber_Negative`, `PyNumber_Or`,
  `PyNumber_Positive`, `PyNumber_Power`, `PyNumber_Remainder`,
  `PyNumber_Rshift`, `PyNumber_Subtract`, `PyNumber_TrueDivide`,
  `PyNumber_Xor`, `PyObject_Call`, `PyObject_CallNoArgs`,
  `PyObject_CallOneArg`, `PyObject_DelItem`, `PyObject_GetAttrString`,
  `PyObject_GetItem`, `PyObject_GetIter`, `PyObject_IsTrue`,
  `PyObject_Repr`, `PyObject_RichCompare`, `PyObject_SetAttrString`,
  `PyObject_SetItem`, `PyObject_Size`, `PyObject_Str`,
  `PyObject_Vectorcall`, `PyRun_SimpleString`, `PySequence_Contains`,
  `PySlice_New`, `PySys_GetObject`, `PySys_WriteStdout`, `PyTuple_GetItem`,
  `PyTuple_New`, `PyTuple_Pack`, `PyTuple_SetItem`, `PyUnicode_DecodeUTF8`,
  `PyUnicode_FromString`, `Py_DecRef`, `Py_EnterRecursiveCall`,
  `Py_Finalize`, `Py_IncRef`, `Py_Initialize`, `Py_IsInitialized`,
  `Py_LeaveRecursiveCall`
- Every one is a real exported function - not a macro, not a `static inline` -
  with a fixed count of word-sized arguments, and a test asserts each is
  exported by the running interpreter's dylib. That is why this compiler never
  reads `Python.h`: the preprocessor could include it, but what is inside is
  macros, inline functions and struct layouts it does not implement, so the
  table is written out instead.

### How fast each one is

300,000 iterations, median of 5, against the interpreter it links. Higher is
better; `1.00×` means the same speed as CPython.

| feature | py2bin | CPython | |
|---|---|---|---|
| integer arithmetic | **5.1 ms** | 8.5 ms | **1.67× faster** |
| comparisons | **3.8 ms** | 4.7 ms | **1.24× faster** |
| `while` loop | 7.5 ms | 6.5 ms | 0.87× |
| direct function call | 7.9 ms | 6.4 ms | 0.81× |
| comprehension | 5.2 ms | 4.0 ms | 0.77× |
| dict store | 10.9 ms | 7.8 ms | 0.72× |
| subscript | 7.1 ms | 3.9 ms | 0.55× |
| attribute read | 7.5 ms | 3.8 ms | 0.51× |
| exception raise/catch | 13.8 ms | 6.7 ms | 0.49× |
| closure call | 14.0 ms | 6.5 ms | 0.46× |
| f-string | 13.6 ms | 5.1 ms | 0.38× |
| float arithmetic | 10.6 ms | 3.4 ms | 0.32× |
| list append | 19.1 ms | 5.3 ms | 0.28× |
| string concatenation | 24.3 ms | 3.3 ms | 0.14× |
| instantiation | 177 ms | 16.3 ms | 0.09× |
| method call | 142 ms | 6.7 ms | 0.05× |

**Integer loops win** because a local the analysis picks out is held in a
machine register, with an overflow check that falls back to unbounded
arithmetic when the value leaves the word. That is what CPython's specialising
interpreter does, and doing anything less was what made this tier slower than
not compiling at all.

**Everything else loses, by a factor that tracks how many C-API calls the
operation costs.** Each one is a real call with the reference-count discipline
around it, where the interpreter's specialised bytecode does the same work
inline. Floats are not held in registers at all yet, which is the same job as
the integers and not done.

**Method calls and instantiation are far worse than the pattern predicts** -
21× and 11× rather than the 2-4× everything else pays. That is not the C-API
overhead; something in the class path is doing work per call that it should do
once. It is the clearest thing to fix next and it is measured here rather than
left out.

## Measured against Nuitka

Same machine (arm64 macOS), same CPython 3.14, same source. Nuitka 2.x with
`--standalone`, driving Apple's clang; this driving its own C compiler.

Run time, median of 5 runs, seconds:

| | this | CPython | Nuitka |
|---|---|---|---|
| integer arithmetic | **0.054** | 0.089 | 0.098 |
| `while` loop | **0.048** | 0.071 | 0.046 |
| nested loops | **0.023** | 0.037 | 0.043 |
| function calls | 0.069 | 0.024 | **0.021** |
| string building | 0.026 | 0.011 | **0.010** |

Startup, `print("x")`, median of 13 runs:

| | startup | on disk |
|---|---|---|
| this, `compile-capi` | **10.2 ms** | **52 KB** |
| CPython | 13.8 ms | - |
| Nuitka `--standalone` | 17.1 ms | 15 MB |

A `compile-capi` binary links the interpreter it was built against and starts
by calling `Py_Initialize`, skipping the interpreter's own startup path -
scanning `sys.path`, finding and unmarshalling `__main__`. Nuitka's standalone
bundle pays to bootstrap its own tree first.

**Loops are faster than the interpreter; calls are not.** The reason is worth
stating rather than hiding, because it explains both halves.

Since 3.11 CPython does not run the bytecode you wrote: it rewrites hot
instructions into specialised forms, so a `+` between two ints becomes an add
on two machine words with a type guard and never reaches `PyNumber_Add`.
Emitting a call to the generic entry point for every operation therefore
produces exactly the code the interpreter has learned to *avoid* - which is why
this tier was once 0.140 s against the interpreter's 0.089 on the first row of
that table.

So it now does what CPython does. A local the analysis picks out is held as a
machine integer, in a register, and arithmetic over such names is emitted twice:
once as machine instructions guarded on a flag, once as the C-API calls it
always was. Only the second arm can produce an integer wider than the word, so
every operation that can leave 64 bits carries the check that says it did and
falls back rather than answering wrongly - `3 ** 200` compiled this way is
still exact. `for i in range(...)` counts in a register, with one copy of the
loop body and a branch inside it rather than two loops, so the binary does not
grow in proportion to how much this helped. It grew 2%.

Calls still lose, and will until there is a calling convention that passes a
machine integer as a machine integer. Today an argument is boxed at the call
and unboxed inside, which costs two allocations where the interpreter's
specialised call costs none.

On a whole application - the manim_app figures under
[Bundling an application](#bundling-an-application) - the shape is the same.

The binary is a third the size because Nuitka compiles every module it reaches,
including the third-party tree, while this compiles the program and ships its
dependencies as bytecode. That does not make the bundle smaller by 20 MB,
because the bytecode has to go somewhere - and asking where the difference
went, category by category, is what found the rest:

| | py2bin | Nuitka |
|---|---|---|
| main binary | 9.2M | 29.6M |
| the interpreter | 6.9M | 7.6M |
| native extensions and their libraries | 30.0M | 22.6M |
| bytecode | 7.8M | - |
| the application's own web assets | 15.8M | 15.8M |

Both carry an interpreter; Nuitka's is `Contents/MacOS/Python` rather than a
framework. The 20 MB the binary saves comes back as 7.8 MB of bytecode and
7.4 MB of native libraries, and the rest was waste this found and now removes:
`--zip-stdlib` took the library from 8.4 MB to 3.5, and `--prune-unused` drops
the `.dSYM` debug companions that some wheels ship - pyobjc alone carries
3.6 MB of DWARF beside a 1 MB extension.

The native libraries were the last of it, and two things were wrong. A wheel
with native dependencies vendors them beside its extension rather than in the
bundle's library directory - Pillow puts nineteen in `PIL/.dylibs` - and the
closure only looked in `Contents/lib`, so it never considered them at all.
And a static walk has to keep every Pillow plugin, because `Image.init()`
imports whatever is beside it, so each optional codec kept its library.
`--exclude` is how a caller says what the walk cannot work out: naming
`PIL.AvifImagePlugin` and `PIL._avif` drops the plugin, the extension, and
then - now that vendored directories are closed over too - the 2.9 MB library
behind it. What the program can no longer do is the caller's to judge.

## Compiling asks for an interpreter and nothing else

There is no `import ctypes` anywhere on the path from Python source to machine
code, which a test asserts by compiling a program in a fresh interpreter and
listing what got loaded. `ctypes` is stdlib and so would pass an
imports-only-the-standard-library check, but it pulls in `ctypes.util` and
through it `subprocess` - and there are Pythons, the one on a phone among them,
where a subprocess is not something a program may have.

`py2bin.cabi` does use `ctypes`, because calling a C-API entry point from
Python is what it is for. What the compiler needs from it is only the table
saying which library each symbol lives in, and that table lives in
`py2bin.cabi_tables`, which imports nothing at all.

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

## Where else this is written down

The PyPI page carries the same grid, the same account of how the pieces fit,
and the install instructions on their own:
**https://pypi.org/project/python-to-binary/**. It is generated from
`README-pypi.md` in this repository, so the two cannot drift without the
drift being visible in a diff.

## Development

```sh
git clone https://github.com/yu314-coder/python_to_binary.git
cd python_to_binary
PYTHONPATH=src python3 -m unittest discover -s tests
```

1328 tests, no dependencies, nothing to install. The suite fails if any module
under `src/` imports `subprocess`, `multiprocessing`, `pty`, `distutils` or
`setuptools`, or names an external toolchain as a value - which is what keeps
the zero-toolchain claim honest rather than aspirational.
