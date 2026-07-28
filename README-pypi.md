# python-to-binary

`py2bin` turns Python into machine code using nothing but the Python standard
library. No Cython, Nuitka, mypyc, Rust, C, C++, PyInstaller, PPCI, bootloader,
assembler, linker or SDK - and no `gcc` or `clang` at any point. The only thing
a build needs is an interpreter.

```sh
pip install python-to-binary
py2bin compile-capi app.py --target darwin-arm64 -o app
```

Source, issues and the full documentation:
**https://github.com/yu314-coder/python_to_binary**

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
CPython's. macOS agrees on 878 and differs on 7; a 100-program slice run
through Wine agrees on 93 and differs on 5. The differences are the same on
every platform and are inherent rather than open - CPython's "Did you mean"
needs a Python frame to suggest from, and the repr of a compiled function
really is a builtin function's.

The **native** tier (`py2bin compile`, no CPython at all) targets all six.

## The paths through it

**`compile-capi`** translates ordinary Python into C that drives the CPython
C API, then compiles that C with py2bin's own C compiler. This is the tier
Nuitka occupies, with Nuitka's dependency removed - Nuitka hands its generated
C to clang, and this compiles it itself. Your logic becomes machine code;
Python's object semantics stay in libpython, which the binary links.

**`compile`** removes CPython entirely: Python AST → py2bin IR → optimizer →
handwritten x86-64/ARM64 instructions → ELF, PE or Mach-O. Nothing is
interpreted and nothing is linked. The price is the smallest accepted subset.

**`freeze` / `bundle`** is compatibility packaging: your program shipped beside
an interpreter that runs it, which is how a project with NumPy or a GUI gets an
artifact at all. It is not translation.

## Using it

```sh
pip install python-to-binary
```

| command | what it does |
|---|---|
| `py2bin compile-capi` | Python → C driving the CPython C API → machine code |
| `py2bin compile` | Python → machine code, no CPython anywhere |
| `py2bin compile-c` | py2bin's own C compiler, on your C |
| `py2bin freeze` | ship the program beside an interpreter |
| `py2bin targets` | list the targets this build knows |

Bundling a real application into a macOS `.app` that carries its own
interpreter and packages:

```sh
py2bin compile-capi app.py --target darwin-arm64 \
  --app --name "My App" --icon icon.icns \
  --embed-python --site ../Resources/site-packages \
  --bundle-site /path/to/venv/lib/python3.14/site-packages \
  --prune-unused --zip-stdlib \
  -o dist/MyApp.app --clean
```

## How it works

Nothing wraps a toolchain; each stage is a module you can read.

```
capi_emit.py        Python AST  ->  C that calls the CPython C API
capi_ints.py          which locals may live in a machine register
c_preprocessor.py   #include, macros, conditionals
c_frontend.py       C  ->  py2bin IR
native/ir.py        the IR itself
native/optimizer.py constant folding, dead code, write merging
native/arm64.py     IR  ->  ARM64 instructions
native/x86_64.py    IR  ->  x86-64, System V and Microsoft x64
native/formats/     Mach-O, PE32+, ELF
freezer.py          bundling: interpreter, packages, pruning, archives
cabi.py             the vetted CPython entry points
```

So `compile-capi` is five stages, all of them in this package: `capi_emit` →
`c_preprocessor` → `c_frontend` → `native.x86_64`/`native.arm64` →
`native.formats.macho`/`pe`.

There is no `import ctypes` anywhere on that path, which a test asserts by
compiling in a fresh interpreter and listing what got loaded. `ctypes` is
standard library and would pass an imports-only-stdlib check, but it pulls in
`subprocess` - and there are Pythons where a subprocess is not something a
program may have.

## Measured against Nuitka

manim_app: 10,100 lines, pywebview + Pillow + pyobjc, built both ways on the
same machine.

| | py2bin | Nuitka |
|---|---|---|
| whole `.app` | **61.2 MB** | 72.6 MB |
| main binary | **9.2 MB** | 29.6 MB |
| bare interpreter start | **9.5 ms** | 16.2 ms |
| start with the app's imports | 52.1 ms | **44.8 ms** |
| compile time | **16.7 s** | minutes |

Run time, median of 5, seconds:

| workload | py2bin | CPython | Nuitka |
|---|---|---|---|
| integer arithmetic | **0.050** | 0.084 | 0.095 |
| `while` loop | **0.045** | 0.070 | 0.045 |
| nested loops | **0.022** | 0.035 | 0.042 |
| function calls | 0.065 | 0.023 | **0.020** |
| string building | 0.025 | 0.011 | **0.009** |

Loops beat both because a local the analysis picks out is held in a register
rather than on the heap, with the overflow check that falls back to unbounded
arithmetic when it leaves the word. Calls still lose: an argument is boxed at
the call and unboxed inside, where the interpreter's specialised call pays
neither.

## Licence

MIT. Full documentation, source and issues:
**https://github.com/yu314-coder/python_to_binary**
