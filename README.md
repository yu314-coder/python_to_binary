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

**`compile-capi`** translates ordinary Python into C that drives the CPython
C API, then compiles that C with py2bin's own C compiler. This is the tier
Nuitka occupies, with Nuitka's dependency removed - Nuitka hands its generated
C to clang, and this compiles it itself. Your logic becomes machine code;
Python's object semantics stay in libpython, which the binary links.

**`compile`** removes CPython entirely: Python AST → py2bin IR → optimizer →
handwritten x86-64/ARM64 instructions → ELF, PE or Mach-O. Nothing is
interpreted and nothing is linked. The price is the smallest accepted subset of
the three.

**`freeze` / `bundle`** is compatibility packaging: your program is shipped
beside an interpreter that runs it, which is how a project with NumPy or a GUI
gets an artifact at all. It is not translation, and this README does not
pretend otherwise.

Three supporting pieces are usable on their own: **`compile-c`**, py2bin's C
compiler, which implements the integer and pointer language including
recursion, function pointers, static storage and a real preprocessor;
**`aot-plan` / `aot-build`**, which refuse to emit anything unless every
reached operation has a CPython-free route; and **`py2bin.cabi`**, the vetted
list of CPython entry points a program may bind.
## Install

```sh
pip install python-to-binary
```

or from a checkout, which needs nothing installed at all:

```sh
git clone https://github.com/yu314-coder/python_to_binary.git
cd python_to_binary
PYTHONPATH=src python3 -m py2bin --help
```

The only requirement is Python 3.10 or newer. There are no dependencies, and
compiling imports neither `ctypes` nor `subprocess` - a build asks for an
interpreter and nothing else, which a test asserts by compiling in a fresh
interpreter and listing what got loaded.

## Platforms

What `compile-capi` - the tier that turns your program into machine code that
drives CPython - can target today.

| | x86-64 | arm64 |
|---|---|---|
| **macOS** | ✅ works | ✅ works |
| **Windows** | ⚠️ partial | ❌ future work |
| **Linux** | ❌ future work | ❌ future work |

**macOS, both architectures.** Verified the same way: the 889-program corpus
compiled for each agrees with CPython on 878 and differs on the same 7 cases,
all of them inherent - CPython's "Did you mean" needs a Python frame to
suggest from, the repr of a compiled function really is a builtin function's,
and so on.

**Windows x86-64 is partial and should not be relied on yet.** It builds a
PE32+ that imports the interpreter from `pythonXY.dll`, and simple programs
run correctly under Wine. But a function that *returns one of its parameters* -
`def g(x): return x` - jumps through a heap pointer and dies. A corpus slice
scores 87 of 100 with that shape as the cause. The build is real and the
remaining bug is in code generation, not in the image format: 518 indirect call
sites all resolve to genuine import slots, and every static and string
reference lands inside the image.

**Windows arm64 and both Linux architectures are future work.** Windows arm64
has no encoder for the import-table call. Linux needs an ELF `.got.plt` and its
relocations, which nothing here writes. Both are the same shape of job as the
two that are done.

The **native** tier (`py2bin compile`, no CPython at all) targets all six, and
the **freeze** tier targets whatever it has a runtime pack for. This grid is
about `compile-capi` only, because that is the tier with the interesting
constraint: it has to bind an external interpreter through the platform's
dynamic linker.

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

**The open Windows bug.** A function that returns one of its parameters -
`def g(x): return x` - jumps through a heap pointer and dies. Wine's backtrace
shows `rip` equal to `rdx`, an address in CPython's heap, so a `PyObject *` is
reaching the instruction pointer. Reading the parameter is fine; returning it
is not. Until that is understood, Windows is a preview.

An exe needs `pythonXY.dll` beside it or on the path. That is what you ship,
not something the compiler can settle.

## The three tiers, stated plainly

Three different things in this repository all end in "a file you can run". They
give very different guarantees, and the difference matters more than the
similarity.

| | (a) `freeze` / `bundle` | (b) CPython C-API path | (c) `compile` native |
|---|---|---|---|
| Shape of the idea | PyInstaller-shaped | Nuitka-shaped | a small AOT compiler |
| What executes your logic | CPython, interpreting your source/bytecode | machine code py2bin wrote | machine code py2bin wrote |
| Is CPython present? | Yes, bundled inside the artifact | Yes, linked as an external shared library | No |
| Accepts arbitrary Python? | Yes, in practice | Most of it: 878 of an 889-program corpus match CPython | No — a small explicit subset |
| Third-party packages | Yes, carried as payload | Only whatever the linked interpreter can already import | No |
| Targets | all implemented targets, given a runtime pack and target wheels | macOS (both), Windows x86-64 partially | all implemented targets |
| Artifact runs standalone? | Yes | **No** — needs that exact CPython installed | Yes |

**(a) Freeze** is compatibility packaging. Your program is not translated; it
is shipped next to an interpreter that runs it. This is the tier that handles
real applications with NumPy, Torch, or a GUI.

**(b) The CPython C-API path** is the Nuitka-shaped tier, with Nuitka's
essential dependency removed: Nuitka hands its generated C to clang, and py2bin
compiles the C itself. `py2bin compile-capi` now goes the whole way from
Python - it emits the C-API calls and compiles them, so the pipeline is
Python → C → machine code with nothing outside the standard library in it.
What survives that constraint is genuinely narrower than Nuitka. See
[What the C-API path supports](#what-the-c-api-path-supports).

**(c) Native compile** removes CPython entirely. Nothing is interpreted and
nothing is linked; the ELF/PE/Mach-O contains only instructions py2bin encoded.
The price is the smallest subset of the three.

None of these tiers invokes gcc, clang, `as`, `ld`, Xcode, or an SDK. py2bin
cannot: the library does not import `subprocess` and never starts a process, and
the test suite fails if that changes.

## What the C-API path supports

This is the honest boundary of tier (b). It is deliberately unflattering.

**How you reach it.** Three entry points, all producing the same IR:

```sh
# 1. Ordinary Python, translated into C-API calls for you. This is the
#    Nuitka-shaped route: Python -> C -> machine code, no clang. Every .py
#    beside the entry that it imports is compiled into the same binary.
PYTHONPATH=src python3 -m py2bin compile-capi program.py \
  --target darwin-arm64 -o program.bin

#    ...or as a macOS .app with an icon. What is inside the bundle is the
#    compiled program, not an interpreter and a copy of the source. --site
#    puts a directory on sys.path, which is how the binary finds packages
#    the linked interpreter was never told about.
PYTHONPATH=src python3 -m py2bin compile-capi app.py \
  --target darwin-arm64 --app --name "My App" --icon icon.icns \
  --site ~/venvs/myapp/lib/python3.12/site-packages -o MyApp.app

#    --embed-python carries the interpreter inside the bundle and names it
#    relative to the executable, so the .app starts on a Mac that does not
#    have this exact CPython installed.
PYTHONPATH=src python3 -m py2bin compile-capi app.py \
  --target darwin-arm64 --app --name "My App" --icon icon.icns \
  --embed-python --site Resources/site-packages \\
  --bundle-site ~/venvs/myapp/lib/python3.12/site-packages --prune-unused \\
  -o MyApp.app

# 2. Python that imports vetted C-API names from py2bin.cabi.
PYTHONPATH=src python3 -m py2bin compile program.py \
  --target darwin-arm64 -o program.bin

# 3. Canonical C that declares the same functions with `extern` prototypes.
PYTHONPATH=src python3 -m py2bin compile-c program.c \
  --target darwin-arm64 -o program.bin
```

The first is what `compile-capi` adds. It writes the C itself - `--emit-c PATH`
keeps a copy to read - and every value in it is a `PyObject *`, so the
interpreter's semantics apply rather than the native subset's:

```python
def fact(n):
    if n < 2:
        return 1
    return n * fact(n - 1)

print(fact(25))          # 15511210043330985984000000, exact
```

The native tier answers that one with a wrapped 64-bit integer. This tier hands
the multiply to `PyNumber_Multiply` and gets Python's own answer. The price is
the one this whole tier pays: the artifact needs libpython, so it is not
standalone.

Reference counting follows a single rule, chosen because it can be checked by
reading rather than by trusting: **every expression yields a reference the
caller owns**, and every statement releases what it finishes with. Reading a
name therefore increments first. A 500,000-iteration loop peaks at 14 MB, which
is what that rule holding looks like from outside.

The two are interconvertible: py2bin's C frontend parses the C into a Python
AST, so the same program can be run under `python3` (where `py2bin.cabi` makes
the identical calls through `ctypes.pythonapi`) and diffed against the compiled
binary. Every C-API feature below is verified that way — build, run natively,
run the same program under CPython, require identical stdout and exit status.

**What goes through.**

- A fixed table of 71 exported CPython entry points: `Py_Initialize`,
  `Py_Finalize`, `Py_IsInitialized`, `PyRun_SimpleString`,
  `PyLong_FromLongLong`, `PyLong_AsLongLong`, `PyUnicode_FromString`,
  `PyNumber_Add`/`Subtract`/`Multiply`/`TrueDivide`, `PyObject_RichCompare`,
  `PyObject_IsTrue`, `PyObject_Str`, `PyObject_Repr`, `PyObject_Size`,
  `PyObject_GetAttrString`, `PyObject_CallNoArgs`, `PyObject_CallOneArg`,
  `PyObject_Call`, `PyTuple_New`, `PyTuple_SetItem`,
  `PyObject_GetIter`, `PyIter_Next`, `PyFloat_FromDouble`,
  `PyFloat_AsDouble`, `PyObject_GetItem`, `PyObject_SetItem`,
  `PyNumber_Remainder`, `PyNumber_FloorDivide`, `PyNumber_Power`, `PyDict_New`,
  `PyDict_SetItem`, `PyTuple_Pack`, `PySequence_Contains`,
  `PyErr_ExceptionMatches`, `PyErr_SetObject`, `PySlice_New`,
  `PyNumber_Or`, `PyNumber_And`, `PyNumber_Xor`,
  `PyNumber_Lshift`, `PyNumber_Rshift`, `PyObject_DelItem`,
  `PyErr_GetRaisedException`, `PyCFunction_New`, `PyTuple_GetItem`,
  `PyObject_SetAttrString`, `PyErr_SetRaisedException`,
  `PyBytes_FromStringAndSize`, `PyUnicode_DecodeUTF8`, `PyNumber_Negative`,
  `PyNumber_Positive`, `PyNumber_Invert`, `Py_EnterRecursiveCall`,
  `Py_LeaveRecursiveCall`, `PyLong_FromString`, `PyImport_AddModule`,
  `PyObject_Vectorcall`,
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

**What `compile-capi` translates from Python.** Everything here is verified by
building it, running it, running the same source under CPython, and requiring
identical stdout and exit status.

- **The whole program, not just its entry file.** Every `.py` beside the entry
  that it imports is compiled and linked into the same image, registered under
  its own name before its body runs — so an `import` of it finds the compiled
  module rather than reading source next to the binary. A three-module,
  10,100-line application compiles to one 9.5 MB Mach-O that runs with none of
  its own `.py` files present.
- Every module has `__name__` and `__file__`, without which
  `if __name__ == "__main__":` never fires and a program cannot find what sits
  beside it. `__file__`, `sys.argv[0]` and any relative `--site` directory are
  resolved from the running binary, so a bundle can be moved.
- Compiled functions carry a signature, so `inspect.signature` describes them
  and frameworks that introspect (pywebview binding a JS API, for one) accept
  them. Defaults read as `None` — the format has no spelling for an arbitrary
  expression.

- Integers, floats, strings and f-strings, including format specifiers and
  `!r`/`!s`/`!a` conversions — `f"{x:.{places}f}"` goes to the interpreter's
  own `format()`, so the mini-language means what it means in Python.
- `list`, `dict`, `tuple` and `set` literals and comprehensions; subscripting
  and slices; every arithmetic, bitwise, boolean and comparison operator,
  including chains such as `0 <= x < n`, which evaluate each operand once.
- `if`/`while`/`for` with `break`, `continue` and tuple targets; unpacking;
  `with`; `del`; `import` and `from`-`import`; attribute access and method
  calls of any arity; keyword arguments; module-level globals.
- Functions with defaults, **nested functions, and lambdas**, to any depth. A
  closure is a real Python callable: the body is compiled to its own C
  function and `PyCFunction_New` wraps it, with what it captured travelling as
  the object CPython hands that function as `self`.
- **Classes**, including inheritance, `__init__`, `__repr__` and other dunder
  methods, class-level attributes, methods with defaults, and `isinstance`. A
  class is what `type(name, bases, namespace)` answers, so the method
  resolution order and attribute lookup are the interpreter's own. Each method
  is a closure wrapped in `functools.partialmethod`, which is what makes it
  bind — a raw `PyCFunction` is not a descriptor, so the instance would never
  arrive. Zero-argument `super()` works: CPython supplies `__class__` and
  `self` through a cell it makes for any method mentioning the name, and a
  compiled method writes the same two values out instead.
- Runaway recursion raises `RecursionError` rather than taking the process
  down: compiled calls use the real stack, and a segfault is not something a
  program can catch or report.
- A call with the wrong number of arguments raises `TypeError` with CPython's
  own message, qualified name included — `outer.<locals>.one() takes 1
  positional argument but 2 were given`.
- `try`/`except` (with `as name` and a tuple of classes), `raise`, and bare
  `raise` to re-raise what a clause is handling. A function body that raises
  with nothing to catch answers `NULL` with the exception still set, exactly
  as a C-API function does, so a `try` around the *call* catches it.
- `finally`, which runs on every way out of the region it protects: falling
  off the end, an exception nothing caught, `return`, `break` and `continue`,
  and through nested clauses in the right order. `with` is one of these: its
  `__exit__` runs however the body ends, and receives the real exception so it
  can suppress.
- Assignment to an attribute or a subscript, and augmented assignment to
  either. `xs[f()] += 1` calls `f` once, as Python does.
- `*args` and `**kwargs` spread into a call, and `[*xs, 3]` /
  `{**base, 'k': 1}` in a literal. A module-level `def` is also a value, so it
  can be passed as a sort key or have arguments spread into it.
- The whole parameter grammar: defaults, `*args`, `**kwargs`, keyword-only
  parameters with their own defaults, and positional-only parameters after
  `/`. Every compiled function takes keywords, so any parameter can be passed
  by name — `show(1, c=9)` reaches `c`.
- `for`/`while`/`try` with an `else` clause; `assert`; annotated assignment
  (`x: int = 5`); chained assignment (`a = b = v`); `del` of a name, an
  attribute or an item; and `print` with `sep=`, `end=`, `file=` or `flush=`.
- **Decorators**, on functions, methods and classes, stacked and with
  arguments — including `@staticmethod`, `@classmethod` and `@property`, which
  are handed the plain callable so their own binding is not obstructed.
- `global`, and dotted imports (`import a.b`, `import a.b as c`).
- `bytes` literals, and dict comprehensions. Integers of any width, and text
  or bytes carrying a zero byte — neither of which has a C type to arrive in.
- Comprehensions have a scope of their own, so `[x * 2 for x in xs]` leaves an
  enclosing `x` alone; unpacking checks how many values there were and raises
  Python's `ValueError`; and a name that exists nowhere, or whose only binding
  did not run, raises `NameError` or `UnboundLocalError` as Python does rather
  than reading an empty slot.

Two things it deliberately refuses rather than answering differently from
Python. A closure captures the *value* a name holds when the closure is made,
where Python captures the variable; where the enclosing scope moves that name
afterwards the two disagree, so that case is a build-time refusal naming the
variable. (At module level there is nothing to refuse: the name lives in the
module's own storage, so `[f() for f in fs]` after a loop of lambdas gives
Python's `[2, 2, 2]`.) `nonlocal`, `async` and generators are not translated
yet, and `functools.wraps` cannot rename a compiled function — see below.

**What py2bin does not do for you on the hand-written routes.** These apply to
routes 2 and 3 above, where you write the C-API calls yourself; `compile-capi`
generates all of this for you.

- **No automatic reference counting.** You call `Py_IncRef`/`Py_DecRef`. py2bin
  emits exactly the calls you wrote and does not verify ownership, so a leak or
  a double-free in your program stays a leak or a double-free.
- **No exception propagation.** A failing C-API call returns `NULL` and the
  error stays pending. py2bin inserts no checks and generates no unwinding;
  checking `PyErr_Occurred` is your program's job.
- **The native subset still applies to what is not a C-API call.** In route 2,
  constructs outside the native frontend's subset are rejected outright, and
  those inside it compile the way tier (c) compiles them, straight to
  instructions — `print` becomes a `write` syscall, not `PyObject_Print`.

**What no route does.**

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
