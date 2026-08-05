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

**0.8.9** is a large correctness and speed release - see *Release notes* below.
see *Release notes* below.

## Platforms

What `compile-capi` - the tier that turns your program into machine code that
drives CPython - can target today.

| | x86-64 | arm64 |
|---|---|---|
| **macOS** | ✅ works | ✅ works · 📦 ships a real app |
| **Windows** | ✅ works · 📦 ships a real app | ✅ works |
| **Linux** | ✅ works | ✅ works · 📦 ships a real app |

📦 marks a target a complete third-party GUI application has been built for and
run on real hardware rather than only a corpus: ManimStudio - 10,100 lines,
pywebview, Pillow, manim - as a Windows x86-64 `.exe`, a macOS arm64 app and a
Linux arm64 executable, all three working. iOS is **not** a py2bin target; that
app's iPad/iPhone build is a separate native Swift port embedding CPython for
`arm64-iphoneos`, and is not this compiler's work.

**An iPad can build all three of them, though.** py2bin has no compiler,
assembler or linker behind it - it writes the machine code and the Mach-O, PE
and ELF itself - so a cross-build is arithmetic and file writing, which a
sandbox allows. Run inside the embedded Python of the iPad app:

| built on iPadOS | artifact | carried off by | opened on the target |
|---|---|---|---|
| Windows x86-64 | `.exe` | USB | ✅ opens and runs |
| macOS arm64 | `.app`, and a `.dmg` of it | USB | ✅ opens and runs |
| Linux arm64 | ELF executable | USB | ✅ opens and runs |

Nothing on that path needs `subprocess`, `ctypes`, `fork` or `exec` - every
target still compiles with all of them removed from the interpreter, which is
a test. The iPad is a build machine and nothing else: an App Store app cannot
`exec` an arbitrary binary, so there is no such thing as a py2bin artifact
that runs on the tablet that made it.

Each working target is held to the same standard: an 889-program corpus is
compiled for it and every program's output and exit code compared against
CPython's. macOS agrees on 878 and differs on 7; a 100-program slice run
through Wine agrees on 93 and differs on 5. The differences are the same on
every platform and are inherent rather than open - CPython's "Did you mean"
needs a Python frame to suggest from, and the repr of a compiled function
really is a builtin function's.

The **native** tier (`py2bin compile`, no CPython at all) targets all six.

## What it guarantees

Names the program binds are the program's. Integers do not stop at 64 bits.
Floats stay floats and `-0.0` stays distinct. Evaluation order is Python's, and
a `__len__` or `__getitem__` runs exactly once. Exceptions - class, message,
`__cause__`, traceback, what `except` matches - are the interpreter's.

Where it knowingly differs: a generator expression is built eagerly;
`builtins.len` and `builtins.str` replaced at run time are not observed
(`print` is); attribute access is slower than the interpreter's, because
matching it means reading `ob_type` out of an object this treats as opaque -
which is what lets one binary run against a CPython it was not built against.

**Inside an `except` block, the exception being handled is not on record.**
`sys.exc_info()` answers `None` there, and an exception raised from a handler
gets no `__context__` - so a traceback loses its "during handling of the above
exception" chain. Both come from one thing: the handler *takes* the exception
rather than also setting it as the thread's handled exception, and CPython's
`PyErr_SetHandledException` is not in the vetted table. The exception itself,
its type, message, `__cause__` from an explicit `raise ... from`, and what
`except` matches are all correct; it is the implicit chaining and the
introspection that are missing.


## The paths through it

Three ways to turn a program into an artifact. They trade the same three things
against each other, and which one you want depends on which you care about.

| | `compile` | `compile-capi` | `freeze` |
|---|---|---|---|
| **speed** on a 30M-iteration loop | **0.05 s** | 0.44 s | 0.74 s |
| **artifact** | **32 KB** | 50 KB | 24 MB |
| **needs Python on the machine?** | **no** | yes, or bundle it | no, it carries one |
| **how much Python works** | a small subset | most of it: 878 of an 889-program corpus[^corpus] | **everything** |
| **third-party packages** | none | any the interpreter can import | **carried inside** |
| what actually runs your logic | machine code | machine code | CPython, interpreting |

**`compile` is the fastest and the smallest.** Python AST → py2bin IR →
optimizer → handwritten x86-64/ARM64 → ELF, PE or Mach-O. There is no
interpreter in the artifact and none on the machine: 14× faster than CPython on
that loop, in 32 KB that runs on a bare system. You pay for it in what it will
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
registers, and so is float arithmetic; attribute and method access are slower,
because each is a real C-API call where the interpreter has a per-site cache
that reads the object's internals directly. The per-feature table is below.

The loop above is deliberately unkind to `compile-capi`: its accumulator is
compared against a parameter, which the register analysis cannot claim, so the
fast path is off. On a loop it can claim, the same tier is 1.17× faster than
CPython; a float loop, once the worst row at 0.32×, is 1.06×; and a call to a
small helper is 2.10×, because the call stops existing and the loop around it
becomes machine arithmetic.

## Using it

```sh
pip install python-to-binary
```

| command | what it does |
|---|---|
| `py2bin make` | three questions, then a bundle - see below |
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
  --prune-unused --zip-stdlib --include web \
  -o dist/MyApp.app --clean
```

`--include PATH` carries a file or directory beside the program - a `web/`
folder, templates, anything it opens at runtime rather than imports. `make`
finds those on its own; this is how to name one it did not.

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

## Three questions, and nothing to type

```sh
py2bin make
```

It asks which file is the program, which machine it is for, and what shape it
should take. Everything else is found or downloaded rather than typed - the
other `.py` files beside it, the libraries it imports, an interpreter for the
target, `web/` and `assets/` if they are there, and an icon if one is.

| the shape offered first | what comes out |
|---|---|
| macOS | a compressed `.dmg` holding the app |
| Windows | one `.exe` that unpacks itself |
| Linux | one executable |

Two other ways in ship in the source distribution: `build.py` runs a clone
with nothing installed, and `get-py2bin.py` fetches py2bin for a machine with
neither - falling back to `curl` or `wget` where Python's own networking is
kept away from the interpreter.

## Bundling for Windows

The executable, the interpreter and the packages share one directory, and one
command assembles it:

```sh
py2bin compile-capi app.py --target windows-x86_64 --crash-log \
  --runtime /path/to/embeddable-cpython \
  --bundle-site /path/to/site-packages \
  -o dist/win/MyApp.exe
```

Neither needs a path on this machine: `--auto-fetch` downloads the
interpreter for the target, and `--fetch-package NAME` downloads and unpacks a
project's wheel. Both are checked against a published hash and cached.

`--bundle-site` copies packages into `Lib\site-packages` and names it on the
interpreter's path, which has to happen together: the embeddable CPython ships
a `pythonXY._pth` naming exactly two places, and once it exists `sys.path` is
those two and nothing else. Packages are invisible until the path file names
them, and the program reports `ModuleNotFoundError` for a directory plainly on
disk - silently, if it is windowed.

A wheel must also match the interpreter's ABI, not only its version: `cp314`
and `cp314t` differ by a character, the second is for the free-threaded build,
and only one loads.

## macOS bundles, signing and disk images

`--app` writes a `.app`; `--embed-python` makes it carry its own interpreter,
so it runs on a Mac with no Python installed. The bundle is signed and sealed
as the last step of the build, once everything is in place, and
`codesign --verify --deep --strict` exits 0 on the result.

The signature is ad-hoc - no Apple Developer ID, no notarisation, since either
needs a paid account and Apple's own tooling. That only matters to Gatekeeper,
which inspects apps carrying a quarantine flag: copied from a USB stick there
is none, downloaded through a browser there is, and then the app needs one
trip through **System Settings → Privacy & Security → Open Anyway**.

`--dmg` writes a mountable disk image beside the bundle. No `hdiutil` is
involved, because nothing in this library may reach for a subprocess; the
filesystem is written byte by byte as ISO 9660 with Joliet, which macOS mounts
with files executable - what an `.app` needs in order to launch.

```sh
py2bin compile-capi app.py --app --dmg -o dist/MyApp.app
```

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
| walrus (`:=`) | ✅ |
| `raise … from …` | ✅ |
| starred unpacking (`a, *b, c = …`) | ✅ |
| `match`: values, `\|`, captures, sequences, guards | ✅ |
| `match`: mapping and class patterns, `__match_args__` | ✅ |
| generators: `yield`, `send`, `yield from`, `return value` | ✅ |
| `async def` / `await`, driven by a real event loop | ✅ |
| `match`: starred sequence patterns (`[a, *rest]`) | ✅ |
| `yield` inside `try` / `except` | ✅ |
| `yield`/`await` inside `try` / `finally` | ✅ |
| `yield`/`await` inside `with`, including suppression | ✅ |
| a `finally` that itself yields; `break` out of one | ✅ |
| `async for` / `async with` | ✅ |
| `nonlocal`, as a cell a closure can rebind | ✅ |
| a closure over a name still moving, with Python's late binding | ✅ |
| unpacking into nested tuples, attributes, subscripts | ✅ |
| `raise SomeError` - a class rather than an instance | ✅ |

A generator cannot be compiled the way the rest is - a C function has one
entry and its locals die with its frame, so it cannot stop in the middle of
itself. It is turned inside out instead: the body is cut into blocks at each
`yield`, the blocks are numbered, and the function becomes a class whose
`__next__` dispatches on which block to run next, with the locals as attributes
because they have to outlive a `return`. The class is then compiled by the
machinery that already compiles classes, so there is no new C and nothing
interpreted at run time.

That covers `yield` as a statement or as a value, `send`, `yield from`,
straight-line code, `if`/`else`, `while`, `for`, `break`, `continue` and a bare
`return`. `next(g)` is `g.send(None)` here as it is in the protocol, and
`yield from` is written as the loop it is before the body is cut - which
forwards iteration but not a `send` into the sub-generator, so a `yield from`
whose value is used is refused rather than quietly answering None.

A `try`/`except` around a `yield` works, and the way it works is worth saying,
because "the handler has to survive the suspension" sounds like it needs
something the cut cannot give. It does not: an exception can only be raised
while a *block* is running, so each block of the guarded region carries the
handler and it is re-established on every entry rather than having to persist
across one.

`await` is the same machine with a second name on it. Awaiting an object with
`__await__` means delegating to the iterator it answers with, and a state
machine is one - so an `async def` compiles to the same class, plus `__await__`
returning itself, and `await x` is PEP 380's expansion of
`yield from x.__await__()`. A real event loop then drives it through `send`
exactly as it drives a coroutine: `asyncio.run`, `asyncio.sleep` and
`asyncio.gather` all work on compiled coroutines.

A `finally` around a `yield` works, and so do `with`, `async for` and
`async with` - see *How `finally` and `with` are handled* above for how, and
for the two shapes that are still refused.

A refusal is a `file:line:col` error, never a silent approximation. On an
889-program corpus, 878 programs produce byte-identical output to CPython; the
7 that differ do so inherently (CPython's "Did you mean" needs a Python frame,
`"v" is "v"` depends on interning) and 4 are refused outright.

### How fast each one is

Measured on an **Apple M4** (10 cores - 4 performance, 6 efficiency - 24 GB,
macOS 27.0, arm64) against **CPython 3.14.3, python.org framework build** -
the interpreter these binaries actually bind, which is not the same as
whichever `python3` is first on PATH and does not perform alike.

300,000 iterations per row, nine fresh processes each, median taken, timing
only the hot loop so neither column pays for start-up. Higher is better.
The harness and cases are in `benchmarks/` in the repository.

| feature | py2bin | CPython | |
|---|---|---|---|
| direct function call | **3.0 ms** | 7.1 ms | **2.39× faster** |
| integer arithmetic | **5.3 ms** | 8.5 ms | **1.61× faster** |
| `while` loop | **4.6 ms** | 6.3 ms | **1.37× faster** |
| comparisons | **3.7 ms** | 4.7 ms | **1.26× faster** |
| `try` that does not raise | **3.5 ms** | 3.6 ms | **1.05× faster** |
| float arithmetic | **5.5 ms** | 5.8 ms | **1.04× faster** |
| `in` on a list | 9.0 ms | 9.0 ms | 1.00× |
| list append | 5.2 ms | 5.2 ms | 1.00× |
| exception raise/catch | 20.1 ms | 19.8 ms | 0.98× |
| comprehension | 5.7 ms | 5.6 ms | 0.98× |
| dict store | 8.4 ms | 8.0 ms | 0.95× |
| `and` / `or` | 6.3 ms | 5.8 ms | 0.93× |
| f-string | 21.9 ms | 17.6 ms | 0.80× |
| `isinstance` | 7.6 ms | 6.0 ms | 0.80× |
| string concatenation | 16.0 ms | 12.5 ms | 0.78× |
| dict lookup by name | 6.5 ms | 4.8 ms | 0.73× |
| subscript | 8.5 ms | 6.1 ms | 0.71× |
| module global read | 5.4 ms | 3.7 ms | 0.69× |
| closure call | 10.2 ms | 6.7 ms | 0.66× |
| `for` over a list | 4.4 ms | 2.9 ms | 0.66× |
| attribute read | 6.3 ms | 3.7 ms | 0.59× |
| attribute write | 6.0 ms | 3.0 ms | 0.49× |
| instantiation | 34.4 ms | 16.3 ms | 0.47× |
| method call | 15.6 ms | 6.7 ms | 0.43× |
| tuple unpacking | 15.4 ms | 5.6 ms | 0.36× |

Ratios are computed from the unrounded timings, so dividing the millisecond
figures as shown gives a slightly different number in the last decimal.

One recorded run, the one in `benchmarks/last-run.json` in the repository.
Repeat it and the figures move by a few per cent either way - which rows beat
the interpreter, and by roughly how much, does not.

### Where those numbers came from

Fourteen of the twenty-five rows sit at 0.80× or better and six beat the interpreter
outright. Most did not a short while ago.

| row | before the fix | after it | what it was |
|---|---|---|---|
| direct function call | 0.81× | **2.10×** | the call hid the arithmetic from the register analysis |
| exception raise/catch | 0.49× | **1.06×** | every raise classified its argument through a Python-level `type()` |
| float arithmetic | 0.32× | **1.06×** | floats were never held in registers at all |
| attribute read | 0.51× | 0.82× | the name was built and hashed at every access |
| string concatenation | 0.14× | 0.80× | literal text was joined at run time, every time |
| list append | 0.28× | 0.72× | a lookup, a bound method and a discarded `None` per call |
| instantiation | 0.09× | 0.51× | `__init__` was reached through a Python-level wrapper |
| method call | 0.05× | 0.40× | so was every other method |

The largest wins were not optimisations but mistakes being removed - a wrapper
written in Python on the method path, a float analysis that did not exist, a
string rebuilt on every iteration of a loop. And they all have one shape:
something stops happening. Adding a cheap test in order to skip expensive work
*inside* the interpreter was tried five times and measured flat or slower every
time, because the extra call through the import table cost more than it saved.

**Arithmetic loops win** because a local the analysis picks out is held in a
machine register - a `long long` for an integer, a `double` for a float - with
an overflow check that falls back to unbounded arithmetic when an integer
leaves the word. That is what CPython's specialising interpreter does, and
doing anything less was what made this tier slower than not compiling at all.

**Everything else loses, by a factor that tracks how many C-API calls the
operation costs.** Each one is a real call with the reference-count discipline
around it, where the interpreter's specialised bytecode does the same work
inline.

**Method calls used to be far worse than that pattern predicted** - 21×,
where everything else paid 2-4×. Every compiled method was wrapped in
`functools.partialmethod` to make it bind, and that wrapper's `__get__` is
written in Python, so each `obj.method` ran interpreted code before the
call could start. CPython's own `instancemethod` does the same binding in
C, and both are now within the general pattern.


### Raising a class

`raise ValueError` names a class and `raise ValueError("x")` an instance, and
the two want different things from the C API - asking `type()` for the class
of a class answers `type`, the metaclass. The plainest raise a program can
write therefore ended in `SystemError: exception <class 'type'> is not a
BaseException subclass`, in compiled code of every kind. Fixed.

### How `finally` and `with` are handled

A generator becomes a class with `__next__`, not a generator: never closed,
never finalised by the collector, so the only ways out of a protected region
are the ones the rewriter can see. The cleanup is not a real `finally:` - a
`yield` returns from `__next__`, so one would fire on every suspension.
It is attached to the raising path as a handler that runs it and re-raises,
while the ordinary path jumps to a block holding the same cleanup. `with`
expands into the try it stands for and takes that path, with `__exit__` looked
up once on the type and suppression honoured.

`async for` and `async with` take the same route, each written out as what it
stands for. A `return` here is signalled by raising `StopIteration`, so the
cleanup's handler had to learn to tell the frame leaving from a real failure -
otherwise `__aexit__` is handed a `StopIteration` where CPython passes `None`.

A `finally` that itself yields works too: the cleanup is a block and a block
may suspend, reached the same way from both paths, with whatever was raised
waiting in a name until it is done. A `break` or `continue` leaving the region
runs a copy of the cleanup first, which is what it would have reached had it
left the ordinary way.

### One file, both ways

On macOS an application *is* a directory - Finder runs `Contents/MacOS/<name>`
and Gatekeeper reads `Contents/Info.plist` beside it. Nuitka says so in its own
help: `--mode=app` is "onefile except on macOS where it creates an app bundle".
What stays open is how much is *inside* the bundle, and `--onefile` folds the
payload into the bundle's own executable.

| | files | to hand over | first start | later starts |
|---|---|---|---|---|
| py2bin `--app` | 495 | 66.0 MB | 84 ms | **84 ms** |
| py2bin `--app --onefile` | **3** | 23.0 MB | 4.3 s | 134 ms |
| py2bin `--app --dmg` | 1 image | **21.8 MB** | 84 ms | **84 ms** |
| Nuitka `--mode=app` | 255 | 73.5 MB | **79 ms** | **79 ms** |

The packed bundle unpacks once into a content-addressed cache and runs from
there. A self-extracting single *executable* is what py2bin builds on Windows
and Linux, and Nuitka where it can - on macOS it declines that shape once
pyobjc is in the graph, which is any pywebview program.

## Measured against Nuitka

manim_app: 10,100 lines, pywebview + Pillow + pyobjc, built both ways on the
same machine.

> **The two bundle tables were taken at 0.8.5 and have not been re-taken.**
> They need the application's own virtualenv staged into wheels, which is not
> a build this repository can run on its own. Every other figure here was
> re-measured for 0.8.7.

| | py2bin | Nuitka |
|---|---|---|
| whole `.app` | **66.0 MB** | 73.5 MB |
| main binary | **8.9 MB** | 28.9 MB |
| native extensions carried | 8.7 MB | 8.7 MB |
| start with the app's imports | 84.4 ms | **78.6 ms** |
| compile time | **20.1 s** | 88.3 s |

Whole-process time - start-up included - median of 5, seconds, on the same
Apple M4 against **Nuitka 4.1.3 `--standalone`**:

| workload | py2bin | CPython | Nuitka |
|---|---|---|---|
| integer arithmetic | **0.061** | 0.099 | 0.094 |
| `while` loop | **0.055** | 0.084 | 0.062 |
| nested loops | **0.021** | 0.025 | 0.027 |
| function calls | **0.020** | 0.038 | 0.036 |
| string building | **0.022** | 0.024 | 0.026 |

Two of those rows finish in under thirty milliseconds, so start-up is a large
share of them - a real difference between the three rather than a distortion.
The margins that are about generated code are the loops and the calls. This
compiler's weaknesses do not show up in a five-loop benchmark at all; for
those read the grid above, where method call and instantiation sit at 0.41×.
Put a hot loop at module level instead of in a function and the loop advantage
goes away, because module-scope names are not narrowed into registers.

### What a build costs

Run time is what a user waits for; build memory is what decides whether the
build runs at all. py2bin never starts a C toolchain. Peak resident set of the
whole build process tree, sampled every 25 ms. Nuitka keeps a ccache and a
module cache and py2bin keeps none, so both answers are given - py2bin's
column is the same in each.

**Cold** - a first build, or CI without a warm cache:

| what is being built | py2bin | | Nuitka | |
|---|---|---|---|---|
| a small program (~10 lines) | **38-42 MB** | **0.1-0.2 s** | 564-719 MB | 17-18 s |
| 200 functions | **186 MB** | **2.0 s** | 678 MB | 18.7 s |
| 1,000 functions | **601 MB** | **7.5 s** | 833 MB | 23.7 s |
| 3,000 functions | **1,514 MB** | **22.0 s** | 1,740 MB | 36.3 s |

**Warm** - Nuitka's cache in place, which is what a second build gets:

| what is being built | py2bin | | Nuitka | |
|---|---|---|---|---|
| a small program | **42 MB** | **0.1 s** | 296-297 MB | 3.7 s |
| 200 functions | **188 MB** | **2.0 s** | 423 MB | 4.9 s |
| 1,000 functions | **603 MB** | **7.4 s** | 713 MB | 8.1 s |
| 3,000 functions | 1,571 MB | 21.3 s | **1,516 MB** | **17.5 s** |

A small build costs a seventh of a warm Nuitka's and a fifteenth of a cold
one, which is the whole reason an iPad can run one. The advantage narrows with
program size - nothing here streams, so the curve is steeper - but on a cold
build it survives to the largest case measured. Only against a *warm* Nuitka
at 3,000 functions does it turn over.

**Yes, the C toolchain is counted** - it is most of Nuitka's column. The
sampler walks the whole process tree and now reports what held the memory, so
the total can name its parts rather than be asserted:

| building | py2bin's tree held | Nuitka's tree held |
|---|---|---|
| a small program | `Python` 40 MB | `Python` 300 MB, `clang` 181 MB, `ld` 150 MB, `codesign` 8 MB |
| 3,000 functions | `Python` 1,567 MB | `ld` 1,210 MB, `clang` 608 MB, `Python` 434 MB |

At scale it is the *linker* rather than the compiler that dominates. py2bin
starts neither: it writes the object code and the container itself, so its tree
is one Python process at every size.

Sampling finer matters here, and the interval was tightened from 25 ms to
10 ms after checking: clang processes are short-lived, and at 25 ms the same
build measured 582 MB where 5 ms saw 643 MB. The published figures were
understating Nuitka - which flattered it rather than py2bin, but was wrong
either way.

Startup, `print("x")`, median of 13 runs, same M4:

| | startup | on disk |
|---|---|---|
| py2bin `compile-capi` | **10.1 ms** | **49 KB** |
| CPython | 13.8 ms | - |
| Nuitka `--standalone` | 15.4 ms | 17.2 MB |

Loops beat both because a local the analysis picks out is held in a register
rather than on the heap, with the overflow check that falls back to unbounded
arithmetic when it leaves the word. Calls still lose: an argument is boxed at
the call and unboxed inside, where the interpreter's specialised call pays
neither.

## Release notes

Newest first. The full history is in the repository.

### 0.8.9 - verdicts, borrowed references, and a leak in every `try`

The largest release so far, and most of it is correctness that turned up while
chasing speed. Fourteen of the twenty-five measured rows now sit at 0.80x or
better and six beat the interpreter.

**A condition wants a verdict, not a value.** Four places computed an object
and then asked what it meant. `if a and b` evaluated the whole chain into a
Python boolean; `if x in xs` looked `True` up *by name on the builtins module*
and handed it to `PyObject_IsTrue`, when `PySequence_Contains` had already
answered; `isinstance(x, C)` found the callable and dispatched through it,
where `PyObject_IsInstance` is what the builtin does; `f"{x}"` asked `str`.
Each goes straight through now, and the short circuit in `and`/`or` is the C
`if` guarding the next side, so a side that must not run has no code reached
rather than a value discarded. **`and`/`or` 0.66x -> 0.94x, `in` on a list
0.37x -> 1.02x, `isinstance` 0.51x -> 0.80x.**

**A `__bool__` that raised was read as true.** `PyObject_IsTrue` answers -1
with an exception set, and -1 is true in C, so a class whose `__bool__` raised
ran the body of the `if` and the program exited 0 where CPython stops. The
same for a comparison that raised. Every verdict is checked where it is
produced now. This was there before any of the work above and is the more
important half of it.

**`f"{x}"` asked `str` where Python asks `__format__`.** For most types those
agree, because `object.__format__` with an empty specifier defers to `str`;
for a type that defines `__format__` they do not. `PyObject_Format` is now
vetted, and an exact `str` skips the call entirely - the same two paths
CPython's `FORMAT_SIMPLE` takes. **f-string 0.73x -> 0.80x.**

**Calls, literals, stores and lookups stopped paying for references they
already hold.** `PyObject_CallOneArg` borrows its argument and the
two-or-more path knew it, but the single-argument path - the commonest call
shape there is - took a reference and dropped it again, and so did the
callable. A pooled literal lives in a static written once at start-up and was
incremented and decremented around every use. `obj.field = v` and `d[k] = v`
did the same with the object and the key, and `d["name"]` and `xs[i]` did it
with the key they look up. All borrowed now, on rules that still refuse to
borrow a global. **Closure call 0.55x -> 0.66x, attribute write 0.47x ->
0.53x, dict lookup by name 0.61x -> 0.71x.**

**Every `try` built the classes its clauses catch, and then leaked them.**
They were built before the body ran, so a `try` paid for them whether it
needed them or not - and nothing released them where the body raised nothing.
`except (ValueError, TypeError)` builds a fresh tuple each evaluation, so a
`try` in a loop leaked one per turn: 400,000 turns held 40 MB against the
interpreter's 15, and 800,000 held 65.

They are built in the handler now, where the body has already raised. Doing
that with an exception set is what CPython refuses - `except (A, B)` failed
with "returned a result with an exception set" - so the exception is lifted
out with `PyErr_GetRaisedException` and put back once they are built, which is
what the match needs anyway. A `try` that does not raise now builds nothing at
all: **0.70x -> 1.04x**, past the interpreter. The raising path pays for the
lift and costs a little, 1.06x -> 0.97x, which is the right way round for a
construct whose point is that it usually does not raise.

**A long function no longer needs a bigger stack frame than a short one.**
Every intermediate the C lowering needed took a stack slot and never gave it
back, so a frame grew with a function's *length*: about 1,800 statements in one
`def` reached the 512 KB budget and the build was refused, which generated code
walks into without doing anything unusual. Forty thousand statements compile
now. The float formatter's scratch and any local outlive the statement that
made them and are excluded, and reclaiming is per statement rather than per
expression, which is what keeps a loop's condition alive across its own body.

**Tuple unpacking was the worst row measured and nothing had measured it.**
Deciding whether a two-item tuple has two items boxed the length, boxed the
expected count twice, ran two `PyObject_RichCompare` calls and asked
`PyObject_IsTrue` of each - eleven C-API calls and five allocations for one
machine comparison - and called `tuple()` first, allocating a copy per unpack.
The length is a machine comparison now and a value that can answer for itself
is taken apart where it stands, with `tuple()` kept for what needs it: a
generator has neither length nor index. **0.18x -> 0.36x, and
`for n, x in enumerate(...)` 0.23x -> 0.50x.**

**Four names that were not the program's.** A decorator written without the `@`
- `greet = trace(greet)` - kept calling the undecorated body, because a
module-level `def` earned a direct C call keyed on the spelling alone. Reading
a global above its assignment handed the program a raw NULL instead of raising
`NameError`, and so did a class used above its `class`. A function that rebound
itself through `global` kept calling its old body. The rule is positional now:
a `def` earns the direct call only when it is the one thing binding that name
at module scope, and only where it is already bound - so a function may still
call one written below it, and recursion keeps the direct call.

**A nested function could not call itself.** Its own name is not bound when the
capture is taken - the `def` being compiled is what binds it - so an ordinary
nested `fact` raised `NameError` on its first recursive call. Mutual recursion
between nested functions is refused at build time with an explanation, because
capture-by-value cannot express it and failing at run time naming a function
written plainly above is the worst way to say so.

**Smaller things.** `+` on strings known to be exact skips the `__add__`
dispatch, and exactness composes so `a + b + c` converts throughout. Branches
whose condition is already a constant are removed, in function bodies as well
as the entry point - 24 operations across the benchmark suite, which is a small
number and is stated rather than implied. The launcher scripts and the
bootstrapper's PowerShell fallback quote or pass by environment what they were
given, so nothing py2bin writes into a shell is read as shell.

**Eight shapes were added to the grid**, which now has twenty-five rows. They
were found by measuring what the suite did not cover, and most were worse than
anything in it. They are published because a grid showing only the shapes a
compiler is good at is a grid measuring itself.

### 0.8.8 - archives, and what they may not do

py2bin's safety story is that it never *runs* what it downloads - no
`setup.py`, no pip, no install hooks. Unpacking is therefore the only moment a
hostile wheel or runtime pack acts at all, and it was not being guarded well
enough. Reported by a static scan; each was reproduced before it was fixed.

**A symbolic link in an archive could write outside the build.** Three callers
each checked containment by resolving where a member would land and comparing
strings. Both halves fail: `resolve()` runs *before* extraction, so a link the
archive is about to create is not there to be resolved through - `esc -> ..`
followed by `esc/passwd` passed both members and `extractall` then followed the
link it had just made - and `/tmp/outsider` starts with `/tmp/out`. Reproduced
on Python 3.11, writing a file outside the destination. Python 3.14 happens to
stop it because `extractall` defaults to the `data` filter there; this project
supports 3.10 upwards, so on most of that range nothing did.

Extraction now happens member by member through one shared implementation
(`py2bin.archives`), never `extractall`, with every member judged before
anything is written. Links are *validated*, not banned: a CPython framework is
a structure of symbolic links, and a runtime pack that lost them would not run.

**A wheel could escape on Windows through a backslash.** `..\..\x` is a
single atom to a POSIX path - no `..` part, not absolute, so it passed - and
then becomes a traversal when Windows splits it again. One of the two copies
of this rule refused the character and the other did not.

**An archive could expand without limit.** Member *count* was capped, which
says nothing about size.

**A fetched runtime is no longer trusted silently.** `--fetch-lock` now exists
on `compile-capi` as well as `freeze`, so the tier most people use can pin what
it downloads. python.org publishes a GPG signature for the Windows embeddable
CPython and no SHA-256, so there is nothing to verify against on a first fetch:
that is now said out loud, with the digest and how to pin it, instead of being
recorded quietly and compared against on later builds.

A refused download now reports as one line rather than a traceback, and a test
asserts no fetching path can go back to unpacking its own way.

### 0.8.7 - long functions, and names that are not bound yet

**A long function no longer needs a bigger stack frame than a short one.**
Every intermediate the C lowering needed took a stack slot and never gave it
back, so a function's frame grew with its *length* rather than with how much
of it was live at once. About eighteen hundred statements in a single `def`
reached the 512 KB budget and the build was refused outright - which generated
code walks into without doing anything unusual. Slots taken for a statement's
temporaries are handed back when it finishes, and the frame is now built from
the high-water mark rather than from what happens to be outstanding at the
end. Forty thousand statements in one function compile and answer correctly,
where 1,900 was refused before.

Two things had to be got right and both have tests. A local, or the scratch
area the float formatter allocates the first time a program prints a double,
outlives the statement that created it - reclaiming those handed a later
statement slots something still held, which did not print wrong numbers but
exhausted the heap. And reclaiming is per statement rather than per
expression, which is what keeps a loop's condition alive across its own body.

**A decorator written without the `@` was skipped.** `greet = trace(greet)`
kept calling the undecorated body: a module-level `def` became a direct C call
keyed on the name alone, so a later rebinding of that name was never
consulted. A `def` now earns the direct call only when it is the one thing
binding the name at module scope.

Three more of the same shape, all asking *does the module bind this name*
where the question is *is it bound yet*: `print(y)` above `y = 5` handed the
program a raw NULL instead of raising `NameError`, the same for a class used
above its `class`, and a call above its own `def` answered rather than
raising. The replacement rule is positional, so a function may still call one
written below it and recursion keeps the direct call. Nine tests pin it, five
of which fail against 0.8.6; no measurable speed cost.

### 0.8.6 - repairing 0.8.5, and six machines

**0.8.5 could not build anything through `py2bin make` or `build.py`.** An
`append` option nobody passed is `None` rather than an empty list, and the
change that stopped `--exclude` fetching what it excluded read it without a
default - so every `--auto-fetch` build without `--exclude` stopped with a
`TypeError`. Fixed, and every `append` option is now checked for a default by
a test.

The three questions offer six machines rather than five (`linux-x86_64` was
missing), Linux gains a one-file shape, and all sixteen target-and-shape
combinations are driven through `build.py` end to end in testing.

A helper naming a module constant, or calling another helper, is now written
out at its call site: `bump(v)` over `weigh(v)` collapses to `v * SCALE + 1`,
0.67x to 1.26x against the interpreter.

### 0.8.5 - a correctness sweep

A correctness sweep. Each of these produced a *wrong result rather than an
error*, which is the worst way for a compiler to be wrong; all are now pinned
by tests.

- A name the program bound was ignored - its own `len`, `str`, `print` or
  `super` lost to the builtin, and a module-level function called through a
  rebound name calling the wrong one.
- A bundle could not find the packages it carried, on Linux: the program asked
  CPython where it was, and CPython - handed no argument vector - answered with
  its own installation.
- `sys.argv` held one entry this compiler had put there, so a command-line tool
  could not read what it was asked to do. It is taken from the operating system
  now.
- `len(5)` answered `-1` instead of raising, leaving the `TypeError` set.
- A two-piece f-string ran `__add__`; an f-string joins.
- A wheel's executable bit was dropped, so a package shipping a helper program
  could not start it.

Two that destroyed things: `--clean` removed whatever was at the output path,
directory and contents; `--include` removed its own source when the output was
in the same directory. Both are refused now.

New: `--onefile` for a macOS `.app` and for targets with no bundle;
`--exclude` reaching the fetch rather than downloading what it excluded; a
syntax error reported with file, line and column instead of a traceback.

Faster too - seven of sixteen measured operations now beat CPython, where two
did. See the table below.

## Licence

MIT. Full documentation, source and issues:
**https://github.com/yu314-coder/python_to_binary**

[^corpus]: That sweep was last run before the optimisation work described
    further down, and its harness was scratch rather than committed, so the
    figure is reported as measured rather than as currently verified. What is
    checked on every change is the 1529-test suite and a differential set that
    demands byte-identical output to CPython.
