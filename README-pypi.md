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

**0.9.3 gives every module the names every module has.** A bare `__spec__`
fell through to the builtins module and answered with *its* spec, so
`__spec__.name` said `"builtins"` - a confident wrong answer to "where am
I?". `__package__`, `__spec__`, `__loader__` and `__builtins__` are the
module's own now.

**0.9.2 put a function's signature and its annotations back together**, so
`inspect.signature(f)` shows the types the source wrote.

**0.9.1 finished what a compiled function can say about itself:**
`f.__annotations__`, and with it `typing.get_type_hints` and
`functools.singledispatch`, which is most of what annotating a function is
for. Only where the program asks - an annotated program that never reads its
annotations compiles to exactly what it compiled to before, and calls are
untouched either way. `globals()` in a module other than the entry now
answers with that module's own names rather than `__main__`'s.

**0.9.0 closed what a compiled function could not do, and what a compiled
*program* could not be.** Metaclasses, `enum` and `dataclasses`, generator and
`async def` methods, async generators, `locals()`, a real `globals()`, `eval`
and `exec`, `sys.exc_info()` inside an `except` - and now `abc.abstractmethod`
and `functools.wraps`, the two decorators that write on the function they are
given, which between them are how a great many programs begin and how nearly
every decorator is written.

A program is also more than a flat directory of files: packages, submodules,
`from . import x`, PEP 420 directories, `importlib.import_module("pkg.thing")`
and everything under `src/` are all compiled into the image now, where before
only a `.py` sitting beside the entry was. Along with them, `f.__doc__`, a
`finally` that keeps the exception it interrupted, the two ends of a
generator's life, `athrow` on an async generator, and the `SyntaxWarning`s
CPython's compiler gives.

Found by compiling language shapes one at a time and comparing each against
CPython - some four hundred of them across this release. What is left is one
fact: a compiled function is a `builtin_function_or_method`, not a Python
function. See *Release notes* below.

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
CPython's. macOS agrees on 886 and differs on 3; a 100-program slice run
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

Where it knowingly differs:
`builtins.len` and `builtins.str` replaced at run time are not observed
(`print` is); attribute access is slower than the interpreter's, because
matching it means reading `ob_type` out of an object this treats as opaque -
which is what lets one binary run against a CPython it was not built against.

Inside an `except` block the exception being handled **is** on record:
`sys.exc_info()` answers with it, including through a call made from the
handler, and an exception raised there takes its `__context__` from it, so a
traceback keeps its "during handling of the above exception" chain. It is put
back on the way out whichever way the clause leaves - falling off the end,
returning, breaking, or raising - and put back to what was there rather than
cleared, so handlers nest.

**A call that names an argument is no longer slow.** It was the worst row
here for a long time - `helper(t, step=1)` measured 0.13x the interpreter,
then 0.18x, then 0.27x as the caller stopped allocating a tuple and a dict
per call and the callee stopped rebuilding a dict from `kwnames`. All of that
was work to make the run-time binding cheaper, and the binding did not need
to happen at run time at all: which parameter a name is for is settled by the
call site and the `def`, and where both are in the same module both are in
front of the compiler. Placed there, `f(a, step=1)` is written as `f(a, 1)`
and is an ordinary call - which means it inlines, and the loop around it
holds its values in machine registers. The row is **2.26x faster than the
interpreter**, from 27.8 ms to 3.4 ms.

What is left goes through the callable, as it always did: a `**mapping`, a
name that is not a parameter, a parameter given twice, a gap with no way to
say "default here", a reordering that would move a side effect, or a callee
that is not a plain function in this module.



## The paths through it

**There are two ways to build, and they are the two you choose between:**

- **`freeze` - ship Python with it.** Your program travels beside a real
  interpreter, the way PyInstaller does it. Quickest to build, and every
  Python program works.
- **`compile-capi` - compile it.** Your program is translated to C and that C
  to machine code by py2bin's own compiler. Slower to build; no source and no
  bytecode in the result.

That is the whole decision, and it is the only one `py2bin make` (or
`build.py` in a clone) asks about.

There is a third, `compile`, which is not a general choice: it accepts a small
subset of the language and no packages at all, in exchange for an artifact
with no interpreter anywhere near it. Reach for it when that is the point.

| | `freeze` | `compile-capi` | `compile` |
|---|---|---|---|
| **speed** on a 30M-iteration loop | 0.74 s | 0.44 s | **0.05 s** |
| **artifact** | 24 MB | 50 KB | **32 KB** |
| **needs Python on the machine?** | no, it carries one | yes, or bundle it | **no** |
| **how much Python works** | **everything** | most of it: 886 of an 889-program corpus[^corpus] | a small subset |
| **third-party packages** | **carried inside** | any the interpreter can import | none |
| what actually runs your logic | CPython, interpreting | machine code | machine code |

**`freeze` is the most complete.** It ships your program beside an interpreter
that runs it, so NumPy, Torch and a GUI toolkit all work exactly as they do
now. Nothing is translated, so nothing is faster; the artifact is the largest
of the three because an interpreter and every dependency are inside it.

**`compile-capi` is the one under active work.** It translates
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

It asks which file is the program, which machine it is for, and which of the
two ways to build it - ship Python with it, or compile it. Everything else is
found or downloaded rather than typed - the other `.py` files beside it, the
libraries it imports, an interpreter for the target, `web/` and `assets/` if
they are there, and an icon if one is. What shape the result takes is not
asked about: one file, always, because that is the thing somebody can send.

| target | ship Python with it | compile it |
|---|---|---|
| macOS | one executable, ~14 MB | a compressed `.dmg` holding the app, ~10 MB |
| Windows | one `.exe`, ~10 MB | one `.exe` that unpacks itself, ~11 MB |
| Linux | needs a Linux machine to build on | one executable, needing Python there |

Freezing needs a whole CPython built for the target machine. One is published
for Windows and is downloaded; for anything else it has to come from a machine
like the target. Where that cannot be done the question is not asked -
compiling is stated and the build goes on.

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
| packages, `pkg/sub/deeper.py`, `from . import x`, PEP 420 directories | ✅ |
| `importlib.import_module("pkg.thing")` with the name written down | ✅ |
| a program that puts its own `src/` on `sys.path` | ✅ |
| `abc.abstractmethod`, `functools.wraps` - both write on a function | ✅ |
| `f.__annotations__`, `typing.get_type_hints`, `singledispatch` | ✅ |
| `f.__doc__` and a class's, so `help()` and `inspect.getdoc` answer | ✅ |
| `globals()` in any module of the program, answering with its own | ✅ |
| CPython's compile-time `SyntaxWarning`s, at compile time | ✅ |
| `athrow` / `asend` / `aclose` on an async generator | ✅ |
| `global` / `nonlocal`, tuple unpacking | ✅ |
| the whole program: modules, packages and relative imports compiled in | ✅ |
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
`yield from` is written out as PEP 380's own expansion before the body is cut
- the whole of it, so a value sent in reaches the sub-iterator, its return
value is the value of the expression, and what is thrown or closed at the
delegating generator is passed on to it. That last part is what makes a
cancelled `asyncio` task run its `finally`: while a generator is delegating
it is not the one suspended, so closing it has to close the sub-iterator.

The two ends of a generator's life are handled where they have no block of
their own: `next` on one already exhausted raises `StopIteration` rather than
going round the dispatch again, `throw` into one that has not started comes
straight back out without running the body, closing one twice is quiet, and
`send` before the first `yield` is refused as Python refuses it.

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
889-program corpus, **886 produce byte-identical stdout and the same exit
code as CPython, nothing is refused, and the other three are not programs
this comparison can be run on**. Each was checked rather than assumed:

- `nameval` prints a function, and CPython does not match *itself* - two runs
  give `<function g at 0x104cff530>` and `<function g at 0x10036f530>`. There
  is no output for anything to agree with.
- `P17_while_masked` prints forever, in both. Twenty thousand lines of each
  hash the same; what differs is only where the timeout fell.
- `fuzz_ws` launches `python3 -m py2bin` through `subprocess` to fuzz the
  compiler. Compiled, `sys.executable` is the binary rather than an
  interpreter, so it is testing something that is not there.

So the honest reading is 886 of 886 comparable programs.

**Comparing stderr as well, 804 match.** The 82 that do not are one thing:
CPython prints a traceback - the frames, the source line, the `~~~^^^` caret -
and a compiled program prints the final `ExceptionType: message` line only,
because there are no Python frames to walk. In 77 of the 82 that last line is
character-for-character CPython's; the other five are the "Did you mean"
suggestion (three), a `SyntaxWarning` about `is` with a literal that is now
given at compile time instead (one), and a `RecursionError` that says how much
stack was used rather than naming the depth limit (one), which is the same
exception reached by the real stack rather than a counter.

**That gap is closed by shipping the source, which is the thing this does not
do.** Of the 82 tracebacks, 81 echo the line of source under the `File` line
and 71 draw a caret under the sub-expression that failed. CPython reads that
line off disk at the moment it prints - give it a filename that is not there
and it prints the `File` line alone. So a compiled program could match these
only by carrying its own source and reading it back, which is most of what
compiling was for. Frames could be synthesised on the unwind path at no cost
to the working path; the source line and the caret could not be, and they are
in almost every one of them.

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
| direct function call | **2.9 ms** | 7.1 ms | **2.48× faster** |
| a call naming an argument | **3.4 ms** | 7.8 ms | **2.27× faster** |
| integer arithmetic | **5.1 ms** | 8.3 ms | **1.62× faster** |
| `while` loop | **4.6 ms** | 6.2 ms | **1.36× faster** |
| comparisons | **3.7 ms** | 4.6 ms | **1.26× faster** |
| float arithmetic | **5.4 ms** | 5.7 ms | **1.05× faster** |
| `try` that does not raise | **3.5 ms** | 3.6 ms | **1.04× faster** |
| `in` on a list | **8.9 ms** | 9.0 ms | **1.01× faster** |
| comprehension | 5.6 ms | 5.6 ms | 1.00× |
| list append | 5.2 ms | 5.2 ms | 0.99× |
| dict store | 8.2 ms | 7.8 ms | 0.96× |
| `and` / `or` | 6.1 ms | 5.7 ms | 0.93× |
| exception raise/catch | 21.9 ms | 19.6 ms | 0.90× |
| f-string | 21.6 ms | 17.6 ms | 0.81× |
| `isinstance` | 7.5 ms | 6.0 ms | 0.81× |
| string concatenation | 15.6 ms | 12.4 ms | 0.80× |
| subscript | 8.4 ms | 6.0 ms | 0.72× |
| module global read | 5.4 ms | 3.7 ms | 0.68× |
| dict lookup by name | 6.8 ms | 4.6 ms | 0.67× |
| attribute read | 6.3 ms | 3.7 ms | 0.60× |
| chained comparison | 11.8 ms | 6.7 ms | 0.57× |
| `for` over a list | 5.3 ms | 2.8 ms | 0.54× |
| attribute write | 6.1 ms | 3.3 ms | 0.54× |
| closure call | 12.6 ms | 6.6 ms | 0.52× |
| instantiation | 37.0 ms | 16.0 ms | 0.43× |
| tuple unpacking | 15.3 ms | 5.5 ms | 0.36× |
| method call | 18.5 ms | 6.6 ms | 0.36× |

Ratios are computed from the unrounded timings, so dividing the millisecond
figures as shown gives a slightly different number in the last decimal.

One recorded run, the one in `benchmarks/last-run.json` in the repository.
Repeat it and the figures move by a few per cent either way - which rows beat
the interpreter, and by roughly how much, does not.

### Where those numbers came from

Fourteen of the twenty-seven rows sit at 0.80× or better and six beat the interpreter
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

**A named argument is placed before anything else looks at the call.** This row
was worked on three times - the caller stopped building a tuple and a dict per
call, then stopped rebuilding the keyword's name, then the callee stopped
turning `kwnames` back into a dict - and after all of it it still sat at 0.28×,
the worst on the grid. Each fix made the run-time binding cheaper; none asked
whether it had to happen at run time.

It does not. Which parameter a name is for is decided by the call site and the
`def`, and when both are in the same module both are in front of the compiler.
`f(a, step=1)` is written as `f(a, 1)` before any other pass runs. The saving
is not the matching: a keyword stopped the call being inlined and stopped it
being a direct C call, so the value came back through a `PyObject` and the loop
around it kept everything boxed. Placed, the same loop runs on machine
registers - 27.8 ms to 3.4 ms, against the interpreter's 7.8. A `**mapping`, a
name that is not a parameter, a parameter given twice, a gap, or a reordering
that would move a side effect are all left for the interpreter, as before.

Looking at it turned up something that was not about speed: `def f(a, /)`
called as `f(1, a=2)` was **accepted in silence** and answered 1, where Python
raises. It now raises what CPython raises, naming every offending parameter in
declaration order.

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
| function calls | **0.020** | 0.037 | 0.035 |
| a call naming an argument | **0.020** | 0.044 | 0.043 |
| integer arithmetic | **0.062** | 0.101 | 0.095 |
| `while` loop | **0.056** | 0.084 | 0.061 |
| nested loops | **0.022** | 0.026 | 0.028 |
| string building | **0.022** | 0.026 | 0.026 |

The keyword row is new here, and it is the one worth reading twice. Before the
placement pass it measured **0.109 s** - two and a half times slower than
Nuitka and slower than not compiling at all. The same case is now 0.020 s,
which is 2.15x faster than Nuitka rather than 2.5x slower. That is one change
in the compiler, not a faster machine: both numbers were taken on this one,
minutes apart, against the same Nuitka build.

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

### 0.9.3 - the names every module has

A bare `__spec__` was not found among a module's globals and fell through to
the builtins module - where it exists and is *its*. So `__spec__.name`
answered `"builtins"`, `__package__` came back with `builtins`' empty string,
and `__builtins__` raised a `NameError` because that is the one the builtins
module does not have.

`__package__`, `__spec__`, `__loader__` and `__builtins__` are now the
module's own, alongside `__name__`, `__file__` and `__doc__` - declared before
anything in the module is written, because a function that mentions one has
to find it there rather than out among the builtins. `__spec__` and
`__loader__` hold None: a compiled module was not loaded by anything, so
there is no loader to name.

Suite 1,808 tests, corpus 886 of 886 comparable.

### 0.9.2 - the signature and the annotations, put back together

A compiled function carries its parameters and defaults in the doc slot, in
the shape `inspect` reads a *builtin's* signature out of - and that shape has
nowhere to say what a parameter was annotated with, because CPython's own
builtins have no annotations to say. So `inspect.signature(f)` came back with
the right parameters and no types, where the same source under CPython shows
both.

The two halves are held separately, so they are put together when somebody
asks for `__signature__` and not before: almost nobody asks, and a program
that does not pays nothing for it.

Suite 1,805 tests, corpus 886 of 886 comparable.

### 0.9.1 - what a function knows about itself

**`f.__annotations__` needed the function to hold a dictionary**, and a
compiled function has no `__dict__`. Every read of it raised - and with it
`typing.get_type_hints` and `functools.singledispatch`, which reads the
annotation on a registered implementation to decide what it is for.

Annotations are now carried in the same holder that `abc.abstractmethod` and
`functools.wraps` write on: evaluated where the `def` is, or written down as
their own source where the module said `from __future__ import annotations`,
with `__globals__` beside them because that is what `get_type_hints` resolves
a written-down annotation against. Since 3.14 what asks a function about its
annotations asks for `__annotate__` and calls it - PEP 649 - so that is set
too.

Only where the program asks: the trigger is a mention of `__annotations__`,
`get_type_hints` or `singledispatch` anywhere in it, and without one an
annotated program compiles to what it compiled to before. Calls are untouched
either way - a module-level function is called directly in C and never goes
through the name. A generator and an `async def` were reporting every
annotation but `return`, which the rewrite into a machine was dropping.

**`globals()` outside the entry module read the entry's.** One slot for the
program where there wanted to be one per module - so a helper's `globals()`
answered with `__main__`'s names, and where the entry needed none at all the
slot was never declared and the program would not compile.

Corpus 886 of 886 comparable. Suite 1,803 tests, conformance corpus 105.

### 0.9.0 - what a compiled function could not do, and now can

Found by probing the language a feature at a time - seventy shapes, each
compiled and its output compared against CPython's - rather than by reading
code. Sixty-seven of the seventy now agree exactly. The three that do not are
one fact rather than three gaps, and it is named at the end.

**Two silent wrong answers, both in code people write every day.**

A default argument was evaluated on every call. Python evaluates it once, when
the `def` runs, and hands every later call the same object - which is what
makes `def f(x=[])` share one list and what the memoisation idiom rests on.
`accum(1), accum(2)` answered `[1] [2]` where Python answers `[1, 2] [1, 2]`.
Nested functions had it worse: `def each(x=i)` in a loop read the loop
variable late, so two closures both answered 1 where Python gives 0 and 1.
Defaults are now evaluated where the `def` is - in statics for a module-level
function, and in the tuple a closure already carries its captures in, which is
what makes the loop case come out right.

A closure captured names by asking *does the module bind this spelling*, and a
parameter of the enclosing function shadows a module name. So

    def d(f):
        def w(): return f() + 1
        return w
    @d
    def f(): return 1

made a `w` that called the module's `f` - which `@d` had just rebound to `w` -
and every call recursed until the stack ran out. The quieter form of the same
hole let `def d(helper)` beside `def helper()` answer from the module's
`helper`: no crash, just the wrong number. What decides a capture is now
whether the name resolves to the module's slot or to something nearer. A
module global is still not captured, on purpose - Python reads a global when
the closure runs.

**Classes.** `class C(metaclass=M)` was refused outright and now works, along
with every keyword in a class header. Three things came with it: the bases are
built before the body, which is the order Python does it in; `__prepare__` is
asked of the most derived metaclass of the bases, which is what `enum` needs -
its namespace is a mapping that notices a member name used twice, and a plain
dict is why every `Enum` subclass failed; and `__annotations__` is recorded,
which is what `dataclasses` reads to find the fields, so every dataclass used
to come out with none.

**Generators and coroutines in classes.** `def items(self): yield` did not
compile, and neither did any `async def` method - both are everywhere. A
generator becomes a machine class and a maker, and a class inside a class body
is not translated, so the machine now goes in front of the class. Two
collisions came out of that, both quiet: a method's first parameter is spelled
the same as the machine's receiver, and every name a generator binds is
rewritten into an attribute of that receiver - so the machine's own
`self.<state>` became `self.self.<state>`, the state lived on the instance,
and iterating yielded the first value for ever.

**`locals()` and `vars()`** answered `None`, and callers then tried to iterate
it. The builtin wants the frame of whoever called it; since 3.13 `locals()` in
a function is an independent snapshot, and a snapshot is what this can build
from the slots it already knows about, unbound names left out as they are
there.

**Inside an `except`, the exception is on record.** `sys.exc_info()` answered
`None` and an exception raised from a handler got no `__context__`, so a
traceback lost its "during handling of the above exception" chain. The vetted
table gains `PyErr_GetHandledException` and `PyErr_SetHandledException`, taking
it from 84 entry points to 86, and the restore rides the mechanism `finally`
already uses so it happens whichever way the clause leaves. This is the one
row that got slower for it: `exception raise/catch` goes from 0.97x to 0.90x,
which is two C-API calls per handler and is what `sys.exc_info()` costs.

**Async comprehensions** - `[x async for x in it]` and the set and dict forms -
are written out as the `async for` they are short for before the state machine
sees them, because the machine cuts at statements and a comprehension is one
expression.

**Smaller, and measured.** A loop whose sequence is empty runs its `else`
again: the flag was set up inside the body, which an empty sequence never
reaches, and it lives in a reused slot, so the `else` read what the last loop
that broke had left there. A local nothing ever reads shares one slot, which
removed the last refusal in the 889-program corpus - a generated file with
67,000 of them needed two slots and asked for 67,001. `inspect.signature`
spells a literal default as itself rather than as `None`.

**Async generators.** `async def` with a `yield` in it, and `async for` over
one. The state machine turns a `yield` and an `await` into the same thing,
and for an async generator the two go to different places - the program's own
values to whoever is iterating, the awaited ones to the event loop. The
program's are marked before the pass that expands `await` runs, so what is
marked is exactly what the program wrote; the object `__anext__` answers with
drives the machine, passes anything unmarked out to the loop, and returns the
payload of the first marked value. `asend`, `athrow` and `aclose` are not
there yet.

**`globals()` is the module's own dictionary**, not a copy of one, so a write
through it changes the program and a `del` through it unbinds. The names live
in C slots for speed and a copy would take writes and drop them - so a module
that asks for `globals()`, or for an `eval`/`exec` that would want one, keeps
its names in the module's dictionary as well and *reads* them from there. Only
the modules that ask pay for it; nothing else changes. One-argument `eval` and
`exec` are given that dictionary, which is what the frame of a module-level
`eval` would have held.

Two things showed up the moment `globals()` started answering truthfully, both
of them ours. py2bin's own start-up left `sys`, `os` and `builtins` bound in
the program's module, so `globals()` listed three names the program never
wrote; it now runs inside a function and cleans up after itself. And an
`except E as e` left `e` bound after the handler, where Python unbinds it
however the handler ends - reading it afterwards is a NameError now, as it is
there.

**Still structural**, with the reason rather than a shrug: `type(f).__name__`
answers `builtin_function_or_method`, `f.__annotations__` raises, and
`sys._getframe()` finds nothing. A compiled function is a `PyCFunction` - no
`__dict__`, no attributes, not subclassable, no frame - and the only way round
it is an interpreted call in front of every call, which is the two fastest
rows on the grid.

**Two decorators that write on functions, which used to be fatal.**
`abc.abstractmethod(f)` does exactly one thing: it sets
`f.__isabstractmethod__` and hands `f` back. `functools.wraps` does the same
six times over. A compiled function has no `__dict__`, so both failed - and
between them they are how a great many programs begin and how nearly every
decorator is written; `class Shape(ABC)` stopped the whole program at import
time. A function the source decorates with either is now handed over inside a
small object that holds what they write and binds like a method afterwards.
The mark travels with the value, which is what keeps a subclass overriding
only half of an interface abstract. Only those two, by the name they are
spelled: every other function stays the plain compiled one.

**A compiled function had no docstring.** Its doc slot carries the signature
`inspect` reads and nothing after it, so `help()` said nothing about anything
and `wraps` copied a `__doc__` of None onto every wrapper. What the function
says now follows the signature, which is where CPython reads `__doc__` from;
classes and modules keep theirs too.

**A program is more than a flat directory of files.** Modules were looked for
in one place - a `.py` beside the entry - so `import helper` was compiled in
and `import pkg` was not, and a program laid out the way most programs are
failed at start-up. Now: `a.b` is `a/b.py` or `a/b/__init__.py`, taken in
that order; a directory with no `__init__.py` is a package all the same (PEP
420); `importlib.import_module("pkg.thing")` and `__import__("helper")` are
followed wherever the name is written down; and everything under `src/` is
reached the way programs reach it, by reading the strings in the
`sys.path.insert` rather than working the expression out. Relative imports are
resolved where they are written, and one that counts past the top of the
program is refused in Python's own words. A linked module's `__file__` keeps
the shape it had in the tree. Checked against CPython on twenty-eight
laid-out programs.

**Which exception is being handled belongs to the call, not the thread.** A
`finally` now runs with the exception it interrupted on record, so what it
raises is chained to it. A body whose `except` or `finally` raised something
of its own used to leave its exception on record for good, so
`sys.exc_info()` in the caller answered with it long afterwards; every call
now gives the caller's back on the way out, and only bodies with a `try` or a
`with` save anything, so an ordinary call pays nothing.

**Smaller things, each found the same way.** `global x` in a class body binds
the module's name. `super()` in a `def` written inside a method says
"super(): no arguments", which is what Python says. A module that imports
`__main__` can read the entry's globals off it. An async generator gained
`athrow`. And the `SyntaxWarning`s CPython's compiler gives - `assert (a,
b)`, `'return' in a 'finally' block`, `"is"` with a literal - are given here
too, in CPython's words and at CPython's line, at the moment a compiled
language gives them: build time.

**Still structural**, and unlikely to change: a compiled function is a
`builtin_function_or_method`, so `type(f).__name__` and `sys._getframe()` do
not answer as they would for a Python function, and a traceback names no
source line because there is no source beside the binary to name.

Corpus 886 of 886 comparable. Suite 1,800 tests, and a conformance corpus of
103 programs run against CPython on every push.

### 0.8.9 - verdicts, borrowed references, and a leak in every `try`

The largest release so far, and most of it is correctness that turned up while
chasing speed. Fourteen of the twenty-seven measured rows sit at 0.80x or
better and six beat the interpreter.

**What moved.** Each of these is the same case before and after; the grid above
is the current measurement.

| | before | now |
|---|---|---|
| `in` on a list | 0.37x | **1.00x** |
| `try` that does not raise | 0.70x | **1.07x** |
| tuple unpacking | 0.18x | **0.36x** |
| a call naming an argument | 0.13x | **2.26x** |
| `isinstance` | 0.51x | **0.81x** |
| `and` / `or` | 0.66x | **0.92x** |
| chained comparison | 0.46x | **0.57x** |
| dict lookup by name | 0.61x | **0.72x** |
| f-string | 0.73x | **0.81x** |
| attribute write | 0.47x | **0.50x** |
| `for n, x in enumerate(...)` | 0.23x | **0.50x** |

Two rows went the other way on purpose. `exception raise/catch` gave up 1.06x
for 1.00x because the raising path now lifts the exception out so that the
*non*-raising path builds nothing at all, and `try` spent a while at 0.65x
while the leak below was fixed by releasing what it built, before it stopped
building it.

#### A condition wants a verdict, not a value

Four places computed an object and then asked what it meant. `if a and b`
evaluated the whole chain into a Python boolean; `if x in xs` looked `True` up
*by name on the builtins module* and handed it to `PyObject_IsTrue`, when
`PySequence_Contains` had already answered; `isinstance(x, C)` found the
callable and dispatched through it, where `PyObject_IsInstance` is what the
builtin does; `0 < i < n` did it once per link.

Each goes straight through now. The short circuit in `and`/`or` is the C `if`
guarding the next side, so a side that must not run has no code reached rather
than a value discarded. A chained comparison whose operands cost nothing to
read twice is rewritten into the `and` Python says it is, so each link picks up
the machine comparison a two-sided one would have had - swapping the call alone
measured nothing, because routing the chain through its own emitter had
bypassed that path.

**A `__bool__` that raised was read as true.** `PyObject_IsTrue` answers -1
with an exception set, and -1 is true in C, so a class whose `__bool__` raised
ran the body of the `if` and the program exited 0 where CPython stops. The same
for a comparison that raised. Every verdict is checked where it is produced
now. This was there before any of the work above and is the more important half
of it.

#### Nothing pays for a reference it already holds

`PyObject_CallOneArg`, `PyObject_SetAttr`, `PyObject_GetItem` and
`PyObject_Vectorcall` all borrow what they are given. The single-argument call
- the commonest call shape there is - took a reference and dropped it again,
and so did the callable, the object in `obj.field = v`, the key in `d[k]` and
`d[k] = v`, and every piece of an f-string. A pooled literal is the safest
borrow there is: it lives in a static written once at start-up, so unlike a
local it holds at module level too. All borrowed now, on rules that still
refuse to borrow a global, because anything a call runs could rebind one.

#### A call that names an argument

`f(a, key=b)` built a tuple for the positional part *and* a dict for the rest,
and made the keyword's name from a C string on every call - two allocations and
a string build to pass one argument by name. Vectorcall carries the values in
one array with their names in a tuple beside it, which is what CPython does and
what every compiled function here already accepts.

The callee stopped rebuilding a dict too: without a `**` parameter there is
nothing to hand leftovers to, so each parameter looks through the names tuple
instead, which for the one or two of each a real call has beats an allocation,
a hash per entry and a probe per parameter. Functions with `**kwargs` keep the
dict, which is what a dict is for. It is still the worst row measured.

`def f(a)` called as `f(1, b=2)` used to run and answer where CPython raises,
because a key was only removed from the keywords when there was a `**` to hand
the rest to - so nothing could tell a keyword that had been taken from one that
matched no parameter. The complaints are worded as CPython words them now,
including which parameter got two values.

#### Everything a `try` was doing that it did not need to

The classes each clause catches were built before the body ran, so every `try`
paid for them whether it needed them or not - and nothing released them where
the body raised nothing. `except (ValueError, TypeError)` builds a fresh tuple
each evaluation, so a `try` in a loop leaked one per turn: 400,000 turns held
40 MB against the interpreter's 15, and 800,000 held 65.

They are built in the handler now, where the body has already raised. Doing
that with an exception set is what CPython refuses, so the exception is lifted
with `PyErr_GetRaisedException` and put back once they are built - which is
what the match needs anyway. A `try` that does not raise builds nothing at all,
and what is never built cannot be leaked.

#### Names that were not the program's

A decorator written without the `@` - `greet = trace(greet)` - kept calling the
undecorated body, because a module-level `def` earned a direct C call keyed on
the spelling alone. Reading a global above its assignment handed the program a
raw NULL instead of raising `NameError`, and so did a class used above its
`class`. A function that rebound itself through `global` kept calling its old
body. The rule is positional now: a `def` earns the direct call only when it is
the one thing binding that name at module scope, and only where it is already
bound - so a function may still call one written below it, and recursion keeps
the direct call.

**A nested function could not call itself.** Its own name is not bound when the
capture is taken - the `def` being compiled is what binds it - so an ordinary
nested `fact` raised `NameError` on its first recursive call. Mutual recursion
between nested functions is refused at build time with an explanation, because
capture-by-value cannot express it and failing at run time naming a function
written plainly above is the worst way to say so.

#### Structure

**A long function no longer needs a bigger stack frame than a short one.** Every
intermediate the C lowering needed took a stack slot and never gave it back, so
a frame grew with a function's *length*: about 1,800 statements in one `def`
reached the 512 KB budget and the build was refused, which generated code walks
into without doing anything unusual. Forty thousand statements compile now.

**Tuple unpacking** boxed the length, boxed the expected count twice, ran two
`PyObject_RichCompare` calls and asked `PyObject_IsTrue` of each - eleven C-API
calls and five allocations for one machine comparison - and called `tuple()`
first, allocating a copy per unpack. The length is a machine comparison now and
a value that can answer for itself is taken apart where it stands, with
`tuple()` kept for what needs it: a generator has neither length nor index.

**Smaller.** `+` on strings known to be exact skips the `__add__` dispatch, and
exactness composes so `a + b + c` converts throughout. Branches whose condition
is already a constant are removed, in function bodies as well as the entry
point - 24 operations across the benchmark suite, which is a small number and
is stated rather than implied. The launcher scripts and the bootstrapper's
PowerShell fallback quote or pass by environment what they were given, so
nothing py2bin writes into a shell is read as shell.

#### The grid grew, and its headline got worse

Ten shapes were added, taking it from seventeen rows to twenty-seven. They were
found by measuring what the suite did not cover, and most were worse than
anything already in it - which is why the fraction beating the interpreter fell
from seven of seventeen to six of twenty-seven while the number at 0.80x or
better rose from ten to fourteen. A grid showing only the shapes a compiler is
good at is a grid measuring itself. One old row was also measuring the wrong
thing: `string concatenation` concatenated only literals, which fold at compile
time, so the generated C held no concatenation at all.

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

[^corpus]: Freshly measured, on this machine, at the commit that carries this
    line: each of the 889 programs compiled with `compile-capi` for the host
    and run, and its stdout and exit code compared against CPython's. The
    harness is scratch rather than committed, which is why the method is
    written out here rather than pointed at. Comparing stderr as well - which
    means comparing tracebacks a compiled program cannot produce - the figure
    is 804. What is checked on every change is the 1749-test suite.
