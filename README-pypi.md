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
| **Windows** | ✅ works | ✅ works |
| **Linux** | ✅ works | ✅ works |

Each working target is held to the same standard: an 889-program corpus is
compiled for it and every program's output and exit code compared against
CPython's. macOS agrees on 878 and differs on 7; a 100-program slice run
through Wine agrees on 93 and differs on 5. The differences are the same on
every platform and are inherent rather than open - CPython's "Did you mean"
needs a Python frame to suggest from, and the repr of a compiled function
really is a builtin function's.

The **native** tier (`py2bin compile`, no CPython at all) targets all six.

## The paths through it

Three ways to turn a program into an artifact. They trade the same three things
against each other, and which one you want depends on which you care about.

| | `compile` | `compile-capi` | `freeze` |
|---|---|---|---|
| **speed** on a 30M-iteration loop | **0.062 s** | 1.27 s | 0.66 s |
| **artifact** | **48 KB** | 66 KB | tens of MB |
| **needs Python on the machine?** | **no** | yes, or bundle it | no, it carries one |
| **how much Python works** | a small subset | most of it: 878 of an 889-program corpus[^corpus] | **everything** |
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

300,000 iterations, median of 5, against the interpreter it links. Higher is
better; `1.00×` means the same speed as CPython.

| feature | py2bin | CPython | |
|---|---|---|---|
| direct function call | **4.5 ms** | 9.6 ms | **2.10× faster** |
| comparisons | **4.9 ms** | 5.8 ms | **1.19× faster** |
| `while` loop | **4.5 ms** | 5.4 ms | **1.19× faster** |
| integer arithmetic | **9.3 ms** | 10.9 ms | **1.17× faster** |
| float arithmetic | **6.3 ms** | 6.7 ms | **1.06× faster** |
| exception raise/catch | **21.1 ms** | 22.4 ms | **1.06× faster** |
| attribute read | 7.0 ms | 5.7 ms | 0.82× |
| comprehension | 3.9 ms | 3.2 ms | 0.81× |
| string concatenation | 22.6 ms | 18.1 ms | 0.80× |
| dict store | 12.0 ms | 9.0 ms | 0.75× |
| list append | 8.8 ms | 6.3 ms | 0.72× |
| closure call | 13.9 ms | 9.3 ms | 0.67× |
| subscript | 12.2 ms | 8.1 ms | 0.67× |
| f-string | 32.2 ms | 20.3 ms | 0.63× |
| instantiation | 35.3 ms | 18.1 ms | 0.51× |
| method call | 28.1 ms | 11.2 ms | 0.40× |

### Where those numbers came from

Nine of the sixteen rows sit at 0.80× or better and six beat the interpreter
outright. Most did not a short while ago.

| row | was | now | what it was |
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

| | py2bin | Nuitka |
|---|---|---|
| whole `.app` | **66.0 MB** | 73.5 MB |
| main binary | **8.9 MB** | 28.9 MB |
| native extensions carried | 8.7 MB | 8.7 MB |
| start with the app's imports | 84.4 ms | **78.6 ms** |
| compile time | **20.1 s** | 88.3 s |

Run time, median of 5, seconds, re-measured against **Nuitka 4.1.3**:

| workload | py2bin | CPython | Nuitka |
|---|---|---|---|
| integer arithmetic | 0.095 | 0.112 | **0.094** |
| `while` loop | **0.046** | 0.054 | 0.053 |
| nested loops | 0.018 | **0.017** | 0.018 |
| function calls | **0.015** | 0.033 | 0.027 |
| string building | 0.016 | **0.012** | 0.014 |

These were first taken against Nuitka 2.x; against 4.1.3 it is faster than
it was, and integer arithmetic is now a tie rather than a win. Where the two
still differ is calls - a small helper is written out at the call site here,
and no better call reaches a call that is not made. String building is the
honest weakness: both lose to the interpreter, and this one by more.

Loops beat both because a local the analysis picks out is held in a register
rather than on the heap, with the overflow check that falls back to unbounded
arithmetic when it leaves the word. Calls still lose: an argument is boxed at
the call and unboxed inside, where the interpreter's specialised call pays
neither.

## Licence

MIT. Full documentation, source and issues:
**https://github.com/yu314-coder/python_to_binary**

[^corpus]: That sweep was last run before the optimisation work described
    further down, and its harness was scratch rather than committed, so the
    figure is reported as measured rather than as currently verified. What is
    checked on every change is the 1529-test suite and a differential set that
    demands byte-identical output to CPython.
