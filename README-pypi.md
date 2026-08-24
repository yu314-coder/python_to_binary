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

**0.9.13 fixes two bugs fatal to macOS builds** - a stack misalignment that
segfaulted every `compile-capi` x86-64 binary, and a carried interpreter whose
code signature no longer described it, which a real Intel Mac refuses to load
at all. Both were invisible on Apple silicon and under Rosetta. **If you build
for macOS, upgrade from 0.9.12 or earlier.**

**Every one of py2bin's six targets has now run on a processor of its own
architecture** - not merely built, not merely inspected, and not on an
emulator. macOS can also be built as **one universal binary** holding both
slices, which runs on Intel and Apple silicon from a single file:

```sh
py2bin compile-capi app.py --target darwin-universal2 --app --dmg -o App.app
```

All three macOS tiers have been run as universal binaries on both an Apple
silicon Mac and a real Intel one: the native binary, a one-file build, and
a frozen `.app` carrying its own CPython.

That sentence was not true a week ago, and **0.9.11 is the release that made
it true**. Both Windows targets had until then only ever been *read* here -
parsed and checked against the format, never started, because there is no
Windows machine on which to start one. The first four runs on real hardware
found four bugs, each fatal to every Windows program the compiler produced: a
native `.exe` the loader refused outright, a frozen one that discarded
everything it printed, a cross-built bundle that put `python.exe` where it
could not find its own DLLs, and a launcher stage that threw its child's
output away. None was visible to output comparison, because in all four the
program never reached a `print` - and none was in the compiled code.
**If you build for Windows, upgrade from anything before 0.9.11.**

**0.9.6 makes `dir()` work** in every scope, and stops `locals()` at module level failing to compile.

**0.9.5 made `freeze` work on Homebrew's Python**, which it did not: the
bundle carried a `bin/python3` that is a stub handing over to a file the
bundle did not have, so it built cleanly and died at start-up. It also fixes
what a frozen program's `__main__` looked like** - `__package__`
was the empty string where a script has None, and `__builtins__` the
dictionary where `__main__` has the module - and says plainly, where the tiers
are chosen between, that the native `compile` tier's integers wrap at 64 bits.

**0.9.4 stopped a `try` in a loop leaking.** A handler holds the exception it
caught and what was being handled before it, and released them only when the
clause fell off its end - so `except E: raise F(...)`, a `return` out of a
handler, and a generator whose handler suspends all leaked 160 bytes a turn.

**0.9.3 gave every module the names every module has.** A bare `__spec__`
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

### One macOS binary for both machines

`darwin-universal2` writes a single artifact holding an Intel slice and an
Apple silicon slice:

```sh
py2bin compile-capi app.py --target darwin-universal2 --app --dmg -o App.app
```

A universal `.app` built this way, carrying its own CPython, has been opened
from its `.dmg` on both an Apple silicon Mac and a real Intel one.

A universal binary is the two programs, whole and unaltered, behind a table
saying where each begins - arithmetic rather than a second compiler. Each slice
keeps its own ad-hoc signature, and the `.app` is sealed over both at once; it
passes `codesign --deep --strict`. The code is there twice; the interpreter is
not, since python.org's framework is already universal2 and a universal bundle
simply stops discarding half of it.

`freeze` can do it from a pack that kept both slices - `runtime-pack
--universal`. One file works as well, storing the payload once, after both
slices rather than inside each. Every slice is signed, x86-64 included, which
it was not before: a fat file is only as signed as its least signed slice.
Refused, with a reason: a universal `.app` *packed into one file*, since
packing re-seals the bundle and a re-signed slice would move the payload the
launcher was already told the position of.

**A 16 KB alignment rule is the thing worth knowing.** A code-signed x86-64
slice on a 4 KB boundary - what `lipo` historically recorded - is killed at
exec on Apple silicon, whose pages are 16 KB, while `codesign` still calls the
file valid and the same bytes run fine on their own. Every slice is placed on
2**14, as Apple's own universal2 builds are.

### A program that is not all one language

Python, C and web assets in one artifact, through the tier that makes real
machine code:

```sh
py2bin compile-capi app.py --native native --include web --app --onefile -o App.app
```

`--native` compiles the C for the same target as the Python and carries the
executable with it; `--include` carries a directory as it is. `py2bin make`
needs neither typed - a `native/` with a `.c` holding a `main` is compiled, and
`web/`, `assets/`, `static/`, `templates/`, `resources/`, `data/` are carried.

The C and the Python do not merge into one image - there is no linker - so the
C is a separate executable inside the bundle and the Python runs it. What is in
one file is the delivery, not the linkage.

### C++, translated to C

A class becomes a struct, a member function a free function taking the object,
a constructor something that initialises one in place - the trick the first
C++ compiler used, which needs nothing the C backend does not already have:

```sh
py2bin cc main.cpp stack.cpp -o app
```

Through: classes, single inheritance, `virtual` (a table of the object's own
functions, installed by its constructor), references, `new`/`delete` on a real
`malloc`, overloading (by argument count, and by type where that is not
enough), templates (one copy per set of arguments, named `Box__int` rather
than a hash), operators, namespaces, and exceptions - a flag and a return,
tested by the caller right after the call, with `try`/`catch` as a jump to a
label.

Three standard headers come with it, each written in py2bin's own C++ subset
and put through the same translator as your code: `<string>`, `<vector>` and
`<iostream>`. So this builds, and prints what clang++ prints:

```cpp
#include <iostream>
int main() { std::cout << "Hello, world!" << std::endl; return 0; }
```

Not implemented: multiple inheritance, `dynamic_cast` and RTTI. There is no
unwinder, so a call that can throw gets a statement of its own - one behind
`&&`, `||` or `?:` is refused with the reason rather than moved to where it
would run at the wrong time.

It is checked by building `tools/cpp_corpus/` twice, once with py2bin and once
with `clang++`, and comparing the output: 66 programs, all agreeing. clang++ is
the yardstick there and never a dependency.

### C, and a project of several files

py2bin has its own C compiler, so a C program is a native executable the same
way a Python one is:

```sh
py2bin cc main.c util.c parser.c -I include -o app
```

Name every `.c` file: there is no linker, so the whole program is compiled as
one translation unit, and a project in several files is joined before it is
compiled. Headers need nothing special, and a diagnostic still names the file
the mistake is in. `py2bin make` offers a `.c` program the same way it offers
a `.py` one.

py2bin's C compiler implements C and ships its own standard headers, with no
system include path; `<stdlib.h>` brings a real `malloc`, written in C on top
of one primitive the compiler provides. **C++ is translated to C** rather than
compiled - see above.

There is an npm wrapper, so a Node project can reach the same thing:
`npx py2bin cc main.c util.c -o app`.

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

| | `freeze` | `compile-capi` |
|---|---|---|
| **speed** on a 30M-iteration loop | 0.74 s | **0.44 s** |
| **artifact** | 24 MB | **50 KB** |
| **needs Python on the machine?** | no, it carries one | yes, or bundle it |
| **how much Python works** | **everything** | most of it: 886 of an 889-program corpus[^corpus] |
| **third-party packages** | **carried inside** | any the interpreter can import |
| what actually runs your logic | CPython, interpreting | machine code |

**`freeze` is the most complete.** It ships your program beside an interpreter
that runs it, so NumPy, Torch and a GUI toolkit all work exactly as they do
now. Nothing is translated, so nothing is faster; the artifact is the larger
of the two because an interpreter and every dependency are inside it.

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

### The third tier, which is not one of the two

`py2bin compile` is out of the table on purpose: it is not something to choose
between, because it takes a small subset of the language and no packages at
all. What it gives back is an artifact with no interpreter in it and none on
the machine - 14× faster on that loop, in 32 KB. Reach for it when *that* is
the point. It is also the compiler behind `py2bin cc`, so C goes through it
whether or not any Python does.

**Its integers are 64 bits wide and they wrap**, which is the one place py2bin
can be quietly wrong rather than refuse: `v = v * 2` seventy times answers 0
where Python answers 1180591620717411303424. Constants are folded exactly.
Both tiers in the table use the interpreter's own arithmetic and are exact.

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

**Apple M4** (10 cores, 24 GB, macOS 27.0, arm64) against **CPython 3.14.3,
python.org framework build** - the interpreter these binaries actually bind.
300,000 iterations a row, nine fresh processes each, median taken, timing only
the hot loop. Higher is better.

| | py2bin | CPython | |
|---|---|---|---|
| direct function call | **2.8 ms** | 6.9 ms | **2.43×** |
| a call naming an argument | **3.3 ms** | 7.7 ms | **2.30×** |
| integer arithmetic | **5.0 ms** | 8.2 ms | **1.65×** |
| `while` loop | **4.4 ms** | 6.4 ms | **1.45×** |
| comparisons | **3.6 ms** | 4.6 ms | **1.26×** |
| float arithmetic | **5.2 ms** | 5.6 ms | **1.07×** |
| `try` that does not raise | **3.4 ms** | 3.6 ms | **1.05×** |
| `in` on a list | **8.8 ms** | 8.9 ms | **1.02×** |
| comprehension | 5.5 ms | 5.4 ms | 0.97× |
| dict store | 8.1 ms | 7.7 ms | 0.96× |
| exception raise/catch | 21.9 ms | 19.2 ms | 0.88× |
| f-string | 21.8 ms | 17.3 ms | 0.80× |
| string concatenation | 15.6 ms | 12.3 ms | 0.78× |
| subscript | 8.3 ms | 5.9 ms | 0.71× |
| attribute read | 6.2 ms | 3.7 ms | 0.60× |
| `for` over a list | 5.2 ms | 2.8 ms | 0.54× |
| instantiation | 36.6 ms | 15.9 ms | 0.43× |
| tuple unpack | 15.0 ms | 5.5 ms | 0.37× |
| method call | 18.2 ms | 6.5 ms | 0.36× |

**Eight of twenty-seven beat the interpreter; fourteen sit at 0.80× or
better.** The wins are where a call, a lookup or an allocation stops happening
at all. The losses track how many C-API calls an operation costs, where the
interpreter's specialised bytecode does the same work inline.

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

### 0.9.13 - what a real Intel Mac refuses

Two bugs, both fatal to macOS builds, both invisible on Apple silicon and
under Rosetta. **If you build for macOS, upgrade.**

- **Every `compile-capi` x86-64 binary segfaulted** before printing anything.
  An image entered through `LC_MAIN` is *called* by dyld, so `rsp` is 8 past
  alignment on entry; a frame that is a multiple of 16 preserved that, and the
  first aligned SSE store in the callee raised a general-protection fault
  inside CPython's start-up.
- **Every macOS freeze bundle carried a mis-signed interpreter.** The framework
  is signed as a bundle, hashing an `Info.plist` and a `_CodeSignature` the
  bundle does not carry, and the stdlib beside it is pruned. Apple silicon and
  Rosetta load it anyway; real Intel refuses the dylib and the program dies
  before it runs. Anything altered is now signed again over what it is.

Not specific to Intel or to universal builds - the signature has been wrong in
every macOS freeze bundle since pruning was added; arm64 never refused one.

### 0.9.12 - all six targets have now been run

No change to the compiler; it records a verification result 0.9.11 was
published too early to carry. Windows arm64 - the one target nothing had ever
started - has since passed on a Windows 11 ARM64 virtual machine, which runs
ARM64 instructions on an ARM64 processor rather than translating them.

Every target has now had its output compared against CPython's on a machine
that actually ran it: darwin-arm64 natively, darwin-x86_64 under Rosetta 2,
both Linux targets in containers, and both Windows targets on the author's
hardware. Until 0.9.11 the two Windows targets had only ever been *read* - and
that gap is exactly where four bugs lived, each fatal to every Windows binary
py2bin produced, none of them in the compiled code.

### 0.9.11 - the Windows binaries had never been started on Windows

Every release before this one was found by compiling a program and
comparing its output against CPython, on macOS and Linux. The Windows
images had only ever been *read*, never started - there is no Windows
machine here. Then somebody started one.

Four bugs over four runs on real hardware. Each was fatal to every Windows
program the compiler produced, and none could have been caught by
comparing output, because in all four the program never reached a
`print`:

- **Every native-tier `.exe` was unloadable.** The addresses *inside* the
  import table were computed against a data section fixed at `0x2000`, true
  only while the code fitted in one page. Past that, they pointed into the
  middle of `.text` and Windows read machine code as a DLL name.
- **The frozen launcher gave its child no standard handles** and no console,
  so the program failed on its first `print` and the traceback went to the
  same missing handle.
- **A cross-built bundle put `python.exe` away from its DLLs.** The launcher
  went to the bundle root while the runtime pack kept its own `runtime/`
  directory. Windows resolves an executable's imports from its own directory,
  so `CreateProcess` refused it outright. Built *on* Windows the two are the
  same directory, which is why it had always worked.
- **The one-file stage then discarded the output** of the program it started,
  with `CreateNoWindow` on a console program - the same mistake one level
  down. The program ran, passed, and exited 0 having printed nothing.

All four are fixed and covered by tests that read the generated image the way
the loader does. Windows x86-64 passes all three tiers on physical hardware
and Windows arm64 on an ARM64 virtual machine - the target nothing had ever
started before this release. The compiled code was correct throughout; every
one of these was packaging.

### 0.9.1 - 0.9.10

Ten releases found by compiling a shape and comparing what came out against
the interpreter - roughly five hundred shapes, and the pattern was that bugs
came from *new kinds* of test rather than more of the same.

- **A program is more than a flat directory of files**: packages, submodules,
  relative imports, PEP 420 namespace directories,
  `importlib.import_module("pkg.thing")`, and everything under `src/`.
- **Three things that write on a function** - `abc.abstractmethod`,
  `functools.wraps`, and an annotated `def` writing `__annotations__` - all
  failed on a `PyCFunction` with no `__dict__`. Each is now handed something
  that can hold what it writes, and with them came `f.__doc__`,
  `inspect.signature` showing annotations, and `singledispatch`.
- **The ends of a generator's life**: `next` past exhaustion stopped answering
  at all, `close` on a fresh one complained, a delegating generator did not
  close what it delegated to - so a cancelled `asyncio` task ran no cleanup -
  and `athrow` did not exist.
- **A `try` in a loop leaked** 160 bytes a turn when its handler raised.
- **`dir()` and comprehension capture** stopped being refusals.
- `__spec__` answered `"builtins"`; `globals()` outside the entry read the
  entry's; two closures that captured nothing were the same closure; a
  Latin-1 source file was refused; and the same source compiled twice gave
  two different binaries.
- **`freeze` did not work at all on Homebrew's Python**, and **Windows ARM64**
  left an invalid instruction in the middle of the code.

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
    is 804. What is checked on every change is the 1932-test suite.
