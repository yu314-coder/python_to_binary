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
call where the interpreter has specialised bytecode. The per-feature table is below.

The loop above is deliberately unkind to `compile-capi`: its accumulator is
compared against a parameter, which the register analysis cannot claim, so the
fast path is off. On a loop it can claim, the same tier is 1.67× faster than
CPython.

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

## Bundling for Windows

The runtime sits beside the executable rather than in a bundle. Two traps,
both silent:

The embeddable CPython ships a `pythonXY._pth` naming exactly two places, and
`sys.path` is those and nothing else - packages in `Lib\site-packages` are
invisible until that file names them, and the program reports
`ModuleNotFoundError` for something plainly on disk. And a wheel must match the
interpreter's ABI, not only its version: `cp314` and `cp314t` are named almost
identically, the second is for the free-threaded build, and only one loads.

`--crash-log` matters here, because a GUI-subsystem executable writes nothing
to a console; with it the program leaves `crash.txt` beside itself.

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
| `yield`/`await` inside `try` / `finally` or `with` | ❌ |
| `async for` / `async with` | ❌ |

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

A `finally` is different and is refused. It has to run when the generator is
*closed* - abandoned, garbage collected - and nothing here is told about a
close. Cleanup that silently does not happen is worse than a refusal, and
`with` is a `finally`, so it is refused too. `async` needs all of this plus a
scheduler and the coroutine protocol.

A refusal is a `file:line:col` error, never a silent approximation. On an
889-program corpus, 878 programs produce byte-identical output to CPython; the
7 that differ do so inherently (CPython's "Did you mean" needs a Python frame,
`"v" is "v"` depends on interning) and 4 are refused outright.

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
