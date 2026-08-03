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
that reads the object's internals directly. See
[what it supports](#what-compile-capi-supports) for the per-feature table.

The loop above is deliberately unkind to `compile-capi`: its accumulator is
compared against a parameter, which the register analysis cannot claim, so the
fast path is off. On a loop it can claim, the same tier is 1.17× faster than
CPython; a float loop, once the worst row at 0.32×, is 1.06×; and a call to a
small helper is 2.10×, because the call stops existing and the loop around it
becomes machine arithmetic.

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
CPython's. macOS agrees on 878 and differs on 7; a 100-program slice through
Wine agrees on 93 and differs on 5. The differences are the same ones on every
platform and are inherent rather than open: CPython's "Did you mean" needs a
Python frame to suggest from, the repr of a compiled function really is a
builtin function's, and `"v" is "v"` depends on an interning the compiler does
not reproduce.

All six. Each was built from one program and the three that can run on this
machine were run - darwin-arm64 natively, both Linux targets in containers -
answering exactly what CPython answers. The two Windows targets and the Intel
Mac are built and checked structurally; there is no Windows, no Intel Mac and
no emulator here, and that is the honest limit of what was verified.

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
| `make` | three questions, then a bundle - the way in with nothing to type |
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
  requirements.py       what a program needs from an index, worked out from
                        what it imports - and never guessed at
  runtime_fetch.py      verified downloads, through a downloader a caller
                        outside this package may replace
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
| `--crash-log` | write `<name>-crash.txt` beside the app if it dies, so a failure on someone else's Mac leaves evidence |
| `--dmg` | also write a compressed `.dmg` beside the `.app` - see below |
| `--include PATH` | carry a file or directory beside the program - web assets, templates, anything it opens rather than imports |
| `--onefile` | fold the bundle into its own executable, so the `.app` holds one real file |

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

### Bundling for Windows

A Windows target has no `.app` to wrap: the executable, the interpreter and
the packages share one directory. One command assembles it:

```sh
py2bin compile-capi app.py --target windows-x86_64 --crash-log \
  --runtime /path/to/embeddable-cpython \
  --bundle-site /path/to/site-packages \
  -o dist/win/MyApp.exe
```

| flag | effect |
|---|---|
| `--runtime DIR` | copy an embeddable CPython in beside the executable |
| `--bundle-site DIR` | copy packages into `Lib\site-packages` **and name it on the interpreter's path** |
| `--crash-log` | write `crash.txt` beside the program if it dies |
| `--auto-fetch` | download the interpreter instead of being told where one is |
| `--fetch-package NAME` | download that project's wheel for the target and unpack it in; repeatable |

### Three questions, and nothing to type

```sh
py2bin make                 # installed with pip
python3 build.py            # from a clone, nothing installed
python3 get-py2bin.py       # no clone either - fetches one first
```

All three ask the same three things: which file is the program, which machine
it is for, and what shape it should take. Everything else is found or
downloaded rather than typed - the other `.py` files beside it, the libraries
it imports, an interpreter for the target, `web/` and `assets/` if they are
there, and an `icon.ico` or `icon.icns` if one is.

| the shape offered first | what comes out |
|---|---|
| macOS | a compressed `.dmg` holding the app |
| Windows | one `.exe` that unpacks itself |
| Linux | one executable |

`get-py2bin.py` is the one for a machine with no pip and no clone. It also
falls back to `curl`, `wget`, `fetch` or PowerShell when Python's own
networking fails - some runtimes keep the network from the interpreter while
the shell beside it can still reach out - and lends that fallback to the
library for the downloads the build itself makes.

Nothing above needs a path on this machine. With `--auto-fetch` the
interpreter is downloaded for the target being built, and `--fetch-package`
takes a name rather than a directory:

```sh
py2bin compile-capi app.py --target windows-x86_64 --crash-log \
  --auto-fetch --fetch-package psutil \
  -o dist/win/MyApp.exe
```

Every download is checked against a hash the index published before it is
used, and cached under `~/.cache/py2bin`, so a second build does not go out
again. A wheel is unpacked rather than carried as an archive - a `.whl` on the
path is a file nothing can import.

A macOS bundle carries a macOS interpreter, and until now that meant only a
Mac could build one. It does not any more: where the machine has no framework
of its own, a portable CPython is downloaded for the target, checked against
the checksum published beside it, and laid out inside the bundle. A Mac still
uses its own, which matches everything else about it.

A project that has no wheel for the target does not stop the build. Soon
after a Python release this is ordinary: a project publishes wheels for the
interpreters that existed when it was released, so the newest version may
have nothing yet - an older one is looked for first, and if none of them fit,
the package is named and the build goes on. The program is compiled either
way and only fails if it actually reaches for what is missing.

Those last two words matter. The embeddable CPython ships a `pythonXY._pth`
naming exactly two places - the zip it came with, and the directory beside it
- and once that file exists `sys.path` is those two entries and nothing else.
Packages copied into `Lib\site-packages` are invisible until the path file
names them, and what the program reports is `ModuleNotFoundError` for a
directory plainly sitting on disk. A windowed executable has no console to say
it in, so it looks like nothing happened at all. Placing packages and naming
them is therefore one step here, not two a caller has to remember to do in
order.

Two things worth knowing when assembling one by hand:

**A wheel has to match the interpreter's ABI, not just its version.** CPython
3.14 publishes `cp314` and `cp314t` wheels, the second built for the
free-threaded interpreter. The names differ by one character and only one of
them loads.

**Prefer the console build while diagnosing.** A GUI-subsystem executable
writes nothing where you can see it; the console one prints the same thing
immediately.

An icon is embedded with `py2bin.windows_icon.install_windows_identity`, which
also sets the name and version shown in the file's properties.

### Signing, and the disk image

A macOS target is signed and sealed as the last step of the build, once the
interpreter, the packages and the program's own files are all in place. Both
halves matter and both are checked by `codesign --verify --deep --strict`,
which exits 0 on what this produces:

| what | how |
|---|---|
| the executable | an ad-hoc SHA-256 signature, which is what the kernel checks in order to run it at all on Apple Silicon |
| the bundle | `CodeResources` hashing every file that ships, with no rules excusing anything from the seal |

The signature is ad-hoc: there is no Apple Developer ID and no notarisation,
because getting either means a paid account and Apple's own tooling. The
practical difference is Gatekeeper, and Gatekeeper only inspects apps carrying
a quarantine flag - which a file copied from a USB stick does not have, and a
file downloaded through a browser does. Downloaded, the app needs one trip
through **System Settings → Privacy & Security → Open Anyway**. Right-click →
Open no longer works; Apple removed it in macOS 15.

`--dmg` writes a mountable disk image beside the bundle:

```sh
py2bin compile-capi app.py --app --dmg -o dist/MyApp.app
```

There is no `hdiutil` behind that, because there cannot be - nothing under
`src/` may reach for a subprocess. The filesystem is written byte by byte, as
ISO 9660 with Joliet rather than the HFS+ `hdiutil` would emit. Two reasons
that fits: it is simple enough to write correctly, with no catalog B-tree, no
allocation bitmap and no extents; and macOS mounts files from it executable,
which is what an `.app` needs in order to launch. Being read-only costs
nothing for something whose purpose is to be dragged to `/Applications`.

Plain ISO 9660 allows eight characters, a dot and three more, which no real
bundle survives, so every name is carried twice - mangled into that shape for
the primary descriptor, and in full UCS-2 for Joliet, which is the tree macOS
reads. A symlink is refused rather than quietly flattened.

The image is compressed, in Apple's own UDZO form: the filesystem cut into
chunks, each deflated, with a table saying where each one went. A bundle is
mostly native code, which deflates to about two fifths - a 23 MB bundle
becomes a 9.5 MB image. What it holds is unchanged; macOS inflates as the
volume is read, so the app that comes out is the same size it always was.

### Measured on a real application

manim_app: 10,100 lines, pywebview + Pillow + pyobjc, built both ways on the
same machine, same CPython 3.14, against Nuitka 4.1.3. Both bundles carry an
interpreter and 73 native extension modules, which is what makes the sizes
comparable at all.

| | py2bin | Nuitka |
|---|---|---|
| whole `.app` | **66.0 MB** | 73.5 MB |
| main binary | **8.9 MB** | 28.9 MB |
| native extensions carried | 8.7 MB | 8.7 MB |
| start with the app's imports | 84.4 ms | **78.6 ms** |
| compile time | **20.1 s** | 88.3 s |

Verified from a copy moved elsewhere on disk: every module the program imports
resolves, a pty opens and echoes, Pillow still round-trips PNG/JPEG/GIF/BMP/
WEBP/TIFF, and the app starts with no traceback.

### One file, both ways

"One file" means different things on different platforms, and on macOS it
means something particular: an application *is* a directory, because Finder
runs `Contents/MacOS/<name>` and Gatekeeper reads `Contents/Info.plist` beside
it. Nuitka says as much in its own help - `--mode=app` is "onefile except on
macOS where it creates an app bundle" - and that is the right call, not a
shortcoming. What it leaves open is how much is *inside* the bundle.

`--onefile` on a macOS build folds the payload into the bundle's own
executable. The bundle stays a bundle; what changes is that it stops being
five hundred files.

| | files | to hand over | first start | later starts |
|---|---|---|---|---|
| py2bin `--app` | 495 | 66.0 MB | 84 ms | **84 ms** |
| py2bin `--app --onefile` | **3** | 23.0 MB | 4.3 s | 134 ms |
| py2bin `--app --dmg` | 1 image | **21.8 MB** | 84 ms | **84 ms** |
| Nuitka `--mode=app` | 255 | 73.5 MB | **79 ms** | **79 ms** |

Sizes and file counts are the application; the timings are a program that
performs the application's imports, because a window that opens cannot be
timed. The packed bundle unpacks itself once into a content-addressed cache
and runs from there, which costs a few seconds the first time and about fifty
milliseconds a run afterwards - the launcher is a shell stub that checks a
marker and `exec`s. Whether that is worth it depends on what is being shipped:
three files that are easy to sign, notarise and copy, against a start that is
as fast as it can be.

Nothing verifies the payload's digest at run time - it names the cache
directory, and it is taken at build time. What guards it is the launcher's own
signature: on arm64 the archive is carried inside `__TEXT`, which the ad-hoc
signature covers, so a payload edited after the build is killed by the kernel
before the stub runs. A flipped byte in the archive was checked and ends in
SIGKILL.

A true self-extracting single *executable* - not a bundle - is what py2bin
builds on Windows and Linux, and what Nuitka's `--mode=onefile` builds where it
can. On macOS Nuitka declines that shape once pyobjc is in the graph
(`package 'Foundation' requires '--mode=app'`), which is any pywebview
program; for a plain program it builds one, 4.5 MB against py2bin's 9.5 MB
`.dmg`, and pays 261 ms of unpacking on every run unless told to cache, or
771 ms once and 38 ms after if it is.

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
| walrus (`:=`) | ✅ |
| `raise … from …` | ✅ |
| starred unpacking (`a, *b, c = …`) | ✅ |
| `match`: values, `\|`, captures, sequences, guards | ✅ |
| `match`: mapping and class patterns, `__match_args__` | ✅ |
| generators: `yield`, `send`, `yield from`, `return value` | ✅ |
| `async def` / `await`, driven by a real event loop | ✅ |
| `match`: starred sequence patterns (`[a, *rest]`) | ✅ |
| `yield` inside `try` / `except` | ✅ |
| `nonlocal`, as a cell a closure can rebind | ✅ |
| a closure over a name still moving, with Python's late binding | ✅ |
| unpacking into nested tuples, attributes, subscripts | ✅ |
| `yield`/`await` inside `try` / `finally` | ✅ |
| `yield`/`await` inside `with`, including suppression | ✅ |
| `async for` / `async with` | ✅ |

### Raising a class

`raise ValueError` names a class; `raise ValueError("x")` names an instance.
The two need different things from the C API, and asking `type()` for the
class of a *class* answers `type`, the metaclass - so the plainest raise a
Python program can write ended in

    SystemError: exception <class 'type'> is not a BaseException subclass

in every compiled program, of every kind, until it was found by an
`async for` whose protocol raises `StopAsyncIteration` without parentheses.
A class is now handed over on its own, which is the shape `PyErr_SetObject`
expects and normalises when the exception is caught.

### How `finally` and `with` are handled

The object a generator becomes is a class with `__iter__`, `__next__` and
`send` - not a generator. It is never closed and never finalised by the
collector, so the ways out of a protected region are only the ones the
rewriter can see: running off the end, and an exception on its way past. Both
are expressible, which is why these compile.

The cleanup is *not* emitted as a real `finally:` around each block. A `yield`
returns from `__next__`, so a real one would fire on the way out of every
suspension. Instead it is attached to the raising path as a synthesized
handler that runs the cleanup and re-raises, and the ordinary path reaches the
same cleanup by jumping to a block of its own.

`with` is expanded into the try it already stands for, and then takes that
same path. The manager and its `__exit__` are looked up once, on the type,
before the body runs, so rebinding the name inside cannot change which object
is left; a flag records whether a handler already dealt with an exception,
since `__exit__` is called once either way and with different arguments.
Returning true from `__exit__` suppresses, as it should.

`async for` and `async with` take the same route: each is written out as
what it stands for - a `while` over the iterator, a `try` around the body -
and the machine cuts up the result. Two details are worth recording, because
both produced answers that looked nothing like their cause. `raise X` where X
is a class had never worked at all, in any compiled program, and an
`async for` raising `StopAsyncIteration` was the first thing to notice. And a
`return` here is signalled by raising `StopIteration`, so the cleanup's
handler saw the frame leaving as a failure and passed `__aexit__` a
`StopIteration` where CPython passes `None`; the two are now told apart.

Both of the shapes this used to refuse now work, and for the same reason:
the cleanup is a block, and a block may suspend. It is reached the same way
from both paths - finishing and raising - so it can hold a `yield` of its own,
with whatever was raised waiting in a name until the cleanup is done and put
back afterwards. A `break` or `continue` leaving the region would jump to the
loop's own blocks and go round it, so a copy of the cleanup runs immediately
before the jump - which is what the jump would have reached had it left the
ordinary way. A `return` needs nothing: an earlier pass already turns it into
a jump that leaves by the ordinary exit.

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

A `finally` around a `yield` works, and so does `with`, `async for` and
`async with` - see *How `finally` and `with` are handled* above for why the
cleanup is not emitted as a real `finally:` and what is still refused.

A refusal is a `file:line:col` error, never a silent approximation. On an
889-program corpus, 878 programs produce byte-identical output to CPython; the
7 that differ do so inherently (CPython's "Did you mean" needs a Python frame,
`"v" is "v"` depends on interning) and 4 are refused outright.

### The interpreter surface it may use

- A fixed table of 83 exported CPython entry points: `PyBytes_FromStringAndSize`, `PyCFunction_New`, `PyDict_New`,
  `PyDict_SetItem`, `PyErr_Clear`, `PyErr_ExceptionMatches`,
  `PyErr_GetRaisedException`, `PyErr_Occurred`, `PyErr_Print`,
  `PyErr_SetObject`, `PyErr_SetRaisedException`, `PyFile_WriteObject`,
  `PyFile_WriteString`, `PyFloat_AsDouble`, `PyFloat_FromDouble`,
  `PyImport_AddModule`, `PyImport_ImportModule`, `PyIter_Next`,
  `PyList_Append`, `PyList_New`, `PyList_SetItem`, `PyLong_AsLongLong`, `PyLong_FromLongLong`,
  `PyLong_FromString`, `PyNumber_Add`, `PyNumber_And`,
  `PyNumber_FloorDivide`, `PyNumber_Invert`, `PyNumber_Lshift`,
  `PyNumber_Multiply`, `PyNumber_Negative`, `PyNumber_Or`,
  `PyNumber_Positive`, `PyNumber_Power`, `PyNumber_Remainder`,
  `PyNumber_Rshift`, `PyNumber_Subtract`, `PyNumber_TrueDivide`,
  `PyNumber_Xor`, `PyObject_Call`, `PyObject_CallNoArgs`,
  `PyObject_CallOneArg`, `PyObject_DelItem`, `PyObject_GetAttr`, `PyObject_GetAttrString`,
  `PyObject_GetItem`, `PyObject_GetIter`, `PyObject_IsInstance`, `PyObject_IsTrue`,
  `PyObject_Repr`, `PyObject_RichCompare`, `PyObject_RichCompareBool`,
  `PyObject_SetAttr`, `PyObject_SetAttrString`,
  `PyObject_SetItem`, `PyObject_Size`, `PyObject_Str`,
  `PyObject_Vectorcall`, `PyObject_VectorcallMethod`,
  `PyInstanceMethod_New`, `PyRun_SimpleString`, `PySequence_Check`, `PySequence_Contains`, `PySequence_GetItem`,
  `PySlice_New`, `PySys_GetObject`, `PySys_WriteStdout`, `PyTuple_GetItem`,
  `PyTuple_New`, `PyTuple_Pack`, `PyTuple_SetItem`, `PyUnicode_Concat`, `PyUnicode_DecodeUTF8`, `PyUnicode_InternFromString`,
  `PyUnicode_Join`,
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

Nine of the sixteen rows above sit at 0.80× or better and six beat the
interpreter outright. Most of them did not a short while ago, and the reason
each moved is worth having in one place, because none of it was a matter of
turning something up.

| row | was | now | what it was |
|---|---|---|---|
| direct function call | 0.81× | **2.10×** | the call hid the arithmetic from the register analysis |
| exception raise/catch | 0.49× | **1.06×** | every raise classified its argument through a Python-level `type()` |
| float arithmetic | 0.32× | **1.06×** | floats were never held in registers at all |
| comprehension | 0.77× | **0.81×** | `[x for x in it]` walked a loop `list(it)` already walks |
| attribute read | 0.51× | 0.82× | the name was built and hashed at every access |
| string concatenation | 0.14× | 0.80× | literal text was joined at run time, every time |
| list append | 0.28× | 0.72× | a lookup, a bound method and a discarded `None` per call |
| instantiation | 0.09× | 0.51× | `__init__` was reached through a Python-level wrapper |
| method call | 0.05× | 0.40× | so was every other method |

Two things run through the whole list. The first is that the largest wins were
not optimisations but *mistakes being removed* - a wrapper written in Python on
the method path, a float analysis that did not exist, a string rebuilt on every
iteration of a loop. The second is that the wins all have one shape: something
stops happening. Adding a cheap test to skip expensive work inside the
interpreter was tried five separate times and measured flat or slower every
time, because the extra call through the import table cost more than it saved.
Those attempts are recorded next to the code that does not do them.

**Arithmetic loops win** because a local the analysis picks out is held in a
machine register - a `long long` for an integer, a `double` for a float - with
an overflow check that falls back to unbounded arithmetic when an integer
leaves the word. That is what CPython's specialising interpreter does, and
doing anything less was what made this tier slower than not compiling at all.
Literal arithmetic is folded before any of it, so `1.5 * 2.0 - 0.5` is the one
constant it computes to.

**A method body is treated like any other function body**, which it was not.
A method is written while the module's own statements are being emitted, and
the flag saying "this is module level" was still set inside it - so the
register analysis and the borrowing below were switched off for every method
in every class. Turning them on there is worth more than either was worth
anywhere else.

**A name bound only by displays is known exactly, with no guard.** `xs = []`
can only ever be a `list`: a display makes one, mutation never changes a type,
and only rebinding could - so a name whose every binding is a display holds
its type as a compile-time fact. `xs.append(v)` is then `PyList_Append` - the
lookup, the bound method and both dispatch layers all go - and `d[k] = v` is
`PyDict_SetItem`. `xs = MyList()` is a call, not a display, so a subclass
keeps its override. This is the static form of the run-time guard measured
seventeen per cent slower above: the knowledge is free because it is decided
once, from the bindings.

**`raise ValueError('x')` skips the class-or-instance question.** Every raise
ran `type(value)` - through a Python-level call - to decide whether it was
handed a class or an instance. For an untouched builtin exception name the
answer is known at compile time: a call on the class answers an instance, the
bare name is the class. The lookup itself stays live, so a replaced
`builtins.ValueError` is still honoured. Exceptions now run at 1.06×.

**`len()` inside a hot expression is a machine integer.** `n = n + len(s)`
was three heap allocations to add a number the C already had: `PyObject_Size`
answers a `long long`, which was boxed, added to a boxed `n`, and stored as an
object - unmaking `n`'s register form for the rest of the loop. The
measurement is hoisted into a slot of the emitter's own, once, before the fast
path's two arms - which is what keeps a program's `__len__` from running twice
when the fast arm declines. `while i < len(xs)` gets the same treatment, and
still measures every iteration, so a list that grows mid-loop is seen growing.

**A comprehension over a `range` counts in a register.** Its target becomes a
name of the emitter's own, registered with the integer analysis, so `x * 2` in
the element is machine arithmetic boxed once, by the store; and with no filter
the list is made at its final length and filled with `PyList_SetItem`, which
steals the reference and never grows the storage. `[x for x in it]` - the
identity - is `list(it)` to the letter and is emitted as exactly that call.
The comprehension row below is the general shape, not the identity: measuring
the shape the optimisation is best at and calling it "comprehension" would be
the benchmark measuring itself.

**An operand that is a local is read without taking a reference.** A local, a
parameter and a capture are C variables this function alone writes, and each
holds its reference for the whole body, so the increment and decrement around
every read were two memory writes to arrive back where they started. A
*global* is not borrowed: anything called while the expression runs can rebind
a module-level name, and the reference the slot held may have been the last
one. Nor is a name any `:=` assigns, that being the one thing which writes a
slot in the middle of an expression.

**The one mechanism that could close this was looked for and found, and it is
still not enough.** An inline cache needs two halves: something to invalidate
it when a class changes, and a cheap way to check that the object in hand is
the type the cache was filled for. `PyType_AddWatcher` and `PyType_Watch`
(3.12) supply the first, through the documented API, with no object layout
involved. The second does not exist: there is no public way to read an
object's type without taking a reference to it. `PyObject_Type` is an
out-of-line call plus a reference count, and that is already more than a
guarded fast path saves - measured twice, once on `list.append`, where adding
the guard made it seventeen per cent *slower*, because a builtin method
already reaches CPython's fast path. Nuitka does specialise these, and can,
because it includes `Python.h` and reads `ob_type` directly. That is the trade
this compiler makes the other way, and it is why one binary here runs against
a CPython it was not built against.

**Attribute access still loses**, and the reason is worth stating plainly
rather than leaving as a number. CPython caches a `LOAD_ATTR` against the
type's version tag and, on a hit, reads the value straight out of the instance
without a lookup. Doing the same here means reading `ob_type` and its version
out of the object, and this compiler treats `PyObject` as opaque and goes
through the documented entry points - which is what lets one binary work
against a CPython it was not built against. `PyObject_GetAttr` does the full
generic lookup every time. That is a deliberate trade, not an oversight.

**Calls that can be written out disappear entirely.** A module-level function
whose body is one expression naming nothing but its own parameters is
substituted at the call site. The gain is not the call saved: `t = add(t, i)`
tells the register analysis nothing, because the value arrives from a call,
while `t = t + i` tells it everything - so the loop around it narrows and the
whole thing becomes machine arithmetic. That is what takes this row past the
interpreter rather than merely level with it.

**Method calls are the worst row, and it was tried.** A compiled method is
wrapped in `instancemethod`, which binds correctly but does not carry
`Py_TPFLAGS_METHOD_DESCRIPTOR`; CPython checks that flag when it fetches a
method to call and when it calls `__init__`, and skips allocating a bound
method for anything that has it. `PyDescr_NewMethod` produces a descriptor
that does. It was implemented, and it works, and it was **reverted**: a method
reached that way is a `builtin_function_or_method` rather than a bound method,
so `inspect.ismethod` answers False and `self` appears in the signature. That
is precisely the breakage that once stopped pywebview from binding a single
method of a compiled application. Compiled methods introspect like Python
methods, and they will keep doing so; the speed is not worth what it costs to
every framework that looks at them.

**What is left below the line loses by a factor that tracks how many C-API
calls the operation costs.** Each is a real call through the import table, with
the reference-count discipline around it, where the interpreter's specialised
bytecode does the same work inline in its own loop. That is also the shape of
every optimisation here that *failed*: five separate attempts to add a cheap
test in order to skip work inside libpython measured flat or slower, because
the extra out-of-line call cost more than the work it skipped. The wins all
have the same shape instead - a call, a lookup or an allocation stops happening
at all.

**Method calls used to be far worse than that pattern predicted** - 21×, where
everything else paid 2-4×. That was not C-API overhead. Every compiled method
was wrapped in `functools.partialmethod` to make it bind, and that wrapper's
`__get__` is written in Python, so each `obj.method` ran interpreted code and
built a `functools.partial` before the call could start. CPython's own
`instancemethod` does the same binding in C. Instantiation was the same wrapper
showing up in `__init__`. Both are now within the general pattern rather than
outside it.

## Measured against Nuitka

Same machine (arm64 macOS), same CPython 3.14, same source. Nuitka 2.x with
`--standalone`, driving Apple's clang; this driving its own C compiler.

Run time, median of 5 runs, seconds, re-measured against **Nuitka 4.1.3** -
several major versions newer than the 2.x these were first taken against, and
faster for it:

| | this | CPython | Nuitka |
|---|---|---|---|
| integer arithmetic | 0.095 | 0.112 | **0.094** |
| `while` loop | **0.046** | 0.054 | 0.053 |
| nested loops | 0.018 | **0.017** | 0.018 |
| function calls | **0.015** | 0.033 | 0.027 |
| string building | 0.016 | **0.012** | 0.014 |

Nuitka moved too. These were first taken against Nuitka 2.x, and against 4.1.3
it is faster than it was: integer arithmetic is now a tie rather than a win,
and it has closed most of the loop gap. Where the two still differ is calls -
a small helper compiled here is written out at the call site, which takes the
arithmetic around it into registers as well, and no amount of a *better* call
reaches a call that is not made. String building is the row to look at if you
want the honest weakness: both compilers lose to the interpreter on it, and
this one loses by more.

Both compile the same source on the same machine against the same CPython
3.14. Nuitka drives Apple's clang; this drives its own C compiler, which is
the whole point of the row below it.

Startup, `print("x")`, median of 13 runs:

| | startup | on disk |
|---|---|---|
| this, `compile-capi` | **12.6 ms** | **49 KB** |
| CPython | 14.7 ms | - |
| Nuitka `--standalone` | 18.1 ms | 17 MB |

A `compile-capi` binary links the interpreter it was built against and starts
by calling `Py_Initialize`, skipping the interpreter's own startup path -
scanning `sys.path`, finding and unmarshalling `__main__`. Nuitka's standalone
bundle pays to bootstrap its own tree first.

**Loops are faster than the interpreter, and so now are calls.** The reason is
worth stating rather than hiding, because it explains both halves - and because
the second half only became true recently.

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

[^corpus]: That sweep was last run before the optimisation work described under
    [how fast each one is](#how-fast-each-one-is), and the harness that ran it
    was scratch rather than committed, so the figure is reported as measured
    rather than as currently verified. What is checked on every change is the
    1529-test suite and a differential set that compiles each program and
    demands byte-identical output to CPython.
