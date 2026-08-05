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
that reads the object's internals directly. See
[what it supports](#what-compile-capi-supports) for the per-feature table.

**`compile` is the fastest and the smallest.** Python AST → py2bin IR →
optimizer → handwritten x86-64/ARM64 → ELF, PE or Mach-O. There is no
interpreter in the artifact and none on the machine: 14× faster than CPython on
that loop, in 32 KB that runs on a bare system. You pay for it in what it will
accept - integers, floats, strings, control flow, your own functions - and it
will not import a package at all.

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
| **macOS** | ✅ works | ✅ works · 📦 ships a real app |
| **Windows** | ✅ works · 📦 ships a real app | ✅ works |
| **Linux** | ✅ works | ✅ works · 📦 ships a real app |

📦 marks a target that a **complete third-party GUI application** has been
built for and run on real hardware, rather than only a corpus: ManimStudio -
10,100 lines, pywebview, Pillow, manim - as a Windows x86-64 `.exe`, a macOS
arm64 app, and a Linux arm64 executable. All three work. That is a stronger
claim than the corpus makes, and it is the author's own report from real
machines rather than something this repository's test rig can reach.

Each working target is held to the same standard: an 889-program corpus is
compiled for it and every program's output and exit code compared against
CPython's. macOS agrees on 886 and differs on 3; a 100-program slice through
Wine agrees on 93 and differs on 5. The differences are the same ones on every
platform and are inherent rather than open: CPython's "Did you mean" needs a
Python frame to suggest from, the repr of a compiled function really is a
builtin function's, and `"v" is "v"` depends on an interning the compiler does
not reproduce.

All six. Each was built from one program and the three that can run on this
machine were run - darwin-arm64 natively, both Linux targets in containers -
answering exactly what CPython answers. The two Windows targets and the Intel
Mac are built and checked structurally; there is no Windows, no Intel Mac and
no emulator here, and that is the honest limit of what *this machine* verified.
Windows x86-64 is no longer only a structural check: a real application built
for it runs on real Windows, reported by the author rather than measured here.

**iOS is not a py2bin target and this grid does not claim it.** ManimStudio
also ships on iPad and iPhone
([App Store](https://apps.apple.com/app/id6764472686)), and that build has
nothing to do with this compiler: it is a from-scratch native Swift port
embedding a full CPython 3.14 for `arm64-iphoneos` from
[python-ios-lib](https://github.com/yu314-coder/python-ios-lib), with the
App Store-compliant packaging worked out in
[CodeBench](https://github.com/yu314-coder/CodeBench). It is listed here only
so the four platforms that application runs on are not mistaken for four
targets py2bin can build - iOS forbids the JIT-adjacent and dynamic-linking
freedoms every py2bin tier depends on, and reaching it needs an embedded
interpreter and an Xcode toolchain instead.

### Building on an iPad

iOS cannot *run* a py2bin binary. It can *produce* one - and this has been
done on a real device, with the artifacts carried off and opened on the
machines they were built for.

py2bin ran inside the embedded CPython of
[ManimStudio for iPad](https://apps.apple.com/app/id6764472686) - the same
`arm64-iphoneos` Python 3.14 from
[python-ios-lib](https://github.com/yu314-coder/python-ios-lib) that
[CodeBench](https://github.com/yu314-coder/CodeBench) ships - and compiled
for three other platforms:

| built on iPadOS | artifact | carried off by | opened on the target |
|---|---|---|---|
| Windows x86-64 | `.exe` | USB | ✅ opens and runs |
| macOS arm64 | `.app`, and a `.dmg` of it | USB | ✅ opens and runs |
| Linux arm64 | ELF executable | USB | ✅ opens and runs |

The table is what the device builds *for*, and iPadOS is not among them: an
App Store app cannot `exec` an arbitrary binary, so there is no such thing as
a py2bin artifact that runs on the tablet that made it. The iPad is a build
machine here, nothing else.

**Why the tablet can do this at all** is the thing worth taking from the
table. py2bin has no compiler, assembler, linker or toolchain behind it - it
writes the machine code, the Mach-O, the PE and the ELF itself, in Python -
so a cross-build is arithmetic and file writing, which is all any sandbox
allows. Held to that claim directly: every target was compiled on this machine
with `subprocess`, `multiprocessing`, `ctypes`, `fcntl`, `pty`, and `os.fork`,
`os.execv`, `os.execve`, `os.posix_spawn`, `os.system` and `os.popen` removed
from the interpreter first. All six builds - `compile-capi` and `compile`,
across macOS, Linux and Windows - produced correct binaries with none of them
present. Nothing on that path asks the operating system for anything iOS
withholds.

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
it is for, and which of the two ways to build it - ship Python with it, or
compile it. Everything else is found or downloaded rather than typed - the
other `.py` files beside it, the libraries it imports, an interpreter for the
target, `web/` and `assets/` if they are there, and an `icon.ico` or
`icon.icns` if one is. What shape the result takes is not asked about: one
file, always, because that is the thing somebody can send.

| target | ship Python with it | compile it |
|---|---|---|
| macOS | one executable, ~14 MB | a compressed `.dmg` holding the app, ~10 MB |
| Windows | one `.exe`, ~10 MB | one `.exe` that unpacks itself, ~11 MB |
| Linux | needs a Linux machine to build on | one executable, needing Python there |

Freezing needs a whole CPython built for the target machine. One is published
for Windows and is downloaded; for anything else it has to come from a machine
like the target. So a Linux target is frozen on Linux, and where that cannot
be done the question is not asked - compiling is stated and the build goes on.
Compiling carries an interpreter on macOS and Windows; nothing is published to
carry for Linux, so a compiled Linux program uses the Python already there.

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

> **These three bundle tables were taken at 0.8.5 and have not been re-taken
> since.** Unlike every other table here they need the application's own
> virtualenv staged into wheels, which is a build this repository cannot run
> on its own. The machine is the same Apple M4 described under *Measured
> against Nuitka*; treat the sizes as accurate to a release or two ago rather
> than to today's `main`.

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

The same application has since been built for **three targets and run on real
machines** - Windows x86-64 as an `.exe`, macOS arm64, and Linux arm64 - and
works on all three. The numbers above are the macOS arm64 build, which is the
one this machine can measure; the other two are the author's report from the
hardware itself, which is evidence this repository's test rig cannot produce
for Windows at all. A fourth platform, iPad and iPhone, is
[on the App Store](https://apps.apple.com/app/id6764472686) and is **not** a
py2bin build - see the note under *Platforms*.

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
| `class C(metaclass=M)`, and the metaclass a base carries | ✅ |
| `__annotations__` in a class body, so `dataclasses` works | ✅ |
| `enum`, `dataclasses` | ✅ |
| `[x async for x in it]`, and the set and dict forms | ✅ |
| `def m(self): yield`, and `async def` methods | ✅ |
| `async def` that yields (an async generator) | ❌ refused |
| `locals()` / `vars()` inside a function | ✅ |
| `globals()`, and one-argument `eval` / `exec` | ❌ refused |
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
889-program corpus, **886 programs produce byte-identical stdout and the same
exit code as CPython, and nothing is refused**. The three that differ are
structural rather than open: the repr of a compiled function really is a
builtin function's and carries an address, so it never matches twice even
against itself; one program prints forever by design, and agrees for as long
as both are allowed to run; and one drives py2bin through `subprocess` to
fuzz it, so what it reaches is the compiled binary rather than an
interpreter.

**Comparing stderr as well, 804 match.** The 82 that do not are one thing:
CPython prints a traceback - the frames, the source line, the `~~~^^^` caret -
and a compiled program prints the final `ExceptionType: message` line only,
because there are no Python frames to walk. In 77 of the 82 that last line is
character-for-character CPython's; the other five are the "Did you mean"
suggestion (three), a compile-time `SyntaxWarning` about `is` with a literal
(one), and a `RecursionError` that says how much stack was used rather than
naming the depth limit (one), which is the same exception reached by the real
stack rather than a counter.

**That gap is closed by shipping the source, which is the thing this does not
do.** Of the 82 tracebacks, 81 echo the line of source under the `File` line
and 71 draw a caret under the sub-expression that failed. CPython reads that
line off disk at the moment it prints - give it a filename that is not there
and it prints the `File` line alone. So a compiled program could match these
only by carrying its own source and reading it back, which is most of what
compiling was for. Frames could be synthesised on the unwind path at no cost
to the working path; the source line and the caret could not be, and they are
in almost every one of them.

### The interpreter surface it may use

- A fixed table of 86 exported CPython entry points:
  `PyBytes_FromStringAndSize`, `PyCFunction_New`, `PyDict_New`,
  `PyDict_SetItem`, `PyErr_Clear`, `PyErr_ExceptionMatches`,
  `PyErr_GetHandledException`, `PyErr_GetRaisedException`, `PyErr_Occurred`,
  `PyErr_Print`, `PyErr_SetHandledException`, `PyErr_SetObject`,
  `PyErr_SetRaisedException`, `PyFile_WriteObject`, `PyFile_WriteString`,
  `PyFloat_AsDouble`, `PyFloat_FromDouble`, `PyImport_AddModule`,
  `PyImport_ImportModule`, `PyInstanceMethod_New`, `PyIter_Next`,
  `PyList_Append`, `PyList_New`, `PyList_SetItem`, `PyLong_AsLongLong`,
  `PyLong_FromLongLong`, `PyLong_FromString`, `PyNumber_Add`, `PyNumber_And`,
  `PyNumber_FloorDivide`, `PyNumber_Invert`, `PyNumber_Lshift`,
  `PyNumber_Multiply`, `PyNumber_Negative`, `PyNumber_Or`,
  `PyNumber_Positive`, `PyNumber_Power`, `PyNumber_Remainder`,
  `PyNumber_Rshift`, `PyNumber_Subtract`, `PyNumber_TrueDivide`,
  `PyNumber_Xor`, `PyObject_Call`, `PyObject_CallNoArgs`,
  `PyObject_CallOneArg`, `PyObject_DelItem`, `PyObject_Format`,
  `PyObject_GetAttr`, `PyObject_GetAttrString`, `PyObject_GetItem`,
  `PyObject_GetIter`, `PyObject_IsInstance`, `PyObject_IsTrue`,
  `PyObject_Repr`, `PyObject_RichCompare`, `PyObject_RichCompareBool`,
  `PyObject_SetAttr`, `PyObject_SetAttrString`, `PyObject_SetItem`,
  `PyObject_Size`, `PyObject_Str`, `PyObject_Vectorcall`,
  `PyObject_VectorcallMethod`, `PyRun_SimpleString`, `PySequence_Check`,
  `PySequence_Contains`, `PySequence_GetItem`, `PySlice_New`,
  `PySys_GetObject`, `PySys_WriteStdout`, `PyTuple_GetItem`, `PyTuple_New`,
  `PyTuple_Pack`, `PyTuple_SetItem`, `PyUnicode_Concat`,
  `PyUnicode_DecodeUTF8`, `PyUnicode_FromString`,
  `PyUnicode_InternFromString`, `PyUnicode_Join`, `Py_DecRef`,
  `Py_EnterRecursiveCall`, `Py_Finalize`, `Py_IncRef`, `Py_Initialize`,
  `Py_IsInitialized`, `Py_LeaveRecursiveCall`
- Every one is a real exported function - not a macro, not a `static inline` -
  with a fixed count of word-sized arguments, and a test asserts each is
  exported by the running interpreter's dylib. That is why this compiler never
  reads `Python.h`: the preprocessor could include it, but what is inside is
  macros, inline functions and struct layouts it does not implement, so the
  table is written out instead.

### How fast each one is

**Measured on an Apple M4** (10 cores - 4 performance, 6 efficiency - 24 GB,
macOS 27.0, arm64) against **CPython 3.14.3 from the python.org framework
build**, which is the interpreter these binaries actually bind. That last part
matters: this machine also carries a Homebrew 3.14.3, the two do not perform
alike, and timing against the wrong one is the easiest way to publish a number
that is not true. The harness asks a compiled program which interpreter it
ended up using and times that one.

300,000 iterations per row, each row run in nine fresh processes with the
median taken, timing only the hot loop inside the process so neither column
pays for start-up. Higher is better; `1.00×` means the same speed as CPython.

Run it yourself - the harness and every case live in [`benchmarks/`](benchmarks):

```sh
python3 benchmarks/run.py
```

| feature | py2bin | CPython | |
|---|---|---|---|
| direct function call | **3.0 ms** | 7.1 ms | **2.40× faster** |
| a call naming an argument | **3.5 ms** | 7.8 ms | **2.26× faster** |
| integer arithmetic | **5.3 ms** | 8.4 ms | **1.59× faster** |
| `while` loop | **4.7 ms** | 6.4 ms | **1.38× faster** |
| comparisons | **3.8 ms** | 4.7 ms | **1.24× faster** |
| float arithmetic | **5.5 ms** | 5.8 ms | **1.06× faster** |
| `try` that does not raise | **3.5 ms** | 3.6 ms | **1.03× faster** |
| `in` on a list | 9.0 ms | 9.0 ms | 1.00× |
| comprehension | 5.7 ms | 5.6 ms | 0.98× |
| list append | 5.2 ms | 5.1 ms | 0.98× |
| exception raise/catch | 20.7 ms | 20.0 ms | 0.97× |
| dict store | 8.5 ms | 7.9 ms | 0.93× |
| `and` / `or` | 6.2 ms | 5.7 ms | 0.92× |
| `isinstance` | 7.5 ms | 6.1 ms | 0.81× |
| f-string | 22.1 ms | 17.7 ms | 0.80× |
| string concatenation | 16.0 ms | 12.5 ms | 0.78× |
| module global read | 5.5 ms | 3.7 ms | 0.68× |
| subscript | 8.9 ms | 6.0 ms | 0.68× |
| dict lookup by name | 6.9 ms | 4.6 ms | 0.67× |
| `for` over a list | 4.4 ms | 2.9 ms | 0.65× |
| attribute read | 6.5 ms | 3.8 ms | 0.58× |
| chained comparison | 12.2 ms | 6.8 ms | 0.56× |
| closure call | 13.0 ms | 6.7 ms | 0.51× |
| attribute write | 6.5 ms | 3.0 ms | 0.46× |
| instantiation | 37.9 ms | 16.4 ms | 0.43× |
| tuple unpacking | 15.5 ms | 5.6 ms | 0.36× |
| method call | 18.8 ms | 6.7 ms | 0.36× |

Ratios are computed from the unrounded timings, so dividing the millisecond
figures as shown gives a slightly different number in the last decimal.

One recorded run, the one in `benchmarks/last-run.json`. Repeat it and the
figures move by a few per cent either way - which rows beat the interpreter,
and by roughly how much, does not.

### Where those numbers came from

Fifteen of the twenty-seven rows above sit at 0.80× or better and seven beat
the interpreter outright. Most of them did not a short while ago, and the reason
each moved is worth having in one place, because none of it was a matter of
turning something up.

The pair of numbers below is a **before-and-after of each fix**, both taken
with the harness of the day on the same machine, so the pair means what it
says. They are not comparable to the grid above, which is a fresh measurement
with different case shapes - where the two disagree, the grid above is the
current answer and this table is the history.

| row | before the fix | after it | what it was |
|---|---|---|---|
| a call naming an argument | 0.28× | **2.26×** | the name was matched to its parameter at run time, and both were known at compile time |
| direct function call | 0.81× | **2.10×** | the call hid the arithmetic from the register analysis |
| exception raise/catch | 0.49× | **1.06×** | every raise classified its argument through a Python-level `type()` |
| float arithmetic | 0.32× | **1.06×** | floats were never held in registers at all |
| comprehension | 0.77× | **0.81×** | `[x for x in it]` walked a loop `list(it)` already walks |
| attribute read | 0.51× | 0.82× | the name was built and hashed at every access |
| string concatenation | 0.14× | 0.80× | literal text was joined at run time, every time |
| list append | 0.28× | 0.72× | a lookup, a bound method and a discarded `None` per call |
| instantiation | 0.09× | 0.51× | `__init__` was reached through a Python-level wrapper |
| method call | 0.05× | 0.40× | so was every other method |

**A named argument is placed before anything else looks at the call.** This
row was worked on three times - the caller stopped building a tuple and a dict
per call, then stopped rebuilding the keyword's name, then the callee stopped
turning `kwnames` back into a dict - and after all of it the row still sat at
0.28×, the worst on the grid. Each fix made the run-time binding cheaper, and
none of them asked whether it had to happen at run time.

It does not. Which parameter a name is for is decided by two things, the call
site and the `def`, and when both are in the same module both are in front of
the compiler. `f(a, step=1)` is written as `f(a, 1)` before any other pass
runs. The saving is not the matching: a keyword stopped the call being inlined
and stopped it being a direct C call, so the value came back through a
`PyObject` and the loop around it kept everything boxed for want of knowing
what it was. Placed, the same loop runs on machine registers - 27.8 ms to 3.4
ms, against the interpreter's 7.8.

Only where it is the same call. A `**mapping`, a name that is not a parameter,
a parameter given twice, a gap with no way to write "default here", or a
reordering that would move a side effect past another one - `f(b=g(), a=h())`
calls `g` first however the parameters are ordered - are all left as written,
and the interpreter binds them at run time as before.

Looking at it turned up something that was not about speed: `def f(a, /)`
called as `f(1, a=2)` was **accepted in silence** and answered 1. A
positional-only parameter is filled from the tuple and never looked for among
the keywords, which is right when a `**kwargs` exists for a name of that
spelling to belong to; without one, Python raises and this did not. It now
raises what CPython raises, naming every offending parameter in declaration
order. A generated corpus of 328 keyword-call shapes - every signature against
every arity in every order, with and without side effects in the arguments -
put 128 of them wrong on that one point, and now agrees with CPython on all
328.

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

**An f-string of a few pieces is concatenated, not gathered and joined.** The
join allocates a tuple to be handed the pieces in and then walks it twice -
once to measure, once to fill - where a chain of concatenations allocates only
the intermediates. For the three-piece f-string most programs write that is
0.034 against 0.038; past four pieces the join's single allocation wins again
and it takes over. Both spellings concatenate rather than add, because an
f-string joins: `+` would ask the left piece's type for `__add__`, and a `str`
subclass out of a `__repr__` can override that where CPython never asks.

**A subscript on a proven list asks no protocol question.** `PySequence_Check`
is a call to answer what the bindings already settled. Dropping it is worth
about five per cent. `PySequence_GetItem` stays, though: reaching for
`PyList_GetItem` instead was measured *slower*, because the borrowed reference
it answers needs an increment that is an out-of-line call from here, where the
one `PySequence_GetItem` takes on its way out is inside the interpreter
already.

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

**A compiled program reads the arguments it was started with.** An embedded
interpreter is handed no argument vector, so `sys.argv` used to hold one entry
this compiler put there - a command-line tool could not read what it had been
asked to do. The arguments are taken from the operating system instead:
`/proc/self/cmdline` on Linux, `_NSGetArgv` on macOS, `GetCommandLineW` on
Windows. Not through the C entry point, whose signature py2bin's own C front
end fixes at `int main(void)`, and which is handed nothing at all on Windows
even where it is not fixed. It is emitted only for a program that mentions
`argv`, so one that never asks pays neither the file read nor the import.

**A builtin the program shadows is the program's.** `def len(x)` of your own,
a local `str`, a `super` bound to something else - the compiler reaches past
the name to the C entry point only when nothing in the program has bound it.
That was not always true, and the failure was silent: a module defining its
own `len` printed the length instead of calling its function.

`print` is checked at run time as well, against the object `builtins` held at
start-up, because replacing it is a thing programs actually do - harnesses
capture output that way. `len` and `str` are not, and that is a trade rather
than an oversight: the check is a dictionary probe on every call, these two
appear in the innermost loops a program has, and it measured a fifth of the
running time of a loop that calls both. Nothing replaces `builtins.len`
without breaking the interpreter's own machinery along with it, so the
exchange is a real cost against an imagined case.

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

## What it guarantees, and what it does not

A compiler that is *nearly* right about semantics is worse than a slow one,
because the difference shows up as a wrong answer rather than an error. This
is what `compile-capi` promises, and where it knowingly stops.

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


### It behaves as CPython does

**Names the program binds are the program's.** `def len(x)` of your own, a
local called `str`, a `super` bound to something else, a module-level `add`
shadowed inside a function - the compiler reaches past a name to a C entry
point only when nothing in the program has bound it. Getting this wrong is
silent, which is why each one is pinned by a test.

**Integers do not stop at 64 bits.** A local the analysis holds in a register
carries an overflow check, and the arm that overflows hands the operation to
`PyNumber_Add` and its unbounded arithmetic. `2 ** 200` is exact.

**Floats are floats.** A value that entered as `1` comes out as `1`, not
`1.0`; `-0.0` stays distinct from `0.0`, which the constant pool learned the
hard way. Division by zero raises where C would answer an infinity.

**Evaluation order is Python's.** Arguments are evaluated left to right before
the call; `print(7, 1 // 0)` writes nothing before it raises; a function
written out at its call site is only written out when the substitution
provably preserves the order and the number of evaluations.

**`__len__`, `__getitem__` and the rest run exactly once.** Every fast path
that has a slow arm hoists what it measured *out* of the arms, because the
slow arm re-evaluates its tree, and a `__len__` that printed would have
printed twice.

**Exceptions are the interpreter's.** The class, the message, the
`__cause__`, the traceback and what `except` matches all come from CPython;
nothing here re-implements them.

**`sys.argv` holds what the process was started with**, recovered from the
operating system - the embedded interpreter is handed no argument vector.

### It knowingly differs

**A generator expression is built eagerly.** `(x for x in source)` gathers
into a list and hands back an iterator over it. An infinite source will not
terminate and the memory is spent up front. Every generator in the programs
this targets is consumed immediately, which is the case the trade is made for.

**`builtins.len` and `builtins.str` replaced at run time are not observed.**
Those two go straight to `PyObject_Size` and `PyObject_Str` when the program
has not bound the name. Checking `builtins` on every call was written and
measured: a dictionary probe in the innermost loops a program has, costing a
fifth of the running time of a loop that calls both. `print` *is* checked,
because harnesses replace it to capture output and the check is nothing
against the write.

**Attribute access is slower than the interpreter's.** CPython caches
`LOAD_ATTR` against the type's version tag and reads the value straight out of
the instance. Doing the same means reading `ob_type` out of an object this
compiler treats as opaque - which is what lets one binary run against a
CPython it was not built against. The trade is deliberate; the cost is in the
table above.

**A metaclass is refused**, by name, rather than approximated.

## When it does not work

The failures worth recognising, and what each one actually means.

**`ModuleNotFoundError` for something you bundled.** The program is not
finding what was carried beside it. Check that the directory `--site` named is
where the packages actually landed; a bundle moved without it will not find
them. This was also a bug of ours through 0.8.4 on Linux, where the program
asked CPython where it was and CPython answered with its own installation -
fixed in 0.8.5.

**A wheel has no build for your target.** `--auto-fetch` says so by name and
carries on without it. PyPI has no macOS wheel for a Windows-only package and
none of pyobjc for Linux. Supply one with `--wheel-dir`, or leave the package
out with `--exclude`, which now also stops it being fetched.

**The Linux binary starts and no window opens.** pywebview and anything like
it needs a GUI toolkit, and a distribution's own bindings are built for the
distribution's Python rather than for the one you are linking. Bundle a Qt
backend with `--fetch-package PySide6-Essentials --fetch-package
PySide6-Addons`, and expect the machine to have Qt's ordinary runtime
libraries - on Debian and Ubuntu `apt install libqt6webenginecore6` pulls the
whole set.

**`libpython3.x.so` not found.** A `compile-capi` binary links the interpreter
rather than carrying one, which is what keeps it small. The machine needs that
CPython. Use `--embed-python` with `--app` on macOS to carry it instead.

**The macOS app will not open on another Mac.** Every Mach-O inside a bundle
has to be signed *after* everything is in place, which `--app` does. If you
change a file inside a built bundle, seal it again.

**`python -m build` started asking which machine to compile for.** This
repository's `build.py` shares a name with the PEP 517 packaging frontend, and
`-m` puts the working directory first on the path - so from a clone the
command found this file, took `--outdir` for the name of a program, and wrote
an app bundle where a wheel was wanted. It now notices (`__spec__` is set for
`-m` and `None` for a script) and hands over to the real frontend, or says how
to install one. `python3 build.py` is unchanged.

**`--icon` did nothing on Linux.** An ELF has nowhere to carry one; Linux
reads an application's icon from a `.desktop` entry. py2bin says so now rather
than accepting the flag in silence.

## Measured against Nuitka

**Apple M4** (10 cores - 4 performance, 6 efficiency - 24 GB, macOS 27.0,
arm64), same CPython 3.14.3 python.org framework build for all three columns,
same source. **Nuitka 4.1.3** with `--standalone`, driving Apple's clang; this
driving its own C compiler.

Whole-process time - start-up included, because that is what someone running
the artifact waits for - median of 5 runs, seconds. The cases and the runner
are in [`benchmarks/`](benchmarks):

```sh
python3 benchmarks/vs_nuitka.py
```

| | this | CPython | Nuitka |
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

Read these as whole-process numbers and not as pure throughput. Two of the
five rows finish in under thirty milliseconds, and start-up is a large share
of those - which is a real difference between the three rather than a
distortion, and the table below measures it directly. The wider margins are
the loops and the calls, where the register work and the inlining show up.

The rows are what `run.py` measures at a finer grain, and the two agree:
calls, integer arithmetic and loops are where this compiler is ahead. Its
weaknesses do not appear here at all, because a whole-program benchmark of
five loops does not touch method dispatch or instantiation - for those, read
the twenty-seven-row grid above, where they sit at 0.36×. **Put a hot loop at
module level rather than in a function and this advantage disappears**: names
at module scope are not narrowed into registers, which cost 1.41× → 0.73× on
the `while` loop when measured directly. Every case here therefore puts its
loop in a function, as any real program does.

Both compile the same source on the same machine against the same CPython
3.14. Nuitka drives Apple's clang; this drives its own C compiler, which is
the whole point of the row below it.

### What a build costs

Run time is what a user waits for; build memory is what decides whether the
build runs at all. py2bin writes the machine code, the object file and the
container in Python and never starts a C toolchain. Nuitka writes C and hands
it to Apple's clang. Peak resident set of the **whole process tree** - parent
and every descendant, summed, sampled every 25 ms:

```sh
python3 benchmarks/build_memory.py
```

Nuitka keeps a ccache and a module cache; py2bin has no build cache of any
kind. So there are two honest answers, and py2bin's column is the same in
both.

**Cold** - a first build, or any CI runner without a warm cache
(`--disable-cache=all`):

| what is being built | py2bin | | Nuitka | |
|---|---|---|---|---|
| a small program (5 cases, ~10 lines) | **42 MB** | **0.1 s** | 557-656 MB | 16.5-16.8 s |
| 200 functions | **186 MB** | **2.0 s** | 681 MB | 18.1 s |
| 1,000 functions | **602 MB** | **7.3 s** | 946 MB | 22.0 s |
| 3,000 functions | **1,567 MB** | **21.3 s** | 1,744 MB | 35.5 s |

**Warm** - Nuitka's cache in place, which is what a second build gets:

| what is being built | py2bin | | Nuitka | |
|---|---|---|---|---|
| a small program | **42 MB** | **0.1 s** | 296-297 MB | 3.7 s |
| 200 functions | **188 MB** | **2.0 s** | 423 MB | 4.9 s |
| 1,000 functions | **603 MB** | **7.4 s** | 713 MB | 8.1 s |
| 3,000 functions | 1,571 MB | 21.3 s | **1,516 MB** | **17.5 s** |

**On a small program a build costs a seventh of a warm Nuitka's and a
fifteenth of a cold one** - 40 MB against 300 or 600. That is the whole reason
an iPad can run one, and it does not depend on which column you pick.

**The advantage narrows as the program grows.** Nothing here streams - the
program, its IR, its C and its machine code are all live at once - so the
curve is steeper than clang's, which compiles a translation unit and forgets
it. Cold, py2bin is still ahead at three thousand functions; warm, it is
slightly behind on both memory and time. For anything of ordinary size the
first rows are what apply.

Measuring this is where the caching had to be pinned down: the same warm
Nuitka build came out at 1,522 MB one run and 1,175 MB the next, which would
have been published as a difference between the compilers rather than a
difference between two runs of one of them.

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
| this, `compile-capi` | **10.1 ms** | **49 KB** |
| CPython | 13.8 ms | - |
| Nuitka `--standalone` | 15.4 ms | 17.2 MB |

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

## Release notes

Newest first. Older releases are in the repository's history.

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

**A decorator written without the `@` was skipped.** `greet = trace(greet)` is
how decoration is spelled when the `@` is not convenient, and py2bin kept
calling the *undecorated* body - quietly, with the right number of arguments
and a plausible answer. A module-level `def` was compiled to a direct C call
keyed on the name alone, so anything else that bound the name later was never
consulted. A `def` now earns the direct call only when it is the one thing
binding that name at module scope.

That was one of four bugs of a single shape, all found by a differential
corpus aimed at the inliner, all present since the tier was written. The
question the compiler was asking was *does the module bind this name* where
the question is *is it bound yet*:

- `print(y)` above `y = 5` skipped the unbound test and handed the program a
  raw NULL. Under `print` that shows as `<NULL>`; anywhere else it is a crash.
  Now a `NameError`, as in Python.
- The same for a class used above its `class` statement.
- `print(later(3))` above `def later(...)` answered instead of raising: the
  callable was built *and bound* at start-up. It is still built at start-up,
  where a failure can be reported cleanly, but bound at the `def`.

The rule that replaced the old one is positional rather than textual, so
nothing ordinary is refused: a function may still call one written below it -
neither runs until the module body reaches the end - and recursion keeps the
direct call, because a function's own name is bound by the time its body runs.
Nine tests pin the behaviour; five of them fail against 0.8.6. There is no
measurable speed cost, and all sixteen `build.py` combinations were rebuilt
and re-checked.

### 0.8.6 - repairing 0.8.5, and six machines

**0.8.5 could not build anything through `py2bin make` or `build.py`.** The
change that stopped `--exclude` fetching what it excluded read the option
without a default, and an `append` option nobody passed is `None` rather than
an empty list - so every `--auto-fetch` build that did not also pass
`--exclude` stopped with a `TypeError`. That is every build the three-question
front end makes. Fixed, pinned, and every `append` option in the command line
is now checked for a default by a test.

The three questions also offer **six machines** rather than five -
`linux-x86_64` was missing - and Linux gains the one-file shape, an executable
carrying its packages and unpacking them on first run. All sixteen
target-and-shape combinations were driven through `build.py` end to end and
checked to produce the format and architecture they promise.

A helper is written out at its call site in more cases: one that names a
module constant, and one that calls another helper. `def bump(v): return
weigh(v) + 1` over `def weigh(v): return v * SCALE` collapses to `v * SCALE +
1` at the call site, which took that shape from 0.67x to 1.26x against the
interpreter. A name is only substituted when the whole module binds it at most
once, so there is only one thing it can mean.

### 0.8.5 - a correctness sweep

A long correctness sweep, and the compiled code got faster while it happened.

### Wrong answers, now right

Each of these produced a *wrong result rather than an error*, which is the
worst way for a compiler to be wrong. All are pinned by tests.

- **A name the program bound was ignored.** A module defining its own `len`
  got the builtin; so did `str`. A local `super` was rewritten into
  `super(__class__, self)` and handed two arguments to something that took
  none. A `def print` of the program's own was skipped, and its output still
  appeared - just not through the function written to produce it. A
  module-level function called through a name a nested scope had rebound
  called the wrong one.
- **A bundle could not find what it carried.** On Linux the program asked
  CPython where it was; CPython, given no argument vector, answered with its
  own installation. So a bundle looked for its packages next to
  `/usr/local/bin/python3.14` and stopped on an import of something inside
  its own executable. It asks the operating system now.
- **`sys.argv` held one entry** this compiler had put there, so a
  command-line tool could not read what it was asked to do.
- **`len(5)` answered `-1`** instead of raising, and left the `TypeError` set
  for whatever ran next to trip over.
- **A two-piece f-string ran `__add__`.** An f-string joins; `+` asks the left
  piece's type, and a `str` subclass out of a `__repr__` could answer.
- **A wheel's executable bit was dropped**, so any package shipping a helper
  program - Qt's `QtWebEngineProcess`, console scripts - could not start it.

### Things it destroyed

- **`--clean` deleted whatever was at the output path.** `-o build` or
  `-o dist` pointing at a directory holding anything else removed it, contents
  and all. A directory is now only removed when it is one py2bin could have
  built, or empty.
- **`--include` deleted its own source** when the output was in the same
  directory, which is what happens building a program in its own tree: it
  cleared the destination first, and the destination *was* the source.

### Now possible

- **`--onefile`** for a macOS `.app`, folding the bundle into the executable
  Finder runs - 498 files and 66 MB down to three files and 23 MB - and for a
  target with no bundle at all, packing the program and everything carried
  beside it into one self-extracting executable.
- **`--exclude` reaches the fetch.** It said the program would not import a
  package and then downloaded it anyway, with its whole dependency tree: 379
  MB of scientific stack beside a program told to leave it out.
- **A syntax error is reported**, with file, line, column and the offending
  line, rather than a traceback through this compiler ending at `<unknown>`.

### Faster

Seven of sixteen measured operations now beat CPython, where two did.
`direct function call` 0.81x to 2.05x, `exception raise/catch` 0.49x to 1.05x,
`float arithmetic` 0.32x to 1.09x, `method call` 0.05x to 0.40x,
`instantiation` 0.09x to 0.51x, `string concatenation` 0.14x to 0.80x. Those
were the figures at the time; the grid under *How fast each one is* is the
current measurement.

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

[^corpus]: Freshly measured, on this machine, at the commit that carries this
    line: each of the 889 programs compiled with `compile-capi` for the host
    and run, and its stdout and exit code compared against CPython's. The
    harness is scratch rather than committed, which is why the method is
    written out here rather than pointed at. Comparing stderr as well - which
    means comparing tracebacks a compiled program cannot produce - the figure
    is 804; see *It behaves as CPython does* for what the other 82 are. What
    is checked on every change is the 1732-test suite.
