# python-to-binary

`py2bin` turns Python into machine code using nothing but the Python standard
library. No Cython, Nuitka, mypyc, Rust, C, C++, PyInstaller, PPCI, bootloader,
assembler, linker or SDK - and no `gcc` or `clang` at any point. The only thing
a build needs is an interpreter.

```sh
pip install python-to-binary
py2bin compile-capi app.py --target darwin-arm64 -o app
```

**Where it stands.** 2,019 tests; a 110-program corpus whose output matches
CPython character for character; 886 of an 889-program corpus likewise, with
the other three not comparable by anything; 1,494 of 1,500 randomly generated
programs; 314 C and C++ programs whose output matches `clang++`, built for all
six targets; eight of twenty-seven benchmark rows faster than the interpreter.
Every one of those numbers is measured, and where a number is not what it
looks like the section that gives it says so.

**All six targets have run on a processor of their own architecture** -
darwin-arm64 here, both Linux targets in containers, and darwin-x86_64,
windows-x86_64 and windows-arm64 on real machines. Nothing in that claim rests
on emulation. macOS can also be built as **one universal binary** that holds
both slices and runs on either machine:

```sh
py2bin compile-capi app.py --target darwin-universal2 --app --dmg -o App.app
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
that reads the object's internals directly. See
[what it supports](#what-compile-capi-supports) for the per-feature table.

### The third tier, which is not one of the two

`py2bin compile` is deliberately out of the table above, because it is not
something to choose between: it accepts a small subset of the language and no
packages at all. Python AST → py2bin IR → optimizer → handwritten
x86-64/ARM64 → ELF, PE or Mach-O. No interpreter in the artifact and none on
the machine: 14× faster than CPython on that loop, in 32 KB that runs on a
bare system. Reach for it when *that* is the point, not when you are deciding
how to ship a program. It is also the compiler behind `py2bin cc`, so C goes
through it whether or not any Python does.

**Its integers are 64 bits wide and they wrap.** That is the one place in
py2bin where a program can be quietly wrong rather than refused, so it is
worth saying plainly: a runtime integer in this tier is a machine word, and
`v = v * 2` run seventy times answers 0 where Python answers
1180591620717411303424. Constants are folded exactly - `2 ** 70` written down
is right - so it is only values the program computes as it runs. Both tiers in
the table use the interpreter's own arithmetic and are exact. If your program
counts past 2^63 anywhere, this is not the tier for it.

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

All six. Each was built from one program, and the four that can be run on this
machine were run here - darwin-arm64 natively, both Linux targets in
containers, and darwin-x86_64 under Rosetta 2 - answering exactly what CPython
answers. darwin-x86_64 and the two Windows targets have since been run on real
hardware of their own architecture; see below.

**Windows x86-64 has since been run on real Windows hardware**, all three
tiers: the native `.exe`, the frozen `.exe` carrying its own CPython, and the
C-API `.exe` driving a downloaded CPython 3.14. That run is the author's,
on a physical machine, not something measured here - and it was worth doing.
It found four bugs that no amount of reading the images had caught, every one
of them fatal to every Windows program the compiler produced, and every one in
the packaging rather than in the compiled code. They are fixed and covered by
tests; the first three releases of this section describe them.

**darwin-x86_64 has been run on a real Intel Mac** - a MacBookPro16,1 - across
all three tiers, each as a universal binary: the native one, a one-file build,
and a frozen `.app` carrying its own CPython. All pass.

It passes under Rosetta 2 here as well, and the difference between those two
sentences is the most expensive thing in this section. Rosetta ran these
perfectly while they carried **two** faults a real Intel CPU refuses outright:
a misaligned stack at the first call into CPython, and a carried interpreter
whose code signature no longer described it. Rosetta enforces neither SSE
alignment nor dylib signatures. What it can tell you is that a program is
*correct*; it cannot tell you the program is *well formed*, and those are
different questions.

**Windows arm64 passes too**, on a Windows 11 ARM64 virtual machine - which
runs ARM64 instructions on an ARM64 processor, so the generated code is
executed rather than translated. The author's report, like the x86-64 one.

That closes the grid, and closes it properly: **every one of the six targets
has now run on a processor of its own architecture** - darwin-arm64 here, both
Linux targets in containers, and darwin-x86_64, windows-x86_64 and
windows-arm64 on the author's machines. Nothing in that list rests on
emulation or translation any more.

It took the whole way round to get there. Two of the six had only ever been
*read*, which missed four bugs that made every Windows binary unusable; and
the sixth was passing under Rosetta while carrying a fault that stopped it
dead on the hardware it was built for.

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

### One macOS binary for both machines

`darwin-universal2` writes a single artifact holding an Intel slice and an
Apple silicon slice, the way Apple's own universal2 builds do:

```sh
py2bin compile-capi app.py --target darwin-universal2 --app --dmg -o App.app
py2bin compile      app.py --target darwin-universal2 -o app
```

A universal binary is not a merged program. It is the two programs, whole and
unaltered, behind a small table saying where each begins - which is why this is
arithmetic rather than a second compiler. Each slice keeps its own ad-hoc
signature, because a signature covers the bytes of the image it was written
over and knows nothing about the wrapper they were later placed in. The `.app`
is sealed afterwards over both slices at once, and passes
`codesign --deep --strict`.

What it costs is size: the code is there twice. The interpreter is not - the
python.org framework is already universal2, and a universal bundle simply stops
throwing half of it away.

**`freeze` can do it too**, from a runtime pack that has kept both slices:

```sh
py2bin runtime-pack --universal -o pack
py2bin freeze app.py --runtime-pack pack --target darwin-universal2 --app --onedir -o App.app
```

`--universal` is asked for rather than detected. python.org's framework is
universal whether or not anyone wants a universal bundle out of it, and quietly
keeping both slices would double the size of every bundle built the way they
always were.

**One file works too**, and stores the payload once rather than per slice:

```sh
py2bin freeze app.py --runtime-pack pack --target darwin-universal2 --onefile -o app
```

The archive goes *after* both slices rather than inside each - an image has to
be told where its payload is, not contain it - and both slices are told the
same position in the finished file. That position is not known until the layout
is, and the layout does not move when the position changes, because the command
pads the number to a fixed width; so the launcher is built twice and the second
pass asserts the length did not change.

**Every slice is signed**, including x86-64, which was emitted unsigned for
years because Intel macOS never asked for one. That stopped being harmless when
the two were joined: a fat file is only as signed as its least signed slice, so
one unsigned half made a whole bundle report as unsigned however carefully the
other had been sealed.

One combination is refused with a reason: a universal `.app` **packed into one
file**. Packing re-seals the bundle, a re-signed slice is not always the length
it was, and here that would move the payload the launcher has already been told
the position of. A universal `.app` and a universal one-file each work; it is
only the two together.

**Intel found a second alignment bug, and only Intel could.** System V wants
`rsp` 16-byte aligned *at the call instruction*. An image the kernel starts is
already aligned; one entered through `LC_MAIN` is not, because dyld *calls* it
and its return address is already on the stack. The entry frame was a multiple
of 16, which preserved that 8 and handed every call from the entry a stack
misaligned by exactly that - and the first `movaps` to a stack slot in the
callee raises a general-protection fault. CPython's start-up does one, so a
`compile-capi` binary segfaulted inside `_PyRuntimeState_Init` before printing
anything.

Rosetta 2 does not enforce the alignment, so this ran perfectly on Apple
silicon, corpus and all. It took a crash report from a real Intel Mac, where
`rbp` and `rsp` were both 8 mod 16 in a frame whose prologue leaves `rbp` at 0.
Internal functions were never affected: they `push rbp` first, which corrects
the 8 before anything else happens. It was the entry alone. **Fixed and
confirmed on the machine that found it.**

**A 16 KB alignment rule is the other thing worth knowing here.** A code-signed
x86-64 slice placed on a 4 KB boundary - which is what `lipo` historically
recorded - is killed at exec on Apple silicon, whose pages are 16 KB. Nothing
about the file says so: `codesign` calls it valid, and the same bytes copied
back out to a file of their own run perfectly. Only in place, at the wrong
offset, does it die, and an *unsigned* slice survives it, which is what makes
the symptom so misleading. Every slice is placed on 2**14, as Apple's own
universal2 builds are.

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

**A program written against somebody else's component builds there too.**
SidecarBridge is a Windows WebView2 app in C++: three sources, a header from
the WebView2 SDK, and the loader DLL that header is for. With `--auto-fetch`
and nothing else named, a folder holding only the three sources produces a
`dist/` with the executable and that DLL beside it, on both `windows-x86_64`
and `windows-arm64`. The header and the library are the only things that came
from anywhere else, and both came over HTTPS from the package that publishes
them - which is the one thing a tablet can do that a toolchain cannot be made
to.

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

### C, and a project of several files

py2bin has its own C compiler, so a C program is a native executable the same
way a Python one is - and with the same absence of a toolchain behind it:

```sh
py2bin cc main.c util.c parser.c -I include -o app
```

Name every `.c` file. **There is no linker**, so the whole program is compiled
as one translation unit; a project split across several files is joined into
one before it is compiled, the way a unity build has always got a single
translation unit out of many. Headers need nothing special - an include guard
is exactly what makes including one twice harmless. A diagnostic still names
the file the mistake is in, because the joined text is mapped back before
anything is reported.

Two files that each define the same `static helper` will collide, which
separate translation units would have allowed. That is reported against the
real file and line rather than guessed at.

`build.py` and `py2bin make` offer a `.c` program the same way they offer a
`.py` one: any `.c` beside it that does not define its own `main` is compiled
with it, and an `include/` directory beside it is searched.

**A `static` object inside a block** is one object with the scope of the name
that declares it, and it keeps its value between calls. That used to be
refused, and for a reason: py2bin inlines a body rather than calling it, so a
function compiled into three call sites would have got three objects instead
of the one C promises. The slot is keyed by the declaration itself now, so
every inlining of that body names the same one — and the initial value is
written with the file-scope objects, because a store where the declaration
stands would run again on every call.

**`printf` is compiled, not called.** The format is read at compile time and
the formatting code is emitted for it, which is why there is no C library
underneath. It writes `%d %i %u %x %X %o %c %s %f %F %e %E %g %G` with the
`h`/`hh`/`l`/`ll`/`z` length modifiers, a precision on the floating ones, and
the `-`, `+`, space and `0` flags with a field width — `%5d`, `%-8s`,
`%08.2f`, `%+d` all pad exactly as C says, including putting the zeros of a
zero-padded field after the sign rather than before it. A width given as `*`
is refused with the reason: it comes from an argument, and the format is read
before there are any.

`sprintf` and `snprintf` are the same formatter pointed at a buffer instead of
at stdout. `snprintf` keeps what fits and answers the length it *would* have
written — which is what lets a caller ask how much room to make — and puts the
terminator where the copy stopped. A size of zero writes nothing at all, as C
says. A program that defines its own `printf`, `sprintf` or `snprintf` gets
that one; these are only what a program calls without having written.

`swprintf` and `swprintf_s` are the same formatter again, storing two bytes a
character on Windows and four elsewhere - whatever `wchar_t` is on the target.
`%ls` reads a wide string and everything else writes the same characters it
always did, one to a cell instead of one to a byte. C++'s array overload -
`swprintf_s(buffer, L"...")`, with no count - takes its room from the array it
is writing to, which is the whole reason a program reaches for it.

**Variadic functions** work, and `<stdarg.h>` is the typedef that goes with
them. The arguments past the named ones are promoted the way C promotes them —
narrow integers to `int`, `float` to `double` — and written into a run of
eight-byte cells whose address travels as one more argument; so a `va_list` is
a pointer into those cells, `va_arg` is a load and a step forward, and `va_end`
has nothing to undo. Passing it as an argument rather than finding it in the
frame is what makes it work the same whether the callee was inlined or really
called, and it means a `va_list` can be handed on to another function, which is
how every logging helper in C is written.

**`#pragma`.** `once` is honoured, and so is every pragma that provably says
nothing about the program: `warning`, `region`, `message`, `comment`, and any
whose second word is `diagnostic` or `system_header` — which is how every
compiler that has them spells them, and which one it is addressed to does not
matter. Refusing those stopped ordinary headers on
their first line — `WebView2.h` begins with `#pragma warning( disable: 4049 )`
— and none of them can change what the C means.

`#pragma pack` *can*, so it is implemented rather than ignored: a cap on how
far a member may be padded forward and on the struct's own alignment, with
`push`, `pop`, a bare `pack()` to go back to the ABI's answer, and a refusal
for a width that is not a power of two. A compiler that ignored it would give
every struct after it the wrong offsets and say nothing, which is why the
other pragmas are still refused by name rather than skipped.

**Bitfields** are laid out, read and written: `unsigned int flags : 3;` packs
into a storage unit of its declared type and the next field continues in the
same unit while it fits, which is what every ABI py2bin targets does. A read
shifts down and masks, and a signed field gets back the sign its own width
carries — three bits holding -1 read as -1 and not as 7. A write is a
read-modify-write, so the fields beside it keep their values. `: 0` closes the
unit without taking any of it, an unnamed field pads without being reachable,
and `&f.a` is refused with the reason, because a bitfield has no address of
its own.

**Braced initialisers** work for whatever they nest: `struct P a = {1, 2}`, a
struct inside a struct, an array of structs, a two-dimensional array, a string
member, a union, a partly-filled list (C zero-fills the rest, and so does
this), and any of them at file scope. One entry point initialises whatever is
at an address, of whatever shape it is, because a member may be an array or a
struct or a scalar and C nests them freely.

**What it is not.** py2bin's C compiler implements C and ships its own copies
of the standard headers (`stdio.h`, `stdlib.h`, `string.h`, `ctype.h`,
`math.h`, `assert.h`, `wchar.h`, `uchar.h`, `stdint.h`, `inttypes.h`,
`limits.h`, `float.h`, `stddef.h`, `stdbool.h`). The ones with functions in
them - `string.h`, `ctype.h`, `math.h`, the allocator in `stdlib.h` - are
written in C and compiled like any other source, so they can be read rather
than taken on trust. It has no system include path: a real system header uses
compiler extensions this does not implement.

**Your own headers are found; a platform SDK's are not.** A folder called
`include`, `inc`, `headers` or `src` beside the program is searched without
being asked, and anywhere else is `--include DIR` (`build.py`) or
`--include-dir` (`py2bin cc`), repeatable. What that will *not* get you is a
vendor SDK: `WebView2.h` and its like are COM — `MIDL_INTERFACE`,
`STDMETHODCALLTYPE`, `__declspec(uuid)`, pure-virtual vtables — and pull in
half the Windows SDK behind them. Finding the file does not help, because the
file is written in a language this compiler does not implement. That is a real
ceiling and not a missing flag.

**A header that is not on this machine can be fetched.** `--auto-fetch`
(`build.py`) says that a header py2bin cannot find here may be looked up and
downloaded; without it nothing reaches the network, which is what keeps a
build the same on a machine that has none. Two places are searched, and which
one first is decided by the name: an include with a directory in it, or one
spelled `.hpp`, belongs to a library published as source, and a bare `.h` is
what a vendor ships in a package.

```
python3 build.py app.cpp --auto-fetch
py2bin fetch-header nlohmann/json.hpp --into vendor
py2bin fetch-header thing.h --from https://example.com/thing.h --into vendor
python3 build.py app.cpp -D SOME_MACRO -I vendor
```

`-D NAME` (or `--define NAME=VALUE`, repeatable) is how you answer a header
that asks for a macro.

**A `#error` says why it was reached.** A header that falls through every
branch of an `#if`/`#elif` chain and stops is telling you what it wanted, so
py2bin lists the branches that did not hold:

```
winnt.h:2638:3: #error You must define NtCurrentTeb() for your architecture
  Reached because none of these held:
      #ifdef WINE_UNIX_LIB
      #elif defined(__i386__) && defined(__GNUC__)
      #elif defined(__x86_64__) && defined(_MSC_VER)
      ...
```

That is the whole answer to "what do I do about this": each line names what
that branch needed, and you can see whether any of them is something you can
arrange. It replaced a guess — py2bin used to read the word "define" in a
`#error` and suggest `-D`, which is wrong here, because that chain never
tests whether `NtCurrentTeb` is defined.

A platform header — `rpc.h`, `objbase.h` — is never published on its own and
is never in a repository named after it: it belongs to a *set*. Those sets are
searched by path, and what comes down is the closure over the header's own
`#include` lines rather than the directory it sits in, which is a few dozen
files where the directory is a few thousand. A small library's directory is
taken whole, because that directory *is* the library.

Each build says which package or repository a header came from, so you can
judge it. What arrives is a header and not a toolchain: whether py2bin's C
understands what is inside it is a separate question, answered by the compiler
in the usual way.

Two sets of Windows headers are known, and which is tried first was decided by
walking both closures and counting. Taking `rpc.h` from each: one gives 80
headers and cannot resolve 11, every one of them a COM header generated from a
`.idl`; the other gives 144 and cannot resolve 13, two of which are its own
core headers — and *every* header in that set includes those at the top,
unconditionally, so nothing from it compiles at all. The first is tried first
for that reason. A fetch says which files a set does not publish, so you can
see what it left out rather than discovering it one build at a time.

**Some headers cannot be fetched by anyone**, and a Windows one usually cannot
be *compiled* here either. A COM header is generated from a `.idl` at build
time and a platform set writes its own core header at configure time — neither
exists as a file. And the parts that do exist are written for one of two
specific compilers: `winnt.h` picks its `NtCurrentTeb()` by testing `__GNUC__`
or `_MSC_VER`, and every branch is inline assembly or that compiler's
intrinsics. py2bin is neither compiler and implements neither, and it does not
claim to be one — a header that believed it was would reach for builtins that
are not there and produce something plausible and wrong instead of a refusal
that says where it stopped. That is a decision, not an oversight: py2bin will
not define another compiler's identity macro to get past a `#error`.

Where such a header offers a branch for a compiler that is neither, `-D` will
take it — that set's `NtCurrentTeb()` has one, and choosing it moves the wall
from `winnt.h` to `winsock.h`. Read what the macro means before you use it:
that one selects a different view of the platform, so it gets further without
getting closer. What gets a program to a COM library is declaring by hand the
two or three interfaces it actually calls, which py2bin's vtables express
directly.

**Which is why `<windows.h>` is py2bin's own too.** Microsoft's is tens of
thousands of declarations written in extensions this compiler does not have.
py2bin ships the part a program usually wants - the types (`DWORD`, `HANDLE`,
`LPCSTR`, ...), the constants, and prototypes for about thirty functions from
`KERNEL32` and `USER32`: `Sleep`, `GetTickCount`, `GetStdHandle`, `WriteFile`,
`CreateFileA`, `SetConsoleOutputCP`, `MessageBoxA`, `MultiByteToWideChar` and
so on. A window is there too - `RegisterClassExW`, `CreateWindowExW`, the
message loop - and so is COM: `CoInitializeEx`, `CoCreateInstance`,
`CoTaskMemFree` and the three `Sys*String` calls. Calling *through* a vtable is
something py2bin could always express; those are how a program comes by the
pointer to call it on, which is what it had no way to do.

**A generated COM header compiles.** The vendor's own `WebView2.h` - 68,921
lines of MIDL output, straight out of the NuGet package - builds for both
Windows targets, and a program that calls through it lands on the slots the
vendor's own tables put its methods at.

MIDL output declares every interface twice and chooses between them:

```c
#if defined(__cplusplus) && !defined(CINTERFACE)
    /* C++ classes */
#else
    /* a table of function pointers */
#endif
```

py2bin defines no `__cplusplus`, so the second is the branch taken - and the
second is the branch it wants, because a COM object *is* that table. What was
needed to read it was small and ordinary once the shape was clear: `sal.h` and
the annotations, `CONST_VTBL`, `STDAPI`, `DEFINE_ENUM_FLAG_OPERATORS`, the
fixed-width `UINT32` family, `IUnknown` and `IStream` in their C shape,
`VARIANT`, `EventRegistrationToken`, and `__declspec` read and dropped -
except `align`, which decides layout and is refused rather than dropped.

Two things in the C front end, both of which every real header needs: an
enumeration constant may now stand in a constant expression, which is how a
generated enum is written (each entry is the one before it plus one), and an
enumerator may be `0xffffffff`, which is how a flag enum spells all its bits.

**A header may ask what this compiler has.** `__has_feature(x)`,
`__has_builtin`, `__has_attribute`, `__has_cpp_attribute` and the rest are
operators rather than macros: they take an argument, and a compiler that does
not have the thing still has to read past it. Left to the rule that turns an
unknown identifier into 0, each became `0(x)` and the `(` was a stray - which
stopped a standard C++ header on its first line of feature detection.
py2bin answers no to all of them, which is what makes a library take the
portable path it keeps for compilers without the extension, and answers
`__has_include` truthfully by looking.

**A class template may be written again for a shape of argument.**
`template <class T> struct is_pointer<T *>` is the whole mechanism a traits
header is made of, and it works now, along with the full form
(`template <> struct Name<int>`). Which copy a use gets is decided the way
C++ decides it: the narrowest pattern that fits. A type named outright beats
one written around a parameter, `T **` beats `T *`, and `<T, T>` - which says
the two arguments are the same type - beats `<T, U>`, which says nothing.

**`class X final { ... };` is a class**, and used not to be. It reads exactly
like `Type name { ... };` - a name, a space, a name, a brace - and the pass
that rewrites a brace initialiser took it for one: the class came out as
`X final( ... );`, turned inside out, with every member after it lost. Any
class written with `final` was destroyed silently. `final` and `override` are
now dropped where they stand, both being checks C++ makes that C cannot.

**A `using` alias written inside a class** is that class's name for a type,
and is resolved wherever the class says it - in its body and in its methods,
spelled out or bare. `using Handler = std::function<void(const string&)>;` is
how a class declares what it will call back into.

**A template may be written inside a template.** `ComPtr<T>::As` is one, and
it could not be read before: a member template's calls are on objects of a
type that does not exist until the class around it has been written out, and
the pass that expands member templates ran first and threw it away as unused.
It runs again afterwards now, and what a member template takes may be spelled
in terms of another template - `As(ComPtr<U> *other)` says what it was handed
only that way - so deduction takes a parameter's spelling apart against the
argument's. Each copy of a class template gets its own copies of the member,
which is what two `ComPtr`s of different interfaces each having an `As`
means.

**`<wrl.h>` is py2bin's own**, which is what a WebView2 program includes for
`ComPtr<T>` - a pointer that counts, releasing what it held when it is given
something else or goes out of scope. `__uuidof(T)` is the `IID_T` a generated
header writes out beside the interface.

**A parameter may stand for however many arguments are left.**
`template <class... Ts>` is a pack: `sizeof...(Ts)` is how many there are,
`Ts...` is the types, and a parameter declared `Rest... rest` becomes one
parameter each. A pack of nothing is a pack. Recursion over one stops the way
C++ stops it - an ordinary function is preferred to a copy of a template, so
`total(int)` written out by hand is the end of `total(int, Rest...)`.

**And a function's return type may decide whether it is a candidate at all.**

```cpp
template <class T>
typename enable_if<is_pointer<T>::value, int>::type kind(T v) { return 1; }
template <class T>
typename enable_if<!is_pointer<T>::value, int>::type kind(T v) { return 2; }
```

Where the guard says no there is no `type` in it, and a function whose return
type does not exist is not a candidate - another of the same name answers the
call instead. That is what SFINAE means and it is what py2bin does: the guard
is worked out for the arguments the call deduced, the class it names is
written out, and if the member is not in it the copy is never made.

**So `<type_traits>` is py2bin's own now**, written the way the standard
describes each answer rather than the way a real library implements it: a
general class that says no and a narrower one that says yes. `is_same`,
`is_pointer`, `is_reference`, `is_const`, `is_void`, `is_integral`,
`is_floating_point`, `is_signed`, `is_unsigned`, `remove_reference`,
`remove_pointer`, `remove_const`, `remove_volatile`, `add_pointer`,
`add_const`, `conditional`, `enable_if`, `integral_constant`, and
`true_type`/`false_type`.
It costs nothing at run time: the copies are made while translating and the
answer is a constant before any code runs.

**The subset was probed rather than assumed.** Forty-eight programs written
across the corners of the language - templates, virtual dispatch, operator
overloading, the containers, lambdas, destructors, strings, statics and
namespaces, the heap, references, plain control flow, structs and unions,
exceptions, conversions, the smart pointers - each built by py2bin and by
`clang++`, run, and the two answers compared. Nineteen disagreed. Five of
those compiled without a word and printed something else, which is the half
worth reading first:

*A `static` local was constructed on every call.* `static Counter c;` inside a
function is built once, the first time control reaches it; this built it each
time through, so a counter that should have answered 1, 2, 3 answered 1, 1, 1.
The flag C++ keeps out of sight is written out now.

*A destructor ran before the answer was read.* C++ works out the returned
value and then takes the scope apart. This did it the other way round, so
`return alive;` in a scope whose destructor decrements `alive` answered with
the count afterwards. The value goes into a temporary first.

*A temporary hoisted out of an unbraced loop body left the loop.* `for (...)
grid[i] = Cell(i * 10);` needs somewhere to build the `Cell`, and everything
here that needs somewhere writes it in front of the statement it found - which
in front of a body with no braces is *above the `for`*, out of the scope, and
out of reach of the `i` it was built from. So the braces C++ lets you leave
out are written in, once, before any of those passes run. Nothing about the
program changes: a block holding one statement is that statement.

*`std::unique_ptr<T> b = std::move(a);` copied the pointer.* `std::move` is a
cast and nothing survived it, so both held the same object: `a` still answered
as though it owned one, and both destructors freed it. Building a `unique_ptr`
from another transfers now. That is not a liberty - a program that *copies* a
`unique_ptr` does not compile in C++ at all, so a move is the only thing it
can ever have been.

*A method answering an object, called in another method's `return`, lost the
caller's space.* An object is answered through a pointer the caller provides;
this emitted the call without it. It was in py2bin's own `std::string`.

The rest failed loudly, which is the right way to fail but still a gap:
`std::vector` of a plain struct could not `push_back` one; `catch (const T &e)`
- the form C++ asks for - built a value from a cast integer, because the
qualifier hid the class name; `friend` functions and free operators had
nowhere to go; `template <int N>` crashed; a `const T &` parameter deduced
nothing from a value argument, which is how nearly every such call is written;
`Class::staticMember()` lost its class if the class also held an enum;
`b.add('x').add('y')` - a builder, and any other chain of reference returns -
was invisible to two passes at once, because both read the text in the
stretches between its literals and the statement was split across two of them.

Each is a program in the corpus now, so each is checked on every sweep.

**A C header under its C++ name is found by rule.** `<cstdarg>` is
`<stdarg.h>` and `<cstdio>` is `<stdio.h>`; C++ renames each C header by
dropping the `.h` and putting a `c` in front, and says the two hold the same
things. Which ones exist is asked of the headers py2bin's C ships rather than
kept as a list beside them - written as a list it went stale, and a program
including `<cstdarg>` was told py2bin does not implement it while
`<stdarg.h>` sat in the same build.

**A standard C++ header py2bin does not implement still says so.** One
spelled the way only a standard header is, that py2bin does not ship and that
no `--include` directory holds, is refused by name with the list of the ones
it does - rather than fetching a real standard library's copy and failing
four thousand lines inside it about something that is not the reason. Your
own copy under `--include` still wins.

**A header that chooses a branch is preprocessed before it is translated.**
The C++ translator runs before the preprocessor and has no `#if`, so pasting
one in meant translating both branches - and the branch meant for C is
written in shapes that mean something else in C++. Leaving it to the
preprocessor instead took the *C* branch, and a program calling an interface
the C++ way - `view->Navigate(url)`, which is how one is written - was told
the struct had no such member.

So the preprocessor runs first for that header alone, with `__cplusplus`
defined, and hands the translator the one branch a C++ compiler would have
been given. Both spellings of the vendor's header work now:

```cpp
ICoreWebView2 *view = ...;
view->Navigate(L"https://example.com");   // slot 5, and the code loads 0x28
```

What that branch carries with it is split in two. The headers that declare
COM interfaces stay in it, because the translator is about to read
`struct ICoreWebView2 : public IUnknown` and a base it cannot see is a base it
cannot lay out; those are reported, so the run that reads the rest of the
program leaves them alone. The rest are plain C - types and prototypes - and
are left to that run entirely, which reads them at the top where it puts
every directive. That is the order they have to be in: `<shellapi.h>` asks
for `HINSTANCE`, and the answer has to be above it.

**A guard is no answer here**, which is worth knowing before reaching for
one: the translator moves every directive to the top of the file it emits, so
a `#ifndef` written around a *declaration* ends up above the thing it was
meant to guard and guards nothing. An `#include` is a directive all through
and survives that move intact, which is why py2bin's own `<unknwn.h>` asks
`<wtypes.h>` for `HRESULT` and `GUID` rather than writing them out again.

**`_mingw.h` is py2bin's own too**, and for a reason worth stating: it is the
one file in the mingw-w64 set that does not exist. Every header in that set
opens with `#include <_mingw.h>`, and what a fetch finds is `_mingw.h.in` -
the template a configure step fills in. What it holds is a description of the
compiler reading it, which extensions it has and how it spells an attribute,
so py2bin is the one that knows the answers. Nearly all of them are nothing,
which is how a compiler without an extension has always been told about it.

**Windows is LLP64, and py2bin now is too on that target.** `long` is four
bytes on Windows and eight everywhere else; py2bin was LP64 on all six, on
the stated reasoning that it shared no layout with a platform C library. That
was true while it compiled nobody's headers but its own and stopped being
true the day it compiled a vendor's. `FORMATETC` holds a `LONG`, and eight
bytes where the platform has four moves every member after it and makes the
struct the wrong size to hand to anything. `size_t` and the pointer-width
integers stay eight bytes there, which is why they are named rather than left
to whatever `long` turned out to be.

**Anonymous struct and union members work**, which C11 has and the SDK uses
throughout: `STGMEDIUM` holds an unnamed union of handles and reaches into it
without naming it. The member is laid out so its size and alignment count,
and looked through so its members are the enclosing struct's.

**`Callback<I>(lambda)` is written out as the class it is.** WRL's helper
builds a COM object around a closure: a reference count, the three `IUnknown`
methods, and the interface's own method forwarding to the body. The vendor's
is a template whose machinery this subset does not have; its *result* is an
ordinary class, and that is what py2bin emits - one per callback, deriving
from the interface, with the lambda's body as the method and the enclosing
object carried in a member.

Innermost first, which matters as soon as there is more than one: a callback
is usually written inside another, and `this` in each means the object the
body *around* it was written in. Taken from the outside in, the outer pass
would already have turned that into its own member and the inner one would
have carried the wrong object.

**Windows starts a desktop program at `wWinMain`, and C starts one at
`main`.** The wrapper between them is written in C, below the entry point it
calls, because there is no C runtime here to link that would do it - the four
things `wWinMain` is handed are all things the program could have asked the
kernel for. The image is marked as a desktop one too, so no console opens in
front of the window.

**`FORMATETC`, `STGMEDIUM`, `DVTARGETDEVICE` and `STATDATA`** are py2bin's
own, transcribed from the published set rather than written from memory - and
every size and offset checked against the same fields computed at the widths
Windows gives them. 32, 24, 16 and 56 bytes, on both Windows targets.

**What py2bin's own headers define is a default, not a claim.** `S_OK` is
`((HRESULT)0)` here and `((HRESULT)0x00000000)` in the set a fetch brings
down, and neither is wrong - so a real header may redefine what py2bin
supplied, and only two definitions that are both somebody else's still clash.
And where a fetched `winerror.h` is on the path, py2bin's `<windows.h>` takes
it, because a set that has one relies on that order: `<urlmon.h>` writes
`#ifndef E_PENDING` around its own spelling, which only does its job if the
real one has been read already.

A DLL somebody else wrote is reached one of two ways. `LoadLibraryW`,
`GetProcAddress` and `FreeLibrary` are all there, which asks for the entry
point by name at run time and lets a program carry on without it. Where the
program calls the vendor's function directly - as everything written against
an SDK does - **`--auto-fetch` works the library out on its own.** A function
the program calls and nothing defines was declared by a header; that header
came out of a package; that package ships the library too, and a library says
what it exports. So py2bin reads the export tables of what it downloaded and
takes the one that has the name, which is an answer that is either right or
absent rather than a guess:

```
Nothing here defines CreateCoreWebView2EnvironmentWithOptions. Looking for the library it is in.
CreateCoreWebView2EnvironmentWithOptions is exported by WebView2Loader.dll, which came with it
```

`--library WebView2Loader.dll` names it outright where that is wanted - where
the header was supplied by hand, say, so there is no package to look in - the
way a build with a linker names an import library:

```bash
py2bin cc main.cpp webview.cpp -I vendor --library WebView2Loader.dll \
  --target windows-x86_64 -o app.exe
```

Every function the program declares and never defines is then an import from
that library, with the shape of the call read off the prototype the program
wrote - which is what a linker reads out of a `.lib`. `NAME:one,two` claims
only the symbols named, for a program calling into two components. py2bin
knows the library behind every function it vets and does not guess at one it
has never seen, so this is asked for rather than assumed: an image naming a
DLL will not start on a machine without it.

**And the library itself is put beside the binary.** A component ships two
things: the header, to compile against, and the library, to run against.
`--auto-fetch` already brings the header down; the same package holds the
library, so the one matching the target's architecture is written next to the
executable. What comes out of `dist/` is a program that starts, rather than
one that needs a component installed by hand - which matters most on the
machine that cannot install anything, and is the one it was written for.

Each is an ordinary import the loader binds, the same mechanism a
program driving CPython already uses, so calling one still needs no toolchain.
A name it does not declare is a name the compiler reports, rather than one
that compiles and fails to resolve; and on a target that is not Windows the
header says so instead of letting the program build against declarations that
cannot bind.

**Two ways that carrying went silently wrong**, both found by building a real
project rather than a test. A build that is wrong about what it carries says
nothing: it reports success, and what fails is the program, later, on somebody
else's machine.

*A library named without its suffix was not carried.* `--library
WebView2Loader` is how the component names itself and how a `CMakeLists.txt`
asks for it - `find_library(NAMES WebView2LoaderStatic WebView2Loader)`. The
step that resolves symbols took the bare name; the step that puts the file
beside the program wanted `.dll` and skipped it without a word. So the build
succeeded, the symbols bound, and the program could not start, because the
library it loads at run time was not there. A bare name is given the suffix
its target spells - `.dll`, `.dylib`, `.so`.

*A source directory named relatively was never looked above.* The finder that
reads what a program opens walks one level up from the sources, because
`src/main.cpp` naming `web` means `../web` nearly every time. `build.py
src/main.cpp` hands it `src` - and `Path("src").parent` is `Path(".")`, whose
parent is itself, so the walk stopped on its first step and the directory
above was never searched. Handed an absolute path the same finder had always
worked, which is why nothing caught it: the build carried nothing, said
nothing, and produced a program with no pages to show. The directory is
resolved before anything walks up from it.

**And so are its pieces.** The SDK splits `<windows.h>` across a dozen files,
and a program is as likely to include one of those - `winnt.h`, `windef.h`,
`basetsd.h`, `winbase.h`, `winuser.h`, `minwindef.h`, `minwinbase.h` - as the
whole. Each of those names is py2bin's own `<windows.h>`, entered once however
many of them a program asks for.

Fetching them instead does not work, and it is worth saying why rather than
leaving it to be found. The published sets are written for a compiler that is
GCC or MSVC, and they check. Wine's `winnt.h` runs nine branches looking for
one of those two paired with an architecture:

```
winnt.h:2638: #error You must define NtCurrentTeb() for your architecture
```

Every branch that would have matched needs something no branch could give:
inline assembly reading `gs:0x30`, a register variable pinned to `x18`, or an
MSVC intrinsic behind `#pragma intrinsic`. py2bin is neither of those
compilers and does not claim to be one, so it brings its own header - the same
answer, and for the same reason, as the COM headers above.

A fetch does not bring one of these along either, and that mattered more than
it sounds: `--auto-fetch` takes the closure over what a header includes, so
fetching anything from a Windows set once left that set's `winnt.h` sitting in
`.py2bin-headers/`. An include directory is searched before a built-in, so
py2bin's own was shadowed by a copy that cannot compile here - for every build
afterwards, which is how a build that had been fixed came back with the same
error. Now a header py2bin ships is never taken along, never taken from that
cache directory even if an older run left one there, and asking for one
outright says so rather than downloading it. A header you name yourself with
`-I` is your own choice and still wins.

**`inline` is accepted and ignored**, along with `__inline` and
`__forceinline`. py2bin decides for itself whether a call is a real call or an
inlined body, so the specifier says nothing to it - but refusing it stopped
every real header at its first small function, since `static inline` is how a
platform header writes one.

**The platform macros are defined**, which they were not before: `_WIN32`,
`_WIN64`, `__APPLE__`, `__linux__`, `__unix__`, `__x86_64__`, `__aarch64__`,
`_M_X64`, `_M_ARM64`. A file that picks its headers with `#ifdef _WIN32` took
the wrong branch on every target until these existed, and the failure was a
missing header rather than anything that pointed at the cause. **C++ is translated to C** rather than compiled:
classes, inheritance, `virtual`, references, templates, overloading, `new`,
exceptions, and py2bin's own `<string>`, `<vector>` and `<iostream>` - see
[C++, translated to C](#c-translated-to-c).

### A program that is not all one language

An application is often Python, some C, and a folder of html/css/js. All three
go into one artifact through `compile-capi` - the tier that produces real
machine code, not an interpreter shipped beside your source:

```sh
py2bin compile-capi app.py --native native --include web \
       --app --name App --onefile --embed-python -o App.app
```

- **`--native PATH`** compiles the C *for the same target as the Python* and
  puts the executable beside it. PATH is the `.c` holding the `main`, or a
  directory holding it; every other `.c` beside it is compiled in, and an
  `include/` directory is searched for headers. Compiled here rather than
  accepted already built, because nothing about a finished executable says
  which machine it was for - and a helper built for the build machine, dropped
  into a Windows bundle, is the failure worth preventing.
- **`--include PATH`** carries a file or directory as it is. Web assets are
  not "supported" so much as *carried*: they are data, and a bundle is a
  filesystem.

`build.py` and `py2bin make` need none of that typed. A `native/` directory
holding a `.c` with a `main` is compiled; `web/`, `assets/`, `static/`,
`templates/`, `resources/` and `data/` are carried. Answer the three questions
and the mixture comes out as one file.

**The C and the Python do not merge into one image.** py2bin has no linker, so
the C is a separate executable inside the bundle and the Python reaches it the
ordinary way - `subprocess`, or `ctypes` for a shared library. What is in one
file is the *delivery*, not the linkage.

One limit of the C compiler is worth knowing before leaning on this: it ships
its own standard headers and has no system include path, so `#include
<stdio.h>` gets py2bin's copy and there is no `#include <sys/socket.h>` to be
had. `<stdlib.h>` brings a real `malloc`, written in C on top of one primitive
the compiler provides, so it can be read rather than taken on trust; it is an
arena, and `free` keeps its promise not to touch what you hand it.

**Text is UTF-8, and the wide literals mean what the platform means.** A
character above 127 written in a source file goes into a plain literal as the
UTF-8 it already was. `L"..."` becomes `wchar_t`, which is four bytes on
POSIX and two on Windows - so a character outside the basic plane becomes a
surrogate pair there, which is what makes Windows' `wchar_t` different rather
than merely narrower. `u"..."` and `U"..."` are UTF-16 and UTF-32 whatever
the target, `u8"..."` is UTF-8, and `\xFF` still names the byte while
`\u00ff` names the character.

### C++, translated to C

py2bin has a C compiler and no C++ one. What it has instead is what the first
C++ compiler was: a **translator**. A class becomes a struct, a member
function becomes a free function whose first parameter is the object, a
constructor initialises one in place. Nothing downstream knows C++ happened —
the C that comes out is compiled by the same backend that compiles C, and
cross-builds to all six targets the same way.

```sh
py2bin cc main.cpp stack.cpp -o app
```

**What goes through.** Classes and structs with data members and member
functions, written in the class or out of it as `Type Class::name`;
constructors and destructors, including destructors at every exit from a
scope and at every `return` inside one; `this`, written or implicit; calls
through an object, a pointer, an array element or `this`; single inheritance,
with the base embedded first so a pointer to the derived object is a pointer
to the base one.

Beyond that, each of these is something a C++ compiler turns into C-shaped
code before it emits anything, which is where they are done here:

| | how |
|---|---|
| **Overloading** | by how many arguments a call passes, and by their types where that is not enough. `show(1)` and `show("a")` become `show__1__int` and `show__1__char_p` |
| **`virtual`** | a pointer to a table of the object's own functions, installed by its constructor. Derived tables keep the base's slot order |
| **References** | a pointer with the dereference written out; call sites take the address |
| **`new` / `delete`** | malloc from `<stdlib.h>`, then the constructor. `new T[n]` records the count in front of the block so `delete[]` can destroy every element |
| **Templates** | one copy per set of arguments used, named after them — `Box__int`, not a hash. Arguments a call does not spell out are deduced from literals and from declarations in view. A member written outside its class (`template<typename T> T Box<T>::get()`) is folded back into it; `template<>` is one copy written by hand and goes straight in under the name the expander would have used; a member template is expanded from its call sites |
| **Namespaces** | flattened. One translation unit, no linker, so scoping is the whole of what a namespace can mean here |
| **Operators** | `a + b` becomes the call the class declared for it, including a value return through a hidden pointer. The right side is an operand, not just a name: `a += " x"` and `a + f(1)` work too. Unary ones as well — `*p`, `!p`, `-v`, `p->m()`, `++c` and `c++` — each told from the two-operand spelling by taking no parameter, which is the only thing that says |
| **Exceptions** | a flag and a return, tested by the caller immediately after the call. `try`/`catch` becomes a jump to a label. A thrown object is copied to the heap so it outlives the frame |
| **Lambdas** | a class with a call operator and a member per capture — which is what the standard says one *is*. `auto` is how one is held directly, because the class's name is generated; `std::function` holds one too, and copies it. `[x]` copies, `[&x]` holds the address and every use follows it, `[this]` holds the enclosing object and bare member names go through it, `[=]`/`[&]` capture what the body uses — including the object, the same rule C++ applies — and `[v = n * 2]` is a member initialised from an expression the scope has no name for. `[](auto a, auto b)` is a member template in C++; here the types are read from the calls, and calls that disagree are refused rather than compiled once and run for both. A lambda written inside another is expanded first, so what the outer one returns can be read: `auto add5 = outer(5);` holds a closure a closure made |
| **Plain structs** | a `struct` with no methods is C already and is emitted exactly as written — but py2bin's C can neither pass nor answer one in a register, so `Point add(Point a, Point b)` gets the same treatment a class does: passed by address and copied on entry, answered through the pointer the caller provides |
| **`dynamic_cast`** | answered from the table the object carries. py2bin has no linker, so a translation unit is the whole program and there is no class it has not seen: an object is a `D` if its table is D's own or belongs to something derived from D. A cast that fails answers null |
| **The rest of it** | enums (plain, scoped, and with an underlying type named), unions, bitfields, `static` data members, member functions and block-scope statics, objects at file scope with or without constructor arguments, nested classes, nested namespaces (`namespace a::b`), member typedefs (`vector<int>::iterator`), range-`for` (over a container, over a plain array, and by reference), member initialiser lists (including a member of class type, built with what the list gave it), default member initialisers (`int n = 7;`), `= default` and `= delete`, `final` and `override`, default arguments, named casts, `explicit`, function-pointer members, `auto`, `using X = Y`, aggregate and braced initialisers, arrays of objects built from a brace list, `bool`/`true`/`false`/`nullptr`, forward declarations, members defined outside their class, prototypes in headers, and free functions that return a class by value |
| **`operator()`** | a call on an object, so `std::sort(v.begin(), v.end(), cmp)` takes a lambda or a function object alike |

**Standard headers**, each written in py2bin's own C++ subset and put
through the same translator as your code — so they are readable, and they are
not special cases in the compiler:

* `<string>` — a fixed-capacity string with `assign`, `size`, `c_str`,
  `append`, `substr`, `find`, `npos`, `push_back`, `operator+`, `operator+=`,
  `operator[]`, comparison, and the free `to_string` and `stoi`.
* `<vector>` — a template, so one concrete class per element type, with
  `push_back`, `pop_back`, `erase`, `insert`, `assign`, `resize`, `reserve`,
  `at`, `front`, `back`, `data`, `begin`/`end` and `iterator`. It grows by
  doubling and the old block stays where it is, because the heap under it is
  an arena that does not reclaim; pretending otherwise would be the dishonest
  part, not the leak. A `vector<vector<int>>` works, and so does `g[0][1]`.
* `<map>` and `<unordered_map>` — entries in one array, so an iterator is a
  pointer to one and `it->first` is an ordinary member read. `find`, `count`,
  `contains`, `at`, `erase`, `operator[]`, `begin`/`end`. It searches from the
  front, which a red-black tree would not: a program holding thousands of keys
  will notice, and one holding dozens will not.
* `<set>` and `<unordered_set>` — the same shape with nothing on the other
  side. Insertion order is kept, which is a stronger promise than the
  unordered ones make and a weaker one than the ordered ones do.
* `<memory>` — `unique_ptr` (which frees what it holds when it goes) and
  `shared_ptr`, with `get`, `release`, `reset`, `operator->` and `operator*`.
  Not move-only and not reference counted: this subset has neither move
  semantics nor atomics, so what is here is the ownership and not the
  machinery C++ uses to enforce it.
* `<sstream>` — `ostringstream` with one `operator<<` per type it can write,
  and `str()`.
* `<array>` — the count lives in the object rather than in the type, because
  a value template argument is not something this subset deduces.
* `<iostream>` — `cout` with one `operator<<` per type it can print, each
  handing the stream back so the next `<<` in the chain has something to be
  called on.
* `<algorithm>` — `sort` (a heapsort: no recursion, no scratch memory),
  `find`, `count`, `fill`, `reverse`, `min`/`max`, `min_element`/`max_element`,
  `swap`. Templates over pointers, which is what a contiguous iterator is, so
  they work on a `vector` and on a plain array alike.
* `<stdexcept>` — `exception` and the four that derive from it, each carrying
  a message and answering `what()`.
* `<filesystem>` — `path` (`filename`, `stem`, `extension`, `parent_path`,
  `operator/`) plus `exists`, `is_directory`, `is_regular_file`, `file_size`,
  `create_directory`, `remove`, `rename`. The `path` half is string work; the
  questions that depend on the platform live in a C header, because `#ifdef`
  is read by the C preprocessor and the C++ translator runs before it.
  `directory_iterator` is *not* there: reading a directory means `getdents`
  on Linux, `getdirentries` on macOS and `FindFirstFile` on Windows, each
  with a struct laid out differently per architecture - and a struct read
  wrong gives plausible answers.
* `<functional>` — `less`, `greater`, `plus`, `equal_to` and the rest of the
  comparison and arithmetic objects, which are small classes with a call
  operator. `std::function<int(int)>` becomes a class that holds any of them,
  built the way `dynamic_cast` is answered: py2bin has no linker, so a
  translation unit is the whole program, and every callable that is ever put
  into one of these is in front of it while it translates. So the class has a
  member per callable and a tag saying which is live, and the call is a
  comparison and a direct call — no thunk, no `void *`. It gives what an
  indirect call cannot: the closure is *copied* into the object, so one held
  as a member outlives the scope its lambda was written in, which is the whole
  reason a program stores one. A variable, a member, a parameter, an element
  of a `vector`, reassigned from a lambda to a function and back, and `if (cb)`
  before calling it — those all work; C++'s conversion to `bool` has no
  spelling here, so `if (cb)` and `if (!cb)` are read as what they mean.
* `<unknwn.h>` — COM's root interface, because nobody publishes one. Every
  open implementation of the Windows API generates `unknwn.h`, `wtypes.h`,
  `objidl.h` and the rest from an `.idl` at build time, and the vendor's own
  set ships inside a toolchain: there is no file to fetch, from any of them.
  So py2bin writes it. What COM *is* is a struct whose first member points at
  a table of function pointers — which is exactly how py2bin lays out a class
  with virtual methods — so the header says that in C++, and a program
  declares an interface by deriving from `IUnknown` the way a generated header
  does. `HRESULT`, `GUID`/`IID`/`CLSID`, `S_OK`, `SUCCEEDED`/`FAILED` come
  with it; `<wtypes.h>`, `<rpcndr.h>` and `<objbase.h>` are there for the C
  side, with the `MIDL_INTERFACE`/`STDMETHODCALLTYPE` spellings a generated
  header uses. An interface, an implementation of it, and a call through an
  interface pointer build for all six targets with nothing underneath but
  py2bin.
* `<utility>` (`pair`), `<numeric>` (`accumulate`), and `<cassert>`,
  `<climits>`, `<cfloat>`, `<cctype>`, `<cstdio>`, `<cstdlib>`, `<cstring>`,
  `<cmath>`, `<cstdint>` as names for the C headers underneath.

```cpp
#include <iostream>
int main() { std::cout << "Hello, world!" << std::endl; return 0; }
```

**What it will not do.** No unwinder, so exception propagation is written out
in the C and a call that can throw is given a statement of its own — which
means one behind `&&`, `||` or `?:`, or in a loop's header, is refused with
the reason rather than moved to where it would run at the wrong time. An
exception reaching the end of `main` aborts in C++; there is no way to raise a
signal here, so the program exits with a status of its own instead. A bare
`throw;` inside a `catch` rethrows what is in flight, and a `try` inside a
`try` nests the way C++ says. Multiple
inheritance and the rest of RTTI (`typeid`, `type_info`) are not implemented;
`dynamic_cast` is, because the table the object already carries answers it. Two overloads that
differ only in types py2bin cannot read are refused rather than guessed at.

**What a build reads.** Nothing off the machine it runs on. Compiling a C or
a C++ program opens the program's own files, py2bin's own source, and the
interpreter running the build — and no third thing. There is no system include
path, no host library is opened to read a symbol out of it, and no toolchain
is looked for. `tests/test_bundles_use_no_system_files.py` says so the only way
it can be said honestly: it watches every file-opening call while a build runs
and compares what was touched against those three places, so a convenience
fallback added later fails a test rather than the next machine.

The names that *do* appear in the output — `/usr/lib/libSystem.B.dylib`,
`/lib64/ld-linux-x86-64.so.2`, `KERNEL32.dll` — are written into the artifact,
never read during the build. They name the target's own loader, which is what
being a native executable for that target means, and they are the same names
whichever machine did the building.

**What it costs.** Translating C++ is the long pole, and it used to grow with
the square of the program: reading brace depth counted from the first
character each time it was asked, and the scan for which functions take a base
pointer walked every definition once per method body. On a program of 320
classes that was 8.9 seconds of CPU; it is 2.1 now, and the same three sweeps
agree with `clang++` either way.

**How it is checked.** One command, three questions:

```sh
tools/cpp_sweep.sh
```

*Meaning*: every program in `tools/cpp_corpus/` is built twice — once by
py2bin, once by `clang++` — run on this machine, and the outputs compared.
Reading the generated C tells you it is well formed and nothing about whether
it means the same thing, and this is the only thing that asks. C programs are
in there beside the C++ ones: the translator writes C, so a gap in the C front
end shows up as a C++ program that will not build, and a C program says which
of the two is at fault.

*Projects*: each directory under `tools/cpp_projects/` is a program in several
files with headers of its own, built from its `main.cpp` the way `build.py` is
handed one — which is how a project reaches py2bin, and not how a single file
does.

*Targets*: every program built for all six machines. A construct can
translate perfectly and still fail to encode for one of them, and nothing else
asks.

Point it at your own program instead of the corpus:

```sh
tools/cpp_sweep.sh check src/main.cpp include
```

Everything is built through `build.py` — the entry point this readme gives
people, and the one `py2bin make` asks the same questions as. A sweep that
used some other route would be checking a path nobody takes, and it was:
`build.py` left the `.exe` off a Windows build, which `py2bin cc` has always
added, so a program built the documented way came out unrunnable and no test
noticed. `build.py` now takes the three answers on the command line
(`--target`, `--how`), which is what lets anything check it.

The corpus is where each of those answers ends up: a program goes in when
something about it was broken, so the thing that broke cannot come back
quietly. It is over two hundred programs now, and every one of them agrees
with `clang++`.

`tools/cpp_sweep.sh check src/main.cpp include` asks the same questions of
your own file — every target, and the comparison — and is the thing to run
before shipping.

It does not run the cross-builds; five of the six machines are not this
computer. 271 programs × 6 targets is 1626 builds, and the two projects are
built for this machine and run.

What is in the corpus is what broke at some point: enums, static members,
nested classes, range-`for`, member initialiser lists, default arguments,
named casts, `operator=`, deep inheritance, a container of base pointers, a
string made from a literal, every header above, and a program for each
container and each operator. `tools/cpp_differential.sh` still works
and is the meaning half under its old name. The test suite additionally
translates every corpus program on each change, so one that stops working is
caught whether or not the sweep is run that day.

The first run of the sweep found nine bugs, every one of which produced C that
compiled cleanly and meant something else: a member called `n` rewrote
`printf("outer\n")` into `printf("outer\this->n")`; a parameter named after a
member answered 200 where the answer is 105; a destructor in a nested block
was emitted at the end of the function. It has gone on finding them —
`return size * size;` read as a declaration of a pointer named `size`; a
`return` inside a block destroying only that block's objects; a rewrite that
looks around itself reading its own position against the wrong text, so a
class deriving from one in a header assigned to the base's *name*, and did so
only when a header was included. clang++ is the yardstick there and never a
dependency; py2bin still builds with no toolchain at all.

### Reaching it from npm

The compiler is a Python program, which is what lets it cross-build by
arithmetic rather than by toolchain. `npm/` in this repository is a thin
wrapper so a Node project can reach it without knowing that:

```sh
npx py2bin cc main.c util.c -I include -o app
```

It finds a Python 3.10 or newer, hands the arguments to it, and passes the
exit code back. If `python-to-binary` is not installed for that interpreter it
says so, and names the exact command for the one it found.

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

The machine list offers **macOS, one binary for both** alongside the six, and
answering it produces a universal `.app` and `.dmg` in one pass. When a build
holds more than one architecture the last line reads them back out of the file
that was written:

```
  done: dist/app.dmg
  (the .app beside it is what the image holds)
  runs on: arm64, x86_64  (one file, both machines)
```

That last line is read from the bytes rather than restated from the question,
because "universal" is a claim about the file and a build that quietly
produced one slice should not look like one that produced two.

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

**What the one-file stub asks of the machine it runs on.** This is the one
place in py2bin where an artifact reaches outside itself, and it is worth
being exact about it. The stub is machine code and the payload travels inside
it, but the unpacking is handed to the platform: on macOS and Linux the stub
runs `/bin/sh -c` over a script that calls `mkdir`, `rm`, `rmdir`, `tar` and
`printf`, and on Windows it starts PowerShell. Run one with `PATH` emptied and
it stops at `mkdir: command not found`. Nothing is *read* from the machine at
build time - the audit above holds for every bundle kind - and the runtime the
bundle carries is its own copy, so this is a run-time dependency on five
ordinary commands and not on anything installed. It is still a dependency, and
the shape that removes it is a stub compiled from C by py2bin's own compiler:
the file primitives it needs (`open`, `read`, `write`, `mkdir`, `rmdir`,
`unlink`) are already in the C tier on all four POSIX targets, and Windows
reaches the same ground through the `<windows.h>` imports. A directory bundle
- anything without `--onefile` - has nothing to unpack and asks for none of
this.

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
| the whole program: modules, packages and relative imports compiled in | ✅ |
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
| `async def` that yields (an async generator), `async for` over one | ✅ |
| `locals()` / `vars()` inside a function | ✅ |
| `globals()`, one-argument `eval` / `exec` | ✅ |
| `x += y` through `__iadd__`, in place | ✅ |
| `@property` with a `@v.setter` | ✅ |
| `...` as a stub body | ✅ |
| `gen.throw` / `gen.close`, `asend` / `aclose` | ✅ |
| `__new__`, `__init_subclass__`, `__class_getitem__` bound as Python binds them | ✅ |
| `lambda i=i:` inside a comprehension | ✅ |
| a closure capturing a comprehension's target (`lambda: i`) | ❌ refused |
| `typing.Generic[T]` as a base, and `__mro_entries__` generally | ✅ |
| a class inside a class body, at any depth | ✅ |
| `pickle` and `copy` of a compiled class or function | ✅ |
| `except*` (PEP 654), exception groups | ✅ |
| `if` / `for` / `try` in a class body | ✅ |
| complex numbers, `f = lambda self: ...` as a method | ✅ |
| `x @ y` and `x @= y`, and every augmented operator | ✅ |
| `xs[1:3] = ys`, `del xs[a:b]`, extended slices | ✅ |
| a module's own `__doc__` | ✅ |
| `from x import *` at module level | ✅ |
| `[await f(x) for x in xs]`, and the set and dict forms | ✅ |
| a string holding a lone surrogate | ✅ |
| `c and await g()`, `await g() if c else x`, `while await g():` | ✅ |
| a generator expression, evaluated when asked rather than at once | ✅ |
| a generator `def` inside an `if`, a `for`, or another generator | ✅ |
| `dir()` with no argument, in any scope | ✅ |
| `abc.abstractmethod`, `functools.wraps` - both write on a function | ✅ |
| `f.__annotations__`, `typing.get_type_hints`, `singledispatch` | ✅ |
| `f.__doc__` and a class's, so `help()` and `inspect.getdoc` answer | ✅ |
| `globals()` in any module of the program, answering with its own | ✅ |
| packages, `pkg/sub/deeper.py`, `from . import x`, PEP 420 directories | ✅ |
| `importlib.import_module("pkg.thing")` with the name written down | ✅ |
| a program that puts its own `src/` on `sys.path` | ✅ |
| CPython's compile-time `SyntaxWarning`s, at compile time | ✅ |
| `type(f).__name__`, `sys._getframe()` | ❌ |
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
`send`. The cleanup is *not* emitted as a real `finally:` around each block - a
`yield` returns from `__next__`, so a real one would fire on the way out of
every suspension. It is attached to the raising path as a synthesized handler
that runs the cleanup and re-raises, and the ordinary path reaches the same
cleanup by jumping to a block of its own.

`with` is expanded into the try it already stands for. The manager and its
`__exit__` are looked up once, on the type, before the body runs, so rebinding
the name inside cannot change which object is left; returning true from
`__exit__` suppresses, as it should. `async for` and `async with` take the
same route.

Because the cleanup is a block, and a block may suspend, it can hold a `yield`
of its own - with whatever was raised waiting in a name until it is done. A
`break` or `continue` leaving the region would jump past it, so a copy runs
immediately before the jump.

Two details are worth recording, because both produced answers that looked
nothing like their cause. `raise X` where X is a class had never worked in any
compiled program, and an `async for` raising `StopAsyncIteration` was the
first thing to notice. And a `return` here is signalled by raising
`StopIteration`, so the cleanup saw the frame leaving as a failure and passed
`__aexit__` a `StopIteration` where CPython passes `None`.

### The interpreter surface it may use

- A fixed table of 100 exported CPython entry points:
  `PyBytes_FromStringAndSize`, `PyCFunction_New`, `PyDict_New`,
  `PyDict_SetItem`, `PyErr_Clear`, `PyErr_ExceptionMatches`,
  `PyErr_GetHandledException`, `PyErr_GetRaisedException`, `PyErr_Occurred`,
  `PyErr_Print`, `PyErr_SetHandledException`, `PyErr_SetObject`,
  `PyErr_SetRaisedException`, `PyFile_WriteObject`, `PyFile_WriteString`,
  `PyFloat_AsDouble`, `PyFloat_FromDouble`, `PyImport_AddModule`,
  `PyImport_ImportModule`, `PyInstanceMethod_New`, `PyIter_Next`,
  `PyList_Append`, `PyList_New`, `PyList_SetItem`, `PyLong_AsLongLong`,
  `PyLong_FromLongLong`, `PyLong_FromString`, `PyNumber_Add`, `PyNumber_And`,
  `PyNumber_FloorDivide`, `PyNumber_InPlaceAdd`, `PyNumber_InPlaceAnd`,
  `PyNumber_InPlaceFloorDivide`, `PyNumber_InPlaceLshift`,
  `PyNumber_InPlaceMatrixMultiply`, `PyNumber_InPlaceMultiply`,
  `PyNumber_InPlaceOr`, `PyNumber_InPlacePower`, `PyNumber_InPlaceRemainder`,
  `PyNumber_InPlaceRshift`, `PyNumber_InPlaceSubtract`,
  `PyNumber_InPlaceTrueDivide`, `PyNumber_InPlaceXor`, `PyNumber_Invert`,
  `PyNumber_Lshift`, `PyNumber_MatrixMultiply`, `PyNumber_Multiply`,
  `PyNumber_Negative`, `PyNumber_Or`, `PyNumber_Positive`, `PyNumber_Power`,
  `PyNumber_Remainder`, `PyNumber_Rshift`, `PyNumber_Subtract`,
  `PyNumber_TrueDivide`, `PyNumber_Xor`, `PyObject_Call`,
  `PyObject_CallNoArgs`, `PyObject_CallOneArg`, `PyObject_DelItem`,
  `PyObject_Format`, `PyObject_GetAttr`, `PyObject_GetAttrString`,
  `PyObject_GetItem`, `PyObject_GetIter`, `PyObject_IsInstance`,
  `PyObject_IsTrue`, `PyObject_Repr`, `PyObject_RichCompare`,
  `PyObject_RichCompareBool`, `PyObject_SetAttr`, `PyObject_SetAttrString`,
  `PyObject_SetItem`, `PyObject_Size`, `PyObject_Str`, `PyObject_Vectorcall`,
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

**Apple M4** (10 cores, 24 GB, macOS 27.0, arm64) against **CPython 3.14.3,
python.org framework build** - the interpreter these binaries actually bind.
This machine also carries a Homebrew 3.14.3; the two do not perform alike, and
timing against the wrong one is the easiest way to publish a number that is
not true. The harness asks a compiled program which interpreter it ended up
using and times that one.

300,000 iterations a row, nine fresh processes each, median taken, timing only
the hot loop so neither column pays for start-up. Higher is better.

```sh
python3 benchmarks/run.py
```

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
| list append | 5.1 ms | 5.0 ms | 0.97× |
| comprehension | 5.5 ms | 5.4 ms | 0.97× |
| dict store | 8.1 ms | 7.7 ms | 0.96× |
| `and` / `or` | 5.9 ms | 5.6 ms | 0.95× |
| exception raise/catch | 21.9 ms | 19.2 ms | 0.88× |
| f-string | 21.8 ms | 17.3 ms | 0.80× |
| `isinstance` | 7.4 ms | 5.9 ms | 0.80× |
| string concatenation | 15.6 ms | 12.3 ms | 0.78× |
| subscript | 8.3 ms | 5.9 ms | 0.71× |
| module global read | 5.3 ms | 3.6 ms | 0.68× |
| dict lookup by string | 6.8 ms | 4.5 ms | 0.66× |
| attribute read | 6.2 ms | 3.7 ms | 0.60× |
| chained compare | 11.7 ms | 6.7 ms | 0.57× |
| `for` over a list | 5.2 ms | 2.8 ms | 0.54× |
| closure call | 12.5 ms | 6.4 ms | 0.51× |
| attribute write | 6.2 ms | 3.0 ms | 0.49× |
| instantiation | 36.6 ms | 15.9 ms | 0.43× |
| tuple unpack | 15.0 ms | 5.5 ms | 0.37× |
| method call | 18.2 ms | 6.5 ms | 0.36× |

**Eight of twenty-seven beat the interpreter; fourteen sit at 0.80× or
better.** The wins are where a call, a lookup or an allocation stops happening
at all - a named argument placed at compile time, arithmetic in registers, a
direct C call. The losses track how many C-API calls an operation costs, where
the interpreter's specialised bytecode does the same work inline.

Two of those numbers are the price of being right rather than fast, and both
were measured before being accepted. `for` over a list went from 0.66× to
0.54× when `x += y` started meaning what Python means - branching to skip the
in-place operator was tried and measured *worse*. And a faster method call
exists: `PyDescr_NewMethod` was implemented, worked, and was reverted, because
a method reached that way is a `builtin_function_or_method` - `inspect.ismethod`
answers False and `self` appears in the signature, which is what once stopped
pywebview binding a compiled method.

How each row got where it is - the fixes, the five optimisations that measured
flat or slower, and the before-and-after of each - is in
[the guide](docs/DETAILED_GUIDE.md).

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

**What is still refused is refused by name**, with a `file:line:col`, rather than
approximated. What is left is `type(f).__name__` and `sys._getframe()`, and
those are structural rather than unfinished: a compiled function is a
`builtin_function_or_method`, which is not spelled `function` and makes no
frame. Each says what to do instead where there is something to do.

Three things used to be on that list because they write on the function they
are given: `abc.abstractmethod` sets one attribute, `functools.wraps` sets
six, and a `def` with annotated parameters writes `__annotations__`. Between
them they are how a great many programs begin, how nearly every decorator is
written, and how most modern Python is typed. A function the source decorates
with either of the first two - or annotates, in a program that reads
annotations anywhere - is now handed over inside a small object that can hold
what they write and binds like a method afterwards.

Only in those cases: every other function stays the plain compiled one, and
the extra hop this costs to call is paid by nobody else. Annotations are
recognised separately from the two decorators because annotating a parameter
is far commoner than asking what the annotation was, and a program that never
asks compiles to exactly what it compiled to before.

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

### Against Nuitka, on getting Python right

Speed is measured further down; this is the other question. Seventy-eight
whole programs, each compiled by both and its output compared against what
CPython answers for the same source - stdout and exit code, character for
character. **Nuitka 4.1.3**, macOS arm64, CPython 3.14.3, the same source and
the same machine for all three columns. Nuitka warns that 3.14 is only
experimentally supported by that release, which is worth knowing before
reading the two it gets wrong.

| | py2bin | Nuitka |
|---|---|---|
| answers exactly what CPython answers | **69** | **76** |
| answers something else | 6 | 2 |
| refuses, with a `file:line:col` | 3 | 0 |

Nuitka is ahead, and it should be: it is a mature project that reimplements
CPython's semantics rather than restricting them. Where the two differ:

| program | py2bin | Nuitka | what it is |
|---|---|---|---|
| `type(f).__name__` | ✗ | ✗ | **neither** says `function`: py2bin says `builtin_function_or_method`, Nuitka says `compiled_function` ([its own `tp_name`](https://github.com/Nuitka/Nuitka/blob/develop/nuitka/build/static_src/CompiledFunctionType.c)) |
| `f.__code__.co_code` | ✗ | ✗ | **neither** has bytecode to give. Nuitka [says so](https://nuitka.net/user-documentation/user-manual.html); py2bin has no code object at all |
| a debugger attached to a compiled function | ✗ | ✗ | **neither**: there is no tracing to attach to |
| `sys._getframe()`, `sys.settrace` | ✗ | ✓ | Nuitka builds real frames; a py2bin function makes none, which is part of why its calls are faster |
| a traceback naming a source line | ✗ | ✓ | Nuitka carries code objects with filenames; py2bin prints the exception line alone |
| `inspect.getsource` | ✗ | ✓ | there is no source beside the binary to read |
| `locals()` inside a generator | refused | ✓ | its names live on the object that runs it, so an answer would be the wrong one |
| `except*` (PEP 654) | ✓ | ✗ | py2bin rewrites it and agrees with CPython on 42,100 shapes; Nuitka answers differently |

What is left is one fact and its consequences: a compiled function is a
`builtin_function_or_method`, so it is not spelled `function`, has no code
object, and makes no frame. That is what makes a direct call 2.4x faster than
the interpreter - the frame is most of what a call costs, and Nuitka builds
one. Two of these rows are not a py2bin problem at all: **neither** compiler
says `function`, and neither has bytecode to hand back.

Everything that *needs* a function to hold something now gets one that can:
`abc.abstractmethod`, `functools.wraps` and an annotated `def` are each handed
a small object that holds what they write and binds like a method. Nothing
else pays for it - a module-level function is called directly in C and never
goes through the name at all.

**A corpus somebody wrote covers what somebody thought of.**
[`tools/fuzz.py`](tools/fuzz.py) covers what nobody thought of: programs drawn
at random from the grammar, compiled, run, and compared with the interpreter
character for character. Seeds are program numbers, so anything it finds is
reproducible.

Of 1,500 generated programs, **1,494 match exactly and none is refused**. The
six that differ all printed a function object - the `PyCFunction` fact again,
and the only difference this has ever turned up.

The corpus is [`tests/programs`](tests/programs). It is run against CPython
before every release, on the machine described above, and
[`.github/workflows/checks.yml`](.github/workflows/checks.yml) is set up to
run it on every push as well - the suite on three operating systems and three
Pythons, the corpus on Linux, and a cross-build for all six targets. **That
workflow has not actually executed**: runs queue and never start on this
account, so every number quoted here comes from running it locally rather
than from a green tick. The workflow is what it says; what has not been
demonstrated is the automation.


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

Run time is what a user waits for; build memory decides whether the build runs
at all. py2bin writes the machine code, the object file and the container in
Python and never starts a C toolchain. Nuitka writes C and hands it to Apple's
clang. Peak resident set of the **whole process tree**, sampled every 25 ms:

```sh
python3 benchmarks/build_memory.py
```

Nuitka keeps a ccache; py2bin has no build cache of any kind, so its column is
the same in both. **Cold** is a first build or a CI runner
(`--disable-cache=all`); **warm** is Nuitka's second build.

| what is being built | py2bin | | Nuitka cold | | Nuitka warm | |
|---|---|---|---|---|---|---|
| a small program (~10 lines) | **42 MB** | **0.1 s** | 557-656 MB | 16.5 s | 296 MB | 3.7 s |
| 200 functions | **186 MB** | **2.0 s** | 681 MB | 18.1 s | 423 MB | 4.9 s |
| 1,000 functions | **602 MB** | **7.3 s** | 946 MB | 22.0 s | 713 MB | 8.1 s |
| 3,000 functions | 1,567 MB | 21.3 s | 1,744 MB | 35.5 s | **1,516 MB** | **17.5 s** |

On a small program a build costs a seventh of a warm Nuitka's and runs in a
tenth of a second. **The advantage narrows as the program grows** and is gone
by three thousand functions: nothing here streams - the whole module is held
as objects, then as C, then as machine code, all at once - so memory grows
with the program while Nuitka's is dominated by a fixed toolchain cost.

**The C toolchain is counted**, and it is most of Nuitka's column: building a
small program, its tree holds `clang` at 181 MB and `ld` at 150 MB beside
Python's 300; at three thousand functions `ld` alone holds 1,210 MB. py2bin's
tree holds one Python process and nothing else.

And what the artifact costs to start:

| | startup | on disk |
|---|---|---|
| this, `compile-capi` | **10.1 ms** | **49 KB** |
| CPython | 13.8 ms | - |
| Nuitka `--standalone` | 15.4 ms | 17.2 MB |

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

The full history, with the reasoning behind each fix, is in
[the guide](docs/DETAILED_GUIDE.md). This is the short form.

### 0.9.13 - a C or C++ program that needs a library is still one file

A program built against somebody else's component came out of `dist/` as two
files: the executable and the DLL the loader wants beside it. Two files is
one more than somebody can send, and the second one is the one that gets
left behind.

The library is folded into the program now. `py2bin cc --onefile` does it on
request, and the path that asks three questions does it whenever there is
anything to fold - SidecarBridge comes out as a single 220 KB `main.exe`
with `WebView2Loader.dll` inside it.

The launcher that stands in front of the program is read off the program:
where that is a desktop program, so is the launcher, because a console
launcher in front of a windowed program flashes a black rectangle on every
start and nobody passes a flag for a thing they did not know was happening.

### 0.9.13 - the program is run once, and what it opened is carried

Reading a program finds every branch without taking any of them. Running it
takes one branch and finds what only that branch knows: a directory whose
name is read out of a config file at run time is written down nowhere a
reader can follow. py2bin does the second and adds it to the first, without
being asked - a bundle missing a file it opens is a bundle that starts and
then cannot do the thing it was built for, and finding that out on somebody
else's machine costs more than a run here does. `--no-watch` turns it off
for a program that should not be started at build time.

Added, never instead. A run is one path through the program, so the error
page nothing failed to reach, the locale nobody selected and the template for
the route nobody visited are all opened by code that did not run. And in a
program with a window the pages are fetched by the engine behind it rather
than by Python, so watching never sees them at all - reading does.

It runs in a daemon thread with a time limit, so a program that never returns
- which is every program with a window - does not hold up the build: what it
opened before it settled is what gets carried. Only a Python program is run;
a C or C++ one is compiled and never started.

### 0.9.13 - which project publishes a module is asked, not remembered

**A library's own files come with it, whatever they are.** py2bin knew 41
import names whose project is spelled differently - `PIL` is `pillow`, `yaml`
is `PyYAML` - and a module outside that list was reported as one it could not
name a project for, so the build went on without it. `certifi` was outside
it, which meant a program calling `certifi.where()` was bundled without
`cacert.pem` and failed the first time it opened a connection.

An installed package records the import names it provides, so the machine
doing the build already knows. That is asked first now, and the list is what
is left for a package that is *not* installed here - a cross build for a
machine this is not, where there is no metadata to read.

Nothing filters a package by file type. `certifi`'s `.pem`, a library's
`.csv`, `.json`, `.dat` or `.html` all travel with it, because what is
carried is the package and not a selection from it.

### 0.9.13 - assets are found by reading the program, not by knowing a layout

**What a program opens is read out of the program.** py2bin used to look for
six directory names beside the entry - `web`, `assets`, `static`,
`templates`, `resources`, `data` - which is six of the names an author might
pick and not the one they did. `ui`, `frontend`, `gui`, `pages`, `site`, an
`index.html` lying loose with no directory at all: none of them were carried,
and the bundle started and could not draw anything.

There is no list now, and no depth. A program says where its files are, and
what it says is worked out without running it:

```python
webview.create_window("App", "ui/index.html")                  # outright
os.path.join(os.path.dirname(__file__), "pages", "index.html")  # in pieces
ROOT = Path(__file__).parent.parent                             # through a
STYLE = ROOT / "assets" / "app.css"                             # constant
```

All three are followed, along with the modules the program imports from
beside it. That is what reaches a directory the program does not sit next to
- `../shared/web` and `parents[2] / "elsewhere"` are as findable as `web`,
because the program said so either way.

What a path assembled at run time cannot be read, so what is beside the
program is carried as well - judged by what is in it, not by what it is
called. A directory of nothing but source is the program and is already
handled; a directory with anything else in it is data and travels whole,
however deep it goes. Version control, caches, virtual environments and the
output directory are passed over, and py2bin says which, because something
not carried is worth seeing rather than guessing at.

### 0.9.13 - the allocator was 32-bit on Windows

**Every `malloc` on a Windows target handed back a truncated address.** The
arena's bump pointer was held in an `unsigned long`, which is eight bytes on
every platform py2bin targets except the one it matters on: Windows is LLP64,
where a `long` is four bytes and a pointer is eight. The kernel maps the arena
wherever it likes and on a 64-bit process that is usually above the four
gigabytes that fit, so the top half of the address was cut off and every
block handed out was a low address belonging to nobody.

Nothing said so. A program that never allocates was unaffected, which is most
of the corpus; one that did either faulted or quietly wrote somewhere else.
It was found by a real program on a real machine: SidecarBridge got as far as
its own error dialog, reporting `E_POINTER` from a WebView2 call it had handed
an object that `new` had just failed to make.

`size_t` now, which is the width of a pointer on every target - which is the
property being relied on, so it is the one named. A corpus program checks
that an address above four gigabytes survives the allocator intact, and a
test reads the header to make sure no address goes back into a `long`.

**And then the first `malloc` on Windows still answered nothing.** The same
program, the same dialog, the same `E_POINTER` - after the address width was
fixed. The cause was next to it and had been there all along.

The Microsoft x64 ABI gives a callee 32 bytes above the return address to
spill its four register arguments into, and makes the *caller* reserve them.
py2bin's module body does. A *function* body does not: the frame a function
gets is `_frame_bytes(slots, 0)`, which is exactly its locals and nothing
else, so a call that does not reserve the space itself lets the callee write
over that function's own first four slots.

`HeapInit` - the one-time `VirtualAlloc` that reserves the arena - was such a
call, and it runs inside `malloc`:

```c
void *malloc(size_t __n) {
    if (__py2bin_heap_end == 0) {
        __py2bin_heap_bump = (size_t)__py2bin_arena();  /* clobbers __n */
    }
    __n = (__n + 15) & ~15;
    if (__n > __py2bin_heap_end - __py2bin_heap_bump) return NULL;
```

`__n` is slot zero. So the first `malloc` in every Windows program compared a
size it no longer held against the arena and answered NULL - and only the
first, because the reservation happens once. Everything after it was correct,
which is exactly what made it invisible: the window opened, the program ran,
and the single allocation that failed was whichever one happened to come
first. In SidecarBridge that was the WebView2 callback, so the loader was
handed a null handler and said `E_POINTER` - and the program blamed a missing
runtime, on a machine where the runtime was installed and answering.

Four other call sites had the same gap - `Write`'s `GetStdHandle` and
`WriteFile`, and both `ExitProcess` paths. All nine Windows call sites reserve
it now: 32 bytes, or 48 where the call takes a fifth argument and writes a
count back above the shadow area. Three tests read the emitted `.text` and
fail if any `call [rip + disp32]` in it has no reservation in front of it.

Twice now a Windows-only mistake in this file has been found by running a real
program on a real machine rather than by anything here. Both were invisible to
2,000 tests and 300 corpus programs for the same reason: they are properties
of an ABI that only one target has, and nothing on this side of the build
executes that target's code.

### 0.9.13 - a Windows WebView2 program, from C++, with no toolchain

SidecarBridge - three C++ files, a fetched WebView2 header and a vendor DLL -
builds to a 627 KB PE32+ for windows-x86_64 with py2bin alone. What it needed
along the way is listed under [C++, translated to C](#c-translated-to-c):
`Callback<I>(lambda)` written out as the class it is - carrying the enclosing
object or carrying nothing, `[]` being as ordinary as `[this]` and having been
refused until a diagnostic needed one - `--library` for a DLL
somebody else shipped, `wWinMain` and the desktop subsystem, `swprintf`, and
the ordinary C++ a corpus program never happens to write - `operator&` on a
holder, a method called on a pointer parameter, `operator=` chosen by what is
being assigned, a destructor that must not run at a `return` above the
declaration it belongs to.

Three of those were silent rather than loud, which is the kind worth naming.
A `{` inside a string literal opened a block that never closed, so the rest of
the statement was lifted out and put back with whatever had been written into
it in the meantime - inside the string. A reference to a class is a pointer
here, which made `r[i]` on one read as an element of an array rather than as
the class's own subscript. And `path a = base() / "web";` read the call as the
whole initialiser and dropped the operator, so the program compiled and
quietly did half of what it says.

### 0.9.13 - what a real Intel Mac refuses

Two bugs, both fatal, both invisible on Apple silicon and invisible under
Rosetta. **If you build anything for macOS, upgrade.**

*Every `compile-capi` x86-64 binary segfaulted before printing anything.*
System V wants `rsp` 16-byte aligned at the call instruction. An image the
kernel starts is already aligned; one entered through `LC_MAIN` is not,
because dyld *calls* it and the return address is already on the stack. The
entry frame was a multiple of 16, which preserved that 8 and handed every call
out of the entry a stack misaligned by exactly it - and the first `movaps` to a
stack slot in the callee raises a general-protection fault. CPython's start-up
does one, so the crash was inside `_PyRuntimeState_Init`. Only the entry was
ever wrong: an internal function pushes `rbp` first, and that push corrects the
8.

*Every macOS freeze bundle carried a mis-signed interpreter.* The framework is
signed as a *bundle* - its code directory hashes an `Info.plist` and a
`_CodeSignature` that a freeze bundle does not carry - and the standard library
beside it is pruned. What shipped was a signature describing something that was
not there, and `codesign` had been saying so for months: "invalid Info.plist".
Apple silicon loads it anyway. Rosetta loads it anyway. A real Intel Mac
refuses the dylib outright, so the program dies before a line of it runs,
naming a library sitting exactly where the bundle put it. The bytes were never
corrupt - the shipped framework is byte-identical to python.org's. It was the
claim attached to it that had stopped being true. Anything this alters is now
signed again over what it actually is.

This was not specific to Intel, or to universal builds. The signature has been
wrong in every macOS freeze bundle since the pruning was added; arm64 simply
never refused one.

### 0.9.12 - all six targets have now been run

No change to the compiler. This records a verification result that 0.9.11 was
published too early to carry.

py2bin has six targets, and until 0.9.11 two of them - both Windows - had only
ever been *read*: parsed, checked against the format, disassembled, never
started, because there is no Windows machine here. 0.9.11 fixed the four bugs
that came out the first time somebody started one. Windows arm64 was still
untried when it went out, and has since passed on a Windows 11 ARM64 virtual
machine, which runs ARM64 instructions on an ARM64 processor - the code is
executed, not translated.

So every target has now had its output compared against CPython's on a machine
that actually ran it: darwin-arm64 natively, both Linux targets in containers,
and darwin-x86_64 and both Windows targets on the author's hardware.

Worth saying plainly, because it is the whole lesson of 0.9.11: reading a
generated image tells you it is well formed, and tells you nothing about
whether it runs. Four bugs lived in that gap, every one of them fatal to every
Windows binary py2bin produced, and not one of them in the compiled code.

### 0.9.11 - the Windows binaries had never been started on Windows

Every release before this one was found by compiling a program and comparing
its output against CPython, on macOS and Linux. The Windows images had only
ever been *read* - parsed, checked against the format, disassembled - and
never started, because there is no Windows machine here.

Then somebody started one. Four bugs, over four runs on real hardware. Every
one was fatal to every Windows program the compiler produced, and not one
could have been caught by comparing output, because in all four the program
never reached a `print`. They are worth reading as a set: they are all the
same kind of mistake, and none of them is in the compiled code.

*Every native-tier `.exe` was unloadable.* The import table's own addresses -
the DLL name, the lookup table, the address table - were computed against a
data section fixed at `0x2000`, which was correct only while the code fitted in
one page. Real programs are forty times that, so the section moved and those
addresses pointed into the middle of the code. Windows read machine code as a
DLL name and refused the image. A sibling bug, the two sections overlapping,
had been found and fixed earlier; the addresses *inside* the table were left
pointing at the old place. Both architectures, every program.

*A frozen `.exe` threw away everything it printed.* The launcher started the
real program with `bInheritHandles` false and `CREATE_NO_WINDOW` set, so the
child inherited none of the launcher's standard handles and got no console
either. It failed on its first `print`, and the traceback went to the same
missing handle. From outside: a silent exit 1 with two empty files.

Both are now checked by reading the generated image the way the loader reads
it - resolving every import RVA and asserting it lands in the data section, on
a program deliberately larger than one page.

**Then the second run found the third.** With the native tier passing, the
frozen executable still exited 1 with nothing to say. Its launcher is a copy
of `python.exe`, and Windows resolves an executable's imported DLLs from the
directory that executable is in - but the launcher was moved to the bundle
root while the runtime pack kept its own `runtime/` directory, leaving
`pythonXY.dll` one level down. `CreateProcess` fails outright in that case,
before any of the program's code runs.

It had always worked when built *on* Windows, because there the runtime is
staged at the bundle root and "root" and "beside the interpreter" are the same
directory. Cross-built they are not. The launcher now goes beside the
interpreter wherever that is; a bundle built on Windows is unchanged.

The one-file launcher also said nothing when it failed, which is why this took
a second run to see: its PowerShell stage wrote errors to the error stream,
and PowerShell serialises that stream as CLIXML when it is redirected, so a
failure arrived as a page of XML containing only a progress record. Errors are
now written to stdout as a sentence.

**And the third run found the same bug a third time.** With the layout fixed
the frozen program ran and exited 0 - printing nothing. Its PowerShell stage
started the program with `CreateNoWindow`, which is the mistake the launcher
stub made one level up: a console program denied a console has nowhere to
write. The program was correct every time; its output was being thrown away.
It is set now only for a windowed build, where suppressing a console is the
point.

Three failures, one shape. A child process is not given its parent's console
by default - not by `CreateProcess`, and not by the .NET wrapper over it - and
a program that cannot write looks exactly like a program with nothing to say.

**The fourth run passed.** All three tiers, on a physical x86-64 Windows
machine: the native `.exe`, the frozen `.exe` carrying its own CPython, and
the C-API `.exe` driving a CPython 3.14 it downloaded itself. Linux passes the
same way, and so does Windows arm64 on an ARM64 virtual machine - the target
that had never once been started until this release.

So all six targets have now been *run*, not merely built and inspected. What
is worth keeping is that the compiled programs were right on every one of
those runs: all four bugs were in where the executable was put and what its
children were allowed to write to. Reading a generated image tells you it is
well formed. It does not tell you it runs.

### 0.9.1 - 0.9.10

Ten releases in one sitting, all of them found by compiling a shape and
comparing what came out against the interpreter. Roughly five hundred shapes
went through, and the pattern was that bugs came from *new kinds* of test
rather than more of the same kind - eighteen sweeps of output comparison went
quiet, and then packaging, a wheel-install test, leak measurement and an
adversarial sweep each found something on their first run.

**A program is more than a flat directory of files.** Only a `.py` beside the
entry was compiled in, so `import pkg` failed at start-up. Now: packages and
submodules, relative imports resolved at compile time, PEP 420 namespace
directories, `importlib.import_module("pkg.thing")` where the name is written
down, and everything under `src/` reached the way programs reach it.

**Three things that write on a function.** `abc.abstractmethod` sets one
attribute, `functools.wraps` sets six, and an annotated `def` writes
`__annotations__`. A compiled function has no `__dict__`, so all three failed
- and between them they are how a great many programs begin, how nearly every
decorator is written, and how most modern Python is typed. Each is now handed
an object that can hold what it writes. With them came `f.__doc__`,
`inspect.signature` showing the annotations, and `singledispatch`.

**The ends of a generator's life.** `next` on an exhausted generator stopped
answering at all; `close` on a fresh one complained; a delegating generator
did not close what it delegated to, so a cancelled `asyncio` task ran no
cleanup. `athrow` did not exist.

**Which exception is being handled belongs to the call, not the thread.** A
`finally` now runs with the exception it interrupted on record, so what it
raises is chained to it; and a body whose handler raised no longer leaves its
exception on record for the caller.

**A `try` in a loop leaked.** A handler holds two references and released them
only when the clause fell off its end, so `except E: raise F(...)` leaked 160
bytes a turn. Found by measuring memory across two identical runs, which is a
thing output comparison cannot see.

**`dir()` and comprehension capture stopped being refusals.** `dir()` is
`sorted(locals())`, which the compiler already builds. And a closure made in a
comprehension now shares the comprehension's variable, as Python does - the
cell is named after the comprehension, so a variable of the same name outside
is never involved.

**Smaller, and each a wrong answer rather than a missing one:** `__spec__` fell
through to the builtins module and answered `"builtins"`; `globals()` outside
the entry module read the entry's; two closures that captured nothing were the
same closure, so a set of them kept one; a source file was always read as
UTF-8, so a Latin-1 file was refused; and the same source compiled twice gave
two different binaries, because a temporary was named after an object's
address.

**`freeze` did not work at all on Homebrew's Python** - the bundle carried a
`bin/python3` that hands over to a file the bundle did not have. It built
cleanly and died at start-up. It survived every earlier test because the
`python3` on the machine that built them is python.org's.

**Windows ARM64 wrote one word where two were reserved**, leaving an invalid
instruction in the middle of the code that any large enough program would
reach.

### 0.9.0 - what a compiled function could not do

Metaclasses, `enum` and `dataclasses`, generator and `async def` methods,
async generators and comprehensions, `locals()`, a real `globals()`, `eval`
and `exec`, and `sys.exc_info()` inside an `except`. Two silent wrong answers
went with them: a default argument was evaluated on every call rather than
once at the `def`, so `def f(x=[])` did not share its list; and a closure could
reach a module-level function where a parameter of the same name shadowed it.

### 0.8.9 - verdicts, borrowed references, and a leak in every `try`

Fourteen of twenty-seven measured rows moved to 0.80× or better. `except`
clauses leaked their exception class per turn; borrowed references were
returned where owned ones were expected; a `with` whose `__exit__` returned a
value did not suppress.

### 0.8.5 - 0.8.8

`0.8.8` refused what an archive may not contain - a member escaping its
directory, a symlink pointing out of it - and pinned the same rule in the
bootstrapper. `0.8.7` fixed long functions overflowing a frame and names read
before binding. `0.8.6` repaired 0.8.5 and added the six-target matrix.
`0.8.5` was the first correctness sweep: wrong answers on integer overflow,
`-0.0`, evaluation order, and `__len__` called twice.

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

1905 tests, no dependencies, nothing to install. Five modules want pytest's
fixtures and skip themselves without it; `python -m pytest tests` runs those
too, for 2005. Two of them compile a program in a *fresh interpreter* and
assert that `ctypes`, `_ctypes` and `subprocess` are absent from `sys.modules`
afterwards - one through the Python path, one through the C++ one. "No
toolchain" is not a promise here; it is a thing the suite checks. The suite fails if any module under `src/` imports
`subprocess`, `multiprocessing`, `pty`, `distutils` or `setuptools`, or names
an external toolchain as a value - which is what keeps the zero-toolchain
claim honest rather than aspirational.

[^corpus]: Freshly measured, on this machine, at the commit that carries this
    line: each of the 889 programs compiled with `compile-capi` for the host
    and run, and its stdout and exit code compared against CPython's. The
    harness is scratch rather than committed, which is why the method is
    written out here rather than pointed at. Comparing stderr as well - which
    means comparing tracebacks a compiled program cannot produce - the figure
    is 804; see *It behaves as CPython does* for what the other 82 are. What
    is checked on every change is the 2005-test suite.
