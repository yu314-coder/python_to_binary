# python-to-binary

`py2bin` is a self-contained compiler and application builder implemented with
the Python standard library. The project itself has no runtime or build
dependencies: no Cython, Nuitka, mypyc, Rust, C, C++, PyInstaller, PPCI, native
bootloader, assembler, linker, or SDK.

It has four deliberately separate execution paths. They must not be confused:

1. **Native compile:** Python AST → py2bin IR → py2bin optimizer → handwritten
   x86-64/ARM64 instructions → ELF, PE, or Mach-O. This invokes no external
   toolchain, and the generated program needs no Python runtime.
2. **Runtime freeze:** arbitrary CPython projects and target-compatible packages
   are collected with an embedded CPython runtime. The default output is one
   self-extracting `.exe` or `.bin`; this is compatibility packaging, not
   native translation of the application.
3. **Lightweight bundle:** `.pyz`, executable `.bin`, and directory formats
   package project code and dependencies but use a compatible target Python.
4. **Portable-C frontend:** a useful typed subset of Python becomes readable C
   source, or a checksummed `.py2cbin` C-source container. Imports automatically
   plan for the compatible CPython bundle instead of pretending native packages
   can be translated from Python source.

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

## Quick start

No installation is required in the repository:

```sh
# Produce Windows, Linux, and macOS x86-64/arm64 artifacts in one command.
# This does not invoke an assembler, linker, compiler, SDK, or target runtime.
PYTHONPATH=src python3 -m py2bin compile-all examples/native_hello.py \
  --output-dir dist/all --mac-app --clean

PYTHONPATH=src python3 -m py2bin compile examples/native_hello.py \
  --output dist/native-hello --clean
./dist/native-hello

# Translate supported Python to portable C without invoking a C compiler.
PYTHONPATH=src python3 -m py2bin emit-c examples/c_program.py \
  --output dist/c_program.c --clean

# Explain whether a program can use the C subset or needs CPython bundling.
PYTHONPATH=src python3 -m py2bin plan-c app/main.py

# List common-library support, or inspect one source file without importing it.
PYTHONPATH=src python3 -m py2bin capabilities
PYTHONPATH=src python3 -m py2bin capabilities app/main.py --json

# Turn an installed/staged package tree into a standards-structured wheel.
PYTHONPATH=src python3 -m py2bin wheel build/package-root \
  --output-dir dist/wheels --name my-package --version 1.0

# Fetch pinned imported source and attempt the real native compiler only.
PYTHONPATH=src python3 -m py2bin compile app.py -o dist/app \
  --source-lock py2bin-sources.lock.json \
  --source-cache /Volumes/D/py2bin-source-cache

# A native Apple Silicon application bundle:
PYTHONPATH=src python3 -m py2bin compile examples/native_hello.py \
  --target darwin-arm64 --app --output dist/NativeHello --clean

# Automatically choose native code when possible and otherwise emit one
# self-extracting embedded-CPython file.
PYTHONPATH=src python3 -m py2bin assemble examples/generic_app/main.py \
  --source-root examples/generic_app --dependency-mode none \
  --output dist/GenericApp --compact --clean
./dist/GenericApp.bin hello

# Short compatible-bundle spelling. `bundle` is an alias for `freeze`, and
# `--optimize-size` is an alias for the explicit `--compact` policy.
py2bin bundle app.py -o dist/App --optimize-size --clean

PYTHONPATH=src python3 -m py2bin analyze examples/hello/main.py
PYTHONPATH=src python3 -m py2bin build examples/hello/main.py \
  --source-root examples/hello --format bin --output dist/hello --clean
./dist/hello
```

Or install it into a virtual environment:

```sh
python3 -m pip install python-to-binary
py2bin build app/main.py --source-root app --output dist/my-app --format bin
```

Install the current GitHub version:

```sh
python3 -m pip install \
  "git+https://github.com/yu314-coder/python_to_binary.git"
```

See the [detailed compiler, bundling, target, and release guide](docs/DETAILED_GUIDE.md).

## Formats

| Command/mode | Output | Application logic | Installed Python needed on target |
|---|---|---|---:|
| `compile` | ELF, PE, Mach-O, macOS `.app` | py2bin-generated machine code | No |
| `emit-c` | `.c` or `.py2cbin` | C source, not yet an executable | N/A |
| `build --format pyz` | Python zip application | CPython bytecode/source | Yes |
| `build --format bin` | Executable Python zip application | CPython bytecode/source | Yes |
| `build --format dir` | Project, packages, and launcher | CPython bytecode/source | Yes |
| `freeze` / `bundle` (default) | One self-extracting PE/ELF/Mach-O file, or macOS `.app` with one embedded payload | CPython bytecode/source | No |
| `freeze --onedir` / `bundle --onedir` | Unpacked embedded-runtime directory | CPython bytecode/source | No |

Every executable above is a valid OS binary or executable launcher, but only
`compile` translates the supported application logic into py2bin-generated CPU
instructions. `freeze` produces a real native launcher around embedded CPython;
calling the contained Python application “natively compiled” would be
incorrect.

## Claims audit

The following wording is intentionally strict. “No installed Python on the
target” is not the same claim as “the application does not use CPython.”

| Claim | Accurate? | Exact meaning |
|---|---:|---|
| No GCC, Clang, assembler, or linker is required | Yes | `compile` writes ELF, PE, and Mach-O bytes directly. `freeze` copies a compatible CPython runtime and installed package files. |
| The target computer does not need Python installed | Yes, for `compile` and `freeze` | A `compile` artifact has no Python runtime. A `freeze` artifact carries its own CPython runtime. The lighter `build` formats still need compatible target Python. |
| A complete frozen application does not use CPython | No | `freeze` embeds and starts CPython; only the supported `compile` subset replaces Python execution with generated machine code. |
| Third-party packages do not need a Python runtime | No | NumPy, Torch, `bpy`, Manim, and similar packages are imported by the embedded CPython runtime in `freeze` mode. |
| Arbitrary Python is translated completely to machine code | No | `compile` currently accepts the documented small, static subset and rejects unsupported syntax with a source location. |
| py2bin itself has no third-party Python dependency | Yes | The package imports only Python standard-library modules, and its build/runtime dependency lists are empty. Tests enforce both properties. |
| The build computer does not need Python | No | Building requires Python 3.10+ to run py2bin. No native compiler toolchain is required for the supported direct-binary path. |

## Source-only builds: exact boundary

“Use only the application source and py2bin” has a precise, limited meaning:

- For programs inside the documented static `compile` subset, it is enough to
  have the source, Python 3.10+, and py2bin on the build computer. py2bin can
  write Windows PE, Linux ELF, and macOS Mach-O files without Wine, Rosetta,
  an assembler, a linker, a target SDK, or a target Python runtime.
- For arbitrary Python, source files alone are not enough. Full Python
  semantics require CPython or a future complete py2bin runtime.
- An imported third-party package must also exist as target-compatible package
  data. An import name does not contain that package's implementation.
- `freeze` embeds the current compatible CPython runtime and copies compatible
  installed packages or supplied wheels. It does not manufacture a missing
  Windows runtime on macOS, translate a macOS native wheel into a Windows
  wheel, or recreate a package whose files were never supplied.

For example, the `manim_app` repository imports `webview`, but does not contain
pywebview's implementation. It also installs Manim and related tools into a
separate environment. A working Windows build therefore needs Windows CPython,
pywebview and its Windows backend, plus every other required package and native
file. With only the `manim_app` source and py2bin, the current compiler must
reject the program. Producing a PE header that cannot start the application
would be a broken artifact, not successful assembly.

`manim_app` also imports `winpty`. Its full terminal mode therefore requires a
matching `pywinpty` wheel; for CPython 3.12 on Windows x86-64 that wheel must
carry the `cp312` and `win_amd64` tags. py2bin extracts its `.pyd`, DLL, agent,
and console helper files as wheel data. Excluding `winpty` intentionally selects
the application's simpler subprocess fallback instead.

This limitation is not specific to pywebview or Manim. “Any Python program and
every third-party package from source alone” would require py2bin to implement
the complete Python language, standard library, extension ABI, GUI frameworks,
and each missing third-party implementation. That work is not complete and is
not claimed.

## Pinned source download and native attempt

`compile --source-lock` detects statically imported non-stdlib modules,
downloads their locked source archives through Python's HTTPS client (no pip),
verifies SHA-256, extracts them without running project code, and attempts the
handwritten native frontend. A local archive path can be locked for offline
builds.

Example lock:

```json
{
  "schema": 1,
  "sources": {
    "demo": {
      "url": "https://github.com/owner/demo/archive/COMMIT.tar.gz",
      "revision": "FULL_COMMIT_ID",
      "sha256": "64_LOWERCASE_HEX_DIGITS",
      "subdirectory": "src"
    }
  }
}
```

Build:

```sh
py2bin compile app.py -o dist/app \
  --source-root . \
  --source-lock py2bin-sources.lock.json \
  --source-cache /Volumes/D/py2bin-source-cache \
  --target darwin-arm64
```

After the lock is supplied, source discovery and fetching are automatic.
py2bin intentionally does not guess a Git repository from an import name:
import names are not globally unique, repositories can be renamed or
compromised, and an unpinned `main` branch is not a reproducible input.

The fetcher:

- accepts credential-free HTTPS URLs or explicit local archive paths;
- requires an immutable revision label and exact SHA-256;
- accepts ZIP or tar archives;
- rejects traversal paths, links, special files, duplicate/case-colliding
  members, excessive member counts, and configured download/expanded-size
  limits;
- records and rechecks a hash of the extracted source tree;
- never imports the downloaded package or runs `setup.py`, build backends, or
  shell commands.

The current successful cross-package native form is deliberately narrow:
`from MODULE import CONSTANT`, where the export is statically evaluable by the
native frontend. The value is lowered into py2bin IR and handwritten
x86-64/ARM64 instructions. Dynamic functions, classes, package initialization,
Cython-generated C, C/C++/Rust/Fortran/CUDA sources, CPython extensions, and
general library imports fail with a source location. This command has no
CPython or PyInstaller fallback, so a failed native conversion cannot be
mistaken for a native artifact.

`fetch-sources` performs only the verified download/extraction phase:

```sh
py2bin fetch-sources app.py --source-root . \
  --source-lock py2bin-sources.lock.json \
  --source-cache /Volumes/D/py2bin-source-cache --json
```

Native compile targets currently implemented are `linux-x86_64` (ELF),
`linux-arm64` (ELF), `darwin-x86_64` and `darwin-arm64` (Mach-O), and
`windows-x86_64` and `windows-arm64` (PE `.exe`).
Run `py2bin targets` to list them. The first native
frontend supports module constants, static strings/f-strings, a signed 64-bit
runtime for variables and arithmetic, comparisons, `if`, `while`,
`for NAME in range(...)`, `break`, `continue`, `print()` of compile-time
values, and integer-expression exit status. It rejects everything else with a
source location rather than producing a subtly incorrect executable.

The word “supports” is intentionally narrow:

| Python feature | `compile` now | Exact behavior |
|---|---:|---|
| Literal `str`, `bytes`, `int`, `float`, `bool`, `None` | Yes | Represented while lowering the static program |
| Single-name assignment and annotation | Yes | Static values are folded; runtime integer values use native stack slots |
| Integer `+`, `-`, `*`, bitwise operations, shifts | Yes | Runtime signed 64-bit instructions; overflow wraps to 64 bits rather than creating Python big integers |
| Integer comparisons and dynamic `if` | Yes | Runtime signed comparisons and native branches |
| Constant arithmetic, Boolean and conditional expressions | Yes | Evaluated at build time when no runtime value is involved |
| Constant `if` | Yes | Only the selected branch is emitted |
| `while`, `for NAME in range(...)`, `break`, `continue` | Yes | Native branches; `range` step must be a nonzero integer constant |
| Simple f-string | Yes | Every formatted value must be compile-time constant |
| `print(...)` | Yes | Constant UTF-8 bytes are emitted through an OS write API/syscall |
| `SystemExit(integer)` / `sys.exit(integer)` | Yes | Constant or runtime integer expression becomes the OS process-exit value |
| Runtime input/arguments, dynamic printing, containers, functions, classes, general exceptions | No | Rejected by `compile`; compatible mode needs CPython |
| Imports | Only restricted `sys` | `import sys` exists solely for `sys.exit`; imported libraries are not compiled |

The bundle-format `bin` uses Python; `py2bin compile` produces actual machine
code. These writers encode executable headers, import tables, system calls, and
instructions directly rather than shelling out to an assembler.

## Self-written optimizer

The native compiler has its own target-independent optimizer. It currently:

- propagates assignment constants;
- folds arithmetic, Boolean, comparison, conditional-expression, and f-string
  constants;
- selects only the reachable arm of a constant `if`;
- removes assignments that have no runtime representation;
- removes empty writes;
- merges adjacent writes to reduce system calls;
- removes operations after the first process exit;
- inserts one canonical successful exit when required.

The compiler also lowers runtime integer variables and structured control flow
to target-independent IR. The x86-64 and ARM64 backends encode stack loads and
stores, arithmetic, comparisons, conditional/unconditional branches, and
process exit directly. That runtime path is not described as constant folding.

These transformations are deterministic and covered by equivalence tests. No
optimizer can truthfully be “fully optimal” for every Python program. py2bin
therefore documents each safe optimization instead of claiming universal
optimality, and unsupported dynamic semantics remain compilation errors.

Select the OS and architecture independently:

```sh
py2bin compile program.py -o dist/program.exe --os windows --arch arm64
py2bin compile program.py -o dist/program --os linux --arch x86_64
py2bin compile program.py -o dist/program --os macos --arch arm64
```

Accepted architecture aliases include `x64`, `amd64`, and `aarch64`. Exact
canonical targets remain available through `--target`.

## Python-to-C path

`emit-c` lowers a deterministic Python subset to ISO-style C source using only
the Python standard library. The current subset includes numeric and string
constants, local variables, arithmetic, comparisons, Boolean expressions,
`if`, `while`, `for range(...)`, functions, returns, `break`, `continue`,
`print`, and simple f-strings.

```sh
py2bin emit-c program.py -o program.c
py2bin emit-c program.py -o program.py2cbin --container
py2bin plan-c program.py
```

A `.py2cbin` file is a versioned, checksummed container holding generated C; it
is not an executable. Turning C into machine code normally requires a C
compiler. The toolchain's `compile` command bypasses that requirement for its
supported native Python subset by writing ELF, Mach-O, or PE bytes directly.
This project never silently invokes a system assembler, linker, or C compiler.

`plan-c` returns `c-source` when direct C translation is safe and
`cpython-bundle` when imports or unsupported Python semantics require the
compatibility path. Programs importing `manim`, `torch`, `transformers`, `bpy`,
or `webview` therefore keep their real CPython and native-extension behavior
rather than receiving incomplete generated C.

## Zero-toolchain contract

Native compilation requires only Python 3.10+ and this repository/package on
the build machine. It does not inspect or invoke `as`, `ld`, `clang`, `gcc`,
Visual Studio, Xcode, a target SDK, Docker, a virtual environment, or a target
Python. `compile-all` cross-compiles every implemented target from the same
process. Its generated native artifacts require only the matching operating
system and CPU.

This guarantee does not create missing inputs. If an application imports a
third-party library, that library's source, wheel, or adapter payload must be
supplied to py2bin. Native libraries such as Torch and `bpy` contain
target-specific compiled code; future standalone framework mode will consume
provided wheels/runtime archives directly without installing them into the
build environment.

## Dependency collection

`py2bin` parses imports without executing the application, maps top-level
packages to installed distributions, and copies every file declared by those
distributions. Dynamic imports can be declared explicitly:

```sh
py2bin build render.py -o dist/render --include manim --include torch
```

Dependency modes are:

- `closure` (default): imported distributions plus their installed dependency
  closure.
- `imported`: only directly imported distributions.
- `none`: project source only.

Use `--exclude MODULE` for optional backends you do not ship. `analyze` returns
exit status 1 when it sees an unresolved import.

### Wheel and prebuilt-Cython pipeline

`py2bin wheel` creates a wheel using only the standard library. The input is an
already-staged tree in the layout that should be installed into
`site-packages`. Python files, package data, and already-built Cython/native
extensions are stored byte-for-byte. `METADATA`, `WHEEL`, `top_level.txt`, and
the SHA-256 `RECORD` are generated by py2bin.

Pure Python example:

```sh
py2bin wheel build/package-root -o dist/wheels \
  --name example-package --version 1.0
```

Prebuilt Windows CPython 3.11 x86-64 Cython extension:

```sh
py2bin wheel build/windows-cp311 -o dist/wheels \
  --name example-native --version 1.0 \
  --python-tag cp311 --abi-tag cp311 --platform-tag win_amd64
```

Native payloads cannot use `py3-none-any`; exact interpreter, ABI, and platform
tags are mandatory. A `.pyx` or `.pxd` file may be included as source data, but
the command reports that it was not compiled. Supply the corresponding
target-built `.pyd` or `.so` when runtime importability is required.

Feed the created wheel directly into the compatible bundle:

```sh
py2bin freeze app/main.py --source-root app -o dist/App \
  --runtime-pack runtimes/windows-cp311-amd64 \
  --target windows-x86_64 \
  --wheel dist/wheels/example_native-1.0-cp311-cp311-win_amd64.whl \
  --icon icon.ico

# Output: dist/App.exe
```

This pipeline does not invoke Cython, a C compiler, or a linker. If Cython is
used, its output must be built before `py2bin wheel`, normally once per target.
Implementing a machine-code backend for arbitrary Cython-generated C would
require a complete C preprocessor/compiler, target ABI, object linker, CPython
C API, and dynamic loader; the current handwritten backend does not claim
those components.

For arbitrary CPython packages, freeze the interpreter and complete package
trees into a target-side bundle:

```sh
PYTHONPATH=src python3 -m py2bin freeze app/main.py \
  --source-root app --output dist/MyApp \
  --include torch --include transformers --clean

./dist/MyApp.bin
```

`freeze` carries the current compatible CPython runtime, standard library,
native extension modules, distribution metadata, and package data. It can also
consume wheels directly without pip or installation:

```sh
py2bin freeze app.py -o dist/App --wheel wheels/custom_backend.whl
```

Frozen bundles are specific to the build runtime's OS, CPU, Python ABI, and
accelerator variant. By default, the unpacked runtime tree is compressed behind
a handwritten native launcher in one `.bin` or `.exe`. `--onedir` keeps that
tree unpacked for inspection and debugging. Cross-target compatibility builds
require an explicit matching runtime pack and complete target-wheel closure.
Dynamic imports still need `--include`.

Windows one-file outputs accept an `.ico` directly. py2bin writes the PE
resources itself; no resource compiler is invoked. For `--app`, it replaces
the inherited Python icon and version information on both the outer one-file
launcher and the embedded app host. `--name` supplies the app/product
identity, and `--icon` supplies the executable icon:

```sh
py2bin freeze app.py -o dist/App --target windows-x86_64 \
  --runtime-pack runtimes/windows-cp311-amd64 \
  --wheel-dir wheels/windows-cp311 --icon icon.ico --app --clean
```

### Windows GUI identity and startup

For Windows, `--app` makes both executable layers GUI-subsystem PE files and
uses the runtime pack's `pythonw.exe` as the inner app host. It suppresses
console windows for the extractor and compatible CPython process. Omit `--app`
for a console program.

| What Windows sees | Implementation | Identity |
|---|---|---|
| Distributed `NAME.exe` | Handwritten py2bin PE launcher carrying the ZIP payload | `NAME`, app version resource, and `--icon` |
| Cached/running `NAME.exe` | Renamed target `pythonw.exe` that loads the bundled CPython DLL | The same `NAME`, version resource, and `--icon`; inherited Python branding is removed |
| Loaded `python3XY.dll` and native packages | Bundled CPython, `.pyd`, and dependency DLLs | Still visible to diagnostic tools because this is the compatibility path |

Before application code creates a window, the bootstrap assigns an explicit
`PythonToBinary.NAME` Windows AppUserModelID using the exact wide-string API
signature. Microsoft documents that this ID identifies the process to the
taskbar and should be set during initial startup before UI is shown:
[AppUserModelIDs](https://learn.microsoft.com/en-us/windows/win32/shell/appids)
and
[`SetCurrentProcessExplicitAppUserModelID`](https://learn.microsoft.com/en-us/windows/win32/api/shobjidl_core/nf-shobjidl_core-setcurrentprocessexplicitappusermodelid).

A GUI toolkit can deliberately override its window icon. A previously pinned
`.lnk` shortcut can also retain its own old icon or AppUserModelID; py2bin does
not rewrite existing Windows shortcuts, so unpin and re-pin the rebuilt
executable when checking a new identity build.

These changes fix Windows presentation and grouping, not compilation mode. A
frozen Manim/Torch/pywebview application still executes bundled CPython. Seeing
`python3XY.dll` in a module list is therefore expected and is not equivalent to
the taskbar incorrectly identifying the application as Python.

The generated isolated-runtime path includes the matching embedded
standard-library archive, such as `python312.zip`; target-compatible runtime
packs must supply that archive or an equivalent `Lib` tree. py2bin rewrites
both the executable-specific and versioned-DLL `_pth` files because Windows
CPython can give the latter priority.

For the fastest first launch of a large Windows GUI, use the unpacked app
layout. It starts `pythonw.exe` directly and performs no one-file extraction:

```sh
py2bin freeze app.py -o dist/App --target windows-x86_64 \
  --runtime-pack runtimes/windows-cp311-amd64 \
  --wheel-dir wheels/windows-cp311 --app --onedir --compact --clean

# Start dist/App/App.exe and distribute the complete dist/App directory.
```

This is a distribution tradeoff, not a different compilation mode. A one-file
build must materialize native `.pyd` and `.dll` files before Windows can load
them; packages such as Torch and `bpy` can therefore make the first launch
substantially slower. The content-addressed one-file cache makes later launches
reuse the extracted tree. The one-file launcher passes its path and original
command line directly to its extraction process instead of querying WMI/CIM.
On a cached Windows launch, the extractor performs one direct
`System.IO.File.Exists` marker check before creating any mutex; the named mutex
and its second race-prevention check are used only on a cache miss. Path,
delete, move, and process-start operations use direct .NET APIs instead of
PowerShell filesystem-provider cmdlets.
The generated bootstrap embeds its entrypoint and AppUserModelID at build time,
avoiding per-launch JSON and `pathlib` work; traceback support is imported only
after an application error. ZIP level 6 is used as the default size/extraction
balance.

On macOS, `freeze --app` wraps the one-file payload in the directory structure
required by the Apple `.app` format. The runtime archive is embedded in
`Contents/MacOS/NAME`; there is no unpacked `Contents/Resources/bundle`.
`--icon` accepts ICNS, a square PNG at a standard icon size, or a
PNG-backed multi-resolution Windows ICO. ICO-to-ICNS conversion is implemented
in pure Python and skips only ICO sizes, such as 48×48, that have no matching
modern ICNS record:

```sh
py2bin freeze app.py --source-root . -o dist/MyApp \
  --app --name MyApp --icon icon.ico --include webview --compact --clean

dist/MyApp.app/Contents/MacOS/MyApp
```

A dependency-free generic resource/argument example is included for smoke
testing independently of large frameworks:

```sh
py2bin freeze examples/generic_app/main.py \
  --source-root examples/generic_app \
  -o dist/GenericApp --app --compact --clean

dist/GenericApp.app/Contents/MacOS/GenericApp hello
```

The generated `Info.plist` declares `AppIcon.icns`, and the icon is stored in
`Contents/Resources`. A frozen app does not require an installed target Python,
but third-party extensions still have to match the build OS, CPU, and Python
ABI.

The macOS app entry in `Contents/MacOS` is a directly executable Mach-O, not a
shell script. py2bin writes its instructions, ad-hoc code signature, resource
seal, and app metadata itself. The launcher starts the embedded CPython runtime
for full-library compatibility; it does not claim that a dynamic pywebview or
Manim application has been translated into the narrow native Python subset.

`--compact` and its descriptive alias `--optimize-size` omit distribution
tests, loose bytecode caches, CPython build/debug support, and documented
GUI/demo modules that are not used by a typical pywebview app. The policy now
applies consistently to installed distributions, supplied wheels, supplied
runtime-pack trees/ZIPs, and the Windows embedded standard-library ZIP. Runtime
`.pyc` members inside that standard-library ZIP are preserved.

On the retained CPython 3.11.9 Windows x86-64 runtime pack, this reduced
21,665,688 bytes to 20,588,888 bytes: 1,076,800 bytes, or 4.97%. Across the
72-wheel heavy-library reference closure, 2,412 test/cache files represented
38,198,995 uncompressed bytes and 9,547,608 bytes in the original compressed
wheels. A minimal real-runtime Windows GUI one-file comparison decreased from
11,539,423 to 10,899,205 bytes, saving 640,218 bytes or 5.55%. Final savings
vary with the application because py2bin recompresses the complete payload.

Leave size optimization off when the packaged program imports package test
suites, `tkinter`, `unittest`, `lib2to3`, or CPython build configuration files
at runtime.

## Heavy-library compatibility

Run `py2bin capabilities` for the catalog below, or
`py2bin capabilities APP.py --json` to audit the imports and native-subset
result of one entry file without importing or executing it.

| Import/project | Fully translated by `compile`? | Can `freeze` carry it? | What is still required |
|---|---:|---:|---|
| NumPy / SciPy / pandas / scikit-learn | No | Conditional | Matching CPython ABI and complete target wheels/native libraries |
| PyTorch / TorchVision | No | Conditional | Target wheels, C++ libraries, accelerator variant, and target GPU driver when used |
| TensorFlow / JAX | No | Conditional | Supported target runtime wheels; JAX also needs matching `jaxlib` |
| Transformers | No | Conditional | CPython, backend such as Torch, dependency closure, model/config/tokenizer files |
| `tokenizers` | No | Conditional | Its Rust extension compiled for the target CPython ABI |
| Manim | No | Conditional | CPython, target wheels, fonts/assets, and media tools such as FFmpeg; LaTeX when used |
| Matplotlib | No | Conditional | Target wheels, NumPy, rendering backend, fonts, and package data |
| Pillow / OpenCV | No | Conditional | Target extension wheels and their native image/media/GUI libraries |
| Blender `bpy` | No | Conditional | Exactly compatible Blender/`bpy`, CPython ABI, resources, OS, and CPU |
| pywebview (`webview`) | No | Conditional | CPython, target dependencies, and the OS webview framework/runtime |
| Gradio / Streamlit | No | Conditional | CPython server packages and frontend assets plus a browser/webview |
| Numba / llvmlite | No | Conditional | Mutually compatible target wheels, including the LLVM components |
| Requests / Flask / Django / FastAPI | No | Conditional | CPython, dependencies, and application templates/static/configuration data |
| Unknown third-party or local import | No by default | Conditional | Its real implementation and complete target-compatible dependency/data closure |

“Conditional” means the necessary files can be collected when they are
actually supplied and compatible. It does not mean every version exists for
every OS, architecture, or CPython ABI, and it does not mean py2bin has
translated the package into its own machine code.

“Supports all libraries” means the collector is generic and does not maintain a
hardcoded allowlist. It cannot guarantee that every third-party binary, driver,
external executable, license, network model, or platform service is portable.
It also cannot make mutually incompatible third-party requirements coexist in
one Python environment. For example, some current Manim and `bpy` releases
require incompatible NumPy major versions; use compatible releases or separate
runtime sidecars rather than bypassing package constraints.

Native compilation of those libraries is a different problem: Torch contains
millions of lines of precompiled C++/CUDA code, `bpy` is coupled to Blender, and
Manim invokes external tools. They can be made self-contained per target by
shipping their native components and a compatible embedded runtime, but they
cannot truthfully become one CPU-independent executable. The long-term native
API is an adapter ABI: pure-Python modules compile through py2bin IR, while
large native libraries link as target-specific prebuilt components.
Until that adapter ABI is complete, `freeze` is the full-compatibility engine.

## Comparison with dante-biase/py2bin

This project is unrelated to
[dante-biase/py2bin](https://github.com/dante-biase/py2bin). That repository
describes itself as a streamlined PyInstaller interface, declares
PyInstaller 3.6, Click 7.1.1, and py2x 1.0 as dependencies, and invokes
PyInstaller's `--onefile` mode in a subprocess. Its README lists macOS as its
compatibility target.

| Capability | This project | dante-biase/py2bin |
|---|---|---|
| Implementation dependencies | Python standard library only | PyInstaller, Click, and py2x |
| Actual Python-to-machine-code path | Yes, for the documented small static subset | No; it delegates packaging to PyInstaller |
| Arbitrary-package compatibility path | Embedded-CPython `freeze` bundle | PyInstaller one-file bundle |
| Direct binary writers | ELF, PE, and Mach-O; x86-64 and ARM64 | None in that wrapper |
| Installed Python required on target | No for `compile` or `freeze` | No for the produced PyInstaller bundle |
| Arbitrary Python becomes native machine code | No | No |

The fair comparison is therefore `freeze` versus its PyInstaller wrapper for
application compatibility, and `compile` versus a real compiler for native
translation. Calling either project's compatibility bundle a complete native
translation would be inaccurate.

## PPCI relationship

The pipeline is inspired by [windelbouwman/ppci](https://github.com/windelbouwman/ppci):
keep parsing, IR, instruction selection, linking, and file formats separate.
Unlike simply wrapping PPCI, this repository starts with its own narrow Python
frontend and direct executable writers so every emitted byte is controlled and
the dependency-free guarantee remains testable. PPCI's own documentation calls
its Python-to-IR support preliminary; this project therefore treats broad
Python compatibility as staged compiler work, not an already-solved claim.

## Runtime behavior

Single-file artifacts extract into a content-addressed cache before execution.
Set `PY2BIN_CACHE_DIR` to control its location. The app can read
`PY2BIN_BUNDLE_ROOT` to locate bundled resources. Rebuilding changes the cache
fingerprint; deleting the cache is safe when no bundled program is running.

“One file” describes distribution, not execution without extraction. The first
launch atomically expands the embedded runtime; later launches reuse it.
Windows launchers use the Windows PowerShell/.NET ZIP facilities included with
normal Windows 10/11 installations. The handwritten PE passes its own path and
original Unicode command line through the child environment, avoiding WMI/CIM
process queries. Cached Windows launches check the completion marker before
allocating the extraction mutex; first-run filesystem work uses direct
`System.IO` methods. Linux and macOS launchers use `/bin/sh` plus `tail`,
`head`, and `tar` from the base operating system. Extremely minimal Windows or
Linux images that remove those OS facilities need `--onedir`.

A macOS `.app` can never literally be one filesystem file because Apple defines
it as a directory bundle. py2bin minimizes it to the native executable carrying
the compressed payload, `Info.plist`, the code-resource seal, and optional
icon.

## Development

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The implementation intentionally depends only on the standard library. The
unit suite performs no package downloads and writes only inside its selected
temporary directory:

```sh
TMPDIR=/path/on/your/data/disk \
  PYTHONPATH=src python3 -m unittest discover -s tests -v
```

GitHub Actions is disabled for this repository at two levels: active workflow
files are absent from `.github/workflows/`, and the repository Actions
permission is disabled. Reference definitions remain inert under
`.github/workflows-disabled/`. Do not move them or re-enable repository Actions
unless the owner explicitly changes this policy.

This repository is standalone. It does not modify or depend on CodeBench or
`python-ios-lib`; those projects can consume a future release as an ordinary
package or copied source dependency when their platform integration is ready.
