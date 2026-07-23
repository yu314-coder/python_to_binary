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
   are collected with an embedded CPython runtime. This is compatibility
   packaging, not native translation of the application.
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

# A native Apple Silicon application bundle:
PYTHONPATH=src python3 -m py2bin compile examples/native_hello.py \
  --target darwin-arm64 --app --output dist/NativeHello --clean

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
| `build --format bin` | Self-extracting Python application | CPython bytecode/source | Yes |
| `build --format dir` | Project, packages, and launcher | CPython bytecode/source | Yes |
| `freeze` | Embedded-runtime directory or macOS `.app` | CPython bytecode/source | No |

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

Native compile targets currently implemented are `linux-x86_64` (ELF),
`linux-arm64` (ELF), `darwin-x86_64` and `darwin-arm64` (Mach-O), and
`windows-x86_64` and `windows-arm64` (PE `.exe`).
Run `py2bin targets` to list them. The first native
frontend milestone supports module constants, constant arithmetic and
comparisons, constant Boolean expressions and branches, f-strings, `print()`,
and integer exit status. It rejects everything else with a source location
rather than producing a subtly incorrect executable.

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

For arbitrary CPython packages, freeze the interpreter and complete package
trees into a target-side bundle:

```sh
PYTHONPATH=src python3 -m py2bin freeze app/main.py \
  --source-root app --output dist/MyApp \
  --include torch --include transformers --clean

./dist/MyApp/MyApp.bin
```

`freeze` carries the current compatible CPython runtime, standard library,
native extension modules, distribution metadata, and package data. It can also
consume wheels directly without pip or installation:

```sh
py2bin freeze app.py -o dist/App --wheel wheels/custom_backend.whl
```

Frozen bundles are specific to the build runtime's OS, CPU, Python ABI, and
accelerator variant. On Unix they contain a `.bin` launcher; on Windows they
contain a copied `.exe` configured by an isolated `._pth` file. Build each
target from a matching runtime or, in a future release, an explicit runtime
pack. Dynamic imports still need `--include`.

On macOS, `freeze --app` wraps that embedded runtime in a launchable application
bundle. `--icon` accepts ICNS, a square PNG at a standard icon size, or a
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

`--compact` omits distribution tests, CPython build support, GUI/demo modules
that are not used by a pywebview app, and generated bytecode caches. Leave it
off when the packaged program imports package test suites, `tkinter`,
`unittest`, `lib2to3`, or CPython build configuration files at runtime.

## Heavy-library compatibility

- **PyTorch:** bundle on the same OS, architecture, Python ABI, and accelerator
  family as the destination. GPU drivers remain a target-system prerequisite.
- **Transformers:** Python code is bundled; downloaded model weights must be
  placed inside the source tree or made available in a target cache. Test true
  offline mode before distribution.
- **Manim:** Python packages and data can be bundled, while programs such as
  ffmpeg and LaTeX plus required fonts remain external unless you ship them in
  your project and configure Manim to use them.
- **bpy:** build from Blender's matching Python or a compatible `bpy` wheel.
  Blender's resources and licensing/distribution requirements are separate.
- **Any other library:** static imports work automatically; use `--include` for
  plugins, entry-point-loaded modules, optional backends, or runtime imports.

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

This repository is standalone. It does not modify or depend on CodeBench or
`python-ios-lib`; those projects can consume a future release as an ordinary
package or copied source dependency when their platform integration is ready.
