# python-to-binary

`py2bin` is a dependency-free compiler and application builder written in pure
Python. It has three deliberately separate execution paths:

1. **Native compile:** Python AST → py2bin IR → machine-code bytes → ELF or
   Mach-O. This invokes no assembler or linker, and the generated executable
   needs no Python installation at runtime.
2. **Compatible bundle:** full CPython projects become a `.pyz`, executable
   `.bin`, directory bundle, or macOS `.app`, including installed package data
   and native extensions used by Manim, PyTorch, Transformers, NumPy, or `bpy`.
3. **Portable-C frontend:** a useful typed subset of Python becomes readable C
   source, or a checksummed `.py2cbin` C-source container. Imports automatically
   plan for the compatible CPython bundle instead of pretending native packages
   can be translated from Python source.

## What “pure Python” means

The **compiler and builder** use only Python's standard library. They do not require a C/C++
compiler, Rust, PyInstaller, Nuitka, Docker, or a native bootloader.

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

| Format | Output | Native extensions | Python required on target |
|---|---|---:|---:|
| `bin` | Executable self-extracting zip application | Yes, after extraction | Yes |
| `pyz` | Python zip application | Yes, after extraction | Yes |
| `dir` | App + dependencies + launcher | Yes | Yes |
| `app` | macOS application bundle | Yes | Yes |

Native compile targets currently implemented are `linux-x86_64` (ELF),
`linux-arm64` (ELF), `darwin-x86_64` and `darwin-arm64` (Mach-O), and
`windows-x86_64` and `windows-arm64` (PE `.exe`).
Run `py2bin targets` to list them. The first native
frontend milestone supports module constants, constant arithmetic and
f-strings, `print()`, and integer exit status. It rejects everything else with
a source location rather than producing a subtly incorrect executable.

The bundle-format `bin` uses Python; `py2bin compile` produces actual machine
code. These writers encode executable headers, import tables, system calls, and
instructions directly rather than shelling out to an assembler.

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

Native compilation of those libraries is a different problem: Torch contains
millions of lines of precompiled C++/CUDA code, `bpy` is coupled to Blender, and
Manim invokes external tools. They can be made self-contained per target by
shipping their native components and a compatible embedded runtime, but they
cannot truthfully become one CPU-independent executable. The long-term native
API is an adapter ABI: pure-Python modules compile through py2bin IR, while
large native libraries link as target-specific prebuilt components.
Until that adapter ABI is complete, `freeze` is the full-compatibility engine.

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

The implementation intentionally depends only on the standard library, and the
tests never download packages or write outside their temporary directory.

This repository is standalone. It does not modify or depend on CodeBench or
`python-ios-lib`; those projects can consume a future release as an ordinary
package or copied source dependency when their platform integration is ready.
