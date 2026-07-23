# python-to-binary detailed guide

`python-to-binary` (`py2bin`) is a standard-library-only toolchain with three
separate jobs:

1. Translate a supported Python subset into portable C source.
2. Translate a smaller static Python subset directly into ELF, PE, or Mach-O
   machine-code files without an assembler or linker.
3. Bundle full CPython applications, package data, native extensions, and an
   optional embedded interpreter when source-level native compilation is not
   compatible with the program.

These modes are intentionally distinct. A dynamic application importing Manim,
PyTorch, Transformers, Blender `bpy`, or pywebview cannot truthfully be
converted into one architecture-independent native executable merely by
rewriting its Python files.

## Installation

From PyPI:

```sh
python3 -m pip install python-to-binary
```

Directly from GitHub:

```sh
python3 -m pip install \
  "git+https://github.com/yu314-coder/python_to_binary.git"
```

From a clone without installing:

```sh
git clone https://github.com/yu314-coder/python_to_binary.git
cd python_to_binary
PYTHONPATH=src python3 -m py2bin --help
```

Python 3.10 or newer is required on the build host. Native compilation itself
does not invoke a C compiler, assembler, linker, target SDK, Docker, or target
Python runtime.

## Choosing an output strategy

| Program | Recommended command | Target Python required |
|---|---|---:|
| Static constants, printing, integer exit | `compile` | No |
| Supported typed Python subset | `emit-c` | C source only |
| Ordinary pure-Python project | `build` or `freeze` | `build`: yes; `freeze`: no |
| pywebview/Manim/Torch/Transformers/bpy | `freeze` | No, but native libraries remain target-specific |

Use `plan-c` before assuming C translation is safe:

```sh
py2bin plan-c app.py
```

It returns `c-source` for the implemented C subset and `cpython-bundle` when
imports or unsupported Python semantics require compatibility mode.

## Native target selection

The implemented target matrix is:

| OS selector | Architecture selector | Canonical target | File format |
|---|---|---|---|
| `linux` | `x86_64`, `x64`, `amd64` | `linux-x86_64` | ELF64 |
| `linux` | `arm64`, `aarch64` | `linux-arm64` | ELF64 |
| `windows` | `x86_64`, `x64`, `amd64` | `windows-x86_64` | PE32+ |
| `windows` | `arm64`, `aarch64` | `windows-arm64` | PE32+ |
| `macos`, `mac`, `darwin`, `osx` | `x86_64`, `x64`, `amd64` | `darwin-x86_64` | Mach-O |
| `macos`, `mac`, `darwin`, `osx` | `arm64`, `aarch64` | `darwin-arm64` | Mach-O |

Examples:

```sh
py2bin compile hello.py -o dist/hello-linux \
  --os linux --arch x86_64 --clean

py2bin compile hello.py -o dist/hello-arm64.exe \
  --os windows --arch arm64 --clean

py2bin compile hello.py -o dist/hello-mac \
  --os macos --arch arm64 --clean
```

The equivalent exact form is:

```sh
py2bin compile hello.py -o dist/hello.exe \
  --target windows-x86_64 --clean
```

Generate all six native formats:

```sh
py2bin compile-all hello.py -o dist/all --clean
```

Generate one selected format through `compile-all`:

```sh
py2bin compile-all hello.py -o dist/windows-arm \
  --os windows --arch arm64 --clean
```

Native files are target-specific. An ARM64 PE file does not run on Linux ARM64,
and an x86-64 Mach-O does not run on Windows x86-64.

## Native Python subset

The current direct-machine-code frontend supports:

- module docstrings;
- constant assignments;
- constant arithmetic and basic f-strings;
- `print(...)`;
- `SystemExit(...)` and `sys.exit(...)`.

Unsupported dynamic statements fail with a file, line, and column. The compiler
does not silently produce a program with different semantics.

```python
name = "native"
answer = 6 * 7
print(f"{name}: {answer}")
raise SystemExit(0)
```

## Python-to-C

```sh
py2bin emit-c program.py -o dist/program.c --clean
py2bin emit-c program.py -o dist/program.py2cbin --container --clean
```

The C frontend supports variables, numeric operations, comparisons, Boolean
expressions, branches, loops, `range`, functions, returns, printing, and simple
f-strings. A `.py2cbin` is a versioned and checksummed C-source container, not
an executable.

Turning emitted C into machine code normally requires an explicitly chosen C
compiler. `py2bin compile` is the no-compiler path for its smaller native
subset.

## Full application freezing

Freeze an application and its current compatible CPython runtime:

```sh
py2bin freeze app.py --source-root . -o dist/App \
  --include webview --compact --clean
```

On macOS, create a native `.app` entrypoint and convert an ICO to ICNS:

```sh
py2bin freeze app.py --source-root . -o dist/App \
  --app --name App --icon icon.ico \
  --include webview --compact --clean
```

The macOS `Contents/MacOS/App` entry is real hardwritten Mach-O machine code.
It forwards command-line arguments and starts the embedded runtime. py2bin
also writes the ad-hoc signature, `Info.plist`, ICNS, and resource seal in
Python. The application logic still runs on embedded CPython for compatibility.

`--compact` removes distribution test suites, bytecode caches, and CPython
build/demo modules. Do not use it when the application intentionally imports
`unittest`, `tkinter`, `lib2to3`, package tests, or CPython build configuration
files.

## Dependencies and native libraries

Static imports are discovered without executing the application.

```sh
py2bin analyze app.py --source-root .
py2bin freeze app.py --source-root . -o dist/App \
  --include dynamically_loaded_plugin
```

Dependency modes:

- `closure`: direct packages plus installed dependency closure;
- `imported`: direct distributions only;
- `none`: project files only.

Native extensions must match the destination OS, CPU, and Python ABI. Build a
Windows bundle from a Windows runtime, a Linux bundle from Linux, and a macOS
bundle from macOS. Cross-compilation currently applies to the narrow native
compiler, not arbitrary CPython extension bundles.

Framework-specific requirements remain external when the framework itself
requires them:

- Manim may require ffmpeg, LaTeX, fonts, and platform libraries.
- Torch GPU builds require compatible drivers and accelerator libraries.
- Transformers model weights must be bundled or available in an offline cache.
- `bpy` requires a compatible Blender/Python build and Blender resources.
- pywebview requires the target platform's webview backend.

## Python API

```python
from pathlib import Path
from py2bin.native import compile_native, resolve_target

target = resolve_target("windows", "arm64")
result = compile_native(
    Path("hello.py"),
    Path("dist/hello.exe"),
    target=target,
    clean=True,
)
print(result.artifact, result.target, result.bytes)
```

Portable C:

```python
from py2bin import compile_to_c, plan_c

source = "print('hello')\n"
print(plan_c(source))
print(compile_to_c(source))
```

## ManimStudio reference test

The repository has been validated against `yu314-coder/manim_app` as a large
reference application:

- application source and local modules copied;
- Monaco, HTML, JavaScript, prompts, fonts, and other resources preserved;
- pywebview and PyObjC dependency closure bundled;
- ICO converted to multi-resolution ICNS;
- native ARM64 Mach-O launcher generated;
- command-line arguments preserved;
- embedded CPython runtime loaded instead of the build-host runtime;
- strict macOS signature and resource verification passed.

The reference source is Windows-first and calls `cmd.exe` and `where`, so those
features remain Windows-specific. py2bin does not rewrite application behavior
to conceal platform assumptions.

## Validation

Run the complete suite:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Inspect targets:

```sh
py2bin targets
```

For generated files, test on the actual destination operating system and CPU.
Wine is useful for some PE checks but is not equivalent to real Windows
certification.

## Publishing and release integrity

Build wheel and source distribution:

```sh
python3 -m build
python3 -m twine check dist/*
```

The project includes GitHub Actions for tests and PyPI trusted publishing.
Publishing to PyPI requires either a configured PyPI trusted publisher or a
valid API token. Never commit tokens or `.pypirc`.

Before a release:

1. Run all tests.
2. Verify all six generated headers.
3. Build and inspect both wheel and source distribution.
4. Install the wheel into an empty virtual environment.
5. Run `py2bin targets` and compile a smoke-test source.
6. Tag the exact commit intended for release.

## Current limitations

- Direct native compilation implements a deliberately narrow Python subset.
- Full-library freeze outputs are specific to the build runtime's platform and
  ABI.
- Windows and Linux frozen-runtime packs are not yet cross-produced from
  macOS.
- The macOS frozen-app native launcher is currently implemented for ARM64;
  x86-64 remains available for narrow native Mach-O output.
- Driver, license, model, font, ffmpeg, LaTeX, and system-service requirements
  cannot be removed by changing executable formats.
