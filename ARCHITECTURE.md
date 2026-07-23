# Compiler architecture

## Product contract

The end goal is a pure-Python toolchain that writes target executables without
calling an assembler or linker. A generated native program must not require
Python on its target computer. The compiler never describes an ABI-specific
binary as “universal”: each artifact declares its OS and CPU target.

The build-host contract is equally strict: Python plus py2bin is sufficient.
Cross-compilation never probes for or executes a target assembler, linker,
compiler, SDK, virtual environment, or emulator. Library code and binary assets
are inputs, not assumed features of the host environment.

The repository also has a compatible bundler. That path exists because broad
CPython compatibility and native compilation are different guarantees. It
provides immediate execution for dynamic programs while the native object
model, runtime, standard library, and extension ABI grow.

The `freeze` engine is the compatibility implementation: it copies a matching
CPython runtime and every declared file in selected distributions, and it can
expand supplied wheels without pip. Native extensions retain the ABI they were
built for instead of being incorrectly treated as Python source.

The portable-C frontend is a third output path. It translates a validated,
typed Python subset to C source and can wrap that source in a deterministic,
checksummed `.py2cbin` container. The container is an interchange artifact, not
machine code. This keeps C generation usable on hosts with only Python while
leaving final C compilation to an explicitly supplied platform toolchain.

## Pipeline

```text
Python source
    |
    +-- AST validation, planning, and type/constant discovery
    |      |
    |      +-- portable C source / .py2cbin
    |      +-- CPython freeze plan for imports or unsupported semantics
    |
    +-- py2bin portable IR (currently Write and Exit)
    |
    +-- target instruction encoder
    |      x86-64                 arm64
    |
    +-- executable image writer
           ELF       Mach-O + ad-hoc signature       PE32+
```

Every layer consumes structured objects and returns text or bytes. No layer
emits assembly text or invokes a platform toolchain. The C backend emits source
only; the native backends hardwrite executable machine code and container
headers.

Frozen macOS apps also use a hardwritten Mach-O entrypoint. That launcher
forwards arguments, prepares paths for the relocated runtime, and starts the
embedded interpreter. Its ad-hoc signature and the app resource seal are
generated directly from Python. Compact freezing prunes package tests and
CPython build-only files without specializing application semantics for the
reference workload.

## Compatibility strategy

Full Python and “all libraries” require four coordinated tracks:

1. **Language/runtime:** tagged objects, reference counting or tracing GC,
   exceptions, functions and closures, classes, iterators, generators, async,
   descriptors, and Python-compatible containers.
2. **Imports:** frozen pure-Python modules compiled into the image, package
   resources, metadata, dynamic-import declarations, and deterministic module
   initialization.
3. **Native ABI:** a stable py2bin extension ABI plus a CPython-compatibility
   bridge. Existing wheels are OS/CPU/Python-ABI-specific and must be collected
   per target; they cannot be translated as Python source.
4. **Library adapters:** NumPy/Torch tensor operations lower to native library
   calls; Transformers packages model assets; Manim declares ffmpeg/LaTeX/font
   resources; `bpy` binds to a compatible Blender runtime.

The executable can be self-contained for one declared target. GPU drivers,
kernel facilities, hardware instruction sets, licenses, and external services
cannot be made portable merely by changing the container format.

## Roadmap

- M0 (implemented): direct ELF/Mach-O/PE writers, x86-64 and arm64 instruction
  encoding, macOS ad-hoc signing, constant lowering, native output and exit.
- M1 (partial): portable-C lowering for integer/float arithmetic, comparisons,
  branches, loops, functions, local variables, printing, and basic strings.
  Native SSA-like IR and equivalent runtime operations remain in progress.
- M2: object ABI, allocator, exceptions, lists/dicts/tuples, modules, frozen
  imports, and a native standard-library core.
- M3: FFI and extension manifests, static/dynamic library import writers,
  resources, and cross-target dependency resolution.
- M4: CPython extension bridge and specialized adapters for NumPy, Torch,
  Transformers, Manim, and Blender.
- M5: optimization passes, debug information, reproducible builds, universal
  macOS containers, signing/notarization integration, and target test farms.

## PPCI comparison

PPCI is a broad compiler infrastructure with many languages and architectures;
its own Python program wrapper describes Python-to-IR as “very preliminary.”
This project borrows the proven separation of frontend, IR, code generation,
and file format, but focuses its roadmap on Python semantics, CPython package
compatibility, self-contained applications, and framework-specific adapters.
No PPCI source is vendored or required at runtime.
