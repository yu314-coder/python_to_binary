# Compiler architecture

## Product contract

The end goal is a pure-Python toolchain that writes target executables without
calling an assembler or linker. A program from the direct-native path must not
require Python on its target computer. The compiler never describes an
ABI-specific binary as “universal”: each artifact declares its OS and CPU
target.

One path is deliberately exempt from the no-Python-on-target rule and says so
in its own output: the CPython C-API tier links an already-compiled interpreter
rather than replacing it. Its binary contains py2bin-encoded instructions for
the program's own logic and an `LC_LOAD_DYLIB` naming the CPython shared
library, so it needs that interpreter present. It still calls no assembler,
linker, or C compiler at build time, which is the invariant that actually
matters here.

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

A fourth path closes that loop: `c_frontend.py` is py2bin's own C compiler. It
lexes and parses C into a C syntax tree, applies C's type rules, and lowers the
result straight to native IR. It deliberately does not reuse Python's `ast`:
C and Python are different languages, and the earlier bridge that reused the
Python tree miscompiled a C `for` by giving it `range` semantics. Nothing in a
Python tree can express a narrow integer type, the address of a local, or
`goto`, so C gets its own front end and its own lowering.

`c_native.py` keeps the older canonical-C bridge, which round-trips the C that
py2bin's *own* generator emits back into a Python AST. That path exists to
prove `emit-c` output means what the Python meant, not to compile C in general.

Because py2bin both generates and parses C, it never reads `Python.h` or any
other system header. Every `PyObject *` is an opaque 64-bit handle: an
incomplete type the compiler refuses to dereference or offset. That is what
removes the need for a preprocessor, macros, and struct layouts, and it is what
makes a handwritten C compiler tractable.

## Pipeline

```text
Python source                       C source
    |                                     |
    |                        c_frontend.py lexer/parser
    |                                     |
    |                            C syntax tree + C types
    |                                     |
    |                        C-specific lowering (no Python AST)
    |                                     |
Python AST                                |
    |                                     |
    +-- AST validation, planning, and type/constant discovery
    |      |                              |
    |      +-- portable C source / .py2cbin
    |      +-- CPython freeze plan for imports or unsupported semantics
    |                                     |
    +-- py2bin portable IR <--------------+
    |      Int*/Float*/Heap*/SlotAddress/ExternCall/CStringConstant/Write/
    |      WriteRuntime/Store/FloatStore/Label/Jump/JumpIfFalse/Exit/ExitValue
    |
    +-- target-independent optimizer
    |
    +-- target instruction encoder
    |      x86-64                 arm64 (+ dyld extern binding)
    |
    +-- executable image writer
           ELF       Mach-O + ad-hoc signature       PE32+
```

An `ExternCall` is the single node behind both the libc adapter ABI and the
CPython C-API tier. On `darwin-arm64` the Mach-O writer emits one
`LC_LOAD_DYLIB` per referenced library with the correct two-level namespace
ordinals, and `cabi.symbol_library` decides which library owns each symbol.
Every other target rejects the node rather than emitting an unbound call.

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
