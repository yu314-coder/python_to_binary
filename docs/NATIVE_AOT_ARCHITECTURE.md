# CPython-free whole-application AOT architecture

This document defines the only mode in py2bin that may be described as
“everything is machine code.” It is a compiler design and acceptance contract,
not a packaging trick.

## Non-negotiable artifact contract

`py2bin aot-build` must either:

1. lower every reachable Python operation to py2bin IR;
2. emit target instructions or call an explicitly declared CPython-free native
   adapter;
3. keep non-code resources such as HTML, CSS, JavaScript, fonts, models, and
   media as data;
4. write the final target artifact without launching an external assembler,
   compiler, linker, SDK, Wine, Rosetta, or target Python; and
5. attest that the result contains no Python source, Python bytecode, CPython
   library, self-extracting runtime payload, or compatibility fallback.

Otherwise the command fails before it writes the artifact.

This contract does **not** apply to `freeze`, `bundle`, compatible `assemble`,
or the lightweight `build` formats. Those commands remain useful packaging
tools, but they execute Python and must not be called whole-program native AOT.

```sh
# Inspect the complete reachable source set without importing the app.
py2bin aot-plan app/main.py --source-root app --json --strict

# Build only if that plan has no blockers, then write a proof record.
py2bin aot-build app/main.py --source-root app \
  --via-c --c-output dist/App.c \
  --target windows-x86_64 --output dist/App.exe \
  --attestation dist/App.aot.json --clean
```

The current command implements this fail-closed gate and the existing native
integer compiler. With `--via-c`, supported imported local functions are
lowered with the application, optimized and inlined, serialized as canonical
whole-program C, reparsed by py2bin, and required to reconstruct identical IR
before machine-code emission. It does not make unsupported Python or libraries
work.

## Why a new runtime is still required

“No CPython runtime” and “no runtime services” are different requirements.
Arbitrary Python requires services for objects, reference tracing or garbage
collection, dynamic types, strings, arbitrary-precision integers, dictionaries,
exceptions, frames, descriptors, classes, generators, coroutines, imports,
and platform I/O. These services can themselves be py2bin-authored machine
code; they do not have to be CPython and do not require `.py` files.

The intended pipeline is:

```text
closed Python source
        |
        v
semantic graph and typed/object IR
        |
        +---- static values ----> unboxed optimized machine operations
        |
        +---- dynamic values ---> py2bin-owned object-runtime calls
        |
        +---- libraries --------> declared CPython-free adapter ABI
        |
        v
handwritten PE / ELF / Mach-O writer
        |
        v
target-specific artifact plus non-code resources
```

An interpreter hidden inside the artifact is forbidden. A native object
runtime is not an interpreter: the application control flow is compiled, while
runtime helper functions implement dynamic language operations. This is the
same unavoidable distinction made by native compilers for other managed
languages.

`eval`, `exec`, runtime `compile`, dynamic `__import__`, and unconstrained
`importlib.import_module` break closed-world analysis. A strict build must
reject them. A future optional solution would be a py2bin-owned compiler
embedded as machine code, but that is a runtime compiler and must be labelled
as such; it may not fall back to CPython.

## Library routes

Third-party code falls into different categories. Treating every wheel as
Python source is incorrect: wheels frequently contain target-specific machine
code and bindings to CPython's object ABI.

| Library class | CPython-free route | What cannot be reused unchanged |
|---|---|---|
| Pure Python package | Analyze all reachable modules and lower their semantics to py2bin IR | Unsupported dynamic Python, import-time behavior that the compiler cannot preserve |
| CPython `.pyd` / `.so` extension | Port its binding layer to the py2bin adapter ABI, or rebuild it against HPy universal ABI after py2bin implements that ABI | `PyInit_*`, `PyObject*`, Python Stable/Limited ABI calls without a compatible object runtime |
| NumPy | (Future work) Implement py2bin ndarray/dtype/ufunc object semantics faithfully, including scalar return types; today the import is rejected | NumPy's CPython C-API table and `PyArrayObject*` bindings |
| PyTorch | Compile Python orchestration and bind directly to a target LibTorch/native engine API | The ordinary `torch` CPython extension and arbitrary Python extension ecosystem |
| Transformers | Compile model orchestration; use py2bin Torch/tokenizer adapters; keep model/config files as data | Dynamic plugin loading, unsupported model Python, CPython-only Torch bindings |
| Tokenizers | A target-specific C ABI adapter to its native engine, or a py2bin rewrite | Its Python extension ABI |
| Matplotlib / Manim | Implement the Python artist/scene semantics; call native render/font/image/video adapters; keep SVG/HTML/media as assets | CPython extension bindings and unported backend/plugin code |
| Blender `bpy` | A dedicated Blender-native embedding/API adapter or a clean process boundary | The normal `bpy` module, which exposes Blender through CPython |
| pywebview | Compile application logic and call a py2bin platform adapter for WKWebView, WebView2, or WebKitGTK | The CPython `webview` package and its Python platform backends |
| Requests | A py2bin sockets, TLS, URL, HTTP, certificate, and encoding stack | CPython standard-library networking modules until ported |
| Gradio / Streamlit | Compile server/state logic to native code and serve HTML/CSS/JS assets through native HTTP/WebSocket adapters | Their dynamic Python plugin/server ecosystem |
| pywinpty | A direct Windows ConPTY/native helper adapter | Its CPython extension wrapper |

Relevant upstream boundaries are explicit:

- CPython extension modules export initialization functions such as
  `PyInit_name` and operate through the Python/C API:
  <https://docs.python.org/3/c-api/extension-modules.html>
- CPython's Stable ABI remains a CPython ABI, not a runtime-independent native
  library contract:
  <https://docs.python.org/3/c-api/stable.html>
- NumPy's C API exposes `PyArrayObject*` and an imported function table:
  <https://numpy.org/doc/stable/reference/c-api/array.html>
- PyTorch publishes a C++ frontend and LibTorch distribution suitable for a
  non-Python adapter:
  <https://docs.pytorch.org/cppdocs/frontend>
- HPy defines a less CPython-specific extension API and is a possible future
  compatibility target:
  <https://docs.hpyproject.org/en/latest/overview.html>
- Apple exposes WKWebView directly as a native platform API:
  <https://developer.apple.com/documentation/webkit/wkwebview>

A native engine linked through an adapter is already machine code. Converting
its binary back to C and recompiling it would not make it “more native”; it
would usually lose correctness, licensing metadata, platform integration, and
hardware dispatch. If the policy is “only py2bin-authored implementation,”
then py2bin must independently reimplement that engine and must reject it until
the rewrite exists. That stricter policy cannot simultaneously promise full
Torch, Blender, NumPy, or browser-engine compatibility today.

## Adapter ABI requirements

Every future adapter must declare:

- supported target triples and CPU features;
- a versioned C-compatible calling convention that uses no `PyObject*`;
- ownership, lifetime, thread, error, and cancellation rules;
- exact native library files and non-code resources included;
- whether subprocesses, drivers, system frameworks, or network access are
  required;
- deterministic discovery rules with no import-time Python execution; and
- tests on the real target OS and CPU.

The final linker/loader layer must support relocations, symbols, imports,
thread-local storage, unwind information, dynamic libraries, resources, code
signing inputs, and platform GUI metadata. The current handcrafted writer
supports the much smaller needs of its present integer programs; general
native adapters require this layer to grow substantially.

## Implementation sequence

1. **Fail-closed whole-app plan and attestation — implemented.** Traverse local
   imports, reject dynamic code and unported imports, compile through the
   direct backend only, then scan/hash the artifact. The optional canonical-C
   route now round-trips the complete accepted IR—including inlined supported
   local-library functions—through a handwritten parser before emission.
2. **Core object runtime — not implemented.** Strings, bytes, lists, tuples,
   dictionaries, sets, large integers, floats, objects, exceptions, and memory
   management.
3. **Full language lowering — not implemented.** Classes, descriptors,
   closures, generators, async, pattern matching, dynamic calls, and Python's
   exact edge-case semantics.
4. **Native standard library — not implemented.** Filesystem, networking, TLS,
   compression, codecs, concurrency, subprocesses, and platform integration.
5. **Adapter ABI and linker — not implemented.** First-party ABI plus foreign
   object/static/dynamic library linking without CPython.
6. **Library ports — not implemented.** Build adapters one library and target
   at a time, backed by conformance tests.
7. **Whole-application optimization.** Reachability, specialization, unboxing,
   constant propagation, dead-code elimination, LTO-like IR optimization, and
   resource deduplication.

The order matters. Attempting library compatibility before object semantics,
the standard library, and a stable adapter ABI merely recreates a CPython
freezer under another name.

## Current truthful status

As of this source tree:

- supported static integer programs can be real CPython-free ELF, PE, or
  Mach-O machine-code artifacts;
- supported integer functions and `None`-returning procedures are inlined
  without Python frames; procedures support native control flow, bare returns,
  constant output, and acyclic procedure calls on every target;
- a runtime IEEE-754 binary64 `float` subset lowers to real SSE2/NEON on every
  target (native-run-verified on `darwin-arm64`);
- on POSIX targets (ELF/Mach-O) a runtime integer `list` and runtime ASCII
  `str` subset lowers onto an anonymous-`mmap` bump arena; the two Windows
  targets reject it (native-run-verified on `darwin-arm64`, built-only
  elsewhere);
- on `darwin-arm64` only, the `py2bin.cabi` adapter ABI binds vetted libc
  symbols through real dyld and is native-run-verified against CPython;
- real NumPy and Torch imports are **rejected** by `compile`, not reimplemented:
  an integer reimplementation would not match their runtime object semantics
  (a reduction is `np.int64` / a 0-d tensor, not a plain `int`);
- `aot-plan` and `aot-build` now make no-fallback behavior explicit and
  machine-checkable;
- `aot-build --via-c` performs a real Python → IR → canonical C → reparsed IR
  → handwritten target-binary pipeline and records it in the attestation;
- HTML/CSS/JavaScript are catalogued as external resources, not claimed as CPU
  instructions;
- arbitrary Python, Manim, real NumPy, real Torch, Transformers, `bpy`,
  pywebview, Gradio, Streamlit, and pywinpty are **not** yet complete
  CPython-free builds.

Therefore a current Manim application must fail `aot-build`. A working package
made by `freeze` is a CPython application bundle, not evidence that the AOT
compiler supports Manim.
