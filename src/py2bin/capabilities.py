"""Truthful native-compilation and compatible-bundle capability reporting.

The catalog in this module is descriptive metadata.  It never imports or
executes an application or a third-party package, so it is safe to use while
deciding whether a build is possible.
"""

from __future__ import annotations
import tokenize

import ast
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

def _read_source(path):
    """A source file's text, decoded the way Python decodes one.

    A file may open with a byte-order mark and may say what it is in a
    `# -*- coding: ... -*-` line; Python honours both, so reading everything
    as UTF-8 refused a Latin-1 file with a codec error naming a byte offset.
    """

    with tokenize.open(path) as stream:
        return stream.read()


from .native.frontend import NativeCompileError, lower


@dataclass(frozen=True, slots=True)
class LibraryCapability:
    """How one import can be handled by the current py2bin backends."""

    module: str
    project: str
    native_aot: str
    compatible_bundle: str
    payload: str
    requirement: str


@dataclass(frozen=True, slots=True)
class EntryCapability:
    """Native-subset result and import-level explanations for one entry file."""

    entry: Path
    native_compile: bool
    native_reason: str
    imports: tuple[str, ...]
    libraries: tuple[LibraryCapability, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "entry": str(self.entry),
            "native_compile": self.native_compile,
            "native_reason": self.native_reason,
            "imports": list(self.imports),
            "libraries": [dataclasses.asdict(item) for item in self.libraries],
        }


def _capability(
    module: str,
    project: str,
    payload: str,
    requirement: str,
    *,
    compatible_bundle: str = "conditional",
) -> LibraryCapability:
    return LibraryCapability(
        module=module,
        project=project,
        native_aot="no",
        compatible_bundle=compatible_bundle,
        payload=payload,
        requirement=requirement,
    )


# This is intentionally a claim matrix, not a list of packaging hooks.  "No"
# means py2bin does not translate that project's Python implementation and
# CPython bindings into independent machine code.  Some projects already ship
# native machine-code components; freeze preserves those target-specific files.
_COMMON_LIBRARIES = (
    _capability(
        "numpy",
        "NumPy",
        "Python plus C/C++ extension modules and native shared libraries",
        "matching CPython ABI and an OS/architecture-compatible wheel",
    ),
    _capability(
        "scipy",
        "SciPy",
        "Python plus compiled C/C++/Fortran extension modules",
        "matching CPython ABI, target wheel, and its native dependency closure",
    ),
    _capability(
        "pandas",
        "pandas",
        "Python plus C/Cython-generated extension modules",
        "matching CPython ABI and target wheels for pandas, NumPy, and dependencies",
    ),
    _capability(
        "sklearn",
        "scikit-learn",
        "Python plus compiled extension modules",
        "matching target wheels for scikit-learn, NumPy, SciPy, and dependencies",
    ),
    _capability(
        "torch",
        "PyTorch",
        "Python bindings plus large precompiled C++/CPU/GPU libraries",
        "target PyTorch wheel, CPython ABI, native libraries, and optional GPU runtime",
    ),
    _capability(
        "torchvision",
        "TorchVision",
        "Python plus compiled image/video operators",
        "a Torch-compatible target wheel and matching native libraries",
    ),
    _capability(
        "tensorflow",
        "TensorFlow",
        "Python bindings plus a precompiled C++ runtime",
        "a supported target wheel, CPython ABI, and target native runtime",
    ),
    _capability(
        "jax",
        "JAX",
        "Python frontend plus jaxlib/XLA native runtime",
        "compatible jax and jaxlib target wheels",
    ),
    _capability(
        "transformers",
        "Transformers",
        "dynamic Python model orchestration; model execution uses optional native backends",
        "CPython, complete dependency wheels, model/config/tokenizer files, and a backend",
    ),
    _capability(
        "tokenizers",
        "Hugging Face Tokenizers",
        "Rust machine code exposed through a CPython extension",
        "matching CPython ABI and target wheel",
    ),
    _capability(
        "huggingface_hub",
        "huggingface_hub",
        "dynamic Python networking, cache, and model-download logic",
        "CPython, dependency wheels, certificates, and separately supplied model data",
    ),
    _capability(
        "manim",
        "Manim Community",
        "dynamic Python plus compiled graphics/media dependencies",
        "CPython, complete target wheels, fonts/assets, and external media tools such as FFmpeg",
    ),
    _capability(
        "matplotlib",
        "Matplotlib",
        "Python plus compiled rendering extensions and package data",
        "matching target wheels, NumPy, backend resources, fonts, and data files",
    ),
    _capability(
        "PIL",
        "Pillow",
        "Python plus compiled image-codec extensions",
        "matching Pillow target wheel and its included or system codec libraries",
    ),
    _capability(
        "cv2",
        "OpenCV Python",
        "precompiled OpenCV C++ libraries exposed through a CPython extension",
        "matching OpenCV target wheel and GUI/media native dependencies",
    ),
    _capability(
        "bpy",
        "Blender Python API",
        "Blender C/C++ machine code exposed to a tightly matched Python ABI",
        "matching bpy/Blender build, Blender resources, CPython ABI, OS, and architecture",
    ),
    _capability(
        "webview",
        "pywebview",
        "Python control layer plus an OS-specific native webview backend",
        "CPython, pywebview dependencies, and the target OS webview framework/runtime",
    ),
    _capability(
        "gradio",
        "Gradio",
        "Python web server plus packaged HTML/CSS/JavaScript frontend assets",
        "CPython, server dependency wheels, frontend assets, and a browser/webview",
    ),
    _capability(
        "streamlit",
        "Streamlit",
        "Python server/runtime plus packaged web frontend assets",
        "CPython, server dependency wheels, frontend assets, and a browser/webview",
    ),
    _capability(
        "numba",
        "Numba",
        "dynamic Python compiler frontend plus llvmlite/LLVM native components",
        "CPython and mutually compatible Numba, NumPy, and llvmlite target wheels",
    ),
    _capability(
        "llvmlite",
        "llvmlite",
        "compiled LLVM bindings and native libraries",
        "matching CPython ABI and target wheel",
    ),
    _capability(
        "onnxruntime",
        "ONNX Runtime",
        "Python bindings plus a precompiled C/C++ inference runtime",
        "matching target wheel and CPU/GPU execution-provider native libraries",
    ),
    _capability(
        "diffusers",
        "Diffusers",
        "dynamic Python model pipelines using Torch/JAX and other backends",
        "CPython, backend and dependency wheels, model/config/tokenizer files",
    ),
    _capability(
        "accelerate",
        "Accelerate",
        "dynamic Python device/distributed-execution orchestration",
        "CPython, target backend wheels, and target accelerator/distributed runtime",
    ),
    _capability(
        "safetensors",
        "Safetensors",
        "Rust machine code exposed through Python bindings",
        "matching target wheel plus separately supplied model tensor files",
    ),
    _capability(
        "pyarrow",
        "Apache Arrow Python",
        "Python bindings plus large precompiled Arrow C++ libraries",
        "matching CPython ABI and target wheel/native library closure",
    ),
    _capability(
        "polars",
        "Polars",
        "Rust machine code exposed through Python bindings",
        "matching CPython ABI and target wheel",
    ),
    _capability(
        "OpenGL",
        "PyOpenGL",
        "dynamic Python/ctypes bindings to the target OpenGL implementation",
        "CPython, PyOpenGL packages, target graphics framework, and compatible driver",
    ),
    _capability(
        "pygame",
        "pygame",
        "Python API plus compiled SDL-based extension modules",
        "matching target wheel, bundled SDL libraries, and target graphics/audio services",
    ),
    _capability(
        "psutil",
        "psutil",
        "Python API plus OS-specific compiled process/system extensions",
        "matching CPython ABI and target wheel",
    ),
    _capability(
        "winpty",
        "pywinpty",
        "Python bindings plus Windows native terminal libraries",
        "matching Windows CPython ABI/architecture and native DLLs",
    ),
    _capability(
        "cryptography",
        "cryptography",
        "Python API plus Rust/OpenSSL native extension components",
        "matching CPython ABI and target wheel/native libraries",
    ),
    _capability(
        "lxml",
        "lxml",
        "compiled XML extension modules using libxml2/libxslt",
        "matching CPython ABI and target wheel/native libraries",
    ),
    _capability(
        "pydantic_core",
        "pydantic-core",
        "Rust machine code exposed through a CPython extension",
        "matching CPython ABI and target wheel",
    ),
    _capability(
        "sqlalchemy",
        "SQLAlchemy",
        "dynamic Python database toolkit with optional compiled accelerators",
        "CPython, dependency packages, and the selected database driver/client libraries",
    ),
    _capability(
        "requests",
        "Requests",
        "dynamic pure-Python HTTP client",
        "CPython, dependency packages, and certificate data",
    ),
    _capability(
        "flask",
        "Flask",
        "dynamic pure-Python web framework and templates",
        "CPython, dependency packages, templates, and static assets",
    ),
    _capability(
        "django",
        "Django",
        "dynamic pure-Python web framework, templates, and application metadata",
        "CPython, dependency packages, settings, templates, migrations, and static assets",
    ),
    _capability(
        "fastapi",
        "FastAPI",
        "dynamic Python ASGI application and type-driven routing",
        "CPython, ASGI server, dependency packages, templates, and static assets",
    ),
)

_BY_MODULE = {item.module: item for item in _COMMON_LIBRARIES}


def common_libraries() -> tuple[LibraryCapability, ...]:
    """Return the stable common-library claim matrix."""

    return _COMMON_LIBRARIES


def library_capability(module: str) -> LibraryCapability:
    """Return a known or conservative generic capability for an import."""

    root = module.partition(".")[0]
    if root == "sys":
        return LibraryCapability(
            module="sys",
            project="Python standard library",
            native_aot="restricted",
            compatible_bundle="yes",
            payload="no CPython payload for the native sys.exit-only special case",
            requirement="native compile accepts only import sys followed by sys.exit(native integer expression)",
        )
    known = _BY_MODULE.get(root)
    if known is not None:
        return known
    if root in getattr(sys, "stdlib_module_names", set()):
        return _capability(
            root,
            "Python standard library",
            "CPython standard-library implementation",
            "embedded CPython runtime containing the requested standard-library module",
            compatible_bundle="yes",
        )
    return _capability(
        root,
        "unknown/local third-party project",
        "not classified; py2bin does not assume an import is independently native",
        "the implementation plus every target-compatible dependency and data file",
    )


def _imports(tree: ast.AST) -> tuple[str, ...]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                result.add("<relative>")
            elif node.module:
                result.add(node.module.partition(".")[0])
    return tuple(sorted(result, key=str.lower))


def _local_source_capability(entry: Path, module: str) -> LibraryCapability | None:
    if module == "<relative>":
        return LibraryCapability(
            module=module,
            project="relative local pure-Python source",
            native_aot="restricted",
            compatible_bundle="yes",
            payload="supported pure integer functions are inlined into native IR",
            requirement=(
                "the resolved relative module must remain inside the source root "
                "and satisfy the restricted native function rules"
            ),
        )
    parts = module.split(".")
    candidates = (
        entry.parent.joinpath(*parts).with_suffix(".py"),
        entry.parent.joinpath(*parts, "__init__.py"),
    )
    if not any(candidate.is_file() for candidate in candidates):
        return None
    return LibraryCapability(
        module=module,
        project="local pure-Python source",
        native_aot="restricted",
        compatible_bundle="yes",
        payload="supported pure integer functions are inlined into native IR",
        requirement=(
            "imported functions must use positional parameters with optional "
            "static integer defaults, supported integer assignments/control "
            "flow, and value returns on every path; executable module top-level "
            "code is rejected"
        ),
    )


def assess_entry(
    entry: Path,
    *,
    experimental_kernels: bool = False,
) -> EntryCapability:
    """Inspect an entry file without executing it or importing its packages.

    ``experimental_kernels`` is accepted for call-signature stability but is
    inert: py2bin no longer reimplements a NumPy/Torch integer subset, because
    such a reimplementation does not match the real packages' runtime object
    semantics. NumPy/Torch imports are reported as unsupported.
    """

    entry = entry.expanduser().resolve()
    if not entry.is_file():
        raise FileNotFoundError(f"entry script does not exist: {entry}")
    source = _read_source(entry)
    tree = ast.parse(source, filename=str(entry))
    imports = _imports(tree)
    try:
        lower(
            entry,
            source,
            (entry.parent,),
        )
    except (NativeCompileError, ValueError) as error:
        native_compile = False
        native_reason = str(error)
    else:
        native_compile = True
        native_reason = (
            "accepted by the current static native subset; no CPython runtime "
            "or third-party library payload is used; supported pure-Python "
            "functions are inlined into native IR"
        )
    return EntryCapability(
        entry=entry,
        native_compile=native_compile,
        native_reason=native_reason,
        imports=imports,
        libraries=tuple(
            _local_source_capability(entry, module)
            or library_capability(module)
            for module in imports
        ),
    )


def format_catalog(libraries: tuple[LibraryCapability, ...] | None = None) -> str:
    """Format the claim matrix as a readable fixed-width table."""

    libraries = libraries or common_libraries()
    rows = [("import", "native AOT", "freeze", "project")]
    rows.extend(
        (
            item.module,
            item.native_aot,
            item.compatible_bundle,
            item.project,
        )
        for item in libraries
    )
    widths = [max(len(row[index]) for row in rows) for index in range(4)]
    return "\n".join(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()
        for row in rows
    )
