"""Strict whole-application AOT planning and attestation.

This module is the explicit opposite of a freezer.  It never selects the
embedded-CPython backend, never copies Python source or bytecode, and never
creates a self-extracting payload.  A build either lowers through the native
frontend or fails before an artifact is written.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ..analyzer import analyze
from .compiler import NativeResult, compile_native, compile_native_module
from .frontend import NativeCompileError, lower
from .ir_c import roundtrip_ir_c
from .optimizer import optimize


_WEB_SUFFIXES = frozenset({".css", ".htm", ".html", ".js", ".mjs", ".cjs", ".wasm"})
_NATIVE_SUFFIXES = frozenset({".a", ".dll", ".dylib", ".lib", ".pyd", ".so"})
_FOREIGN_SOURCE_SUFFIXES = frozenset(
    {".c", ".cc", ".cpp", ".cxx", ".cu", ".f", ".f90", ".h", ".hpp", ".pyx", ".pxd", ".rs"}
)
_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        "__pycache__",
        "build",
        "dist",
    }
)
_RUNTIME_MARKERS = (
    b"\nPY2BIN-ONEFILE-PAYLOAD-",
    b"py2bin_bootstrap.py",
    b"runpy.run_path(entry",
)


class AOTPlanError(ValueError):
    """Raised when a requested build is not completely CPython-free."""


@dataclass(frozen=True, slots=True)
class AOTApplicationPlan:
    entry: Path
    source_root: Path
    buildable: bool
    compiler_reason: str
    imports: tuple[str, ...]
    reachable_python: tuple[Path, ...]
    web_assets: tuple[Path, ...]
    native_payloads: tuple[Path, ...]
    foreign_sources: tuple[Path, ...]
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": 1,
            "entry": str(self.entry),
            "source_root": str(self.source_root),
            "buildable": self.buildable,
            "backend": "py2bin-native-aot" if self.buildable else None,
            "compiler_reason": self.compiler_reason,
            "imports": list(self.imports),
            "reachable_python": [str(path) for path in self.reachable_python],
            "web_assets": [str(path) for path in self.web_assets],
            "native_payloads": [str(path) for path in self.native_payloads],
            "foreign_sources": [str(path) for path in self.foreign_sources],
            "blockers": list(self.blockers),
            "guarantees": {
                "uses_cpython": False,
                "contains_python_source": False,
                "contains_python_bytecode": False,
                "uses_self_extraction": False,
                "uses_fallback_backend": False,
            },
        }


@dataclass(frozen=True, slots=True)
class AOTAttestation:
    artifact: Path
    target: str
    sha256: str
    bytes: int
    operations: int
    pipeline: str
    ir_roundtrip_verified: bool
    canonical_c_sha256: str | None
    cpython_free: bool
    python_payload_free: bool
    extraction_free: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": 1,
            "artifact": str(self.artifact),
            "target": self.target,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "operations": self.operations,
            "backend": "py2bin-native-aot",
            "pipeline": self.pipeline,
            "ir_roundtrip_verified": self.ir_roundtrip_verified,
            "canonical_c_sha256": self.canonical_c_sha256,
            "cpython_free": self.cpython_free,
            "python_payload_free": self.python_payload_free,
            "extraction_free": self.extraction_free,
        }


@dataclass(frozen=True, slots=True)
class AOTBuildResult:
    native: NativeResult
    plan: AOTApplicationPlan
    attestation: AOTAttestation
    attestation_path: Path | None
    c_source: str | None
    c_artifact: Path | None


def _included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return not any(
        part in _IGNORED_PARTS or part.startswith(".")
        for part in relative.parts
    )


def _local_candidate(root: Path, module: str) -> Path | None:
    parts = module.split(".")
    module_path = root.joinpath(*parts).with_suffix(".py")
    package_path = root.joinpath(*parts, "__init__.py")
    if module_path.is_file():
        return module_path
    if package_path.is_file():
        return package_path
    return None


def _dynamic_code_blockers(paths: tuple[Path, ...]) -> tuple[str, ...]:
    blockers: list[str] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            blockers.append(f"{path}: cannot parse closed-world source: {error}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name: str | None = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            ):
                name = f"importlib.{node.func.attr}"
            if name in {
                "__import__",
                "compile",
                "eval",
                "exec",
                "importlib.import_module",
            }:
                blockers.append(
                    f"{path}:{getattr(node, 'lineno', 1)}: dynamic code operation "
                    f"{name}() is incompatible with closed-world AOT"
                )
    return tuple(blockers)


def plan_aot_application(
    entry: Path,
    *,
    source_root: Path | None = None,
    experimental_kernels: bool = False,
) -> AOTApplicationPlan:
    """Return an exact no-fallback plan without importing application code."""

    entry = entry.expanduser().resolve()
    source_root = (source_root or entry.parent).expanduser().resolve()
    if not entry.is_file():
        raise FileNotFoundError(f"entry script does not exist: {entry}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"source root does not exist: {source_root}")
    if not entry.is_relative_to(source_root):
        raise ValueError("entry must be inside source_root")

    analysis = analyze(
        entry,
        source_root,
        dependency_mode="imported",
    )
    reachable_python = tuple(sorted(analysis.local_files, key=lambda path: path.as_posix()))
    files = tuple(
        sorted(
            (
                path
                for path in source_root.rglob("*")
                if path.is_file() and _included(path, source_root)
            ),
            key=lambda path: path.as_posix(),
        )
    )
    web_assets = tuple(path for path in files if path.suffix.lower() in _WEB_SUFFIXES)
    native_payloads = tuple(path for path in files if path.suffix.lower() in _NATIVE_SUFFIXES)
    foreign_sources = tuple(
        path for path in files if path.suffix.lower() in _FOREIGN_SOURCE_SUFFIXES
    )

    blockers = list(_dynamic_code_blockers(reachable_python))
    imports = tuple(sorted(analysis.modules, key=str.lower))
    allowed_external = {"__future__", "sys"}
    if experimental_kernels:
        allowed_external.update({"numpy", "torch"})
    for module in imports:
        root_name = module.partition(".")[0]
        if (
            root_name in allowed_external
            or _local_candidate(source_root, module) is not None
        ):
            continue
        if root_name in getattr(sys, "stdlib_module_names", set()):
            blockers.append(
                f"import {module}: standard-library module needs a py2bin-native "
                "implementation; embedding CPython's stdlib is forbidden"
            )
        else:
            blockers.append(
                f"import {module}: package needs a CPython-free py2bin adapter "
                "or must lower from supplied source; wheels/extensions are not "
                "silently bundled"
            )
    for module in sorted(analysis.unresolved, key=str.lower):
        blockers.append(
            f"import {module}: implementation is unresolved and cannot be "
            "proven native"
        )

    source = entry.read_text(encoding="utf-8")
    try:
        lower(
            entry,
            source,
            (source_root,),
            experimental_kernels=experimental_kernels,
        )
    except (NativeCompileError, ValueError) as error:
        compiler_reason = str(error)
        blockers.insert(0, compiler_reason)
    else:
        compiler_reason = (
            "entry and every reached operation lower through py2bin native IR; "
            "no compatibility backend is selected"
        )

    unique_blockers = tuple(dict.fromkeys(blockers))
    return AOTApplicationPlan(
        entry,
        source_root,
        not unique_blockers,
        compiler_reason,
        imports,
        reachable_python,
        web_assets,
        native_payloads,
        foreign_sources,
        unique_blockers,
    )


def require_aot_application(
    entry: Path,
    *,
    source_root: Path | None = None,
    experimental_kernels: bool = False,
) -> AOTApplicationPlan:
    plan = plan_aot_application(
        entry,
        source_root=source_root,
        experimental_kernels=experimental_kernels,
    )
    if not plan.buildable:
        raise AOTPlanError(
            f"strict CPython-free AOT plan found {len(plan.blockers)} blocker(s); "
            f"first: {plan.blockers[0]}"
        )
    return plan


def _artifact_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    return tuple(
        sorted(
            (candidate for candidate in path.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(path).as_posix(),
        )
    )


def _artifact_sha256(path: Path, files: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()
    for file in files:
        relative = file.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def attest_aot_artifact(
    result: NativeResult,
    *,
    pipeline: str = "python-ir-machine",
    ir_roundtrip_verified: bool = False,
    canonical_c_sha256: str | None = None,
) -> AOTAttestation:
    """Verify that a direct-native result contains no Python/freeze payload."""

    artifact = result.artifact
    files = _artifact_files(artifact)
    forbidden_paths = [
        file
        for file in files
        if file.suffix.lower() in {".py", ".pyc", ".pyo"}
        or "Python.framework" in file.parts
        or file.name.lower().startswith("libpython")
    ]
    if forbidden_paths:
        raise AOTPlanError(
            "native artifact unexpectedly contains Python runtime payload: "
            + str(forbidden_paths[0])
        )
    for file in files:
        data = file.read_bytes()
        marker = next((item for item in _RUNTIME_MARKERS if item in data), None)
        if marker is not None:
            raise AOTPlanError(
                f"native artifact unexpectedly contains compatibility marker "
                f"{marker!r}: {file}"
            )
    return AOTAttestation(
        artifact,
        result.target,
        _artifact_sha256(artifact, files),
        result.bytes,
        result.operations,
        pipeline,
        ir_roundtrip_verified,
        canonical_c_sha256,
        True,
        True,
        True,
    )


def build_aot_application(
    entry: Path,
    output: Path,
    *,
    target: str | None = None,
    clean: bool = False,
    app: bool = False,
    source_root: Path | None = None,
    experimental_kernels: bool = False,
    attestation: Path | None = None,
    via_c: bool = False,
    c_output: Path | None = None,
) -> AOTBuildResult:
    """Build only with handwritten native backends; never freeze or fallback."""

    entry = entry.expanduser().resolve()
    source_root = (source_root or entry.parent).expanduser().resolve()
    plan = require_aot_application(
        entry,
        source_root=source_root,
        experimental_kernels=experimental_kernels,
    )
    attestation_path = (
        attestation.expanduser().resolve() if attestation is not None else None
    )
    output_path = output.expanduser().resolve()
    c_artifact = c_output.expanduser().resolve() if c_output is not None else None
    if c_artifact is not None and not via_c:
        raise ValueError("c_output requires via_c=True")
    if attestation_path == output_path:
        raise ValueError("attestation path must be different from the artifact path")
    if c_artifact is not None and c_artifact in {output_path, attestation_path}:
        raise ValueError("C, attestation, and artifact paths must be different")
    if attestation_path is not None and attestation_path.exists() and not clean:
        raise FileExistsError(
            f"attestation already exists: {attestation_path} (use --clean)"
        )
    if c_artifact is not None and c_artifact.exists() and not clean:
        raise FileExistsError(f"C output already exists: {c_artifact} (use --clean)")
    c_source: str | None = None
    if via_c:
        module, _optimization = optimize(
            lower(
                entry,
                entry.read_text(encoding="utf-8"),
                (source_root,),
                experimental_kernels=experimental_kernels,
            )
        )
        c_source, reconstructed = roundtrip_ir_c(module)
        native = compile_native_module(
            entry,
            reconstructed,
            output,
            target=target,
            clean=clean,
            app=app,
        )
        pipeline = "python-ir-c-ir-machine"
    else:
        native = compile_native(
            entry,
            output,
            target=target,
            clean=clean,
            app=app,
            source_roots=(source_root,),
            experimental_kernels=experimental_kernels,
        )
        pipeline = "python-ir-machine"
    proof = attest_aot_artifact(
        native,
        pipeline=pipeline,
        ir_roundtrip_verified=via_c,
        canonical_c_sha256=(
            hashlib.sha256(c_source.encode("utf-8")).hexdigest()
            if c_source is not None
            else None
        ),
    )
    if c_artifact is not None:
        assert c_source is not None
        c_artifact.parent.mkdir(parents=True, exist_ok=True)
        c_artifact.write_text(c_source, encoding="utf-8", newline="\n")
    if attestation_path is not None:
        attestation_path.parent.mkdir(parents=True, exist_ok=True)
        attestation_path.write_text(
            json.dumps(proof.as_dict(), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return AOTBuildResult(
        native,
        plan,
        proof,
        attestation_path,
        c_source,
        c_artifact,
    )


__all__ = [
    "AOTApplicationPlan",
    "AOTAttestation",
    "AOTBuildResult",
    "AOTPlanError",
    "attest_aot_artifact",
    "build_aot_application",
    "plan_aot_application",
    "require_aot_application",
]
