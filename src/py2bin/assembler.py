from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .freezer import FreezeResult, freeze
from .native import NativeCompileError, compile_native
from .native.compiler import host_target
from .runtime_packs import inspect_runtime_pack


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    artifact: Path
    launcher: Path
    backend: str
    target: str
    bytes: int
    reason: str


def assemble(
    entry: Path,
    output: Path,
    *,
    mode: str = "auto",
    target: str | None = None,
    source_root: Path | None = None,
    includes: tuple[str, ...] = (),
    excludes: tuple[str, ...] = (),
    wheels: tuple[Path, ...] = (),
    dependency_mode: str = "closure",
    runtime_pack: Path | None = None,
    app: bool = False,
    name: str | None = None,
    icon: Path | None = None,
    compact: bool = False,
    clean: bool = False,
    onefile: bool = True,
) -> AssemblyResult:
    """Choose real native compilation or a compatible embedded-runtime bundle."""

    if mode not in {"auto", "native", "compatible"}:
        raise ValueError("mode must be auto, native, or compatible")
    if target is None:
        target = (
            inspect_runtime_pack(runtime_pack).target
            if runtime_pack is not None
            else host_target()
        )
    native_error: NativeCompileError | None = None
    if mode != "compatible":
        try:
            result = compile_native(
                entry,
                output,
                target=target,
                clean=clean,
                app=app,
            )
        except NativeCompileError as error:
            if mode == "native":
                raise
            native_error = error
        else:
            launcher = (
                result.artifact / "Contents" / "MacOS" / entry.stem
                if app
                else result.artifact
            )
            return AssemblyResult(
                result.artifact,
                launcher,
                "native",
                result.target,
                result.bytes,
                "the program is inside the documented static native subset",
            )

    frozen: FreezeResult = freeze(
        entry,
        output,
        source_root,
        includes,
        excludes,
        wheels,
        dependency_mode,
        clean,
        app=app,
        name=name,
        icon=icon,
        compact=compact,
        runtime_pack=runtime_pack,
        target=target,
        onefile=onefile,
    )
    reason = (
        "compatible mode was requested"
        if native_error is None
        else f"native subset rejected the program: {native_error}"
    )
    return AssemblyResult(
        frozen.bundle,
        frozen.launcher,
        "compatible",
        frozen.target,
        frozen.bytes,
        reason,
    )
