"""Pinned-source to handwritten native binary orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .native import NativeResult, compile_native
from .source_fetch import FetchedSource, fetch_sources_for_entry


@dataclass(frozen=True, slots=True)
class SourceNativeResult:
    native: NativeResult
    fetched: tuple[FetchedSource, ...]
    imports: tuple[str, ...]


def compile_locked_sources(
    entry: Path,
    output: Path,
    *,
    source_lock: Path,
    source_cache: Path,
    source_root: Path | None = None,
    target: str | None = None,
    clean: bool = False,
    app: bool = False,
) -> SourceNativeResult:
    """Fetch pinned source imports and attempt only real native compilation.

    There is deliberately no compatible-runtime fallback. If downloaded code
    is outside the direct native subset, NativeCompileError is returned to the
    caller and no artifact is presented as native.
    """

    entry = entry.expanduser().resolve()
    source_root = (source_root or entry.parent).expanduser().resolve()
    fetched = fetch_sources_for_entry(
        entry,
        source_root,
        source_lock,
        source_cache,
    )
    native = compile_native(
        entry,
        output,
        target=target,
        clean=clean,
        app=app,
        source_roots=fetched.roots,
    )
    return SourceNativeResult(native, fetched.fetched, fetched.imports)
