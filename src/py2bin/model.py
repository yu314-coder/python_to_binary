from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ArtifactKind(str, Enum):
    DIRECTORY = "dir"
    PYZ = "pyz"
    BIN = "bin"
    APP = "app"


@dataclass(slots=True)
class BuildConfig:
    entry: Path
    output: Path
    kind: ArtifactKind = ArtifactKind.BIN
    name: str | None = None
    source_root: Path | None = None
    includes: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    dependency_mode: str = "closure"
    python: str = "/usr/bin/env python3"
    icon: Path | None = None
    clean: bool = False

    def normalized(self) -> "BuildConfig":
        entry = self.entry.expanduser().resolve()
        source_root = (self.source_root or entry.parent).expanduser().resolve()
        if self.dependency_mode not in {"none", "imported", "closure"}:
            raise ValueError("dependency_mode must be none, imported, or closure")
        return BuildConfig(
            entry=entry,
            output=self.output.expanduser().resolve(),
            kind=ArtifactKind(self.kind),
            name=self.name or entry.stem,
            source_root=source_root,
            includes=tuple(dict.fromkeys(self.includes)),
            excludes=tuple(dict.fromkeys(self.excludes)),
            dependency_mode=self.dependency_mode,
            python=self.python,
            icon=self.icon.expanduser().resolve() if self.icon is not None else None,
            clean=self.clean,
        )


@dataclass(slots=True)
class ImportAnalysis:
    modules: set[str] = field(default_factory=set)
    unresolved: set[str] = field(default_factory=set)
    local_files: set[Path] = field(default_factory=set)
    distributions: set[str] = field(default_factory=set)
    hook_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BuildResult:
    artifact: Path
    manifest: Path | None
    files: int
    bytes: int
    analysis: ImportAnalysis
