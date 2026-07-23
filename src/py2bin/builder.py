from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import shutil
import stat
import tempfile
import zipapp
from pathlib import Path, PurePath

from .analyzer import analyze
from .icons import install_macos_icon, macos_info_plist
from .model import ArtifactKind, BuildConfig, BuildResult


_IGNORED_PARTS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", "dist", "build"}


def _safe_remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _copy_project(
    source: Path,
    destination: Path,
    excluded_paths: tuple[Path, ...] = (),
) -> None:
    excluded = {path.resolve() for path in excluded_paths}

    def ignored(directory: str, names: list[str]) -> set[str]:
        parent = Path(directory)
        return {
            name
            for name in names
            if name in _IGNORED_PARTS
            or name.endswith((".pyc", ".pyo"))
            or (parent / name).resolve() in excluded
        }

    shutil.copytree(source, destination, ignore=ignored)


def _copy_distribution(name: str, destination: Path, compact: bool = False) -> None:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return
    for item in distribution.files or ():
        relative = Path(*PurePath(item).parts)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        if compact and (
            any(part in {"PyObjCTest", "test", "tests", "__pycache__"} for part in relative.parts)
            or relative.name.endswith((".pyc", ".pyo"))
        ):
            continue
        source = Path(distribution.locate_file(item))
        target = destination / relative
        if source.is_dir():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except FileNotFoundError:
            # Editable installs may list generated or absent files.
            continue


def _write_runtime(stage: Path, config: BuildConfig, analysis) -> Path:
    runtime = stage / "runtime"
    runtime.mkdir()
    bootstrap_source = Path(__file__).with_name("bootstrap.py")
    shutil.copy2(bootstrap_source, runtime / "bootstrap.py")
    (stage / "__main__.py").write_text(
        "from runtime.bootstrap import main\nmain()\n", encoding="utf-8"
    )
    assert config.source_root is not None
    entry_relative = config.entry.relative_to(config.source_root)
    manifest = {
        "schema": 1,
        "name": config.name,
        "entry": entry_relative.as_posix(),
        "python": config.python,
        "build_platform": os.uname().sysname if hasattr(os, "uname") else os.name,
        "distributions": sorted(analysis.distributions, key=str.lower),
        "unresolved_imports": sorted(analysis.unresolved),
        "notes": analysis.hook_notes,
    }
    manifest_path = stage / "py2bin-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _launcher(path: Path, bundle: Path, python: str) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        f'exec {python} "$(dirname "$0")/{bundle.name}/runtime/bootstrap.py" "$@"\n',
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_app(
    stage: Path,
    output: Path,
    name: str,
    python: str,
    icon: Path | None = None,
) -> Path:
    app = output if output.suffix == ".app" else output.with_suffix(".app")
    executable = app / "Contents" / "MacOS" / name
    resources = app / "Contents" / "Resources"
    resources.mkdir(parents=True)
    icon_filename = install_macos_icon(icon, resources)
    shutil.copytree(stage, resources / "bundle")
    executable.parent.mkdir(parents=True)
    executable.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        f'ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../Resources/bundle" && pwd)"\n'
        f'exec {python} "$ROOT/runtime/bootstrap.py" "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    (app / "Contents" / "Info.plist").write_bytes(
        macos_info_plist(name, name, icon_filename)
    )
    return app


def _measure(path: Path) -> tuple[int, int]:
    if path.is_file():
        return 1, path.stat().st_size
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def build(raw_config: BuildConfig) -> BuildResult:
    config = raw_config.normalized()
    assert config.source_root is not None and config.name is not None
    if not config.entry.is_file():
        raise FileNotFoundError(f"entry script does not exist: {config.entry}")
    if not config.entry.is_relative_to(config.source_root):
        raise ValueError("entry must be inside source_root")
    if config.icon is not None and config.kind is not ArtifactKind.APP:
        raise ValueError("--icon currently requires --format app")
    requested_output = config.output
    if config.kind is ArtifactKind.PYZ and requested_output.suffix != ".pyz":
        requested_output = requested_output.with_suffix(".pyz")
    elif config.kind is ArtifactKind.APP and requested_output.suffix != ".app":
        requested_output = requested_output.with_suffix(".app")
    config.output = requested_output
    if config.output.exists():
        if not config.clean:
            raise FileExistsError(f"output already exists: {config.output} (use --clean)")
        _safe_remove(config.output)

    analysis = analyze(
        config.entry,
        config.source_root,
        config.includes,
        config.excludes,
        config.dependency_mode,
    )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="py2bin-", dir=config.output.parent) as temporary:
        stage = Path(temporary) / "stage"
        stage.mkdir()
        _copy_project(config.source_root, stage / "app")
        packages = stage / "site-packages"
        packages.mkdir()
        for distribution in sorted(analysis.distributions, key=str.lower):
            _copy_distribution(distribution, packages)
        _write_runtime(stage, config, analysis)

        if config.kind is ArtifactKind.DIRECTORY:
            artifact = config.output
            shutil.copytree(stage, artifact)
            _launcher(artifact.parent / f"{artifact.name}.run", artifact, config.python)
        elif config.kind is ArtifactKind.APP:
            artifact = _make_app(stage, config.output, config.name, config.python, config.icon)
        else:
            artifact = config.output
            zipapp.create_archive(stage, artifact, interpreter=config.python, compressed=True)
            artifact.chmod(artifact.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    files, size = _measure(artifact)
    manifest = artifact / "py2bin-manifest.json" if artifact.is_dir() else None
    return BuildResult(artifact, manifest, files, size, analysis)
