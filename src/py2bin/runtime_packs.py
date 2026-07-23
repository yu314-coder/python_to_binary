from __future__ import annotations

import json
import re
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


MANIFEST_NAME = "py2bin-runtime.json"
SUPPORTED_TARGETS = {
    "linux-x86_64",
    "linux-arm64",
    "darwin-x86_64",
    "darwin-arm64",
    "windows-x86_64",
    "windows-arm64",
}
_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass(frozen=True, slots=True)
class RuntimePackInfo:
    source: Path
    target: str
    python: str
    executable: Path
    environment: dict[str, str]


def _relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"runtime pack {field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"runtime pack {field} escapes the pack root: {value!r}")
    return Path(*path.parts) if path.parts else Path(".")


def _validate_manifest(source: Path, data: object) -> RuntimePackInfo:
    if not isinstance(data, dict):
        raise ValueError("runtime pack manifest must be a JSON object")
    if data.get("schema") != 1:
        raise ValueError("runtime pack schema must be 1")
    target = data.get("target")
    if target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"runtime pack target must be one of: {', '.join(sorted(SUPPORTED_TARGETS))}"
        )
    python = data.get("python")
    if not isinstance(python, str) or not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", python):
        raise ValueError("runtime pack python must look like 3.12 or 3.12.10")
    executable = _relative_path(data.get("executable"), "executable")
    raw_environment = data.get("environment", {})
    if not isinstance(raw_environment, dict):
        raise ValueError("runtime pack environment must be a JSON object")
    environment: dict[str, str] = {}
    for key, value in raw_environment.items():
        if not isinstance(key, str) or not _ENVIRONMENT_NAME.fullmatch(key):
            raise ValueError(f"invalid runtime pack environment name: {key!r}")
        environment[key] = _relative_path(value, f"environment[{key}]").as_posix()
    return RuntimePackInfo(source, target, python, executable, environment)


def inspect_runtime_pack(source: Path) -> RuntimePackInfo:
    source = source.expanduser().resolve()
    if source.is_dir():
        manifest = source / MANIFEST_NAME
        if not manifest.is_file():
            raise FileNotFoundError(f"runtime pack manifest does not exist: {manifest}")
        return _validate_manifest(
            source, json.loads(manifest.read_text(encoding="utf-8"))
        )
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            try:
                data = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            except KeyError as error:
                raise ValueError(
                    f"runtime pack ZIP must contain {MANIFEST_NAME} at its root"
                ) from error
        return _validate_manifest(source, data)
    raise FileNotFoundError(
        f"runtime pack must be a directory or ZIP file: {source}"
    )


def _safe_archive_member(name: str) -> Path | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    return Path(*path.parts)


def install_runtime_pack(source: Path, destination: Path) -> RuntimePackInfo:
    """Copy a validated runtime pack into a bundle staging root."""

    info = inspect_runtime_pack(source)
    destination.mkdir(parents=True, exist_ok=True)
    if info.source.is_dir():
        for source_path in sorted(info.source.rglob("*")):
            relative = source_path.relative_to(info.source)
            if relative == Path(MANIFEST_NAME):
                continue
            if source_path.is_symlink():
                raise ValueError(f"runtime pack symlinks are not accepted: {relative}")
            target = destination / relative
            if source_path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                raise ValueError(f"runtime pack collides with bundle path: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
    else:
        with zipfile.ZipFile(info.source) as archive:
            for member in archive.infolist():
                relative = _safe_archive_member(member.filename)
                unix_mode = (member.external_attr >> 16) & 0xFFFF
                if (
                    relative is None
                    or relative == Path(MANIFEST_NAME)
                    or member.is_dir()
                    or stat.S_ISLNK(unix_mode)
                ):
                    continue
                target = destination / relative
                if target.exists():
                    raise ValueError(
                        f"runtime pack collides with bundle path: {relative}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as input_stream, target.open("wb") as output:
                    shutil.copyfileobj(input_stream, output)
                if unix_mode:
                    target.chmod(unix_mode & 0o777)
    executable = destination / info.executable
    if not executable.is_file():
        raise ValueError(
            f"runtime pack executable is missing after extraction: {info.executable}"
        )
    return info


def write_runtime_manifest(
    destination: Path,
    *,
    target: str,
    python: str,
    executable: Path,
    environment: dict[str, str],
) -> RuntimePackInfo:
    data = {
        "schema": 1,
        "target": target,
        "python": python,
        "executable": executable.as_posix(),
        "environment": environment,
    }
    manifest = destination / MANIFEST_NAME
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return _validate_manifest(destination.resolve(), data)
