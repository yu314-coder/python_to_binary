"""Dependency-free wheel creation for source and prebuilt native payloads."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VERSION = re.compile(r"^[0-9][a-z0-9.!+]*$")
_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.]*$")
_NATIVE_SUFFIXES = (".pyd", ".so", ".dylib", ".dll")
_SKIP_SUFFIXES = (".pyc", ".pyo")


@dataclass(frozen=True, slots=True)
class WheelBuildResult:
    wheel: Path
    files: int
    bytes: int
    native_files: tuple[str, ...]
    cython_sources: tuple[str, ...]
    tag: str


def _distribution_component(value: str) -> str:
    return re.sub(r"[-_.]+", "_", value).lower()


def _wheel_file(path: Path, source_root: Path, output_directory: Path) -> bool:
    relative = path.relative_to(source_root)
    try:
        output_inside_source = output_directory.relative_to(source_root)
    except ValueError:
        output_inside_source = None
    if (
        output_inside_source is not None
        and output_inside_source.parts
        and relative.parts[: len(output_inside_source.parts)] == output_inside_source.parts
    ):
        return False
    if "__pycache__" in relative.parts:
        return False
    if any(part.endswith((".dist-info", ".egg-info")) for part in relative.parts):
        return False
    if path.name == ".DS_Store" or path.suffix in _SKIP_SUFFIXES:
        return False
    return path.is_file()


def _top_levels(paths: list[str]) -> tuple[str, ...]:
    result: set[str] = set()
    for name in paths:
        first = name.partition("/")[0]
        if first.endswith(".py"):
            root = first[:-3]
        else:
            root = first.partition(".")[0]
        if root.isidentifier():
            result.add(root)
    return tuple(sorted(result))


def _record_row(path: str, data: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return path, f"sha256={digest.rstrip(b'=').decode('ascii')}", str(len(data))


def build_payload_wheel(
    source_root: Path,
    output_directory: Path,
    *,
    name: str,
    version: str,
    python_tag: str = "py3",
    abi_tag: str = "none",
    platform_tag: str = "any",
    requirements: tuple[str, ...] = (),
    clean: bool = False,
) -> WheelBuildResult:
    """Package a tree without compiling or executing any of its files.

    Prebuilt Cython/native extensions are preserved byte-for-byte.  A native
    payload requires explicit Python ABI and platform tags so a generic wheel
    cannot accidentally be advertised as portable.
    """

    source_root = source_root.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"wheel source directory does not exist: {source_root}")
    if output_directory == source_root:
        raise ValueError("wheel output directory must not equal the packaged source root")
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid wheel distribution name: {name!r}")
    if not _VERSION.fullmatch(version):
        raise ValueError(
            f"wheel version must already be normalized and contain no '-' or '_': {version!r}"
        )
    if any(not requirement.strip() or "\n" in requirement or "\r" in requirement for requirement in requirements):
        raise ValueError("wheel requirements must be nonempty single-line Requires-Dist values")
    for label, value in (
        ("Python", python_tag),
        ("ABI", abi_tag),
        ("platform", platform_tag),
    ):
        if not _TAG.fullmatch(value):
            raise ValueError(f"invalid wheel {label} tag: {value!r}")

    source_files: list[Path] = []
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"wheel source tree contains unsupported symbolic link: {path}")
        if _wheel_file(path, source_root, output_directory):
            source_files.append(path)
    if not source_files:
        raise ValueError(f"wheel source directory contains no package files: {source_root}")
    relative_names = [path.relative_to(source_root).as_posix() for path in source_files]
    native_files = tuple(
        name
        for name in relative_names
        if name.lower().endswith(_NATIVE_SUFFIXES) or ".so." in name.lower()
    )
    cython_sources = tuple(
        name for name in relative_names if name.lower().endswith((".pyx", ".pxd"))
    )
    if native_files and (abi_tag == "none" or platform_tag == "any"):
        raise ValueError(
            "wheel contains native files but has a portable tag; supply the exact "
            "--python-tag, --abi-tag, and --platform-tag for the prebuilt payload"
        )

    distribution = _distribution_component(name)
    version_component = version
    tag = f"{python_tag}-{abi_tag}-{platform_tag}"
    filename = f"{distribution}-{version_component}-{tag}.whl"
    output_directory.mkdir(parents=True, exist_ok=True)
    target = output_directory / filename
    if target.exists() and not clean:
        raise FileExistsError(f"wheel already exists: {target} (use --clean)")

    dist_info = f"{distribution}-{version_component}.dist-info"
    metadata_lines = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
    ]
    metadata_lines.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    metadata = ("\n".join(metadata_lines) + "\n\n").encode("utf-8")
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        "Generator: py2bin\n"
        f"Root-Is-Purelib: {'false' if native_files else 'true'}\n"
        f"Tag: {tag}\n"
    ).encode("utf-8")
    top_level = ("\n".join(_top_levels(relative_names)) + "\n").encode("utf-8")

    files: list[tuple[str, bytes]] = [
        (relative, source.read_bytes())
        for relative, source in zip(relative_names, source_files)
    ]
    files.extend(
        (
            (f"{dist_info}/METADATA", metadata),
            (f"{dist_info}/WHEEL", wheel_metadata),
            (f"{dist_info}/top_level.txt", top_level),
        )
    )
    record_path = f"{dist_info}/RECORD"
    rows = [_record_row(path, data) for path, data in files]
    rows.append((record_path, "", ""))
    record = io.StringIO()
    csv.writer(record, lineterminator="\n").writerows(rows)
    files.append((record_path, record.getvalue().encode("utf-8")))

    with tempfile.TemporaryDirectory(
        prefix="py2bin-wheel-", dir=output_directory
    ) as temporary:
        staged = Path(temporary) / filename
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as archive:
            for relative, data in files:
                info = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, data)
        if target.exists():
            target.unlink()
        staged.replace(target)

    return WheelBuildResult(
        target,
        len(files),
        target.stat().st_size,
        native_files,
        cython_sources,
        tag,
    )
