"""Minimal PEP 517 backend so this project has no build dependencies."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import tarfile
import zipfile
from pathlib import Path

_VERSION = "0.2.1"
_DISTRIBUTION = f"python_to_binary-{_VERSION}"
_ZIP_EPOCH = 315532800  # 1980-01-01 UTC, the oldest timestamp ZIP accepts.


def _metadata() -> str:
    root = Path(__file__).resolve().parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")
    return (
        "Metadata-Version: 2.1\n"
        "Name: python-to-binary\n"
        f"Version: {_VERSION}\n"
        "Summary: Pure-Python Python-to-C, native binary compiler, and CPython application bundler\n"
        "Requires-Python: >=3.10\n"
        "License: MIT\n"
        "Project-URL: Homepage, https://github.com/yu314-coder/python_to_binary\n"
        "Project-URL: Repository, https://github.com/yu314-coder/python_to_binary\n"
        "Project-URL: Documentation, https://github.com/yu314-coder/python_to_binary/blob/main/docs/DETAILED_GUIDE.md\n"
        "Description-Content-Type: text/markdown\n\n"
        + readme
        + "\n"
    )


def build_wheel(wheel_directory: str, config_settings=None, metadata_directory=None) -> str:
    del config_settings, metadata_directory
    root = Path(__file__).resolve().parents[2]
    name = f"{_DISTRIBUTION}-py3-none-any.whl"
    target = Path(wheel_directory) / name
    dist_info = f"{_DISTRIBUTION}.dist-info"
    rows: list[tuple[str, str, str]] = []
    files: dict[str, bytes] = {}
    source_root = root / "src"
    for source in sorted((source_root / "py2bin").rglob("*.py")):
        files[source.relative_to(source_root).as_posix()] = source.read_bytes()
    files[f"{dist_info}/METADATA"] = _metadata().encode()
    files[f"{dist_info}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: py2bin\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    files[f"{dist_info}/entry_points.txt"] = b"[console_scripts]\npy2bin = py2bin.cli:main\n"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as wheel:
        for path, data in files.items():
            wheel.writestr(path, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            rows.append((path, f"sha256={digest}", str(len(data))))
        rows.append((f"{dist_info}/RECORD", "", ""))
        record = io.StringIO()
        csv.writer(record, lineterminator="\n").writerows(rows)
        wheel.writestr(f"{dist_info}/RECORD", record.getvalue())
    return name


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    del config_settings
    return []


def build_sdist(sdist_directory: str, config_settings=None) -> str:
    del config_settings
    root = Path(__file__).resolve().parents[2]
    archive_name = f"{_DISTRIBUTION}.tar.gz"
    target = Path(sdist_directory) / archive_name
    prefix = _DISTRIBUTION
    included = [
        root / ".gitignore",
        root / "ARCHITECTURE.md",
        root / "LICENSE",
        root / "README.md",
        root / "pyproject.toml",
        root / "docs",
        root / "examples",
        root / "src",
        root / "tests",
    ]

    def normalized(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = _ZIP_EPOCH
        return info

    with tarfile.open(target, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        metadata = _metadata().encode("utf-8")
        metadata_info = tarfile.TarInfo(f"{prefix}/PKG-INFO")
        metadata_info.size = len(metadata)
        metadata_info = normalized(metadata_info)
        archive.addfile(metadata_info, io.BytesIO(metadata))
        for source in included:
            if source.exists():
                archive.add(
                    source,
                    arcname=f"{prefix}/{source.relative_to(root).as_posix()}",
                    recursive=True,
                    filter=normalized,
                )
    return archive_name


def get_requires_for_build_sdist(config_settings=None) -> list[str]:
    del config_settings
    return []


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings=None) -> str:
    del config_settings
    name = f"{_DISTRIBUTION}.dist-info"
    target = Path(metadata_directory) / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "METADATA").write_text(_metadata(), encoding="utf-8", newline="\n")
    (target / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: py2bin\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
        newline="\n",
    )
    return name
