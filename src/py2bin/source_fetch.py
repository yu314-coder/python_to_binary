"""Pinned, non-executing source acquisition for native compilation attempts."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


LOCK_SCHEMA = 1
SOURCE_MANIFEST = ".py2bin-source.json"
DEFAULT_MAX_DOWNLOAD = 1024 * 1024 * 1024
DEFAULT_MAX_EXPANDED = 4 * 1024 * 1024 * 1024
DEFAULT_MAX_MEMBERS = 200_000


@dataclass(frozen=True, slots=True)
class SourceSpec:
    module: str
    url: str | None
    path: Path | None
    revision: str
    sha256: str
    subdirectory: str


@dataclass(frozen=True, slots=True)
class SourceLock:
    path: Path
    sources: dict[str, SourceSpec]


@dataclass(frozen=True, slots=True)
class FetchedSource:
    module: str
    root: Path
    revision: str
    sha256: str
    origin: str


@dataclass(frozen=True, slots=True)
class SourceFetchResult:
    roots: tuple[Path, ...]
    fetched: tuple[FetchedSource, ...]
    imports: tuple[str, ...]


def _safe_relative(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path: {value!r}")
    return path


def load_source_lock(path: Path) -> SourceLock:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source lock does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid source lock JSON: {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != LOCK_SCHEMA:
        raise ValueError(f"source lock must use schema {LOCK_SCHEMA}: {path}")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, dict):
        raise ValueError("source lock 'sources' must be an object")
    sources: dict[str, SourceSpec] = {}
    for module, raw in raw_sources.items():
        if not isinstance(module, str) or not module.isidentifier():
            raise ValueError(f"invalid source-lock import name: {module!r}")
        if not isinstance(raw, dict):
            raise ValueError(f"source-lock entry {module!r} must be an object")
        url = raw.get("url")
        local = raw.get("path")
        if (url is None) == (local is None):
            raise ValueError(
                f"source-lock entry {module!r} requires exactly one of url or path"
            )
        if url is not None and not isinstance(url, str):
            raise ValueError(f"source-lock URL for {module!r} must be a string")
        local_path = None
        if local is not None:
            if not isinstance(local, str):
                raise ValueError(f"source-lock path for {module!r} must be a string")
            candidate = Path(local).expanduser()
            local_path = (
                candidate.resolve()
                if candidate.is_absolute()
                else (path.parent / candidate).resolve()
            )
        revision = raw.get("revision")
        digest = raw.get("sha256")
        subdirectory = raw.get("subdirectory", "")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"source-lock entry {module!r} requires a revision")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise ValueError(
                f"source-lock entry {module!r} requires a 64-digit SHA-256"
            )
        if not isinstance(subdirectory, str):
            raise ValueError(
                f"source-lock subdirectory for {module!r} must be a string"
            )
        _safe_relative(subdirectory, f"source-lock subdirectory for {module!r}")
        sources[module] = SourceSpec(
            module,
            url,
            local_path,
            revision.strip(),
            digest.lower(),
            subdirectory,
        )
    return SourceLock(path, sources)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    spec: SourceSpec,
    downloads: Path,
    *,
    max_download: int,
) -> Path:
    downloads.mkdir(parents=True, exist_ok=True)
    cached = downloads / f"{spec.sha256}.archive"
    if cached.is_file():
        if _hash(cached) != spec.sha256:
            raise ValueError(f"cached source archive has wrong SHA-256: {cached}")
        return cached
    if spec.path is not None:
        if not spec.path.is_file():
            raise FileNotFoundError(
                f"locked source archive for {spec.module!r} does not exist: {spec.path}"
            )
        if spec.path.stat().st_size > max_download:
            raise ValueError(f"source archive exceeds download limit: {spec.path}")
        origin_stream = spec.path.open("rb")
        final_url = None
    else:
        assert spec.url is not None
        parsed = urllib.parse.urlsplit(spec.url)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError(
                f"source URL for {spec.module!r} must be credential-free HTTPS"
            )
        request = urllib.request.Request(
            spec.url,
            headers={"User-Agent": "py2bin-source-fetch/1"},
        )
        origin_stream = urllib.request.urlopen(request, timeout=60)
        final_url = origin_stream.geturl()
        if urllib.parse.urlsplit(final_url).scheme != "https":
            origin_stream.close()
            raise ValueError(
                f"source URL for {spec.module!r} redirected outside HTTPS"
            )
    try:
        with tempfile.NamedTemporaryFile(
            prefix="py2bin-download-", dir=downloads, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = origin_stream.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_download:
                    raise ValueError(
                        f"source archive for {spec.module!r} exceeds download limit"
                    )
                digest.update(chunk)
                temporary.write(chunk)
    except BaseException:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)
        raise
    finally:
        origin_stream.close()
    if digest.hexdigest() != spec.sha256:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"source archive SHA-256 mismatch for {spec.module!r}; "
            f"expected {spec.sha256}, received {digest.hexdigest()}"
        )
    temporary_path.replace(cached)
    return cached


def _archive_destination(root: Path, name: str) -> Path:
    relative = _safe_relative(name, "archive member")
    if not relative.parts:
        raise ValueError("archive contains an empty member name")
    return root.joinpath(*relative.parts)


def _extract_zip(
    archive_path: Path,
    root: Path,
    *,
    max_expanded: int,
    max_members: int,
) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > max_members:
            raise ValueError("source ZIP contains too many members")
        total = 0
        seen: set[str] = set()
        for member in members:
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"source ZIP contains a symbolic link: {member.filename}")
            destination = _archive_destination(root, member.filename)
            relative_key = destination.relative_to(root).as_posix().casefold()
            if relative_key in seen:
                raise ValueError(
                    f"source ZIP contains a duplicate/case-colliding member: {member.filename}"
                )
            seen.add(relative_key)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            total += member.file_size
            if total > max_expanded:
                raise ValueError("expanded source ZIP exceeds size limit")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)


def _extract_tar(
    archive_path: Path,
    root: Path,
    *,
    max_expanded: int,
    max_members: int,
) -> None:
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        if len(members) > max_members:
            raise ValueError("source tar archive contains too many members")
        total = 0
        seen: set[str] = set()
        for member in members:
            destination = _archive_destination(root, member.name)
            relative_key = destination.relative_to(root).as_posix().casefold()
            if relative_key in seen:
                raise ValueError(
                    f"source tar contains a duplicate/case-colliding member: {member.name}"
                )
            seen.add(relative_key)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(
                    f"source tar contains a link or special file: {member.name}"
                )
            total += member.size
            if total > max_expanded:
                raise ValueError("expanded source tar archive exceeds size limit")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read source tar member: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)


def _extract(
    archive_path: Path,
    root: Path,
    *,
    max_expanded: int,
    max_members: int,
) -> None:
    if zipfile.is_zipfile(archive_path):
        _extract_zip(
            archive_path,
            root,
            max_expanded=max_expanded,
            max_members=max_members,
        )
        return
    if tarfile.is_tarfile(archive_path):
        _extract_tar(
            archive_path,
            root,
            max_expanded=max_expanded,
            max_members=max_members,
        )
        return
    raise ValueError(f"locked source is not a ZIP or tar archive: {archive_path}")


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source cache contains a symbolic link: {path}")
        if not path.is_file() or path.name == SOURCE_MANIFEST:
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def fetch_source(
    spec: SourceSpec,
    cache: Path,
    *,
    max_download: int = DEFAULT_MAX_DOWNLOAD,
    max_expanded: int = DEFAULT_MAX_EXPANDED,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> FetchedSource:
    """Fetch and safely extract one immutable source archive without executing it."""

    cache = cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = _download(spec, cache / "downloads", max_download=max_download)
    target = cache / "sources" / f"{spec.module}-{spec.sha256[:16]}"
    manifest_path = target / SOURCE_MANIFEST
    if target.exists():
        if not manifest_path.is_file():
            raise ValueError(f"source cache target exists without manifest: {target}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != spec.sha256:
            raise ValueError(f"source cache target has unexpected content identity: {target}")
        if manifest.get("tree_sha256") != _tree_hash(target):
            raise ValueError(f"source cache target failed extracted-tree verification: {target}")
        return FetchedSource(
            spec.module,
            target,
            spec.revision,
            spec.sha256,
            spec.url or str(spec.path),
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="py2bin-source-", dir=cache) as temporary:
        extraction = Path(temporary) / "extracted"
        extraction.mkdir()
        _extract(
            archive_path,
            extraction,
            max_expanded=max_expanded,
            max_members=max_members,
        )
        children = list(extraction.iterdir())
        repository_root = children[0] if len(children) == 1 and children[0].is_dir() else extraction
        relative = _safe_relative(
            spec.subdirectory, f"source-lock subdirectory for {spec.module!r}"
        )
        selected = repository_root.joinpath(*relative.parts)
        if not selected.is_dir():
            raise ValueError(
                f"source-lock subdirectory for {spec.module!r} does not exist: "
                f"{spec.subdirectory!r}"
            )
        staged = Path(temporary) / target.name
        shutil.copytree(selected, staged)
        tree_sha256 = _tree_hash(staged)
        (staged / SOURCE_MANIFEST).write_text(
            json.dumps(
                {
                    "schema": 1,
                    "module": spec.module,
                    "revision": spec.revision,
                    "sha256": spec.sha256,
                    "tree_sha256": tree_sha256,
                    "origin": spec.url or str(spec.path),
                    "subdirectory": spec.subdirectory,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staged.replace(target)
    return FetchedSource(
        spec.module,
        target,
        spec.revision,
        spec.sha256,
        spec.url or str(spec.path),
    )


def _candidate(roots: tuple[Path, ...], module: str) -> Path | None:
    parts = module.split(".")
    for root in roots:
        module_path = root.joinpath(*parts).with_suffix(".py")
        package_path = root.joinpath(*parts, "__init__.py")
        if module_path.is_file():
            return module_path
        if package_path.is_file():
            return package_path
    return None


def _imports(path: Path) -> list[tuple[str, int, str | None]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, int, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, 0, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "", node.level, path.as_posix()))
    return imports


def _relative_candidate(path: Path, module: str, level: int) -> Path | None:
    base = path.parent
    for _ in range(level - 1):
        base = base.parent
    relative = base.joinpath(*module.split(".")) if module else base
    candidates = (relative.with_suffix(".py"), relative / "__init__.py")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def fetch_sources_for_entry(
    entry: Path,
    source_root: Path,
    lock_path: Path,
    cache: Path,
) -> SourceFetchResult:
    """Resolve statically imported source archives from a pinned lock."""

    entry = entry.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    if not entry.is_file():
        raise FileNotFoundError(f"entry script does not exist: {entry}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    lock = load_source_lock(lock_path)
    casefolded: dict[str, SourceSpec] = {}
    for name, spec in lock.sources.items():
        key = name.lower()
        if key in casefolded:
            raise ValueError(f"source lock contains case-colliding import names: {name}")
        casefolded[key] = spec
    roots: list[Path] = [source_root]
    fetched: dict[str, FetchedSource] = {}
    discovered: set[str] = set()
    pending = [entry]
    visited: set[Path] = set()
    stdlib = getattr(sys, "stdlib_module_names", set())
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for module, level, _origin in _imports(path):
            if level:
                candidate = _relative_candidate(path, module, level)
                if candidate is not None:
                    pending.append(candidate)
                continue
            if not module:
                continue
            root_name = module.partition(".")[0]
            if root_name in stdlib:
                continue
            discovered.add(root_name)
            candidate = _candidate(tuple(roots), module)
            if candidate is None:
                spec = casefolded.get(root_name.lower())
                if spec is None:
                    raise ValueError(
                        f"no pinned source-lock entry for imported module {root_name!r}"
                    )
                fetched_source = fetched.get(spec.module)
                if fetched_source is None:
                    fetched_source = fetch_source(spec, cache)
                    fetched[spec.module] = fetched_source
                    roots.append(fetched_source.root)
                candidate = _candidate(tuple(roots), module)
                if candidate is None:
                    raise ValueError(
                        f"fetched source for {root_name!r} does not provide import {module!r}"
                    )
            pending.append(candidate)
    return SourceFetchResult(
        tuple(roots[1:]),
        tuple(fetched.values()),
        tuple(sorted(discovered, key=str.lower)),
    )
