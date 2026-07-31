"""Fetch target CPython runtimes and target wheels over verified HTTPS.

A cross-target compatible bundle needs two inputs that a foreign build machine
does not have: the target's CPython runtime and target-compatible wheels. Both
are published, so py2bin can retrieve them itself instead of failing. This
module does that with the same rules as :mod:`py2bin.source_fetch`:

* credential-free HTTPS only, and a redirect may not leave HTTPS;
* an explicit download size limit;
* SHA-256 verification of every downloaded file;
* a content-addressed cache, so a rebuild re-uses bytes it already verified;
* archive extraction that rejects traversal paths, links, and special files.

pip, setuptools, virtualenv, and the network stack of any third-party package
are never used; this is :mod:`urllib.request` plus :mod:`zipfile` from the
standard library.

Integrity model, stated exactly: PyPI and python.org are asked over HTTPS for a
file and its SHA-256, and the download must match that digest. On the first
fetch that is trust-on-first-use, so the digest of every file is written to a
lock file. When a lock file is supplied, the recorded digest is authoritative
and a changed artifact is an error, which makes later builds reproducible and
auditable.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .runtime_packs import RuntimePackInfo, write_runtime_manifest

DEFAULT_MAX_DOWNLOAD = 512 * 1024 * 1024
DEFAULT_MAX_MEMBERS = 100_000
LOCK_SCHEMA = 1
_USER_AGENT = "py2bin-runtime-fetch/1"

_EMBED_URL = "https://www.python.org/ftp/python/{version}/python-{version}-embed-{arch}.zip"
_PYPI_PROJECT = "https://pypi.org/pypi/{project}/json"
_PYPI_RELEASE = "https://pypi.org/pypi/{project}/{version}/json"

_VERSION = re.compile(r"\d+\.\d+\.\d+")
_PROJECT_NAME = re.compile(r"[A-Za-z0-9._-]+")

# python.org publishes the Windows embeddable distribution per architecture.
_EMBED_ARCH = {"windows-x86_64": "amd64", "windows-arm64": "arm64"}

# A wheel's platform tag names an architecture as well as an OS, and matching
# only the OS part is how an x86-64 wheel ends up installed for arm64. Each
# target therefore lists the architecture suffixes it accepts, plus the
# portable tags that carry no native code at all.
_PLATFORM_RULES = {
    "windows-x86_64": ("win_amd64",),
    "windows-arm64": ("win_arm64",),
    "linux-x86_64": ("_x86_64",),
    "linux-arm64": ("_aarch64", "_arm64"),
    "darwin-x86_64": ("_x86_64", "_universal2", "_intel", "_fat64"),
    "darwin-arm64": ("_arm64", "_universal2"),
}
_PLATFORM_OS = {
    "windows-x86_64": ("win",),
    "windows-arm64": ("win",),
    "linux-x86_64": ("manylinux", "musllinux", "linux"),
    "linux-arm64": ("manylinux", "musllinux", "linux"),
    "darwin-x86_64": ("macosx",),
    "darwin-arm64": ("macosx",),
}


class FetchError(RuntimeError):
    """A runtime or wheel could not be retrieved or verified."""


@dataclass(frozen=True, slots=True)
class FetchedFile:
    name: str
    url: str
    sha256: str
    path: Path


@dataclass(slots=True)
class FetchLock:
    """Recorded digests, so a later build verifies rather than trusts."""

    path: Path | None = None
    entries: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> "FetchLock":
        if path is None or not path.is_file():
            return cls(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise FetchError(f"invalid fetch lock JSON: {path}: {error}") from error
        if data.get("schema") != LOCK_SCHEMA:
            raise FetchError(f"fetch lock must use schema {LOCK_SCHEMA}: {path}")
        entries = data.get("files", {})
        if not isinstance(entries, dict):
            raise FetchError("fetch lock 'files' must be an object")
        return cls(path, entries)

    def expected(self, name: str) -> str | None:
        entry = self.entries.get(name)
        return entry.get("sha256") if isinstance(entry, dict) else None

    def record(self, name: str, url: str, sha256: str) -> None:
        self.entries[name] = {"url": url, "sha256": sha256}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"schema": LOCK_SCHEMA, "files": dict(sorted(self.entries.items()))},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )


#: How bytes are fetched. Replaceable, and that is the point: some Pythons are
#: shipped without a working ssl module, or with the network kept away from
#: the interpreter while the shell beside it can still reach out - a code
#: editor on a tablet, typically. Nothing here may start a subprocess, because
#: the runtimes that need this are the same ones that forbid one. So a caller
#: outside this package - get-py2bin.py, which may - sets this to something
#: that shells out to curl or wget, and everything below goes through it.
#:
#: A replacement takes (url, label) and answers the bytes. The HTTPS check
#: below still applies: it happens before this is called.
DOWNLOADER = None


def _open_https(url: str, label: str):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise FetchError(f"{label} URL must be credential-free HTTPS: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    stream = urllib.request.urlopen(request, timeout=60)
    if urllib.parse.urlsplit(stream.geturl()).scheme != "https":
        stream.close()
        raise FetchError(f"{label} URL redirected outside HTTPS: {url}")
    return stream


def _open_stream(url: str, label: str):
    """Something to read the response from, whichever downloader is in use.

    A replacement answers all the bytes at once, so they are wrapped to look
    like the stream the caller below reads in chunks. The hash is still
    computed from what actually arrives, so a replacement cannot weaken the
    check that the download is what the index said it would be.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise FetchError(f"{label} URL must be credential-free HTTPS: {url}")
    if DOWNLOADER is not None:
        import io

        payload = DOWNLOADER(url, label)
        if not isinstance(payload, (bytes, bytearray)):
            raise FetchError(f"the installed downloader returned no bytes for {label}")
        return io.BytesIO(bytes(payload))
    return _open_https(url, label)


def _read_bytes(url: str, label: str, limit: int) -> bytes:
    """Read a whole response, through the replacement if one was installed."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise FetchError(f"{label} URL must be credential-free HTTPS: {url}")
    if DOWNLOADER is not None:
        payload = DOWNLOADER(url, label)
        if not isinstance(payload, (bytes, bytearray)):
            raise FetchError(f"the installed downloader returned no bytes for {label}")
        return bytes(payload)
    with _open_https(url, label) as stream:
        return stream.read(limit)


def _read_json(url: str, label: str) -> dict:
    payload = _read_bytes(url, label, 32 * 1024 * 1024)
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FetchError(f"{label} returned invalid JSON: {url}") from error
    if not isinstance(data, dict):
        raise FetchError(f"{label} returned an unexpected document: {url}")
    return data


def download_verified(
    url: str,
    cache: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    max_download: int = DEFAULT_MAX_DOWNLOAD,
) -> tuple[Path, str]:
    """Download ``url`` into the content-addressed ``cache``.

    Returns the cached path and its SHA-256. When ``expected_sha256`` is given
    the download must match it, otherwise the digest is computed and returned
    so the caller can record it.
    """

    cache.mkdir(parents=True, exist_ok=True)
    if expected_sha256 is not None:
        cached = cache / f"{expected_sha256}.blob"
        if cached.is_file() and _hash_file(cached) == expected_sha256:
            return cached, expected_sha256

    stream = _open_stream(url, label)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="py2bin-fetch-", dir=cache, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_download:
                    raise FetchError(
                        f"{label} exceeds the {max_download}-byte download limit: {url}"
                    )
                digest.update(chunk)
                temporary.write(chunk)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    finally:
        stream.close()

    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        temporary_path.unlink(missing_ok=True)
        raise FetchError(
            f"{label} SHA-256 mismatch for {url}; "
            f"expected {expected_sha256}, received {actual}"
        )
    destination = cache / f"{actual}.blob"
    temporary_path.replace(destination)
    return destination, actual


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> PurePosixPath | None:
    if not name or name.startswith("/") or "\\" in name:
        return None
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in {"..", ""} for part in relative.parts):
        return None
    return relative


def extract_zip(archive_path: Path, destination: Path) -> int:
    """Extract a ZIP, rejecting traversal paths, links, and special files."""

    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > DEFAULT_MAX_MEMBERS:
            raise FetchError(f"archive contains too many members: {archive_path}")
        for member in members:
            if member.is_dir():
                continue
            relative = _safe_member(member.filename)
            if relative is None:
                raise FetchError(
                    f"archive member escapes its root: {member.filename!r}"
                )
            unix_mode = (member.external_attr >> 16) & 0xFFFF
            if unix_mode and not stat.S_ISREG(unix_mode) and unix_mode & 0xF000:
                raise FetchError(
                    f"archive member is not a regular file: {member.filename!r}"
                )
            target = destination / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            written += 1
    return written


# --- Windows embeddable CPython ---------------------------------------------


def fetch_windows_runtime(
    version: str,
    target: str,
    output: Path,
    *,
    cache: Path,
    lock: FetchLock | None = None,
    clean: bool = False,
) -> RuntimePackInfo:
    """Download python.org's Windows embeddable CPython as a runtime pack."""

    if target not in _EMBED_ARCH:
        raise FetchError(
            "an embeddable CPython runtime is published for "
            f"{', '.join(sorted(_EMBED_ARCH))}, not {target!r}"
        )
    if not _VERSION.fullmatch(version):
        raise FetchError(f"runtime version must look like 3.12.9, not {version!r}")
    output = output.expanduser().resolve()
    if output.exists() and not clean:
        raise FetchError(f"output already exists: {output} (use --clean)")

    url = _EMBED_URL.format(version=version, arch=_EMBED_ARCH[target])
    name = f"cpython-{version}-{target}"
    lock = lock if lock is not None else FetchLock()
    archive, digest = download_verified(
        url,
        cache,
        label="Windows embeddable runtime",
        expected_sha256=lock.expected(name),
    )
    lock.record(name, url, digest)

    if output.exists():
        import shutil

        shutil.rmtree(output) if output.is_dir() else output.unlink()
    runtime_root = output / "runtime"
    extract_zip(archive, runtime_root)

    executable = runtime_root / "python.exe"
    if not executable.is_file():
        raise FetchError(
            f"embeddable runtime does not contain python.exe: {url}"
        )
    # The embeddable distribution ships its standard library as pythonXY.zip,
    # which py2bin's isolated-runtime path already understands.
    info = write_runtime_manifest(
        output,
        target=target,
        python=version,
        executable=executable.relative_to(output),
        environment={"PYTHONHOME": "runtime"},
    )
    lock.save()
    return info


# --- wheels ------------------------------------------------------------------


def _wheel_tags(filename: str) -> tuple[str, str, list[str], list[str], list[str]]:
    stem = filename[: -len(".whl")]
    parts = stem.split("-")
    if len(parts) < 5:
        raise FetchError(f"malformed wheel filename: {filename}")
    name, version = parts[0], parts[1]
    python_tags, abi_tags, platform_tags = parts[-3], parts[-2], parts[-1]
    return (
        name,
        version,
        python_tags.split("."),
        abi_tags.split("."),
        platform_tags.split("."),
    )


def wheel_is_compatible(filename: str, target: str, python_version: str) -> bool:
    """Return whether a wheel filename can be installed for the target."""

    try:
        _name, _version, python_tags, abi_tags, platform_tags = _wheel_tags(filename)
    except FetchError:
        return False
    major, _, minor = python_version.partition(".")
    cp = f"cp{major}{minor}"
    target_version = (int(major), int(minor))
    # A free-threaded build (for example cp313t) needs a matching interpreter.
    if any(tag.startswith("cp") and tag.endswith("t") for tag in abi_tags):
        return False
    if not any(tag in {"none", "abi3", cp} for tag in abi_tags):
        return False
    stable_abi = "abi3" in abi_tags

    def python_tag_ok(tag: str) -> bool:
        if tag in {cp, f"py{major}{minor}", f"py{major}"}:
            return True
        if stable_abi and tag.startswith("cp"):
            # The stable ABI is forward compatible: a cp37-abi3 wheel loads on
            # every later CPython 3.x, so accept any tag at or below the target.
            digits = tag[2:]
            if digits.isdigit() and len(digits) >= 2:
                built = (int(digits[0]), int(digits[1:]))
                return built <= target_version
        return False

    if not any(python_tag_ok(tag) for tag in python_tags):
        return False
    families = _PLATFORM_OS.get(target, ())
    suffixes = _PLATFORM_RULES.get(target, ())

    def platform_ok(tag: str) -> bool:
        if tag == "any":
            return True  # no native code, so any machine can run it
        if not any(tag.startswith(family) for family in families):
            return False
        # The OS matches; now the architecture must too.
        return any(tag.endswith(suffix) or tag == suffix for suffix in suffixes)

    return any(platform_ok(tag) for tag in platform_tags)


def _abi_rank(filename: str, python_version: str) -> tuple[int, int]:
    """Prefer an exact cp-ABI wheel, then abi3, then pure Python."""

    _n, _v, _py, abi_tags, platform_tags = _wheel_tags(filename)
    major, _, minor = python_version.partition(".")
    cp = f"cp{major}{minor}"
    if cp in abi_tags:
        abi_score = 0
    elif "abi3" in abi_tags:
        abi_score = 1
    else:
        abi_score = 2
    platform_score = 1 if platform_tags == ["any"] else 0
    return abi_score, platform_score


def select_wheel(
    files: list[dict], target: str, python_version: str
) -> dict | None:
    candidates = [
        item
        for item in files
        if item.get("packagetype") == "bdist_wheel"
        and not item.get("yanked")
        and isinstance(item.get("filename"), str)
        and wheel_is_compatible(item["filename"], target, python_version)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: _abi_rank(item["filename"], python_version))
    return candidates[0]


#: Portable CPython builds for macOS, which python.org does not publish - it
#: ships an installer, and unpacking one needs the installer. These are plain
#: archives of a prefix, with a checksum file beside them.
_STANDALONE_LATEST = (
    "https://api.github.com/repos/astral-sh/python-build-standalone"
    "/releases/latest"
)
_STANDALONE_ARCH = {"darwin-arm64": "aarch64", "darwin-x86_64": "x86_64"}


def fetch_macos_runtime(
    version: str, target: str, output: Path, *, cache: Path
) -> Path:
    """Download a macOS CPython and answer the shared library inside it.

    A macOS bundle links against a macOS libpython, and only a Mac has one -
    which stops a bundle being built anywhere else, including on the tablet
    someone is writing the program on. It does not have to: portable builds
    of exactly this are published, so the machine doing the building needs to
    be no particular machine at all.
    """
    architecture = _STANDALONE_ARCH.get(target)
    if architecture is None:
        raise FetchError(f"no portable CPython is published for {target}")
    wanted = f"{version.split('.')[0]}.{version.split('.')[1]}"

    release = _read_json(_STANDALONE_LATEST, "the portable CPython index")
    assets = {item.get("name"): item for item in release.get("assets") or []}
    pattern = re.compile(
        rf"^cpython-({re.escape(wanted)}\.\d+)\+\d+-"
        rf"{architecture}-apple-darwin-install_only\.tar\.gz$"
    )
    matches = sorted(name for name in assets if name and pattern.match(name))
    if not matches:
        raise FetchError(
            f"the portable CPython release {release.get('tag_name', '?')} has "
            f"no {wanted} build for {target}"
        )
    name = matches[-1]

    # The checksums are published beside the archives, so what arrives can be
    # held to what was published rather than merely to what was served.
    digest = None
    if "SHA256SUMS" in assets:
        listing = _read_bytes(
            assets["SHA256SUMS"]["browser_download_url"],
            "the portable CPython checksums",
            8 * 1024 * 1024,
        ).decode("utf-8", "replace")
        for line in listing.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == name:
                digest = parts[0]
                break
    if digest is None:
        raise FetchError(f"no published checksum for {name}")

    archive, _got = download_verified(
        assets[name]["browser_download_url"],
        cache,
        expected_sha256=digest,
        label=f"portable CPython {wanted} for {target}",
    )
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            place = (output / member.name).resolve()
            if not str(place).startswith(str(output.resolve())):
                raise FetchError(f"the archive writes outside its directory: {member.name}")
        bundle.extractall(output)
    found = sorted(output.rglob("libpython*.dylib"))
    if not found:
        raise FetchError(f"no libpython in the archive {name}")
    return found[0]


def _version_key(text: str):
    """Order releases newest-first, well enough to pick a usable one.

    Not a full PEP 440 implementation - it does not need to be. It has to put
    2.6.1 after 2.6.0 and both after 1.9, and treat anything it cannot read as
    older, so a pre-release never wins over a plain one.
    """
    parts = []
    for piece in text.split("."):
        digits = ""
        for character in piece:
            if not character.isdigit():
                break
            digits += character
        parts.append((int(digits) if digits else -1, piece))
    return parts


def _earlier_releases(document: dict):
    """Every release this project has published, newest first."""
    releases = document.get("releases")
    if not isinstance(releases, dict):
        return []
    ordered = sorted(releases, key=_version_key, reverse=True)
    return [
        (name, releases[name])
        for name in ordered
        if isinstance(releases.get(name), list) and releases[name]
    ]


def fetch_wheel(
    project: str,
    target: str,
    python_version: str,
    output: Path,
    *,
    cache: Path,
    version: str | None = None,
    lock: FetchLock | None = None,
) -> FetchedFile:
    """Download one target-compatible wheel for ``project`` from PyPI."""

    if not _PROJECT_NAME.fullmatch(project):
        raise FetchError(f"invalid PyPI project name: {project!r}")
    url = (
        _PYPI_RELEASE.format(project=project, version=version)
        if version
        else _PYPI_PROJECT.format(project=project)
    )
    document = _read_json(url, f"PyPI metadata for {project}")
    files = document.get("urls") or []
    if not isinstance(files, list):
        raise FetchError(f"PyPI returned unexpected files for {project}")
    chosen = select_wheel(files, target, python_version)
    release = document.get("info", {}).get("version", "?")
    if chosen is None and version is None:
        # The newest release has nothing for this target, which is ordinary
        # soon after a Python release: a project builds wheels for the
        # interpreters that existed when it was published. An older release
        # often has one, and an older release of the right shape is far more
        # use than a build that stops.
        older = _earlier_releases(document)
        for candidate, candidate_files in older:
            found = select_wheel(candidate_files, target, python_version)
            if found is not None:
                chosen, release = found, candidate
                break
    if chosen is None:
        only_sdist = bool(files) and all(
            item.get("packagetype") == "sdist" for item in files
        )
        if only_sdist:
            raise FetchError(
                f"{project} {release} publishes only a source distribution, and "
                "py2bin does not execute setup.py or a build backend. Build the "
                "wheel once (py2bin wheel works for an already-staged tree) and "
                "pass it with --wheel/--wheel-dir"
            )
        raise FetchError(
            f"PyPI has no {target} wheel for {project} {release} on Python "
            f"{python_version}; supply one with --wheel/--wheel-dir"
        )
    filename = chosen["filename"]
    expected = chosen.get("digests", {}).get("sha256")
    lock = lock if lock is not None else FetchLock()
    recorded = lock.expected(filename)
    if recorded is not None and expected is not None and recorded != expected:
        raise FetchError(
            f"PyPI now serves a different {filename}; locked {recorded}, "
            f"PyPI reports {expected}"
        )
    blob, digest = download_verified(
        chosen["url"],
        cache,
        label=f"wheel {filename}",
        expected_sha256=recorded or expected,
    )
    lock.record(filename, chosen["url"], digest)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    destination.write_bytes(blob.read_bytes())
    return FetchedFile(filename, chosen["url"], digest, destination)
