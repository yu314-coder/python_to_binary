"""Find a header py2bin cannot find on this machine, and bring it here.

py2bin ships the C standard headers and its own C++ ones, and it has no
system include path - that is deliberate, and it is what makes a build the
same on every machine. What it leaves out is a header that belongs to
somebody else: a vendor SDK, a small single-file library, whatever the
program was written against. Those are on the web, and this fetches one.

Two ways in, and both are the author's decision rather than something that
happens on its own:

  * a name that is looked up in a package index and downloaded, which is what
    `--auto-fetch` does when a header is missing;
  * a URL given outright, for a header that is not in any index.

Nothing here runs on its own. A build without `--auto-fetch` and without a
URL never reaches the network, which is what keeps `py2bin cc` a thing you
can run on a machine that has none.

What arrives is a header, not a toolchain. A header written in one
compiler's extensions still will not compile here - the fetch puts the file
where the preprocessor can see it, and whether py2bin's C understands what is
in it is a separate question that the compiler answers in the usual way.
"""

from __future__ import annotations

import json
import re
import struct
import zipfile
import urllib.parse
from pathlib import Path

from .runtime_fetch import (
    FetchError,
    _read_bytes,
    _read_json,
    download_verified,
    extract_zip,
    read_range,
)

__all__ = [
    "HeaderFetchError",
    "components_fetched",
    "found_headers",
    "fetch_header",
    "fetch_header_from",
    "fetch_library",
    "libraries_offered",
    "search_index",
]


class HeaderFetchError(FetchError):
    """A header could not be found or could not be brought here."""


#: The two places a header is looked for, in this order. Both answer JSON
#: over HTTPS and need no account, which is what makes them usable from a
#: build: no token to keep and nothing to log in to.
#:
#:  * a package index, which is where a vendor SDK is published - the
#:    WebView2 headers, a platform's own, anything shipped as a package;
#:  * a source host, which is where a header-only C++ library lives, and
#:    those are usually not packaged anywhere at all.
_INDEX_QUERY = "https://azuresearch-usnc.nuget.org/query"
_INDEX_PACKAGE = "https://api.nuget.org/v3-flatcontainer"
_SOURCE_QUERY = "https://api.github.com/search/repositories"
_SOURCE_TREE = "https://api.github.com/repos"
_SOURCE_RAW = "https://raw.githubusercontent.com"

#: A ceiling on what one repository's headers may come to.
_MAX_HEADERS = 600
_MAX_HEADER_BYTES = 64 * 1024 * 1024

#: Repositories that hold a whole *set* of headers rather than one library's.
#: A platform header - `rpc.h`, `objbase.h`, `unknwn.h` - is never published
#: on its own and is never in a repository named after it, so searching by
#: name never finds one. These are asked by path instead: the file list of a
#: repository arrives in a single request, and the header is looked for in it.
#:
#: This one is the header set that a program written against the Windows API
#: is written against. It is headers only - no compiler, no library, nothing
#: py2bin runs - and it is what makes `#include <windows.h>`-shaped code
#: reachable at all, since the vendor's own set ships with a toolchain and is
#: published nowhere a build can fetch it from.
#: Two of them, because neither is complete on its own: each generates part
#: of its set from templates its own build system fills in, and what one
#: generates the other often ships. Tried in order, and the one that answered
#: last is tried first next time - the headers of a set belong together, and
#: a program that needed one usually needs its neighbours.
#: In this order because the closures were walked and counted. Neither set is
#: complete as published - each generates part of itself at build time - but
#: they are incomplete in different places. Taking `rpc.h` from each:
#:
#:   the reimplementation  80 headers, 11 it cannot resolve, every one of
#:                         them a COM header generated from a `.idl`;
#:   the runtime package  144 headers, 13 it cannot resolve, two of which are
#:                         its own core header and its unicode header - and
#:                         *every* header in that set includes those at the
#:                         top, unconditionally, so nothing from it compiles
#:                         without a file its configure step writes.
#:
#: So the first is tried first: what it is missing is reached only by the
#: part of a program that uses COM, and what the second is missing is reached
#: by everything. A set that does not hold the header at all falls through.
_COLLECTIONS = ["wine-mirror/wine", "mingw-w64/mingw-w64"]

#: File lists already fetched, by (repository, branch). One of these is a
#: single request answering thousands of paths, and a build that fetches a
#: dozen headers from one set would otherwise ask a dozen times - which is
#: most of an anonymous caller's hourly allowance.
_TREES: "dict[tuple[str, str], list[str]]" = {}

#: How many results a search looks at. A header that is not in the first few
#: packages named after it is not going to be in the twentieth.
_RESULTS = 8

#: A ceiling on what one fetch will pull down. A package holding a header is
#: a few megabytes; one that is not is not the package that was wanted.
_MAX_PACKAGE = 96 * 1024 * 1024

#: Where a fetched header is kept, beside the program that asked for it. A
#: directory rather than the program's own, so that what was downloaded stays
#: told apart from what was written.
CACHE_DIRECTORY = ".py2bin-headers"


#: What a candidate that did not answer raises. `FetchError` is py2bin's own
#: way of saying "not here"; the rest are how the network and the archive
#: reader say it. A guessed repository that does not exist answers 404, which
#: urllib raises as an HTTPError - an OSError, and not a FetchError - and a
#: package that downloads but is not an archive raises BadZipFile. Caught for
#: `FetchError` alone, the first candidate failing that way ended the whole
#: search, and a header one of the curated sets holds was reported missing.
_DID_NOT_ANSWER = (FetchError, OSError, zipfile.BadZipFile)


def _valid_name(name: str) -> str:
    """A header name that cannot escape the directory it is written into."""

    spelled = name.strip().replace("\\", "/")
    if not spelled or spelled.startswith("/") or ".." in spelled.split("/"):
        raise HeaderFetchError(f"{name!r} is not a header name that can be fetched")
    if not re.fullmatch(r"[A-Za-z0-9._+-]+(?:/[A-Za-z0-9._+-]+)*", spelled):
        raise HeaderFetchError(f"{name!r} is not a header name that can be fetched")
    return spelled


def found_headers(where: Path) -> "list[Path]":
    """Every header already fetched into `where`."""

    if not where.is_dir():
        return []
    return sorted(
        path
        for path in where.rglob("*")
        if path.is_file() and path.suffix.lower() in (".h", ".hpp", ".hxx", ".hh")
    )


def search_index(name: str, *, results: int = _RESULTS) -> "list[str]":
    """Package ids that might hold this header, most likely first.

    The search is on the header's own stem, because that is what a package
    holding it is nearly always called - `WebView2.h` lives in a package with
    `WebView2` in its name - and there is nothing else in the name to go on.
    """

    return [package for package, _version in _search_entries(name, results=results)]


def _search_entries(
    name: str, *, results: int = _RESULTS
) -> "list[tuple[str, str | None]]":
    """Package ids and the newest release of each, as one search answers them.

    The index says which version it is listing, and a search with
    pre-releases left out lists the newest release: for a caller that will
    go on to open the package, that spares asking the version list of every
    candidate. None where the index did not say."""

    stem = _valid_name(name).rsplit("/", 1)[-1]
    stem = re.sub(r"\.(h|hpp|hxx|hh)$", "", stem, flags=re.I)
    if not stem:
        raise HeaderFetchError(f"{name!r} has nothing in it to search for")
    query = urllib.parse.urlencode(
        {"q": stem, "take": str(max(1, results)), "prerelease": "false"}
    )
    answer = _read_json(f"{_INDEX_QUERY}?{query}", f"a search for {stem}")
    data = answer.get("data")
    if not isinstance(data, list):
        raise HeaderFetchError(f"the index answered nothing usable for {stem!r}")
    found: "list[tuple[str, str | None]]" = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        package = entry.get("id")
        if not isinstance(package, str) or not package:
            continue
        if any(package == seen for seen, _version in found):
            continue
        version = entry.get("version")
        if not isinstance(version, str) or not version or "-" in version:
            version = None
        found.append((package, version))
    return found


def _latest_version(package: str) -> str:
    """The newest published version of a package, as the index reports it."""

    lowered = urllib.parse.quote(package.lower(), safe="")
    answer = _read_json(
        f"{_INDEX_PACKAGE}/{lowered}/index.json", f"the versions of {package}"
    )
    versions = answer.get("versions")
    if not isinstance(versions, list) or not versions:
        raise HeaderFetchError(f"{package} has no published version")
    # Newest last, which is the order the index publishes them in. A
    # pre-release is skipped: a build should not quietly pick one up.
    for version in reversed(versions):
        if isinstance(version, str) and "-" not in version:
            return version
    raise HeaderFetchError(f"{package} has only pre-release versions")


def _package_url(package: str, version: str) -> str:
    lowered = urllib.parse.quote(package.lower(), safe="")
    spelled = urllib.parse.quote(version.lower(), safe="")
    return f"{_INDEX_PACKAGE}/{lowered}/{spelled}/{lowered}.{spelled}.nupkg"


def fetch_header(
    name: str,
    into: Path,
    *,
    cache: "Path | None" = None,
    results: int = _RESULTS,
    say=lambda message: None,
) -> Path:
    """Look `name` up in the index, download what holds it, and keep the header.

    Answers where the header was written. Every file the package holds that is
    a header is kept, not only the one asked for: a header includes its
    neighbours, and fetching one at a time would ask again immediately.
    """

    wanted = _valid_name(name)
    stem = wanted.rsplit("/", 1)[-1]
    if stem in _SUPPLIED:
        raise HeaderFetchError(
            f"{stem} is one of the headers py2bin ships, so there is nothing "
            f"to fetch. A published copy is written for a compiler that is "
            f"GCC or MSVC, and does not compile here"
        )
    into = Path(into)
    # Fetched before. Under the name it is included by first: `openssl/evp.h`
    # is kept as `openssl/evp.h`, and a check that looked only for the stem
    # missed it and asked the network again on every build.
    for already in (into / wanted, into / stem):
        if already.is_file():
            return already
    reasons: "list[str]" = []
    # Which index to ask first. A name with a directory in it, or one spelled
    # the way only C++ spells a header, belongs to a library published as
    # source; a bare `.h` is what a vendor SDK ships in a package. Asking the
    # likely one first is the difference between one download and eight.
    source_first = "/" in wanted or stem.lower().endswith((".hpp", ".hxx", ".hh"))
    if source_first:
        found = _from_source(
            wanted, into, results=results, say=say, reasons=reasons
        )
        if found is not None:
            return found
    try:
        packages = search_index(wanted, results=results)
    except _DID_NOT_ANSWER as error:
        packages = []
        reasons.append(f"the package index: {error}")
    for package in packages:
        try:
            version = _latest_version(package)
            say(f"  trying {package} {version}")
            found = _take_headers(
                _package_url(package, version),
                f"{package} {version}",
                into,
                stem,
                cache=cache,
            )
        except _DID_NOT_ANSWER as error:
            reasons.append(f"{package}: {error}")
            continue
        if found is not None:
            say(f"  {stem} came from the package {package} {version}")
            return found
        reasons.append(f"{package}: holds no {stem}")
    if not source_first:
        # Nothing packaged holds it, so look where the source is. A
        # header-only C++ library is usually published nowhere else.
        found = _from_source(
            wanted, into, results=results, say=say, reasons=reasons
        )
        if found is not None:
            return found
    spelled = "\n    ".join(reasons) or "nothing was found to try"
    raise HeaderFetchError(
        f"{stem!r} was not found:\n    {spelled}\n"
        f"  Some headers are not published as files at all - a COM or IDL one "
        f"is generated from a .idl by a tool that runs at build time, and a "
        f"platform's own set often has a core header its configure step "
        f"writes. Nothing can fetch those.\n"
        f"  Give the URL of the header itself, or put it in a directory and "
        f"name that with --include"
    )


def exports_of(image: bytes) -> "set[str]":
    """Every name a PE exports, read out of its own export table.

    py2bin writes these; reading one is the same layout in the other
    direction. This is what lets a build *check* which library holds the
    function a program calls rather than guess from a name - a package ships
    several DLLs and only one of them exports what is wanted.

    An image that is not a PE, or is one with no exports, answers with
    nothing rather than raising: it is one candidate among several.
    """

    try:
        if image[:2] != b"MZ":
            return set()
        at = struct.unpack_from("<I", image, 0x3C)[0]
        if image[at:at + 4] != b"PE\0\0":
            return set()
        sections = struct.unpack_from("<H", image, at + 6)[0]
        optional = struct.unpack_from("<H", image, at + 20)[0]
        head = at + 24
        magic = struct.unpack_from("<H", image, head)[0]
        # The data directories sit after the optional header's fixed part,
        # which is eight bytes longer in a 64-bit image than in a 32-bit one.
        directories = head + (112 if magic == 0x20B else 96)
        rva, size = struct.unpack_from("<II", image, directories)
        if not rva or not size:
            return set()
        table = []
        for index in range(sections):
            spot = head + optional + index * 40
            start, where, raw, offset = struct.unpack_from("<IIII", image, spot + 8)
            table.append((where, max(start, raw), offset))

        def at_rva(wanted: int) -> "int | None":
            for where, span, offset in table:
                if where <= wanted < where + span:
                    return offset + (wanted - where)
            return None

        base = at_rva(rva)
        if base is None:
            return set()
        # NumberOfNames at 24, AddressOfNames at 32. Between them sits
        # AddressOfFunctions, which is the ordinal table and not this one.
        count = struct.unpack_from("<I", image, base + 24)[0]
        names_rva = struct.unpack_from("<I", image, base + 32)[0]
        walk = at_rva(names_rva)
        if walk is None:
            return set()
        found: "set[str]" = set()
        for index in range(min(count, _MOST_EXPORTS)):
            spot = at_rva(struct.unpack_from("<I", image, walk + index * 4)[0])
            if spot is None:
                continue
            end = image.index(b"\0", spot)
            found.add(image[spot:end].decode("ascii", "replace"))
        return found
    except (struct.error, ValueError, IndexError):
        return set()


#: A guard on a malformed image, not a real limit: a DLL with more exports
#: than this is not one anybody links against by name.
_MOST_EXPORTS = 200000


#: Where a package keeps the machine-code half of what it ships, by the
#: architecture a target names. NuGet's own conventions, in the order a
#: package is likely to use them.
_RUNTIME_FOLDERS = {
    "x86_64": ("runtimes/win-x64/native/", "build/native/x64/", "/x64/"),
    "arm64": ("runtimes/win-arm64/native/", "build/native/arm64/", "/arm64/"),
}


def library_exporting(
    symbol: str,
    into: Path,
    architecture: str,
    *,
    cache: "Path | None" = None,
    say=lambda message: None,
) -> "Path | None":
    """The library that exports `symbol`, kept in `into`. None if none does.

    A program calling into somebody else's component names the function and
    not the library it lives in - the header declares it and a build with a
    linker is handed the import library separately. What py2bin has instead
    is the package the header came from: it holds the library too, and the
    library says what it exports. So the question is answered by looking
    rather than by guessing from a name, and the answer is either right or
    absent.
    """

    into = Path(into)
    folders = _RUNTIME_FOLDERS.get(architecture)
    if folders is None:
        return None
    store = Path(cache) if cache is not None else into / ".cache"
    if not store.is_dir():
        return None
    for archive in sorted(store.glob("*.blob")):
        found = _exporter_in(archive, into, symbol, folders)
        if found is not None:
            say(f"  {symbol} is exported by {found.name}, which came with it")
            return found
    return None


def _exporter_in(
    archive: Path, into: Path, symbol: str, folders: "tuple[str, ...]"
) -> "Path | None":
    """Which library in one package exports `symbol`, for this machine."""

    import tempfile
    import zipfile

    try:
        with zipfile.ZipFile(archive) as held:
            names = [
                item.filename
                for item in held.infolist()
                if item.filename.lower().endswith(".dll")
                and any(f.lower() in item.filename.lower() for f in folders)
            ]
            if not names:
                return None
            for spelled in names:
                if symbol not in exports_of(held.read(spelled)):
                    continue
                into.mkdir(parents=True, exist_ok=True)
                written = into / spelled.rsplit("/", 1)[-1]
                written.write_bytes(held.read(spelled))
                return written
    except (OSError, zipfile.BadZipFile, KeyError):
        return None
    return None


def fetch_library(
    name: str,
    into: Path,
    architecture: str,
    *,
    cache: "Path | None" = None,
    results: int = _RESULTS,
    components: "tuple[str, ...]" = (),
    say=lambda message: None,
) -> Path:
    """Download the shared library `name` and keep the copy for this machine.

    A program built against somebody else's component needs two things from
    them: the header, to compile, and the library itself, to run. py2bin
    fetches the header already; this is the other half, and without it the
    binary that comes out cannot start on a machine that has not installed
    the component separately - which is the whole difficulty on a machine
    that cannot install anything.

    Only what the program asked for by name. A package holds a great many
    files and nearly all of them are somebody else's business; the one
    matching the name given, for the architecture being built, is the one
    that was asked for.

    ``components`` is what the fetched headers are about - `openssl` for a
    program that included `openssl/evp.h` - and is where the index is asked
    next when a search for the file's own name finds nothing: a package is
    named after the component and not after the file in it, and OpenSSL's
    `libcrypto-3-x64.dll` is in no package called that. Only the file named
    is taken from whatever the component's packages turn out to be: the
    name carries the version the program was written against, so the
    package that ships a different version of the component ships a file of
    a different name and is passed over.
    """

    stem = _valid_name(name).rsplit("/", 1)[-1]
    if not stem.lower().endswith(".dll"):
        raise HeaderFetchError(
            f"{stem!r} is not a shared library name that can be fetched"
        )
    into = Path(into)
    already = into / stem
    if already.is_file():
        return already
    wanted = _RUNTIME_FOLDERS.get(architecture)
    if wanted is None:
        raise HeaderFetchError(
            f"nothing is known about where a package keeps a library for "
            f"{architecture!r}"
        )
    reasons: "list[str]" = []
    # What was downloaded for the header first, and without asking the
    # network again. A component ships its header and its library in one
    # package - fetching the header brought the library with it - and the
    # package is named after the component rather than after the file, so
    # searching an index for `WebView2Loader` finds everything except the
    # package that holds it.
    store = Path(cache) if cache is not None else into / ".cache"
    for archive in sorted(store.glob("*.blob")) if store.is_dir() else ():
        found = _library_from(archive, into, stem, wanted)
        if found is not None:
            say(f"  {stem} came with the package the header came from")
            return found
    try:
        packages = search_index(stem, results=results)
    except _DID_NOT_ANSWER as error:
        packages = []
        reasons.append(f"the package index: {error}")
    for package in packages:
        try:
            version = _latest_version(package)
            say(f"  trying {package} {version}")
            found = _take_library(
                _package_url(package, version),
                f"{package} {version}",
                into,
                stem,
                wanted,
                cache=cache,
            )
        except _DID_NOT_ANSWER as error:
            reasons.append(f"{package}: {error}")
            continue
        if found is not None:
            say(f"  {stem} came from the package {package} {version}")
            return found
        reasons.append(f"{package}: holds no {stem} for {architecture}")
    for component in components:
        found = _library_of_component(
            component, stem, into, wanted, architecture, cache=cache, say=say,
            reasons=reasons,
        )
        if found is not None:
            return found
    spelled = "\n    ".join(reasons) or "nothing was found to try"
    raise HeaderFetchError(
        f"{stem!r} was not found:\n    {spelled}\n"
        f"  The program names it with --library, so it has to be beside the "
        f"binary to run. Put a copy there yourself, or install the component "
        f"on the machine that will run it"
    )


#: How many of a component's packages are looked at. A component that is
#: also a .NET namespace - OpenSSL is - has dozens of managed packages named
#: after it ahead of the one that ships the native library, and each is
#: turned over by reading its directory alone, so looking at many is cheap.
_COMPONENT_RESULTS = 40


def _library_of_component(
    component: str,
    stem: str,
    into: Path,
    folders: "tuple[str, ...]",
    architecture: str,
    *,
    cache: "Path | None" = None,
    say=lambda message: None,
    reasons: "list[str] | None" = None,
) -> "Path | None":
    """The library `stem` out of a package named after `component`, if one
    of the first few the index lists ships it for this machine.

    Each candidate is asked what it holds before anything is downloaded -
    a zip says so in its directory, which sits at its end - and only a
    package that lists the file has the file taken out of it, by that
    member's own byte range. The package OpenSSL ships its Windows
    libraries in is 136MB of every build flavour of every architecture;
    the one library wanted is 8MB of it.
    """

    reasons = reasons if reasons is not None else []
    try:
        entries = _search_entries(component, results=_COMPONENT_RESULTS)
    except _DID_NOT_ANSWER as error:
        reasons.append(f"the package index, asked for {component}: {error}")
        return None
    say(f"  looking through the {len(entries)} packages named after {component}")
    for package, version in entries:
        try:
            if version is None:
                version = _latest_version(package)
            label = f"{package} {version}"
            url = _package_url(package, version)
            members = _members_of(url, label)
            if members is None:
                # A server that answers no range, or a directory too big for
                # the format this reads: the whole package, then, within the
                # ceiling any fetch has.
                say(f"  trying {label}")
                found = _take_library(url, label, into, stem, folders, cache=cache)
                if found is not None:
                    say(f"  {stem} came from the package {label}")
                    return found
                continue
            member = _member_named(members, stem, folders)
            if member is None:
                continue
            say(f"  trying {label}")
            into.mkdir(parents=True, exist_ok=True)
            written = into / stem
            written.write_bytes(_member_bytes(url, label, member, _MAX_PACKAGE))
        except _DID_NOT_ANSWER as error:
            reasons.append(f"{package}: {error}")
            continue
        say(f"  {stem} came from the package {label}")
        return written
    reasons.append(
        f"none of the {len(entries)} packages named after {component} holds "
        f"{stem} for {architecture}"
    )
    return None


def libraries_offered(
    component: str, architecture: str, *, say=lambda message: None
) -> "list[tuple[str, list[str]]]":
    """Which shared libraries the packages named after `component` ship for
    this machine: (package and version, library names), in the index's order.

    For the message that says a library has to be named: it is a list of
    what could be named, read off each package's directory and nothing
    more. Which of them matches the headers the program compiled against is
    the program's author's to say - the versions differ, and a library of
    one version behind the headers of another is a program that starts and
    computes something else.
    """

    folders = _RUNTIME_FOLDERS.get(architecture)
    if folders is None:
        return []
    offered: "list[tuple[str, list[str]]]" = []
    try:
        entries = _search_entries(component, results=_COMPONENT_RESULTS)
    except _DID_NOT_ANSWER:
        return offered
    say(f"  looking through the {len(entries)} packages named after {component}")
    for package, version in entries:
        try:
            if version is None:
                version = _latest_version(package)
            members = _members_of(_package_url(package, version), f"{package} {version}")
        except _DID_NOT_ANSWER:
            continue
        if not members:
            continue
        names: "list[str]" = []
        for member in members:
            name = member[0].rsplit("/", 1)[-1]
            if not name.lower().endswith(".dll") or name in names:
                continue
            if any(folder.lower() in member[0].lower() for folder in folders):
                names.append(name)
        if names:
            offered.append((f"{package} {version}", names))
    return offered


#: How much of the end of an archive is read to find its directory. The
#: end-of-directory record is 22 bytes plus a comment of up to 64KB, so
#: this many bytes hold it whatever the comment.
_ZIP_TAIL = 64 * 1024 + 22

#: A zip member as its central directory lists it: name, method, compressed
#: size, size, CRC-32, and the offset of its local header.
_Member = "tuple[str, int, int, int, int, int]"


def _members_of(url: str, label: str) -> "list[_Member] | None":
    """The directory of the zip at `url`, read without the rest of it.

    The directory sits at the end of a zip, so the end is what is asked for.
    None where the server answered the whole file rather than the end of it
    - nothing of it is read then - or where the archive is written in the
    64-bit form this does not read, and the caller falls back to a whole
    download with the ceiling any download has.
    """

    answered = read_range(url, label, -_ZIP_TAIL)
    if answered is None:
        return None
    tail, size = answered
    at = tail.rfind(b"PK\x05\x06")
    if at < 0 or at + 22 > len(tail):
        raise FetchError(f"{label} is not a zip archive: {url}")
    entries, directory_size, directory_at = struct.unpack_from("<HII", tail, at + 10)
    if entries == 0xFFFF or directory_size == 0xFFFFFFFF or directory_at == 0xFFFFFFFF:
        return None
    tail_at = size - len(tail)
    if directory_at >= tail_at:
        directory = tail[directory_at - tail_at : directory_at - tail_at + directory_size]
    else:
        answered = read_range(url, label, directory_at, directory_at + directory_size)
        if answered is None:
            return None
        directory = answered[0]
    members: "list[_Member]" = []
    at = 0
    while at + 46 <= len(directory) and directory[at : at + 4] == b"PK\x01\x02":
        method = struct.unpack_from("<H", directory, at + 10)[0]
        crc, compressed, expanded = struct.unpack_from("<III", directory, at + 16)
        name_length, extra_length, comment_length = struct.unpack_from(
            "<HHH", directory, at + 28
        )
        local_at = struct.unpack_from("<I", directory, at + 42)[0]
        name = directory[at + 46 : at + 46 + name_length].decode("utf-8", "replace")
        members.append((name, method, compressed, expanded, crc, local_at))
        at += 46 + name_length + extra_length + comment_length
    if len(members) != entries:
        raise FetchError(f"{label} has a directory that does not add up: {url}")
    return members


def _member_named(
    members: "list[_Member]", stem: str, folders: "tuple[str, ...]"
) -> "_Member | None":
    """The member that is the library `stem` for this machine, if listed:
    the same rule ``_library_from`` applies to an extracted package."""

    named = [
        member
        for member in members
        if member[0].rsplit("/", 1)[-1].lower() == stem.lower()
    ]
    for folder in folders:
        for member in named:
            if folder.lower() in member[0].lower():
                return member
    return None


def _member_bytes(url: str, label: str, member: "_Member", limit: int) -> bytes:
    """One member of the zip at `url`, by its own byte range, inflated here.

    The local header repeats the name and may carry a different extra
    field, so the range asked for allows for the longest one and the data
    is found from the lengths the header itself states. The CRC-32 the
    directory recorded is checked against what came out: a range answered
    from the wrong bytes is a library that loads and then does something
    else.
    """

    import zlib

    name, method, compressed, expanded, crc, local_at = member
    if expanded > limit or compressed > limit:
        raise FetchError(f"{name} in {label} exceeds the {limit}-byte download limit")
    if method not in (0, 8):
        raise FetchError(f"{name} in {label} is stored in a way this cannot read")
    most = local_at + 30 + len(name.encode("utf-8")) + 0xFFFF + compressed
    answered = read_range(url, label, local_at, most)
    if answered is None:
        raise FetchError(f"{label} stopped answering ranges: {url}")
    held = answered[0]
    if held[:4] != b"PK\x03\x04":
        raise FetchError(f"{name} in {label} is not where its directory says")
    name_length, extra_length = struct.unpack_from("<HH", held, 26)
    start = 30 + name_length + extra_length
    data = held[start : start + compressed]
    if len(data) != compressed:
        raise FetchError(f"{name} in {label} is shorter than its directory says")
    if method == 8:
        try:
            inflater = zlib.decompressobj(-15)
            data = inflater.decompress(data, expanded) + inflater.flush()
        except zlib.error as error:
            raise FetchError(f"{name} in {label} did not inflate: {error}") from error
    if len(data) != expanded or (zlib.crc32(data) & 0xFFFFFFFF) != crc:
        raise FetchError(f"{name} in {label} did not come out as its directory says")
    return data


def components_fetched(into: Path) -> "list[str]":
    """What the headers kept in `into` are about, as the includes spelled it.

    A header included through a directory - `openssl/evp.h` - belongs to
    the component the directory is named after, and a bare one - `zlib.h` -
    to the one its own name is. That is how a package index knows the
    component too, so these are the names to ask it for.
    """

    into = Path(into)
    if not into.is_dir():
        return []
    found: "list[str]" = []
    for path in sorted(into.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            found.append(path.name)
        elif path.suffix.lower() in (".h", ".hpp", ".hxx", ".hh"):
            found.append(path.stem)
    return found


def _take_library(
    url: str,
    label: str,
    into: Path,
    stem: str,
    folders: "tuple[str, ...]",
    *,
    cache: "Path | None" = None,
) -> "Path | None":
    """Download one package and keep the named library for this machine."""

    into.mkdir(parents=True, exist_ok=True)
    store = Path(cache) if cache is not None else into / ".cache"
    archive, _digest = download_verified(
        url, store, label=label, max_download=_MAX_PACKAGE
    )
    return _library_from(archive, into, stem, folders)


def _library_from(
    archive: Path, into: Path, stem: str, folders: "tuple[str, ...]"
) -> "Path | None":
    """The named library for this machine, out of one downloaded package."""

    import tempfile
    import zipfile

    try:
        with zipfile.ZipFile(archive) as held:
            if not any(
                item.filename.rsplit("/", 1)[-1].lower() == stem.lower()
                for item in held.infolist()
            ):
                return None
    except (OSError, zipfile.BadZipFile):
        return None
    with tempfile.TemporaryDirectory(dir=str(into)) as work:
        opened = Path(work)
        extract_zip(archive, opened)
        candidates = [
            path
            for path in opened.rglob("*")
            if path.is_file() and path.name.lower() == stem.lower()
        ]
        for folder in folders:
            for path in candidates:
                if folder.lower() in path.as_posix().lower():
                    written = into / stem
                    written.write_bytes(path.read_bytes())
                    return written
        return None


def search_source(name: str, *, results: int = _RESULTS) -> "list[str]":
    """Repositories that might hold this header, most starred first.

    The words of the name are the query: `nlohmann/json.hpp` is asked for as
    "nlohmann json", which is what the repository holding it is called. The
    extension is dropped - it says the file is a header and nothing else.
    """

    spelled = _valid_name(name)
    words = [
        word
        for word in re.split(r"[/._-]+", spelled)
        if word and word.lower() not in ("h", "hpp", "hxx", "hh")
    ]
    if not words:
        raise HeaderFetchError(f"{name!r} has nothing in it to search for")
    query = urllib.parse.urlencode(
        {
            "q": " ".join(dict.fromkeys(words)),
            "sort": "stars",
            "order": "desc",
            "per_page": str(max(1, results)),
        }
    )
    answer = _read_json(f"{_SOURCE_QUERY}?{query}", f"a search for {' '.join(words)}")
    items = answer.get("items")
    if not isinstance(items, list):
        raise HeaderFetchError(f"the source host answered nothing usable for {name!r}")
    found: "list[str]" = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        full = entry.get("full_name")
        if isinstance(full, str) and "/" in full and full not in found:
            found.append(full)
            branch = entry.get("default_branch")
            if isinstance(branch, str) and branch:
                # Kept so that reading the files does not cost a second
                # request: an unauthenticated caller gets sixty an hour, and
                # spending two per repository ran out during one build.
                _BRANCHES[full] = branch
    return found


#: Default branches already learned, so they are not asked for twice.
_BRANCHES: "dict[str, str]" = {}


def _from_source(
    wanted: str,
    into: Path,
    *,
    results: int,
    say,
    reasons: "list[str]",
) -> "Path | None":
    """Find the header in a repository and keep it, with its neighbours."""

    # `nlohmann/json.hpp` names its own repository: an include path with a
    # directory in it is nearly always `<who>/<what>`, and asking for that
    # outright beats any search on the words.
    repositories: "list[str]" = []
    if "/" in wanted:
        holder, _, spelled = wanted.rpartition("/")
        named = re.sub(r"\.(h|hpp|hxx|hh)$", "", spelled, flags=re.I)
        if holder and named:
            repositories.append(f"{holder}/{named}")
    # The curated sets before any search, because a search ranks repositories
    # by their names and a platform header's name is a common word. Asking
    # for `winnt.h` returned a leaked NT5 source dump, an nmap script and a
    # Program Manager clone ahead of either header set, and the dump held a
    # 725-line file under that name - so the build got it, and nothing about
    # it was the header. A set that does not hold the header falls through in
    # one cached request, which is what makes trying them first affordable.
    for full in _COLLECTIONS:
        if full not in repositories:
            repositories.append(full)
    try:
        for full in search_source(wanted, results=results):
            if full not in repositories:
                repositories.append(full)
    except _DID_NOT_ANSWER as error:
        # Not fatal any more: the sets are already on the list, and they are
        # where a platform header was always going to come from.
        reasons.append(f"the source host: {error}")
    for full in repositories:
        try:
            say(f"  trying {full}")
            found = _take_from_repository(full, wanted, into, say=say)
        except _DID_NOT_ANSWER as error:
            reasons.append(f"{full}: {error}")
            continue
        if found is not None:
            say(f"  {wanted} came from {full}")
            if full in _COLLECTIONS:
                # Its neighbours are there too, so ask it first next time.
                _COLLECTIONS.remove(full)
                _COLLECTIONS.insert(0, full)
            return found
        reasons.append(f"{full}: holds no {wanted}")
    return None


def _take_from_repository(
    full: str, wanted: str, into: Path, *, say
) -> "Path | None":
    """Look through one repository's files for the header, and keep what is near it."""

    owner, _, repository = full.partition("/")
    branch = _BRANCHES.get(full)
    if branch is None:
        described = _read_json(
            f"{_SOURCE_TREE}/{urllib.parse.quote(owner)}/"
            f"{urllib.parse.quote(repository)}",
            f"the description of {full}",
        )
        branch = described.get("default_branch")
        if not isinstance(branch, str) or not branch:
            raise HeaderFetchError(f"{full} has no default branch")
        _BRANCHES[full] = branch
    paths = _TREES.get((full, branch))
    if paths is None:
        listed = _read_json(
            f"{_SOURCE_TREE}/{urllib.parse.quote(owner)}/"
            f"{urllib.parse.quote(repository)}/git/trees/"
            f"{urllib.parse.quote(branch)}?recursive=1",
            f"the files of {full}",
        )
        tree = listed.get("tree")
        if not isinstance(tree, list):
            raise HeaderFetchError(f"{full} answered no file list")
        paths = [
            entry["path"]
            for entry in tree
            if isinstance(entry, dict)
            and entry.get("type") == "blob"
            and isinstance(entry.get("path"), str)
        ]
        _TREES[(full, branch)] = paths
    matching = [
        path for path in paths if path == wanted or path.endswith("/" + wanted)
    ]
    if not matching:
        return None
    # The shortest path is the one the library means people to include; a
    # longer one is usually a copy under a test or a third-party directory.
    chosen = min(matching, key=len)
    # The directory it is written against, which is what `#include
    # "nlohmann/json.hpp"` names: as many parents up as the wanted name has
    # pieces.
    depth = wanted.count("/") + 1
    root = "/".join(chosen.split("/")[:-depth])
    prefix = f"{root}/" if root else ""
    # Every file under it, not only the ones spelled like a header: an
    # `#include` names a *file*, and a header set includes its resource
    # headers - `winnt.rh` - and its generated fragments by name. Filtering
    # those out of the closure left the build asking for one that was there.
    under = [path for path in paths if path.startswith(prefix)]
    headers = [
        path
        for path in under
        if path.lower().endswith((".h", ".hpp", ".hxx", ".hh", ".inc", ".ipp"))
    ]
    into.mkdir(parents=True, exist_ok=True)
    if len(headers) <= _WHOLE_DIRECTORY:
        # A small directory *is* the library, and taking it whole costs
        # little and cannot miss a header hidden behind a condition.
        return _take_paths(full, branch, headers, prefix, chosen, into, say)
    # A big one is a platform's whole set - thousands of files, nearly none
    # of them wanted. Take the header asked for and the ones it reaches: a
    # header's own `#include` lines say what it needs, and that closure is a
    # few dozen where the directory is a few thousand.
    say(f"  {root or '/'} holds {len(headers)} headers; taking what {wanted} reaches")
    return _take_closure(full, branch, under, prefix, chosen, into, say, paths)


def _take_paths(
    full: str,
    branch: str,
    paths: "list[str]",
    prefix: str,
    chosen: str,
    into: Path,
    say,
) -> "Path | None":
    """Download every one of these and keep it under `into`."""

    if len(paths) > _MAX_HEADERS:
        raise HeaderFetchError(
            f"{full} keeps {len(paths)} headers under {prefix or '/'}, which "
            f"is more than one build will pull down"
        )
    spent = 0
    answer: "Path | None" = None
    for path in paths:
        payload = _read_bytes(
            f"{_SOURCE_RAW}/{full}/{urllib.parse.quote(branch)}/"
            f"{urllib.parse.quote(path)}",
            f"{path} from {full}",
            _MAX_HEADER_BYTES,
        )
        spent += len(payload)
        if spent > _MAX_HEADER_BYTES:
            raise HeaderFetchError(
                f"{full} keeps more than {_MAX_HEADER_BYTES // (1024 * 1024)}MB "
                f"of headers under {prefix or '/'}"
            )
        written = into / path[len(prefix):]
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_bytes(payload)
        if path == chosen:
            answer = written
    return answer


def _take_closure(
    full: str,
    branch: str,
    paths: "list[str]",
    prefix: str,
    chosen: str,
    into: Path,
    say,
    everywhere: "list[str] | None" = None,
) -> "Path | None":
    """Download the header asked for and every header it reaches.

    A header says what it needs in its own `#include` lines, so the set that
    has to come down is a closure over those and not a whole directory. The
    conditions around them are not read: a header included only on some
    platform comes too, which costs a file and cannot leave one out.
    """

    by_name = {path[len(prefix):]: path for path in paths}
    # A set does not have to keep all of itself in one directory. mingw-w64
    # writes its headers under `include` and the support files they all
    # include - `_mingw_unicode.h` and its neighbours - under `crt`, so a
    # closure that only ever looked below the header asked for could not
    # resolve them and the build stopped on the first one. Anywhere else in
    # the same repository is still that set; the shortest path wins, the way
    # it does when the header itself is chosen.
    for path in sorted(everywhere or (), key=len):
        stem = path.rsplit("/", 1)[-1]
        by_name.setdefault(stem, path)
    answer: "Path | None" = None
    pending = [chosen[len(prefix):]]
    seen: "set[str]" = set()
    taken: "set[str]" = set()
    #: What the closure reaches and this set does not hold. Not an error -
    #: an `#include` behind a condition this does not read may never be
    #: reached - but worth saying, because a set that generates part of
    #: itself has no file to fetch for those at all.
    unresolved: "set[str]" = set()
    spent = 0
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = by_name.get(name)
        if path is None:
            continue
        if len(seen) > _MAX_HEADERS:
            raise HeaderFetchError(
                f"{chosen} reaches more than {_MAX_HEADERS} headers in {full}, "
                f"which is more than one build will pull down"
            )
        payload = _read_bytes(
            f"{_SOURCE_RAW}/{full}/{urllib.parse.quote(branch)}/"
            f"{urllib.parse.quote(path)}",
            f"{path} from {full}",
            _MAX_HEADER_BYTES,
        )
        spent += len(payload)
        if spent > _MAX_HEADER_BYTES:
            raise HeaderFetchError(
                f"{chosen} reaches more than "
                f"{_MAX_HEADER_BYTES // (1024 * 1024)}MB of headers in {full}"
            )
        written = into / name
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_bytes(payload)
        taken.add(name)
        if path == chosen:
            answer = written
        for reached in _INCLUDED.findall(payload.decode("utf-8", "replace")):
            spelled = reached.strip().replace("\\", "/")
            # A header py2bin ships is not taken even when the set has one.
            # Its copy would land in the cache directory beside the ones that
            # were wanted, and an include directory is searched before a
            # built-in - so one fetch of anything from a Windows set left
            # that set's `winnt.h` shadowing py2bin's own for every build
            # afterwards, which is how a fixed build came back broken.
            if spelled in _SUPPLIED or spelled.rsplit("/", 1)[-1] in _SUPPLIED:
                continue
            if spelled in by_name:
                if spelled not in seen:
                    pending.append(spelled)
            else:
                unresolved.add(spelled)
    say(f"  took {len(taken)} headers, {spent // 1024}KB")
    if unresolved:
        # Said rather than left to be discovered: a set that generates part
        # of itself at build time has no file to fetch for those, and the
        # build will ask for the first one it actually reaches.
        say(
            f"  {full} publishes no file for: "
            f"{', '.join(sorted(unresolved)[:8])}"
            f"{' and more' if len(unresolved) > 8 else ''}"
        )
    return answer


#: How many headers a directory may hold before it is taken by closure rather
#: than whole. A library's own include directory is a handful; a platform's is
#: thousands, nearly none of them wanted.
_WHOLE_DIRECTORY = 60

#: `#include <x/y.h>` and `#include "x/y.h"` - what a header says it needs.
_INCLUDED = re.compile(r'(?m)^[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]')


def _supplied() -> "frozenset[str]":
    """The headers py2bin ships, which a set does not have to hold."""

    from .c_preprocessor import _BUILTIN_HEADERS

    return frozenset(_BUILTIN_HEADERS)


_SUPPLIED = _supplied()




def fetch_header_from(
    url: str, name: str, into: Path, *, cache: "Path | None" = None
) -> Path:
    """Write the header at `url` into `into` under `name`.

    For a header that is in no index - one file in somebody's repository,
    which is how a good many libraries are shipped.
    """

    wanted = _valid_name(name)
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    payload = _read_bytes(url, f"the header {wanted}", 16 * 1024 * 1024)
    written = into / wanted
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_bytes(payload)
    return written


def _take_headers(
    url: str,
    label: str,
    into: Path,
    stem: str,
    *,
    cache: "Path | None" = None,
) -> "Path | None":
    """Download one package and keep every header in it. None if it holds none."""

    import tempfile

    into.mkdir(parents=True, exist_ok=True)
    store = Path(cache) if cache is not None else into / ".cache"
    archive, _digest = download_verified(
        url, store, label=label, max_download=_MAX_PACKAGE
    )
    with tempfile.TemporaryDirectory(dir=str(into)) as work:
        opened = Path(work)
        extract_zip(archive, opened)
        answer: "Path | None" = None
        for header in found_headers(opened):
            written = into / header.name
            written.write_bytes(header.read_bytes())
            if header.name == stem:
                answer = written
        return answer
