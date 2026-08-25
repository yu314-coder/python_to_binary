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
import urllib.parse
from pathlib import Path

from .runtime_fetch import (
    FetchError,
    _read_bytes,
    _read_json,
    download_verified,
    extract_zip,
)

__all__ = [
    "HeaderFetchError",
    "found_headers",
    "fetch_header",
    "fetch_header_from",
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
_MAX_HEADERS = 400
_MAX_HEADER_BYTES = 32 * 1024 * 1024

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
    found: "list[str]" = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        package = entry.get("id")
        if isinstance(package, str) and package and package not in found:
            found.append(package)
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
    into = Path(into)
    already = into / stem
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
    except FetchError as error:
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
        except FetchError as error:
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
        f"  Give the URL of the header itself, or put it in a directory and "
        f"name that with --include"
    )


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
    try:
        for full in search_source(wanted, results=results):
            if full not in repositories:
                repositories.append(full)
    except FetchError as error:
        reasons.append(f"the source host: {error}")
        if not repositories:
            return None
    for full in repositories:
        try:
            say(f"  trying {full}")
            found = _take_from_repository(full, wanted, into, say=say)
        except FetchError as error:
            reasons.append(f"{full}: {error}")
            continue
        if found is not None:
            say(f"  {wanted} came from {full}")
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
    # pieces. Everything under that comes too, because a header includes its
    # neighbours and one at a time would ask again immediately.
    depth = wanted.count("/") + 1
    root = "/".join(chosen.split("/")[:-depth])
    prefix = f"{root}/" if root else ""
    neighbours = [
        path
        for path in paths
        if path.startswith(prefix)
        and path.lower().endswith((".h", ".hpp", ".hxx", ".hh", ".inc", ".ipp"))
    ]
    if len(neighbours) > _MAX_HEADERS:
        raise HeaderFetchError(
            f"{full} keeps {len(neighbours)} headers under {root or '/'}, "
            f"which is more than one build will pull down"
        )
    into.mkdir(parents=True, exist_ok=True)
    spent = 0
    answer: "Path | None" = None
    for path in neighbours:
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
                f"of headers under {root or '/'}"
            )
        written = into / path[len(prefix):]
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_bytes(payload)
        if path == chosen:
            answer = written
    return answer


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
