"""Fetching a header py2bin cannot find here.

Nothing in this file reaches the network. `runtime_fetch.DOWNLOADER` is the
seam the whole module already goes through - it exists so a caller can supply
its own way of getting bytes - and these tests supply canned ones.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from py2bin import header_fetch, runtime_fetch
from py2bin.header_fetch import (
    HeaderFetchError,
    _valid_name,
    components_fetched,
    fetch_header,
    fetch_header_from,
    fetch_library,
    found_headers,
    libraries_offered,
    search_index,
    search_source,
)
from py2bin.interactive import _header_that_is_missing


class _Answers:
    """Stands in for the downloader, answering from a table of URLs."""

    def __init__(self, table):
        self.table = table
        self.asked = []

    def __call__(self, url, label):
        self.asked.append(url)
        for prefix, payload in self.table.items():
            if url.startswith(prefix):
                if isinstance(payload, Exception):
                    # A canned failure: how a candidate that is not there
                    # answers, which is not always a `FetchError`.
                    raise payload
                return payload
        raise runtime_fetch.FetchError(f"nothing canned for {url}")


class _Downloader:
    def __init__(self, table):
        self.answers = _Answers(table)

    def __enter__(self):
        self.before = runtime_fetch.DOWNLOADER
        runtime_fetch.DOWNLOADER = self.answers
        return self.answers

    def __exit__(self, *_):
        runtime_fetch.DOWNLOADER = self.before


def _zip_of(files: dict, compression=zipfile.ZIP_STORED) -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return buffer.getvalue()


class _Ranges:
    """Stands in for the range downloader: answers a slice of what the table
    holds, the way a server that honours `Range` does, and keeps a record of
    every range asked - which is how a test tells a member-wise fetch from a
    whole download."""

    def __init__(self, table, corrupt=None):
        self.table = table
        self.asked = []
        self.corrupt = corrupt

    def __call__(self, url, label, start, stop):
        for prefix, payload in self.table.items():
            if url.startswith(prefix):
                whole = payload[start:] if start < 0 else payload[start:stop]
                if self.corrupt is not None and start >= 0:
                    whole = self.corrupt(whole)
                self.asked.append((url, start, stop, len(whole)))
                return whole, len(payload)
        raise runtime_fetch.FetchError(f"nothing canned for {url}")

    def __enter__(self):
        self.before = runtime_fetch.RANGE_DOWNLOADER
        runtime_fetch.RANGE_DOWNLOADER = self
        return self

    def __exit__(self, *_):
        runtime_fetch.RANGE_DOWNLOADER = self.before


#: What the module's own tables and caches held when it was imported, taken by
#: asking the module rather than by naming them one at a time - a cache added
#: later is then covered without anybody remembering to come back here. The
#: dunders are left alone: `__builtins__` is a dict too, and is not this
#: module's to put back.
_AS_IMPORTED = {
    name: holder.copy()
    for name, holder in vars(header_fetch).items()
    if not name.startswith("__") and isinstance(holder, (dict, set, list))
}


def _as_it_was_imported():
    """Put every module-level table and cache back to what it held at import."""

    for name, contents in _AS_IMPORTED.items():
        holder = getattr(header_fetch, name)
        holder.clear()
        if isinstance(holder, list):
            holder.extend(contents)
        else:
            holder.update(contents)


class _Fetching(unittest.TestCase):
    """A test that starts, and leaves, the module's state as it was imported.

    `_TREES` and `_BRANCHES` are keyed by repository and live as long as the
    process, which is the whole point of them: a build fetching a dozen headers
    from one set asks for its file list once rather than a dozen times. Between
    tests it means a tree canned by one answers for another that names the same
    repository, and a test passes on its own and fails in the full module.
    """

    def setUp(self):
        self.addCleanup(_as_it_was_imported)
        _as_it_was_imported()


class HeaderNameTests(_Fetching):
    def test_a_name_that_could_escape_the_directory_is_refused(self):
        for spelled in ("../secret.h", "/etc/passwd", "a/../../b.h", ""):
            with self.subTest(spelled=spelled):
                with self.assertRaises(HeaderFetchError):
                    _valid_name(spelled)

    def test_a_name_with_a_space_in_it_is_refused(self):
        # Not because a space is dangerous, but because nothing writes one:
        # a name that odd is a sign of something other than a header.
        with self.assertRaises(HeaderFetchError):
            _valid_name("my header.h")

    def test_an_ordinary_name_is_kept_as_it_is(self):
        self.assertEqual(_valid_name("WebView2.h"), "WebView2.h")
        self.assertEqual(_valid_name("nlohmann/json.hpp"), "nlohmann/json.hpp")

    def test_a_backslash_is_read_as_a_separator(self):
        # Which is how a program written on Windows spells an include.
        self.assertEqual(_valid_name(r"nlohmann\json.hpp"), "nlohmann/json.hpp")


class MissingHeaderTests(_Fetching):
    def test_the_name_is_read_back_out_of_the_refusal(self):
        message = "x.cpp:2:2: cannot find the header 'WebView2.h'. py2bin looked in:"
        self.assertEqual(_header_that_is_missing(message), "WebView2.h")

    def test_any_other_refusal_is_not_one_of_these(self):
        self.assertIsNone(_header_that_is_missing("x.c:1:1: expected a ';'"))


class SearchTests(_Fetching):
    def test_the_package_index_is_asked_for_the_stem(self):
        answer = json.dumps({"data": [{"id": "Microsoft.Web.WebView2"}]}).encode()
        with _Downloader({"https://azuresearch": answer}) as asked:
            found = search_index("WebView2.h")
        self.assertEqual(found, ["Microsoft.Web.WebView2"])
        self.assertIn("q=WebView2", asked.asked[0])

    def test_the_source_host_is_asked_for_the_words_of_the_name(self):
        answer = json.dumps(
            {"items": [{"full_name": "nlohmann/json", "default_branch": "develop"}]}
        ).encode()
        with _Downloader({"https://api.github.com/search": answer}) as asked:
            found = search_source("nlohmann/json.hpp")
        self.assertEqual(found, ["nlohmann/json"])
        # The extension says the file is a header and nothing else, so it is
        # not part of what is searched for.
        self.assertIn("nlohmann+json", asked.asked[0])
        self.assertNotIn("hpp", asked.asked[0])

    def test_an_index_that_answers_something_else_is_refused(self):
        with _Downloader({"https://azuresearch": b'{"data": "no"}'}):
            with self.assertRaises(HeaderFetchError):
                search_index("thing.h")


class FetchFromAPackageTests(_Fetching):
    def test_the_header_and_its_neighbours_are_kept(self):
        package = _zip_of(
            {
                "build/native/include/WebView2.h": "/* the one */",
                "build/native/include/WebView2Interop.h": "/* beside it */",
                "lib/net45/thing.dll": "not a header",
            }
        )
        table = {
            "https://azuresearch": json.dumps(
                {"data": [{"id": "Microsoft.Web.WebView2"}]}
            ).encode(),
            "https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/index.json":
                json.dumps({"versions": ["1.0.1", "1.0.2"]}).encode(),
            "https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/1.0.2":
                package,
        }
        with tempfile.TemporaryDirectory() as work:
            into = Path(work) / "headers"
            with _Downloader(table):
                kept = fetch_header("WebView2.h", into)
            self.assertEqual(kept.name, "WebView2.h")
            self.assertEqual(kept.read_text(), "/* the one */")
            names = {path.name for path in found_headers(into)}
            self.assertIn("WebView2Interop.h", names)

    def test_a_pre_release_is_not_picked_up(self):
        table = {
            "https://azuresearch": json.dumps({"data": [{"id": "Thing"}]}).encode(),
            "https://api.nuget.org/v3-flatcontainer/thing/index.json": json.dumps(
                {"versions": ["2.0.0-beta1"]}
            ).encode(),
            "https://api.github.com/search": json.dumps({"items": []}).encode(),
        }
        with tempfile.TemporaryDirectory() as work:
            with _Downloader(table):
                with self.assertRaises(HeaderFetchError) as refused:
                    fetch_header("Thing.h", Path(work))
        self.assertIn("pre-release", str(refused.exception))

    def test_a_package_that_does_not_hold_it_is_said_so_by_name(self):
        table = {
            "https://azuresearch": json.dumps({"data": [{"id": "Thing"}]}).encode(),
            "https://api.nuget.org/v3-flatcontainer/thing/index.json": json.dumps(
                {"versions": ["1.0.0"]}
            ).encode(),
            "https://api.nuget.org/v3-flatcontainer/thing/1.0.0": _zip_of(
                {"readme.txt": "nothing here"}
            ),
            "https://api.github.com/search": json.dumps({"items": []}).encode(),
        }
        with tempfile.TemporaryDirectory() as work:
            with _Downloader(table):
                with self.assertRaises(HeaderFetchError) as refused:
                    fetch_header("Thing.h", Path(work))
        self.assertIn("Thing: holds no Thing.h", str(refused.exception))


class FetchedBeforeTests(_Fetching):
    def test_a_header_kept_under_its_included_name_is_not_asked_for_again(self):
        # `openssl/evp.h` is kept as `openssl/evp.h`, and the check for an
        # earlier fetch looked only for `evp.h`: every build asked the network
        # for a set it already had, until the host said no more for the hour.
        with tempfile.TemporaryDirectory() as directory:
            into = Path(directory)
            kept = into / "openssl" / "evp.h"
            kept.parent.mkdir()
            kept.write_text("int evp;\n", encoding="utf-8")
            with _Downloader({}) as network:
                self.assertEqual(fetch_header("openssl/evp.h", into), kept)
            self.assertEqual(network.asked, [])


class FetchFromSourceTests(_Fetching):
    #: A name with a directory in it names its own repository, so the source
    #: host is asked first and the package index is never reached.
    _TREE = json.dumps(
        {
            "tree": [
                {"type": "blob", "path": "single_include/nlohmann/json.hpp"},
                {"type": "blob", "path": "include/nlohmann/json.hpp"},
                {"type": "blob", "path": "include/nlohmann/detail/macro.hpp"},
                {"type": "blob", "path": "test/src/unit.cpp"},
                {"type": "tree", "path": "include"},
            ]
        }
    ).encode()

    def _table(self):
        return {
            "https://api.github.com/repos/nlohmann/json/git/trees": self._TREE,
            "https://api.github.com/repos/nlohmann/json": json.dumps(
                {"default_branch": "develop"}
            ).encode(),
            "https://raw.githubusercontent.com/nlohmann/json/develop/"
            "include/nlohmann/json.hpp": b"/* the one */",
            "https://raw.githubusercontent.com/nlohmann/json/develop/"
            "include/nlohmann/detail/macro.hpp": b"/* beside it */",
            "https://raw.githubusercontent.com/nlohmann/json/develop/"
            "single_include/nlohmann/json.hpp": b"/* the packed copy */",
            "https://api.github.com/search": json.dumps({"items": []}).encode(),
        }

    def test_the_shortest_path_is_the_one_the_library_means(self):
        with tempfile.TemporaryDirectory() as work:
            into = Path(work) / "headers"
            with _Downloader(self._table()):
                kept = fetch_header("nlohmann/json.hpp", into)
            # Two copies of the header, and the shorter path is the one under
            # `include/` - which is the directory a library means people to
            # put on their include path.
            self.assertEqual(kept.read_text(), "/* the one */")
            self.assertEqual(kept, into / "nlohmann" / "json.hpp")
            # And what it includes came with it, one directory down.
            self.assertTrue((into / "nlohmann" / "detail" / "macro.hpp").is_file())

    def test_a_guess_that_answers_404_does_not_end_the_search(self):
        """The repository read out of the include path may not exist.

        `who/what.h` is asked of `who/what` first, because an include with a
        directory in it nearly always names its own repository. When it does
        not, the host answers 404 - which urllib raises as an HTTPError, an
        OSError and not a `FetchError`. Only `FetchError` was caught, so that
        first guess took every later candidate with it and a header one of
        the curated sets holds was reported missing.
        """

        missing = urllib.error.HTTPError(
            "https://api.github.com/repos/who/what", 404, "Not Found", {}, None
        )
        table = {
            "https://api.github.com/repos/who/what": missing,
            "https://api.github.com/repos/wine-mirror/wine/git/trees": json.dumps(
                {"tree": [{"type": "blob", "path": "include/who/what.h"}]}
            ).encode(),
            "https://api.github.com/repos/wine-mirror/wine": json.dumps(
                {"default_branch": "master"}
            ).encode(),
            "https://raw.githubusercontent.com/wine-mirror/wine/": b"/* from the set */",
            "https://api.github.com/search": json.dumps({"items": []}).encode(),
        }
        with tempfile.TemporaryDirectory() as work:
            into = Path(work) / "headers"
            with _Downloader(table) as asked:
                kept = fetch_header("who/what.h", into)
            self.assertEqual(kept.read_text(), "/* from the set */")
        # It really did try the guess first, and really did carry on past it.
        self.assertTrue(any("repos/who/what" in one for one in asked.asked))
        self.assertTrue(any("wine-mirror" in one for one in asked.asked))

    def test_the_package_index_is_not_asked_when_the_name_names_a_repository(self):
        with tempfile.TemporaryDirectory() as work:
            with _Downloader(self._table()) as asked:
                fetch_header("nlohmann/json.hpp", Path(work))
        self.assertFalse([one for one in asked.asked if "azuresearch" in one])

    def test_a_repository_without_it_is_said_so_by_name(self):
        table = {
            "https://api.github.com/repos/who/what/git/trees": json.dumps(
                {"tree": [{"type": "blob", "path": "README.md"}]}
            ).encode(),
            "https://api.github.com/repos/who/what": json.dumps(
                {"default_branch": "main"}
            ).encode(),
            "https://api.github.com/search": json.dumps({"items": []}).encode(),
            "https://azuresearch": json.dumps({"data": []}).encode(),
        }
        with tempfile.TemporaryDirectory() as work:
            with _Downloader(table):
                with self.assertRaises(HeaderFetchError) as refused:
                    fetch_header("who/what.hpp", Path(work))
        self.assertIn("who/what: holds no who/what.hpp", str(refused.exception))


class CarriedOverTests(_Fetching):
    """Nothing one test cans is left to answer for the next one."""

    def test_a_file_list_from_an_earlier_test_does_not_answer_here(self):
        """Two tests here name `wine-mirror/wine` and can different trees.

        Whichever ran first used to fill `_TREES` for both, so the second
        failed for a reason that had nothing to do with what it tested - and
        passed again as soon as it was run on its own.
        """

        header_fetch._BRANCHES["wine-mirror/wine"] = "master"
        header_fetch._TREES[("wine-mirror/wine", "master")] = ["README.md"]
        ran = unittest.TestResult()
        FetchFromSourceTests(
            "test_a_guess_that_answers_404_does_not_end_the_search"
        ).run(ran)
        self.assertEqual([text for _which, text in ran.errors + ran.failures], [])


class CollectionTests(_Fetching):
    """A platform header is in a *set*, never in a repository named after it."""

    _TREE = json.dumps(
        {
            "tree": [
                {"type": "blob", "path": "headers/include/commctrl.h"},
                {"type": "blob", "path": "headers/include/commctrl_dce.h"},
                {"type": "blob", "path": "headers/include/windef.h"},
                {"type": "blob", "path": "headers/include/unrelated.h"},
            ]
            # Enough of them that the directory is taken by closure rather
            # than whole; the real one holds fifteen hundred.
            + [
                {"type": "blob", "path": f"headers/include/other{index}.h"}
                for index in range(80)
            ]
        }
    ).encode()

    def _table(self, collection):
        raw = "https://raw.githubusercontent.com/" + collection + "/main/"
        return {
            "https://api.github.com/search": json.dumps({"items": []}).encode(),
            "https://azuresearch": json.dumps({"data": []}).encode(),
            f"https://api.github.com/repos/{collection}/git/trees": self._TREE,
            f"https://api.github.com/repos/{collection}": json.dumps(
                {"default_branch": "main"}
            ).encode(),
            raw + "headers/include/commctrl.h": b'#include "commctrl_dce.h"\n#include <stdio.h>\n',
            raw + "headers/include/commctrl_dce.h": b'#include "windef.h"\n',
            raw + "headers/include/windef.h": b"/* the end */\n",
        }

    def test_a_header_is_found_in_a_set_that_is_not_named_after_it(self):
        from py2bin.header_fetch import _COLLECTIONS

        collection = _COLLECTIONS[0]
        with tempfile.TemporaryDirectory() as work:
            into = Path(work) / "headers"
            with _Downloader(self._table(collection)):
                kept = fetch_header("commctrl.h", into)
            self.assertEqual(kept, into / "commctrl.h")

    def test_only_the_headers_it_reaches_come_with_it(self):
        """A platform's include directory is thousands of files.

        What has to come down is the closure over a header's own `#include`
        lines, which is a handful - taking the directory whole would download
        a set nearly none of which is wanted.
        """

        from py2bin.header_fetch import _COLLECTIONS

        collection = _COLLECTIONS[0]
        with tempfile.TemporaryDirectory() as work:
            into = Path(work) / "headers"
            with _Downloader(self._table(collection)):
                fetch_header("commctrl.h", into)
            names = {path.name for path in found_headers(into)}
            # windef.h is reached, and is not taken: py2bin ships that one,
            # and a downloaded copy in the cache directory would shadow it.
            self.assertEqual(names, {"commctrl.h", "commctrl_dce.h"})
            self.assertNotIn("unrelated.h", names)

    def test_a_header_py2bin_ships_is_not_taken_along(self):
        """The cache directory is searched before a built-in, so a copy left
        there shadows py2bin's own for every build afterwards - which is how
        a Windows build that had been fixed came back with the same error."""

        from py2bin.header_fetch import _COLLECTIONS, _SUPPLIED

        self.assertIn("windef.h", _SUPPLIED)
        collection = _COLLECTIONS[0]
        with tempfile.TemporaryDirectory() as work:
            into = Path(work) / "headers"
            with _Downloader(self._table(collection)):
                fetch_header("commctrl.h", into)
            self.assertFalse((into / "windef.h").exists())

    def test_asking_outright_for_one_py2bin_ships_says_so(self):
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaises(HeaderFetchError) as caught:
                fetch_header("winnt.h", Path(work))
            self.assertIn("py2bin ships", str(caught.exception))


#: A component's packages as the index lists them: the managed ones first,
#: named after the component and shipping nothing native, and the one with
#: the library far down the list.
_STEM_SEARCH = "https://azuresearch-usnc.nuget.org/query?q=libthing-3-x64.dll&"
_COMPONENT_SEARCH = "https://azuresearch-usnc.nuget.org/query?q=thing&"
_MANAGED = "https://api.nuget.org/v3-flatcontainer/thing.managed/1.0.0/"
_NATIVE = "https://api.nuget.org/v3-flatcontainer/thing-native/3.5.5/"
_LIBRARY = bytes(range(256)) * 300  # what the library's bytes are, compressible


def _component_table(package: bytes) -> dict:
    return {
        _STEM_SEARCH: json.dumps({"data": []}).encode(),
        _COMPONENT_SEARCH: json.dumps(
            {
                "data": [
                    {"id": "Thing.Managed", "version": "1.0.0"},
                    {"id": "thing-native", "version": "3.5.5"},
                ]
            }
        ).encode(),
        _MANAGED: _zip_of({"lib/net45/Thing.dll": b"MZ managed", "Thing.nuspec": ""}),
        _NATIVE: package,
    }


def _native_package(compression=zipfile.ZIP_DEFLATED) -> bytes:
    return _zip_of(
        {
            "thing-native.nuspec": "<package/>",
            "include/thing/thing.h": "int thing(void);",
            "runtimes/win-x86/native/libthing-3.dll": b"MZ x86",
            "runtimes/win-x64/native/libthing-3-x64.dll": _LIBRARY,
            "runtimes/win-x64/native/libthing-other-x64.dll": b"MZ other",
            "runtimes/win-arm64/native/libthing-3-arm64.dll": b"MZ arm64",
        },
        compression,
    )


class LibraryOfAComponentTests(_Fetching):
    """`--library libthing-3-x64.dll` names a file no package is called after;
    the package is called after the component the headers are about."""

    def test_the_library_is_taken_out_of_the_components_package_by_its_range(self):
        package = _native_package()
        with tempfile.TemporaryDirectory() as work:
            into = Path(work) / "dist"
            with _Downloader(_component_table(package)) as network, _Ranges(
                _component_table(package)
            ) as ranges:
                found = fetch_library(
                    "libthing-3-x64.dll", into, "x86_64", components=("thing",)
                )
            self.assertEqual(found, into / "libthing-3-x64.dll")
            self.assertEqual(found.read_bytes(), _LIBRARY)
        # The stem was searched for first, the component second.
        self.assertTrue(any(url.startswith(_STEM_SEARCH) for url in network.asked))
        self.assertTrue(any(url.startswith(_COMPONENT_SEARCH) for url in network.asked))
        # Neither package was downloaded whole: each was asked for its
        # directory - the end of the file - and the one holding the library
        # for the member's own bytes, from where its directory says the
        # member starts, which is less than the package.
        self.assertFalse(any(url.startswith(_NATIVE) for url in network.asked))
        asked_of_native = [
            (start, stop, got)
            for url, start, stop, got in ranges.asked
            if url.startswith(_NATIVE)
        ]
        self.assertEqual(asked_of_native[0][0], -(64 * 1024 + 22))
        import io

        member = zipfile.ZipFile(io.BytesIO(package)).getinfo(
            "runtimes/win-x64/native/libthing-3-x64.dll"
        )
        self.assertEqual(asked_of_native[1][0], member.header_offset)
        self.assertLess(asked_of_native[1][2], len(package))
        self.assertTrue(any(url.startswith(_MANAGED) for url, *_rest in ranges.asked))

    def test_a_stored_member_comes_out_the_same_way(self):
        package = _native_package(zipfile.ZIP_STORED)
        with tempfile.TemporaryDirectory() as work:
            into = Path(work)
            with _Downloader(_component_table(package)), _Ranges(
                _component_table(package)
            ):
                found = fetch_library(
                    "libthing-3-x64.dll", into, "x86_64", components=("thing",)
                )
            self.assertEqual(found.read_bytes(), _LIBRARY)

    def test_a_server_that_answers_no_range_is_asked_for_the_whole_package(self):
        package = _native_package()

        def no_ranges(url, label, start, stop):
            return None

        with tempfile.TemporaryDirectory() as work:
            into = Path(work)
            before = runtime_fetch.RANGE_DOWNLOADER
            runtime_fetch.RANGE_DOWNLOADER = no_ranges
            try:
                with _Downloader(_component_table(package)) as network:
                    found = fetch_library(
                        "libthing-3-x64.dll",
                        into,
                        "x86_64",
                        components=("thing",),
                        cache=into / ".cache",
                    )
            finally:
                runtime_fetch.RANGE_DOWNLOADER = before
            self.assertEqual(found.read_bytes(), _LIBRARY)
        self.assertTrue(any(url.startswith(_NATIVE) for url in network.asked))

    def test_a_member_that_does_not_come_out_as_listed_is_refused_by_name(self):
        package = _native_package()

        def flipped(data: bytes) -> bytes:
            # Past the local header and its 42-character name: in the data.
            return data[:80] + bytes(byte ^ 0xFF for byte in data[80:88]) + data[88:]

        with tempfile.TemporaryDirectory() as work:
            with _Downloader(_component_table(package)), _Ranges(
                _component_table(package), corrupt=flipped
            ):
                with self.assertRaises(HeaderFetchError) as refused:
                    fetch_library(
                        "libthing-3-x64.dll", Path(work), "x86_64", components=("thing",)
                    )
            self.assertIn("libthing-3-x64.dll in thing-native 3.5.5", str(refused.exception))
            self.assertEqual(list(Path(work).iterdir()), [])

    def test_a_package_that_holds_it_for_another_machine_only_is_passed_over(self):
        package = _native_package()
        with tempfile.TemporaryDirectory() as work:
            with _Downloader(_component_table(package)), _Ranges(
                _component_table(package)
            ):
                with self.assertRaises(HeaderFetchError) as refused:
                    fetch_library(
                        "libthing-3-x64.dll", Path(work), "arm64", components=("thing",)
                    )
        self.assertIn(
            "none of the 2 packages named after thing holds libthing-3-x64.dll for arm64",
            str(refused.exception),
        )

    def test_what_the_headers_are_about_is_read_off_the_cache(self):
        with tempfile.TemporaryDirectory() as work:
            into = Path(work)
            (into / "openssl").mkdir()
            (into / "openssl" / "evp.h").write_text("")
            (into / ".cache").mkdir()
            (into / "zlib.h").write_text("")
            (into / "notes.txt").write_text("")
            self.assertEqual(components_fetched(into), ["openssl", "zlib"])
            self.assertEqual(components_fetched(into / "absent"), [])

    def test_what_could_be_named_is_listed_by_package(self):
        package = _native_package()
        with _Downloader(_component_table(package)), _Ranges(_component_table(package)):
            self.assertEqual(
                libraries_offered("thing", "x86_64"),
                [("thing-native 3.5.5", ["libthing-3-x64.dll", "libthing-other-x64.dll"])],
            )
            self.assertEqual(
                libraries_offered("thing", "arm64"),
                [("thing-native 3.5.5", ["libthing-3-arm64.dll"])],
            )
            self.assertEqual(libraries_offered("thing", "riscv"), [])


class FetchFromAUrlTests(_Fetching):
    def test_a_header_named_outright_is_written_where_it_was_asked_for(self):
        with tempfile.TemporaryDirectory() as work:
            into = Path(work) / "headers"
            with _Downloader({"https://example.invalid": b"/* mine */"}):
                kept = fetch_header_from(
                    "https://example.invalid/thing.h", "vendor/thing.h", into
                )
            self.assertEqual(kept, into / "vendor" / "thing.h")
            self.assertEqual(kept.read_text(), "/* mine */")

    def test_a_url_that_is_not_https_is_refused(self):
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaises(runtime_fetch.FetchError):
                fetch_header_from("http://example.invalid/x.h", "x.h", Path(work))


if __name__ == "__main__":
    unittest.main()
