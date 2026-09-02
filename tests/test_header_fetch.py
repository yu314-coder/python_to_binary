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
    fetch_header,
    fetch_header_from,
    found_headers,
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


def _zip_of(files: dict) -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return buffer.getvalue()


class HeaderNameTests(unittest.TestCase):
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


class MissingHeaderTests(unittest.TestCase):
    def test_the_name_is_read_back_out_of_the_refusal(self):
        message = "x.cpp:2:2: cannot find the header 'WebView2.h'. py2bin looked in:"
        self.assertEqual(_header_that_is_missing(message), "WebView2.h")

    def test_any_other_refusal_is_not_one_of_these(self):
        self.assertIsNone(_header_that_is_missing("x.c:1:1: expected a ';'"))


class SearchTests(unittest.TestCase):
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


class FetchFromAPackageTests(unittest.TestCase):
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


class FetchFromSourceTests(unittest.TestCase):
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

        # `_TREES` is a module-level cache of file lists by repository and
        # branch. It is what stops a build that fetches a dozen headers from
        # one set asking a dozen times, and it also outlives a test - so a
        # canned tree from an earlier one answers for `wine-mirror/wine` here
        # and this passes or fails by the order the suite happens to run in.
        header_fetch._TREES.clear()
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


class CollectionTests(unittest.TestCase):
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


class FetchFromAUrlTests(unittest.TestCase):
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
