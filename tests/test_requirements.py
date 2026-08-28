"""Working out what a program needs, without guessing at it."""

import tempfile
import time
import unittest
from pathlib import Path

from py2bin.requirements import KNOWN_PROJECTS, discover


class DiscoveryTests(unittest.TestCase):
    def _tree(self, files: dict) -> Path:
        directory = Path(tempfile.mkdtemp())
        for name, text in files.items():
            (directory / name).write_text(text)
        return directory / "main.py"

    def test_a_module_beside_the_program_needs_no_download(self):
        entry = self._tree({
            "main.py": "import helper\nprint(helper.x)\n",
            "helper.py": "x = 1\n",
        })
        found = discover(entry)
        self.assertIn("helper", found.local)
        self.assertEqual(found.projects, [])

    def test_the_standard_library_is_not_a_download(self):
        entry = self._tree({"main.py": "import json, os, sys\n"})
        found = discover(entry)
        self.assertEqual(found.projects, [])
        self.assertEqual(found.unknown, [])
        self.assertIn("json", found.standard)

    def test_an_import_is_translated_to_the_project_that_publishes_it(self):
        entry = self._tree({"main.py": "from PIL import Image\nimport cv2\n"})
        found = discover(entry)
        self.assertEqual(found.projects, ["opencv-python", "pillow"])

    def test_an_unknown_import_is_reported_rather_than_guessed(self):
        # The failure mode of a guess is downloading a stranger's package,
        # so a name that is not in the checked table is handed back instead.
        entry = self._tree({"main.py": "import someone_elses_thing\n"})
        found = discover(entry)
        self.assertEqual(found.projects, [])
        self.assertEqual(found.unknown, ["someone_elses_thing"])

    def test_imports_are_followed_through_local_modules(self):
        entry = self._tree({
            "main.py": "import helper\n",
            "helper.py": "import psutil\n",
        })
        self.assertEqual(discover(entry).projects, ["psutil"])

    def test_a_relative_import_is_not_taken_for_a_project(self):
        entry = self._tree({"main.py": "from . import sibling\n"})
        self.assertEqual(discover(entry).projects, [])
        self.assertEqual(discover(entry).unknown, [])

    def test_every_translation_differs_from_its_import_name_or_is_deliberate(self):
        # A table entry that merely repeats the name is fine, but a typo that
        # points at a different project is not something to find at runtime.
        for name, project in KNOWN_PROJECTS.items():
            self.assertTrue(project.strip(), f"{name} maps to nothing")
            self.assertEqual(project, project.lower(), f"{project} is not normalised")


if __name__ == "__main__":
    unittest.main()


class LentDownloaderTests(unittest.TestCase):
    """The library cannot shell out, so a caller that may can lend a way in.

    Some Pythons ship without a working ssl module, or keep the network away
    from the interpreter while the shell beside it can still reach out. The
    fetcher therefore takes a replacement rather than assuming urllib works.
    """

    def setUp(self):
        from py2bin import runtime_fetch

        self.module = runtime_fetch
        self.addCleanup(setattr, runtime_fetch, "DOWNLOADER", None)

    def test_a_replacement_is_used_instead_of_urllib(self):
        seen = []

        def lent(url, label):
            seen.append((url, label))
            return b'{"ok": true}'

        self.module.DOWNLOADER = lent
        answer = self.module._read_json("https://example.invalid/x", "a label")
        self.assertEqual(answer, {"ok": True})
        self.assertEqual(seen, [("https://example.invalid/x", "a label")])

    def test_a_replacement_cannot_smuggle_in_plain_http(self):
        self.module.DOWNLOADER = lambda url, label: b"{}"
        with self.assertRaises(self.module.FetchError):
            self.module._read_json("http://example.invalid/x", "a label")

    def test_a_replacement_that_answers_nothing_is_refused(self):
        self.module.DOWNLOADER = lambda url, label: None
        with self.assertRaises(self.module.FetchError):
            self.module._read_json("https://example.invalid/x", "a label")

    def test_the_stream_path_takes_it_too(self):
        self.module.DOWNLOADER = lambda url, label: b"payload"
        stream = self.module._open_stream("https://example.invalid/x", "a label")
        self.assertEqual(stream.read(), b"payload")


class OlderReleaseTests(unittest.TestCase):
    """A project publishes wheels for the interpreters that existed then.

    Soon after a Python release the newest version of something often has
    nothing for it, while an older version does. An older wheel of the right
    shape is far more use than a build that stops.
    """

    def test_releases_are_ordered_newest_first(self):
        from py2bin.runtime_fetch import _earlier_releases

        document = {
            "releases": {
                "1.9": [{"filename": "a"}],
                "2.6.1": [{"filename": "b"}],
                "2.6.0": [{"filename": "c"}],
                "2.10.0": [{"filename": "d"}],
            }
        }
        self.assertEqual(
            [name for name, _ in _earlier_releases(document)],
            ["2.10.0", "2.6.1", "2.6.0", "1.9"],
        )

    def test_a_release_with_no_files_is_skipped(self):
        from py2bin.runtime_fetch import _earlier_releases

        document = {"releases": {"1.0": [], "2.0": [{"filename": "x"}]}}
        self.assertEqual([name for name, _ in _earlier_releases(document)], ["2.0"])

    def test_a_document_without_releases_answers_nothing(self):
        from py2bin.runtime_fetch import _earlier_releases

        self.assertEqual(_earlier_releases({}), [])

    def test_a_prerelease_does_not_outrank_the_plain_version(self):
        from py2bin.runtime_fetch import _version_key

        self.assertLess(_version_key("2.6.0.dev2"), _version_key("2.6.1"))


class WheelRequirementTests(unittest.TestCase):
    """What a downloaded wheel says it stands on.

    A program imports pywebview and never mentions proxy_tools; pywebview
    mentions it, in the metadata beside the code. Reading only the program's
    imports fetches the package without the things under it, and the bundle
    then fails on an import nobody wrote.
    """

    def _dist_info(self, body: str) -> Path:
        room = Path(tempfile.mkdtemp())
        info = room / "thing-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(body)
        return room

    def test_requirements_are_read(self):
        from py2bin.requirements import required_by

        room = self._dist_info(
            "Name: thing\nRequires-Dist: proxy-tools\nRequires-Dist: bottle>=0.12\n"
        )
        self.assertEqual(required_by(room), ["proxy-tools", "bottle"])

    def test_an_extra_is_not_opted_into(self):
        from py2bin.requirements import required_by

        room = self._dist_info(
            "Name: thing\n"
            "Requires-Dist: needed\n"
            'Requires-Dist: only-for-tests; extra == "test"\n'
        )
        self.assertEqual(required_by(room), ["needed"])

    def test_a_version_specifier_is_not_part_of_the_name(self):
        from py2bin.requirements import required_by

        room = self._dist_info("Name: thing\nRequires-Dist: cffi>=1.0,!=1.2\n")
        self.assertEqual(required_by(room), ["cffi"])

    def test_a_marker_that_is_not_an_extra_still_counts(self):
        # A platform marker may well be true on the target being built for,
        # and the fetch refuses what does not fit anyway.
        from py2bin.requirements import required_by

        room = self._dist_info(
            'Name: thing\nRequires-Dist: pyobjc; sys_platform == "darwin"\n'
        )
        self.assertEqual(required_by(room), ["pyobjc"])

    def test_nothing_to_read_is_not_an_error(self):
        from py2bin.requirements import required_by

        self.assertEqual(required_by(Path(tempfile.mkdtemp())), [])


class AskedRatherThanListed(unittest.TestCase):
    """Which project publishes a module is looked up, not remembered.

    A list holds the names somebody thought of, and a program imports the
    one they did not. `certifi` was outside the list for as long as there
    was one, so a bundle that needed its `cacert.pem` was built without it
    and failed the first time it opened a connection.
    """

    def test_an_installed_package_is_found_without_being_listed(self) -> None:
        from importlib.metadata import packages_distributions

        from py2bin.requirements import KNOWN_PROJECTS, publisher_of

        known = packages_distributions()
        outside = [
            name
            for name in known
            if name not in KNOWN_PROJECTS and not name.startswith("_")
        ]
        if not outside:
            self.skipTest("every installed import name is also in the list")
        for name in sorted(outside)[:20]:
            self.assertEqual(publisher_of(name), sorted(known[name])[0], name)

    def test_the_list_is_the_fallback_and_not_the_first_word(self) -> None:
        from py2bin.requirements import publisher_of

        # An import name nothing installed provides falls back to the list.
        self.assertEqual(publisher_of("PIL"), "pillow")
        # And one nothing knows is unknown rather than guessed at.
        self.assertIsNone(publisher_of("a_module_nobody_publishes_xyzzy"))


class WatchingAddsToReading(unittest.TestCase):
    """Running the program finds what reading it cannot, and vice versa.

    Reading finds every branch without taking any of them. Running takes one
    branch and finds what only that branch knows - a directory whose name is
    read out of a file at run time. Neither is the whole answer, so watching
    adds to reading rather than replacing it.
    """

    def _project(self, where: Path) -> Path:
        (where / "app").mkdir()
        (where / "far" / "skin").mkdir(parents=True)
        (where / "far" / "skin" / "index.html").write_text("<h1>x</h1>\n")
        (where / "far" / "skin" / "site.css").write_text("body{}\n")
        (where / "app" / "which.txt").write_text("../far/skin\n")
        (where / "app" / "main.py").write_text(
            "import os\n"
            "HERE = os.path.dirname(os.path.abspath(__file__))\n"
            "folder = open(os.path.join(HERE, 'which.txt')).read().strip()\n"
            "open(os.path.join(HERE, folder, 'index.html')).read()\n"
        )
        return where / "app" / "main.py"

    def test_watching_is_on_unless_it_is_turned_off(self) -> None:
        import inspect

        from py2bin.interactive import main

        self.assertIs(
            inspect.signature(main).parameters["watch"].default,
            True,
            "a bundle missing a file it opens is worse than the cost of a run",
        )

    def test_a_name_read_at_run_time_is_found_only_by_running(self) -> None:
        from py2bin.interactive import _what_it_opens

        with tempfile.TemporaryDirectory() as spelled:
            program = self._project(Path(spelled))
            here = program.parent
            read, _skipped, _outside = _what_it_opens(program, here, watch=False)
            self.assertNotIn(
                "skin", {path.name for path in read},
                "reading should not find a directory named nowhere in the code",
            )
            both, _skipped, _outside = _what_it_opens(program, here, watch=True)
            self.assertIn(
                "skin", {path.name for path in both},
                "running it should find the directory it opened",
            )
            # And it keeps what reading already had.
            self.assertLessEqual({p.name for p in read}, {p.name for p in both})

    def test_a_program_that_never_returns_does_not_hold_up_the_build(self) -> None:
        from py2bin import interactive

        with tempfile.TemporaryDirectory() as spelled:
            where = Path(spelled)
            (where / "page.html").write_text("<h1>x</h1>\n")
            (where / "app.py").write_text(
                "import os, time\n"
                "HERE = os.path.dirname(os.path.abspath(__file__))\n"
                "open(os.path.join(HERE, 'page.html')).read()\n"
                "while True: time.sleep(3600)\n"
            )
            was, interactive._WATCH_SECONDS = interactive._WATCH_SECONDS, 2.0
            try:
                began = time.monotonic()
                carried, _s, _o = interactive._what_it_opens(
                    where / "app.py", where, watch=True
                )
            finally:
                interactive._WATCH_SECONDS = was
            self.assertLess(time.monotonic() - began, 30.0)
            self.assertIn("page.html", {path.name for path in carried})
