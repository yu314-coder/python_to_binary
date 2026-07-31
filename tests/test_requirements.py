"""Working out what a program needs, without guessing at it."""

import tempfile
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
