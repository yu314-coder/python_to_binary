"""The version is written once and agreed on everywhere it appears."""

import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _module_version() -> str:
    for line in (_ROOT / "src" / "py2bin" / "__init__.py").read_text().splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise AssertionError("no __version__ in py2bin/__init__.py")


class VersionAgreementTests(unittest.TestCase):
    def test_pyproject_matches_the_module(self):
        text = (_ROOT / "pyproject.toml").read_text()
        found = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        self.assertIsNotNone(found, "pyproject.toml names no version")
        self.assertEqual(
            found.group(1),
            _module_version(),
            "pyproject.toml and py2bin.__version__ disagree; a build would be "
            "labelled with whichever the backend happened to read",
        )

    def test_the_backend_reads_rather_than_repeats_it(self):
        from py2bin.build_backend import _VERSION

        self.assertEqual(_VERSION, _module_version())


class HostTargetTests(unittest.TestCase):
    """A build that names no target must behave like one that names the host's.

    Not naming --target is the ordinary way to build for the machine you are
    on, and the sealing and Windows-layout steps both asked the target whether
    it started with a platform name. Unset, that target is None, and every
    --app build without an explicit --target aborted before it was sealed.
    """

    def test_the_cli_never_asks_a_bare_target_for_its_prefix(self):
        text = (_ROOT / "src" / "py2bin" / "cli.py").read_text()
        self.assertNotIn(
            'target.startswith("darwin")',
            text.replace("chosen.startswith", "SAFE"),
        )
        self.assertNotIn(
            'target.startswith("windows-")',
            text.replace("chosen.startswith", "SAFE"),
        )
