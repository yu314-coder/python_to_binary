from pathlib import Path
from unittest import mock
import tempfile
import unittest

from py2bin.analyzer import _distribution_closure, analyze


class AnalyzeTests(unittest.TestCase):
    def test_dependency_closure_visits_more_than_one_distribution(self):
        class Distribution:
            def __init__(self, name, requires=()):
                self.metadata = {"Name": name}
                self.requires = requires

        distributions = [
            Distribution("First_Package", ["Second.Package>=1"]),
            Distribution("Second.Package"),
        ]
        with mock.patch(
            "py2bin.analyzer.metadata.distributions", return_value=distributions
        ):
            self.assertEqual(
                _distribution_closure({"First-Package"}),
                {"First_Package", "Second.Package"},
            )

    def test_finds_local_and_standard_library_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "main.py"
            entry.write_text("import json\nimport helper\n", encoding="utf-8")
            (root / "helper.py").write_text("import pathlib\n", encoding="utf-8")
            result = analyze(entry, root, dependency_mode="none")
            self.assertEqual(result.modules, {"json", "helper", "pathlib"})
            self.assertIn(root / "helper.py", result.local_files)
            self.assertFalse(result.unresolved)

    def test_reports_missing_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "main.py"
            entry.write_text("import definitely_not_installed_xyz\n", encoding="utf-8")
            result = analyze(entry, root, dependency_mode="imported")
            self.assertEqual(result.unresolved, {"definitely_not_installed_xyz"})

    def test_follows_relative_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            entry = package / "main.py"
            entry.write_text("from . import helper\n", encoding="utf-8")
            helper = package / "helper.py"
            helper.write_text("import definitely_not_installed_relative_xyz\n", encoding="utf-8")
            result = analyze(entry, root, dependency_mode="imported")
            self.assertIn(helper, result.local_files)
            self.assertEqual(result.unresolved, {"definitely_not_installed_relative_xyz"})


if __name__ == "__main__":
    unittest.main()
