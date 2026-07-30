"""Laying a Windows program out beside the interpreter it needs."""

import tempfile
import unittest
from pathlib import Path

from py2bin.windows_bundle import (
    BundleError,
    SITE_PACKAGES,
    SITE_PACKAGES_ENTRY,
    carry_packages,
    carry_runtime,
    name_site_packages,
)

_PTH = "python314.zip\n.\n\n# Uncomment to run site.main() automatically\n#import site\n"


class PathFileTests(unittest.TestCase):
    def test_the_entry_is_spelled_the_way_windows_reads_it(self):
        # Built on a Mac, read by Windows: the separator is the target's.
        self.assertEqual(SITE_PACKAGES_ENTRY, "Lib\\site-packages")

    def test_site_packages_is_named_after_the_directory_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "python314._pth").write_text(_PTH)
            self.assertEqual(name_site_packages(root), 1)
            lines = [line for line in
                     (root / "python314._pth").read_text().splitlines()
                     if line and not line.startswith("#")]
            self.assertEqual(lines, ["python314.zip", ".", "Lib\\site-packages"])

    def test_naming_it_twice_changes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "python314._pth").write_text(_PTH)
            name_site_packages(root)
            before = (root / "python314._pth").read_text()
            self.assertEqual(name_site_packages(root), 0)
            self.assertEqual((root / "python314._pth").read_text(), before)

    def test_an_interpreter_with_no_path_file_is_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(name_site_packages(Path(directory)), 0)


class CarryTests(unittest.TestCase):
    def test_packages_land_in_site_packages_and_are_on_the_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "out"
            root.mkdir()
            (root / "python314._pth").write_text(_PTH)
            source = Path(directory) / "site"
            (source / "widget").mkdir(parents=True)
            (source / "widget" / "__init__.py").write_bytes(b"x = 1\n")
            carry_packages(root, (source,))
            self.assertTrue(
                (root / SITE_PACKAGES / "widget" / "__init__.py").is_file()
            )
            self.assertIn(
                "Lib\\site-packages",
                (root / "python314._pth").read_text(),
            )

    def test_a_runtime_without_a_dll_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "runtime"
            empty.mkdir()
            with self.assertRaises(BundleError):
                carry_runtime(Path(directory) / "out", empty)

    def test_a_runtime_is_copied_whole(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            (runtime / "Lib").mkdir(parents=True)
            (runtime / "python314.dll").write_bytes(b"\x00" * 32)
            (runtime / "Lib" / "os.py").write_bytes(b"pass\n")
            root = Path(directory) / "out"
            root.mkdir()
            copied = carry_runtime(root, runtime)
            self.assertEqual(copied, 37)
            self.assertTrue((root / "python314.dll").is_file())
            self.assertTrue((root / "Lib" / "os.py").is_file())


if __name__ == "__main__":
    unittest.main()
