from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from py2bin.build_backend import build_sdist, build_wheel


class BuildBackendTests(unittest.TestCase):
    def test_wheel_contains_native_subpackages(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel_name = build_wheel(directory)
            with zipfile.ZipFile(Path(directory) / wheel_name) as wheel:
                names = set(wheel.namelist())
            self.assertIn("py2bin/native/compiler.py", names)
            self.assertIn("py2bin/native/formats/pe.py", names)
            self.assertIn("py2bin/onefile.py", names)

    def test_sdist_has_metadata_docs_and_no_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            sdist_name = build_sdist(directory)
            with tarfile.open(Path(directory) / sdist_name) as archive:
                members = archive.getmembers()
                names = {member.name for member in members}
            prefix = sdist_name.removesuffix(".tar.gz")
            self.assertIn(f"{prefix}/PKG-INFO", names)
            self.assertIn(f"{prefix}/docs/DETAILED_GUIDE.md", names)
            self.assertFalse(
                any("__pycache__" in name or name.endswith((".pyc", ".pyo")) for name in names)
            )
            self.assertTrue(
                all(member.mtime >= 315532800 for member in members)
            )


if __name__ == "__main__":
    unittest.main()
