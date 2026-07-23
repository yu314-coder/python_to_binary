from pathlib import Path
import os
import plistlib
import struct
import subprocess
import sys
import tempfile
import unittest

from py2bin import ArtifactKind, BuildConfig, build


class BuilderTests(unittest.TestCase):
    def _project(self, root: Path) -> Path:
        entry = root / "main.py"
        entry.write_text("from helper import value\nprint(value)\n", encoding="utf-8")
        (root / "helper.py").write_text("value = 'bundled-ok'\n", encoding="utf-8")
        return entry

    def test_builds_and_runs_bin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            entry = self._project(project)
            artifact = root / "demo.bin"
            result = build(BuildConfig(entry, artifact, dependency_mode="none"))
            environment = os.environ.copy()
            environment["PY2BIN_CACHE_DIR"] = str(root / "cache")
            run = subprocess.run(
                [sys.executable, str(result.artifact)],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(run.stdout.strip(), "bundled-ok")

    def test_builds_directory_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            entry = self._project(project)
            artifact = root / "demo"
            result = build(
                BuildConfig(entry, artifact, kind=ArtifactKind.DIRECTORY, dependency_mode="none")
            )
            run = subprocess.run(
                [sys.executable, str(result.artifact / "runtime" / "bootstrap.py")],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.stdout.strip(), "bundled-ok")
            self.assertTrue((root / "demo.run").exists())

    def test_pyz_suffix_is_cleaned_at_resolved_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            entry = self._project(project)
            stale = root / "demo.pyz"
            stale.write_text("stale", encoding="utf-8")
            result = build(
                BuildConfig(
                    entry,
                    root / "demo",
                    kind=ArtifactKind.PYZ,
                    dependency_mode="none",
                    clean=True,
                )
            )
            self.assertEqual(result.artifact, stale)
            self.assertGreater(stale.stat().st_size, 5)

    def test_app_converts_png_backed_ico_and_declares_icon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            entry = self._project(project)
            png = (
                b"\x89PNG\r\n\x1a\n"
                + struct.pack(">I", 13)
                + b"IHDR"
                + struct.pack(">II", 256, 256)
                + b"\x08\x06\x00\x00\x00"
            )
            ico = root / "icon.ico"
            ico.write_bytes(
                struct.pack("<HHH", 0, 1, 1)
                + struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(png), 22)
                + png
            )
            result = build(
                BuildConfig(
                    entry,
                    root / "Demo",
                    kind=ArtifactKind.APP,
                    dependency_mode="none",
                    icon=ico,
                )
            )
            icon = result.artifact / "Contents" / "Resources" / "AppIcon.icns"
            self.assertEqual(icon.read_bytes()[:4], b"icns")
            with (result.artifact / "Contents" / "Info.plist").open("rb") as stream:
                info = plistlib.load(stream)
            self.assertEqual(info["CFBundleIconFile"], "AppIcon.icns")


if __name__ == "__main__":
    unittest.main()
