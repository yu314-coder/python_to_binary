from pathlib import Path
import platform
import plistlib
import struct
import subprocess
import tempfile
import unittest
import zipfile

from py2bin.freezer import _frozen_macos_app, _shell_launcher, extract_wheel
from py2bin.native.launcher import macos_shell_launcher


class FreezerTests(unittest.TestCase):
    def test_native_macos_launcher_is_a_macho(self):
        image = macos_shell_launcher("exit 0", machine="arm64")
        self.assertEqual(image[:4], b"\xcf\xfa\xed\xfe")

    @unittest.skipUnless(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "native launcher execution requires Apple Silicon",
    )
    def test_native_macos_launcher_forwards_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "launcher"
            launcher.write_bytes(macos_shell_launcher("printf '%s' \"$1\""))
            launcher.chmod(0o755)
            run = subprocess.run(
                [str(launcher), "forwarded"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.stdout, "forwarded")

    def test_posix_launcher_has_no_path_dependent_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "App.bin"
            _shell_launcher(launcher, Path("runtime/bin/python3"), {"PYTHONHOME": "runtime"})
            text = launcher.read_text(encoding="utf-8")
            self.assertNotIn("dirname", text)
            self.assertNotIn("/usr/bin/env", text)
            self.assertIn('exec "$ROOT/runtime/bin/python3"', text)

    def test_extracts_wheel_packages_data_native_files_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "demo-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("demo/__init__.py", "value = 42\n")
                archive.writestr("demo/data/model.json", "{}")
                archive.writestr("demo/native.pyd", b"native")
                archive.writestr("demo-1.0.dist-info/METADATA", "Name: demo\n")
                archive.writestr("demo-1.0.data/purelib/plugin.py", "enabled=True\n")
                archive.writestr("../escape", "bad")
            destination = root / "packages"
            destination.mkdir()
            count = extract_wheel(wheel, destination)
            self.assertEqual(count, 5)
            self.assertTrue((destination / "demo" / "data" / "model.json").exists())
            self.assertTrue((destination / "demo" / "native.pyd").exists())
            self.assertTrue((destination / "demo-1.0.dist-info" / "METADATA").exists())
            self.assertTrue((destination / "plugin.py").exists())
            self.assertFalse((root / "escape").exists())

    def test_frozen_macos_app_wraps_payload_and_icon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload"
            payload.mkdir()
            payload_launcher = payload / "ManimStudio.bin"
            payload_launcher.write_text("#!/bin/sh\n", encoding="utf-8")
            png = (
                b"\x89PNG\r\n\x1a\n"
                + struct.pack(">I", 13)
                + b"IHDR"
                + struct.pack(">II", 128, 128)
                + b"\x08\x06\x00\x00\x00"
            )
            icon = root / "icon.ico"
            icon.write_bytes(
                struct.pack("<HHH", 0, 1, 1)
                + struct.pack("<BBBBHHII", 128, 128, 0, 0, 1, 32, len(png), 22)
                + png
            )
            app = root / "ManimStudio.app"
            launcher = _frozen_macos_app(
                payload, app, "ManimStudio", payload_launcher, icon
            )
            self.assertTrue(launcher.is_file())
            self.assertTrue(
                (app / "Contents" / "Resources" / "bundle" / "ManimStudio.bin").is_file()
            )
            self.assertEqual(
                (app / "Contents" / "Resources" / "AppIcon.icns").read_bytes()[:4],
                b"icns",
            )
            self.assertTrue(
                (app / "Contents" / "_CodeSignature" / "CodeResources").is_file()
            )
            with (app / "Contents" / "Info.plist").open("rb") as stream:
                info = plistlib.load(stream)
            self.assertEqual(info["CFBundleIconFile"], "AppIcon.icns")


if __name__ == "__main__":
    unittest.main()
