from pathlib import Path
import os
import platform
import plistlib
import struct
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

from py2bin.freezer import zip_bytecode, _frozen_macos_app, _shell_launcher, extract_wheel
from py2bin.native.launcher import macos_shell_launcher
from py2bin.onefile import _powershell_script, create_onefile


class FreezerTests(unittest.TestCase):
    def test_windows_onefile_script_uses_launcher_environment_without_wmi(self):
        script = _powershell_script(
            offset=1234,
            digest="0" * 64,
            launcher="Demo.exe",
        )
        self.assertIn("$env:PY2BIN_ONEFILE_SELF", script)
        self.assertIn("$env:PY2BIN_ONEFILE_COMMAND", script)
        self.assertNotIn("Get-CimInstance", script)
        self.assertNotIn("Win32_Process", script)
        self.assertNotIn("Join-Path", script)
        self.assertNotIn("Test-Path", script)
        self.assertNotIn("Remove-Item", script)
        self.assertNotIn("Move-Item", script)
        self.assertNotIn("New-Object", script)
        self.assertEqual(
            script.count("if(![IO.File]::Exists($m))"),
            2,
        )
        self.assertLess(
            script.index("if(![IO.File]::Exists($m))"),
            script.index("[Threading.Mutex]::new"),
        )
        self.assertIn(
            "$si=[Diagnostics.ProcessStartInfo]::new()",
            script,
        )

    @unittest.skipUnless(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "self-extracting Mach-O runs only on Apple Silicon",
    )
    def test_onefile_macho_extracts_once_and_forwards_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload"
            payload.mkdir()
            inner = payload / "inner.bin"
            inner.write_text(
                '#!/bin/sh\nprintf "onefile:%s" "$1"\n',
                encoding="utf-8",
            )
            inner.chmod(0o755)
            output = root / "Demo.bin"
            result = create_onefile(
                payload,
                output,
                target="darwin-arm64",
                launcher=inner,
            )
            environment = os.environ.copy()
            environment["PY2BIN_CACHE_DIR"] = str(root / "cache")
            first = subprocess.run(
                [str(output), "forwarded"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            second = subprocess.run(
                [str(output), "cached"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(first.stdout, "onefile:forwarded")
            self.assertEqual(second.stdout, "onefile:cached")
            self.assertEqual(output.read_bytes()[:4], b"\xcf\xfa\xed\xfe")
            self.assertGreater(result.archive_bytes, 0)
            self.assertEqual(
                len(list((root / "cache").rglob(".py2bin-complete"))),
                1,
            )

    def test_native_macos_launcher_is_a_macho(self):
        image = macos_shell_launcher("exit 0", machine="arm64")
        self.assertEqual(image[:4], b"\xcf\xfa\xed\xfe")

    def test_x86_64_macos_launcher_reads_the_initial_stack(self):
        # The x86-64 Mach-O writer uses LC_UNIXTHREAD, which starts execution
        # at the raw entry point with argc/argv on the initial process stack.
        # Only the arm64 image uses LC_MAIN, where they arrive in registers.
        # Reading rdi/rsi/rdx here would read uninitialised registers, and the
        # launcher would exit 64 instead of running the program.
        image = macos_shell_launcher("exit 0", machine="x86_64")
        prologue = (
            b"\x49\x89\xe5"  # mov r13, rsp
            b"\x4d\x8b\x65\x00"  # mov r12, [r13]  (argc)
            b"\x4d\x8d\x75\x08"  # lea r14, [r13+8] (argv)
        )
        self.assertIn(prologue, image)
        # The LC_MAIN register convention must not be used for this target.
        self.assertNotIn(b"\x49\x89\xfc\x49\x89\xf6\x49\x89\xd7", image)

    def test_x86_64_app_launcher_is_emitted_unsigned(self):
        # arm64 macOS requires a code signature, so that launcher embeds an
        # ad-hoc one sealing Info.plist/CodeResources. Intel macOS still loads
        # unsigned executables, so the same request must produce a valid
        # x86-64 Mach-O rather than being refused.
        image = macos_shell_launcher(
            "exit 0",
            machine="x86_64",
            info_plist=b"<plist/>",
            code_resources=b"<plist/>",
        )
        self.assertEqual(image[:4], b"\xcf\xfa\xed\xfe")
        # cputype in the Mach-O header: CPU_TYPE_X86_64 is 0x01000007.
        self.assertEqual(
            int.from_bytes(image[4:8], "little"), 0x01000007
        )

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

    def test_compact_wheel_keeps_runtime_payload_and_omits_tests_and_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "demo-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("demo/__init__.py", "value = 42\n")
                archive.writestr("demo/data/schema.json", "{}")
                archive.writestr("demo/native.pyd", b"native")
                archive.writestr("demo/compiled_only.pyc", b"runtime")
                archive.writestr("demo/tests/test_api.py", "assert True\n")
                archive.writestr(
                    "demo/__pycache__/module.pyc",
                    b"bytecode",
                )
                archive.writestr(
                    "demo-1.0.dist-info/METADATA",
                    "Name: demo\nVersion: 1.0\n",
                )
            destination = root / "packages"
            destination.mkdir()
            count = extract_wheel(wheel, destination, compact=True)
            self.assertEqual(count, 5)
            self.assertTrue((destination / "demo" / "__init__.py").is_file())
            self.assertTrue(
                (destination / "demo" / "data" / "schema.json").is_file()
            )
            self.assertTrue((destination / "demo" / "native.pyd").is_file())
            self.assertTrue(
                (destination / "demo" / "compiled_only.pyc").is_file()
            )
            self.assertTrue(
                (
                    destination
                    / "demo-1.0.dist-info"
                    / "METADATA"
                ).is_file()
            )
            self.assertFalse((destination / "demo" / "tests").exists())
            self.assertFalse(
                (destination / "demo" / "__pycache__").exists()
            )

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
            with mock.patch(
                "py2bin.native.launcher.platform.machine", return_value="arm64"
            ):
                launcher = _frozen_macos_app(
                    payload,
                    app,
                    "ManimStudio",
                    payload_launcher,
                    icon,
                    Path("runtime/bin/python3"),
                    {"PYTHONHOME": "runtime"},
                    "darwin-arm64",
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


class ZipBytecodeTests(unittest.TestCase):
    """The carried library, packed into the archive the interpreter expects."""

    def _bundle(self, root: Path) -> Path:
        """A bundle shaped like the real thing, with one native package."""

        bundle = root / "App.app"
        library = bundle / "Contents" / "lib" / "python3.14"
        (library / "json").mkdir(parents=True)
        (library / "lib-dynload").mkdir(parents=True)
        (library / "native" / "__pycache__").mkdir(parents=True)
        (library / "os.pyc").write_bytes(b"os bytecode" * 40)
        (library / "json" / "__init__.pyc").write_bytes(b"json bytecode" * 40)
        (library / "lib-dynload" / "select.so").write_bytes(b"\xcf\xfa\xed\xfe" * 10)
        # A package holding an extension: dyld needs a file, so it stays put.
        (library / "native" / "ext.so").write_bytes(b"\xcf\xfa\xed\xfe" * 10)
        (library / "native" / "__pycache__" / "helper.cpython-314.pyc").write_bytes(
            b"helper" * 40
        )
        return bundle

    def test_the_library_moves_into_the_archive_the_interpreter_looks_for(self):
        # `{prefix}/lib/pythonXY.zip` is on sys.path whether or not it exists,
        # so no path setup is needed - the name has to be exactly that.
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._bundle(Path(directory))
            zip_bytecode(bundle)
            archive = bundle / "Contents" / "lib" / "python314.zip"
            self.assertTrue(archive.is_file())
            with zipfile.ZipFile(archive) as packed:
                names = set(packed.namelist())
        self.assertIn("os.pyc", names)
        self.assertIn("json/__init__.pyc", names)

    def test_a_package_holding_an_extension_is_left_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._bundle(Path(directory))
            zip_bytecode(bundle)
            library = bundle / "Contents" / "lib" / "python3.14"
            with zipfile.ZipFile(library.parent / "python314.zip") as packed:
                names = set(packed.namelist())
            self.assertNotIn("native/helper.pyc", names)
            self.assertTrue(
                (library / "native" / "__pycache__" /
                 "helper.cpython-314.pyc").is_file()
            )
            self.assertTrue((library / "native" / "ext.so").is_file())

    def test_the_name_in_the_archive_is_the_one_import_asks_for(self):
        # `__pycache__/helper.cpython-314.pyc` is imported as `helper`, so a
        # cache directory in the archive would put every module out of reach.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "App.app"
            library = bundle / "Contents" / "lib" / "python3.14" / "pkg"
            (library / "__pycache__").mkdir(parents=True)
            (library / "__pycache__" / "part.cpython-314.pyc").write_bytes(b"x" * 90)
            zip_bytecode(bundle)
            with zipfile.ZipFile(
                bundle / "Contents" / "lib" / "python314.zip"
            ) as packed:
                self.assertEqual(packed.namelist(), ["pkg/part.pyc"])

    def test_storing_is_offered_for_a_filesystem_that_compresses_already(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._bundle(Path(directory))
            zip_bytecode(bundle, compress=False)
            with zipfile.ZipFile(
                bundle / "Contents" / "lib" / "python314.zip"
            ) as packed:
                methods = {item.compress_type for item in packed.infolist()}
        self.assertEqual(methods, {zipfile.ZIP_STORED})

    def test_a_bundle_with_no_carried_library_is_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "App.app"
            (bundle / "Contents" / "MacOS").mkdir(parents=True)
            self.assertEqual(zip_bytecode(bundle), 0)
            self.assertEqual(list(bundle.rglob("*.zip")), [])
