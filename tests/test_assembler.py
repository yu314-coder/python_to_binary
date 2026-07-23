from pathlib import Path
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

from py2bin.assembler import assemble
from py2bin.freezer import freeze, inspect_wheel
from py2bin.native.formats.pe import write_pe_shell_launcher
from py2bin.runtime_packs import MANIFEST_NAME, inspect_runtime_pack
from py2bin.windows_icon import (
    RT_GROUP_ICON,
    RT_ICON,
    RT_VERSION,
    _existing_resources,
    _pe_layout,
)


class AssemblerTests(unittest.TestCase):
    def _runtime_pack(
        self,
        root: Path,
        target: str = "linux-x86_64",
        python: str = "3.11.9",
    ) -> Path:
        pack = root / "runtime-pack"
        executable = pack / "runtime" / "bin" / "python3"
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        if target.startswith("windows-"):
            machine = "arm64" if target.endswith("arm64") else "x86_64"
            executable.write_bytes(
                write_pe_shell_launcher(
                    b"cmd.exe /d /c exit 0 ",
                    machine,
                )
            )
            (executable.parent / "python311._pth").write_text(
                "python311.zip\n.\n#import site\n",
                encoding="utf-8",
            )
            (executable.parent / "pythonw.exe").write_bytes(
                write_pe_shell_launcher(
                    b"cmd.exe /d /c exit 0 ",
                    machine,
                    windowed=True,
                )
            )
        (pack / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "schema": 1,
                    "target": target,
                    "python": python,
                    "executable": "runtime/bin/python3",
                    "environment": {
                        "PYTHONHOME": "runtime",
                        "LD_LIBRARY_PATH": "runtime/lib",
                    },
                }
            ),
            encoding="utf-8",
        )
        return pack

    def _wheel(
        self,
        root: Path,
        filename: str = "demo-1.0-py3-none-any.whl",
        requirements: tuple[str, ...] = (),
    ) -> Path:
        wheel = root / filename
        metadata = ["Metadata-Version: 2.1", "Name: demo", "Version: 1.0"]
        metadata.extend(f"Requires-Dist: {item}" for item in requirements)
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("demo/__init__.py", "value = 42\n")
            archive.writestr("demo-1.0.dist-info/top_level.txt", "demo\n")
            archive.writestr(
                "demo-1.0.dist-info/METADATA", "\n".join(metadata) + "\n\n"
            )
        return wheel

    def test_runtime_pack_and_wheel_are_inspected_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = self._runtime_pack(root)
            wheel = self._wheel(root)
            pack_info = inspect_runtime_pack(pack)
            wheel_info = inspect_wheel(wheel)
            self.assertEqual(pack_info.target, "linux-x86_64")
            self.assertEqual(pack_info.executable, Path("runtime/bin/python3"))
            self.assertEqual(wheel_info.name, "demo")
            self.assertEqual(wheel_info.top_levels, ("demo",))

    def test_cross_target_freeze_uses_pack_and_target_wheel_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            entry = project / "main.py"
            entry.write_text("import demo\nprint(demo.value)\n", encoding="utf-8")
            result = freeze(
                entry,
                root / "bundle",
                project,
                wheels=(self._wheel(root),),
                runtime_pack=self._runtime_pack(root),
                target="linux-x86_64",
                onefile=False,
            )
            self.assertEqual(result.target, "linux-x86_64")
            self.assertEqual(result.python, "3.11.9")
            self.assertTrue((result.bundle / "site-packages" / "demo" / "__init__.py").is_file())
            self.assertTrue((result.bundle / "runtime" / "bin" / "python3").is_file())
            self.assertFalse((result.bundle / MANIFEST_NAME).exists())
            self.assertIn("demo", result.distributions)

    def test_windows_app_is_windowed_onefile_with_stdlib_zip_and_icon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "main.py"
            entry.write_text("print('hello')\n", encoding="utf-8")
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
                + struct.pack(
                    "<BBBBHHII", 128, 128, 0, 0, 1, 32, len(png), 22
                )
                + png
            )
            result = freeze(
                entry,
                root / "Demo-0.2.1",
                root,
                dependency_mode="none",
                runtime_pack=self._runtime_pack(
                    root,
                    target="windows-x86_64",
                ),
                target="windows-x86_64",
                icon=icon,
                app=True,
            )
            image = result.bundle.read_bytes()
            pe_offset = int.from_bytes(image[0x3C:0x40], "little")
            optional = pe_offset + 24
            subsystem = int.from_bytes(
                image[optional + 68:optional + 70],
                "little",
            )
            resource_rva = int.from_bytes(
                image[optional + 112 + 16:optional + 112 + 20],
                "little",
            )
            marker = b"\nPY2BIN-ONEFILE-PAYLOAD-V1:"
            marker_at = image.index(marker)
            payload_at = image.index(b"\n", marker_at + len(marker)) + 1
            with zipfile.ZipFile(io.BytesIO(image[payload_at:])) as archive:
                python_path = archive.read(
                    "Demo-0.2.1._pth"
                ).decode("utf-8")
                dll_path = archive.read(
                    "runtime/bin/python311._pth"
                ).decode("utf-8")
                inner_image = archive.read("Demo-0.2.1.exe")
                manifest = json.loads(
                    archive.read("py2bin-freeze.json").decode("utf-8")
                )
                bootstrap = archive.read(
                    "py2bin_bootstrap.py"
                ).decode("utf-8")
            outer_resources = _existing_resources(image, _pe_layout(image))
            inner_layout = _pe_layout(inner_image)
            inner_resources = _existing_resources(
                inner_image,
                inner_layout,
            )
            inner_optional = int(inner_layout["optional"])
            inner_subsystem = struct.unpack_from(
                "<H",
                inner_image,
                inner_optional + 68,
            )[0]
            version_key = (RT_VERSION, 1, 0x0409)
            self.assertTrue(result.onefile)
            self.assertEqual(result.bundle.suffix, ".exe")
            self.assertEqual(result.bundle.name, "Demo-0.2.1.exe")
            self.assertEqual(result.files, 1)
            self.assertEqual(image[:2], b"MZ")
            self.assertEqual(subsystem, 2)
            self.assertEqual(inner_subsystem, 2)
            self.assertGreater(resource_rva, 0)
            self.assertIn(b"PY2BIN-ONEFILE-PAYLOAD-V1", image)
            self.assertIn(b"GetModuleFileNameW", image)
            self.assertIn(b"GetCommandLineW", image)
            self.assertIn(b"SetEnvironmentVariableW", image)
            self.assertIn(
                "PY2BIN_ONEFILE_SELF".encode("utf-16-le"),
                image,
            )
            self.assertIn("python311.zip\n", python_path)
            self.assertIn("import site\n", python_path)
            self.assertEqual(dll_path, python_path)
            self.assertIn(version_key, outer_resources)
            self.assertIn(version_key, inner_resources)
            self.assertIn(
                "Demo-0.2.1".encode("utf-16-le"),
                outer_resources[version_key].data,
            )
            self.assertIn(
                "Demo-0.2.1".encode("utf-16-le"),
                inner_resources[version_key].data,
            )
            for resources in (outer_resources, inner_resources):
                self.assertIn((RT_GROUP_ICON, 1, 0x0409), resources)
                self.assertTrue(
                    any(key[0] == RT_ICON for key in resources)
                )
            self.assertEqual(
                manifest["windows_app_user_model_id"],
                "PythonToBinary.Demo021",
            )
            self.assertIn(
                "SetCurrentProcessExplicitAppUserModelID",
                bootstrap,
            )
            self.assertIn(
                "setter.argtypes = [ctypes.c_wchar_p]",
                bootstrap,
            )
            self.assertIn(
                "setter.restype = ctypes.c_long",
                bootstrap,
            )
            self.assertNotIn("import json", bootstrap)
            self.assertNotIn("from pathlib import Path", bootstrap)
            self.assertIn(
                "except BaseException:\n        import traceback",
                bootstrap,
            )
            compile(bootstrap, "py2bin_bootstrap.py", "exec")
            bootstrap_root = root / "bootstrap-run"
            bootstrap_app = bootstrap_root / "app"
            bootstrap_app.mkdir(parents=True)
            (bootstrap_root / "site-packages").mkdir()
            (bootstrap_root / "py2bin_bootstrap.py").write_text(
                bootstrap,
                encoding="utf-8",
            )
            (bootstrap_app / "main.py").write_text(
                "print('bootstrap-ok')\n",
                encoding="utf-8",
            )
            bootstrap_run = subprocess.run(
                [sys.executable, str(bootstrap_root / "py2bin_bootstrap.py")],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(bootstrap_run.stdout, "bootstrap-ok\n")

    def test_windows_app_onedir_uses_windowed_runtime_without_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "main.py"
            entry.write_text("print('hello')\n", encoding="utf-8")
            result = freeze(
                entry,
                root / "Demo",
                root,
                dependency_mode="none",
                runtime_pack=self._runtime_pack(
                    root,
                    target="windows-x86_64",
                ),
                target="windows-x86_64",
                app=True,
                onefile=False,
            )
            self.assertFalse(result.onefile)
            self.assertTrue(result.bundle.is_dir())
            launcher = result.bundle / "Demo.exe"
            image = launcher.read_bytes()
            layout = _pe_layout(image)
            resources = _existing_resources(image, layout)
            optional = int(layout["optional"])
            subsystem = struct.unpack_from("<H", image, optional + 68)[0]
            version_key = (RT_VERSION, 1, 0x0409)
            self.assertEqual(
                subsystem,
                2,
            )
            self.assertIn(version_key, resources)
            self.assertIn(
                "Demo".encode("utf-16-le"),
                resources[version_key].data,
            )
            self.assertFalse(
                any(
                    key[0] in {RT_ICON, RT_GROUP_ICON}
                    for key in resources
                )
            )
            manifest = json.loads(
                (result.bundle / "py2bin-freeze.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["windows_app_user_model_id"],
                "PythonToBinary.Demo",
            )
            self.assertIn(
                "SetCurrentProcessExplicitAppUserModelID",
                (
                    result.bundle / "py2bin_bootstrap.py"
                ).read_text(encoding="utf-8"),
            )

    def test_cross_target_rejects_wrong_platform_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "main.py"
            entry.write_text("import demo\n", encoding="utf-8")
            wheel = self._wheel(root, "demo-1.0-cp311-cp311-win_amd64.whl")
            with self.assertRaisesRegex(ValueError, "does not match target"):
                freeze(
                    entry,
                    root / "bundle",
                    root,
                    wheels=(wheel,),
                    runtime_pack=self._runtime_pack(root),
                )

    def test_cross_target_requires_unconditional_wheel_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "main.py"
            entry.write_text("import demo\n", encoding="utf-8")
            wheel = self._wheel(root, requirements=("missing-runtime>=1",))
            with self.assertRaisesRegex(ValueError, "complete wheel closure"):
                freeze(
                    entry,
                    root / "bundle",
                    root,
                    wheels=(wheel,),
                    runtime_pack=self._runtime_pack(root),
                )

    def test_windows_freeze_collects_pywinpty_native_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "main.py"
            entry.write_text(
                "import winpty\nfrom winpty.enums import Backend\n",
                encoding="utf-8",
            )
            wheel = root / "pywinpty-3.0.5-cp311-cp311-win_amd64.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("winpty/__init__.py", "class PTY: pass\n")
                archive.writestr("winpty/enums.py", "class Backend: pass\n")
                archive.writestr(
                    "winpty/_winpty.cp311-win_amd64.pyd",
                    b"native-extension",
                )
                archive.writestr("winpty/conpty.dll", b"conpty")
                archive.writestr("winpty/winpty.dll", b"winpty")
                archive.writestr("winpty/winpty-agent.exe", b"agent")
                archive.writestr("winpty/OpenConsole.exe", b"console")
                archive.writestr(
                    "pywinpty-3.0.5.dist-info/top_level.txt",
                    "winpty\n",
                )
                archive.writestr(
                    "pywinpty-3.0.5.dist-info/METADATA",
                    "Metadata-Version: 2.1\n"
                    "Name: pywinpty\n"
                    "Version: 3.0.5\n\n",
                )
            result = freeze(
                entry,
                root / "TerminalApp",
                root,
                wheels=(wheel,),
                runtime_pack=self._runtime_pack(
                    root,
                    target="windows-x86_64",
                ),
                target="windows-x86_64",
            )
            image = result.bundle.read_bytes()
            marker = b"\nPY2BIN-ONEFILE-PAYLOAD-V1:"
            marker_at = image.index(marker)
            payload_at = image.index(b"\n", marker_at + len(marker)) + 1
            with zipfile.ZipFile(io.BytesIO(image[payload_at:])) as archive:
                names = set(archive.namelist())
            self.assertIn(
                "site-packages/winpty/_winpty.cp311-win_amd64.pyd",
                names,
            )
            self.assertIn("site-packages/winpty/conpty.dll", names)
            self.assertIn("site-packages/winpty/winpty-agent.exe", names)
            self.assertIn("pywinpty", result.distributions)

    def test_assemble_uses_native_then_compatible_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native_source = root / "native.py"
            native_source.write_text("print('native')\n", encoding="utf-8")
            native = assemble(
                native_source,
                root / "native.bin",
                target="linux-x86_64",
            )
            self.assertEqual(native.backend, "native")
            self.assertEqual(native.artifact.read_bytes()[:4], b"\x7fELF")

            dynamic_source = root / "dynamic.py"
            dynamic_source.write_text(
                "for value in range(1):\n    print(value)\n", encoding="utf-8"
            )
            compatible = assemble(
                dynamic_source,
                root / "dynamic",
                target="linux-x86_64",
                source_root=root,
                runtime_pack=self._runtime_pack(root),
            )
            self.assertEqual(compatible.backend, "compatible")
            self.assertTrue(compatible.launcher.is_file())
            self.assertTrue(compatible.artifact.is_file())
            self.assertEqual(compatible.artifact.suffix, ".bin")
            self.assertEqual(compatible.artifact.read_bytes()[:4], b"\x7fELF")


if __name__ == "__main__":
    unittest.main()
