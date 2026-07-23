from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import tempfile
import unittest
import zipfile

from py2bin.cli import main
from py2bin.freezer import freeze, inspect_wheel
from py2bin.runtime_packs import MANIFEST_NAME
from py2bin.wheel_builder import build_payload_wheel


class PayloadWheelTests(unittest.TestCase):
    @staticmethod
    def _runtime_pack(root: Path) -> Path:
        pack = root / "runtime-pack"
        executable = pack / "runtime" / "bin" / "python3"
        executable.parent.mkdir(parents=True)
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        (pack / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "schema": 1,
                    "target": "linux-x86_64",
                    "python": "3.11.9",
                    "executable": "runtime/bin/python3",
                    "environment": {"PYTHONHOME": "runtime"},
                }
            ),
            encoding="utf-8",
        )
        return pack

    def test_builds_standard_pure_python_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            package = source / "demo"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("value = 42\n", encoding="utf-8")
            (package / "data.txt").write_text("payload\n", encoding="utf-8")
            result = build_payload_wheel(
                source,
                root / "dist",
                name="demo-package",
                version="1.2.3",
            )
            info = inspect_wheel(result.wheel)
            self.assertEqual(info.name, "demo-package")
            self.assertEqual(info.top_levels, ("demo",))
            self.assertEqual(result.tag, "py3-none-any")
            with zipfile.ZipFile(result.wheel) as archive:
                names = set(archive.namelist())
                record = archive.read("demo_package-1.2.3.dist-info/RECORD")
            self.assertIn("demo/data.txt", names)
            self.assertIn(b"demo/__init__.py,sha256=", record)

    def test_native_payload_requires_truthful_target_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            package = source / "native_demo"
            package.mkdir(parents=True)
            extension = package / "speed.cp311-win_amd64.pyd"
            extension.write_bytes(b"MZ-prebuilt-placeholder")
            (package / "speed.pyx").write_text("cpdef int add(int a, int b): return a+b\n")
            with self.assertRaisesRegex(ValueError, "portable tag"):
                build_payload_wheel(
                    source,
                    root / "dist",
                    name="native-demo",
                    version="1.0",
                )
            result = build_payload_wheel(
                source,
                root / "dist",
                name="native-demo",
                version="1.0",
                python_tag="cp311",
                abi_tag="cp311",
                platform_tag="win_amd64",
            )
            self.assertEqual(
                result.native_files,
                ("native_demo/speed.cp311-win_amd64.pyd",),
            )
            self.assertEqual(result.cython_sources, ("native_demo/speed.pyx",))
            self.assertEqual(
                inspect_wheel(result.wheel).platform_tag,
                "win_amd64",
            )

    def test_wheel_cli_reports_that_cython_source_was_not_compiled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "module.pyx").write_text("value = 42\n", encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "wheel",
                        str(source),
                        "--output-dir",
                        str(root / "dist"),
                        "--name",
                        "cython-source",
                        "--version",
                        "0.1",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertIn("Cython source files were packaged but not compiled", output.getvalue())

    def test_created_wheel_flows_into_frozen_runtime_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel_source = root / "wheel-source"
            package = wheel_source / "demo"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("value = 42\n", encoding="utf-8")
            wheel = build_payload_wheel(
                wheel_source,
                root / "dist",
                name="demo",
                version="1.0",
            ).wheel
            application = root / "application"
            application.mkdir()
            entry = application / "main.py"
            entry.write_text("import demo\nprint(demo.value)\n", encoding="utf-8")
            result = freeze(
                entry,
                root / "bundle",
                application,
                wheels=(wheel,),
                runtime_pack=self._runtime_pack(root),
                target="linux-x86_64",
                onefile=False,
            )
            self.assertTrue(
                (result.bundle / "site-packages" / "demo" / "__init__.py").is_file()
            )


if __name__ == "__main__":
    unittest.main()
