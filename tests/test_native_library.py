from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from py2bin.cli import main
from py2bin.native import audit_native_library


class NativeLibraryAuditTests(unittest.TestCase):
    def test_audit_uses_real_frontend_and_classifies_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "samplelib"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "maths.py").write_text(
                "def sum_to(limit: int) -> int:\n"
                "    total = 0\n"
                "    for value in range(limit + 1):\n"
                "        total += value\n"
                "    return total\n",
                encoding="utf-8",
            )
            (package / "dynamic.py").write_text(
                "def make_list(value: int):\n"
                "    return [value]\n",
                encoding="utf-8",
            )
            (package / "engine.so").write_bytes(b"already-native")
            (package / "viewer.js").write_text("export const ok = true;\n", encoding="utf-8")

            report = audit_native_library(package, source_roots=(root,))

            by_name = {item.name: item for item in report.functions}
            self.assertTrue(by_name["sum_to"].native)
            self.assertFalse(by_name["make_list"].native)
            self.assertFalse(report.fully_native)
            self.assertEqual(report.native_payloads, ((package / "engine.so").resolve(),))
            self.assertEqual(report.web_assets, ((package / "viewer.js").resolve(),))
            self.assertTrue(
                any("CPython/C ABI adapter" in blocker for blocker in report.blockers)
            )
            self.assertIn(
                "signed 64-bit native integer subset",
                by_name["make_list"].reason,
            )

    def test_cli_json_and_strict_library_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "numberlib"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "helper.py").write_text(
                "def total(limit: int) -> int:\n"
                "    result = 0\n"
                "    for value in range(limit + 1):\n"
                "        result += value\n"
                "    return result\n",
                encoding="utf-8",
            )
            entry = root / "main.py"
            entry.write_text(
                "from numberlib.helper import total\n"
                "raise SystemExit(total(9))\n",
                encoding="utf-8",
            )

            output = StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "audit-library",
                        str(package),
                        "--source-root",
                        str(root),
                        "--json",
                        "--strict",
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["fully_native"])
            self.assertEqual(payload["native_functions"], 1)

            artifact = root / "main.exe"
            status = main(
                [
                    "compile",
                    str(entry),
                    "--source-root",
                    str(root),
                    "--strict-library-root",
                    str(package),
                    "--target",
                    "windows-x86_64",
                    "--output",
                    str(artifact),
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(artifact.read_bytes()[:2], b"MZ")

    def test_strict_library_build_rejects_dynamic_function_without_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "dynamiclib"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "bad.py").write_text(
                "def values(item):\n"
                "    return [item]\n",
                encoding="utf-8",
            )
            entry = root / "main.py"
            entry.write_text("raise SystemExit(0)\n", encoding="utf-8")
            artifact = root / "blocked"
            error = StringIO()
            with redirect_stderr(error):
                status = main(
                    [
                        "compile",
                        str(entry),
                        "--source-root",
                        str(root),
                        "--strict-library-root",
                        str(package),
                        "--target",
                        "linux-x86_64",
                        "--output",
                        str(artifact),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("strict native library audit found", error.getvalue())
            self.assertFalse(artifact.exists())


if __name__ == "__main__":
    unittest.main()
