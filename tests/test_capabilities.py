from pathlib import Path
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from py2bin.capabilities import (
    assess_entry,
    common_libraries,
    library_capability,
)
from py2bin.cli import main


class CapabilityTests(unittest.TestCase):
    def test_common_catalog_makes_no_false_native_library_claim(self):
        catalog = {item.module: item for item in common_libraries()}
        for expected in (
            "numpy",
            "torch",
            "transformers",
            "tokenizers",
            "manim",
            "matplotlib",
            "bpy",
            "webview",
            "gradio",
            "streamlit",
            "OpenGL",
            "pygame",
            "psutil",
            "winpty",
            "onnxruntime",
            "pyarrow",
            "polars",
        ):
            self.assertIn(expected, catalog)
            self.assertEqual(catalog[expected].native_aot, "no")
            self.assertEqual(catalog[expected].compatible_bundle, "conditional")

    def test_unknown_import_is_conservatively_rejected_for_native_aot(self):
        result = library_capability("private_package.feature")
        self.assertEqual(result.native_aot, "no")
        self.assertEqual(result.compatible_bundle, "conditional")

    def test_entry_report_does_not_import_third_party_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / "common.py"
            entry.write_text(
                "import torch\n"
                "from transformers import AutoModel\n"
                "import manim, bpy, webview\n",
                encoding="utf-8",
            )
            result = assess_entry(entry)
            self.assertFalse(result.native_compile)
            self.assertEqual(
                result.imports,
                ("bpy", "manim", "torch", "transformers", "webview"),
            )
            self.assertIn("native subset", result.native_reason)

    def test_entry_report_accepts_real_native_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / "native.py"
            entry.write_text("answer = 6 * 7\nprint(answer)\n", encoding="utf-8")
            result = assess_entry(entry)
            self.assertTrue(result.native_compile)
            self.assertEqual(result.imports, ())

    def test_entry_report_accepts_runtime_integer_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / "native_loop.py"
            entry.write_text(
                "total = 0\n"
                "for value in range(5):\n"
                "    total += value\n"
                "raise SystemExit(total)\n",
                encoding="utf-8",
            )
            result = assess_entry(entry)
            self.assertTrue(result.native_compile)

    def test_capabilities_cli_emits_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / "app.py"
            entry.write_text("import numpy\n", encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                status = main(["capabilities", str(entry), "--json"])
            self.assertEqual(status, 0)
            report = json.loads(output.getvalue())
            self.assertFalse(report["native_compile"])
            self.assertEqual(report["libraries"][0]["module"], "numpy")

    def test_capabilities_strict_reports_unsupported_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / "app.py"
            entry.write_text("import numpy\n", encoding="utf-8")
            with redirect_stdout(StringIO()):
                status = main(["capabilities", str(entry), "--strict"])
            self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
