from pathlib import Path
import tempfile
import unittest

from py2bin.cli import main
from py2bin.csource import compile_to_c, decode_c_container, encode_c_container, plan_c


PROGRAM = """
def square(value: int) -> int:
    return value * value

total = 0
for index in range(1, 6):
    total += square(index)
print(f"result={total}")
"""


class CSourceTests(unittest.TestCase):
    def test_functions_loops_and_fstrings(self):
        result = compile_to_c(PROGRAM)
        self.assertIn("long long square(long long value)", result)
        self.assertIn("for (index = 1;", result)
        self.assertIn('printf("result=%lld\\n", total);', result)

    def test_container_round_trip_and_checksum(self):
        c_source = compile_to_c(PROGRAM)
        binary = encode_c_container(c_source)
        self.assertEqual(decode_c_container(binary), c_source)
        with self.assertRaisesRegex(ValueError, "checksum"):
            decode_c_container(binary[:-1] + bytes([binary[-1] ^ 1]))

    def test_heavy_libraries_use_compatible_bundle_plan(self):
        result = plan_c("import bpy, manim, torch, webview\nfrom transformers import AutoModel\n")
        self.assertEqual(result.backend, "cpython-bundle")
        self.assertEqual(result.imports, ("bpy", "manim", "torch", "transformers", "webview"))

    def test_cli_writes_c_and_container(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(PROGRAM, encoding="utf-8")
            self.assertEqual(main(["emit-c", str(entry), "-o", str(root / "program.c")]), 0)
            self.assertEqual(main(["emit-c", str(entry), "-o", str(root / "program.py2cbin"), "--container"]), 0)
            self.assertTrue((root / "program.c").read_text().startswith("/* Generated"))
            self.assertEqual((root / "program.py2cbin").read_bytes()[:8], b"PY2CBIN\0")


if __name__ == "__main__":
    unittest.main()
