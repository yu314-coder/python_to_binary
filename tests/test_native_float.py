from __future__ import annotations

import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

from py2bin.native import NativeCompileError, compile_all, compile_native


_HOST_IS_DARWIN_ARM64 = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)

_MAGIC = {
    "windows": b"MZ",
    "linux": b"\x7fELF",
    "darwin": b"\xcf\xfa\xed\xfe",
}


def _build_all(source: str) -> dict[str, Path]:
    directory = tempfile.mkdtemp()
    root = Path(directory)
    entry = root / "program.py"
    entry.write_text(source, encoding="utf-8")
    results = compile_all(entry, root / "dist")
    return {result.target: result.artifact for result in results}


class NativeFloatTests(unittest.TestCase):
    """IEEE-754 double runtime lowered to real SSE2/NEON machine code."""

    def _run_arm64(self, source: str, expected_exit: int) -> None:
        artifacts = _build_all(source)
        # Every implemented target must produce a valid native image.
        for target, artifact in artifacts.items():
            magic = _MAGIC[target.split("-")[0]]
            self.assertEqual(
                artifact.read_bytes()[: len(magic)],
                magic,
                f"{target} float image has a broken header",
            )
        if _HOST_IS_DARWIN_ARM64:
            run = subprocess.run([str(artifacts["darwin-arm64"])])
            self.assertEqual(run.returncode, expected_exit)

    def test_runtime_float_accumulation_truncates_to_exit_code(self):
        self._run_arm64(
            "total = 0.0\n"
            "for i in range(1, 5):\n"
            "    total = total + 1.5\n"
            "raise SystemExit(int(total))\n",
            6,
        )

    def test_runtime_float_multiply_and_divide(self):
        self._run_arm64(
            "x = 0.0\n"
            "for i in range(1, 4):\n"
            "    x = x + 2.5\n"  # 7.5
            "y = x * 2.0\n"  # 15.0
            "z = y / 3.0\n"  # 5.0
            "raise SystemExit(int(z))\n",
            5,
        )

    def test_runtime_float_subtract_and_negate(self):
        self._run_arm64(
            "a = 0.0\n"
            "for i in range(1, 6):\n"
            "    a = a + 4.0\n"  # 20.0
            "b = a - 3.0\n"  # 17.0
            "c = -b\n"  # -17.0
            "raise SystemExit(int(-c))\n",
            17,
        )

    def test_mixed_integer_and_float_promotes(self):
        self._run_arm64(
            "n = 0\n"
            "for i in range(1, 5):\n"
            "    n = n + 3\n"  # int 12
            "f = n + 0.5\n"  # promoted to 12.5
            "raise SystemExit(int(f))\n",
            12,
        )

    def test_float_builtin_widens_runtime_integer(self):
        self._run_arm64(
            "x = 0.0\n"
            "for i in range(1, 8):\n"
            "    x = x + float(i)\n"  # 1+..+7 = 28.0
            "raise SystemExit(int(x))\n",
            28,
        )

    def test_float_comparison_drives_a_branch(self):
        self._run_arm64(
            "x = 0.0\n"
            "for i in range(1, 4):\n"
            "    x = x + 1.25\n"  # 3.75
            "flag = 0\n"
            "if x > 3.5:\n"
            "    flag = 7\n"
            "raise SystemExit(flag)\n",
            7,
        )

    def test_float_comparison_controls_a_while_loop(self):
        self._run_arm64(
            "x = 0.0\n"
            "steps = 0\n"
            "while x <= 5.0:\n"  # 0..5 -> 6 iterations
            "    x = x + 1.0\n"
            "    steps = steps + 1\n"
            "raise SystemExit(steps)\n",
            6,
        )

    def test_float_matches_cpython_reference(self):
        # The generated arm64 machine code must agree with CPython's own
        # double arithmetic for the same program.
        source = (
            "x = 0.0\n"
            "for i in range(1, 10):\n"
            "    x = x + 0.1\n"
            "raise SystemExit(int(x * 10.0))\n"
        )
        # Compute the reference value the honest way: run the same math in Python.
        reference_x = 0.0
        for _ in range(1, 10):
            reference_x = reference_x + 0.1
        expected = int(reference_x * 10.0) & 0xFF
        self._run_arm64(source, expected)

    def test_runtime_float_division_requires_constant_divisor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(
                "d = 0.0\n"
                "for i in range(1, 3):\n"
                "    d = d + 1.0\n"
                "x = 6.0 / d\n"  # runtime divisor: rejected, not silent inf/NaN
                "raise SystemExit(int(x))\n",
                encoding="utf-8",
            )
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn("constant divisor", str(caught.exception))

    def test_float_division_by_zero_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(
                "n = 0.0\n"
                "for i in range(1, 3):\n"
                "    n = n + 1.0\n"
                "x = n / 0.0\n"
                "raise SystemExit(int(x))\n",
                encoding="utf-8",
            )
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn("division by zero", str(caught.exception))

    def test_float_needs_explicit_int_in_integer_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(
                "x = 0.0\n"
                "for i in range(1, 3):\n"
                "    x = x + 1.0\n"
                "raise SystemExit(x)\n",  # float where an integer is required
                encoding="utf-8",
            )
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn("int(", str(caught.exception))

    def test_variable_cannot_switch_between_int_and_float(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(
                "x = 0\n"
                "for i in range(1, 3):\n"
                "    x = x + 1\n"
                "if x > 1:\n"
                "    x = 2.5\n"  # was int, now float: rejected
                "raise SystemExit(int(x))\n",
                encoding="utf-8",
            )
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn("int and float", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
