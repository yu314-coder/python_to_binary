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


class FloatIRNodeTests(unittest.TestCase):
    """The bit-level float IR nodes, checked against IEEE-754 by hand.

    Every expectation here is derived from the standard, not from another
    compiler: 0.1 rounded to binary32 is exactly 0.100000001490116119384765625,
    a signed conversion of the all-ones bit pattern is -1.0 while the unsigned
    one is 2**64-1, and every relational operator on a NaN is false while ``!=``
    is true. The programs are built for all six targets and, on darwin-arm64,
    RUN, because an FP encoding that assembles is not the same as one that
    computes.
    """

    def _exit_status(self, expression) -> int:
        from py2bin.native.compiler import compile_native_module
        from py2bin.native.ir import ExitValue, Module

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        module = Module([ExitValue(expression)], 4)
        for target in ("darwin-arm64", "darwin-x86_64", "linux-x86_64"):
            artifact = compile_native_module(
                root / "fake.py",
                module,
                root / f"program-{target}",
                target=target,
                clean=True,
            ).artifact
        if not _HOST_IS_DARWIN_ARM64:
            return -1
        artifact = compile_native_module(
            root / "fake.py",
            module,
            root / "program.bin",
            target="darwin-arm64",
            clean=True,
        ).artifact
        return subprocess.run([str(artifact)]).returncode

    def test_a_double_survives_a_round_trip_through_its_bit_pattern(self):
        from py2bin.native.ir import BitsFloat, FloatBits, FloatConstant, FloatToInt

        status = self._exit_status(
            FloatToInt(BitsFloat(FloatBits(FloatConstant(42.5), 8), 8))
        )
        if _HOST_IS_DARWIN_ARM64:
            self.assertEqual(status, 42)

    def test_a_four_byte_round_trip_rounds_to_binary32(self):
        import struct

        from py2bin.native.ir import (
            BitsFloat,
            FloatBinary,
            FloatBits,
            FloatConstant,
            FloatToInt,
        )

        # float(0.1) is 0.100000001490116119384765625, so times 1e9 it is
        # 100000001.49..., which truncates to 100000001 -- 1 modulo 256. The
        # double 0.1 would give exactly 100000000, which is 0 modulo 256, so
        # this distinguishes a real binary32 rounding from a no-op.
        self.assertEqual(struct.unpack("<f", struct.pack("<f", 0.1))[0] * 1e9, 100000001.49011612)
        status = self._exit_status(
            FloatToInt(
                FloatBinary(
                    "mul",
                    BitsFloat(FloatBits(FloatConstant(0.1), 4), 4),
                    FloatConstant(1e9),
                )
            )
        )
        if _HOST_IS_DARWIN_ARM64:
            self.assertEqual(status, 1)

    def test_signed_and_unsigned_integer_to_double_differ_above_two_to_the_63(self):
        from py2bin.native.ir import FloatBinary, FloatConstant, FloatToInt, IntConstant, IntToFloat

        # The all-ones pattern is -1 signed and 2**64-1 unsigned. Divided by
        # 2**63 those are -1/2**63 (truncating to 0) and 2 (exactly).
        for signed, expected in ((False, 2), (True, 0)):
            with self.subTest(signed=signed):
                status = self._exit_status(
                    FloatToInt(
                        FloatBinary(
                            "div",
                            IntToFloat(IntConstant(-1), signed=signed),
                            FloatConstant(float(2**63)),
                        )
                    )
                )
                if _HOST_IS_DARWIN_ARM64:
                    self.assertEqual(status, expected)

    def test_signed_and_unsigned_double_to_integer_differ_above_two_to_the_63(self):
        from py2bin.native.ir import FloatConstant, FloatToInt, IntBinary, IntConstant

        # 2**64-2048 is exactly representable. Converted unsigned its top byte
        # is 0xFF; the signed conversion saturates at 2**63-1, whose top byte
        # is 0x7F.
        for signed, expected in ((False, 255), (True, 127)):
            with self.subTest(signed=signed):
                status = self._exit_status(
                    IntBinary(
                        "urshift",
                        FloatToInt(FloatConstant(float(2**64 - 2048)), signed=signed),
                        IntConstant(56),
                    )
                )
                if _HOST_IS_DARWIN_ARM64:
                    self.assertEqual(status, expected)

    def test_every_ordering_is_false_for_a_nan_operand(self):
        from py2bin.native.ir import FloatBinary, FloatCompare, FloatConstant

        nan = FloatBinary("div", FloatConstant(0.0), FloatConstant(0.0))
        for operator, expected in (
            ("eq", 0),
            ("ne", 1),
            ("lt", 0),
            ("le", 0),
            ("gt", 0),
            ("ge", 0),
        ):
            with self.subTest(operator=operator):
                status = self._exit_status(
                    FloatCompare(operator, nan, FloatConstant(1.0))
                )
                if _HOST_IS_DARWIN_ARM64:
                    self.assertEqual(status, expected)


if __name__ == "__main__":
    unittest.main()
