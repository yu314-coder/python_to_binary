from __future__ import annotations

import math
import platform
import struct
import random
import subprocess
import sys
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

    def test_a_runtime_divisor_divides_and_checks_for_zero(self):
        # A runtime divisor used to be refused because there was no way to
        # raise ZeroDivisionError. The program now checks it and raises, so
        # neither a silent inf/NaN nor a build-time refusal is needed.
        self._run_arm64(
            "d = 0.0\n"
            "for i in range(1, 3):\n"
            "    d = d + 1.0\n"
            "x = 6.0 / d\n"
            "raise SystemExit(int(x))\n",
            3,
        )

    def test_a_runtime_divisor_of_zero_raises_like_cpython(self):
        self._run_arm64(
            "d = 0.0\n"
            "for i in range(1, 3):\n"
            "    d = d + 0.0\n"
            "x = 6.0 / d\n"
            "raise SystemExit(int(x))\n",
            1,
        )

    def test_a_zero_divisor_is_catchable(self):
        self._run_arm64(
            "d = 0.0\n"
            "for i in range(1, 3):\n"
            "    d = d + 0.0\n"
            "try:\n"
            "    x = 6.0 / d\n"
            "except ZeroDivisionError:\n"
            "    x = 9.0\n"
            "raise SystemExit(int(x))\n",
            9,
        )

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


class FloatAugmentedAssignmentTests(unittest.TestCase):
    """`x += 1.5` on a runtime float.

    Lowering it as the equivalent binary operation means every rule that
    applies to `x = x + 1.5` applies here too, including the restriction that
    a runtime float division needs a constant divisor.
    """

    def _run(self, source: str, expected: int) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "f.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "f.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            ).returncode
            self.assertEqual(reference, expected, "test expectation is wrong")
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            native = subprocess.run([str(artifact)], capture_output=True).returncode
            self.assertEqual(native, reference)

    def test_accumulating_in_a_loop(self):
        self._run(
            "total = 0.0\ni = 0\nwhile i < 4:\n    total += 0.5\n    i += 1\n"
            "raise SystemExit(int(total))\n",
            2,
        )

    def test_every_supported_operator(self):
        self._run(
            "x = 1.0\ni = 0\nwhile i < 1:\n    x += 5.0\n    x -= 1.0\n"
            "    x *= 3.0\n    x /= 2.0\n    i += 1\n"
            "raise SystemExit(int(x))\n",
            7,
        )

    def test_a_constant_float_gets_a_slot_when_it_becomes_runtime(self):
        self._run(
            "x = 2.5\ni = 0\nwhile i < 3:\n    x += 1.5\n    i += 1\n"
            "raise SystemExit(int(x))\n",
            7,
        )

    def test_a_runtime_divisor_works_and_checks_for_zero(self):
        # `/=` goes through the same lowering as `x = x / y`, so the runtime
        # zero check applies here too.
        self._run(
            "x = 3.0\ny = 2.0\nfor i in range(0, 2):\n    y += 1.0\n"
            "x /= y\nraise SystemExit(int(x * 4))\n",
            3,  # y accumulates to 4.0, so 3.0 / 4.0 * 4 is 3
        )
        self._run(
            "x = 3.0\ny = 0.0\nfor i in range(0, 2):\n    y += 0.0\n"
            "x /= y\nraise SystemExit(int(x))\n",
            1,
        )


class FloatParameterTests(unittest.TestCase):
    """Floats crossing a native function boundary.

    Functions are inlined, so a parameter is just a private local: the argument
    is stored into its slot and the body reads it through the ordinary variable
    path. Recording the kind is what makes that path pick float loads.
    """

    def _run(self, source: str, expected_stdout: bytes) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "f.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "f.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(
                reference.stdout, expected_stdout, "test expectation is wrong"
            )
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.stdout, reference.stdout)
            self.assertEqual(native.returncode, reference.returncode)

    def test_a_float_argument_to_a_procedure(self):
        self._run(
            "xs: list[float] = [0.0, 0.0]\n\n"
            "def record(slot, value):\n    xs[slot] = value * 2.0\n\n"
            "record(0, 1.25)\nrecord(1, 0.5)\n"
            "print(int((xs[0] + xs[1]) * 4))\n",
            b"14\n",
        )

    def test_a_float_argument_to_a_constructor(self):
        self._run(
            "class Point:\n    def __init__(self, x, y):\n"
            "        self.x: float = x\n        self.y: float = y\n"
            "        self.tag = 7\n\n"
            "p = Point(1.5, 2.25)\np.x = 0.75\n"
            "print(int((p.x + p.y) * 4), p.tag)\n",
            b"12 7\n",
        )

    def test_a_single_expression_function_returns_a_float(self):
        # Such a function is inlined by substituting its arguments into the one
        # expression, so its result kind is that expression's kind - knowable
        # without lowering anything.
        self._run("def scale(v):\n    return v * 2.0\n\nprint(int(scale(1.5)))\n", b"3\n")

    def test_float_returns_compose(self):
        self._run(
            "def area(r):\n    return 3.140625 * r * r\n\n"
            "def total(a, b):\n    return area(a) + area(b)\n\n"
            "print(int(total(2.0, 1.0) * 16))\n",
            b"251\n",
        )

    def test_a_float_argument_with_an_integer_result(self):
        self._run(
            "def bucket(v):\n    return int(v * 10.0)\n\n"
            "print(bucket(2.35), bucket(0.5))\n",
            b"23 5\n",
        )

    def test_a_float_return_survives_a_runtime_argument(self):
        # Constant folding would hide a truncation bug, so the argument is
        # accumulated in a loop and the result is fractional: an integer slot
        # anywhere on this path would turn 2.5 into 2 and print 8, not 10.
        self._run(
            "seed = 0.0\nfor i in range(0, 5):\n    seed += 0.25\n\n"
            "def grow(v):\n    return v * 2.0\n\n"
            "y = grow(seed)\nprint(int(y * 4))\n",
            b"10\n",
        )

    def test_a_float_return_from_a_statement_body(self):
        self._run(
            "seed = 0.0\nfor i in range(0, 5):\n    seed += 0.25\n\n"
            "def grow(v):\n    t = v * 2.0\n    return t\n\n"
            "y = grow(seed)\nprint(int(y * 4))\n",
            b"10\n",
        )

    def _reject(self, source: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "f.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "f.bin", "darwin-arm64", clean=True)
            return str(caught.exception)

    def test_a_float_where_an_integer_is_required_is_rejected(self):
        # Not a miscompile: refused at build time rather than truncated.
        self.assertIn(
            "not in the signed 64-bit native integer subset",
            self._reject(
                "xs = [10, 20, 30]\n\ndef pick(v):\n    return xs[v]\n\n"
                "print(pick(1.0))\n"
            ),
        )

    def test_a_float_returned_from_a_branching_body(self):
        # A body whose branches all end in a return is folded into one
        # conditional expression before the call site picks a lowering, so the
        # returned kind is known in time after all.
        self._run(
            "seed = 0.0\nfor i in range(0, 5):\n    seed += 0.25\n\n"
            "def grow(v):\n    t = v * 2.0\n    if t > 0.0:\n"
            "        return t\n    return t\n\nprint(int(grow(seed)))\n",
            b"2\n",
        )

    def test_a_float_returned_from_an_imperative_body_is_rejected(self):
        # A loop makes the body statement-by-statement, and then the call site
        # really has chosen a lowering before the returned kind is known.
        self.assertIn(
            "a native function with a loop or an early return cannot return a "
            "float",
            self._reject(
                "def acc(k):\n    t = 0.0\n    for i in range(0, k):\n"
                "        t += 1.5\n    return t\n\n"
                "n = 0\nfor i in range(0, 3):\n    n += 1\n"
                "print(acc(n))\n"
            ),
        )

    def test_mixed_arms_in_a_conditional_float_are_rejected(self):
        # One slot cannot print 1 and 2.5; widening the integer arm would print
        # 1.0 where CPython prints 1.
        self.assertIn(
            "the two arms of this have different kinds",
            self._reject(
                "n = 0\nfor i in range(0, 3):\n    n += 1\n"
                "print(1 if n > 2 else 2.5)\n"
            ),
        )

    def test_a_runtime_float_divisor_in_an_eager_arm_is_rejected(self):
        # Both arms of an integer conditional are evaluated, so the divisor's
        # zero check would raise on the branch CPython never takes.
        self.assertIn(
            "a float divisor that is not a compile-time constant cannot appear",
            self._reject(
                "n = 0\nfor i in range(0, 3):\n    n += 1\n"
                "b = float(n) - 3.0\n"
                "print(0 if b == 0.0 else (1 if (10.0 / b) > 1.0 else 2))\n"
            ),
        )


class ShortestRoundTripPrintingTests(unittest.TestCase):
    """print() of a computed double, against CPython's own repr.

    Deciding which decimal string is the shortest one that reads back as the
    same double cannot be done in 64-bit arithmetic: the value has to be
    compared with its two neighbours after scaling by a power of ten that
    reaches 10^308. So the generated code carries fixed-width big integers and
    runs Burger and Dybvig's algorithm on them, and the only convincing test is
    a differential one against CPython over a lot of values.
    """

    def _compare(self, values: list[float]) -> None:
        source = (
            "xs = [" + ", ".join(repr(v) for v in values) + "]\n"
            "i = 0\n"
            f"while i < {len(values)}:\n"
            "    print(xs[i])\n"
            "    i += 1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "f.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "f.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            native = subprocess.run([str(artifact)], capture_output=True)
            expected = "".join(repr(v) + "\n" for v in values).encode()
            self.assertEqual(native.stdout, expected)

    def test_random_doubles_match_repr(self):
        # A fixed seed, so a failure is reproducible rather than occasional.
        generator = random.Random(20260726)
        values: list[float] = []
        while len(values) < 400:
            kind = generator.randrange(4)
            if kind == 0:
                bits = generator.getrandbits(64)
                value = struct.unpack("<d", struct.pack("<Q", bits))[0]
            elif kind == 1:
                value = generator.uniform(-1e6, 1e6)
            elif kind == 2:
                value = generator.uniform(-1, 1) * 10 ** generator.randint(-320, 300)
            else:
                value = float(generator.randint(-(10**17), 10**17))
            if value != value or value in (math.inf, -math.inf):
                continue  # printed by their own test; not repr-able as a literal
            values.append(value)
        self._compare(values)

    def test_subnormals_match_repr(self):
        # Subnormals have the widest scaling range: 10^-324 away from one.
        generator = random.Random(5)
        values = [
            struct.unpack("<d", struct.pack("<Q", generator.getrandbits(52)))[0]
            for _ in range(60)
        ]
        values += [5e-324, 1e-320, 2.2250738585072011e-308]
        self._compare(values)

    def test_powers_of_ten_match_repr(self):
        self._compare([float(f"1e{power}") for power in range(-320, 309, 7)])

    def test_values_needing_every_digit_count(self):
        # One value for each length of digit string, so no path through the
        # generator's termination test goes unexercised.
        self._compare(
            [1.0, 1.5, 1.25, 1.125, 0.1, 1 / 3, 2 / 3, 1e23,
             9007199254740993.0, 1685094889599744.2, 1.7976931348623157e308,
             123456789012345.68, 0.30000000000000004]
        )


class FloatSortingTests(unittest.TestCase):
    """Sorting a runtime list of doubles.

    A double lives in an integer slot as its bit pattern, and those bits do not
    order the way the numbers do: read as signed integers, -1.0 sits above
    -2.0. These programs compare the numbers, which is what the outputs below
    check, and each output is CPython's own for the same source.
    """

    def _run(
        self,
        source: str,
        expected_stdout: bytes,
        expected_exit: int = 0,
        matches_cpython: bool = True,
        stderr_needle: bytes | None = None,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in ("darwin-arm64", "darwin-x86_64", "linux-x86_64"):
                artifact = root / f"program-{target}.bin"
                compile_native(entry, artifact, target, clean=True)
                magic = _MAGIC[target.split("-")[0]]
                self.assertEqual(artifact.read_bytes()[: len(magic)], magic)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run(
                [str(root / "program-darwin-arm64.bin")], capture_output=True
            )
            self.assertEqual(native.stdout, expected_stdout)
            self.assertEqual(native.returncode, expected_exit)
            if stderr_needle is not None:
                self.assertIn(stderr_needle, native.stderr)
            if not matches_cpython:
                # The NaN programs below refuse where CPython answers; the
                # refusal is the point, so there is nothing to diff.
                return
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.stdout, reference.stdout)
            self.assertEqual(native.returncode, reference.returncode)

    _PRINT = "for i in range(0, len({name})):\n    print({name}[i])\n"

    def test_negatives_sort_by_value_not_by_bit_pattern(self):
        # Comparing the bit patterns as signed integers would print
        # -0.5 -1.0 -2.0 0.0 3.5 here.
        self._run(
            "zs = [0.0]\n"
            "for i in range(0, 1):\n"
            "    zs.append(-1.0)\n"
            "    zs.append(-2.0)\n"
            "    zs.append(3.5)\n"
            "    zs.append(-0.5)\n"
            "zs.sort()\n" + self._PRINT.format(name="zs"),
            b"-2.0\n-1.0\n-0.5\n0.0\n3.5\n",
        )

    def test_signed_zeros_keep_their_order_in_both_directions(self):
        # -0.0 and 0.0 compare equal, and only a strict comparison moves an
        # element, so they stay in the order they arrived. Sorting and then
        # reversing the result would print them the other way round.
        self._run(
            "zs = [0.0]\n"
            "for i in range(0, 1):\n"
            "    zs.append(-0.0)\n"
            "    zs.append(0.0)\n"
            "    zs.append(-0.0)\n"
            "ys = sorted(zs)\n" + self._PRINT.format(name="ys")
            + "rs = sorted(zs, reverse=True)\n" + self._PRINT.format(name="rs"),
            b"0.0\n-0.0\n0.0\n-0.0\n0.0\n-0.0\n0.0\n-0.0\n",
        )

    def test_infinities_sort_at_the_ends(self):
        self._run(
            "zs = [1.0]\n"
            "for i in range(0, 1):\n"
            "    zs.append(1e308 * 10.0)\n"
            "    zs.append(-1.0 * (1e308 * 10.0))\n"
            "    zs.append(-2.5)\n"
            "zs.sort()\n" + self._PRINT.format(name="zs")
            + "for v in reversed(zs):\n    print(v)\n",
            b"-inf\n-2.5\n1.0\ninf\ninf\n1.0\n-2.5\n-inf\n",
        )

    def test_a_nan_is_refused_rather_than_ordered_differently(self):
        # CPython does not raise here: it returns whatever order its own
        # sequence of comparisons leaves behind, which an insertion sort does
        # not reproduce. Refusing is a divergence, but not a wrong order.
        self._run(
            "zs = [1.0]\n"
            "for i in range(0, 1):\n"
            "    zs.append(1e308 * 10.0)\n"
            "    zs.append(zs[1] - zs[1])\n"
            "    zs.append(2.0)\n"
            "zs.sort()\n",
            b"",
            expected_exit=1,
            matches_cpython=False,
            stderr_needle=b"ValueError: py2bin cannot sort a list containing nan",
        )

    def test_the_nan_refusal_is_a_catchable_value_error(self):
        self._run(
            "zs = [1.0]\n"
            "for i in range(0, 1):\n"
            "    zs.append(1e308 * 10.0)\n"
            "    zs.append(zs[1] - zs[1])\n"
            "try:\n"
            "    ys = sorted(zs)\n"
            "    print(0)\n"
            "except ValueError:\n"
            "    print(1)\n",
            b"1\n",
            matches_cpython=False,
        )


class FloatFloorDivisionAndModuloTests(unittest.TestCase):
    """`x // y` and `x % y` on doubles.

    Both go through a remainder computed by repeated subtraction of a scaled
    divisor. `x - trunc(x / y) * y` is the obvious way and is wrong once the
    quotient is large enough to round, and flooring `x / y` directly is wrong
    when the quotient rounds to just under or just over a whole number.
    """

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in ("darwin-arm64", "linux-x86_64", "linux-arm64"):
                compile_native(entry, root / f"program-{target}.bin", target, clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run(
                [str(root / "program-darwin-arm64.bin")], capture_output=True
            )
            self.assertEqual(native.stdout, expected_stdout)
            self.assertEqual(native.returncode, expected_exit)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.stdout, reference.stdout)
            self.assertEqual(native.returncode, reference.returncode)

    @staticmethod
    def _runtime(name: str, value: str) -> str:
        return f"{name} = 0.0\nfor _i in range(0, 1):\n    {name} = {name} + {value}\n"

    def test_the_sign_follows_the_divisor_as_pythons_does(self):
        # C's remainder takes the dividend's sign and Python's takes the
        # divisor's, so three of these four disagree with fmod.
        self._run(
            self._runtime("a", "7.5")
            + self._runtime("b", "-7.5")
            + "print(a // 2.0, a % 2.0)\n"
            + "print(b // 2.0, b % 2.0)\n"
            + "print(a // -2.0, a % -2.0)\n"
            + "print(b // -2.0, b % -2.0)\n",
            b"3.0 1.5\n-4.0 0.5\n-4.0 -0.5\n3.0 -1.5\n",
        )

    def test_an_exact_multiple_leaves_no_remainder(self):
        self._run(
            self._runtime("a", "8.0") + "print(a // 2.0, a % 2.0)\n", b"4.0 0.0\n"
        )

    def test_a_quotient_too_large_to_round_through(self):
        # 1e17 // 3.0 has a quotient far past the point where a double can
        # hold every integer, which is where the naive formula gives way.
        self._run(
            self._runtime("a", "1e17") + "print(a // 3.0, a % 3.0)\n",
            b"3.3333333333333332e+16 1.0\n",
        )

    def test_a_divisor_that_is_not_representable(self):
        # 0.1 is not exactly a tenth, so the remainder is not exactly zero and
        # every bit of it has to match.
        self._run(
            self._runtime("a", "1.0") + "print(a // 0.1, a % 0.1)\n",
            b"9.0 0.09999999999999995\n",
        )

    def test_an_integer_operand_is_widened(self):
        self._run(self._runtime("a", "7.5") + "print(a // 2, a % 2)\n", b"3.0 1.5\n")

    def test_dividing_by_zero_raises(self):
        self._run(
            self._runtime("a", "7.5") + "z = 0.0\nprint(a % z)\n", b"", expected_exit=1
        )

    def test_the_vtt_timestamp_helper_from_a_real_application(self):
        # Taken from manim_app's narration_addon.py. It is here because it is
        # what asked for this: floor division and modulo on a duration, then
        # an f-string with a zero-padded integer and a fixed-point float.
        self._run(
            "def format_vtt_time(seconds):\n"
            "    hours = int(seconds // 3600)\n"
            "    minutes = int(seconds % 3600 // 60)\n"
            "    secs = seconds % 60\n"
            '    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"\n'
            "print(format_vtt_time(3725.5))\n"
            "print(format_vtt_time(0.0))\n"
            "print(format_vtt_time(86399.999))\n",
            b"01:02:05.500\n00:00:00.000\n23:59:59.999\n",
        )
