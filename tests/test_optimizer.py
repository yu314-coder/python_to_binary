from pathlib import Path
import tempfile
import unittest

from py2bin.native import compile_native
from py2bin.native.frontend import NativeCompileError, lower
from py2bin.native.ir import Exit, Module, Write
from py2bin.native.optimizer import optimize


class NativeOptimizerTests(unittest.TestCase):
    def test_merges_writes_and_removes_code_after_exit(self):
        module = Module(
            [
                Write(b"first"),
                Write(b" second"),
                Write(b""),
                Exit(7),
                Write(b"unreachable"),
                Exit(0),
            ]
        )
        optimized, report = optimize(module)
        self.assertEqual(optimized.operations, [Write(b"first second"), Exit(7)])
        self.assertEqual(report.before, 6)
        self.assertEqual(report.after, 2)
        self.assertEqual(report.merged_writes, 1)
        self.assertEqual(report.removed_operations, 4)

    def test_frontend_folds_constants_and_removes_dead_branch(self):
        source = """
answer = 6 * 7
enabled = answer == 42 and True
if enabled:
    print(f"answer={answer}")
else:
    print("dead")
"""
        module = lower(Path("optimized.py"), source)
        optimized, _report = optimize(module)
        self.assertEqual(optimized.operations, [Write(b"answer=42\n"), Exit(0)])

    def test_compiler_drops_unreachable_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "exit_early.py"
            source.write_text(
                "import sys\nprint('live')\nsys.exit(3)\nprint('dead')\n",
                encoding="utf-8",
            )
            result = compile_native(source, root / "exit_early", "darwin-arm64")
            self.assertEqual(result.operations, 2)

    def test_identity_comparison_rejects_non_singletons(self):
        # The fold found the real fault, so it stays the reported one even
        # though the runtime path was tried afterwards and also failed. The
        # message is reported once: it is not wrapped in a second location.
        with self.assertRaisesRegex(
            NativeCompileError,
            r"^identity\.py:1:4: identity comparison is limited to None, True, "
            r"or False$",
        ):
            lower(Path("identity.py"), 'if "value" is "value":\n    print("bad")\n')

    def test_runtime_condition_reports_the_lowering_failure(self):
        # A runtime condition never folds, so "not a compile-time constant"
        # would be true of every one of these and name nothing.
        # A float, a list and a string in a condition used to be here too.
        # They lower now - each is true when it is non-zero or non-empty, as
        # Python says - so what is left is the conditions that still cannot be
        # lowered at all.
        cases = [
            (
                "x = 0\nfor i in range(0, 3):\n    x += 1\n",
                "x is None",
                r"native code has no runtime 'is'",
            ),
            (
                "n = 0\nfor i in range(0, 3):\n    n += 1\nif n > 5:\n    a = [1]\n",
                "a",
                r"'a' may be unbound here",
            ),
        ]
        for setup, test, expected in cases:
            with self.subTest(test=test):
                source = f'{setup}if {test}:\n    print("y")\n'
                with self.assertRaisesRegex(NativeCompileError, expected):
                    lower(Path("cond.py"), source)

    def test_runtime_condition_failure_survives_a_boolean_operator(self):
        source = (
            "x = 0\nfor i in range(0, 3):\n    x += 1\n"
            'if x is None and True:\n    print("y")\n'
        )
        with self.assertRaisesRegex(
            NativeCompileError, r"native code has no runtime 'is'"
        ):
            lower(Path("cond.py"), source)


if __name__ == "__main__":
    unittest.main()
