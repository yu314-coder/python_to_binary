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
        with self.assertRaisesRegex(
            NativeCompileError, "identity comparison is limited"
        ):
            lower(Path("identity.py"), 'if "value" is "value":\n    print("bad")\n')


if __name__ == "__main__":
    unittest.main()
