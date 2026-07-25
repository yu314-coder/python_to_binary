from __future__ import annotations

import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from py2bin.native import NativeCompileError, compile_native


_HOST_IS_DARWIN_ARM64 = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)

_MAGIC = {
    "windows": b"MZ",
    "linux": b"\x7fELF",
    "darwin": b"\xcf\xfa\xed\xfe",
}

# POSIX targets get a real anonymous-mmap arena. Windows lists/strings are a
# documented gap (they would need VirtualAlloc wired into the PE import table),
# so heap programs are rejected for the two windows targets, not mis-emitted.
_POSIX_TARGETS = (
    "linux-x86_64",
    "linux-arm64",
    "darwin-x86_64",
    "darwin-arm64",
)
_WINDOWS_TARGETS = ("windows-x86_64", "windows-arm64")


class NativeHeapTests(unittest.TestCase):
    """Runtime bump-arena heap, integer lists, and ASCII strings, lowered to
    real machine code and verified against CPython on darwin-arm64."""

    def _run(self, source: str, expected_exit: int, expected_stdout: bytes = b"") -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")

            # Every POSIX target must at least build a structurally valid image.
            for target in _POSIX_TARGETS:
                artifact = root / f"program-{target}.bin"
                compile_native(entry, artifact, target, clean=True)
                magic = _MAGIC[target.split("-")[0]]
                self.assertEqual(
                    artifact.read_bytes()[: len(magic)],
                    magic,
                    f"{target} heap image has a broken header",
                )

            # Windows heap support is an explicit, documented gap: reject, do
            # not emit a PE that would run incorrectly.
            for target in _WINDOWS_TARGETS:
                with self.assertRaises(NativeCompileError) as caught:
                    compile_native(entry, root / f"program-{target}.exe", target, clean=True)
                self.assertIn("windows", str(caught.exception).lower())

            if _HOST_IS_DARWIN_ARM64:
                native = subprocess.run(
                    [str(root / "program-darwin-arm64.bin")], capture_output=True
                )
                self.assertEqual(native.returncode, expected_exit)
                self.assertEqual(native.stdout, expected_stdout)
                # Cross-check the expectation against CPython itself.
                reference = subprocess.run(
                    [sys.executable, str(entry)], capture_output=True
                )
                self.assertEqual(native.returncode, reference.returncode)
                self.assertEqual(native.stdout, reference.stdout)

    # --- integer lists ------------------------------------------------------

    def test_list_index_load_store_and_len(self):
        # The canonical slice example: xs[1] = xs[0] + xs[2] -> 40.
        self._run(
            "xs = [10, 20, 30]\n"
            "xs[1] = xs[0] + xs[2]\n"
            "raise SystemExit(xs[1])\n",
            40,
        )

    def test_list_runtime_index_sum(self):
        self._run(
            "xs = [5, 6, 7, 8]\n"
            "total = 0\n"
            "for i in range(4):\n"
            "    total = total + xs[i]\n"
            "raise SystemExit(total)\n",
            26,
        )

    def test_list_runtime_index_store(self):
        self._run(
            "xs = [0, 0, 0, 0, 0]\n"
            "for i in range(5):\n"
            "    xs[i] = i * i\n"
            "raise SystemExit(xs[3] + xs[4])\n",
            25,
        )

    def test_list_len_is_runtime_header(self):
        self._run("xs = [1, 2, 3, 4, 5, 6, 7]\nraise SystemExit(len(xs))\n", 7)

    # --- runtime strings ----------------------------------------------------

    def test_runtime_string_concat_len(self):
        # s built in a loop is a genuine runtime string (not constant folded).
        self._run(
            "s = \"\"\n"
            "for i in range(3):\n"
            "    s = s + \"ab\"\n"
            "raise SystemExit(len(s))\n",
            6,
        )

    def test_runtime_string_concat_print_and_len(self):
        self._run(
            "a = \"\"\n"
            "for i in range(2):\n"
            "    a = a + \"xy\"\n"
            "b = \"Z!\"\n"
            "print(a + b)\n"
            "raise SystemExit(len(a + b))\n",
            6,
            b"xyxyZ!\n",
        )

    def test_runtime_string_longer_payload(self):
        self._run(
            "w = \"hello world \"\n"
            "s = \"\"\n"
            "for i in range(5):\n"
            "    s = s + w\n"
            "print(s)\n"
            "raise SystemExit(len(s) & 255)\n",
            60,
            b"hello world hello world hello world hello world hello world \n",
        )

    def test_runtime_string_alias_and_empty(self):
        self._run(
            "a = \"\"\n"
            "for i in range(1):\n"
            "    a = a + \"\"\n"
            "b = \"tail\"\n"
            "print(a + b)\n"
            "raise SystemExit(len(a + b))\n",
            4,
            b"tail\n",
        )

    # --- honest rejections --------------------------------------------------

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    def test_non_ascii_runtime_string_is_rejected(self):
        # Byte-length would disagree with CPython's code-point len(); reject
        # rather than emit a binary whose len() is wrong.
        self._reject(
            "s = \"\"\n"
            "for i in range(3):\n"
            "    s = s + \"é\"\n"
            "raise SystemExit(len(s))\n",
            "ASCII",
        )

    def test_constant_out_of_range_index_is_rejected(self):
        self._reject(
            "xs = [1, 2, 3]\nraise SystemExit(xs[5])\n",
            "out of range",
        )

    def test_negative_constant_index_counts_from_the_end(self):
        # Python semantics: xs[-1] is the last element.
        self._run("xs = [10, 20, 30]\nraise SystemExit(xs[-1])\n", 30)

    def test_negative_constant_index_beyond_length_is_rejected(self):
        self._reject(
            "xs = [1, 2, 3]\nraise SystemExit(xs[-4])\n",
            "out of range",
        )

    # --- runtime bounds checking -------------------------------------------
    #
    # A runtime index cannot be proved in range at build time. These programs
    # must behave exactly like CPython: negative indices count from the end,
    # and an out-of-range index reports IndexError and exits 1 instead of
    # reading or writing outside the list.

    def test_runtime_negative_index_counts_from_the_end(self):
        self._run(
            "xs = [10, 20, 30]\ni = len(xs) - 1\nraise SystemExit(xs[i])\n",
            30,
        )

    def test_runtime_out_of_range_read_reports_index_error(self):
        self._run(
            "xs = [10, 20, 30]\ni = len(xs) + 2\nraise SystemExit(xs[i])\n",
            1,
        )

    def test_runtime_out_of_range_store_reports_index_error(self):
        self._run(
            "xs = [1, 2, 3]\ni = len(xs) + 1\nxs[i] = 5\nraise SystemExit(xs[0])\n",
            1,
        )

    def test_runtime_negative_index_within_range_counts_from_the_end(self):
        # i = -2 is a valid Python index for a 3-element list.
        self._run(
            "xs = [1, 2, 3]\ni = len(xs) - 5\nraise SystemExit(xs[i])\n",
            2,
        )

    def test_runtime_negative_out_of_range_reports_index_error(self):
        # i = -4 is out of range for a 3-element list.
        self._run(
            "xs = [1, 2, 3]\ni = len(xs) - 7\nraise SystemExit(xs[i])\n",
            1,
        )

    def test_index_error_diagnostic_goes_to_stderr(self):
        # The diagnostic must not corrupt the program's stdout.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(
                "xs = [1, 2, 3]\n"
                "print(\"kept\")\n"
                "i = len(xs) + 1\n"
                "raise SystemExit(xs[i])\n",
                encoding="utf-8",
            )
            artifact = root / "program.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            run = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(run.returncode, 1)
            self.assertEqual(run.stdout, b"kept\n")
            self.assertIn(b"IndexError", run.stderr)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(run.returncode, reference.returncode)
            self.assertEqual(run.stdout, reference.stdout)

    def test_runtime_index_in_conditional_expression_is_rejected(self):
        # Both arms of a conditional expression are lowered eagerly, so the
        # bounds check would run even when Python would not evaluate the arm.
        self._reject(
            "xs = [10, 20, 30]\n"
            "i = len(xs) + 2\n"
            "v = xs[i] if i < len(xs) else 7\n"
            "raise SystemExit(v)\n",
            "conditional expression",
        )

    def test_equivalent_if_statement_form_is_accepted(self):
        self._run(
            "xs = [10, 20, 30]\n"
            "i = len(xs) + 2\n"
            "v = 7\n"
            "if i < len(xs):\n"
            "    v = xs[i]\n"
            "raise SystemExit(v)\n",
            7,
        )

    def test_list_of_floats_is_rejected(self):
        self._reject(
            "xs = [1.5, 2.5]\nraise SystemExit(int(xs[0]))\n",
            "signed 64-bit integers",
        )

    def test_list_used_as_bare_integer_is_rejected(self):
        self._reject(
            "xs = [1, 2, 3]\nraise SystemExit(xs)\n",
            "needs indexing or len()",
        )

    def test_string_used_as_bare_integer_is_rejected(self):
        self._reject(
            "s = \"\"\n"
            "for i in range(2):\n"
            "    s = s + \"a\"\n"
            "raise SystemExit(s)\n",
            "needs len()",
        )

    def test_variable_cannot_switch_between_int_and_string(self):
        self._reject(
            "x = 0\n"
            "for i in range(2):\n"
            "    x = x + 1\n"
            "if x > 0:\n"
            "    x = \"now a string\"\n"
            "raise SystemExit(len(x))\n",
            "cannot change type",
        )


if __name__ == "__main__":
    unittest.main()
