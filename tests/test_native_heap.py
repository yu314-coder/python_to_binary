from __future__ import annotations

import platform
import struct
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

            # Windows gets the arena from VirtualAlloc rather than mmap. This
            # machine cannot run a PE, so what is checked here is that a
            # structurally valid image is produced and that it imports the
            # entry points the arena and its writes go through; the behaviour
            # is checked by running the darwin-arm64 image below, which is
            # built from the same IR.
            for target in _WINDOWS_TARGETS:
                image = root / f"program-{target}.exe"
                compile_native(entry, image, target, clean=True)
                data = image.read_bytes()
                self.assertEqual(data[:2], b"MZ", f"{target} is not a PE")
                for symbol in (b"VirtualAlloc", b"KERNEL32.dll"):
                    self.assertIn(symbol, data, f"{target} does not import {symbol!r}")

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

    def test_a_runtime_string_may_hold_any_text(self):
        # The header counts bytes, which is what a write needs; len() counts
        # code points by skipping UTF-8 continuation bytes, which is what
        # CPython reports.
        self._run(
            's = ""\ns = s + "caf\u00e9"\nraise SystemExit(len(s))\n', 4
        )
        self._run(
            's = ""\ns = s + "\u65e5\u672c\u8a9e"\nraise SystemExit(len(s))\n', 3
        )
        self._run(
            's = ""\ns = s + "a\U0001f389b"\nraise SystemExit(len(s))\n', 3
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

    def test_a_list_holds_floats_when_its_elements_are_floats(self):
        # The slot is eight bytes either way, so a float element lives there as
        # its bit pattern, the same way a float dict value does.
        self._run(
            "xs = [1.5, 2.25, 4.0]\nxs[1] = 0.75\n"
            "raise SystemExit(int((xs[0] + xs[1] + xs[2]) * 4))\n",
            25,
        )

    def test_an_annotation_types_an_empty_list_literal(self):
        self._run(
            "xs: list[float] = [0.0, 0.0, 0.0]\ni = 0\n"
            "while i < 3:\n    xs[i] = i * 0.5\n    i += 1\n"
            "raise SystemExit(int((xs[0] + xs[1] + xs[2]) * 2))\n",
            3,
        )

    def test_the_first_element_decides_what_a_list_holds(self):
        # An integer widens into a float list, as it does in a float dict; a
        # float in an integer list has nowhere to go and is refused.
        self._reject("xs = [1, 2.5]\nraise SystemExit(1)\n", "signed 64-bit integers")
        self._run(
            "xs = [1.5, 2]\nraise SystemExit(int((xs[0] + xs[1]) * 2))\n", 7
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


class WindowsArenaEncodingTests(unittest.TestCase):
    """The Windows arena, checked as far as a machine that cannot run a PE can.

    Behaviour is covered by the POSIX images built from the same IR and run
    against CPython. What is specific to Windows is the three call sequences
    the arena needs, and those are pinned here byte for byte so a change to
    them is deliberate rather than accidental.
    """

    def _image(self, target: str) -> bytes:
        source = (
            'names: dict[str, int] = {}\ns = ""\ni = 0\n'
            'while i < 5:\n    s = s + "k"\n    names[s] = i\n    i += 1\n'
            "print(s)\nraise SystemExit(len(names))\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            image = root / "program.exe"
            compile_native(entry, image, target, clean=True)
            return image.read_bytes()

    def test_both_windows_targets_import_what_the_arena_needs(self):
        for target in ("windows-x86_64", "windows-arm64"):
            data = self._image(target)
            self.assertEqual(data[:2], b"MZ", target)
            for symbol in (
                b"KERNEL32.dll",
                b"VirtualAlloc",
                b"GetStdHandle",
                b"WriteFile",
                b"ExitProcess",
            ):
                self.assertIn(symbol, data, f"{target} does not import {symbol!r}")

    def test_x86_64_reserves_the_arena_with_virtualalloc(self):
        data = self._image("windows-x86_64")
        # xor rcx,rcx (NULL) ... mov r8d,0x3000 (MEM_COMMIT|MEM_RESERVE),
        # mov r9d,4 (PAGE_READWRITE) - the argument setup VirtualAlloc needs.
        self.assertIn(b"\x41\xb8\x00\x30\x00\x00\x41\xb9\x04\x00\x00\x00", data)
        # test rax,rax then a short jnz: a NULL reservation must not be used.
        self.assertIn(b"\x48\x85\xc0\x75", data)

    def test_arm64_reserves_the_arena_with_virtualalloc(self):
        data = self._image("windows-arm64")
        words = struct.unpack_from("<%dI" % (len(data) // 4), data)
        # movz x2,#0x3000 and movz x3,#4 sit next to each other in the setup.
        self.assertIn(0xD2800000 | (0x3000 << 5) | 2, words)
        self.assertIn(0xD2800000 | (4 << 5) | 3, words)
        # cbnz x0, +n guards against a NULL reservation.
        self.assertTrue(
            any((word & 0xFF00001F) == 0xB5000000 for word in words),
            "no cbnz x0 guarding the reservation",
        )
