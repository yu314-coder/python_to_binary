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

    # --- runtime string methods ---------------------------------------------
    #
    # Every source below builds its strings in a loop, because the front end
    # folds aggressively and a folded receiver would test CPython's own
    # evaluator instead of the emitted code. `_run` diffs stdout and the exit
    # status against CPython running the same file.

    def test_find_and_index_report_character_offsets(self):
        # A byte-offset implementation answers 3 and 7 here.
        self._run(
            's = ""\n'
            "for i in range(0, 1):\n"
            '    s = s + "héllo wörld"\n'
            'print(s.find("llo"), s.find("wörld"), s.find("zz"), '
            's.index("ö"))\n',
            0,
            "2 6 -1 7\n".encode("utf-8"),
        )

    def test_count_skips_overlaps_and_counts_empty_by_character(self):
        self._run(
            's = ""\n'
            "for i in range(0, 1):\n"
            '    s = s + "aaa"\n'
            'e = ""\n'
            "for i in range(0, 1):\n"
            '    e = e + "é"\n'
            'print(s.count("aa"), s.count("a"), s.count(""), e.count(""), '
            's.count("b"))\n',
            0,
            b"1 3 4 2 0\n",
        )

    def test_replace_grows_shrinks_and_handles_an_empty_needle(self):
        self._run(
            's = ""\n'
            "for i in range(0, 1):\n"
            '    s = s + "aaa"\n'
            'u = ""\n'
            "for i in range(0, 1):\n"
            '    u = u + "é"\n'
            'print(s.replace("a", "bb"))\n'
            'print("[" + s.replace("a", "") + "]")\n'
            'print(s.replace("", "-"))\n'
            'print(u.replace("", "-"))\n'
            'print(s.replace("aa", "X"))\n',
            0,
            "bbbbbb\n[]\n-a-a-a-\n-é-\nXa\n".encode("utf-8"),
        )

    def test_strip_removes_unicode_whitespace(self):
        # The ASCII five would leave U+00A0 and U+3000 in place and diverge on
        # all three calls.
        self._run(
            's = ""\n'
            "for i in range(0, 1):\n"
            '    s = s + "  x　 "\n'
            'print("[" + s.strip() + "]", "[" + s.lstrip() + "]", '
            '"[" + s.rstrip() + "]")\n',
            0,
            "[x] [x　 ] [  x]\n".encode("utf-8"),
        )

    def test_strip_of_only_whitespace_is_empty(self):
        self._run(
            's = ""\n'
            "for i in range(0, 1):\n"
            '    s = s + "    "\n'
            'print("[" + s.strip() + "]", len(s.strip()), len(s.lstrip()), '
            "len(s.rstrip()))\n",
            0,
            b"[] 0 0 0\n",
        )

    def test_padding_counts_characters_not_bytes(self):
        self._run(
            's = ""\n'
            "for i in range(0, 1):\n"
            '    s = s + "é"\n'
            't = ""\n'
            "for i in range(0, 1):\n"
            '    t = t + "ab"\n'
            'print("[" + s.rjust(3) + "]", "[" + s.zfill(3) + "]", '
            '"[" + t.center(5) + "]", "[" + t.center(6) + "]", '
            '"[" + t.center(1) + "]", "[" + t.ljust(5) + "]")\n',
            0,
            "[  é] [00é] [  ab ] [  ab  ] [ab] [ab   ]\n".encode(
                "utf-8"
            ),
        )

    def test_zfill_keeps_a_leading_sign_in_front(self):
        self._run(
            'n = ""\n'
            "for i in range(0, 1):\n"
            '    n = n + "-5"\n'
            'p = ""\n'
            "for i in range(0, 1):\n"
            '    p = p + "+7"\n'
            'e = ""\n'
            "for i in range(0, 1):\n"
            '    e = e + ""\n'
            'print("[" + n.zfill(4) + "]", "[" + p.zfill(4) + "]", '
            '"[" + e.zfill(3) + "]", "[" + n.zfill(1) + "]")\n',
            0,
            b"[-005] [+007] [000] [-5]\n",
        )

    def test_string_predicates_print_as_bools(self):
        # Without renders_as_bool knowing about these, they would print 1 and 0.
        self._run(
            's = ""\n'
            "for i in range(0, 1):\n"
            '    s = s + "abc"\n'
            'print(s.startswith("ab"), s.endswith("bc"), s.startswith(""), '
            "s.isdigit(), s.isalpha())\n"
            'flag = s.isalpha()\n'
            "print(flag)\n",
            0,
            b"True True True False True\nTrue\n",
        )

    def test_empty_haystack_answers_like_cpython(self):
        self._run(
            's = ""\n'
            "for i in range(0, 1):\n"
            '    s = s + ""\n'
            'print("[" + s.strip() + "]", s.find("a"), s.count("a"), '
            's.startswith(""), s.find(""), s.count(""))\n',
            0,
            b"[] -1 0 True 0 1\n",
        )

    def test_ascii_case_methods_match_cpython(self):
        self._run(
            's = ""\n'
            "for i in range(0, 1):\n"
            "    s = s + \"MiXed 42! they're\"\n"
            "print(s.upper())\n"
            "print(s.lower())\n"
            "print(s.capitalize())\n"
            "print(s.title())\n",
            0,
            b"MIXED 42! THEY'RE\nmixed 42! they're\n"
            b"Mixed 42! they're\nMixed 42! They'Re\n",
        )

    def test_methods_apply_to_any_string_expression(self):
        self._run(
            'a = ""\n'
            "for i in range(0, 1):\n"
            '    a = a + "  héllo  "\n'
            "def tag(t):\n"
            '    return t + "!"\n'
            'print("[" + a.strip().replace("é", "e").upper() + "]", '
            '"[" + tag(a.strip())[1:].rjust(6) + "]")\n',
            0,
            "[HELLO] [ éllo!]\n".encode("utf-8"),
        )

    def test_index_raises_a_catchable_value_error(self):
        self._run(
            's = ""\n'
            "for i in range(0, 1):\n"
            '    s = s + "abc"\n'
            "try:\n"
            '    print(s.index("zz"))\n'
            "except ValueError:\n"
            '    print("caught")\n',
            0,
            b"caught\n",
        )

    def test_a_literal_that_reads_the_name_it_replaces(self):
        # The new block is built into a slot of its own and moved onto the name
        # afterwards. Building it in place would put the new, empty block's
        # address in the name before the right-hand side read the old one.
        self._run(
            "xs = [1]\n"
            "w = 0\n"
            "while w < 3:\n"
            "    xs = [xs[0] + 1]\n"
            "    w += 1\n"
            "print(xs[0])\n",
            0,
            b"4\n",
        )
        self._run(
            "d = {}\n"
            "d[0] = 1\n"
            "w = 0\n"
            "while w < 3:\n"
            "    d = {0: d[0] + 1}\n"
            "    w += 1\n"
            "print(d[0])\n",
            0,
            b"4\n",
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
        # One list holds one kind. A float in an integer list has nowhere to
        # go, and an integer in a float list used to be widened - which prints
        # 2.0 where CPython prints 2, because a list element keeps whatever
        # object was put in it rather than converting to the list's type.
        self._reject("xs = [1, 2.5]\nraise SystemExit(1)\n", "signed 64-bit integers")
        self._reject("xs = [1.5, 2]\nprint(xs[1])\n", "write 1.0 rather than 1")
        self._run(
            "xs = [1.5, 2.0]\nraise SystemExit(int((xs[0] + xs[1]) * 2))\n", 7
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


class ListSortingTests(unittest.TestCase):
    """`sorted()`, `list.sort()` and `reversed()` over runtime lists.

    Every expectation is CPython's own output for the same source, diffed
    against the darwin-arm64 binary. The lists are built by appending inside a
    loop so that the front end cannot fold them away and answer from its own
    evaluator instead of from the machine code.
    """

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
                artifact = root / f"program-{target}.bin"
                compile_native(entry, artifact, target, clean=True)
                magic = _MAGIC[target.split("-")[0]]
                self.assertEqual(
                    artifact.read_bytes()[: len(magic)],
                    magic,
                    f"{target} sorting image has a broken header",
                )
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
            if expected_exit == 0:
                self.assertEqual(native.stdout, reference.stdout)
                self.assertEqual(native.returncode, reference.returncode)

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    _BUILD = (
        "xs = []\n"
        "for i in range(0, 1):\n"
        "    xs.append(3)\n"
        "    xs.append(1)\n"
        "    xs.append(4)\n"
        "    xs.append(1)\n"
        "    xs.append(5)\n"
    )
    _PRINT = "for i in range(0, len({name})):\n    print({name}[i])\n"

    def test_sorted_builds_a_new_list_and_leaves_the_source_alone(self):
        self._run(
            self._BUILD
            + "ys = sorted(xs)\n"
            + self._PRINT.format(name="ys")
            + self._PRINT.format(name="xs"),
            b"1\n1\n3\n4\n5\n3\n1\n4\n1\n5\n",
        )

    def test_a_second_name_for_the_sorted_list_is_rejected(self):
        # sort() reorders in place, which an alias would see - but the alias
        # itself cannot be made sound, because an append moves the block and
        # only one name follows it. A slice copies instead.
        self._reject(
            self._BUILD + "ys = xs\nxs.sort()\n" + self._PRINT.format(name="ys"),
            "not a reference to it",
        )

    def test_a_slice_copy_is_an_independent_list(self):
        self._run(
            self._BUILD + "ys = xs[:]\nxs.sort()\n" + self._PRINT.format(name="ys"),
            b"3\n1\n4\n1\n5\n",
        )

    def test_reverse_true_flips_the_comparison(self):
        self._run(
            self._BUILD
            + "ys = sorted(xs, reverse=True)\n"
            + self._PRINT.format(name="ys")
            + "xs.sort(reverse=True)\n"
            + self._PRINT.format(name="xs"),
            b"5\n4\n3\n1\n1\n5\n4\n3\n1\n1\n",
        )

    def test_sorting_is_signed_at_the_ends_of_the_range(self):
        # An unsigned comparison, or a subtract-and-test-the-sign one, puts
        # INT64_MIN above the positives instead of below them.
        self._run(
            "xs = []\n"
            "for i in range(0, 1):\n"
            "    xs.append(9223372036854775807)\n"
            "    xs.append(-9223372036854775807 - 1)\n"
            "    xs.append(0)\n"
            "    xs.append(-1)\n"
            "    xs.append(1)\n"
            "xs.sort()\n" + self._PRINT.format(name="xs"),
            b"-9223372036854775808\n-1\n0\n1\n9223372036854775807\n",
        )

    def test_an_empty_and_a_single_element_list_sort(self):
        self._run(
            "xs = []\nxs.sort()\nprint(len(xs))\n"
            "ys = sorted(xs)\nprint(len(ys))\n"
            "zs = []\n"
            "for i in range(0, 1):\n    zs.append(7)\n"
            "zs.sort()\nprint(zs[0])\n",
            b"0\n0\n7\n",
        )

    def test_reversed_walks_the_same_block_backwards(self):
        self._run(
            self._BUILD + "for v in reversed(xs):\n    print(v)\n",
            b"5\n1\n4\n1\n3\n",
        )

    def test_reversed_re_reads_the_list_the_way_cpythons_iterator_does(self):
        # Appending moves the block to a bigger one. CPython's reverse iterator
        # holds the list, not a snapshot of it, so the write to xs[0] is seen.
        self._run(
            "xs = []\n"
            "for i in range(0, 1):\n"
            "    xs.append(1)\n"
            "    xs.append(2)\n"
            "    xs.append(3)\n"
            "    xs.append(4)\n"
            "for v in reversed(xs):\n"
            "    xs.append(9)\n"
            "    xs[0] = 77\n"
            "    print(v)\n",
            b"4\n3\n2\n77\n",
        )

    def test_a_sorted_copy_can_still_grow(self):
        # The copy is allocated at exactly its own length, so the first append
        # has to move it rather than write past the block.
        self._run(
            self._BUILD
            + "ys = sorted(xs)\n"
            + "for i in range(0, 2):\n    ys.append(9)\n"
            + self._PRINT.format(name="ys"),
            b"1\n1\n3\n4\n5\n9\n9\n",
        )

    def test_sorted_is_iterable_and_composable(self):
        self._run(
            self._BUILD
            + "for v in sorted(sorted(xs), reverse=True):\n    print(v)\n"
            + "for v in reversed(sorted(xs[1:4])):\n    print(v)\n",
            b"5\n4\n3\n1\n1\n4\n1\n1\n",
        )

    def test_a_key_or_a_comparison_callable_is_rejected(self):
        self._reject(self._BUILD + "ys = sorted(xs, key=abs)\n", "does not support key=")
        self._reject(self._BUILD + "ys = sorted(xs, cmp=abs)\n", "does not support cmp=")
        self._reject(self._BUILD + "xs.sort(key=abs)\n", "does not support key=")
        self._reject(self._BUILD + "xs.sort(abs)\n", "takes no positional argument")

    def test_a_runtime_reverse_direction_is_rejected(self):
        self._reject(
            self._BUILD + "n = 0\nfor i in range(0, 2):\n    n += 1\n"
            "ys = sorted(xs, reverse=n)\n",
            "reverse= to be the constant True or False",
        )

    def test_sorting_something_that_is_not_a_runtime_list_is_rejected(self):
        self._reject('ys = sorted("abc")\n', "takes a runtime list of integers")
        self._reject("d = {1: 2}\nys = sorted(d)\n", "takes a runtime list of integers")
        self._reject("for v in reversed(range(0, 3)):\n    print(v)\n", "reversed()")

    def test_reversed_outside_a_for_header_is_rejected(self):
        self._reject(self._BUILD + "ys = reversed(xs)\n", "produces a sequence")

    def test_sorting_a_list_of_bools_is_rejected(self):
        # sorted([True, False]) is [False, True] and prints as bools, but a
        # sorted copy does not inherit which container held bools, so the
        # result would print as 0 and 1.
        for source in (
            "bs = [True, False]\nys = sorted(bs)\n",
            "bs = [True, False]\nbs.sort()\n",
            "bs = [True, False]\nfor b in reversed(bs):\n    print(b)\n",
        ):
            self._reject(source, "holds bools")

    def test_rebinding_the_list_a_reversed_loop_walks_is_rejected(self):
        self._reject(
            "xs = []\n"
            "for i in range(0, 1):\n    xs.append(1)\n"
            "for v in reversed(xs):\n    xs = [9]\n",
            "is the list this loop is walking",
        )

    def test_a_name_a_reversed_loop_may_never_bind_is_rejected(self):
        self._reject(
            "xs = []\nfor v in reversed(xs):\n    print(v)\nprint(v)\n",
            "may be unbound",
        )

    # --- runtime string methods: what is refused, and how --------------------

    def test_non_ascii_case_methods_stop_the_program(self):
        # CPython prints CAFÉ. Getting that right needs the Unicode case
        # tables, which are not in the image, so the binary must refuse rather
        # than print a byte-flipped CAFé. The refusal is a write and an exit,
        # not a raise, so an `except Exception:` cannot swallow it.
        source = (
            's = ""\n'
            "for i in range(0, 1):\n"
            '    s = s + "café"\n'
            "try:\n"
            "    print(s.upper())\n"
            "except Exception:\n"
            '    print("swallowed")\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            binary = root / "program.bin"
            compile_native(entry, binary, "darwin-arm64", clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run([str(binary)], capture_output=True)
            self.assertEqual(native.returncode, 1)
            self.assertEqual(native.stdout, b"")
            self.assertIn(b"str.upper() is limited to ASCII text", native.stderr)

    def test_a_constant_non_ascii_case_receiver_is_rejected_at_build_time(self):
        self._reject(
            'print("café".upper())\n', "is limited to ASCII text"
        )
        self._reject(
            'print("é".isalpha())\n', "is limited to ASCII text"
        )

    def test_an_uncaught_index_miss_reports_a_value_error(self):
        source = (
            's = ""\n'
            "for i in range(0, 1):\n"
            '    s = s + "abc"\n'
            'print(s.index("zz"))\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            binary = root / "program.bin"
            compile_native(entry, binary, "darwin-arm64", clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run([str(binary)], capture_output=True)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.returncode, reference.returncode)
            self.assertEqual(native.stdout, reference.stdout)
            self.assertIn(b"ValueError: substring not found", native.stderr)
            self.assertIn(b"ValueError: substring not found", reference.stderr)

    _RUNTIME_STRING = 's = ""\nfor i in range(0, 1):\n    s = s + "abc"\n'

    def test_a_stopping_string_method_in_an_eager_arm_is_rejected(self):
        # Both arms of a conditional expression are lowered, so the ASCII guard
        # and the index() raise would fire on a branch CPython never takes.
        flag = "f = len(s) > 0\n"
        for source in (
            self._RUNTIME_STRING + flag + "n = len(s.upper()) if f else 0\n",
            self._RUNTIME_STRING + flag + 'n = s.index("a") if f else 0\n',
            self._RUNTIME_STRING + flag + "b = f and s.isdigit()\n",
        ):
            self._reject(source, "conditional expression")

    def test_a_discarded_string_method_call_is_rejected(self):
        self._reject(
            self._RUNTIME_STRING + 's.replace("a", "b")\n',
            "returns a new string and changes nothing",
        )

    def test_unsupported_string_method_arguments_are_rejected(self):
        self._reject(
            self._RUNTIME_STRING + 'print(s.find("a", 1))\n',
            "native str.find() takes 1 argument",
        )
        self._reject(
            self._RUNTIME_STRING + 'print(s.center(5, "*"))\n',
            "native str.center() takes 1 argument",
        )
        self._reject(
            self._RUNTIME_STRING + "print(s.find(1))\n",
            "native str.find() takes a string",
        )
        self._reject(
            self._RUNTIME_STRING + 'print(s.zfill("3"))\n',
            "native str.zfill() takes an integer width",
        )

    def test_mixing_a_string_predicate_with_a_number_in_a_list_is_rejected(self):
        self._reject(
            self._RUNTIME_STRING + 'xs = [s.startswith("a"), 1]\n',
            "mixed container is refused",
        )


class ComprehensionTests(unittest.TestCase):
    """Multi-clause comprehensions and generator expressions in aggregates.

    Every expectation is CPython's own output for the same source, diffed
    against the darwin-arm64 binary. Sources are built from names rather than
    literals so the front end cannot fold the comprehension away and answer
    from its own evaluator instead of from the machine code.
    """

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
                artifact = root / f"program-{target}.bin"
                compile_native(entry, artifact, target, clean=True)
                magic = _MAGIC[target.split("-")[0]]
                self.assertEqual(
                    artifact.read_bytes()[: len(magic)],
                    magic,
                    f"{target} comprehension image has a broken header",
                )
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

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    _RUNTIME_N = "n = 0\nfor i in range(0, 3):\n    n = n + 1\n"

    def test_generator_in_sum_builds_no_list(self):
        self._run(
            "xs = [1, 2, 3]\nprint(sum(v * 2 for v in xs))\n", b"12\n"
        )

    def test_generator_over_a_runtime_range(self):
        self._run(
            self._RUNTIME_N
            + "print(sum(v * v for v in range(0, n + 2)))\n"
            + "print(i)\n",
            b"30\n2\n",
        )

    def test_generator_aggregates_do_not_allocate_per_call(self):
        # A list comprehension here would ask the arena for a block a million
        # times and exhaust it; the generator asks for nothing at all.
        self._run(
            "xs = [1, 2, 3]\n"
            "t = 0\n"
            "k = 0\n"
            "while k < 200000:\n"
            "    t = t + sum(v for v in xs)\n"
            "    k = k + 1\n"
            "print(t)\n",
            b"1200000\n",
        )

    def test_min_and_max_over_an_empty_generator_raise(self):
        self._run(
            "xs = []\n"
            "try:\n"
            "    print(min(v for v in xs))\n"
            "except ValueError:\n"
            "    print('caught')\n"
            "ys = [7, 3, 9]\n"
            "print(max(v for v in ys))\n"
            "print(min(v for v in ys))\n",
            b"caught\n9\n3\n",
        )

    def test_any_and_all_including_empty_sources(self):
        self._run(
            "xs = []\n"
            "ys = [0, 0]\n"
            "zs = [0, 1]\n"
            "print(any(v for v in xs))\n"
            "print(all(v for v in xs))\n"
            "print(any(v for v in ys))\n"
            "print(all(v for v in zs))\n"
            "print(any(zs))\n"
            "print(all(ys))\n",
            b"False\nTrue\nFalse\nFalse\nTrue\nFalse\n",
        )

    def test_nested_clauses_produce_the_cross_product(self):
        self._run(
            "xs = [1, 2]\n"
            "ys = [3, 4, 5]\n"
            "r = [a * b for a in xs for b in ys]\n"
            "print(len(r))\n"
            "print(r[0])\n"
            "print(r[5])\n"
            "print(sum(r))\n",
            b"6\n3\n10\n36\n",
        )

    def test_conditions_on_more_than_one_clause(self):
        self._run(
            "xs = [1, 2, 3]\n"
            "ys = [4, 5]\n"
            "r = [a * b for a in xs if a != 2 for b in ys if b != 4]\n"
            "print(len(r))\n"
            "print(r[0])\n"
            "print(r[1])\n",
            b"2\n5\n15\n",
        )

    def test_two_ifs_on_one_clause_and_a_condition_that_rejects_everything(self):
        self._run(
            "xs = [1, 2, 3]\n"
            "a = [q for q in xs if q > 99]\n"
            "b = [q for q in xs if q > 1 if q < 3]\n"
            "print(len(a))\n"
            "print(len(b))\n"
            "print(b[0])\n",
            b"0\n1\n2\n",
        )

    def test_an_empty_source_in_either_position(self):
        self._run(
            "xs = []\n"
            "ys = [1, 2]\n"
            "outer = [a * b for a in xs for b in ys]\n"
            "inner = [a * b for a in ys for b in xs]\n"
            "print(len(outer))\n"
            "print(len(inner))\n"
            "print(sum(v for v in xs))\n",
            b"0\n0\n0\n",
        )

    def test_every_clause_target_keeps_its_own_scope(self):
        self._run(
            "a = 7\n"
            "b = 8\n"
            "xs = [1, 2]\n"
            "ys = [3, 4]\n"
            "r = [a + b for a in xs for b in ys]\n"
            "print(len(r))\n"
            "print(r[0])\n"
            "print(r[3])\n"
            "print(a)\n"
            "print(b)\n",
            b"4\n4\n6\n7\n8\n",
        )

    def test_two_clauses_may_bind_the_same_name(self):
        self._run(
            "xs = [1, 2]\n"
            "ys = [5, 6, 7]\n"
            "r = [a for a in xs for a in ys]\n"
            "print(len(r))\n"
            "print(r[0])\n"
            "print(r[5])\n",
            b"6\n5\n7\n",
        )

    def test_a_bool_element_still_prints_as_a_bool(self):
        self._run(
            self._RUNTIME_N
            + "xs = [n > 1, n < 1]\n"
            + "r = [v for v in xs]\n"
            + "print(r[0])\n"
            + "for v in xs:\n"
            + "    print(v)\n"
            + "print(xs[0:2][0])\n"
            + "print(max(xs))\n"
            + "print(max(v for v in xs))\n"
            + "zs = [1, 2, 3]\n"
            + "print([q > 1 for q in zs][2])\n"
            + "print(sum(q > 1 for q in zs))\n",
            b"True\nTrue\nFalse\nTrue\nTrue\nTrue\nTrue\n2\n",
        )

    def test_float_elements_survive_the_nested_rewrite(self):
        self._run(
            "xs = [1.5, 2.5]\n"
            "ys = [1, 2]\n"
            "r = [a * 2 for a in xs]\n"
            "f = [a * 0.5 for a in ys for b in ys]\n"
            "print(r[0])\n"
            "print(r[1])\n"
            "print(len(f))\n"
            "print(f[3])\n",
            b"3.0\n5.0\n4\n1.0\n",
        )

    def test_a_product_that_would_not_fit_the_arena_reports_memory_error(self):
        # The spans multiply to more than an i64 holds. A wrapping product
        # would come out negative, be clamped to zero, and then be written far
        # past a header-only block without the arena guard ever seeing it.
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("needs the host to run the darwin-arm64 image")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "huge.py"
            entry.write_text(
                self._RUNTIME_N
                + "m = n * 1500000000\n"
                + "r = [1 for a in range(0, m) for b in range(0, m)]\n"
                + "print(len(r))\n",
                encoding="utf-8",
            )
            artifact = root / "huge.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.returncode, 1)
            self.assertEqual(native.stdout, b"")
            self.assertIn(b"MemoryError", native.stderr)

    def test_a_generator_expression_anywhere_else_is_rejected(self):
        for source in (
            "xs = [1, 2]\ng = (v for v in xs)\nprint(1)\n",
            "xs = [1, 2]\nprint(v for v in xs)\n",
            "xs = [1, 2]\ndef f(y):\n    return y\nprint(f(v for v in xs))\n",
        ):
            self._reject(source, "generator expression is a lazy object")

    def test_a_later_source_may_not_depend_on_an_earlier_target(self):
        self._reject(
            "xs = [1, 2]\nr = [b for a in xs for b in range(0, a)]\nprint(len(r))\n",
            "cannot depend on an earlier target",
        )

    def test_an_inner_source_that_could_raise_when_hoisted_is_rejected(self):
        # CPython never evaluates the inner source when the outer one is
        # empty, so hoisting a division out of it would raise where CPython
        # would not.
        self._reject(
            "xs = []\n"
            "p = 1\n"
            "q = 0\n"
            "r = [1 for a in xs for b in range(0, p // q)]\n",
            "must therefore be",
        )

    def test_a_float_element_in_an_aggregate_is_rejected(self):
        self._reject(
            "xs = [1.0, 2.0]\nprint(sum(v * 0.5 for v in xs))\n",
            "would need a float accumulator",
        )

    def test_set_and_dict_comprehensions_are_rejected_by_name(self):
        # There IS a runtime set now, so the reason is no longer that one: a
        # set comprehension would build a set nothing may iterate.
        self._reject(
            "xs = [1, 2]\ns = {v for v in xs}\nprint(1)\n",
            "would build a set nothing may iterate",
        )
        self._reject(
            "xs = [1, 2]\nd = {v: v * 2 for v in xs}\nprint(1)\n",
            "no runtime dict",
        )

    def test_reordering_a_comprehension_of_bools_is_rejected(self):
        self._reject(
            self._RUNTIME_N
            + "xs = [n > 1, n < 1]\n"
            + "print(sorted([v for v in xs])[0])\n",
            "holds bools",
        )


class DeleteListElementTests(unittest.TestCase):
    """`del xs[i]`: the tail shifts down and the header length drops by one."""

    _RUNTIME_N = "n = 0\nfor i in range(0, 3):\n    n = n + 1\n"

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
                artifact = root / f"program-{target}.bin"
                compile_native(entry, artifact, target, clean=True)
                magic = _MAGIC[target.split("-")[0]]
                self.assertEqual(
                    artifact.read_bytes()[: len(magic)],
                    magic,
                    f"{target} del image has a broken header",
                )
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

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    def test_deleting_from_the_middle_front_and_back(self):
        self._run(
            self._RUNTIME_N
            + "xs = [10, 20, 30, 40]\n"
            + "del xs[1]\n"
            + "print(len(xs), xs[0], xs[1], xs[2])\n"
            + "del xs[-1]\n"
            + "print(len(xs), xs[0], xs[1])\n"
            + "del xs[n - 3]\n"
            + "print(len(xs), xs[0])\n"
            + "del xs[0]\n"
            + "print(len(xs))\n",
            b"3 10 30 40\n2 10 30\n1 30\n0\n",
        )

    def test_a_second_name_for_one_list_is_rejected(self):
        # A list variable holds the block, not a reference to it. `del` shifts
        # in place and would be visible through both names, but an append moves
        # the block and writes the new address to only one slot - so the two
        # names would disagree. Refusing the alias is what keeps them honest.
        self._reject(
            self._RUNTIME_N + "xs = [10, 20, 30]\nys = xs\ndel xs[1]\n"
            "print(len(ys))\n",
            "not a reference to it",
        )

    def test_deleting_a_float_element(self):
        # A float lives in the element as its bit pattern, so the same 8-byte
        # word shift moves it.
        self._run(
            self._RUNTIME_N
            + "fs = [1.5, 2.5, 3.5]\ndel fs[1]\n"
            + "print(len(fs), fs[0], fs[1])\n",
            b"2 1.5 3.5\n",
        )

    def test_a_deleted_bool_list_still_prints_True(self):
        self._run(
            self._RUNTIME_N
            + "bs = [n > 1, n < 1, n > 2]\ndel bs[1]\n"
            + "print(len(bs), bs[0], bs[1])\n",
            b"2 True True\n",
        )

    def test_an_out_of_range_del_raises_IndexError(self):
        self._run(
            self._RUNTIME_N
            + "xs = [1, 2]\n"
            + "try:\n    del xs[n + 5]\nexcept IndexError:\n    print('caught')\n"
            + "print(len(xs))\n"
            + "del xs[n + 5]\n",
            b"caught\n2\n",
            expected_exit=1,
        )

    def test_deleting_inside_a_loop_shrinks_the_same_block(self):
        self._run(
            self._RUNTIME_N
            + "xs = [1, 2, 3, 4, 5]\n"
            + "for i in range(0, n):\n    del xs[0]\n"
            + "print(len(xs), xs[0], xs[1])\n",
            b"2 4 5\n",
        )

    def test_one_del_statement_taking_a_dict_key_and_a_list_element(self):
        # Python evaluates the targets left to right, and the two kinds of
        # target are lowered by different code; one statement drives both.
        self._run(
            "d = {5: 1.5, 3: -0.25, 9: 0.0}\nxs = [1, 2, 3]\n"
            "del d[9], xs[0]\n"
            "for k, v in d.items():\n    print(k, v)\n"
            "print(len(xs), xs[0])\n",
            b"5 1.5\n3 -0.25\n2 2\n",
        )

    def test_del_on_a_dict_slice_is_rejected_by_name(self):
        self._reject(
            "d = {1: 2}\ndel d[0:1]\nprint(len(d))\n",
            "a slice is unhashable",
        )

    def test_del_on_a_name_is_rejected_by_name(self):
        self._reject(
            "x = 1\ndel x\nprint(2)\n",
            "a native variable is a stack slot holding a value",
        )

    def test_del_on_a_slice_is_rejected_by_name(self):
        self._reject(
            "xs = [1, 2, 3]\ndel xs[0:2]\nprint(len(xs))\n",
            "del on a list slice is not supported",
        )

    def test_del_on_an_attribute_is_rejected_by_name(self):
        self._reject(
            "class C:\n    def __init__(self):\n        self.a = 1\n"
            "c = C()\ndel c.a\nprint(1)\n",
            "del on an attribute is not supported",
        )

    def test_del_while_a_for_loop_walks_the_list_is_rejected(self):
        # The walk takes the length once, so a shortened list would yield an
        # element CPython's iterator skips. Refused rather than mis-yielded.
        self._reject(
            self._RUNTIME_N
            + "xs = [1, 2, 3]\nfor v in xs:\n    del xs[0]\nprint(len(xs))\n",
            "while a for loop is walking it",
        )
        self._reject(
            self._RUNTIME_N
            + "xs = [1, 2, 3]\nfor k, v in enumerate(xs):\n    del xs[0]\n"
            + "print(len(xs))\n",
            "while a for loop is walking it",
        )

    def test_deleting_a_different_list_inside_a_walk_is_allowed(self):
        self._run(
            self._RUNTIME_N
            + "xs = [1, 2, 3]\nys = [4, 5, 6, 7]\n"
            + "for v in xs:\n    del ys[0]\n"
            + "print(len(ys), ys[0])\n",
            b"1 7\n",
        )


class NestedListTests(unittest.TestCase):
    """Lists whose elements are strings or other lists.

    An element is eight bytes either way, so a string or an inner list travels
    as the address of its block and only the element kind has to be carried
    along. Every expectation here is CPython's own output for the same source,
    diffed against the darwin-arm64 binary.
    """

    _RUNTIME_N = "n = 0\nfor i in range(0, 3):\n    n = n + 1\n"

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
                artifact = root / f"program-{target}.bin"
                compile_native(entry, artifact, target, clean=True)
                magic = _MAGIC[target.split("-")[0]]
                self.assertEqual(
                    artifact.read_bytes()[: len(magic)],
                    magic,
                    f"{target} nested-list image has a broken header",
                )
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

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    def test_a_literal_of_literals_indexes_and_iterates(self):
        self._run(
            "xs = [[1, 2], [3]]\n"
            "print(xs[0][1])\nprint(len(xs[0]))\nprint(len(xs))\n"
            "for row in xs:\n    print(len(row))\n    print(row[0])\n",
            b"2\n2\n2\n2\n1\n1\n3\n",
        )

    def test_an_empty_literal_takes_its_kind_from_the_annotation(self):
        self._run(
            "xs: list[list[int]] = []\n"
            + self._RUNTIME_N
            + "for i in range(0, 3):\n    xs.append([i, n])\n"
            "print(xs[2][1])\nprint(len(xs))\n",
            b"3\n3\n",
        )
        self._reject(
            "xs: list[list[int]] = []\nxs.append(7)\n",
            "this list holds lists of signed 64-bit integers",
        )

    def test_an_annotation_nests_to_any_depth(self):
        self._run(
            "xs: list[list[list[int]]] = []\n"
            + self._RUNTIME_N
            + "xs.append([[n], [n, n]])\n"
            "print(xs[0][1][0])\nprint(len(xs[0][1]))\n",
            b"3\n2\n",
        )

    def test_a_nested_bool_still_prints_as_a_bool(self):
        # `xs[0][1]` has no name whose bookkeeping could say it is a bool, so
        # the answer has to come from the list's own element kind.
        self._run(
            self._RUNTIME_N
            + "xs = [[n > 1, n < 1], [n < 1]]\n"
            "print(xs[0][0])\nprint(xs[0][1])\n"
            "for row in xs:\n    print(row[0])\n",
            b"True\nFalse\nTrue\nFalse\n",
        )

    def test_a_nested_float_keeps_its_bit_pattern(self):
        self._run(
            "xs: list[list[float]] = [[1.5, -0.0], [2.25]]\n"
            "print(xs[0][1])\nprint(xs[1][0])\n",
            b"-0.0\n2.25\n",
        )

    def test_a_list_of_strings_holds_runtime_ones_too(self):
        self._run(
            'parts = ["a", "b"]\ns = ""\ns = s + "c"\n'
            "parts.append(s)\n"
            "print(parts[0])\nprint(parts[2])\nprint(len(parts))\n"
            "print(len(parts[2]))\n"
            "for w in parts:\n    print(w)\n",
            b"a\nc\n3\n1\na\nb\nc\n",
        )

    def test_membership_over_strings_compares_the_bytes(self):
        # The words are block addresses and two equal strings are two
        # allocations, so identity would answer False where CPython says True.
        self._run(
            'parts = ["a", "b"]\ns = ""\ns = s + "a"\n'
            "print(s in parts)\nprint(\"c\" in parts)\n",
            b"True\nFalse\n",
        )

    def test_slices_and_comprehensions_produce_nested_lists(self):
        self._run(
            "xs = [[1, 2], [3], [4, 5, 6]]\n"
            "ys = xs[1:]\nprint(len(ys))\nprint(ys[1][2])\n"
            "zs = [row[0] for row in xs]\nprint(zs[2])\n"
            "ws = [[k, k] for k in range(0, 3)]\nprint(ws[2][1])\n",
            b"2\n6\n4\n2\n",
        )

    def test_appending_a_name_that_is_also_an_element_is_rejected(self):
        # Appending moves the block and writes the new address back to the
        # variable's slot only. The element would be left on the abandoned
        # copy, printing 4 where CPython prints 5.
        self._reject(
            "inner = [1, 2, 3, 4]\nxs: list[list[int]] = []\n"
            "xs.append(inner)\ninner.append(5)\nprint(len(xs[0]))\n",
            "is stored inside another container somewhere in this module",
        )
        # The refusal is whole-module, because a loop's back edge puts the
        # append textually before the store that shared the block.
        self._reject(
            "xs: list[list[int]] = []\nrow = [1]\n"
            "for i in range(0, 3):\n    row.append(i)\n    xs.append(row)\n",
            "is stored inside another container somewhere in this module",
        )
        # A copy is the way out, and leaves the two lists independent.
        self._run(
            "inner = [1, 2, 3, 4]\nxs: list[list[int]] = []\n"
            "xs.append(inner[:])\ninner.append(5)\n"
            "print(len(xs[0]))\nprint(len(inner))\n",
            b"4\n5\n",
        )

    def test_appending_to_a_name_bound_from_an_element_is_rejected(self):
        self._reject(
            "xs = [[1, 2, 3, 4]]\nrow = xs[0]\nrow.append(5)\n",
            "names a list that is an element of another one",
        )
        self._reject(
            "xs = [[1, 2, 3, 4]]\nfor row in xs:\n    row.append(5)\n",
            "names a list that is an element of another one",
        )
        self._reject(
            "xs = [[1, 2]]\nxs[0].append(5)\n",
            "is only called on a list held by a name",
        )

    def test_writes_that_do_not_move_the_block_stay_allowed(self):
        # `xs[i] = v` and `del xs[i]` rewrite the block in place, which CPython
        # sees through every reference to it, so the alias is honest there.
        self._run(
            "xs: list[list[int]] = []\nrow = [1, 2, 3, 4]\n"
            "xs.append(row)\nrow[0] = 9\ndel row[1]\n"
            "print(xs[0][0])\nprint(len(xs[0]))\n",
            b"9\n3\n",
        )

    def test_growing_the_outer_list_carries_the_inner_blocks(self):
        self._run(
            "xs: list[list[int]] = []\n"
            + self._RUNTIME_N
            + "for i in range(0, 8):\n    xs.append([i, i * n])\n"
            "print(len(xs))\nprint(xs[7][1])\nprint(xs[0][0])\n",
            b"8\n21\n0\n",
        )

    def test_a_mixed_list_is_rejected_from_either_side(self):
        self._reject("xs = [1, [2]]\nprint(xs[0])\n", "one list holds one kind")
        self._reject("xs = [[2], 1]\nprint(len(xs))\n", "one list holds one kind")
        self._reject('xs = ["a", 1]\nprint(len(xs))\n', "one list holds one kind")

    def test_printing_a_list_of_numbers(self):
        # CPython prints the repr of every element, which for integers, floats
        # and bools is the same text they print on their own.
        self._run("xs = [1, 2, 3]\nprint(xs)\n", b"[1, 2, 3]\n")
        self._run("xs = [1.5, 2.0, -0.0]\nprint(xs)\n", b"[1.5, 2.0, -0.0]\n")
        self._run("xs: list[int] = []\nprint(xs)\n", b"[]\n")
        self._run("xs = [7]\nprint(xs)\n", b"[7]\n")

    def test_printing_a_list_of_strings_or_lists_is_rejected(self):
        # The repr of a runtime string means choosing its quote character and
        # its backslash escapes, which is not implemented.
        self._reject("xs = [[1, 2]]\nprint(xs)\n", "renders a list of integers")
        self._reject('parts = ["a"]\nprint(parts[0:1])\n', "renders a list of integers")

    def test_printing_an_inner_list_of_numbers(self):
        # The outer list cannot be printed, because its elements are lists; one
        # of those elements can, because its own elements are numbers.
        self._run("xs = [[1, 2], [3]]\nprint(xs[0])\nprint(xs[1])\n",
                  b"[1, 2]\n[3]\n")

    def test_reordering_and_reducing_strings_or_lists_is_rejected(self):
        # Sorting a list of strings used to be here too, and min() and max()
        # over one have since joined it: both compare the text rather than the
        # block addresses. sum() stays rejected because adding strings up has
        # no answer of the same kind - CPython raises TypeError there - so only
        # the message it is rejected with has moved.
        self._reject(
            'parts = ["a"]\nprint(sum(parts))\n',
            "only min() and max() have an answer",
        )
        self._reject("xs = [[1], [2]]\nxs.sort()\n", "compare block addresses")
        self._reject("xs = [[1], [2]]\nprint([1] in xs)\n", "over a list of lists")


class StringMinMaxTests(unittest.TestCase):
    """`min()` and `max()` over a list of strings.

    They answer with one of the elements, so over strings the answer is a
    string. The comparison is the text one sorting already uses; comparing the
    slots would order by where the arena put each block.
    """

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
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

    def test_the_answer_is_one_of_the_strings(self):
        self._run(
            'xs = ["pear", "apple", "fig"]\nprint(max(xs), min(xs))\n',
            b"pear apple\n",
        )

    def test_a_runtime_string_takes_part(self):
        self._run(
            's = ""\nfor _i in range(0, 1):\n    s = s + "zebra"\n'
            'xs = ["apple", s, "mango"]\nprint(max(xs), min(xs))\n',
            b"zebra apple\n",
        )

    def test_a_prefix_sorts_before_what_extends_it(self):
        self._run('xs = ["ab", "a", "abc"]\nprint(min(xs), max(xs))\n', b"a abc\n")

    def test_non_ascii_compares_by_code_point(self):
        # Read as signed, a lead byte is negative and the accented letter would
        # come out as the smallest of the three.
        self._run(
            'xs = ["\u00e9", "a", "z"]\nprint(max(xs), min(xs))\n',
            "\u00e9 a\n".encode("utf-8"),
        )

    def test_an_empty_list_raises(self):
        self._run("xs: list[str] = []\nprint(max(xs))\n", b"", expected_exit=1)

    def test_the_result_is_a_string_and_behaves_like_one(self):
        self._run(
            'xs = ["b", "aa"]\nlongest = max(xs)\nprint(longest, len(longest))\n',
            b"b 1\n",
        )


class SplitAndJoinTests(unittest.TestCase):
    """`str.split()` and `str.join()`, which needed a list of strings first."""

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
                artifact = root / f"program-{target}.bin"
                compile_native(entry, artifact, target, clean=True)
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

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    def _runtime(self, name: str, text: str) -> str:
        return f'{name} = ""\n{name} = {name} + {text!r}\n'

    def test_whitespace_split_drops_every_empty_piece(self):
        self._run(
            self._runtime("s", "  a  b  ")
            + "print(len(s.split()))\n"
            + "for w in s.split():\n    print(\"[\" + w + \"]\")\n"
            + self._runtime("t", "   ")
            + "print(len(t.split()))\n"
            + self._runtime("u", "")
            + "print(len(u.split()))\n",
            b"2\n[a]\n[b]\n0\n0\n",
        )

    def test_whitespace_split_uses_the_unicode_set(self):
        # The same 29 code points `.strip()` uses, so a non-breaking space and
        # an ideographic space separate here exactly as they do in CPython.
        self._run(
            self._runtime("s", "\t\n x   y 　 z \n")
            + "print(len(s.split()))\n"
            + "for w in s.split():\n    print(w)\n",
            "3\nx\ny\nz\n".encode("utf-8"),
        )

    def test_a_separator_split_keeps_the_empty_pieces(self):
        self._run(
            self._runtime("t", ",a,")
            + self._runtime("c", ",")
            + "print(len(t.split(c)))\n"
            + 'print(t.split(c)[0] == "")\nprint(t.split(c)[2] == "")\n'
            + self._runtime("u", "")
            + "print(len(u.split(c)))\n"
            + self._runtime("v", "aaa")
            + self._runtime("aa", "aa")
            + "print(len(v.split(aa)))\nprint(v.split(aa)[1])\n",
            b"3\nTrue\nTrue\n1\n2\na\n",
        )

    def test_an_empty_separator_raises_a_catchable_value_error(self):
        self._run(
            self._runtime("s", "abc")
            + "sep = s[0:0]\n"
            + "try:\n    print(len(s.split(sep)))\n"
            + "except ValueError:\n    print(\"caught\")\n"
            + "print(len(s.split(s[0:1])))\n",
            b"caught\n2\n",
        )
        # A separator already known to be empty can only fail, so say so at
        # build time rather than emitting a program that always dies.
        self._reject('print(len("abc".split("")))\n', "raises ValueError: empty separator")

    def test_join_concatenates_with_one_allocation(self):
        self._run(
            'parts = ["a", "b", "c"]\n'
            'print("".join(parts))\nprint("-".join(parts))\n'
            'empty: list[str] = []\nprint("-".join(empty) == "")\n'
            + self._runtime("s", "x y z")
            + 'print("|".join(s.split()))\n',
            b"abc\na-b-c\nTrue\nx|y|z\n",
        )

    def test_the_shapes_outside_the_subset_are_rejected(self):
        self._reject(
            self._runtime("s", "a b") + 'print(len(s.split(" ", 1)))\n',
            "the maxsplit form is not in the subset",
        )
        self._reject(
            'xs = [1, 2]\nprint("-".join(xs))\n',
            "takes a list of strings",
        )


class StringIndexingTests(unittest.TestCase):
    """`s[i]` and `for ch in s` - both walk by code point, not by byte."""

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
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

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    def _runtime(self, name: str, text: str) -> str:
        return f'{name} = ""\n{name} = {name} + {text!r}\n'

    # --- indexing ----------------------------------------------------------

    def test_index_returns_a_code_point_not_a_byte(self):
        # "héllo" is six bytes and five code points. Indexing by byte would
        # answer with half of the é here.
        self._run(
            self._runtime("s", "héllo")
            + "print(s[0])\nprint(s[1])\nprint(s[4])\n",
            "h\né\no\n".encode("utf-8"),
        )

    def test_negative_index_counts_from_the_end(self):
        self._run(
            self._runtime("s", "héllo") + "print(s[-1])\nprint(s[-5])\n",
            "o\nh\n".encode("utf-8"),
        )

    def test_index_out_of_range_reports_index_error(self):
        # A slice clamps and answers with "", but an index must raise.
        self._run(
            self._runtime("s", "ab") + "print(s[5])\n", b"", expected_exit=1
        )

    def test_negative_index_out_of_range_reports_index_error(self):
        self._run(
            self._runtime("s", "ab") + "print(s[-3])\n", b"", expected_exit=1
        )

    def test_a_string_can_be_rebuilt_by_indexing_it(self):
        self._run(
            self._runtime("s", "héllo")
            + "out = ''\n"
            + "i = 0\n"
            + "while i < len(s):\n    out = out + s[i]\n    i = i + 1\n"
            + "print(out)\nprint(out == s)\n",
            "héllo\nTrue\n".encode("utf-8"),
        )

    # --- iteration ---------------------------------------------------------

    def test_iteration_yields_one_code_point_at_a_time(self):
        self._run(
            self._runtime("s", "héllo")
            + "n = 0\nfor ch in s:\n    n = n + 1\n    print(ch)\nprint(n)\n",
            "h\né\nl\nl\no\n5\n".encode("utf-8"),
        )

    def test_iterating_an_empty_string_runs_the_body_zero_times(self):
        self._run(
            self._runtime("s", "")
            + "for ch in s:\n    print('body ran')\nprint('done')\n",
            b"done\n",
        )

    def test_break_skips_the_else_body(self):
        self._run(
            self._runtime("s", "abc")
            + "for ch in s:\n    if ch == 'b':\n        break\n"
            + "else:\n    print('no b')\n"
            + "print('after')\n",
            b"after\n",
        )

    def test_the_else_body_runs_when_nothing_broke(self):
        self._run(
            self._runtime("s", "abc")
            + "for ch in s:\n    if ch == 'z':\n        break\n"
            + "else:\n    print('no z')\n",
            b"no z\n",
        )

    def test_continue_resumes_at_the_next_code_point(self):
        self._run(
            self._runtime("s", "a,b,,c")
            + "out = ''\n"
            + "for ch in s:\n    if ch == ',':\n        continue\n    out = out + ch\n"
            + "print(out)\n",
            b"abc\n",
        )

    def test_iteration_reverses_a_string(self):
        self._run(
            self._runtime("s", "héllo")
            + "out = ''\nfor ch in s:\n    out = ch + out\nprint(out)\n",
            "olléh\n".encode("utf-8"),
        )

    def test_nested_loops_over_the_same_string_keep_separate_positions(self):
        self._run(
            self._runtime("s", "ab")
            + "for a in s:\n    for b in s:\n        print(a + b)\n",
            b"aa\nab\nba\nbb\n",
        )

    def test_reading_the_loop_name_afterwards_is_refused(self):
        # The string may be empty, and then CPython leaves the name unbound.
        # The slot would hold an unrelated address, so this is refused rather
        # than dereferenced - the same rule a list loop already follows.
        self._reject(
            self._runtime("s", "abc") + "for ch in s:\n    pass\nprint(ch)\n",
            "may be unbound",
        )

    def test_a_name_bound_before_the_loop_survives_it(self):
        self._run(
            "ch = 'z'\n"
            + self._runtime("s", "")
            + "for ch in s:\n    pass\nprint(ch)\n",
            b"z\n",
        )


class ConditionalBindingTests(unittest.TestCase):
    """A name one branch binds and another does not.

    CPython raises NameError. There is no run-time bit recording whether a slot
    was written, so the read is refused at build time - the alternative is what
    these programs used to do: print a stale constant, or, for a dict, probe an
    address that is not a table until the process is killed.
    """

    _RUNTIME = "n = 0\nfor i in range(0, 3):\n    n = n + 1\n"

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
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

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))
            # CPython must agree that the program is broken, so that what is
            # refused here is not something that would have worked.
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(reference.returncode, 1)
            self.assertIn(b"NameError", reference.stderr)

    def test_every_kind_bound_in_one_branch_only_is_refused(self):
        for binding, use in (
            ("x = 1", "x"),
            ("f = 1.5", "f + 1"),
            ("s = 'hi'", "len(s)"),
            ("xs = [1]", "len(xs)"),
            ("xs = [1]", "xs[0]"),
            ("d = {1: 2}", "len(d)"),
            ("d = {1: 2}", "d[1]"),
            ("d = {1: 2}", "1 in d"),
            ("st = {1, 2}", "len(st)"),
            ("tp = (1, 2)", "tp[0]"),
        ):
            with self.subTest(use=use):
                self._reject(
                    f"{self._RUNTIME}if n > 5:\n    {binding}\nprint({use})\n",
                    "may be unbound",
                )

    def test_binding_in_every_branch_is_accepted(self):
        self._run(
            self._RUNTIME
            + "if n > 5:\n    d = {1: 2}\nelse:\n    d = {3: 4}\nprint(len(d))\n",
            b"1\n",
        )

    def test_an_elif_chain_that_binds_everywhere_is_accepted(self):
        # An elif is a nested If in the tree, so this only works if the
        # analysis descends into the else branch rather than stopping there.
        self._run(
            self._RUNTIME
            + "if n > 5:\n    x = 1\nelif n > 2:\n    x = 2\nelse:\n    x = 3\n"
            + "print(x)\n",
            b"2\n",
        )

    def test_a_branch_that_cannot_fall_through_does_not_count(self):
        # Nothing reaches the print by way of the raise, so `x` is bound on
        # every path that does reach it.
        self._run(
            self._RUNTIME
            + "if n > 5:\n    raise SystemExit(3)\nelse:\n    x = 9\nprint(x)\n",
            b"9\n",
        )

    def test_rebinding_after_the_branch_clears_the_refusal(self):
        self._run(
            self._RUNTIME
            + "if n > 5:\n    xs = [1]\nxs = [9, 9]\nprint(len(xs))\n",
            b"2\n",
        )

    def test_using_the_name_inside_its_own_branch_is_accepted(self):
        self._run(
            self._RUNTIME + "if n > 1:\n    d = {1: 2}\n    print(len(d))\n",
            b"1\n",
        )

    def test_print_writes_nothing_when_a_later_argument_raises(self):
        # print() evaluates all of its arguments and then writes them, so the
        # first value must not reach the terminal when the second raises.
        # This used to print "1 " and then report the error.
        self._run(
            self._RUNTIME
            + "z = n - 3\n"
            + "print(7, 5 // z)\n",
            b"",
            expected_exit=1,
        )

    def test_pre_evaluating_arguments_keeps_bools_and_kinds_intact(self):
        # The pre-evaluation binds each scalar to a hidden name, and a name
        # has to render exactly as the expression did - True, not 1.
        self._run(
            self._RUNTIME
            + "xs = [1, 2]\nd = {1: 2}\n"
            + "print('x', n * 3, 1.5 + n, xs, d, n > 1)\n",
            b"x 9 4.5 [1, 2] {1: 2} True\n",
        )

    def test_a_function_defined_under_a_runtime_condition_is_refused(self):
        # Both bodies exist, but a call is inlined from one of them chosen at
        # build time. This used to compile and run the branch that did not.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(
                self._RUNTIME
                + "if n > 1:\n    def g(v):\n        return v + 1\n"
                + "else:\n    def g(v):\n        return v + 100\n"
                + "print(g(5))\n",
                encoding="utf-8",
            )
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn("only known at run time", str(caught.exception))
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(reference.stdout, b"6\n")


class DictGetAndListMethodsTests(unittest.TestCase):
    """`d.get()`, and the list methods that were missing: pop, insert, remove,
    index and count."""

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
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
            if expected_exit:
                # The message on stderr must be the one CPython gives, not
                # just the same exception type.
                self.assertEqual(
                    native.stderr.strip().splitlines()[-1],
                    reference.stderr.strip().splitlines()[-1],
                )

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    # --- dict.get -----------------------------------------------------------

    def test_get_answers_the_value_or_the_default(self):
        self._run(
            "d = {1: 10, 2: 20}\nprint(d.get(1, 0), d.get(9, -1))\n", b"10 -1\n"
        )

    def test_get_works_on_string_keys_and_float_values(self):
        self._run(
            's = ""\ns = s + "a"\n'
            'd = {"a": 1.5}\nprint(d.get(s, 0.0), d.get("zz", 9.25))\n',
            b"1.5 9.25\n",
        )

    def test_get_sees_a_deleted_key_as_absent(self):
        # A delete leaves a tombstone, which the probe must walk over rather
        # than stop at.
        self._run(
            "d = {1: 10, 2: 20}\ndel d[1]\nprint(d.get(1, -1), d.get(2, -1))\n",
            b"-1 20\n",
        )

    def test_the_one_argument_form_is_refused(self):
        self._reject("d = {1: 10}\nprint(d.get(1))\n", "None is not in the subset")

    # --- pop ---------------------------------------------------------------

    def test_pop_takes_the_last_element_by_default(self):
        self._run("xs = [1, 2, 3]\nprint(xs.pop(), xs)\n", b"3 [1, 2]\n")

    def test_pop_takes_the_element_at_an_index(self):
        self._run(
            "xs = [1, 2, 3]\nprint(xs.pop(0), xs)\nprint(xs.pop(-1), xs)\n",
            b"1 [2, 3]\n3 [2]\n",
        )

    def test_pop_answers_strings_and_floats(self):
        self._run('xs = ["a", "b"]\nprint(xs.pop(), len(xs))\n', b"b 1\n")
        self._run("ys = [1.5, 2.5]\nprint(ys.pop(0), ys)\n", b"1.5 [2.5]\n")

    def test_popping_an_empty_list_uses_cpythons_wording(self):
        self._run("xs = [1]\nxs.pop()\nxs.pop()\n", b"", expected_exit=1)

    def test_popping_out_of_range_uses_cpythons_wording(self):
        self._run("xs = [1]\nxs.pop(5)\n", b"", expected_exit=1)

    def test_pop_while_a_loop_walks_the_list_is_refused(self):
        self._reject(
            "xs = [1, 2, 3]\nfor v in xs:\n    xs.pop()\n",
            "cannot shorten",
        )

    # --- insert, remove, index, count --------------------------------------

    def test_insert_puts_the_value_at_the_index(self):
        self._run("xs = [1, 3]\nxs.insert(1, 2)\nprint(xs)\n", b"[1, 2, 3]\n")

    def test_insert_clamps_instead_of_raising(self):
        # Unlike indexing, insert() never raises: past the end it appends and
        # before the start it prepends.
        self._run(
            "xs = [1, 2]\nxs.insert(99, 9)\nxs.insert(-99, 0)\nprint(xs)\n",
            b"[0, 1, 2, 9]\n",
        )

    def test_insert_grows_the_block(self):
        self._run(
            "xs: list[int] = []\nfor i in range(0, 5):\n    xs.insert(0, i)\n"
            "print(xs)\n",
            b"[4, 3, 2, 1, 0]\n",
        )

    def test_remove_drops_the_first_match_only(self):
        self._run("xs = [1, 2, 3, 2]\nxs.remove(2)\nprint(xs)\n", b"[1, 3, 2]\n")

    def test_removing_something_absent_reports_value_error(self):
        self._run("xs = [1]\nxs.remove(9)\n", b"", expected_exit=1)

    def test_index_reports_the_first_position(self):
        self._run("xs = [5, 6, 7, 6]\nprint(xs.index(6), xs.index(5))\n", b"1 0\n")

    def test_indexing_something_absent_reports_value_error(self):
        self._run("xs = [5]\nprint(xs.index(9))\n", b"", expected_exit=1)

    def test_count_tallies_every_match(self):
        self._run("xs = [1, 2, 2, 3]\nprint(xs.count(2), xs.count(9))\n", b"2 0\n")

    def test_index_and_count_work_on_strings_and_floats(self):
        self._run(
            'xs = ["a", "b", "a"]\nprint(xs.index("b"), xs.count("a"))\n', b"1 2\n"
        )
        self._run("ys = [1.5, 1.5, 2.0]\nprint(ys.count(1.5), ys.index(2.0))\n", b"2 2\n")

    def test_print_keeps_argument_order_when_one_of_them_mutates(self):
        # The pre-evaluation that keeps a raising argument from printing the
        # earlier ones must not reorder these: len() has to see the shortened
        # list, because pop() is written first.
        self._run("xs = [1, 2, 3]\nprint(xs.pop(), len(xs))\n", b"3 2\n")


class CodePointAndArithmeticBuiltinTests(unittest.TestCase):
    """`ord()`, `chr()`, `divmod()` and `sum(xs, start)`."""

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
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

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    def _runtime(self, name: str, text: str) -> str:
        return f'{name} = ""\n{name} = {name} + {text!r}\n'

    def test_ord_reads_every_utf8_width(self):
        self._run(
            self._runtime("a", "A")
            + self._runtime("b", "\u00e9")
            + self._runtime("c", "\u4e2d")
            + self._runtime("d", "\U0001f600")
            + "print(ord(a), ord(b), ord(c), ord(d))\n",
            b"65 233 20013 128512\n",
        )

    def test_ord_of_a_longer_string_reports_type_error(self):
        self._run(self._runtime("s", "ab") + "print(ord(s))\n", b"", expected_exit=1)

    def test_chr_writes_every_utf8_width(self):
        self._run(
            "print(chr(65))\nprint(chr(233))\nprint(chr(20013))\nprint(chr(128512))\n",
            "A\n\u00e9\n\u4e2d\n\U0001f600\n".encode("utf-8"),
        )

    def test_chr_and_ord_round_trip_through_iteration(self):
        # The decoder, the encoder and the code-point walk all have to agree
        # on where one character ends and the next begins.
        self._run(
            self._runtime("s", "h\u00e9llo\u4e2d\U0001f600")
            + "out = ''\nfor ch in s:\n    out = out + chr(ord(ch))\n"
            + "print(out, out == s)\n",
            "h\u00e9llo\u4e2d\U0001f600 True\n".encode("utf-8"),
        )

    def test_chr_out_of_range_reports_value_error(self):
        self._run(
            "n = 0\nfor i in range(0, 5):\n    n = n + 1\nprint(chr(1114112 + n))\n",
            b"",
            expected_exit=1,
        )

    def test_chr_of_a_lone_surrogate_is_reported(self):
        # CPython hands back a string that fails later, when it is written out.
        # A native string is its UTF-8 bytes, so there is nothing to hand back
        # and the report happens here instead.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text("print(chr(55296))\n", encoding="utf-8")
            compile_native(entry, root / "program.bin", "darwin-arm64", clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run([str(root / "program.bin")], capture_output=True)
            self.assertEqual(native.returncode, 1)
            self.assertIn(b"surrogate", native.stderr)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(reference.returncode, 1)

    def test_divmod_answers_both_halves(self):
        self._run("q, r = divmod(17, 5)\nprint(q, r)\n", b"3 2\n")

    def test_divmod_follows_python_rounding_for_negatives(self):
        # Python floors, so -17 // 5 is -4 and the remainder is positive.
        self._run(
            "n = 0\nfor i in range(0, 17):\n    n = n + 1\n"
            "q, r = divmod(0 - n, 5)\nprint(q, r)\n",
            b"-4 3\n",
        )

    def test_divmod_evaluates_each_operand_once(self):
        # Both halves read the same operands, so a side-effecting one must not
        # run twice: this pops a single element.
        self._run("xs = [7]\nq, r = divmod(xs.pop(), 2)\nprint(q, r, len(xs))\n", b"3 1 0\n")

    def test_divmod_by_zero_raises(self):
        self._run(
            "n = 0\nfor i in range(0, 3):\n    n = n + 1\n"
            "q, r = divmod(10, n - 3)\nprint(q, r)\n",
            b"",
            expected_exit=1,
        )

    def test_divmod_outside_a_two_name_assignment_says_so(self):
        self._reject("print(divmod(7, 2))\n", "q, r = divmod(a, b)")

    def test_sum_takes_a_start_value(self):
        self._run("xs = [1, 2]\nprint(sum(xs, 10))\n", b"13\n")
        self._run("ys: list[int] = []\nprint(sum(ys, 5))\n", b"5\n")

    def test_a_float_start_is_refused(self):
        self._reject("xs = [1, 2]\nprint(sum(xs, 1.5))\n", "start must be one too")


class StringOrderingTests(unittest.TestCase):
    """`<`, `<=`, `>`, `>=` between strings, and the sorting they unlock.

    UTF-8 was built so that comparing the bytes of two sequences puts them in
    the same order as comparing the code points, so a byte walk gives the order
    CPython gives without decoding anything.
    """

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
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

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    def _runtime(self, name: str, text: str) -> str:
        return f'{name} = ""\n{name} = {name} + {text!r}\n'

    def test_ordering_two_runtime_strings(self):
        self._run(
            self._runtime("t", "b")
            + 'print(t < "c", t > "c", t <= "b", t >= "b")\n',
            b"True False True True\n",
        )

    def test_a_prefix_sorts_before_the_longer_string(self):
        self._run(
            self._runtime("a", "ab")
            + self._runtime("b", "abc")
            + 'print(a < b, b < a, a < "ab", a <= "ab")\n',
            b"True False False True\n",
        )

    def test_an_empty_string_sorts_first(self):
        self._run(
            self._runtime("a", "")
            + self._runtime("b", "x")
            + "print(a < b, b < a, a < a, a <= a)\n",
            b"True False False True\n",
        )

    def test_bytes_are_unsigned_so_non_ascii_sorts_last(self):
        # Read as signed, a lead byte is negative and "\u00e9" would sort
        # before "z" instead of after it.
        self._run(
            self._runtime("a", "z")
            + self._runtime("b", "\u00e9")
            + "print(a < b, b < a)\n",
            b"True False\n",
        )

    def test_a_chain_compares_each_pair(self):
        # The idiom this was all for: a digit test on a loop variable.
        self._run(
            self._runtime("s", "12a4")
            + "n = 0\nfor ch in s:\n    if '0' <= ch <= '9':\n        n = n + 1\n"
            + "print(n)\n",
            b"3\n",
        )

    def test_sorting_a_list_of_strings(self):
        self._run(
            'xs = ["pear", "apple", "fig"]\nxs.sort()\nprint("|".join(xs))\n',
            b"apple|fig|pear\n",
        )

    def test_sorting_handles_prefixes_duplicates_and_the_empty_string(self):
        self._run(
            'xs = ["ab", "a", "abc", "a", ""]\nxs.sort()\nprint("|".join(xs))\n',
            b"|a|a|ab|abc\n",
        )

    def test_sorted_leaves_the_original_alone_and_reverse_works(self):
        self._run(
            'xs = ["b", "a", "c"]\n'
            'print("|".join(sorted(xs)), "|".join(xs))\n'
            'print("|".join(sorted(xs, reverse=True)))\n',
            b"a|b|c b|a|c\nc|b|a\n",
        )

    def test_split_sort_join_round_trip(self):
        self._run(
            self._runtime("s", "pear apple fig")
            + 'w = s.split()\nw.sort()\nprint(" ".join(w))\n',
            b"apple fig pear\n",
        )

    def test_a_list_of_lists_still_has_no_order(self):
        self._reject("xs = [[1], [2]]\nxs.sort()\n", "compare block addresses")


class ZipAndRoundTests(unittest.TestCase):
    """`for a, b in zip(...)` and `round(x)`."""

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
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

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    def test_zip_walks_two_lists_together(self):
        self._run(
            "a = [1, 2, 3]\nb = [10, 20, 30]\n"
            "for x, y in zip(a, b):\n    print(x + y)\n",
            b"11\n22\n33\n",
        )

    def test_zip_stops_with_the_shortest(self):
        self._run(
            "a = [1, 2, 3]\nb = [10, 20]\n"
            "for x, y in zip(a, b):\n    print(x, y)\nprint('done')\n",
            b"1 10\n2 20\ndone\n",
        )

    def test_zip_takes_more_than_two_lists(self):
        self._run(
            "a = [1, 2]\nb = [3, 4]\nc = [5, 6]\n"
            "for x, y, z in zip(a, b, c):\n    print(x, y, z)\n",
            b"1 3 5\n2 4 6\n",
        )

    def test_zip_over_different_element_kinds(self):
        self._run(
            'names = ["a", "b"]\nvals = [1.5, 2.5]\n'
            "for n, v in zip(names, vals):\n    print(n, v)\n",
            b"a 1.5\nb 2.5\n",
        )

    def test_zip_keeps_bools_printing_as_bools(self):
        self._run(
            "flags = [True, False]\nns = [1, 2]\n"
            "for f, n in zip(flags, ns):\n    print(f, n)\n",
            b"True 1\nFalse 2\n",
        )

    def test_an_empty_list_ends_the_walk_before_it_starts(self):
        self._run(
            "a: list[int] = []\nb = [1]\n"
            "for x, y in zip(a, b):\n    print(x, y)\nprint('done')\n",
            b"done\n",
        )

    def test_break_skips_the_else_body(self):
        self._run(
            "a = [1, 2, 3]\nb = [4, 5, 6]\n"
            "for x, y in zip(a, b):\n    if x == 2:\n        break\n"
            "else:\n    print('no two')\nprint('after')\n",
            b"after\n",
        )

    def test_the_name_count_must_match_the_list_count(self):
        self._reject(
            "a = [1]\nb = [2]\nfor x in zip(a, b):\n    print(x)\n",
            "walks 2 lists",
        )

    def test_round_breaks_ties_toward_the_even_number(self):
        # 2.5 rounds down and 3.5 rounds up, which is Python and not the
        # away-from-zero rule.
        self._run(
            "".join(
                f"x = 0.0\nx = x + {value!r}\nprint(round(x))\n"
                for value in (2.5, 3.5, -2.5, -1.5, 0.5, -0.5)
            ),
            b"2\n4\n-2\n-2\n0\n0\n",
        )

    def test_round_on_ordinary_values(self):
        self._run(
            "".join(
                f"x = 0.0\nx = x + {value!r}\nprint(round(x))\n"
                for value in (1.4999, 2.4999999, 123.456, -123.456, 7.0, -7.0)
            ),
            b"1\n2\n123\n-123\n7\n-7\n",
        )

    def test_round_of_an_integer_is_itself(self):
        self._run("n = 0\nfor i in range(0, 5):\n    n = n + 1\nprint(round(n))\n", b"5\n")

    def test_the_two_argument_form_says_what_it_would_take(self):
        self._reject("x = 1.55\nprint(round(x, 1))\n", "rounds in decimal")


class RenderingAFunctionParameterTests(unittest.TestCase):
    """`str(n)` and f-strings over a function's own parameters.

    A string-returning function is inlined with its string parameters bound to
    the caller's blocks. Its numeric parameters were not bound at all, so any
    call that had to render one - which is most of what a formatting helper
    does - was refused as "not in the integer subset".
    """

    def _run(self, source: str, expected_stdout: bytes) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
                compile_native(entry, root / f"program-{target}.bin", target, clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run(
                [str(root / "program-darwin-arm64.bin")], capture_output=True
            )
            self.assertEqual(native.stdout, expected_stdout)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.stdout, reference.stdout)

    def test_str_of_an_integer_parameter(self):
        self._run('def tag(n):\n    return "n" + str(n)\nprint(tag(3))\n', b"n3\n")

    def test_an_f_string_over_parameters_of_every_kind(self):
        self._run(
            'def row(name, n, x, flag):\n    return f"{name}: {n} / {x} / {flag}"\n'
            'print(row("a", 2, 0.25, True))\n',
            b"a: 2 / 0.25 / True\n",
        )

    def test_a_bool_argument_keeps_printing_as_a_bool(self):
        # Nothing tells a bool from an integer at run time, so which one was
        # passed has to be carried across the call.
        self._run(
            'def show(b):\n    return f"{b}"\n'
            "n = 0\nfor i in range(0, 3):\n    n = n + 1\n"
            "print(show(True), show(1), show(n > 2), show(n))\n",
            b"True 1 True 3\n",
        )

    def test_a_parameter_shadows_an_outer_name_of_its_own(self):
        # The outer v is a build-time constant. Inside the function the name
        # means the argument, so the folded value has to be dropped.
        self._run(
            'v = True\ndef show(v):\n    return f"{v}"\nprint(show(7), f"{v}")\n',
            b"7 True\n",
        )

    def test_a_defaulted_parameter_can_be_rendered(self):
        self._run(
            'def tag(n, k=2):\n    return f"{n}-{k}"\nprint(tag(1), tag(1, 5))\n',
            b"1-2 1-5\n",
        )
        self._run('def tag(k=7):\n    return f"k{k}"\nprint(tag(), tag(9))\n', b"k7 k9\n")

    def test_a_defaulted_bool_stays_a_bool(self):
        # The default is stored as the number it also is, so the boolness has
        # to survive that on its own.
        self._run(
            'def row(n, flag=True):\n    return f"{n} {flag}"\n'
            "print(row(1), row(1, False), row(1, 1))\n",
            b"1 True 1 False 1 1\n",
        )

    def test_a_runtime_argument_and_a_nested_call(self):
        self._run(
            'def inner(n):\n    return f"[{n}]"\n'
            'def outer(n):\n    return "x" + inner(n + 1)\n'
            "k = 0\nfor i in range(0, 4):\n    k = k + 1\n"
            "print(outer(k), outer(k * 2))\n",
            b"x[5] x[9]\n",
        )

    def test_a_body_with_statements_before_the_return(self):
        self._run(
            'def label(n):\n    prefix = "#"\n    return prefix + str(n)\n'
            "print(label(12))\n",
            b"#12\n",
        )

    def test_the_results_can_be_collected_and_joined(self):
        self._run(
            'def tag(n):\n    return f"#{n}"\n'
            'parts = [tag(1), tag(2)]\nprint("|".join(parts))\n',
            b"#1|#2\n",
        )


class ReturningAStringTests(unittest.TestCase):
    """A string returned from a branching body, a loop, or a method.

    The result of an inlined call lives in one slot. A string is the address of
    a block and a number is a number, and nothing at run time tells them apart,
    so the call site has to know which it is holding before the body is
    inlined. It reads the body to find out: parameters get stand-ins of the
    right kind, locals are typed by a walk over the assignments, and every
    return has to agree.
    """

    def _run(self, source: str, expected_stdout: bytes) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
                compile_native(entry, root / f"program-{target}.bin", target, clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run(
                [str(root / "program-darwin-arm64.bin")], capture_output=True
            )
            self.assertEqual(native.stdout, expected_stdout)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.stdout, reference.stdout)

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    _SIGN = 'def sign(n):\n    if n < 0:\n        return "neg"\n    return "pos"\n'

    def test_a_branching_body_can_answer_with_a_string(self):
        self._run(self._SIGN + "print(sign(3), sign(-3))\n", b"pos neg\n")

    def test_a_chain_of_three_branches(self):
        self._run(
            "def size(n):\n"
            '    if n < 10:\n        return "small"\n'
            '    if n < 100:\n        return "medium"\n'
            '    return "large"\n'
            "print(size(5), size(50), size(500))\n",
            b"small medium large\n",
        )

    def test_a_branch_that_renders_its_parameter(self):
        self._run(
            "def sign(n):\n"
            '    if n < 0:\n        return "neg " + str(n)\n'
            '    return "pos " + str(n)\n'
            "print(sign(3), sign(-3))\n",
            b"pos 3 neg -3\n",
        )

    def test_a_local_built_by_a_loop_can_be_returned(self):
        # The local's kind comes from the walk over the assignments; without it
        # the call site would read the block's address as a number.
        self._run(
            'def stars(n):\n    out = ""\n'
            '    for i in range(0, n):\n        out = out + "*"\n    return out\n'
            'print(stars(3), stars(0) == "")\n',
            b"*** True\n",
        )

    def test_the_result_behaves_like_any_other_string(self):
        self._run(
            self._SIGN
            + 's = sign(3)\nprint(s, len(s), s == "pos")\n'
            + 'parts = [sign(1), sign(-1)]\nprint("|".join(parts))\n',
            b"pos 3 True\npos|neg\n",
        )

    def test_a_conditional_expression_over_two_strings(self):
        self._run(
            "n = 0\nfor i in range(0, 3):\n    n = n + 1\n"
            'print("big" if n > 2 else "small")\n',
            b"big\n",
        )

    def test_a_method_can_answer_with_a_string(self):
        self._run(
            "class P:\n"
            "    def __init__(self, x):\n        self.x = x\n"
            '    def label(self):\n        return "x=" + str(self.x)\n'
            "p = P(4)\nq = P(9)\nprint(p.label(), q.label())\n",
            b"x=4 x=9\n",
        )

    def test_a_branching_method_and_one_taking_arguments(self):
        self._run(
            "class P:\n"
            "    def __init__(self, x):\n        self.x = x\n"
            "    def sign(self):\n"
            '        if self.x < 0:\n            return "neg"\n        return "pos"\n'
            '    def tag(self, k):\n        return f"{self.x}-{k}"\n'
            "a = P(3)\nb = P(-3)\nprint(a.sign(), b.sign(), a.tag(9))\n",
            b"pos neg 3-9\n",
        )

    def test_a_method_that_answers_a_number_is_unchanged(self):
        self._run(
            "class P:\n"
            "    def __init__(self, x):\n        self.x = x\n"
            "    def get(self):\n        return self.x * 2\n"
            "p = P(4)\nprint(p.get())\n",
            b"8\n",
        )

    def test_returns_that_disagree_are_refused(self):
        self._reject(
            'def bad(n):\n    if n < 0:\n        return "neg"\n    return 1\n'
            "print(bad(3))\n",
            "one arm of this conditional is a string",
        )

    def test_a_string_result_used_as_a_number_is_refused(self):
        self._reject(self._SIGN + "print(sign(1) + 1)\n", "not in the native string subset")


class SteppedSliceTests(unittest.TestCase):
    """`xs[a:b:step]` on a list, and `s[::-1]` on a string."""

    def _run(self, source: str, expected_stdout: bytes) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
                compile_native(entry, root / f"program-{target}.bin", target, clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run(
                [str(root / "program-darwin-arm64.bin")], capture_output=True
            )
            self.assertEqual(native.stdout, expected_stdout)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.stdout, reference.stdout)

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    def _runtime(self, name: str, text: str) -> str:
        return f'{name} = ""\n{name} = {name} + {text!r}\n'

    def test_a_list_reverses_and_strides(self):
        self._run(
            "xs = [1, 2, 3, 4, 5]\nprint(xs[::-1], xs[::2], xs[1::2])\n",
            b"[5, 4, 3, 2, 1] [1, 3, 5] [2, 4]\n",
        )

    def test_bounds_clamp_the_way_python_clamps_them(self):
        # Going backwards the defaults and the clamps are different: the start
        # is the last index and the stop is just before the first.
        self._run(
            "xs = [0, 1, 2, 3, 4, 5, 6]\n"
            "print(xs[1:6:2], xs[5:0:-2], xs[-2::-1], xs[:-3:-1])\n",
            b"[1, 3, 5] [5, 3, 1] [5, 4, 3, 2, 1, 0] [6, 5]\n",
        )

    def test_a_stride_that_selects_nothing(self):
        self._run(
            "xs = [1, 2, 3]\nprint(xs[::-1][3:], xs[2:0:1], xs[0:2:-1])\n",
            b"[] [] []\n",
        )

    def test_a_step_of_one_still_copies_the_block(self):
        self._run(
            "xs = [1, 2, 3, 4]\nprint(xs[1:3], xs[:], xs[-2:])\n",
            b"[2, 3] [1, 2, 3, 4] [3, 4]\n",
        )

    def test_a_string_reverses_by_code_point(self):
        # Byte by byte would split the é and the emoji into their parts.
        self._run(
            self._runtime("s", "héllo中\U0001f600")
            + "r = s[::-1]\nprint(r)\nprint(r[::-1] == s, len(r) == len(s))\n",
            "\U0001f600中olléh\nTrue True\n".encode("utf-8"),
        )

    def test_reversing_an_empty_and_a_one_character_string(self):
        self._run(
            self._runtime("s", "")
            + self._runtime("t", "x")
            + 'print(s[::-1] == "", t[::-1])\n',
            b"True x\n",
        )

    def test_a_wider_string_step_says_why_it_is_refused(self):
        self._reject(
            self._runtime("s", "abcdef") + "print(s[::2])\n",
            "measured from its lead byte",
        )

    def test_a_step_of_zero_is_refused(self):
        self._reject("xs = [1, 2]\nprint(xs[::0])\n", "non-zero integer constant")


class TruthOfAContainerTests(unittest.TestCase):
    """`if xs:`, `not s`, `while queue:` - true when it is not empty.

    A container's slot holds the address of its block, which is never zero, so
    reading it as a number made every container true. It was refused for that
    reason; the count answers the question properly.
    """

    def _run(self, source: str, expected_stdout: bytes) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
                compile_native(entry, root / f"program-{target}.bin", target, clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run(
                [str(root / "program-darwin-arm64.bin")], capture_output=True
            )
            self.assertEqual(native.stdout, expected_stdout)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.stdout, reference.stdout)

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    def test_an_empty_container_is_false_and_a_full_one_true(self):
        self._run(
            's = ""\ns = s + "abc"\ne = ""\n'
            "xs = [1]\nys: list[int] = []\n"
            "d = {1: 2}\nempty: dict[int, int] = {}\n"
            "print(bool(s), bool(e), bool(xs), bool(ys), bool(d), bool(empty))\n",
            b"True False True False True False\n",
        )

    def test_not_reports_emptiness(self):
        self._run(
            's = ""\ns = s + "abc"\ne = ""\nxs = [1]\nys: list[int] = []\n'
            "print(not s, not e, not xs, not ys, not not xs)\n",
            b"False True False True True\n",
        )

    def test_a_container_drives_if_and_while(self):
        self._run(
            "xs = [1, 2, 3]\n"
            "while xs:\n    xs.pop()\n"
            "if xs:\n    print('wrong')\nelse:\n    print('drained')\n",
            b"drained\n",
        )

    def test_a_runtime_float_is_true_when_it_is_not_zero(self):
        self._run(
            "n = 0.0\nfor i in range(0, 3):\n    n = n + 1.0\n"
            "z = 0.0\nfor i in range(0, 1):\n    z = z + 0.0\n"
            "print(bool(n), bool(z), not z)\n",
            b"True False True\n",
        )

    def test_boolean_operators_combine_truths_not_values(self):
        # `s and n` is a question about each operand's truth. Combining the
        # values instead made this false for n == 2, because 1 & 2 is 0.
        self._run(
            's = ""\ns = s + "a"\nn = 0\nfor i in range(0, 2):\n    n = n + 1\n'
            "xs = [1]\nys: list[int] = []\n"
            "print(bool(s and n), bool(xs and s and n), bool(ys or xs), bool(ys and xs))\n",
            b"True True True False\n",
        )

    def test_the_value_form_of_and_or_is_still_refused(self):
        # `xs and ys` answers with one of the two, and one slot cannot hold
        # either kind. Only the truth question is answered.
        self._reject(
            "xs = [1]\nys = [2]\nzs = xs and ys\nprint(len(zs))\n",
            "needs indexing or len()",
        )

    def test_arithmetic_on_a_container_is_still_refused(self):
        self._reject("d = {1: 1}\nx = d + 1\nprint(x)\n", "needs len()")

    def test_the_escape_helper_from_a_real_application(self):
        # Taken from manim_app's app.py, which is what this was added for: a
        # guard clause on an empty string followed by chained replaces.
        self._run(
            "def escape(text):\n"
            '    if not text:\n        return ""\n'
            '    return text.replace("\\\\", "\\\\\\\\").replace("\'", "\\\\\'")\n'
            "s = ''\ns = s + \"a'b\\\\c\"\n"
            'print(escape(s))\nprint(escape("") == "")\n',
            b"a\\'b\\\\c\nTrue\n",
        )


class AssertRepeatAndEnumerateTests(unittest.TestCase):
    """`assert`, `[v] * n`, `k in d.keys()`, and `enumerate()` over a string."""

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
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
            if expected_exit:
                self.assertEqual(
                    native.stderr.strip().splitlines()[-1],
                    reference.stderr.strip().splitlines()[-1],
                )

    def _reject(self, source: str, needle: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "bad.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "bad.bin", "darwin-arm64")
            self.assertIn(needle, str(caught.exception))

    _N = "n = 0\nfor _i in range(0, 3):\n    n = n + 1\n"

    def test_a_passing_assert_costs_nothing_visible(self):
        self._run(self._N + "assert n == 3\nprint('ok')\n", b"ok\n")

    def test_a_failing_assert_reports_cpythons_message(self):
        self._run(self._N + "assert n == 5\nprint('never')\n", b"", expected_exit=1)
        self._run(
            self._N + "assert n == 5, 'n should be five'\n", b"", expected_exit=1
        )

    def test_an_assert_can_be_caught(self):
        self._run(
            "try:\n" + "".join("    " + line + "\n" for line in self._N.splitlines())
            + "    assert n == 5\nexcept AssertionError:\n    print('caught')\n",
            b"caught\n",
        )

    def test_an_assert_tests_a_container_for_emptiness(self):
        self._run("xs: list[int] = []\nassert not xs\nprint('ok')\n", b"ok\n")

    def test_a_runtime_assert_message_is_refused(self):
        # The message is written into the image, so it cannot be built while
        # the program runs.
        self._reject(
            's = ""\nfor _i in range(0, 1):\n    s = s + "x"\n'
            "assert False, s\n",
            "known at build time",
        )

    def test_a_list_repeats(self):
        self._run(
            "print([0] * 5)\nprint([1, 2] * 3)\n",
            b"[0, 0, 0, 0, 0]\n[1, 2, 1, 2, 1, 2]\n",
        )

    def test_the_count_can_be_a_runtime_value_and_either_side(self):
        self._run(
            self._N + "print([7] * n)\nprint(n * [1])\n",
            b"[7, 7, 7]\n[1, 1, 1]\n",
        )

    def test_a_count_of_zero_or_less_gives_an_empty_list(self):
        self._run(
            "n = 0\nfor _i in range(0, 1):\n    n = n - 2\nprint([9] * 0, [9] * n)\n",
            b"[] []\n",
        )

    def test_a_repeated_list_grows_and_is_indexed_like_any_other(self):
        self._run(
            self._N + "xs = [0] * n\nxs[1] = 5\nxs.append(9)\nprint(xs, len(xs))\n",
            b"[0, 5, 0, 9] 4\n",
        )

    def test_repeating_an_empty_list_is_refused(self):
        # There is nothing in it to read an element kind from.
        self._reject("xs = [] * 3\nprint(xs)\n", "element kind nothing states")

    def test_keys_searches_what_the_dict_searches(self):
        self._run(
            "d = {1: 2, 3: 4}\nprint(1 in d.keys(), 9 in d.keys(), 9 not in d.keys())\n",
            b"True False True\n",
        )

    def test_enumerate_walks_a_string_by_code_point(self):
        self._run(
            's = ""\ns = s + "héllo"\n'
            "for i, ch in enumerate(s):\n    print(i, ch)\n",
            "0 h\n1 é\n2 l\n3 l\n4 o\n".encode("utf-8"),
        )

    def test_enumerate_over_a_string_takes_a_start(self):
        self._run(
            's = ""\ns = s + "ab"\n'
            "for i, ch in enumerate(s, 1):\n    print(i, ch)\n",
            b"1 a\n2 b\n",
        )

    def test_enumerate_over_an_empty_string_runs_zero_times(self):
        self._run(
            's = ""\nfor i, ch in enumerate(s):\n    print(i, ch)\nprint("done")\n',
            b"done\n",
        )


class MoreStringMethodTests(unittest.TestCase):
    """`isalnum`, `isspace`, `islower`, `isupper`, `removeprefix`,
    `removesuffix`."""

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
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
    def _runtime(name: str, text: str) -> str:
        return f'{name} = ""\n{name} = {name} + {text!r}\n'

    def test_isalnum_and_isspace(self):
        self._run(
            self._runtime("a", "abc123")
            + self._runtime("b", " \t\n")
            + self._runtime("c", "ab!")
            + "print(a.isalnum(), b.isspace(), c.isalnum(), a.isspace())\n",
            b"True True False False\n",
        )

    def test_islower_and_isupper_need_a_cased_character(self):
        # "123" is neither: there is nothing cased in it to agree with.
        self._run(
            self._runtime("a", "abc1")
            + self._runtime("b", "ABC1")
            + self._runtime("c", "123")
            + self._runtime("d", "aB")
            + "print(a.islower(), b.isupper())\n"
            + "print(c.islower(), c.isupper())\n"
            + "print(d.islower(), d.isupper())\n",
            b"True True\nFalse False\nFalse False\n",
        )

    def test_every_class_test_is_false_for_the_empty_string(self):
        self._run(
            'e = ""\n'
            "print(e.isalnum(), e.isspace(), e.islower(), e.isupper())\n",
            b"False False False False\n",
        )

    def test_a_non_ascii_receiver_stops_the_program(self):
        # The same documented limit the other class tests have: the Unicode
        # tables are not in the image, so a wrong answer is not offered.
        # This is the one place these tests cannot hold the answer against
        # CPython's, because the answers deliberately differ: CPython says
        # True and the native program stops. Both halves are asserted, so the
        # divergence is recorded rather than assumed.
        #
        # Built in a loop so it is not a build-time constant - a constant one
        # is caught before the program runs, which is a different path.
        source = (
            's = ""\nfor _i in range(0, 1):\n    s = s + "café"\n'
            "print(s.isalnum())\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            binary = root / "program.bin"
            compile_native(entry, binary, "darwin-arm64", clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run([str(binary)], capture_output=True)
            self.assertEqual(native.returncode, 1)
            self.assertEqual(native.stdout, b"")
            self.assertIn(b"limited to ASCII", native.stderr)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(reference.stdout, b"True\n")

    def test_removeprefix_and_removesuffix(self):
        self._run(
            self._runtime("s", "prefix-body")
            + self._runtime("t", "body.txt")
            + 'print(s.removeprefix("prefix-"), s.removeprefix("nope"))\n'
            + 'print(t.removesuffix(".txt"), t.removesuffix(".zip"))\n',
            b"body prefix-body\nbody body.txt\n",
        )

    def test_an_affix_may_be_non_ascii(self):
        # These compare bytes, and a valid UTF-8 sequence cannot start in the
        # middle of another, so no decoding is needed and no guard applies.
        self._run(
            self._runtime("s", "héllo!")
            + 'print(s.removesuffix("!"), s.removeprefix("hé"))\n',
            "héllo llo!\n".encode("utf-8"),
        )

    def test_removing_the_whole_string_or_nothing(self):
        self._run(
            self._runtime("s", "ab")
            + 'print(s.removeprefix("ab") == "", s.removesuffix("ab") == "")\n'
            + 'print(s.removeprefix(""), s.removesuffix(""))\n',
            b"True True\nab ab\n",
        )


class SplitLinesTests(unittest.TestCase):
    """`str.splitlines()` over the universal-newline set."""

    def _run(self, source: str, expected_stdout: bytes) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
                compile_native(entry, root / f"program-{target}.bin", target, clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run(
                [str(root / "program-darwin-arm64.bin")], capture_output=True
            )
            self.assertEqual(native.stdout, expected_stdout)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.stdout, reference.stdout)

    @staticmethod
    def _runtime(text: str) -> str:
        return f's = ""\nfor _i in range(0, 1):\n    s = s + {text!r}\n'

    def test_it_splits_on_newlines(self):
        self._run(
            self._runtime("a\nb\nc") + 'print(len(s.splitlines()), "|".join(s.splitlines()))\n',
            b"3 a|b|c\n",
        )

    def test_a_trailing_break_makes_no_extra_piece(self):
        # This is the whole difference from split("\n"), which makes two.
        self._run(
            self._runtime("a\n") + 'print(len(s.splitlines()), len(s.split("\\n")))\n',
            b"1 2\n",
        )

    def test_a_blank_line_is_an_empty_piece(self):
        self._run(
            self._runtime("a\n\nb") + 'print(len(s.splitlines()), "|".join(s.splitlines()))\n',
            b"3 a||b\n",
        )

    def test_crlf_is_one_break_and_a_lone_cr_is_another(self):
        self._run(
            self._runtime("a\r\nb") + 'print(len(s.splitlines()), "|".join(s.splitlines()))\n',
            b"2 a|b\n",
        )
        self._run(self._runtime("a\rb") + "print(len(s.splitlines()))\n", b"2\n")

    def test_an_empty_string_has_no_lines(self):
        self._run('s = ""\nprint(len(s.splitlines()))\n', b"0\n")
        self._run(self._runtime("\n\n") + "print(len(s.splitlines()))\n", b"2\n")

    def test_the_other_ascii_breaks(self):
        # Vertical tab, form feed and the three file/group/record separators.
        self._run(
            self._runtime("a\vb\fc\x1cd") + "print(len(s.splitlines()))\n", b"4\n"
        )

    def test_the_three_breaks_that_are_not_ascii(self):
        # NEL, LINE SEPARATOR and PARAGRAPH SEPARATOR, matched as the byte
        # sequences they are rather than refused.
        self._run(
            self._runtime("a\u2028b\u2029c\u0085d")
            + 'print(len(s.splitlines()), "|".join(s.splitlines()))\n',
            b"4 a|b|c|d\n",
        )

    def test_the_pieces_keep_their_own_non_ascii_text(self):
        self._run(
            self._runtime("héllo\n中") + 'print("|".join(s.splitlines()))\n',
            "héllo|中\n".encode("utf-8"),
        )


class UncaughtExceptionMessageTests(unittest.TestCase):
    """An exception that reaches no handler keeps the message it was raised with.

    A raise that goes straight out of the program always printed its message,
    because the message was known where the raise was. One that passes through
    a `try` whose handlers do not match went through the dispatch instead, and
    the dispatch only carried an identifier - so "ValueError" was printed where
    CPython printed "ValueError: v".
    """

    def _run(self, source: str, stdout: bytes, stderr_tail: bytes, exit_code: int) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
                compile_native(entry, root / f"program-{target}.bin", target, clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run(
                [str(root / "program-darwin-arm64.bin")], capture_output=True
            )
            self.assertEqual(native.stdout, stdout)
            self.assertEqual(native.stderr.strip(), stderr_tail)
            self.assertEqual(native.returncode, exit_code)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.stdout, reference.stdout)
            self.assertEqual(native.returncode, reference.returncode)
            expected = reference.stderr.strip().splitlines()
            self.assertEqual(
                native.stderr.strip(), expected[-1] if expected else b""
            )

    def test_a_message_survives_a_handler_that_does_not_match(self):
        self._run(
            'print("before")\ntry:\n    raise ValueError("v")\n'
            "except TypeError:\n    pass\n",
            b"before\n",
            b"ValueError: v",
            1,
        )

    def test_it_survives_two_levels_of_them(self):
        self._run(
            "try:\n    try:\n        raise KeyError('inner')\n"
            "    except IndexError:\n        pass\n"
            "except TypeError:\n    pass\n",
            b"",
            b"KeyError: 'inner'",
            1,
        )

    def test_key_error_prints_the_keys_repr(self):
        # KeyError's own __str__ shows the repr, so the quotes are part of the
        # message and the quote character depends on what is inside.
        self._run(
            "try:\n    raise KeyError(\"it's\")\nexcept IndexError:\n    pass\n",
            b"",
            b'KeyError: "it\'s"',
            1,
        )

    def test_a_message_from_inside_a_function(self):
        self._run(
            "def check(n):\n"
            '    if n < 0:\n        raise ValueError("negative")\n'
            "    return n\n"
            "try:\n    print(check(-1))\nexcept TypeError:\n    pass\n",
            b"",
            b"ValueError: negative",
            1,
        )

    def test_an_exception_raised_without_a_message(self):
        self._run(
            "try:\n    raise ValueError\nexcept TypeError:\n    pass\n",
            b"",
            b"ValueError",
            1,
        )

    def test_a_missing_integer_key_is_named(self):
        # KeyError shows the key, which is only known while the program runs,
        # so the message is built there rather than written into the image.
        self._run("d = {1: 2}\nprint(d[5])\n", b"", b"KeyError: 5", 1)
        self._run(
            "d = {1: 2}\nk = 0\nfor _i in range(0, 9):\n    k = k + 1\n"
            "print(d[k])\n",
            b"",
            b"KeyError: 9",
            1,
        )

    def test_a_negative_key_and_one_past_a_rehash(self):
        self._run(
            "d = {1: 2}\nk = 0\nfor _i in range(0, 3):\n    k = k - 1\n"
            "print(d[k])\n",
            b"",
            b"KeyError: -3",
            1,
        )
        self._run(
            "d: dict[int, int] = {}\nfor i in range(0, 40):\n    d[i] = i\n"
            "print(d[999])\n",
            b"",
            b"KeyError: 999",
            1,
        )

    def test_a_deleted_key_reads_as_missing(self):
        self._run("d = {1: 2, 3: 4}\ndel d[1]\nprint(d[1])\n", b"", b"KeyError: 1", 1)

    def test_a_string_key_keeps_the_general_wording(self):
        # Its repr would have to choose a quote character and escape what is
        # inside it, and deciding what is printable needs the Unicode tables
        # that are not in the image. The type and the exit status still match.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text('d = {"a": 1}\nprint(d["z"])\n', encoding="utf-8")
            binary = root / "program.bin"
            compile_native(entry, binary, "darwin-arm64", clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run([str(binary)], capture_output=True)
            self.assertEqual(native.returncode, 1)
            self.assertIn(b"KeyError", native.stderr)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(reference.returncode, 1)
            self.assertIn(b"KeyError: 'z'", reference.stderr)

    def test_a_caught_exception_is_unaffected(self):
        self._run(
            'try:\n    raise ValueError("v")\nexcept ValueError:\n    print("caught")\n',
            b"caught\n",
            b"",
            0,
        )


class PartitionTests(unittest.TestCase):
    """`str.partition()` and `str.rpartition()`, which answer a tuple."""

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in _POSIX_TARGETS:
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
    def _runtime(text: str) -> str:
        return f's = ""\nfor _i in range(0, 1):\n    s = s + {text!r}\n'

    def test_it_cuts_at_the_first_and_the_last_separator(self):
        self._run(
            self._runtime("key=value=more")
            + 'a, b, c = s.partition("=")\nprint(a, b, c)\n'
            + 'd, e, f = s.rpartition("=")\nprint(d, e, f)\n',
            b"key = value=more\nkey=value = more\n",
        )

    def test_an_absent_separator_fills_a_different_end(self):
        # This is the one place the two differ beyond direction: partition
        # puts the whole string first, rpartition puts it last.
        self._run(
            self._runtime("abc")
            + 'a, b, c = s.partition("|")\nprint("[" + a + "][" + b + "][" + c + "]")\n'
            + 'd, e, f = s.rpartition("|")\nprint("[" + d + "][" + e + "][" + f + "]")\n',
            b"[abc][][]\n[][][abc]\n",
        )

    def test_a_separator_at_either_end(self):
        self._run(
            self._runtime("=v")
            + 'a, b, c = s.partition("=")\nprint("[" + a + "][" + b + "][" + c + "]")\n',
            b"[][=][v]\n",
        )
        self._run(
            self._runtime("v=")
            + 'a, b, c = s.partition("=")\nprint("[" + a + "][" + b + "][" + c + "]")\n',
            b"[v][=][]\n",
        )

    def test_the_result_can_be_indexed_instead_of_unpacked(self):
        self._run(
            self._runtime("k=v") + 'print(s.partition("=")[0], s.partition("=")[2])\n',
            b"k v\n",
        )

    def test_a_multi_character_separator(self):
        self._run(
            self._runtime("a::b::c") + 'a, b, c = s.rpartition("::")\nprint(a, b, c)\n',
            b"a::b :: c\n",
        )

    def test_a_separator_that_is_not_ascii(self):
        # The cut is by byte, which is safe because a valid UTF-8 separator
        # can only match at a code-point boundary.
        self._run(
            self._runtime("héllo→wörld") + 'a, b, c = s.partition("→")\nprint(a, c)\n',
            "héllo wörld\n".encode("utf-8"),
        )

    def test_an_empty_separator_raises(self):
        self._run(
            self._runtime("ab") + 'a, b, c = s.partition("")\nprint(a)\n',
            b"",
            expected_exit=1,
        )


class FileAccessTests(unittest.TestCase):
    """Reading and writing files through the open/read/write/close syscalls.

    A file is not an object here - there is nothing to hold one - so the
    surface is narrow: `open(path).read()` answers a string, and a name bound
    by `with open(path, "w") as f` accepts `f.write(...)` inside that block and
    means nothing outside it.
    """

    _POSIX = ("darwin-arm64", "linux-x86_64", "linux-arm64", "darwin-x86_64")

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> str:
        """Compile for every POSIX target, run the host one, and return the
        directory it ran in so a test can look at what it wrote."""

        directory = tempfile.mkdtemp()
        root = Path(directory)
        entry = root / "program.py"
        entry.write_text(source, encoding="utf-8")
        for target in self._POSIX:
            compile_native(entry, root / f"program-{target}.bin", target, clean=True)
        if not _HOST_IS_DARWIN_ARM64:
            return directory
        native = subprocess.run(
            [str(root / "program-darwin-arm64.bin")], capture_output=True, cwd=directory
        )
        self.assertEqual(native.stdout, expected_stdout)
        self.assertEqual(native.returncode, expected_exit)
        return directory

    def _compare(self, source: str, *, files: dict[str, str] | None = None) -> None:
        """Run the compiled program and CPython in two clean directories and
        require the same stdout, exit status, and files left behind."""

        results = []
        for runner in ("native", "cpython"):
            directory = Path(tempfile.mkdtemp())
            entry = directory / "program.py"
            entry.write_text(source, encoding="utf-8")
            for name, text in (files or {}).items():
                (directory / name).write_text(text, encoding="utf-8")
            if runner == "native":
                binary = directory / "program.bin"
                compile_native(entry, binary, "darwin-arm64", clean=True)
                if not _HOST_IS_DARWIN_ARM64:
                    return
                finished = subprocess.run(
                    [str(binary)], capture_output=True, cwd=directory
                )
            else:
                finished = subprocess.run(
                    [sys.executable, str(entry)], capture_output=True, cwd=directory
                )
            written = {
                item.name: item.read_text(encoding="utf-8")
                for item in sorted(directory.iterdir())
                if item.suffix not in {".py", ".bin"}
            }
            results.append((finished.stdout, finished.returncode, written))
        self.assertEqual(results[0], results[1])

    def test_reading_a_whole_file(self):
        self._compare(
            'text = open("in.txt").read()\nprint(len(text))\nprint(text)\n',
            files={"in.txt": "hello\nworld\n"},
        )

    def test_the_with_form_binds_the_file_for_the_block(self):
        self._compare(
            'with open("in.txt") as f:\n    text = f.read()\nprint(text.strip())\n',
            files={"in.txt": "hello\nworld\n"},
        )

    def test_reading_and_splitting_into_lines(self):
        self._compare(
            'text = open("in.txt").read()\n'
            'for line in text.splitlines():\n    print("[" + line + "]")\n',
            files={"in.txt": "a\nb\n"},
        )

    def test_a_file_longer_than_one_read(self):
        # The buffer doubles, so a file past the chunk size exercises the grow
        # path and the copy that comes with it.
        self._compare(
            'text = open("big.txt").read()\nprint(len(text))\n',
            files={"big.txt": "x" * 10000},
        )

    def test_an_empty_file(self):
        self._compare(
            'text = open("empty.txt").read()\nprint(len(text), text == "")\n',
            files={"empty.txt": ""},
        )

    def test_writing_and_reading_back(self):
        self._compare(
            'with open("out.txt", "w") as f:\n    f.write("round trip")\n'
            'print(open("out.txt").read())\n'
        )

    def test_appending_to_what_is_there(self):
        self._compare(
            'with open("out.txt", "w") as f:\n    f.write("a")\n'
            'with open("out.txt", "a") as f:\n    f.write("b")\n'
            'print(open("out.txt").read())\n'
        )

    def test_writing_text_built_while_running(self):
        self._compare(
            'lines = ["x", "y"]\n'
            'with open("out.txt", "w") as f:\n    f.write("\\n".join(lines))\n'
            'print(open("out.txt").read())\n'
        )

    def test_a_missing_file_raises_and_can_be_caught(self):
        self._compare('text = open("nope.txt").read()\nprint(text)\n')
        self._compare(
            'try:\n    text = open("nope.txt").read()\n'
            'except OSError:\n    print("caught")\n'
        )

    def test_the_windows_targets_refuse_it(self):
        # The Windows path would need CreateFile and its handles rather than
        # these syscalls, so it is refused rather than silently left out.
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / "program.py"
            entry.write_text('print(open("in.txt").read())\n', encoding="utf-8")
            for target in ("windows-x86_64", "windows-arm64"):
                with self.assertRaises(ValueError) as caught:
                    compile_native(
                        entry, Path(directory) / "p.exe", target, clean=True
                    )
                self.assertIn("POSIX only", str(caught.exception))


class CommandLineArgumentTests(unittest.TestCase):
    """`sys.argv` - the count and the strings the process was started with.

    Where they arrive differs by kernel and not by executable format, which is
    the thing that had to be measured rather than assumed: macOS hands them to
    the entry point in registers for a static image as well as for a dynamic
    one, and Linux leaves them on the stack.
    """

    _POSIX = ("darwin-arm64", "linux-x86_64", "linux-arm64", "darwin-x86_64")

    def _run(self, source: str, arguments: list[str], expected: bytes,
             expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            for target in self._POSIX:
                compile_native(entry, root / f"p-{target}.bin", target, clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            binary = root / "p-darwin-arm64.bin"
            native = subprocess.run(
                [str(binary), *arguments], capture_output=True, cwd=directory
            )
            self.assertEqual(native.stdout, expected)
            self.assertEqual(native.returncode, expected_exit)
            reference = subprocess.run(
                [sys.executable, str(entry), *arguments],
                capture_output=True,
                cwd=directory,
            )
            self.assertEqual(native.stdout, reference.stdout)
            self.assertEqual(native.returncode, reference.returncode)

    def test_the_count_includes_the_program_itself(self):
        self._run("import sys\nprint(len(sys.argv))\n", [], b"1\n")
        self._run("import sys\nprint(len(sys.argv))\n", ["a", "b", "c"], b"4\n")

    def test_the_arguments_are_ordinary_strings(self):
        self._run(
            "import sys\nname = sys.argv[1]\n"
            'print(name.upper(), name + "!", len(name))\n',
            ["abc"],
            b"ABC abc! 3\n",
        )

    def test_walking_them_with_a_loop(self):
        self._run(
            "import sys\n"
            "for i in range(1, len(sys.argv)):\n    print(i, sys.argv[i])\n",
            ["a", "bb", "ccc"],
            b"1 a\n2 bb\n3 ccc\n",
        )

    def test_an_argument_may_hold_spaces_or_non_ascii(self):
        self._run(
            'import sys\nprint("[" + sys.argv[1] + "]")\n',
            ["two words"],
            b"[two words]\n",
        )
        self._run(
            "import sys\nprint(sys.argv[1], len(sys.argv[1]))\n",
            ["héllo"],
            "héllo 5\n".encode("utf-8"),
        )

    def test_a_negative_index_counts_from_the_end(self):
        self._run("import sys\nprint(sys.argv[-1])\n", ["x", "y", "last"], b"last\n")

    def test_an_index_past_the_end_raises(self):
        self._run("import sys\nprint(sys.argv[9])\n", ["only"], b"", expected_exit=1)

    def test_it_reaches_through_the_dynamic_mach_o_writer_too(self):
        # A program with an extern call is written as a dynamic image that dyld
        # enters like a C main. That is the second of the two Mach-O entry
        # paths, and it has to answer the same.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(
                "import sys\nfrom py2bin.cabi import getpid\n"
                "print(len(sys.argv), sys.argv[1])\nprint(getpid() > 0)\n",
                encoding="utf-8",
            )
            binary = root / "program.bin"
            compile_native(entry, binary, "darwin-arm64", clean=True)
            self.assertIn(b"__LINKEDIT", binary.read_bytes())
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run([str(binary), "hello"], capture_output=True)
            self.assertEqual(native.stdout, b"2 hello\nTrue\n")

    def test_the_windows_targets_refuse_it(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / "program.py"
            entry.write_text("import sys\nprint(len(sys.argv))\n", encoding="utf-8")
            for target in ("windows-x86_64", "windows-arm64"):
                with self.assertRaises(ValueError) as caught:
                    compile_native(entry, Path(directory) / "p.exe", target, clean=True)
                self.assertIn("POSIX only", str(caught.exception))

    def test_a_tool_that_reads_the_files_it_is_given(self):
        # argv and file access together, which is the point of both.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("one two\nthree\n", encoding="utf-8")
            entry = root / "program.py"
            entry.write_text(
                "import sys\n"
                "for i in range(1, len(sys.argv)):\n"
                "    text = open(sys.argv[i]).read()\n"
                "    words = 0\n"
                "    for line in text.splitlines():\n"
                "        words += len(line.split())\n"
                "    print(sys.argv[i], words)\n",
                encoding="utf-8",
            )
            binary = root / "program.bin"
            compile_native(entry, binary, "darwin-arm64", clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run(
                [str(binary), "a.txt"], capture_output=True, cwd=directory
            )
            self.assertEqual(native.stdout, b"a.txt 3\n")
            reference = subprocess.run(
                [sys.executable, str(entry), "a.txt"],
                capture_output=True,
                cwd=directory,
            )
            self.assertEqual(native.stdout, reference.stdout)
