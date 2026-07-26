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

    def test_del_on_a_dict_is_rejected_by_name(self):
        self._reject(
            "d = {}\nd[1] = 2\ndel d[1]\nprint(len(d))\n",
            "the open-addressing table has no tombstone state",
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

    def test_printing_a_list_is_still_rejected(self):
        self._reject("xs = [[1, 2]]\nprint(xs)\n", "cannot render a runtime")
        self._reject("xs = [[1, 2]]\nprint(xs[0])\n", "cannot render a runtime")
        self._reject('parts = ["a"]\nprint(parts[0:1])\n', "cannot render a runtime")

    def test_reordering_and_reducing_strings_or_lists_is_rejected(self):
        self._reject('parts = ["b", "a"]\nparts.sort()\n', "compare block addresses")
        self._reject('parts = ["b", "a"]\nys = sorted(parts)\n', "compare block addresses")
        self._reject('parts = ["a"]\nprint(sum(parts))\n', "compare block addresses")
        self._reject("xs = [[1], [2]]\nxs.sort()\n", "compare block addresses")
        self._reject("xs = [[1], [2]]\nprint([1] in xs)\n", "over a list of lists")


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
