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

_POSIX_TARGETS = (
    "linux-x86_64",
    "linux-arm64",
    "darwin-x86_64",
    "darwin-arm64",
)

# A tuple that a name holds lives in the arena, so a runtime value forces the
# same value through the same block on every target.
_RUNTIME = "n = 0\nfor i in range(0, 3):\n    n += 1\n"


class NativeTupleTests(unittest.TestCase):
    """Fixed-length tuples with a kind per element, lowered to machine code and
    verified against CPython on darwin-arm64."""

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
                    f"{target} tuple image has a broken header",
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

    # --- shape and per-element kinds ----------------------------------------

    def test_one_tuple_holds_three_different_kinds(self):
        # This is the whole reason a tuple is not a list: the elements do not
        # have to agree, because every index that reads one is a constant.
        self._run(
            _RUNTIME
            + 't = (n, 2.5, "x")\n'
            "print(t[0])\nprint(t[1])\nprint(t[2])\nprint(len(t))\n",
            b"3\n2.5\nx\n3\n",
        )

    def test_a_bool_element_keeps_its_identity_beside_a_number(self):
        # `[True, 1]` is refused because one list is one kind; a tuple keeps
        # them apart, and t[0] must still print True rather than 1.
        self._run(
            _RUNTIME + "t = (n > 2, n)\nprint(t[0], t[1])\nprint(t)\n",
            b"True 3\n(True, 3)\n",
        )

    def test_a_whole_tuple_prints_the_repr_of_every_element(self):
        self._run(
            _RUNTIME + 't = (n, 2.5, "x", n > 2)\nprint(t)\n',
            b"(3, 2.5, 'x', True)\n",
        )

    def test_the_one_element_and_empty_forms(self):
        # `(1,)` is a tuple and `(1)` is not, and repr keeps the comma that
        # says which.
        self._run(
            _RUNTIME
            + "one = (n,)\nbare = n,\nempty = ()\nplain = (n)\n"
            "print(one)\nprint(bare)\nprint(empty)\nprint(plain)\n"
            "print(len(one), len(empty))\n",
            b"(3,)\n(3,)\n()\n3\n1 0\n",
        )

    def test_a_string_element_reprs_with_the_quotes_cpython_picks(self):
        self._run(
            _RUNTIME + "t = (n, \"a'b\", 'c\"d')\nprint(t)\n",
            b"(3, \"a'b\", 'c\"d')\n",
        )

    # --- indexing -----------------------------------------------------------

    def test_a_runtime_index_walks_a_tuple_of_one_kind(self):
        self._run(
            _RUNTIME + "h = (10, 20, 30)\nprint(h[n - 1])\nprint(h[-n])\n",
            b"30\n10\n",
        )

    def test_a_runtime_index_out_of_range_raises_a_catchable_index_error(self):
        self._run(
            _RUNTIME
            + "h = (10, 20, 30)\n"
            "try:\n"
            "    print(h[n])\n"
            "except IndexError:\n"
            '    print("caught")\n',
            b"caught\n",
        )

    def test_an_uncaught_runtime_index_error_exits_one(self):
        self._run(_RUNTIME + "h = (10, 20, 30)\nprint(h[n])\n", b"", 1)

    # --- unpacking and iteration --------------------------------------------

    def test_unpacking_gives_each_name_its_own_kind(self):
        self._run(
            _RUNTIME + 't = (n, 1.5, "hi", n > 2)\na, b, c, d = t\n'
            "print(a, b, c, d, len(c))\n",
            b"3 1.5 hi True 2\n",
        )

    def test_iteration_over_a_tuple_of_one_kind(self):
        self._run(
            _RUNTIME + "h = (n, n * 2, n * 3)\nfor v in h:\n    print(v)\n",
            b"3\n6\n9\n",
        )

    def test_iterating_an_empty_tuple_leaves_the_name_unbound(self):
        self._run(
            "t = ()\nfor v in t:\n    print(v)\nprint(len(t))\n", b"0\n"
        )

    def test_iteration_carries_a_bool_through(self):
        self._run(
            _RUNTIME
            + "flags = (n > 2, n > 5)\nfor f in flags:\n    print(f)\n",
            b"True\nFalse\n",
        )

    # --- aliasing -----------------------------------------------------------

    def test_a_second_name_for_a_tuple_is_the_same_block(self):
        # A list refuses this because appending moves the block; a tuple never
        # moves, so both names stay right.
        self._run(
            _RUNTIME + "h = (n, n + 1)\nu = h\nprint(u[0], h[0])\nprint(u)\n",
            b"3 3\n(3, 4)\n",
        )

    def test_a_literal_that_reads_the_name_it_replaces(self):
        # The new block is built before the name is moved onto it, so the
        # right-hand side still reads the old one.
        self._run(
            _RUNTIME
            + "t = (n, n)\n"
            "while t[0] > 0:\n"
            "    print(t[0])\n"
            "    t = (t[0] - 1, t[1])\n",
            b"3\n2\n1\n",
        )

    # --- honest rejections --------------------------------------------------

    def test_a_runtime_index_into_a_mixed_tuple_is_rejected(self):
        self._reject(
            _RUNTIME + "t = (1, 2.5)\nprint(t[n])\n",
            "needs a constant index",
        )

    def test_a_constant_index_out_of_range_is_a_build_error(self):
        self._reject(
            "t = (1, 2, 3)\nprint(t[3])\n",
            "native tuple index 3 is out of range",
        )

    def test_comparing_two_whole_tuples_is_rejected(self):
        # The fall-through would compare block addresses and answer False
        # where CPython answers True.
        self._reject(
            "t = (1, 2)\nu = (1, 2)\nprint(t == u)\n",
            "cannot be compared",
        )

    def test_printing_a_tuple_with_a_runtime_string_is_rejected(self):
        self._reject(
            'parts = ["a", "b"]\ns = parts[0]\nt = (1, s)\nprint(t)\n',
            "string elements are known at build time",
        )

    def test_a_tuple_element_cannot_be_assigned_or_deleted(self):
        self._reject("t = (1, 2)\nt[0] = 5\n", "a native tuple is immutable")
        self._reject("t = (1, 2)\ndel t[0]\n", "del on a native tuple")

    def test_a_tuple_cannot_be_a_dict_key(self):
        self._reject(
            "d = {}\nd[(1, 2)] = 3\n",
            "a native dict key is a signed 64-bit integer or a runtime string",
        )

    def test_a_native_function_cannot_return_a_tuple(self):
        self._reject(
            "def f():\n    return (1, 2)\nprint(f()[0])\n",
            "expression is not a native runtime tuple",
        )

    def test_slicing_and_membership_are_rejected(self):
        self._reject("t = (1, 2, 3)\nprint(t[0:1])\n", "tuple slicing")
        self._reject("t = (1, 2)\nprint(1 in t)\n", "`in` over a native tuple")

    def test_a_nested_container_cannot_be_an_element(self):
        self._reject("t = ((1, 2), 3)\n", "a native tuple element is")
        self._reject("t = ([1], 2)\n", "a native tuple element is")

    def test_iterating_a_mixed_tuple_is_rejected(self):
        self._reject(
            't = (1, "a")\nfor v in t:\n    print(v)\n',
            "needs every element to be the same kind",
        )

    def test_changing_a_tuple_variable_s_shape_is_rejected(self):
        self._reject(
            "t = (1, 2)\nt = (3, 4, 5)\n",
            "cannot change type between tuple:int,int and tuple:int,int,int",
        )

    def test_unpacking_needs_matching_lengths(self):
        self._reject(
            "t = (1, 2, 3)\na, b = t\n", "2 names, 3 values"
        )

    def test_a_runtime_tuple_index_is_rejected_in_an_eager_arm(self):
        # Its bounds check would run on a branch CPython never evaluates.
        self._reject(
            _RUNTIME + "h = (10, 20, 30)\nx = h[n] if n > 5 else 0\n",
            "cannot appear in a conditional expression",
        )


if __name__ == "__main__":
    unittest.main()
