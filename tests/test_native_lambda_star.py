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


class NativeLambdaAndStarredCallTests(unittest.TestCase):
    """A lambda bound to a name is the one-expression function it stands for,
    and `f(*xs)` is spelled out against a parameter count known at build time.
    Both are checked by running a real binary and diffing it against CPython.
    """

    def _run(self, source: str, expected_stdout: bytes, expected_exit: int = 0) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "program.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not _HOST_IS_DARWIN_ARM64:
                self.skipTest("running the image needs a darwin-arm64 host")
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.stdout, expected_stdout)
            self.assertEqual(native.returncode, expected_exit)
            # The expectation is only worth anything if CPython agrees with it.
            reference = subprocess.run([sys.executable, str(entry)], capture_output=True)
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

    # --- lambdas ------------------------------------------------------------

    def test_lambda_parameters_defaults_and_no_parameters(self):
        self._run(
            "f = lambda x: x + 1\n"
            "g = lambda a, b=10: a * b\n"
            "z = lambda: 7\n"
            "n = 0\n"
            "for i in range(0, 3):\n"
            "    n += 1\n"
            "print(f(n), g(n), g(n, 4), z())\n",
            b"4 30 12 7\n",
        )

    def test_lambda_returning_a_bool_prints_true_not_one(self):
        # The lambda goes through the same return-kind machinery as a def, so
        # the bool has to survive the call rather than decay to an integer.
        self._run(
            "h = lambda x: x > 2\n"
            "n = 0\n"
            "for i in range(0, 3):\n"
            "    n += 1\n"
            "print(h(n + 2), h(n - 2))\n",
            b"True False\n",
        )

    def test_lambda_returning_a_float_and_a_string(self):
        self._run(
            "q = lambda x: x / 2.0\n"
            "name = lambda s: s + '!'\n"
            "n = 0\n"
            "for i in range(0, 3):\n"
            "    n += 1\n"
            "s = 'hi'\n"
            "if n == 99:\n"
            "    s = 'other'\n"
            "print(q(n + 2))\n"
            "print(name(s))\n",
            b"2.5\nhi!\n",
        )

    def test_lambda_reads_a_free_variable_at_the_time_of_the_call(self):
        # CPython looks the free name up when the lambda runs, not when it is
        # written, so the second call must see the newer value.
        self._run(
            "n = 0\n"
            "for i in range(0, 3):\n"
            "    n += 1\n"
            "r = lambda x: x + n\n"
            "print(r(10))\n"
            "n = 100\n"
            "print(r(10))\n",
            b"13\n110\n",
        )

    def test_lambda_inside_a_function_body(self):
        self._run(
            "def outer(v):\n"
            "    g = lambda x: x * 2\n"
            "    return g(v) + 1\n"
            "n = 0\n"
            "for i in range(0, 3):\n"
            "    n += 1\n"
            "print(outer(n))\n",
            b"7\n",
        )

    def test_lambda_name_defined_in_a_function_does_not_leak_outward(self):
        self._reject(
            "def outer():\n"
            "    g = lambda x: x * 2\n"
            "    return g(3)\n"
            "print(outer())\n"
            "print(g(3))\n",
            "not in the signed 64-bit native integer subset",
        )

    def test_unbound_lambda_positions_are_rejected(self):
        needle = "the only lambda with a representation is"
        self._reject("k = [lambda x: x]\nprint(1)\n", needle)
        self._reject("def m():\n    return lambda x: x\nprint(1)\n", needle)
        self._reject("print((lambda x: x)(1))\n", needle)
        self._reject("def take(h):\n    return 1\nprint(take(lambda x: x))\n", needle)
        # Bound, but inside a block that may not run, so which code the name
        # means would depend on what happened.
        self._reject(
            "c = 0\n"
            "for i in range(0, 1):\n"
            "    c += 1\n"
            "if c == 1:\n"
            "    f = lambda x: x + 1\n"
            "print(f(1))\n",
            needle,
        )

    def test_rebinding_a_lambda_name_is_rejected(self):
        needle = "bound again on line"
        self._reject("f = lambda x: x\nf = lambda x: x + 1\nprint(f(1))\n", needle)
        self._reject("f = lambda x: x\ndef f(x):\n    return x + 2\nprint(f(1))\n", needle)
        self._reject("f = lambda x: x + 1\nfor f in range(0, 2):\n    pass\nprint(f)\n", needle)

    def test_reading_a_lambda_name_as_a_value_is_rejected(self):
        needle = "so it can only be called"
        self._reject("f = lambda x: x\ng = f\nprint(g(1))\n", needle)
        self._reject("f = lambda x: x + 1\nprint(f)\n", needle)

    def test_a_lambda_keyword_argument_keeps_the_callee_s_own_message(self):
        # sort() knows better than the general lambda rule why it cannot take
        # a callable, so its message has to win.
        self._reject(
            "xs = [3, 1]\nxs.sort(key=lambda v: v)\nprint(xs[0])\n",
            "native sorting does not support key=",
        )

    # --- starred calls ------------------------------------------------------

    def test_star_of_a_literal_a_name_and_a_tuple(self):
        self._run(
            "def add(a, b):\n"
            "    return a + b\n"
            "n = 0\n"
            "for i in range(0, 3):\n"
            "    n += 1\n"
            "xs = [4, n]\n"
            "t = (6, n)\n"
            "print(add(*[1, n]))\n"
            "print(add(*xs))\n"
            "print(add(*t))\n",
            b"4\n7\n9\n",
        )

    def test_star_mixed_with_written_out_arguments_and_a_second_star(self):
        self._run(
            "def three(a, b, c):\n"
            "    return a * 100 + b * 10 + c\n"
            "n = 0\n"
            "for i in range(0, 3):\n"
            "    n += 1\n"
            "p = [2, n]\n"
            "print(three(1, *p))\n"
            "print(three(*[1], *p))\n",
            b"123\n123\n",
        )

    def test_a_bool_keeps_its_identity_through_a_starred_call(self):
        self._run(
            "def first(a, b):\n"
            "    return a\n"
            "n = 0\n"
            "for i in range(0, 3):\n"
            "    n += 1\n"
            "bs = [n > 1, n > 5]\n"
            "print(first(*bs))\n",
            b"True\n",
        )

    def test_floats_and_strings_survive_a_starred_call(self):
        self._run(
            "def total(a, b):\n"
            "    return a + b\n"
            "n = 0\n"
            "for i in range(0, 3):\n"
            "    n += 1\n"
            "fs = [1.5, n / 2.0]\n"
            "ss = ['ab', 'cd']\n"
            "if n == 99:\n"
            "    print(1)\n"
            "print(total(*fs))\n"
            "print(total(*ss))\n",
            b"3.0\nabcd\n",
        )

    def test_star_reaches_methods_constructors_and_procedures(self):
        self._run(
            "class C:\n"
            "    def __init__(self, a, b):\n"
            "        self.a = a\n"
            "        self.b = b\n"
            "    def add(self, x, y):\n"
            "        return x + y\n"
            "def show(a, b):\n"
            "    print(a + b)\n"
            "xs = [1, 2]\n"
            "o = C(*xs)\n"
            "print(o.a + o.b)\n"
            "print(o.add(*xs))\n"
            "show(*xs)\n",
            b"3\n3\n3\n",
        )

    def test_a_length_that_is_not_a_build_time_fact_is_rejected(self):
        needle = "needs a known number of arguments"
        head = "def add(a, b):\n    return a + b\n"
        # Appended to, so the length stopped being known.
        self._reject(head + "xs = [1, 2]\nxs.append(3)\nprint(add(*xs))\n", needle)
        self._reject(head + "xs = [1, 2, 3]\ndel xs[0]\nprint(add(*xs))\n", needle)
        # Reassigned inside a block that may not run.
        self._reject(
            head + "xs = [1, 2]\nfor i in range(0, 2):\n    xs = [1, 2, 3]\nprint(add(*xs))\n",
            needle,
        )
        # Never had a known length here at all.
        length = "* accepts a list or tuple literal"
        self._reject(head + "xs = [1, 2]\nprint(add(*xs[:]))\n", length)
        self._reject(head + "print(add(*range(0, 2)))\n", length)
        self._reject(head + "s = 'ab'\nprint(add(*s))\n", length)
        self._reject(head + "d = {1: 2, 3: 4}\nprint(add(*d))\n", length)
        self._reject(
            head + "def outer(ys):\n    return add(*ys)\nprint(outer([1, 2]))\n", length
        )

    def test_star_arity_must_match_the_parameter_count(self):
        head = "def add(a, b):\n    return a + b\n"
        self._reject(
            head + "xs = [1, 2, 3]\nprint(add(*xs))\n",
            "accepts at most 2 positional arguments",
        )
        self._reject(head + "print(add(*[1]))\n", "missing required argument")
        # A default covers the parameter the star did not reach.
        self._run(
            "def add(a, b=5):\n    return a + b\nprint(add(*[1]))\n", b"6\n"
        )

    def test_star_beside_a_call_in_the_same_argument_list_is_rejected(self):
        # CPython unpacks the list after that call has run, so a call that
        # appended to it would change how many arguments the star stands for.
        self._reject(
            "def add(a, b, c):\n"
            "    return a + b + c\n"
            "def side():\n"
            "    return 1\n"
            "xs = [1, 2]\n"
            "print(add(side(), *xs))\n",
            "cannot share its argument list with",
        )

    def test_star_to_a_callee_that_is_not_defined_here_is_rejected(self):
        self._reject("xs = [1, 2]\nprint(*xs)\n", "take their arguments written out")
        self._reject(
            "xs = [-7]\nprint(abs(*xs))\n", "only when calling a function, lambda"
        )

    def test_mapping_expansion_is_rejected(self):
        self._reject(
            "def add(a, b):\n    return a + b\nd = {'a': 1, 'b': 2}\nprint(add(**d))\n",
            "do not support ** mapping expansion",
        )

    # --- the two features crossed -------------------------------------------

    def test_a_starred_call_to_a_lambda_keeps_bool_and_float_kinds(self):
        self._run(
            "pick = lambda a, b: a\n"
            "tot = lambda a, b: a + b\n"
            "n = 0\n"
            "for i in range(0, 3):\n"
            "    n += 1\n"
            "bs = [n > 1, n > 5]\n"
            "fs = [1.5, n / 2.0]\n"
            "print(pick(*bs))\n"
            "print(tot(*fs))\n",
            b"True\n3.0\n",
        )

    # --- the length a branch invalidated ------------------------------------

    def test_a_list_reassigned_in_a_branch_loses_its_build_time_length(self):
        # The longer list is only assigned on a path that does not run, so the
        # index is out of range at run time exactly as CPython finds it.
        self._run(
            "c = 0\n"
            "for i in range(0, 1):\n"
            "    c += 1\n"
            "ys = [1, 2, 3]\n"
            "if c == 0:\n"
            "    ys = [7, 8, 9, 10]\n"
            "print(ys[3])\n",
            b"",
            expected_exit=1,
        )


if __name__ == "__main__":
    unittest.main()
