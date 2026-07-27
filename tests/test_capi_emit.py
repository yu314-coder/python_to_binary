"""Python translated into C that drives the CPython C API.

The tier Nuitka occupies, reached without Nuitka's toolchain: py2bin writes
the C and py2bin's own C compiler turns it into machine code, so no clang,
assembler or linker takes part. What it buys over the native tier is Python's
own semantics - most visibly integers that do not stop at 64 bits.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from py2bin.capi_emit import CApiEmitError, python_to_capi_c
from py2bin.c_native import compile_c_native

_HOST_IS_DARWIN_ARM64 = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)


class CApiEmitTests(unittest.TestCase):
    """Translation, then compilation, then the answer held against CPython."""

    def _run(self, source: str, expected: bytes) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            generated = root / "program.c"
            generated.write_text(
                python_to_capi_c(source, str(entry)), encoding="utf-8"
            )
            # The C is always produced, on every host: it is the part this
            # module is responsible for. Compiling and running it needs the
            # one platform whose C-API binding is wired up.
            self.assertIn("PyLong_FromLongLong", generated.read_text())
            if not _HOST_IS_DARWIN_ARM64:
                return
            binary = root / "program.bin"
            compile_c_native(generated, binary, target="darwin-arm64", clean=True)
            native = subprocess.run([str(binary)], capture_output=True)
            self.assertEqual(native.stdout, expected)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.stdout, reference.stdout)

    def _reject(self, source: str, needle: str) -> None:
        with self.assertRaises(CApiEmitError) as caught:
            python_to_capi_c(source, "program.py")
        self.assertIn(needle, str(caught.exception))

    def test_arithmetic_goes_through_the_interpreter(self):
        self._run("print(2 + 3 * 4 - 1)\n", b"13\n")

    def test_integers_do_not_stop_at_sixty_four_bits(self):
        # The whole reason this tier exists. PyNumber_Multiply on two PyLongs
        # is the same arbitrary-precision multiply the interpreter performs,
        # where the native tier wraps and answers 0.
        self._run(
            "x = 1\ni = 0\nwhile i < 100:\n    x = x * 2\n    i = i + 1\nprint(x)\n",
            b"1267650600228229401496703205376\n",
        )

    def test_recursion_and_a_factorial_that_overflows_a_register(self):
        self._run(
            "def fact(n):\n"
            "    if n < 2:\n        return 1\n"
            "    return n * fact(n - 1)\n"
            "print(fact(25))\n",
            b"15511210043330985984000000\n",
        )

    def test_strings_concatenate_and_str_converts(self):
        self._run(
            'a = "hello"\nb = "world"\nprint(a + " " + b)\n', b"hello world\n"
        )
        self._run('n = 12345\nprint("n=" + str(n))\n', b"n=12345\n")

    def test_text_outside_ascii_survives_the_round_trip(self):
        # The literal is emitted as octal escapes so the C stays ASCII, and the
        # embedded interpreter's stdout is set to UTF-8 - without that a print
        # of this stops with a UnicodeEncodeError about the encoding rather
        # than anything to do with the program.
        self._run('print("héllo 中")\n', "héllo 中\n".encode("utf-8"))

    def test_true_division_answers_a_float(self):
        self._run("print(7 / 2)\n", b"3.5\n")

    def test_a_condition_and_a_loop(self):
        self._run(
            "x = 10\nif x > 5:\n    print('big')\nelse:\n    print('small')\n",
            b"big\n",
        )
        self._run(
            "total = 0\ni = 0\nwhile i < 100:\n    total = total + i\n"
            "    i = i + 1\nprint(total)\n",
            b"4950\n",
        )

    def test_several_values_on_one_print(self):
        self._run('print(1, "two", 3)\n', b"1 two 3\n")

    def test_a_long_loop_does_not_leak(self):
        # Every expression yields an owned reference and every statement
        # releases what it finishes with. If that rule were broken this would
        # keep one PyLong per iteration and the run would grow without bound.
        self._run(
            "i = 0\ntotal = 0\nwhile i < 200000:\n    total = total + i\n"
            "    i = i + 1\nprint(total)\n",
            b"19999900000\n",
        )

    def test_a_compiled_program_can_import_and_use_a_module(self):
        # This is what the tier is for. The interpreter is present and its
        # import machinery works, so the compiled program reaches anything
        # installed beside it - including modules that are themselves C
        # extensions, which is what `math` is.
        self._run("import math\nprint(math.sqrt(2))\n", b"1.4142135623730951\n")
        self._run(
            "import math\nprint(math.factorial(30))\n",
            b"265252859812191058636308480000000\n",
        )

    def test_modules_that_are_mostly_python(self):
        self._run("import json\nprint(json.dumps(42))\n", b"42\n")
        self._run('import re\nprint(re.escape("a.b"))\n', b"a\\.b\n")

    def test_methods_on_ordinary_objects(self):
        self._run(
            's = "Hello World"\nprint(s.upper())\nprint(s.lower())\n',
            b"HELLO WORLD\nhello world\n",
        )

    def test_len_goes_through_the_object_protocol(self):
        self._run('print(len("hello"))\n', b"5\n")

    def test_a_call_with_several_arguments(self):
        # Nought and one arguments have their own entry points; beyond that the
        # arguments go into a tuple. PyTuple_SetItem steals the reference it is
        # handed, so nothing is released after it.
        self._run("import math\nprint(math.pow(2, 10))\n", b"1024.0\n")
        self._run('s = "a-b-c"\nprint(s.replace("-", "+"))\n', b"a+b+c\n")
        self._run(
            "import math\nprint(math.gcd(math.factorial(10), 48))\n", b"48\n"
        )

    def test_a_loop_calling_into_a_module_does_not_leak(self):
        self._run(
            "import math\ni = 0\ntotal = 0\n"
            "while i < 50000:\n    total = total + math.gcd(i, 12)\n    i = i + 1\n"
            "print(total)\n",
            b"166670\n",
        )

    def _run_failing(self, source: str, needle: bytes) -> None:
        """A program that raises: the message and the status must both match."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            generated = root / "program.c"
            generated.write_text(
                python_to_capi_c(source, str(entry)), encoding="utf-8", newline="\n"
            )
            if not _HOST_IS_DARWIN_ARM64:
                return
            binary = root / "program.bin"
            compile_c_native(generated, binary, target="darwin-arm64", clean=True)
            native = subprocess.run([str(binary)], capture_output=True)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.returncode, 1)
            self.assertEqual(native.returncode, reference.returncode)
            self.assertIn(needle, native.stderr)
            self.assertIn(needle, reference.stderr)

    def test_a_failing_call_stops_with_the_interpreters_own_message(self):
        # A C-API function answers NULL and leaves an exception set. Letting
        # that NULL travel is how `1 + "x"` came to print `<NULL>` and exit 0
        # where CPython raises. Every C-API result is checked now, and an
        # uncaught exception leaves with status 1, which is what CPython does.
        self._run_failing('print(1 + "x")\n', b"TypeError")
        self._run_failing("print(1 / 0)\n", b"ZeroDivisionError")

    def test_a_failing_import_stops_rather_than_carrying_on(self):
        self._run_failing(
            "import nonexistent_module_xyz\nprint(1)\n", b"ModuleNotFoundError"
        )

    def test_a_missing_attribute_and_a_bad_argument(self):
        self._run_failing("import math\nprint(math.no_such_thing)\n", b"AttributeError")
        self._run_failing('import math\nprint(math.sqrt("x"))\n', b"TypeError")

    def test_for_loops_use_the_iterator_protocol(self):
        # Whatever the object offers, because the interpreter is the one being
        # asked - unlike the native tier, which has to know the shape of every
        # iterable it supports.
        self._run("for i in range(5):\n    print(i)\n", b"0\n1\n2\n3\n4\n")
        self._run('for ch in "abc":\n    print(ch)\n', b"a\nb\nc\n")
        self._run("for x in [10, 20]:\n    print(x)\n", b"10\n20\n")

    def test_nested_loops_and_accumulation(self):
        self._run(
            "total = 0\nfor i in range(1000):\n    total = total + i\nprint(total)\n",
            b"499500\n",
        )
        self._run(
            "for i in range(2):\n    for j in range(2):\n"
            "        print(i * 10 + j)\n",
            b"0\n1\n10\n11\n",
        )

    def test_builtins_come_from_the_interpreter(self):
        # Nothing here reimplements range, sum or sorted. The builtins module
        # is imported once and these are read off it.
        self._run("print(sum(range(10)))\n", b"45\n")
        self._run('print(sorted("banana"))\n', b"['a', 'a', 'a', 'b', 'n', 'n']\n")
        self._run("xs = list(range(5))\nprint(xs, len(xs))\n", b"[0, 1, 2, 3, 4] 5\n")

    def test_list_literals_and_append(self):
        # PyList_Append does not steal its reference, unlike PyTuple_SetItem,
        # so each element is released after it goes in.
        self._run("xs = [1, 2, 3]\nprint(xs, len(xs))\n", b"[1, 2, 3] 3\n")
        self._run(
            "xs = []\nfor i in range(4):\n    xs.append(i * i)\nprint(xs)\n",
            b"[0, 1, 4, 9]\n",
        )

    def test_break_continue_and_augmented_assignment(self):
        self._run(
            "for i in range(10):\n    if i > 2:\n        break\n    print(i)\n",
            b"0\n1\n2\n",
        )
        self._run(
            "for i in range(5):\n    if i == 2:\n        continue\n    print(i)\n",
            b"0\n1\n3\n4\n",
        )
        self._run("x = 0\nfor i in range(5):\n    x += i\nprint(x)\n", b"10\n")

    def test_the_remaining_arithmetic(self):
        self._run("print(17 % 5, 17 // 5)\n", b"2 3\n")
        self._run("print(2 ** 100)\n", b"1267650600228229401496703205376\n")

    def test_dict_and_tuple_literals(self):
        # PyDict_SetItem steals neither reference, unlike PyTuple_SetItem, so
        # both go back after the pair is stored.
        self._run(
            'd = {"a": 1, "b": 2}\nprint(d, len(d), d["a"])\n',
            b"{'a': 1, 'b': 2} 2 1\n",
        )
        self._run('print((1, 2, 3))\nt = (1, "two")\nprint(t[1])\n', b"(1, 2, 3)\ntwo\n")
        self._run('d = {"xs": [1, 2, 3]}\nprint(d["xs"][1])\n', b"2\n")

    def test_f_strings(self):
        self._run(
            'n = 42\nname = "world"\nprint(f"hello {name}, n is {n}")\n',
            b"hello world, n is 42\n",
        )
        self._run(
            'for i in range(3):\n    print(f"{i} squared is {i * i}")\n',
            b"0 squared is 0\n1 squared is 1\n2 squared is 4\n",
        )

    def test_and_or_yield_an_operand_and_short_circuit(self):
        # `1 and 2` is 2 in Python, not True. And the second operand must not
        # run when the first settles the answer - getting the condition the
        # wrong way round makes `0 and boom()` call boom.
        self._run('print(1 and 2, 0 or 3, [] or "empty")\n', b"2 3 empty\n")
        self._run(
            "def boom():\n    return 1 / 0\nprint(0 and boom())\n", b"0\n"
        )
        self._run(
            "def boom():\n    return 1 / 0\nprint(1 or boom())\n", b"1\n"
        )

    def test_membership_and_subscript_assignment(self):
        # PySequence_Contains answers 1, 0 or -1, and the -1 is a failure
        # rather than a false - treating it as false would turn a raised
        # exception into an answer.
        self._run('print("a" in "abc", "z" in "abc")\n', b"True False\n")
        self._run("print(2 in [1, 2], 9 not in [1, 2])\n", b"True True\n")
        self._run(
            'd = {}\nd["k"] = 1\nd["k"] = d["k"] + 1\nprint(d)\n', b"{'k': 2}\n"
        )

    def test_a_parameter_the_body_reassigns(self):
        # A parameter is storage, not a fresh local: Python rebinds it. The
        # body owns its parameters, so overwriting one releases what it held
        # instead of dropping a reference the caller still owns.
        self._run(
            "def halve_until_one(n):\n"
            "    steps = 0\n"
            "    while n != 1:\n        n = n // 2\n        steps += 1\n"
            "    return steps\n"
            "print(halve_until_one(1024))\n",
            b"10\n",
        )

    def test_a_function_called_many_times_does_not_leak(self):
        # Every name a body binds owns a reference, and leaving without
        # releasing them leaks one per call. Temporaries are not released here
        # - they were released where they were consumed, and doing it twice
        # crashed outright, which is how this was found.
        self._run(
            "def work(a, b):\n    c = a + b\n    c = c * 2\n    return c\n"
            "total = 0\nfor i in range(200000):\n    total = work(i, 1)\n"
            "print(total)\n",
            b"400000\n",
        )

    def test_what_is_not_translated_says_so(self):
        self._reject("class A:\n    pass\n", "has no C-API translation here yet")
        self._reject(
            'print(f"{x:>3}")\n', "format specifier or conversion is not"
        )

        # An unknown name is no longer refused at build time: it is looked up
        # in builtins while the program runs, which is how range() and sum()
        # work. One that does not exist fails then, with AttributeError rather
        # than the NameError CPython gives - the same exit status, a different
        # type, and worth recording rather than papering over.
        self._reject("print(x)\n", "used before it is assigned")

    def test_the_generated_c_declares_what_it_needs_and_no_headers(self):
        # Python.h carries function-pointer typedefs and macros this project's
        # C front end does not parse, so the generated C declares the dozen
        # entry points it uses and includes nothing.
        generated = python_to_capi_c("print(1 + 1)\n", "program.py")
        self.assertNotIn("#include", generated)
        self.assertIn("typedef struct _object PyObject;", generated)
        self.assertIn("extern PyObject *PyNumber_Add", generated)


if __name__ == "__main__":
    unittest.main()
