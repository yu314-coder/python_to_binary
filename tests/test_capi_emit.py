"""Python translated into C that drives the CPython C API.

The tier Nuitka occupies, reached without Nuitka's toolchain: py2bin writes
the C and py2bin's own C compiler turns it into machine code, so no clang,
assembler or linker takes part. What it buys over the native tier is Python's
own semantics - most visibly integers that do not stop at 64 bits.
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import platform
import shutil
import sysconfig
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from py2bin.capi_emit import (
    CApiEmitter,
    CApiEmitError,
    python_program_to_capi_c,
    python_to_capi_c,
)
from py2bin.c_native import compile_c_native

def _square_png(size: int = 16) -> bytes:
    """A plain opaque square, built here so the test carries no binary blob.

    An icon must be square and one of the sizes macOS keeps in an `.icns`.
    """

    import binascii
    import struct
    import zlib

    raw = b"".join(b"\0" + b"\x40\x80\xc0\xff" * size for _ in range(size))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", binascii.crc32(kind + payload))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


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
        """A program that raises: message, status and prior output all match.

        The output matters as much as the message. The interpreter buffers
        stdout and `exit()` does not run its shutdown, so a program that
        printed and then raised showed nothing at all - and this helper, by
        looking only at stderr, had nothing to say about it.
        """

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
            self.assertEqual(native.stdout, reference.stdout)

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

    def test_try_except_catches_what_the_interpreter_raised(self):
        # Inside a try, a failing C-API call becomes a jump to the handler
        # rather than the end of the process, and PyErr_ExceptionMatches asks
        # whether the exception is the class this clause catches - the same
        # question the interpreter asks.
        self._run(
            "try:\n    print(1 / 0)\nexcept ZeroDivisionError:\n"
            '    print("caught")\n',
            b"caught\n",
        )
        self._run(
            'try:\n    print(1 + "x")\nexcept ZeroDivisionError:\n'
            '    print("wrong")\nexcept TypeError:\n    print("right")\n',
            b"right\n",
        )
        self._run(
            'd = {"a": 1}\ntry:\n    print(d["z"])\nexcept KeyError:\n'
            '    print("no such key")\n',
            b"no such key\n",
        )

    def test_a_try_that_does_not_fire_and_a_bare_except(self):
        self._run(
            'try:\n    print("fine")\nexcept ValueError:\n    print("never")\n'
            'print("after")\n',
            b"fine\nafter\n",
        )
        self._run(
            'try:\n    print(1 / 0)\nexcept:\n    print("caught anything")\n',
            b"caught anything\n",
        )

    def test_try_nested_in_a_loop_and_in_a_function(self):
        self._run(
            "for i in range(4):\n    try:\n        print(10 // (i - 2))\n"
            '    except ZeroDivisionError:\n        print("skip")\n',
            b"-5\n-10\nskip\n10\n",
        )
        self._run(
            "def safe(a, b):\n    try:\n        return a // b\n"
            "    except ZeroDivisionError:\n        return 0\n"
            "print(safe(10, 2), safe(10, 0))\n",
            b"5 0\n",
        )
        self._run(
            "try:\n    try:\n        print(1 / 0)\n"
            '    except TypeError:\n        print("inner wrong")\n'
            'except ZeroDivisionError:\n    print("outer caught")\n',
            b"outer caught\n",
        )

    def test_an_unmatched_exception_carries_on_outward(self):
        self._run_failing(
            'try:\n    print(1 / 0)\nexcept TypeError:\n    print("wrong")\n',
            b"ZeroDivisionError",
        )

    def test_from_imports_including_a_submodule(self):
        # `from matplotlib import pyplot` names a submodule that has not been
        # imported yet, so an attribute lookup alone finds nothing. Python's
        # import system tries the submodule at that point and so does this.
        self._run("from math import sqrt, pi\nprint(sqrt(2), pi)\n",
                  b"1.4142135623730951 3.141592653589793\n")
        self._run("import numpy as np\nprint(np.arange(5).sum())\n", b"10\n")

    def test_keyword_arguments(self):
        # The positional part goes in a tuple and the keywords in a dict, for
        # PyObject_Call. PyDict_SetItem does not steal, so both go back.
        self._run("print(sorted([3, 1, 2], reverse=True))\n", b"[3, 2, 1]\n")

    def test_tuple_unpacking(self):
        self._run("a, b = (1, 2)\nprint(a, b)\n", b"1 2\n")

    def test_the_scientific_stack_runs_from_a_compiled_binary(self):
        # numpy is a thin Python layer over C and Fortran, and none of it is
        # translated here - the interpreter loads it exactly as it always
        # does. That is what the tier buys.
        self._run(
            "import numpy\na = numpy.arange(10)\nprint(a.sum(), a.mean())\n",
            b"45 4.5\n",
        )
        self._run(
            "import numpy\n"
            "m = numpy.array([[4.0, 7.0], [2.0, 6.0]])\n"
            "print(round(numpy.linalg.det(m), 6))\n",
            b"10.0\n",
        )

    def test_what_is_not_translated_says_so(self):
        # A format specifier is no longer a refusal - it goes to `format()`,
        # whose mini-language is the interpreter's own. See
        # test_f_string_format_specifiers_go_to_format for what it now does.
        self._reject("raise\n", "nothing to re-raise")
        # `*args` and `**kwargs` in a parameter list are no longer refused;
        # see test_a_function_takes_star_args_and_keywords for what they do.
        self._reject("nonlocal x\n", "`nonlocal` is not translated")
        # `async def`, `async for` and `async with` are no longer refused;
        # see AsyncTests for what they do.

        # An unknown name is no longer refused at build time: it is looked up
        # in builtins while the program runs, which is how range() and sum()
        # work. One that does not exist fails then, with AttributeError rather
        # than the NameError CPython gives - the same exit status, a different
        # type, and worth recording rather than papering over.
        # A name that is not local, not global and not a builtin is no longer
        # a build-time refusal: it is looked up in builtins while the program
        # runs, and fails there with AttributeError rather than NameError.
        # `import x.y` is no longer refused: a dotted import binds what
        # Python binds. See test_a_dotted_import_binds_what_python_binds.

    def test_a_nested_function_captures_from_the_scope_around_it(self):
        """A closure is a real callable, backed by compiled C.

        `PyCFunction_New` wraps the C function; what it captured travels as
        the object CPython hands back as `self`, which is how a plain C
        function comes to have state of its own.
        """

        self._run(
            "def outer(n):\n"
            "    def add(x):\n"
            "        return x + n\n"
            "    return add(5), add(50)\n"
            "print(outer(3))\n",
            b"(8, 53)\n",
        )

    def test_a_closure_outlives_the_call_that_made_it(self):
        self._run(
            "def make(n):\n"
            "    def step(x):\n"
            "        return x * n\n"
            "    return step\n"
            "fs = [make(2), make(3)]\n"
            "print(fs[0](10), fs[1](10))\n",
            b"20 30\n",
        )

    def test_a_lambda_is_a_value_like_any_other(self):
        self._run(
            "rows = [{'k': 3}, {'k': 1}, {'k': 2}]\n"
            "print([r['k'] for r in sorted(rows, key=lambda r: r['k'])])\n"
            "print([f(2) for f in [lambda x: x * x, lambda x: x + 100]])\n",
            b"[1, 2, 3]\n[4, 102]\n",
        )

    def test_a_closure_takes_defaults_like_any_other_function(self):
        self._run(
            "def outer(n):\n"
            "    def add(x, step=10):\n"
            "        return x + n + step\n"
            "    return add(1), add(1, 100)\n"
            "print(outer(0))\n",
            b"(11, 101)\n",
        )

    def test_closures_nest(self):
        self._run(
            "def outer(a):\n"
            "    def middle(b):\n"
            "        def inner(c):\n"
            "            return a + b + c\n"
            "        return inner\n"
            "    return middle\n"
            "print(outer(1)(2)(3))\n",
            b"6\n",
        )

    def test_a_capture_the_scope_moves_afterwards_is_refused(self):
        """Python closes over the variable, this closes over the value.

        Where the captured name is settled by the time the closure is made the
        two agree, and where it is not they do not - so the cases where they
        would disagree are refused rather than quietly answered differently.
        """

        self._reject(
            "def f():\n"
            "    n = 1\n"
            "    def g():\n"
            "        return n\n"
            "    n = 2\n"
            "    return g()\n",
            "binds again afterwards",
        )
        self._reject(
            "def f():\n"
            "    out = []\n"
            "    for i in range(3):\n"
            "        out.append(lambda: i)\n"
            "    return out\n",
            "binds again afterwards",
        )

    def test_a_module_level_capture_keeps_pythons_late_binding(self):
        """Read from the module's own storage, so it moves when Python's does.

        The trap this reproduces is the well-known one: every lambda made in
        the loop sees the value the name ended on, not the one it had.
        """

        self._run(
            "fs = []\n"
            "for i in range(3):\n"
            "    fs.append(lambda: i)\n"
            "print([f() for f in fs])\n",
            b"[2, 2, 2]\n",
        )

    def test_a_raising_function_hands_the_failure_to_its_caller(self):
        """A body with nothing to catch answers NULL with the exception set.

        That is what every C-API function does, and doing the same is what
        lets a `try` around the *call* catch what the body raised. Ending the
        process there instead made an exception uncatchable across a call.
        """

        self._run(
            "def risky(n):\n"
            "    return 10 // n\n"
            "try:\n"
            "    print(risky(0))\n"
            "except ZeroDivisionError as e:\n"
            "    print('caught', e)\n"
            "print(risky(5))\n",
            b"caught division by zero\n2\n",
        )

    def test_a_bare_raise_sets_the_exception_being_handled_again(self):
        self._run(
            "def risky(n):\n"
            "    try:\n"
            "        return 10 // n\n"
            "    except ZeroDivisionError:\n"
            "        print('logged')\n"
            "        raise\n"
            "try:\n"
            "    risky(0)\n"
            "except ZeroDivisionError as e:\n"
            "    print('outer', e)\n",
            b"logged\nouter division by zero\n",
        )

    def test_a_comparison_chain_evaluates_each_operand_once(self):
        """`0 <= x < 10` is not `0 <= x and x < 10`: x is computed once.

        A rewrite into `and` would call it twice, which anything with a side
        effect notices, so the operands go into slots the links read.
        """

        self._run(
            "def loud(v):\n"
            "    print('evaluated', v)\n"
            "    return v\n"
            "print([0 <= x < 10 for x in [-1, 0, 5, 10]])\n"
            "print(0 <= loud(3) < 10)\n"
            "print(5 < loud(1) < loud(99))\n"
            "print(1 < 2 < 3 < 4, 1 < 2 < 3 > 99)\n",
            b"[False, True, True, False]\n"
            b"evaluated 3\nTrue\n"
            b"evaluated 1\nFalse\n"
            b"True False\n",
        )

    def test_f_string_format_specifiers_go_to_format(self):
        """The mini-language is the interpreter's, not a re-implementation."""

        self._run(
            "x = 3.14159\n"
            "n = 42\n"
            "places = 3\n"
            'print(f"{x:.2f} {n:05d} {n:>6} {n:#x} {n:,}")\n'
            'print(f"{x:.{places}f}")\n'
            "print(f\"{'hi'!r} {'hi'!s}\")\n",
            b"3.14 00042     42 0x2a 42\n3.142\n'hi' hi\n",
        )

    def test_print_evaluates_every_argument_before_writing_any(self):
        """A call evaluates all of its arguments before any of it runs.

        Interleaving let `print("x:", loud())` write "x: " before loud()
        spoke, and let `print(7, 1 // 0)` write "7 " before raising.
        """

        self._run(
            "def loud(v):\n"
            "    print('side effect')\n"
            "    return v\n"
            "print('value:', loud(1))\n",
            b"side effect\nvalue: 1\n",
        )

    def test_a_class_is_built_by_the_interpreters_own_type(self):
        """`type(name, bases, namespace)` - so it is CPython's class machinery.

        A method is a closure wrapped in `functools.partialmethod`. A raw
        `PyCFunction` is not a descriptor and would never bind, so the instance
        would simply not arrive; `partialmethod` passes it first, which lands
        at position zero of the argument tuple - where the compiled body
        already reads its first parameter from.
        """

        self._run(
            "class Point:\n"
            "    kind = 'point'\n"
            "    def __init__(self, x, y):\n"
            "        self.x = x\n"
            "        self.y = y\n"
            "    def norm2(self):\n"
            "        return self.x * self.x + self.y * self.y\n"
            "    def shifted(self, dx, dy=1):\n"
            "        return Point(self.x + dx, self.y + dy)\n"
            "    def __repr__(self):\n"
            "        return 'Point(' + str(self.x) + ', ' + str(self.y) + ')'\n"
            "p = Point(3, 4)\n"
            "print(p, p.norm2(), p.kind)\n"
            "print(p.shifted(1), p.shifted(1, 10))\n",
            b"Point(3, 4) 25 point\nPoint(4, 5) Point(4, 14)\n",
        )

    def test_a_class_inherits(self):
        self._run(
            "class A:\n"
            "    def value(self):\n"
            "        return 1\n"
            "class B(A):\n"
            "    def value(self):\n"
            "        return A.value(self) + 10\n"
            "print([o.value() for o in [A(), B()]], isinstance(B(), A))\n",
            b"[1, 11] True\n",
        )

    def test_a_closure_called_back_from_cpython_reaches_its_globals(self):
        """The regression that made compiled closures unsound.

        Static storage used to live in a mapping whose base sat in X28 for the
        whole run. That holds while every call goes outward. A compiled
        closure can be called *inward* - CPython holds it and calls it from
        inside its own frames - and X28 is callee-saved, so a live CPython
        frame that uses it has its own value there. The callback then read a
        module global through whatever CPython left, and `sorted(key=...)`
        segfaulted. Statics now live in the image's writable __DATA and are
        addressed PC-relatively, which no caller can disturb.
        """

        self._run(
            "SCALE = 10\n"
            "def run():\n"
            "    rows = [3, 1, 2]\n"
            "    print(sorted(rows, key=lambda v: SCALE - v))\n"
            "    print(list(map(lambda v: v * SCALE, rows)))\n"
            "    print(max(rows, key=lambda v: SCALE - v))\n"
            "run()\n",
            b"[3, 2, 1]\n[30, 10, 20]\n1\n",
        )

    def test_finally_runs_on_every_way_out(self):
        """Falling off the end, an exception, a return, a break, a continue.

        Each records *why* it is leaving and jumps to the clause, which runs
        once and then does what the reason says. The exception is taken before
        the clause runs, because CPython refuses to build anything Python-side
        while one is set, and put back after with its traceback intact.
        """

        self._run(
            "def plain(n):\n"
            "    try:\n"
            "        return 10 // n\n"
            "    finally:\n"
            "        print('clean', n)\n"
            "print(plain(5))\n"
            "try:\n"
            "    plain(0)\n"
            "except ZeroDivisionError as e:\n"
            "    print('caught', e)\n"
            "def looped(items):\n"
            "    out = []\n"
            "    for v in items:\n"
            "        try:\n"
            "            if v < 0:\n"
            "                continue\n"
            "            if v > 10:\n"
            "                break\n"
            "            out.append(v)\n"
            "        finally:\n"
            "            print('saw', v)\n"
            "    return out\n"
            "print(looped([1, -2, 3, 99, 4]))\n",
            b"clean 5\n2\nclean 0\ncaught division by zero\n"
            b"saw 1\nsaw -2\nsaw 3\nsaw 99\n[1, 3]\n",
        )

    def test_nested_finally_clauses_both_run(self):
        self._run(
            "def nested():\n"
            "    try:\n"
            "        try:\n"
            "            return 'value'\n"
            "        finally:\n"
            "            print('inner')\n"
            "    finally:\n"
            "        print('outer')\n"
            "print(nested())\n",
            b"inner\nouter\nvalue\n",
        )

    def test_arguments_and_keywords_spread_into_a_call(self):
        self._run(
            "rest = [1, 2]\n"
            "more = {'reverse': True}\n"
            "print(max(*[3, 1, 2]))\n"
            "print(sorted(*[[3, 1, 2]], **more))\n"
            "print(dict(**{'x': 1}, y=2))\n"
            "print([*rest, 3], {**more, 'reverse': False})\n",
            b"3\n[3, 2, 1]\n{'x': 1, 'y': 2}\n[1, 2, 3] {'reverse': False}\n",
        )

    def test_a_module_level_def_is_also_a_value(self):
        """The `def` compiles to a plain C function, which is not an object.

        A wrapper makes it one, so `sorted(xs, key=weight)` has something to
        pass and `weight(*row)` has a way to say how many arguments it is
        passing. Both spellings reach the same body.
        """

        self._run(
            "def weight(row):\n"
            "    return row['k']\n"
            "rows = [{'k': 3}, {'k': 1}]\n"
            "print([r['k'] for r in sorted(rows, key=weight)])\n"
            "print(list(map(weight, rows)))\n"
            "print(weight(*[{'k': 9}]))\n",
            b"[1, 3]\n[3, 1]\n9\n",
        )

    def test_a_function_takes_star_args_and_keywords(self):
        """`*args`, `**kwargs`, and keyword-only parameters with defaults."""

        self._run(
            "def total(*values, scale=1, **named):\n"
            "    return sum(values) * scale, sorted(named.items())\n"
            "print(total(1, 2, 3))\n"
            "print(total(1, 2, scale=10, tag='x'))\n"
            "print(total())\n",
            b"(6, [])\n(30, [('tag', 'x')])\n(0, [])\n",
        )

    def test_a_parameter_can_be_passed_by_name(self):
        """Python lets any parameter be passed by name.

        A compiled function that only read the argument *tuple* answered
        `show(1, c=9)` with c's default and said nothing about it, which is the
        worst way to be wrong. Every compiled function now takes keywords and
        binds by position first, then by name.
        """

        self._run(
            "def show(a, b=2, c=3):\n"
            "    return (a, b, c)\n"
            "print(show(1, c=9), show(c=1, a=2), show(1, 2, 3))\n",
            b"(1, 2, 9) (2, 2, 1) (1, 2, 3)\n",
        )

    def test_global_binds_the_modules_own_name(self):
        """`global` used to be accepted and ignored.

        The assignment bound a local of the same spelling, so the module never
        saw the change - accepted, silent, and wrong.
        """

        self._run(
            "count = 0\n"
            "def bump(n):\n"
            "    global count\n"
            "    count = count + n\n"
            "    return count\n"
            "def shadow():\n"
            "    count = 99\n"
            "    return count\n"
            "print(bump(3), bump(4), count)\n"
            "print(shadow(), count)\n",
            b"3 7 7\n99 7\n",
        )

    def test_a_dotted_import_binds_what_python_binds(self):
        self._run(
            "import os.path\n"
            "import xml.etree.ElementTree as ET\n"
            "print(os.path.basename('/a/b/c.txt'))\n"
            "print(ET.fromstring(\"<r><c v='1'/></r>\")[0].get('v'))\n",
            b"c.txt\n1\n",
        )

    def test_bytes_literals(self):
        self._run(
            "data = b'hello'\n"
            "print(data, len(data), data[0])\n"
            "print(b'caf\\xc3\\xa9'.decode('utf-8'))\n"
            "print(data + b' world', data.hex())\n",
            "b'hello' 5 104\ncaf\u00e9\nb'hello world' 68656c6c6f\n".encode("utf-8"),
        )

    def test_positional_only_parameters(self):
        """`/` means the name is not a keyword, so it can be one for `**kw`.

        `write_function` read only `args.args`, which leaves the
        positional-only ones out entirely - so `def f(a, /, b)` bound `b` from
        the first argument and dropped `a`, accepted and silently wrong.
        """

        self._run(
            "def f(a, b, /, c, d=4, *, e=5):\n"
            "    return (a, b, c, d, e)\n"
            "def g(a, /, **kw):\n"
            "    return (a, sorted(kw.items()))\n"
            "print(f(1, 2, 3))\n"
            "print(f(1, 2, c=3, d=9, e=8))\n"
            "print(g(1, a=2, z=3))\n",
            b"(1, 2, 3, 4, 5)\n(1, 2, 3, 9, 8)\n(1, [('a', 2), ('z', 3)])\n",
        )

    def test_temporary_slots_are_reused_between_statements(self):
        """One slot per live value, not one per subexpression ever written.

        A temporary is dead once the statement that made it has finished, so
        the count is wound back at each statement boundary. Without that, a
        7,000-line module wanted a larger entry frame than py2bin gives a
        frame at all - the failure was a build-time refusal naming the stack,
        which said nothing about the real cause.
        """

        source = "".join(f"print({n} + {n} * 2 - 1, [{n}, {n} + 1])\n" for n in range(200))
        generated = python_to_capi_c(source, "program.py")
        slots = len(re.findall(r"^\s+PyObject \*_t\d+ = 0;$", generated, re.M))
        self.assertLess(slots, 30, "temporaries are not being reused")
        # And it still runs: reuse must not hand a slot out while it is live.
        self._run(
            "".join(f"print({n} + {n} * 2 - 1, [{n}, {n} + 1])\n" for n in range(3)),
            b"-1 [0, 1]\n2 [1, 2]\n5 [2, 3]\n",
        )

    def test_a_comprehension_has_a_scope_of_its_own(self):
        """`[x * 2 for x in xs]` must not touch the enclosing `x`.

        The target was bound as an ordinary name, so a comprehension whose
        variable happened to share a spelling with something outside it
        overwrote that - `print(x)` afterwards answered with the
        comprehension's last item. Accepted, silent, and wrong.
        """

        self._run(
            "x = 7\n"
            "xs = [0, 1, 2, 3]\n"
            "print(x, [x * 2 for x in xs])\n"
            "print(sum(x for x in xs), x)\n"
            "print({x: x for x in xs}, x)\n"
            "print({x for x in xs}, x)\n"
            "n = 'keep'\n"
            "print([[n for n in range(2)] for n in range(2)], n)\n",
            b"7 [0, 2, 4, 6]\n6 7\n{0: 0, 1: 1, 2: 2, 3: 3} 7\n{0, 1, 2, 3} 7\n"
            b"[[0, 1], [0, 1]] keep\n",
        )

    def test_negation_keeps_the_sign_of_zero(self):
        """`-x` is not `0 - x`.

        It was, for want of an entry point, on the reasoning that they are the
        same operation. `0 - 0.0` is positive zero where `-0.0` is negative
        zero, so a list of floats came back with one sign quietly changed.
        """

        self._run(
            "print([1.0, 0.0, -0.0, -1.5])\n"
            "print(-0.0, 0.0 == -0.0, +5, ~5, -(-3))\n",
            b"[1.0, 0.0, -0.0, -1.5]\n-0.0 True 5 -6 3\n",
        )

    def test_unpacking_checks_how_many_there_were(self):
        """`a, b = (1, 2, 3)` bound two names and said nothing.

        Going through a tuple first is what makes the length knowable, and it
        also makes unpacking work on any iterable rather than only on
        something indexable.
        """

        self._run(
            "try:\n"
            "    a, b = (1, 2, 3)\n"
            "except ValueError as e:\n"
            "    print('too many:', e)\n"
            "try:\n"
            "    a, b, c = (1, 2)\n"
            "except ValueError as e:\n"
            "    print('too few:', e)\n"
            "a, b = 'xy'\n"
            "print(a, b)\n",
            b"too many: too many values to unpack (expected 2, got 3)\n"
            b"too few: not enough values to unpack (expected 3, got 2)\n"
            b"x y\n",
        )

    def test_output_survives_an_uncaught_exception(self):
        """The interpreter buffers stdout and `exit()` does not run shutdown.

        A program that printed and then raised showed nothing at all, which
        hides the very output that says how far it got.
        """

        self._run_failing(
            "print('before the failure')\nraise ValueError('boom')\n",
            b"ValueError: boom",
        )

    def test_a_name_that_does_not_exist_raises_NameError(self):
        """Past builtins there is nowhere else to look.

        The failed lookup left an AttributeError naming the *builtins module*,
        which is neither the program's name nor its problem; left set, the
        next thing done turned into `SystemError: ... returned a result with
        an exception set`, which names neither.
        """

        self._run(
            "try:\n"
            "    missing_function(1)\n"
            "except NameError as e:\n"
            "    print('caught:', e)\n",
            b"caught: name 'missing_function' is not defined\n",
        )

    def test_a_name_whose_only_binding_did_not_run(self):
        """A slot the program binds may never have been written to.

        `d` is a name of the module even when the only `d = ...` sits in an
        `if` that did not run. Reading it found NULL, and `Py_IncRef(NULL)`
        became `SystemError: null argument to internal routine`, which says
        nothing about `d`.
        """

        self._run_failing(
            "n = 0\nif n > 5:\n    d = {1: 2}\nprint(len(d))\n",
            b"NameError: name 'd' is not defined",
        )
        self._run(
            "def f(flag):\n"
            "    if flag:\n"
            "        v = 1\n"
            "    return v\n"
            "print(f(True))\n"
            "try:\n"
            "    f(False)\n"
            "except UnboundLocalError as e:\n"
            "    print('caught:', e)\n",
            b"1\ncaught: cannot access local variable 'v' where it is not "
            b"associated with a value\n",
        )

    def test_an_unbound_name_error_carries_the_name(self):
        """`e.name` is what CPython sets, and what its display reads.

        The compiled binary cannot produce CPython's "Did you mean: 'id'?" -
        that suggestion is computed by searching a Python *frame*, and a
        compiled program has none. The attribute itself is set, so a program
        that catches the error and reads it gets what Python gives.
        """

        self._run(
            "try:\n"
            "    print(nope)\n"
            "except NameError as e:\n"
            "    print(repr(e.name), str(e))\n",
            b"'nope' name 'nope' is not defined\n",
        )

    def test_the_unbound_check_is_not_emitted_where_it_cannot_fire(self):
        """A name a statement of this body has already bound needs no test.

        Testing every read put a third more C into a large module. A read is
        left alone when an unconditional statement above it bound the name,
        when it is a `for`/`with`/`except` target inside its own body, or when
        it is a builtin - and the check that remains is one call to a helper
        written once rather than a dozen lines at each of a thousand sites.
        """

        settled = python_to_capi_c(
            "import os\n"
            "total = 0\n"
            "for item in [1, 2]:\n"
            "    total = total + item\n"
            "print(total, os, Exception)\n",
            "program.py",
        )
        self.assertNotIn("_py2bin_unbound", settled)

        # And it is emitted where the name really can be unbound: reading the
        # loop target *after* the loop finds nothing when the sequence was
        # empty, which is a NameError in Python too.
        after = python_to_capi_c(
            "for item in []:\n    pass\nprint(item)\n", "program.py"
        )
        self.assertIn("_py2bin_unbound", after)
        self._run_failing(
            "for item in []:\n    pass\nprint(item)\n",
            b"NameError: name 'item' is not defined",
        )

    def test_zero_argument_super(self):
        """`super()` is `super(__class__, self)`.

        CPython supplies those two through a cell it creates for any method
        that mentions the name. A compiled method has no cell, so the two
        values are written out - the same ones, named rather than implied. The
        class is read when the method runs, by which time it exists.
        """

        self._run(
            "class A:\n"
            "    def __init__(self, v):\n"
            "        self.v = v\n"
            "    def describe(self):\n"
            "        return 'A(' + str(self.v) + ')'\n"
            "class B(A):\n"
            "    def __init__(self, v, extra):\n"
            "        super().__init__(v)\n"
            "        self.extra = extra\n"
            "    def describe(self):\n"
            "        return 'B[' + super().describe() + ', ' + str(self.extra) + ']'\n"
            "class C(B):\n"
            "    def describe(self):\n"
            "        return 'C{' + super().describe() + '}'\n"
            "print(C(3, 4).describe())\n",
            b"C{B[A(3), 4]}\n",
        )

    def test_too_many_arguments_is_refused_like_python(self):
        """The extras used to sit unread in the tuple.

        A call with the wrong shape ran anyway and answered - CPython's own
        demo of this is `super().__init__(1, 2)` against an `__init__(self,
        v)`, which raises there and returned here. The message follows
        CPython's, including the qualified name, which is the one part a
        compiled function cannot read off itself.
        """

        self._run(
            "def outer():\n"
            "    def one(a):\n"
            "        return a\n"
            "    def two(a, b=2):\n"
            "        return (a, b)\n"
            "    def rest(a, *more):\n"
            "        return (a, more)\n"
            "    for call, label in [\n"
            "        (lambda: one(1, 2), 'one(1,2)'),\n"
            "        (lambda: two(1, 2, 3), 'two(1,2,3)'),\n"
            "        (lambda: one(), 'one()'),\n"
            "    ]:\n"
            "        try:\n"
            "            print(label, '->', call())\n"
            "        except TypeError as e:\n"
            "            print(label, '->', e)\n"
            "    print('ok:', rest(1, 2, 3), two(1))\n"
            "outer()\n",
            b"one(1,2) -> outer.<locals>.one() takes 1 positional argument "
            b"but 2 were given\n"
            b"two(1,2,3) -> outer.<locals>.two() takes from 1 to 2 positional "
            b"arguments but 3 were given\n"
            b"one() -> outer.<locals>.one() missing 1 required positional "
            b"argument: 'a'\n"
            b"ok: (1, (2, 3)) (1, 2)\n",
        )

    def test_a_method_is_named_for_its_class(self):
        self._run(
            "class A:\n"
            "    def __init__(self, v):\n"
            "        self.v = v\n"
            "try:\n"
            "    A(1, 2)\n"
            "except TypeError as e:\n"
            "    print(e)\n",
            b"A.__init__() takes 2 positional arguments but 3 were given\n",
        )

    def test_runaway_recursion_is_an_exception_not_a_dead_process(self):
        """Compiled calls use the real stack, which the OS takes away silently.

        A recursion with no base case segfaulted where CPython raises
        RecursionError - and a segfault is not something a program can catch,
        report, or clean up after. Each compiled body counts itself in and out
        through the interpreter's own depth counter, which is what CPython
        does for every call it makes.
        """

        self._run(
            "def deep(n):\n"
            "    return deep(n + 1)\n"
            "try:\n"
            "    deep(0)\n"
            "except RecursionError:\n"
            "    print('caught RecursionError')\n",
            b"caught RecursionError\n",
        )

    def test_the_depth_counter_is_given_back(self):
        """A level entered and not left is never recovered.

        The interpreter would come to believe the stack is deeper than it is
        and refuse calls that are perfectly fine, so every way out of a body -
        a return, a raise, a `finally` - counts back out. Recursion that
        returns normally, over and over, is what shows the counting balances.
        """

        self._run(
            "def fact(n):\n"
            "    if n < 2:\n"
            "        return 1\n"
            "    return n * fact(n - 1)\n"
            "def guarded(n):\n"
            "    try:\n"
            "        return fact(n)\n"
            "    finally:\n"
            "        pass\n"
            "for _ in range(400):\n"
            "    guarded(20)\n"
            "print(fact(25))\n",
            b"15511210043330985984000000\n",
        )

    def test_loop_and_try_else_clauses(self):
        """`else` runs when the loop was not left by `break`.

        Not "when it ended early": the test failing, or the sequence running
        out, is the ordinary way out and the else runs then. Nested loops each
        need their own flag or an inner `break` would silence an outer else.
        """

        self._run(
            "for n in [1, 2]:\n"
            "    pass\n"
            "else:\n"
            "    print('for-else')\n"
            "for n in [1, 2]:\n"
            "    break\n"
            "else:\n"
            "    print('not reached')\n"
            "i = 0\n"
            "while i < 2:\n"
            "    i += 1\n"
            "else:\n"
            "    print('while-else', i)\n"
            "out = []\n"
            "for a in [1, 2]:\n"
            "    for b in [1, 2]:\n"
            "        break\n"
            "    else:\n"
            "        out.append('inner')\n"
            "else:\n"
            "    out.append('outer')\n"
            "print(out)\n"
            "try:\n"
            "    v = 1\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    print('try-else', v)\n",
            b"for-else\nwhile-else 2\n['outer']\ntry-else 1\n",
        )

    def test_with_closes_however_the_body_ends(self):
        """__exit__ used to be written after the body.

        So it ran only when the body fell off the end - a `break`, a `return`
        or an exception left without it, and the thing the `with` exists to
        close was not closed. Silently.
        """

        self._run(
            "class C:\n"
            "    def __init__(self, tag):\n"
            "        self.tag = tag\n"
            "    def __enter__(self):\n"
            "        return self.tag\n"
            "    def __exit__(self, kind, value, trace):\n"
            "        print('exit', self.tag, kind.__name__ if kind else None)\n"
            "        return False\n"
            "for i in range(3):\n"
            "    with C('loop') as t:\n"
            "        if i == 1:\n"
            "            break\n"
            "def leaves():\n"
            "    with C('ret') as t:\n"
            "        return 'returned'\n"
            "print(leaves())\n"
            "try:\n"
            "    with C('raise') as t:\n"
            "        raise ValueError('boom')\n"
            "except ValueError as e:\n"
            "    print('caught', e)\n",
            b"exit loop None\nexit loop None\nexit ret None\nreturned\n"
            b"exit raise ValueError\ncaught boom\n",
        )

    def test_an_exit_that_returns_true_suppresses(self):
        """The three arguments are the exception, so __exit__ can swallow it."""

        self._run(
            "class S:\n"
            "    def __enter__(self):\n"
            "        return self\n"
            "    def __exit__(self, kind, value, trace):\n"
            "        print('suppressing', kind.__name__)\n"
            "        return True\n"
            "with S():\n"
            "    raise KeyError('hidden')\n"
            "print('carried on')\n",
            b"suppressing KeyError\ncarried on\n",
        )

    def test_annotations_assert_del_and_chained_assignment(self):
        self._run(
            "x: int = 5\n"
            "y: str\n"
            "a = b = [0]\n"
            "a.append(1)\n"
            "print(x, a, b, a is b)\n"
            "class P:\n"
            "    n: int = 3\n"
            "    def __init__(self):\n"
            "        self.v: int = 7\n"
            "print(P.n, P().v)\n"
            "q = 1\n"
            "del q\n"
            "try:\n"
            "    print(q)\n"
            "except NameError as e:\n"
            "    print('caught:', e)\n"
            "try:\n"
            "    assert False, 'boom ' + str(3)\n"
            "except AssertionError as e:\n"
            "    print('assert:', e)\n",
            b"5 [0, 1] [0, 1] True\n3 7\ncaught: name 'q' is not defined\n"
            b"assert: boom 3\n",
        )

    def test_print_keywords_go_to_the_interpreters_print(self):
        """`end=`, `sep=`, `file=`, `flush=` and a spread argument.

        The fast path writes straight to sys.stdout and knows none of them.
        """

        self._run(
            "print('a', 'b', sep='-')\n"
            "print('no newline', end='')\n"
            "print('|', flush=True)\n"
            "parts = ['x', 'y']\n"
            "print(*parts, sep='+')\n"
            "print()\n",
            b"a-b\nno newline|\nx+y\n\n",
        )

    def test_an_integer_of_any_width(self):
        """A Python integer has no width; a C literal does.

        `-9223372036854775808` is one literal in Python and two nodes in the
        tree, so negating afterwards needs the positive half to exist first -
        and that is exactly one past what a signed 64-bit integer holds. C has
        no literal for the most negative value either, so it is written as a
        subtraction that never leaves the range.
        """

        self._run(
            "print(-9223372036854775808, 9223372036854775807)\n"
            "print(9223372036854775808, -9223372036854775809)\n"
            "print(10 ** 30, 2 ** 100)\n"
            "print(123456789012345678901234567890 + 1)\n",
            b"-9223372036854775808 9223372036854775807\n"
            b"9223372036854775808 -9223372036854775809\n"
            b"1000000000000000000000000000000 "
            b"1267650600228229401496703205376\n"
            b"123456789012345678901234567891\n",
        )

    def test_text_and_bytes_carrying_a_zero_byte(self):
        """A zero byte is a character in Python and an end in C.

        A literal with one went through `PyUnicode_FromString`, which stops
        there, so the string arrived truncated. The decoder that is told how
        long it is reads every byte.
        """

        self._run(
            "s = 'a\\0b'\n"
            "print(len(s), s.split('\\0'))\n"
            "b = b'x\\0y\\0z'\n"
            "print(len(b), b.split(b'\\0'))\n",
            b"3 ['a', 'b']\n5 [b'x', b'y', b'z']\n",
        )

    def test_decorators(self):
        """`@a` then `@b` on a `def` is `a(b(f))`, applied from the bottom up."""

        self._run(
            "def tag(label):\n"
            "    def outer(f):\n"
            "        def wrapper(*a):\n"
            "            return label + ':' + str(f(*a))\n"
            "        return wrapper\n"
            "    return outer\n"
            "@tag('A')\n"
            "@tag('B')\n"
            "def value():\n"
            "    return 1\n"
            "print(value())\n"
            "def register(k):\n"
            "    def outer(cls):\n"
            "        cls.tag = k\n"
            "        return cls\n"
            "    return outer\n"
            "@register('marked')\n"
            "class D:\n"
            "    pass\n"
            "print(D.tag)\n"
            "import functools\n"
            "@functools.lru_cache(maxsize=None)\n"
            "def slow(n):\n"
            "    return n * 2\n"
            "print(slow(4), slow(4))\n",
            b"A:B:1\nmarked\n8 8\n",
        )

    def test_the_descriptor_decorators(self):
        """A decorated method is handed the plain callable, not the wrapper.

        `staticmethod`, `classmethod` and `property` are descriptors that do
        their own binding, so `partialmethod` must not get in the way - and an
        ordinary wrapping decorator returns a Python function, which binds by
        itself and passes the instance as the first argument, which is where a
        compiled method reads it from anyway.
        """

        self._run(
            "class C:\n"
            "    @staticmethod\n"
            "    def stat(x):\n"
            "        return 'stat ' + str(x)\n"
            "    @classmethod\n"
            "    def named(cls, x):\n"
            "        return cls.__name__ + ' ' + str(x)\n"
            "    @property\n"
            "    def prop(self):\n"
            "        return 'prop'\n"
            "    def plain(self):\n"
            "        return 'plain'\n"
            "c = C()\n"
            "print(C.stat(1), c.stat(2))\n"
            "print(C.named(3), c.named(4))\n"
            "print(c.prop, c.plain())\n",
            b"stat 1 stat 2\nC 3 C 4\nprop plain\n",
        )

    def test_functools_wraps_cannot_rename_a_compiled_function(self):
        """A compiled function is a builtin function object.

        `functools.wraps` copies `__name__` onto the wrapper by assigning it,
        and that attribute is read-only on this kind of object. The failure is
        an AttributeError naming the attribute, which is the truth rather than
        a silent difference - but it does mean the idiom does not work here.
        """

        source = (
            "import functools\n"
            "def shout(f):\n"
            "    @functools.wraps(f)\n"
            "    def wrapper(n):\n"
            "        return f(n)\n"
            "    return wrapper\n"
            "@shout\n"
            "def greet(n):\n"
            "    return n\n"
            "print(greet(1))\n"
        )
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
            # This one is asserted as a *difference*, not a shared failure:
            # under CPython the program runs, and here it does not.
            self.assertEqual(reference.returncode, 0)
            self.assertEqual(native.returncode, 1)
            self.assertIn(b"AttributeError", native.stderr)
            self.assertIn(b"__name__", native.stderr)

    def test_a_program_of_several_modules_is_linked_into_one_image(self):
        """The entry's own imports are compiled, not read as source.

        Compiling only the entry left the rest of a multi-file program to be
        found as `.py` beside the binary - so most of it was never compiled.
        Each module beside the entry is now compiled and registered under its
        own name before its body runs, which is what makes an `import` of it
        find the compiled one.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "helper.py").write_text(
                "COUNT = 0\n"
                "class Greeter:\n"
                "    def __init__(self, who):\n"
                "        self.who = who\n"
                "    def hello(self):\n"
                "        return 'hello ' + self.who\n"
                "def bump():\n"
                "    global COUNT\n"
                "    COUNT += 1\n"
                "    return COUNT\n",
                encoding="utf-8",
            )
            entry = root / "program.py"
            entry.write_text(
                "import helper\n"
                "from helper import Greeter\n"
                "print(Greeter('world').hello())\n"
                "print(helper.bump(), helper.bump())\n"
                "print(helper.COUNT, helper.__name__, __name__)\n",
                encoding="utf-8",
            )
            generated, linked = python_program_to_capi_c(entry)
            self.assertEqual(linked, ["helper"])
            if not _HOST_IS_DARWIN_ARM64:
                return
            source = root / "program.c"
            source.write_text(generated, encoding="utf-8", newline="\n")
            binary = root / "program.bin"
            compile_c_native(source, binary, target="darwin-arm64", clean=True)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True, cwd=root
            )
            # Run it where none of the program's source can be found, which is
            # what proves the modules travelled inside the binary. `helper.COUNT`
            # is the part that catches a module object left holding stale values:
            # bump() writes the slot, and the attribute has to follow.
            (root / "helper.py").unlink()
            (root / "program.py").unlink()
            native = subprocess.run([str(binary)], capture_output=True, cwd=root)
            self.assertEqual(
                native.stdout, b"hello world\n1 2\n2 helper __main__\n"
            )
            self.assertEqual(native.stdout, reference.stdout)

    def test_every_module_has_its_dunders(self):
        """`if __name__ == '__main__':` is how a script says where it starts.

        Without it the entry point of a program simply does not run, and
        `os.path.abspath(__file__)` is how a program finds what sits beside it.
        """

        self._run(
            "print(__name__)\n"
            "print(__file__.endswith('.py'))\n"
            "if __name__ == '__main__':\n"
            "    print('entry point ran')\n",
            b"__main__\nTrue\nentry point ran\n",
        )

    def test_a_compiled_function_has_a_signature(self):
        """`inspect.signature` said "unsupported callable" for every one.

        A compiled function is a builtin function object and carries no
        signature unless its doc begins with one in the shape CPython reads
        `__text_signature__` out of. Anything that introspects refused to work
        with them - pywebview would not bind a single method of a compiled
        application. Defaults read as None: the format has no spelling for an
        arbitrary Python expression.
        """

        source = (
            "import inspect\n"
            "def plain(a, b=2, *rest, key=None, **kw):\n"
            "    return a\n"
            "class C:\n"
            "    def method(self, x, y=1):\n"
            "        return x\n"
            "print(list(inspect.signature(plain).parameters))\n"
            "print(list(inspect.signature(C().method).parameters))\n"
            "print(inspect.ismethod(C().method))\n"
        )
        # The names and the binding agree with CPython exactly, which is what
        # a caller and an introspecting framework act on.
        self._run(
            source,
            b"['a', 'b', 'rest', 'key', 'kw']\n['x', 'y']\nTrue\n",
        )
        # What does not agree is pinned here rather than left to drift: the
        # format has no spelling for an arbitrary Python expression, so every
        # default reads as None.
        self.assertIn(
            '"plain(a, b=None, *rest, key=None, **kw)',
            python_to_capi_c(source, "program.py"),
        )

    def test_a_compiled_program_can_be_a_macos_app_with_an_icon(self):
        """The compiled binary, in the bundle macOS expects, with its icon.

        The `.app` is only a directory shape and a plist - what is inside it
        here is the compiled program itself, not an interpreter and a copy of
        the source.
        """

        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("the C-API path is darwin-arm64 only")
        import plistlib

        from py2bin.c_native import compile_c_native

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text("print('windowed')\n", encoding="utf-8")
            source = root / "program.c"
            source.write_text(
                python_to_capi_c(entry.read_text(), str(entry)),
                encoding="utf-8",
                newline="\n",
            )
            icon = root / "icon.png"
            icon.write_bytes(_square_png())
            bundle = root / "Program.app"
            compile_c_native(
                source,
                bundle,
                target="darwin-arm64",
                clean=True,
                app=True,
                app_name="My Program",
                icon=icon,
            )
            executable = bundle / "Contents" / "MacOS" / "My Program"
            self.assertTrue(executable.is_file())
            plist = plistlib.loads(
                (bundle / "Contents" / "Info.plist").read_bytes()
            )
            self.assertEqual(plist["CFBundleName"], "My Program")
            self.assertEqual(plist["CFBundleExecutable"], "My Program")
            self.assertEqual(plist["CFBundleIconFile"], "AppIcon.icns")
            icns = bundle / "Contents" / "Resources" / "AppIcon.icns"
            self.assertTrue(icns.is_file())
            self.assertEqual(icns.read_bytes()[:4], b"icns")
            ran = subprocess.run([str(executable)], capture_output=True)
            self.assertEqual(ran.stdout, b"windowed\n")

    def test_extra_search_paths_are_baked_in(self):
        """The linked interpreter's search path is the build machine's.

        A compiled binary carries the program but not its dependencies, and
        the interpreter it links knows nothing of where they were installed -
        so an application that is otherwise complete stops at
        ModuleNotFoundError for a package plainly present on the machine.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            elsewhere = root / "packages"
            elsewhere.mkdir()
            (elsewhere / "only_here.py").write_text(
                "VALUE = 'found it'\n", encoding="utf-8"
            )
            entry = root / "program.py"
            entry.write_text(
                "import only_here\nprint(only_here.VALUE)\n", encoding="utf-8"
            )
            generated, _linked = python_program_to_capi_c(
                entry, (str(elsewhere),)
            )
            self.assertIn("sys.path.insert", generated)
            self.assertIn("_py2bin_dir", generated)
            if not _HOST_IS_DARWIN_ARM64:
                return
            source = root / "program.c"
            source.write_text(generated, encoding="utf-8", newline="\n")
            binary = root / "program.bin"
            compile_c_native(source, binary, target="darwin-arm64", clean=True)
            # Run somewhere else entirely, with nothing helping it along: the
            # path has to have travelled inside the binary.
            environment = {
                key: value
                for key, value in os.environ.items()
                if key != "PYTHONPATH"
            }
            ran = subprocess.run(
                [str(binary)], capture_output=True, cwd="/", env=environment
            )
            self.assertEqual(ran.stdout, b"found it\n")

    def test_a_binary_finds_itself_wherever_it_is_moved(self):
        """`__file__` and a relative `--site` follow the binary, not the build.

        Baking the build path meant `os.path.dirname(__file__)` named a
        directory on the machine that compiled it, so a bundle that was moved
        looked for its own files somewhere that did not exist. An embedded
        interpreter resolves `sys.executable` to the host program, which is
        what lets a compiled artifact find itself.
        """

        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("the C-API path is darwin-arm64 only")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            built = root / "built"
            built.mkdir()
            (built / "pkgs").mkdir()
            (built / "pkgs" / "carried.py").write_text(
                "NAME = 'carried along'\n", encoding="utf-8"
            )
            entry = built / "program.py"
            entry.write_text(
                "import os, sys, carried\n"
                "print(carried.NAME)\n"
                "print(os.path.dirname(os.path.abspath(__file__))\n"
                "      == os.path.dirname(os.path.abspath(sys.executable)))\n"
                "print(sys.argv[0] == sys.executable)\n",
                encoding="utf-8",
            )
            # A relative --site is kept relative all the way through: the CLI
            # once resolved it against the build directory, which produced an
            # absolute path to somewhere that never existed, and the bundle
            # failed to import what was sitting inside it.
            from py2bin.cli import _site_paths

            relative, absolute = _site_paths(["pkgs", str(root)])
            self.assertEqual(relative, "pkgs")
            self.assertTrue(Path(absolute).is_absolute())
            generated, _linked = python_program_to_capi_c(entry, ("pkgs",))
            source = built / "program.c"
            source.write_text(generated, encoding="utf-8", newline="\n")
            binary = built / "program.bin"
            compile_c_native(source, binary, target="darwin-arm64", clean=True)

            # Move the whole thing somewhere the build never heard of, and
            # take nothing else with it.
            moved = root / "moved" / "deeper"
            moved.mkdir(parents=True)
            shutil.copy(binary, moved / "program.bin")
            shutil.copytree(built / "pkgs", moved / "pkgs")
            environment = {
                key: value
                for key, value in os.environ.items()
                if key != "PYTHONPATH"
            }
            ran = subprocess.run(
                [str(moved / "program.bin")],
                capture_output=True,
                cwd="/",
                env=environment,
            )
            self.assertEqual(ran.stdout, b"carried along\nTrue\nTrue\n")

    def test_an_embedded_interpreter_makes_the_bundle_portable(self):
        """The interpreter inside the bundle, named relative to the executable.

        A compiled artifact names its interpreter in an LC_LOAD_DYLIB, and dyld
        resolves that before a line of the program runs - so the build
        machine's absolute path is a refusal to launch anywhere else, with no
        message from the program at all.

        The framework *layout* has to be carried, not the bare library: the
        signature on that library seals its neighbouring Resources/Info.plist,
        and a dylib lifted out on its own is reported as modified.
        """

        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("the C-API path is darwin-arm64 only")
        from py2bin.c_native import compile_c_native
        from py2bin.cli import _embedded_python_path
        from py2bin.freezer import embed_cpython_in_app

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(
                "import json, sys\n"
                "print(json.dumps({'ok': True}))\n"
                "print(sys.prefix)\n",
                encoding="utf-8",
            )
            source = root / "program.c"
            source.write_text(
                python_to_capi_c(entry.read_text(), str(entry)),
                encoding="utf-8",
                newline="\n",
            )
            bundle = root / "Program.app"
            compile_c_native(
                source,
                bundle,
                target="darwin-arm64",
                clean=True,
                app=True,
                app_name="Program",
                python_dylib=_embedded_python_path(),
            )
            embed_cpython_in_app(bundle)
            # What is carried is what runs: no build-time pieces, no dead
            # architecture, no libraries nothing references.
            carried = bundle / "Contents" / "lib" / f"python{sysconfig.get_config_var('py_version_short')}"
            self.assertFalse(list(carried.glob("config-*")))
            self.assertFalse((carried / "ensurepip").exists())
            interpreter = (
                bundle / "Contents" / "Frameworks" / "Python.framework"
            )
            thin = subprocess.run(
                ["file", *[str(x) for x in interpreter.rglob("Python")]],
                capture_output=True,
                text=True,
            ).stdout
            self.assertNotIn("universal binary", thin)

            executable = bundle / "Contents" / "MacOS" / "Program"
            loaded = subprocess.run(
                ["otool", "-L", str(executable)], capture_output=True, text=True
            ).stdout
            self.assertIn("@executable_path/../Frameworks/Python.framework", loaded)
            self.assertNotIn("/Library/Frameworks/Python.framework", loaded)

            # Somewhere the build never heard of, with nothing helping it.
            moved = root / "elsewhere"
            moved.mkdir()
            shutil.copytree(bundle, moved / "Program.app", symlinks=True)
            environment = {
                key: value
                for key, value in os.environ.items()
                if key != "PYTHONPATH"
            }
            ran = subprocess.run(
                [str(moved / "Program.app" / "Contents" / "MacOS" / "Program")],
                capture_output=True,
                cwd="/",
                env=environment,
            )
            lines = ran.stdout.decode().splitlines()
            self.assertEqual(lines[0], '{"ok": true}')
            # The standard library it used is the one inside the bundle.
            self.assertTrue(
                lines[1].endswith("Program.app/Contents"), lines[1]
            )

    def test_a_generator_expression_is_an_iterator(self):
        """Gathered eagerly, but handed back as an iterator.

        A list answers `for` and `sum()` identically and `next()` not at all,
        so `next((p for p in candidates if ...), None)` stopped with "'list'
        object is not an iterator" - which names nothing the program wrote.
        This is what a real application hit; the unit tests had only ever fed
        a generator expression straight to something that iterates it.
        """

        self._run(
            "candidates = ['a', 'bb', 'ccc']\n"
            "print(next((p for p in candidates if len(p) == 2), None))\n"
            "print(next((p for p in candidates if p == 'zzz'), 'fallback'))\n"
            "g = (v * 2 for v in [1, 2, 3])\n"
            "print(next(g), next(g), list(g))\n"
            "print(sum(v for v in [1, 2, 3]), max(v for v in [4, 9, 2]))\n",
            b"bb\nfallback\n2 4 [6]\n6 9\n",
        )

    def test_a_call_passes_its_arguments_in_an_array(self):
        """Not in a tuple built for the purpose.

        Every call allocated a tuple, filled it, and freed it. The interpreter
        stopped using the tuple protocol years ago, and calling a builtin from
        compiled code measured slower than the same call interpreted. The array
        is reused per arity within a function: the arguments are all computed
        before any is stored, so a nested call has finished with it before the
        outer one begins to fill it.

        Vectorcall *borrows* its arguments, where PyTuple_SetItem steals - so
        each is released after the call rather than given away.
        """

        generated = python_to_capi_c(
            "def f(a, b, c):\n    return a + b + c\n"
            "print(f(1, 2, 3), max(4, 5, 6))\n",
            "program.py",
        )
        self.assertIn("PyObject_Vectorcall", generated)
        self.assertIn("_args3[3]", generated)
        self._run(
            "xs = [3, 1, 2]\n"
            "def three(a, b, c):\n"
            "    return (a, b, c)\n"
            "print(three(1, 2, 3))\n"
            "print(max(4, 5, 6), sorted(xs), min(1, 2, 3))\n"
            "print(three(*xs))\n"
            "def outer():\n"
            "    def inner(a, b):\n"
            "        return a * b\n"
            "    return [inner(i, 2) for i in range(4)]\n"
            "print(outer())\n",
            b"(1, 2, 3)\n6 [1, 2, 3] 1\n(3, 1, 2)\n[0, 2, 4, 6]\n",
        )

    def test_a_compiled_function_is_called_without_a_tuple(self):
        """METH_FASTCALL: the arguments arrive in the caller's own array.

        A compiled function was registered METH_VARARGS, so CPython packed the
        arguments back into a tuple to call one - undoing at the boundary the
        work the caller had just saved. That is why a nested function, which is
        always reached that way, was the slowest thing a compiled program could
        call.
        """

        generated = python_to_capi_c(
            "def outer():\n"
            "    def inner(a, b=2, *rest, key=None, **kw):\n"
            "        return (a, b, rest, key, sorted(kw))\n"
            "    return inner\n",
            "program.py",
        )
        # METH_FASTCALL | METH_KEYWORDS
        self.assertIn("ml_flags = 130;", generated)
        self.assertIn("PyObject **_args, long long _nargs", generated)
        # The count is a C integer, so asking whether there are too many is a
        # C comparison. It used to build two Python integers and ask the
        # interpreter to compare them, on every call.
        self.assertIn("if (_nargs >", generated)

    def test_a_function_reached_as_a_value_checks_its_own_arity(self):
        """The wrapper had no check at all.

        A call written in the source is checked at build time; one reached
        through a variable has no call site to look at, so too many arguments
        were accepted in silence and a missing required one was passed on as
        NULL.
        """

        self._run(
            "def show(a, b=2, c=3):\n"
            "    return (a, b, c)\n"
            "def via(f):\n"
            "    try:\n"
            "        f(1, 2, 3, 4)\n"
            "    except TypeError as e:\n"
            "        print('too many:', e)\n"
            "    try:\n"
            "        f()\n"
            "    except TypeError as e:\n"
            "        print('too few:', e)\n"
            "    return f(1, c=9)\n"
            "print(via(show))\n",
            b"too many: show() takes from 1 to 3 positional arguments but 4 "
            b"were given\n"
            b"too few: show() missing 1 required positional argument: 'a'\n"
            b"(1, 2, 9)\n",
        )

    def test_pruning_keeps_what_a_package_reaches_by_name(self):
        """A package kept whole needs what *its* modules import, too.

        Only a fraction of a Python installation is ever reached - of nearly
        twelve thousand standard-library files one application touched under
        two hundred - so dropping the rest is the difference between a bundle
        larger than other compilers produce and one smaller. The hazard is
        that a static walk cannot see an import built from a name.

        This is the case that proved it: the codec registry imports
        `encodings.idna` by name, idna imports `stringprep`, and reading only
        `encodings/__init__.py` never mentions either. The pruned bundle
        started, ran, and then failed inside `socket.getfqdn` with "unknown
        encoding: idna".
        """

        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("the C-API path is darwin-arm64 only")
        from py2bin.c_native import compile_c_native
        from py2bin.cli import _embedded_python_path
        from py2bin.freezer import (
            compile_bundle_sources,
            embed_cpython_in_app,
            prune_unreachable,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            # Nothing here mentions idna or stringprep by name.
            entry.write_text(
                "import socket\n"
                "print('hello'.encode('idna'))\n"
                "print(type(socket.getfqdn()) is str)\n",
                encoding="utf-8",
            )
            source = root / "program.c"
            source.write_text(
                python_to_capi_c(entry.read_text(), str(entry)),
                encoding="utf-8",
                newline="\n",
            )
            bundle = root / "Program.app"
            compile_c_native(
                source,
                bundle,
                target="darwin-arm64",
                clean=True,
                app=True,
                app_name="Program",
                python_dylib=_embedded_python_path(),
            )
            embed_cpython_in_app(bundle)
            freed = prune_unreachable(bundle, entry)
            self.assertGreater(freed, 1_000_000, "pruning did nothing")
            compile_bundle_sources(bundle)
            environment = {
                key: value
                for key, value in os.environ.items()
                if key != "PYTHONPATH"
            }
            ran = subprocess.run(
                [str(bundle / "Contents" / "MacOS" / "Program")],
                capture_output=True,
                cwd="/",
                env=environment,
            )
            self.assertEqual(ran.returncode, 0, ran.stderr.decode()[-400:])
            self.assertEqual(ran.stdout, b"b'hello'\nTrue\n")

    def test_the_builtins_are_fetched_once(self):
        """About fifty distinct names, fetched thousands of times.

        Every `None`, `True`, `type` and `str` the emitter reaches for was a
        lookup by name in the builtins dictionary - a hash and a probe, to find
        something that cannot move. They are fetched once into file-scope slots
        at start-up, and a use is an increment on a slot already in hand.
        """

        generated = python_to_capi_c(
            "xs = [1, 2]\n"
            "print([str(x) for x in xs], tuple(xs), x is None if xs else True)\n",
            "program.py",
        )
        self.assertIn("_py2bin_b0", generated)
        # The slot is filled once, at start-up, not at each use.
        self.assertEqual(generated.count('_py2bin_b0 = PyObject_GetAttrString'), 1)
        self._run(
            "xs = [1, 2, 3]\n"
            "print([str(x) for x in xs])\n"
            "print(tuple(xs), sorted(xs, reverse=True), None is None, bool(xs))\n"
            "print(len(xs), max(xs), abs(-2), type(xs).__name__)\n",
            b"['1', '2', '3']\n(1, 2, 3) [3, 2, 1] True True\n3 3 2 list\n",
        )

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


class MachineIntegerTests(unittest.TestCase):
    """Integers held in a register rather than on the heap.

    The tier's reason for being slower than the interpreter was that it wrote
    one generic C-API call per operation, which is precisely what CPython's
    specialising interpreter learned to skip. These cover the fast path that
    closes that gap, and - more importantly - the run-time checks that make it
    give up rather than answer wrongly.
    """

    # The same build-and-compare helper, without inheriting the whole suite
    # along with it - a subclass would run every one of its tests again.
    _run = CApiEmitTests._run

    def test_an_accumulator_is_added_in_a_register(self):
        source = (
            "def total():\n"
            "    n = 0\n"
            "    for i in range(5):\n"
            "        n = n + i * 2 - 1\n"
            "    return n\n"
            "print(total())\n"
        )
        written = python_to_capi_c(source, "program.py")
        # The accumulator is three C variables, and the loop advances a
        # counter rather than asking an iterator for an object.
        self.assertIn("long long n_n", written)
        self.assertIn("int s_n", written)
        self.assertIn("n_n = ", written)
        self._run(source, b"15\n")

    def test_overflow_falls_back_to_unbounded_integers(self):
        # Python's integers do not stop at 64 bits. The fast path has to
        # notice that it left the word and hand the work back, or a program
        # that counts past 2**63 answers something no Python would.
        self._run(
            "def big():\n"
            "    n = 1\n"
            "    for i in range(200):\n"
            "        n = n * 3\n"
            "    return n\n"
            "print(big())\n",
            str(3 ** 200).encode() + b"\n",
        )

    def test_a_wide_accumulator_comes_back_from_the_slow_path(self):
        self._run(
            "def mixed():\n"
            "    n = 2 ** 70\n"
            "    n = n + 1\n"
            "    n = n - 2 ** 70\n"
            "    return n\n"
            "print(mixed())\n",
            b"1\n",
        )

    def test_a_range_wider_than_the_word_still_iterates(self):
        # `range(2**70, 2**70 + 3)` is ordinary Python. The counter cannot
        # hold those bounds, so the loop takes the iterator protocol instead.
        self._run(
            "def wide():\n"
            "    out = 0\n"
            "    for i in range(2 ** 70, 2 ** 70 + 3):\n"
            "        out = out + (i - 2 ** 70)\n"
            "    return out\n"
            "print(wide())\n",
            b"3\n",
        )

    def test_a_name_that_stops_holding_an_integer_still_works(self):
        # The same name holds an int on one pass and a string on the next.
        # Nothing about the representation forbids that; the arithmetic simply
        # takes the long way round when the flag says it is an object.
        self._run(
            "def swing(flip):\n"
            "    n = 1\n"
            "    if flip:\n"
            "        n = 'a'\n"
            "    return n + n\n"
            "print(swing(0), swing(1))\n",
            b"2 aa\n",
        )

    def test_a_counted_loop_is_declined_when_range_is_rebound(self):
        source = (
            "def range(n):\n"
            "    return [10, 20]\n"
            "def use():\n"
            "    out = 0\n"
            "    for i in range(2):\n"
            "        out = out + i\n"
            "    return out\n"
            "print(use())\n"
        )
        # The counter is what would call it; the extern declaration is
        # always there, so the call is what the test has to look for.
        self.assertNotIn("= PyLong_AsLongLong(", python_to_capi_c(source, "p.py"))
        self._run(source, b"30\n")

    def test_a_negative_step_counts_down(self):
        self._run(
            "def down():\n"
            "    out = 0\n"
            "    for i in range(5, 0, -1):\n"
            "        out = out * 10 + i\n"
            "    return out\n"
            "print(down())\n",
            b"54321\n",
        )

    def test_an_empty_range_runs_its_else(self):
        self._run(
            "def none():\n"
            "    seen = 0\n"
            "    for i in range(0):\n"
            "        seen = seen + 1\n"
            "    else:\n"
            "        seen = seen + 100\n"
            "    return seen\n"
            "print(none())\n",
            b"100\n",
        )

    def test_break_leaves_a_counted_loop(self):
        self._run(
            "def stop():\n"
            "    seen = 0\n"
            "    for i in range(100):\n"
            "        if i == 4:\n"
            "            break\n"
            "        seen = seen + i\n"
            "    else:\n"
            "        seen = seen + 1000\n"
            "    return seen\n"
            "print(stop())\n",
            b"6\n",
        )

    def test_a_step_of_zero_raises_the_way_range_does(self):
        source = (
            "def bad():\n"
            "    for i in range(0, 5, 0):\n"
            "        print(i)\n"
            "try:\n"
            "    bad()\n"
            "except ValueError as error:\n"
            "    print(error)\n"
        )
        self._run(source, b"range() arg 3 must not be zero\n")

    def test_a_comparison_of_two_integers_needs_no_object(self):
        source = (
            "def count():\n"
            "    i = 0\n"
            "    hits = 0\n"
            "    while i < 20:\n"
            "        if i > 15:\n"
            "            hits = hits + 100\n"
            "        i = i + 1\n"
            "    return hits\n"
            "print(count())\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertIn("n_i < 20", written)
        self._run(source, b"400\n")

    def test_a_comparison_against_a_wide_value_is_still_right(self):
        self._run(
            "def wide():\n"
            "    n = 2 ** 70\n"
            "    i = 5\n"
            "    return [i < n, n < i, i == 5]\n"
            "print(wide())\n",
            b"[True, False, True]\n",
        )

    def test_an_unbound_local_is_still_refused(self):
        source = (
            "def early():\n"
            "    if 0:\n"
            "        n = 1\n"
            "    return n + 1\n"
            "try:\n"
            "    early()\n"
            "except UnboundLocalError as error:\n"
            "    print('caught')\n"
        )
        self._run(source, b"caught\n")

    def test_the_loop_variable_survives_the_loop(self):
        self._run(
            "def after():\n"
            "    for i in range(4):\n"
            "        pass\n"
            "    return i\n"
            "print(after())\n",
            b"3\n",
        )

    def test_a_counted_loop_variable_can_be_reassigned_inside(self):
        # Rebinding the target does not disturb the counter: Python's `for`
        # takes the next value from the sequence, not from the name.
        self._run(
            "def clobber():\n"
            "    seen = 0\n"
            "    for i in range(4):\n"
            "        seen = seen + i\n"
            "        i = 100\n"
            "    return seen\n"
            "print(clobber())\n",
            b"6\n",
        )

    def test_floor_division_floors_the_way_python_does(self):
        # C truncates toward zero and takes the remainder's sign from the
        # dividend; Python floors and takes it from the divisor. Every sign
        # combination, held against the interpreter.
        self._run(
            "def f():\n"
            "    out = []\n"
            "    for a in range(-9, 10):\n"
            "        for b in range(-4, 5):\n"
            "            if b == 0:\n"
            "                continue\n"
            "            out.append((a // b, a % b))\n"
            "    return out\n"
            "print(f() == [(a // b, a % b)\n"
            "              for a in range(-9, 10)\n"
            "              for b in range(-4, 5) if b != 0])\n",
            b"True\n",
        )

    def test_dividing_by_zero_raises_where_it_should(self):
        self._run(
            "def f():\n"
            "    n = 3\n"
            "    d = 0\n"
            "    try:\n"
            "        return n // d\n"
            "    except ZeroDivisionError as error:\n"
            "        return str(error)\n"
            "print(f())\n",
            b"division by zero\n",
        )

    def test_a_bitwise_operation_on_negative_values_agrees(self):
        self._run(
            "def f():\n"
            "    out = []\n"
            "    for a in range(-4, 5):\n"
            "        for b in range(-4, 5):\n"
            "            out.append((a & b, a | b, a ^ b))\n"
            "    return out\n"
            "print(f() == [(a & b, a | b, a ^ b)\n"
            "              for a in range(-4, 5) for b in range(-4, 5)])\n",
            b"True\n",
        )

    def test_a_bool_is_not_quietly_turned_into_an_integer(self):
        # `True` is an `int` as far as `isinstance` is concerned, and a fast
        # path that narrowed it would hand back `1` where Python says `True`.
        self._run(
            "def f():\n"
            "    flag = True\n"
            "    n = 2\n"
            "    return [flag + n, flag, flag is True]\n"
            "print(f())\n",
            b"[3, True, True]\n",
        )


class CApiCommandLineTests(unittest.TestCase):
    """What `compile-capi` refuses, and how it says so."""

    def test_embedding_the_interpreter_needs_an_app_to_put_it_in(self):
        """A refusal a user can act on, not a traceback.

        `main` has no parser in scope, so reaching for one to report this
        turned the refusal into a NameError - which tells a reader nothing
        about the flag they passed.
        """

        from py2bin.cli import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "p.py").write_text("print(1)\n", encoding="utf-8")
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                code = main(
                    [
                        "compile-capi", str(root / "p.py"),
                        "--target", "darwin-arm64",
                        "--embed-python",
                        "-o", str(root / "p.bin"),
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("--embed-python needs --app", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())


class RelativeImportTests(unittest.TestCase):
    """`from . import x`, with the dots counted at compile time.

    A relative import is relative to where the importing module *is*, and that
    is settled once it has been compiled - so the dots need not be carried into
    the binary. Resolving them here also means no new entry point: asking the
    interpreter would need `PyImport_ImportModuleLevelObject` and a globals
    mapping to read `__package__` from, where spelling the answer out needs
    only the `PyImport_ImportModule` every absolute import already uses.
    """

    def _imports(self, name: str, origin: str, source: str) -> set[str]:
        emitter = CApiEmitter(Path("/x/main.py"))
        written = emitter.program(
            [
                (name, ast.parse(source), origin),
                ("__main__", ast.parse("pass\n"), "/x/main.py"),
            ]
        )
        return set(re.findall(r'PyImport_ImportModule\("([^"]+)"\)', written))

    def test_one_dot_is_the_package_the_module_is_in(self):
        found = self._imports("pkg.sub.mod", "/x/pkg/sub/mod.py", "from . import helper\n")
        self.assertIn("pkg.sub", found)
        # `helper` is read off the package, and imported as a submodule if it
        # is not an attribute - which is what Python does here too.
        self.assertIn("pkg.sub.helper", found)

    def test_two_dots_go_one_package_up(self):
        found = self._imports("pkg.sub.mod", "/x/pkg/sub/mod.py", "from .. import top\n")
        self.assertIn("pkg", found)
        self.assertNotIn("pkg.sub", found)

    def test_a_named_module_is_joined_to_the_resolved_package(self):
        found = self._imports("pkg.sub.mod", "/x/pkg/sub/mod.py", "from .thing import a\n")
        self.assertIn("pkg.sub.thing", found)

    def test_a_packages_own_init_counts_from_itself(self):
        # `__package__` for `pkg/sub/__init__.py` is `pkg.sub`, not `pkg`, so
        # one dot means the package itself rather than its parent.
        found = self._imports("pkg.sub", "/x/pkg/sub/__init__.py", "from . import inner\n")
        self.assertIn("pkg.sub", found)
        self.assertIn("pkg.sub.inner", found)

    def test_a_relative_import_in_a_script_is_refused_as_python_refuses_it(self):
        with self.assertRaises(CApiEmitError) as caught:
            self._imports("top", "/x/top.py", "from . import nope\n")
        self.assertIn("no known parent package", str(caught.exception))

    def test_climbing_past_the_top_is_refused(self):
        with self.assertRaises(CApiEmitError) as caught:
            self._imports("pkg.mod", "/x/pkg/mod.py", "from ... import far\n")
        self.assertIn("beyond top-level package", str(caught.exception))

    def test_an_absolute_import_is_unchanged(self):
        found = self._imports("pkg.mod", "/x/pkg/mod.py", "from json import loads\n")
        self.assertIn("json", found)


class NewlyTranslatedSyntaxTests(unittest.TestCase):
    """Constructs the tier used to refuse, each held against CPython."""

    _run = CApiEmitTests._run

    def test_the_walrus_binds_and_is_the_value(self):
        self._run(
            "if (n := 5) > 1:\n"
            "    print(n)\n"
            "print([y for x in [1, 2, 3] if (y := x * 2) > 2])\n",
            b"5\n[4, 6]\n",
        )

    def test_raise_from_attaches_the_cause(self):
        # `__cause__` has a setter that also sets `__suppress_context__`, so
        # assigning it is the whole of what `from` means.
        self._run(
            "try:\n"
            "    raise ValueError('outer') from KeyError('inner')\n"
            "except ValueError as error:\n"
            "    print(type(error.__cause__).__name__, error.__cause__)\n",
            b"KeyError 'inner'\n",
        )

    def test_raise_from_instantiates_a_class_first(self):
        # `raise TypeError from X` names a class, and a class has nowhere to
        # keep a cause - Python instantiates before attaching.
        self._run(
            "try:\n"
            "    raise TypeError from RuntimeError('r')\n"
            "except TypeError as error:\n"
            "    print(type(error).__name__, type(error.__cause__).__name__)\n",
            b"TypeError RuntimeError\n",
        )

    def test_a_star_takes_what_the_names_leave(self):
        self._run(
            "a, *b = [1, 2, 3, 4]\n"
            "*c, d = (1, 2, 3)\n"
            "e, *f, g = [1, 2, 3, 4, 5]\n"
            "h, *i = [9]\n"
            "print(a, b, c, d, e, f, g, h, i)\n",
            b"1 [2, 3, 4] [1, 2] 3 1 [2, 3, 4] 5 9 []\n",
        )

    def test_a_star_over_a_string_and_in_a_for_target(self):
        self._run(
            "j, *k, m = 'xyz'\n"
            "print(j, k, m)\n"
            "for p, *q in [[1, 2, 3], [4, 5]]:\n"
            "    print(p, q)\n",
            b"x ['y'] z\n1 [2, 3]\n4 [5]\n",
        )

    def test_a_star_says_how_many_it_needed(self):
        # The only unpacking whose failure is one-sided: there can never be
        # too many values for a star, only too few for the fixed names.
        self._run(
            "try:\n"
            "    z, y, *x = [1]\n"
            "except ValueError as error:\n"
            "    print(error)\n",
            b"not enough values to unpack (expected at least 2, got 1)\n",
        )

    def test_match_takes_the_first_case_that_fits(self):
        self._run(
            "def classify(v):\n"
            "    match v:\n"
            "        case 0:\n"
            "            return 'zero'\n"
            "        case 1 | 2 | 3:\n"
            "            return 'small'\n"
            "        case None:\n"
            "            return 'none'\n"
            "        case [a, b]:\n"
            "            return f'pair {a},{b}'\n"
            "        case [x] if x > 100:\n"
            "            return f'big single {x}'\n"
            "        case [x]:\n"
            "            return f'single {x}'\n"
            "        case n:\n"
            "            return f'other {n}'\n"
            "for v in (0, 2, None, [7, 8], [500], [4], 'hi'):\n"
            "    print(classify(v))\n",
            b"zero\nsmall\nnone\npair 7,8\nbig single 500\nsingle 4\nother hi\n",
        )

    def test_a_sequence_pattern_needs_the_right_length(self):
        self._run(
            "def f(v):\n"
            "    match v:\n"
            "        case [a, b]:\n"
            "            return 'two'\n"
            "        case _:\n"
            "            return 'not two'\n"
            "print(f([1, 2]), f([1]), f([1, 2, 3]), f('ab'))\n",
            b"two not two not two not two\n",
        )

    def test_a_sequence_pattern_is_built_without_the_variadic_entry_point(self):
        # PyTuple_Pack is variadic, and Apple's arm64 ABI puts variadic
        # arguments on the stack where this backend passes registers. Calling
        # it with a fixed prototype segfaults rather than answering wrongly.
        written = python_to_capi_c(
            "match [1]:\n    case [a]:\n        pass\n", "program.py"
        )
        self.assertNotIn("= PyTuple_Pack(", written)

    def test_mapping_and_class_patterns_translate(self):
        self._run(
            "class Point:\n"
            "    __match_args__ = ('x', 'y')\n"
            "    def __init__(self, x, y):\n"
            "        self.x = x\n"
            "        self.y = y\n"
            "def describe(v):\n"
            "    match v:\n"
            "        case {'kind': 'circle', 'r': r}:\n"
            "            return f'circle r={r}'\n"
            "        case {'kind': k, **rest}:\n"
            "            return f'{k} plus {sorted(rest)}'\n"
            "        case Point(0, 0):\n"
            "            return 'origin'\n"
            "        case Point(x=0, y=y):\n"
            "            return f'y axis {y}'\n"
            "        case Point(a, b):\n"
            "            return f'point {a},{b}'\n"
            "        case _:\n"
            "            return 'unknown'\n"
            "print(describe({'kind': 'circle', 'r': 3}))\n"
            "print(describe({'kind': 'box', 'w': 1}))\n"
            "print(describe(Point(0, 0)), describe(Point(0, 7)))\n"
            "print(describe(Point(2, 3)), describe(42))\n",
            b"circle r=3\nbox plus ['w']\norigin y axis 7\npoint 2,3 unknown\n",
        )

    def test_a_star_in_a_sequence_pattern_takes_the_middle(self):
        self._run(
            "def g(v):\n"
            "    match v:\n"
            "        case [first, *middle, last]:\n"
            "            return f'{first}..{middle}..{last}'\n"
            "        case [a, *rest]:\n"
            "            return f'head {a} rest {rest}'\n"
            "        case _:\n"
            "            return 'no'\n"
            "print(g([1, 2, 3, 4]), g([9]), g([]), g('x'))\n",
            b"1..[2, 3]..4 head 9 rest [] no no\n",
        )

    def test_two_stars_in_one_pattern_are_refused(self):
        with self.assertRaises(CApiEmitError) as caught:
            python_to_capi_c(
                "match p:\n    case [*a, *b]:\n        pass\n", "program.py"
            )
        self.assertIn("two starred names", str(caught.exception))


class GeneratorTests(unittest.TestCase):
    """`yield`, by turning the function inside out.

    A compiled C function cannot stop in the middle of itself: it has one entry
    and its locals die with its frame. So the body is cut into blocks at each
    `yield`, the blocks are numbered, and the function becomes a class whose
    `__next__` dispatches on which block to run next - with the locals as
    attributes, because they have to outlive a `return`. The class is compiled
    by the machinery that already compiles classes; no new C, no new entry
    point, and nothing interpreted at run time.
    """

    _run = CApiEmitTests._run

    def test_a_while_loop_generator(self):
        self._run(
            "def counter(limit):\n"
            "    n = 0\n"
            "    while n < limit:\n"
            "        yield n\n"
            "        n = n + 1\n"
            "print(list(counter(5)), sum(counter(10)))\n",
            b"[0, 1, 2, 3, 4] 45\n",
        )

    def test_a_for_loop_generator_and_straight_line_yields(self):
        self._run(
            "def squares(xs):\n"
            "    for x in xs:\n"
            "        yield x * x\n"
            "def three():\n"
            "    yield 1\n"
            "    yield 2\n"
            "    yield 3\n"
            "print(list(squares([1, 2, 3])), list(three()))\n",
            b"[1, 4, 9] [1, 2, 3]\n",
        )

    def test_break_and_continue_go_to_the_right_loop(self):
        # Once the loop is cut into blocks it is no longer a loop; what runs is
        # the dispatch loop, and a `break` left as written would leave that -
        # ending the generator instead of the loop.
        self._run(
            "def upto(n):\n"
            "    i = 0\n"
            "    while True:\n"
            "        if i >= n:\n"
            "            break\n"
            "        yield i\n"
            "        i = i + 1\n"
            "def evens(xs):\n"
            "    for x in xs:\n"
            "        if x % 2:\n"
            "            continue\n"
            "        yield x\n"
            "print(list(upto(4)), list(evens([1, 2, 3, 4, 5, 6])))\n",
            b"[0, 1, 2, 3] [2, 4, 6]\n",
        )

    def test_an_early_return_ends_the_iteration(self):
        # A bare `return` left as written becomes `return None` out of
        # `__next__`, which the protocol reads as *yielding* None, forever.
        self._run(
            "def early(xs):\n"
            "    for x in xs:\n"
            "        if x == 3:\n"
            "            return\n"
            "        yield x\n"
            "print(list(early([1, 2, 3, 4])))\n",
            b"[1, 2]\n",
        )

    def test_nested_loops_and_branches(self):
        self._run(
            "def pairs(n):\n"
            "    for i in range(n):\n"
            "        for j in range(i):\n"
            "            yield (i, j)\n"
            "def branchy(n):\n"
            "    if n > 0:\n"
            "        yield 'pos'\n"
            "    else:\n"
            "        yield 'neg'\n"
            "    yield 'done'\n"
            "print(list(pairs(4)))\n"
            "print(list(branchy(1)), list(branchy(-1)))\n",
            b"[(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]\n"
            b"['pos', 'done'] ['neg', 'done']\n",
        )

    def test_a_generator_that_never_yields_is_empty(self):
        self._run(
            "def empty():\n"
            "    if False:\n"
            "        yield 1\n"
            "print(list(empty()))\n",
            b"[]\n",
        )

    def test_it_is_a_real_iterator(self):
        self._run(
            "def three():\n"
            "    yield 1\n"
            "    yield 2\n"
            "    yield 3\n"
            "g = three()\n"
            "print(next(g), next(g), [x for x in three()], iter(g) is g)\n",
            b"1 2 [1, 2, 3] True\n",
        )

    def test_the_shapes_it_cannot_express_are_refused_by_name(self):
        for source, needle in (
            ("def f(*xs):\n    yield 1\n", "*args"),
        ):
            with self.subTest(source=source):
                with self.assertRaises(CApiEmitError) as caught:
                    python_to_capi_c(source, "program.py")
                self.assertIn(needle, str(caught.exception))


class GeneratorDelegationTests(unittest.TestCase):
    """`yield from`, and a `yield` that receives what `send` puts in."""

    _run = CApiEmitTests._run

    def test_yield_from_delegates_iteration(self):
        # Written as the loop it is, before the body is cut into blocks.
        self._run(
            "def inner():\n"
            "    yield 1\n"
            "    yield 2\n"
            "def outer():\n"
            "    yield 0\n"
            "    yield from inner()\n"
            "    yield from [7, 8]\n"
            "    yield 9\n"
            "print(list(outer()))\n",
            b"[0, 1, 2, 7, 8, 9]\n",
        )

    def test_send_reaches_a_yield_used_as_a_value(self):
        self._run(
            "def echo():\n"
            "    while True:\n"
            "        got = yield\n"
            "        if got is None:\n"
            "            return\n"
            "        yield got * 2\n"
            "g = echo()\n"
            "next(g)\n"
            "print(g.send(5))\n",
            b"10\n",
        )

    def test_next_is_send_of_none(self):
        self._run(
            "def taking():\n"
            "    got = yield 'first'\n"
            "    yield got\n"
            "g = taking()\n"
            "print(next(g), next(g))\n",
            b"first None\n",
        )

    def test_a_delegation_in_an_expression_it_cannot_lift_is_refused(self):
        # Lifting it out of an `and` would run it whether or not the first
        # operand was true.
        with self.assertRaises(CApiEmitError) as caught:
            python_to_capi_c(
                "def f(c):\n    x = c and (yield from g())\n    yield x\n",
                "program.py",
            )
        self.assertIn("`and`", str(caught.exception))


class GeneratorHandlerTests(unittest.TestCase):
    """`try`/`except` around a `yield`, which the handler has to survive.

    The difficulty is that the generator stops inside the `try` and the handler
    has to still be there when it starts again. It is, because an exception can
    only be raised while a *block* is running: each block of the guarded region
    carries the handler, so it is re-established on every entry rather than
    having to persist across one.
    """

    _run = CApiEmitTests._run

    def test_a_handler_survives_the_suspension(self):
        self._run(
            "def guarded(xs):\n"
            "    for x in xs:\n"
            "        try:\n"
            "            if x == 0:\n"
            "                raise ValueError('zero')\n"
            "            yield 10 // x\n"
            "        except ValueError as error:\n"
            "            yield f'caught {error}'\n"
            "print(list(guarded([1, 0, 2])))\n",
            b"[10, 'caught zero', 5]\n",
        )

    def test_an_exception_raised_after_a_yield_is_caught(self):
        self._run(
            "def after():\n"
            "    try:\n"
            "        yield 'before'\n"
            "        raise KeyError('k')\n"
            "    except KeyError:\n"
            "        yield 'handled'\n"
            "    yield 'done'\n"
            "print(list(after()))\n",
            b"['before', 'handled', 'done']\n",
        )

    def test_nested_handlers_catch_innermost_first(self):
        self._run(
            "def nested():\n"
            "    try:\n"
            "        try:\n"
            "            yield 1\n"
            "            raise ValueError('inner')\n"
            "        except ValueError:\n"
            "            yield 2\n"
            "            raise TypeError('outer')\n"
            "    except TypeError:\n"
            "        yield 3\n"
            "print(list(nested()))\n",
            b"[1, 2, 3]\n",
        )

    def test_several_clauses_a_tuple_and_a_bare_except(self):
        self._run(
            "def many(kind):\n"
            "    try:\n"
            "        if kind == 'v':\n"
            "            raise ValueError('v')\n"
            "        if kind == 'k':\n"
            "            raise KeyError('k')\n"
            "        yield 'none raised'\n"
            "    except ValueError:\n"
            "        yield 'value'\n"
            "    except (KeyError, IndexError):\n"
            "        yield 'key or index'\n"
            "    yield 'end'\n"
            "def bare():\n"
            "    try:\n"
            "        yield 1\n"
            "        raise RuntimeError('boom')\n"
            "    except:\n"
            "        yield 2\n"
            "print(list(many('v')), list(many('k')), list(many('n')))\n"
            "print(list(bare()))\n",
            b"['value', 'end'] ['key or index', 'end'] ['none raised', 'end']\n"
            b"[1, 2]\n",
        )

    def test_what_no_clause_matches_still_propagates(self):
        self._run(
            "def escapes():\n"
            "    try:\n"
            "        yield 1\n"
            "        raise TypeError('not caught here')\n"
            "    except ValueError:\n"
            "        yield 'wrong'\n"
            "try:\n"
            "    print(list(escapes()))\n"
            "except TypeError as error:\n"
            "    print('propagated:', error)\n",
            b"propagated: not caught here\n",
        )

    def test_a_with_around_a_yield_compiles(self):
        # Expanded into the try it already stands for, then cut into blocks by
        # the same path a written-out try takes.
        python_to_capi_c(
            "def f(c):\n    with c:\n        yield 1\n", "program.py"
        )

    def test_a_return_inside_a_finally_region_still_runs_the_cleanup(self):
        # An earlier pass turns the return into a jump that leaves by the
        # ordinary exit, which is the one that passes the cleanup.
        python_to_capi_c(
            "def f():\n    try:\n        yield 1\n        return\n"
            "    finally:\n        pass\n",
            "program.py",
        )

    def test_a_finally_around_a_yield_compiles(self):
        # The object this produces is a class with __next__, not a generator,
        # so it is never closed and never finalised by the collector. The only
        # ways out of the region are the ones the rewriter can see: running off
        # the end, and an exception on its way past.
        python_to_capi_c(
            "def f():\n    try:\n        yield 1\n    finally:\n        pass\n",
            "program.py",
        )

    def test_what_a_finally_still_refuses_says_why(self):
        for source, needle in (
            # Suspending while unwinding needs the cleanup cut into blocks too.
            ("def f():\n    try:\n        yield 1\n    finally:\n        yield 2\n",
             "itself contains"),
            # These leave without passing the block holding the cleanup.
            # A break goes round the block holding the cleanup.
            ("def f():\n    for i in (1,):\n        try:\n            yield i\n"
             "            break\n        finally:\n            pass\n", "break"),
        ):
            with self.subTest(source=source):
                with self.assertRaises(CApiEmitError) as caught:
                    python_to_capi_c(source, "program.py")
                self.assertIn(needle, str(caught.exception))


class AsyncTests(unittest.TestCase):
    """`async def` and `await`, which are the generator machine renamed.

    Awaiting an object with `__await__` means delegating to the iterator it
    answers with - so a coroutine is a state machine that also says
    `__await__`, which is what it is in CPython too. `await x` is then PEP
    380's expansion of `yield from x.__await__()`, and the event loop drives
    it through `send` exactly as it drives a real coroutine.
    """

    _run = CApiEmitTests._run

    def test_awaiting_a_compiled_coroutine(self):
        self._run(
            "import asyncio\n"
            "async def inner(n):\n"
            "    return n * 2\n"
            "async def middle(n):\n"
            "    doubled = await inner(n)\n"
            "    return doubled + 1\n"
            "async def main():\n"
            "    return await middle(3) + await inner(10)\n"
            "print(asyncio.run(main()))\n",
            b"27\n",
        )

    def test_a_real_suspension_through_the_event_loop(self):
        # asyncio.sleep actually yields to the loop, so this only passes if
        # the state machine suspends and resumes the way a coroutine does.
        self._run(
            "import asyncio\n"
            "async def slow(name, delay):\n"
            "    await asyncio.sleep(delay)\n"
            "    return name\n"
            "async def sequential():\n"
            "    first = await slow('a', 0.01)\n"
            "    second = await slow('b', 0.01)\n"
            "    return [first, second]\n"
            "print(asyncio.run(sequential()))\n",
            b"['a', 'b']\n",
        )

    def test_gather_runs_compiled_coroutines_together(self):
        self._run(
            "import asyncio\n"
            "async def slow(name, delay):\n"
            "    await asyncio.sleep(delay)\n"
            "    return name\n"
            "async def together():\n"
            "    return await asyncio.gather(slow('x', 0.02), slow('y', 0.01))\n"
            "print(asyncio.run(together()))\n",
            b"['x', 'y']\n",
        )

    def test_a_handler_inside_a_coroutine(self):
        self._run(
            "import asyncio\n"
            "async def guarded():\n"
            "    try:\n"
            "        await asyncio.sleep(0.01)\n"
            "        raise ValueError('inside')\n"
            "    except ValueError as error:\n"
            "        return f'caught {error}'\n"
            "print(asyncio.run(guarded()))\n",
            b"caught inside\n",
        )

    def test_awaits_that_cannot_be_lifted_are_refused(self):
        # Lifting an await out of a conditional would run it unconditionally,
        # and out of a comprehension would run it once instead of many times.
        for source, needle in (
            ("import asyncio\nasync def f(c):\n    return c and await g()\n", "`and`"),
            ("import asyncio\nasync def f(c):\n    return await g() if c else 1\n", "arm"),
            ("import asyncio\nasync def f(xs):\n    return [await g(x) for x in xs]\n", "comprehension"),
        ):
            with self.subTest(source=source):
                with self.assertRaises(CApiEmitError) as caught:
                    python_to_capi_c(source, "program.py")
                self.assertIn(needle, str(caught.exception))


class DelegationReturnTests(unittest.TestCase):
    """`yield from` written out as PEP 380 defines it."""

    _run = CApiEmitTests._run

    def test_a_sub_generators_return_value_is_the_expressions_value(self):
        self._run(
            "def sub():\n"
            "    got = yield 'sub asks'\n"
            "    yield f'sub got {got}'\n"
            "    return 'sub returned'\n"
            "def outer():\n"
            "    answer = yield from sub()\n"
            "    yield f'outer saw {answer}'\n"
            "g = outer()\n"
            "print(next(g))\n"
            "print(g.send('a value'))\n"
            "print(next(g))\n",
            b"sub asks\nsub got a value\nouter saw sub returned\n",
        )

    def test_a_generator_may_return_a_value(self):
        self._run(
            "def counted():\n"
            "    yield 1\n"
            "    return 99\n"
            "def uses():\n"
            "    value = yield from counted()\n"
            "    yield value\n"
            "print(list(uses()))\n",
            b"[1, 99]\n",
        )


class EntryNameTests(unittest.TestCase):
    """A program may call one of its own functions `main`.

    The module body compiles to the C entry point, and the renderer told it
    from the program's own functions by the name "main" - so `def main()` was
    dropped and every call to it dangled at the C stage.
    """

    _run = CApiEmitTests._run

    def test_a_function_called_main_is_compiled(self):
        self._run("def main():\n    return 7\nprint(main())\n", b"7\n")


class CrashLogTests(unittest.TestCase):
    """A windowed application has no console, so a traceback needs a file."""

    def test_a_crash_log_program_writes_the_traceback_beside_itself(self):
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("the C-API path is darwin-arm64 only on this host")
        from py2bin.capi_emit import python_program_to_capi_c

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text("raise ValueError('deliberate')\n", encoding="utf-8")
            written = root / "program.c"
            generated, _linked = python_program_to_capi_c(entry, crash_log=True)
            written.write_text(generated, encoding="utf-8")
            binary = root / "program.bin"
            compile_c_native(written, binary, target="darwin-arm64", clean=True)
            subprocess.run([str(binary)], capture_output=True, cwd=directory)
            report = root / "crash.txt"
            self.assertTrue(report.is_file(), "no crash.txt was written")
            text = report.read_text(encoding="utf-8")
        self.assertIn("ValueError: deliberate", text)
        self.assertIn("argv:", text)

    def test_without_the_flag_nothing_is_written(self):
        written = python_to_capi_c("raise ValueError('x')\n", "program.py")
        self.assertNotIn("_py2bin_crash_report", written)


class StorageAgreementTests(unittest.TestCase):
    """A name has one storage, and every reference must use it.

    A name the register analysis picked out is three C variables; a name whose
    storage is the module's is one. Deciding that separately at the binding and
    at the read let a `for` loop write `g_arg` while the body read `v_arg` -
    a NULL that reached PySequence_Contains and segfaulted a real application,
    with no exception and so no traceback to go on.
    """

    _run = CApiEmitTests._run

    def test_a_for_target_is_read_from_where_the_loop_wrote_it(self):
        self._run(
            "cmd = ['a b', 'c']\n"
            "for arg in cmd:\n"
            "    if ' ' in arg:\n"
            "        print('space in', arg)\n"
            "    else:\n"
            "        print('no space in', arg)\n",
            b"space in a b\nno space in c\n",
        )

    def test_a_module_level_loop_target_uses_the_module_slot(self):
        written = python_to_capi_c(
            "total = 0\n"
            "for n in [1, 2]:\n"
            "    total = total + n\n"
            "print(total)\n",
            "program.py",
        )
        # One storage for `n`: the module's. Not a register pair as well.
        self.assertIn("g_n = ", written)
        self.assertNotIn("long long n_n", written)

    def test_the_registers_are_still_used_inside_a_function(self):
        # The fix must not switch the optimisation off where it is correct.
        written = python_to_capi_c(
            "def total():\n"
            "    n = 0\n"
            "    for i in range(5):\n"
            "        n = n + i\n"
            "    return n\n"
            "print(total())\n",
            "program.py",
        )
        self.assertIn("long long n_n", written)
        self.assertIn("long long n_i", written)


class AsynchronousIterationTests(unittest.TestCase):
    """`async for` and `async with`, written out as what they stand for.

    Both expand into shapes the machine already cuts into blocks - a call, an
    `await` inside a `try`, a flag - rather than needing the machine to learn
    about the asynchronous protocols directly.
    """

    def test_async_for_compiles(self):
        python_to_capi_c(
            "import asyncio\n"
            "async def f(xs):\n"
            "    total = 0\n"
            "    async for x in xs:\n"
            "        total += x\n"
            "    return total\n",
            "program.py",
        )

    def test_async_with_compiles(self):
        python_to_capi_c(
            "import asyncio\n"
            "async def f(c):\n"
            "    async with c as held:\n"
            "        return held\n",
            "program.py",
        )

    def test_an_async_for_else_says_why_it_is_refused(self):
        with self.assertRaises(CApiEmitError) as caught:
            python_to_capi_c(
                "import asyncio\n"
                "async def f(xs):\n"
                "    async for x in xs:\n"
                "        pass\n"
                "    else:\n"
                "        pass\n",
                "program.py",
            )
        self.assertIn("else", str(caught.exception))


class RaisingAClassTests(unittest.TestCase):
    """`raise SomeError` names a class, and that is not `type(SomeError)`.

    Asking type() for the class of a class answers `type`, the metaclass, and
    handing that to PyErr_SetObject produced a SystemError for the plainest
    raise a Python program can write.
    """

    def test_raising_a_class_compiles(self):
        python_to_capi_c("def f():\n    raise ValueError\n", "program.py")

    def test_raising_an_instance_still_compiles(self):
        python_to_capi_c("def f():\n    raise ValueError('x')\n", "program.py")

    def test_raising_a_class_from_a_generator_compiles(self):
        python_to_capi_c(
            "def g():\n    yield 1\n    raise ValueError\n", "program.py"
        )


class ComprehensionTargetTests(unittest.TestCase):
    """A comprehension clause binds whatever a `for` statement would.

    `for k, v in d.items()` is ordinary Python and was refused inside a
    comprehension while working perfectly as a statement - a gap that turned
    up when a real application was compiled.
    """

    def test_a_pair_target_compiles(self):
        python_to_capi_c(
            "d = {'a': 1}\nprint([f'{k}{v}' for k, v in d.items()])\n",
            "program.py",
        )

    def test_a_dict_comprehension_takes_a_pair(self):
        python_to_capi_c(
            "d = {'a': 1}\nprint({v: k for k, v in d.items()})\n", "program.py"
        )

    def test_a_longer_target_compiles(self):
        python_to_capi_c(
            "rows = [(1, 2, 3)]\nprint([a + b + c for a, b, c in rows])\n",
            "program.py",
        )

    def test_the_names_stay_inside_the_comprehension(self):
        # A comprehension has a scope of its own, so binding k there must not
        # disturb the k outside it.
        c = python_to_capi_c(
            "d = {'a': 1}\nk = 'outer'\nprint([k for k, v in d.items()], k)\n",
            "program.py",
        )
        self.assertIn("PyObject_GetIter", c)

    def test_a_clause_storing_through_a_subscript_keeps_the_outer_name(self):
        # `for d[k] in ...` reads d and k to find where the item goes; it
        # binds neither. Treating them as bound would give the comprehension
        # its own d, shadowing the dictionary being written to.
        python_to_capi_c(
            "d = {}\nk = 'a'\nprint([1 for d[k] in [9]], d)\n", "program.py"
        )
