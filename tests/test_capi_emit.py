"""Python translated into C that drives the CPython C API.

The tier Nuitka occupies, reached without Nuitka's toolchain: py2bin writes
the C and py2bin's own C compiler turns it into machine code, so no clang,
assembler or linker takes part. What it buys over the native tier is Python's
own semantics - most visibly integers that do not stop at 64 bits.
"""

from __future__ import annotations

import platform
import re
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
        self._reject("@dec\ndef f():\n    pass\n", "decorated function")
        self._reject("async def f():\n    pass\n", "no C-API translation")

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
