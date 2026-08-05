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



def _body_of(generated: str, name: str) -> str:
    """The body of one generated C function, found by its definition.

    Every function is forward-declared before it is defined, so searching for
    the name alone finds the declaration and slices from there - which is a
    different function's body, and an assertion about it means nothing.
    """

    marker = f"{name}(" 
    start = generated.index(marker, generated.index(marker) + 1)
    while generated[start:].split("\n", 1)[0].rstrip().endswith(";"):
        start = generated.index(marker, start + 1)
    return generated[start : generated.index("\n}\n", start)]

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

    def test_a_module_function_rebound_later_is_called_through_the_name(self):
        # `def f` gave the name a direct C call keyed on the spelling alone,
        # so the rebinding below was never consulted: this answered 2 twice
        # where Python answers 2 then 101. Same class as the `len`/`str`
        # shortcuts - a shortcut keyed on a name must first check the program
        # has not bound that name somewhere else.
        self._run(
            "def f(a): return a + 1\n"
            "print(f(1))\n"
            "f = lambda a: a + 100\n"
            "print(f(1))\n",
            b"2\n101\n",
        )

    def test_a_function_decorated_by_reassignment_is_not_bypassed(self):
        # The idiom the case above exists for. `greet = trace(greet)` is how a
        # decorator is spelled without the `@`, and a direct call would run the
        # undecorated body forever.
        self._run(
            "def greet(name): return 'hi ' + name\n"
            "def trace(inner):\n"
            "    def wrapped(name): return '[' + inner(name) + ']'\n"
            "    return wrapped\n"
            "greet = trace(greet)\n"
            "print(greet('you'))\n",
            b"[hi you]\n",
        )

    def test_calling_a_function_above_its_def_raises_name_error(self):
        # The `def` binds the name when it runs, not when the file is read.
        # The callable used to be made at start-up and bound with it, so this
        # answered 21 where Python raises.
        self._run_failing(
            "print(later(3))\ndef later(x): return x * 7\n",
            b"NameError",
        )

    def test_reading_a_global_above_its_assignment_raises_name_error(self):
        # `certain_globals` asked only whether the module binds the name
        # anywhere, not whether it had yet - so the NULL test was skipped and
        # the raw NULL reached `print`, which showed it as `<NULL>`. Anywhere
        # less forgiving than `print` that NULL is a crash.
        self._run_failing("print(y)\ny = 5\n", b"NameError")

    def test_a_class_used_above_its_statement_raises_name_error(self):
        self._run_failing(
            "class C: pass\nprint(D)\nclass D: pass\n", b"NameError"
        )

    def test_a_global_bound_only_in_a_branch_that_did_not_run(self):
        # The pre-existing half of the same test: bound later *and* only
        # conditionally. This one always raised; it is here so the positional
        # rule cannot be "fixed" by dropping the test that motivated it.
        self._run_failing(
            "if False:\n    z = 1\nprint(z)\n", b"NameError"
        )

    def test_a_function_still_reaches_a_global_defined_below_it(self):
        # The rule is positional, not textual: `helper` runs after the whole
        # module body has, so `LIMIT` is bound by then. Refusing this would
        # break the ordinary shape of a Python file.
        self._run(
            "def helper(v): return v + LIMIT\n"
            "LIMIT = 10\n"
            "print(helper(5))\n",
            b"15\n",
        )

    def test_a_function_may_call_one_defined_below_it(self):
        # Same reasoning for calls. `first` is written above `second` and
        # neither runs until the module body reaches the last line.
        self._run(
            "def first(v): return second(v) + 1\n"
            "def second(v): return v * 2\n"
            "print(first(4))\n",
            b"9\n",
        )

    def test_recursion_stays_on_the_direct_path(self):
        # A function's own name is bound by the time its body runs, which is
        # what keeps the positional rule from disabling the direct call for
        # the case it matters most for.
        self._run(
            "def down(n):\n"
            "    if n <= 0: return 0\n"
            "    return n + down(n - 1)\n"
            "print(down(100))\n",
            b"5050\n",
        )

    def test_a_nested_function_can_call_itself(self):
        # A closure captures by value when it is made, and its own name is not
        # bound at that moment - the `def` being compiled is what binds it -
        # so the capture took a NULL and the first recursive call raised
        # `NameError` naming the function it was standing in. The slot is now
        # declared before the closure is built and filled once the callable
        # exists.
        self._run(
            "def go():\n"
            "    def fact(n):\n"
            "        return 1 if n <= 1 else n * fact(n - 1)\n"
            "    return fact(10)\n"
            "print(go())\n",
            b"3628800\n",
        )

    def test_a_nested_function_reading_one_defined_below_is_refused(self):
        # Mutual recursion between two nested functions cannot work under
        # capture-by-value: the second name is simply absent when the first
        # closure is made. It used to fail at run time with a NameError
        # naming a function plainly written above it, which is the worst way
        # to say so.
        self._reject(
            "def go():\n"
            "    def even(n):\n"
            "        return True if n == 0 else odd(n - 1)\n"
            "    def odd(n):\n"
            "        return False if n == 0 else even(n - 1)\n"
            "    return even(10)\n",
            "bound further down the enclosing scope",
        )

    def test_nested_helpers_in_order_still_work(self):
        # The refusal above must not cost the ordinary shape, where the
        # helper is written before what uses it.
        self._run(
            "def go():\n"
            "    def helper(v):\n"
            "        return v + 1\n"
            "    def uses(v):\n"
            "        return helper(v) * 2\n"
            "    return uses(3)\n"
            "print(go())\n",
            b"8\n",
        )

    def test_a_function_that_rebinds_itself_through_global(self):
        # `global a` inside `a` binds the module's `a` from a scope the
        # module-scope walk does not enter, so the direct C call survived and
        # the function went on calling itself after Python would have been
        # calling the replacement.
        self._run(
            "def a(v):\n"
            "    global a\n"
            "    a = lambda x: x * 100\n"
            "    return v + 1\n"
            "print(a(1), a(1))\n",
            b"2 100\n",
        )

    def test_an_f_string_asks_for_format_not_str(self):
        # `f"{x}"` is `format(x, "")`, which is `__format__`. For most types
        # that defers to `str` and the two agree; for a type that defines
        # `__format__` they do not, and this answered `str` for it.
        self._run(
            "class T:\n"
            "    def __str__(self):\n"
            "        return 'STR'\n"
            "    def __repr__(self):\n"
            "        return 'REPR'\n"
            "    def __format__(self, spec):\n"
            "        return 'FMT:' + spec\n"
            "t = T()\n"
            "print(f'{t} {t!r} {t!s} {t:xyz}')\n",
            b"FMT: REPR STR FMT:xyz\n",
        )

    def test_an_exact_string_needs_no_formatting_call_at_all(self):
        # CPython's FORMAT_SIMPLE takes the same two paths: an exact `str` is
        # already its own formatting, so there is nothing to ask.
        generated = python_to_capi_c(
            "def f(name):\n"
            "    greeting = 'hello'\n"
            "    return f'{greeting} there'\n"
            "print(f('x'))\n",
            "program.py",
        )
        body = generated[generated.index("static PyObject *f_f("):]
        body = body[: body.index("\n}\n")]
        self.assertNotIn("PyObject_Format", body)
        self.assertNotIn("PyObject_Str", body)

    def test_concatenating_exact_strings_skips_the_add_dispatch(self):
        # `+` has to go through `PyNumber_Add` because a `str` subclass may
        # override `__add__`. Where both sides are certainly exact there is
        # none to find, and the answer is the same either way.
        self._run(
            "class Loud(str):\n"
            "    def __add__(self, other):\n"
            "        return 'HIJACKED'\n"
            "plain = 'a'\n"
            "print(plain + 'b', Loud('a') + 'b')\n",
            b"ab HIJACKED\n",
        )

    def test_unpacking_checks_its_length_with_a_machine_comparison(self):
        # Deciding whether a two-item tuple has two items used to box the
        # length, box the expected count twice, run two `PyObject_RichCompare`
        # calls and ask `PyObject_IsTrue` of each - eleven C-API calls and five
        # allocations for one comparison. `a, b = pair` ran at 0.18x the
        # interpreter and the check was almost all of it.
        generated = python_to_capi_c(
            "def f(pair):\n    a, b = pair\n    return a + b\nprint(f((1, 2)))\n",
            "program.py",
        )
        body = generated[generated.index("static PyObject *f_f("):]
        body = body[: body.index("\n}\n")]
        self.assertNotIn("PyObject_RichCompare", body)

    def test_unpacking_a_sequence_makes_no_copy_of_it(self):
        # `tuple()` is what makes unpacking work for any iterable, but for a
        # tuple or a list - which is what almost every unpack holds - it
        # allocated a copy per unpack and freed it again.
        generated = python_to_capi_c(
            "def f(pair):\n    a, b = pair\n    return a + b\nprint(f((1, 2)))\n",
            "program.py",
        )
        body = generated[generated.index("static PyObject *f_f("):]
        body = body[: body.index("\n}\n")]
        self.assertIn("PySequence_Check", body)

    def test_unpacking_still_takes_any_iterable(self):
        # The fast path may not narrow what unpacking accepts. A generator has
        # no length and no index and must still come apart.
        self._run(
            "def gen():\n"
            "    yield 1\n"
            "    yield 2\n"
            "a, b = gen()\n"
            "c, d = 'xy'\n"
            "e, f = range(2)\n"
            "print(a, b, c, d, e, f)\n",
            b"1 2 x y 0 1\n",
        )

    def test_unpacking_something_with_no_length_still_works(self):
        # `PySequence_Check` is true for a class defining only `__getitem__`,
        # and such a class has no `__len__` - so the size is tried and a
        # failure sends it back to the general path rather than out of the
        # program.
        self._run(
            "class Weird:\n"
            "    def __getitem__(self, i):\n"
            "        if i > 1:\n"
            "            raise IndexError\n"
            "        return i * 5\n"
            "a, b = Weird()\n"
            "print(a, b)\n",
            b"0 5\n",
        )

    def test_unpacking_the_wrong_length_says_so_the_way_python_does(self):
        self._run_failing(
            "a, b = (1, 2, 3)\n", b"too many values to unpack (expected 2, got 3)"
        )

    def test_unpacking_too_few_says_so_too(self):
        self._run_failing(
            "a, b, c = (1, 2)\n",
            b"not enough values to unpack (expected 3, got 2)",
        )

    def test_a_membership_test_builds_no_boolean(self):
        # `PySequence_Contains` answers 1, 0 or -1 - already the verdict. The
        # condition used to build `True` by looking the name up on the
        # builtins module, then ask `PyObject_IsTrue` what it had built.
        generated = python_to_capi_c(
            "def f(xs):\n    if 5 in xs:\n        return 1\n    return 0\n"
            "print(f([5]))\n",
            "program.py",
        )
        body = generated[generated.index("static PyObject *f_f("):]
        body = body[: body.index("\n}\n")]
        self.assertIn("PySequence_Contains", body)
        self.assertNotIn('GetAttrString(_py2bin_builtins, "True")', body)

    def test_membership_still_answers_with_a_value_where_one_is_wanted(self):
        self._run(
            "xs = [1, 2]\n"
            "print(1 in xs, 9 in xs, 1 not in xs, 9 not in xs)\n",
            b"True False False True\n",
        )

    def test_a_container_that_raises_is_not_read_as_a_verdict(self):
        # -1 is a failure, not a false. Treating it as one would swallow the
        # exception and take the wrong branch.
        self._run_failing(
            "class Boom:\n"
            "    def __contains__(self, value):\n"
            "        raise ValueError('no')\n"
            "if 1 in Boom():\n"
            "    print('unreachable')\n",
            b"ValueError",
        )

    def test_isinstance_goes_straight_to_the_entry_point(self):
        generated = python_to_capi_c(
            "def f(x):\n    if isinstance(x, int):\n        return 1\n    return 0\n"
            "print(f(1))\n",
            "program.py",
        )
        body = generated[generated.index("static PyObject *f_f("):]
        body = body[: body.index("\n}\n")]
        self.assertIn("PyObject_IsInstance", body)

    def test_a_program_with_its_own_isinstance_gets_its_own(self):
        # The bug class this project has hit five times: a shortcut keyed on a
        # name must first check the program has not bound that name.
        self._run(
            "def isinstance(a, b):\n"
            "    return 'MINE'\n"
            "print(isinstance(5, int))\n",
            b"MINE\n",
        )

    def test_isinstance_passed_as_a_value_is_still_the_builtin(self):
        self._run("f = isinstance\nprint(f(5, int))\n", b"True\n")

    def test_a_try_that_does_not_raise_releases_what_it_built(self):
        """The classes a clause catches are built before the body runs.

        They were never released on the path where the body raised nothing,
        so `except (ValueError, TypeError)` - whose class expression builds a
        fresh tuple each time it is evaluated - leaked one tuple per
        execution. Measured by what it costs rather than by reading the C:
        four hundred thousand turns held 40 MB against the interpreter's 15,
        and it grew with the count.
        """

        self._run(
            "import resource\n"
            "def go():\n"
            "    t = 0\n"
            "    for i in range(800000):\n"
            "        try:\n"
            "            t += 1\n"
            "        except (ValueError, TypeError):\n"
            "            pass\n"
            "    return t\n"
            "go()\n"
            "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024\n"
            "print(peak < 35000)\n",
            b"True\n",
        )

    def test_a_matched_clause_releases_the_ones_after_it(self):
        # A clause that matches jumps out, so the classes of every clause
        # after it are never tested and nothing else would release them.
        self._run(
            "import resource\n"
            "def go():\n"
            "    t = 0\n"
            "    for i in range(800000):\n"
            "        try:\n"
            "            raise ValueError('x')\n"
            "        except (ValueError, TypeError):\n"
            "            t += 1\n"
            "        except (KeyError, IndexError):\n"
            "            pass\n"
            "    return t\n"
            "print(go())\n"
            "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024\n"
            "print(peak < 35000)\n",
            b"800000\nTrue\n",
        )

    def test_a_bare_except_still_catches_after_a_tuple_clause(self):
        # Not a leak test: Python requires a bare `except` to be last, so
        # there is never a clause after it whose class would go unreleased.
        # This is here because the first draft asserted otherwise and passed
        # against the unfixed compiler, which is how the impossibility was
        # noticed.
        self._run(
            "def go():\n"
            "    t = 0\n"
            "    for i in range(1000):\n"
            "        try:\n"
            "            raise IndexError\n"
            "        except (ValueError, TypeError):\n"
            "            pass\n"
            "        except:\n"
            "            t += 1\n"
            "    return t\n"
            "print(go())\n",
            b"1000\n",
        )

    def test_an_unexpected_keyword_is_refused(self):
        # `def f(a)` called as `f(1, b=2)` ran and answered where CPython
        # raises. Nothing could tell a keyword that had been taken from one
        # that matched no parameter, because the key was only removed when
        # there was a `**kwargs` to hand the rest to.
        self._run_failing(
            "def f(a):\n    return a\nf(1, b=2)\n",
            b"got an unexpected keyword argument 'b'",
        )

    def test_a_keyword_call_passes_names_beside_the_values(self):
        # A tuple *and* a dict were allocated per call, and the keyword's name
        # was built from a C string every time - two allocations and a string
        # build to pass one argument by name.
        generated = python_to_capi_c(
            "def helper(v, step):\n    return v + step\n"
            "print(helper(1, step=2))\n",
            "program.py",
        )
        # Scoped to the calling function: `PyObject_Call` still has other
        # users - a format specifier, a spread call - and asserting on the
        # whole file failed against the fixed compiler as readily as the
        # broken one, which is no test at all.
        body = generated[generated.index("static PyObject *f__value"):]
        body = body[: body.index("\n}\n")]
        self.assertIn("PyObject_Vectorcall", generated)
        # A tuple of names built once at start-up, not a dict per call.
        self.assertIn("_py2bin_kw0", generated)
        self.assertNotIn("PyDict_New()", body)

    def test_keyword_shapes_still_bind_the_way_python_binds_them(self):
        self._run(
            "def f(a, b=9, c=8):\n"
            "    return (a, b, c)\n"
            "print(f(1), f(1, c=3), f(1, b=2, c=3), f(c=3, a=1))\n",
            b"(1, 9, 8) (1, 9, 3) (1, 2, 3) (1, 9, 3)\n",
        )

    def test_a_starred_signature_still_collects_the_rest(self):
        self._run(
            "def f(a, *rest, k=0, **more):\n"
            "    return (a, rest, k, sorted(more.items()))\n"
            "print(f(1, 2, 3, k=4, z=5))\n",
            b"(1, (2, 3), 4, [('z', 5)])\n",
        )

    def test_a_chained_comparison_in_a_condition_builds_no_object(self):
        # Every link built a `True` or a `False` through `PyObject_RichCompare`
        # and then asked `PyObject_IsTrue` what it had built. Where the
        # operands cost nothing to read twice - a name or a literal - the
        # chain is written out as the `and` Python says it means, so each link
        # picks up the machine comparison a two-sided one would have had.
        generated = python_to_capi_c(
            "def f(n):\n"
            "    t = 0\n"
            "    for i in range(n):\n"
            "        if 0 < i < 99:\n"
            "            t += 1\n"
            "    return t\n"
            "print(f(10))\n",
            "program.py",
        )
        body = generated[generated.index("static PyObject *f_f("):]
        body = body[: body.index("\n}\n")]
        self.assertNotIn("= PyObject_RichCompare(", body)

    def test_a_chain_evaluates_its_middle_operand_once(self):
        # The rewrite is only allowed where an operand costs nothing to read
        # twice. A call does not, so it keeps the slots - and `f()` must run
        # once however many links mention it.
        self._run(
            "def f():\n"
            "    print('called')\n"
            "    return 5\n"
            "print(1 < f() < 10)\n",
            b"called\nTrue\n",
        )

    def test_a_chain_stops_at_the_first_false_link(self):
        self._run(
            "def side(tag, value):\n"
            "    print('eval', tag)\n"
            "    return value\n"
            "print(side('a', 5) < side('b', 1) < side('c', 9))\n",
            b"eval a\neval b\nFalse\n",
        )

    def test_a_chain_that_raises_mid_way_is_not_read_as_a_verdict(self):
        self._run_failing(
            "class Boom:\n"
            "    def __lt__(self, other):\n"
            "        raise ValueError('cmp')\n"
            "if Boom() < 1 < 2:\n"
            "    print('unreachable')\n",
            b"ValueError",
        )

    def test_a_keyword_call_builds_no_dictionary_in_the_callee(self):
        # The wrapper turned `kwnames` back into a dict and probed it once per
        # parameter. Without a `**` there is nothing to hand leftovers to, so
        # there is nothing a dict is for: each parameter looks through the
        # tuple instead.
        generated = python_to_capi_c(
            "def helper(v, step):\n    return v + step\n"
            "print(helper(1, step=2))\n",
            "program.py",
        )
        # The *definition*, not the forward declaration that shares its name -
        # slicing from the first mention took the declaration line and a
        # neighbouring function, and the assertions then held for reasons
        # unrelated to the change.
        wrapper = _body_of(generated, "f__value0_helper")
        self.assertNotIn("PyDict_New()", wrapper)
        self.assertIn("PyObject_RichCompareBool", wrapper)

    def test_a_starred_signature_keeps_its_dictionary(self):
        # `**more` is exactly what a dict is for, so that path is unchanged.
        generated = python_to_capi_c(
            "def helper(a, **more):\n    return (a, more)\n"
            "print(helper(1, z=2))\n",
            "program.py",
        )
        self.assertIn("PyDict_New()", generated)

    def test_the_keyword_complaints_are_worded_as_cpython_words_them(self):
        # All four of these were wrong at some point in getting here: two
        # reported a missing argument because the defaults were supplied
        # before the keywords were judged, and one named the wrong parameter
        # because a count cannot say *which* name went unclaimed.
        self._run(
            "def f(a, b):\n    return (a, b)\n"
            "def g(a, b=0, *, c=0):\n    return (a, b, c)\n"
            "for attempt in range(4):\n"
            "    try:\n"
            "        if attempt == 0: f(1, c=2)\n"
            "        elif attempt == 1: f(1, a=2)\n"
            "        elif attempt == 2: g(1, 2, d=3)\n"
            "        else: g(1, b=2, a=3)\n"
            "    except TypeError as e:\n"
            "        print(e)\n",
            b"f() got an unexpected keyword argument 'c'\n"
            b"f() got multiple values for argument 'a'\n"
            b"g() got an unexpected keyword argument 'd'\n"
            b"g() got multiple values for argument 'a'\n",
        )

    def test_a_condition_of_ands_builds_no_boolean_object(self):
        # `if a and b` wants a verdict, not a value. The whole chain used to be
        # evaluated into a Python object and then asked what it meant, which
        # cost each side the machine comparison it would otherwise have had:
        # `i > 5` ran at 1.22x the interpreter and `i > 5 and i < n` at 0.66x.
        # Each side goes through the same `truth` path now, and the short
        # circuit is a C `if` around the next one.
        generated = python_to_capi_c(
            "def f(n):\n"
            "    t = 0\n"
            "    for i in range(n):\n"
            "        if i > 5 and i < 15:\n"
            "            t += 1\n"
            "    return t\n"
            "print(f(20))\n",
            "program.py",
        )
        body = generated[generated.index("f_f("):]
        # The comparisons stay machine comparisons; nothing asks an object.
        self.assertNotIn("PyObject_IsTrue", body.split("static PyObject *f_")[1])

    def test_short_circuit_still_short_circuits(self):
        # The side that must not run must not have its code reached, not
        # merely have its value discarded - `0 and f()` may not call `f`.
        self._run(
            "def boom():\n"
            "    print('ran')\n"
            "    return True\n"
            "print(bool(0 and boom()))\n"
            "print(bool(1 or boom()))\n"
            "print(bool(1 and boom()))\n",
            b"False\nTrue\nran\nTrue\n",
        )

    def test_a_condition_still_consults_dunder_bool_in_order(self):
        # The fast path is only allowed where it changes nothing. An object
        # deciding its own truth must still be asked, once, in order.
        self._run(
            "class B:\n"
            "    def __init__(self, v, tag):\n"
            "        self.v = v\n"
            "        self.tag = tag\n"
            "    def __bool__(self):\n"
            "        print('asked', self.tag)\n"
            "        return self.v\n"
            "print(bool(B(False, 'one') and B(True, 'two')))\n"
            "print(bool(B(False, 'three') or B(True, 'four')))\n",
            # `one` is asked twice on purpose, and CPython agrees: the `and`
            # asks to decide whether to short-circuit, and `bool()` then asks
            # the object the `and` handed back - which is that same object.
            # The first draft of this test expected one call and was wrong
            # about Python, not about the compiler.
            b"asked one\nasked one\nFalse\nasked three\nasked four\nTrue\n",
        )

    def test_a_failure_inside_a_chain_is_not_read_as_a_verdict(self):
        # `truth` answers -1 for failure, which is neither true nor false. A
        # chain that treated it as one would swallow the exception and carry
        # on with the wrong answer.
        self._run_failing(
            "class Angry:\n"
            "    def __bool__(self):\n"
            "        raise ValueError('no')\n"
            "if Angry() and True:\n"
            "    print('unreachable')\n",
            b"ValueError",
        )

    def test_not_is_a_verdict_too(self):
        self._run(
            "print(not 0, not 1, not [], not [0], not None, not '')\n"
            "print(not (1 and 0), not (0 or 1), not not 5)\n",
            b"True False True False True True\nTrue False True\n",
        )

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

    def test_a_capture_the_scope_moves_afterwards_agrees_with_python(self):
        """Python closes over the variable, so a closure sees the later value.

        Capturing by value sees the earlier one, and these two cases used to
        be refused rather than answered differently. They are not refused now:
        a name whose value is still moving is put in a cell, which is the
        variable back again, and both scopes reach the same place through it.
        """

        for source in (
            "def f():\n"
            "    n = 1\n"
            "    def g():\n"
            "        return n\n"
            "    n = 2\n"
            "    return g()\n",
            "def f():\n"
            "    out = []\n"
            "    for i in range(3):\n"
            "        out.append(lambda: i)\n"
            "    return out\n",
        ):
            with self.subTest(source=source):
                self.assertIn("_py2bin_cell_", python_to_capi_c(source, "p.py"))

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


class MachineDoubleTests(unittest.TestCase):
    """Floats held in a register rather than on the heap.

    The same exercise as the integers, and until it existed the float half was
    the worse of the two: an integer loop already beat the interpreter while
    the same loop written in floats ran at four tenths of its speed.
    """

    _run = CApiEmitTests._run

    def test_a_float_accumulator_is_added_in_a_register(self):
        source = (
            "def total():\n"
            "    t = 0.0\n"
            "    i = 0\n"
            "    while i < 4:\n"
            "        t = t + 1.5\n"
            "        i = i + 1\n"
            "    return t\n"
            "print(total())\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertIn("double d_t", written)
        self.assertIn("int s_t", written)
        self._run(source, b"6.0\n")

    def test_the_result_is_a_float_and_not_an_integer(self):
        # The whole point of keeping the two representations apart. A value
        # that went out as `6` rather than `6.0` would be a wrong program.
        source = (
            "def total():\n"
            "    t = 0.0\n"
            "    i = 0\n"
            "    while i < 3:\n"
            "        t = t + 2.0\n"
            "        i = i + 1\n"
            "    return t\n"
            "print(type(total()).__name__, total())\n"
        )
        self._run(source, b"float 6.0\n")

    def test_division_by_zero_still_raises(self):
        # C answers an infinity where Python raises, so the fast path has to
        # hand the zero to the slow arm rather than compute it.
        source = (
            "def go():\n"
            "    a = 6.0\n"
            "    b = 0.0\n"
            "    b = b + 0.0\n"
            "    a = a * 1.0\n"
            "    try:\n"
            "        return a / b\n"
            "    except ZeroDivisionError:\n"
            "        return -1.0\n"
            "print(go())\n"
        )
        self._run(source, b"-1.0\n")

    def test_a_name_that_stops_being_a_float_still_reads_back(self):
        source = (
            "def go(flag):\n"
            "    t = 1.5\n"
            "    t = t + 1.5\n"
            "    if flag:\n"
            "        t = 'text'\n"
            "    return t\n"
            "print(go(False), go(True))\n"
        )
        self._run(source, b"3.0 text\n")

    def test_an_integer_loop_is_not_turned_into_a_float_one(self):
        source = (
            "def total():\n"
            "    n = 0\n"
            "    i = 0\n"
            "    while i < 4:\n"
            "        n = n + 3\n"
            "        i = i + 1\n"
            "    return n\n"
            "print(type(total()).__name__, total())\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertNotIn("double d_n", written)
        self._run(source, b"int 12\n")


class MethodBindingTests(unittest.TestCase):
    """A compiled method binds the way a plain Python function binds.

    This was `functools.partialmethod`, whose `__get__` is written in Python,
    so every `obj.method` ran interpreted code and allocated a `partial`. The
    replacement is CPython's own `instancemethod`, and these pin the parts of
    the behaviour that a swap of binder could quietly have changed.
    """

    _run = CApiEmitTests._run

    def test_an_instance_method_receives_the_instance(self):
        source = (
            "class C:\n"
            "    def __init__(self, n):\n"
            "        self.n = n\n"
            "    def doubled(self):\n"
            "        return self.n * 2\n"
            "print(C(21).doubled())\n"
        )
        self._run(source, b"42\n")

    def test_reaching_the_method_through_the_class_passes_self_by_hand(self):
        # `C.f` is the plain function on a class, as it is in Python 3, so it
        # takes the instance as its first argument.
        source = (
            "class C:\n"
            "    def __init__(self):\n"
            "        self.n = 7\n"
            "    def get(self):\n"
            "        return self.n\n"
            "print(C.get(C()))\n"
        )
        self._run(source, b"7\n")

    def test_a_bound_method_can_be_stored_and_called_later(self):
        source = (
            "class C:\n"
            "    def __init__(self):\n"
            "        self.n = 3\n"
            "    def add(self, k):\n"
            "        return self.n + k\n"
            "f = C().add\n"
            "print(f(4))\n"
        )
        self._run(source, b"7\n")

    def test_a_subclass_inherits_and_overrides(self):
        source = (
            "class A:\n"
            "    def name(self):\n"
            "        return 'A'\n"
            "    def shout(self):\n"
            "        return self.name() + '!'\n"
            "class B(A):\n"
            "    def name(self):\n"
            "        return 'B'\n"
            "print(A().shout(), B().shout())\n"
        )
        self._run(source, b"A! B!\n")

    def test_a_method_call_with_keywords_still_works(self):
        # Keywords take the other path - the one that fetches the attribute
        # and calls it - so both arms of `method_call` are covered.
        source = (
            "class C:\n"
            "    def go(self, a, b=10):\n"
            "        return a * b\n"
            "print(C().go(3), C().go(3, b=2))\n"
        )
        self._run(source, b"30 6\n")


class ConstantFoldingTests(unittest.TestCase):
    """Literal arithmetic computed once, at compile time - and only when safe."""

    _run = CApiEmitTests._run

    def test_a_literal_expression_is_folded(self):
        source = "x = 1.5 * 2.0 - 0.5\nprint(x)\n"
        written = python_to_capi_c(source, "program.py")
        self.assertIn("2.5", written)
        self._run(source, b"2.5\n")

    def test_a_division_by_zero_is_left_for_run_time(self):
        # Folding it would either refuse a program Python accepts or run one
        # Python does not.
        source = (
            "try:\n"
            "    print(1 // 0)\n"
            "except ZeroDivisionError:\n"
            "    print('raised')\n"
        )
        self._run(source, b"raised\n")

    def test_bools_are_not_folded_into_integers(self):
        source = "print(True + True, type(True + True).__name__)\n"
        self._run(source, b"2 int\n")

    def test_an_enormous_power_is_left_for_run_time(self):
        written = python_to_capi_c("x = 2 ** 100000\n", "program.py")
        # The folded value would be tens of kilobytes of digits in the binary.
        self.assertNotIn("PyLong_FromString(\"1" + "0" * 60, written)


class JoinedStringTests(unittest.TestCase):
    """f-strings built in one pass rather than by repeated concatenation."""

    _run = CApiEmitTests._run

    def test_the_pieces_are_joined_not_added(self):
        # Five pieces - literal, value, literal, value, literal - which is
        # past the point where the chain of concatenations stops being the
        # cheaper shape, so this one is gathered and joined.
        source = "n = 42\nw = 'x'\nprint(f'a{n}b{w}c')\n"
        written = python_to_capi_c(source, "program.py")
        self.assertIn("= PyUnicode_Join(", written)
        self._run(source, b"a42bxc\n")

    def test_conversions_and_specifiers_still_mean_what_they_mean(self):
        source = (
            "v = 1.5\n"
            "w = 'x'\n"
            "places = 2\n"
            "print(f'{v:.3f} {v!r} {w!s} {w!a}')\n"
            "print(f'{v:.{places}f}')\n"
        )
        self._run(source, b"1.500 1.5 x 'x'\n1.50\n")

    def test_an_empty_f_string_is_empty(self):
        self._run("print(repr(f''), repr(f'{1}'))\n", b"'' '1'\n")


class ArgumentBindingTests(unittest.TestCase):
    """The fast prologue must not change what any call shape does."""

    _run = CApiEmitTests._run

    def test_the_exact_arity_call_binds_positionally(self):
        source = (
            "def f(a, b, c):\n"
            "    return a * 100 + b * 10 + c\n"
            "print(f(1, 2, 3))\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertIn("!_kwnames && _nargs ==", written)
        self._run(source, b"123\n")

    def test_keywords_still_reach_the_right_parameters(self):
        source = (
            "def f(a, b, c):\n"
            "    return a * 100 + b * 10 + c\n"
            "print(f(1, c=3, b=2), f(1, 2, c=3))\n"
        )
        self._run(source, b"123 123\n")

    def test_too_few_and_too_many_are_still_refused(self):
        # Reached through a value: a call that names the function directly has
        # its arity checked when the program is compiled, so the run-time
        # refusal - which is what the fast prologue could have skipped - only
        # happens on the path that goes through the wrapper.
        source = (
            "def f(a, b):\n"
            "    return a + b\n"
            "g = f\n"
            "for count in (1, 3):\n"
            "    try:\n"
            "        g(*range(count))\n"
            "    except TypeError:\n"
            "        print('TypeError')\n"
        )
        self._run(source, b"TypeError\nTypeError\n")

    def test_defaults_and_star_args_take_the_general_path(self):
        source = (
            "def f(a, b=5, *rest, **named):\n"
            "    return (a, b, rest, sorted(named))\n"
            "print(f(1))\n"
            "print(f(1, 2, 3, x=4))\n"
        )
        self._run(source, b"(1, 5, (), [])\n(1, 2, (3,), ['x'])\n")


class MethodBodyTests(unittest.TestCase):
    """A method body gets the same treatment as any other function body.

    It did not. A method is written while the module's own statements are
    being emitted, and the flag saying "this is module level" was still set
    inside it - so the register analysis and the borrowing of locals were both
    switched off for every method in every class, which is the one place they
    matter most.
    """

    _run = CApiEmitTests._run

    def test_an_integer_accumulator_in_a_method_is_held_in_a_register(self):
        source = (
            "class Acc:\n"
            "    def run(self, n):\n"
            "        t = 0\n"
            "        i = 0\n"
            "        while i < n:\n"
            "            t = t + i * 2\n"
            "            i = i + 1\n"
            "        return t\n"
            "print(Acc().run(10))\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertIn("long long n_t", written)
        self._run(source, b"90\n")

    def test_a_float_accumulator_in_a_method_is_held_in_a_register(self):
        source = (
            "class Acc:\n"
            "    def run(self, n):\n"
            "        f = 0.0\n"
            "        i = 0\n"
            "        while i < n:\n"
            "            f = f + 1.5\n"
            "            i = i + 1\n"
            "        return f\n"
            "r = Acc().run(4)\n"
            "print(r, type(r).__name__)\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertIn("double d_f", written)
        self._run(source, b"6.0 float\n")

    def test_a_module_global_read_in_a_method_still_reaches_the_module(self):
        # The flag also decides where a name lives. Clearing it inside a method
        # must not turn a module-level name into a local of the method.
        source = (
            "LIMIT = 3\n"
            "class C:\n"
            "    def go(self):\n"
            "        return LIMIT * 2\n"
            "    def set(self):\n"
            "        global LIMIT\n"
            "        LIMIT = 9\n"
            "        return LIMIT\n"
            "c = C()\n"
            "print(c.go(), c.set(), c.go(), LIMIT)\n"
        )
        self._run(source, b"6 9 18 9\n")


class BorrowedOperandTests(unittest.TestCase):
    """Operands read without taking a reference, where that is sound.

    A local, parameter or capture is a C variable this function alone writes,
    and it holds its reference for the whole body - so the increment and
    decrement around every read are two writes to arrive back where they
    started. These cover the cases where that reasoning does *not* hold.
    """

    _run = CApiEmitTests._run

    def test_a_local_is_borrowed(self):
        source = (
            "def go():\n"
            "    a = [1]\n"
            "    b = [2]\n"
            "    return a + b\n"
            "print(go())\n"
        )
        self._run(source, b"[1, 2]\n")

    def test_a_global_rebound_during_the_expression_is_not_borrowed(self):
        # The call in the middle drops the module's last reference to `data`.
        # A borrowed read would be looking at freed memory; an owned one keeps
        # it alive for the operation, which is why globals are refused.
        source = (
            "data = [1, 2, 3]\n"
            "def clear():\n"
            "    global data\n"
            "    data = None\n"
            "    return [9]\n"
            "def go():\n"
            "    return data + clear()\n"
            "print(go(), data)\n"
        )
        self._run(source, b"[1, 2, 3, 9] None\n")

    def test_a_walrus_rebinding_mid_expression_is_not_borrowed(self):
        # A walrus is the one thing that writes a slot in the middle of an
        # expression, so a name any walrus assigns is never borrowed.
        source = (
            "def go():\n"
            "    x = [1]\n"
            "    return x + (x := [2]) + x\n"
            "print(go())\n"
        )
        self._run(source, b"[1, 2, 2]\n")

    def test_an_attribute_owner_that_is_a_call_is_still_released(self):
        source = (
            "class C:\n"
            "    def __init__(self):\n"
            "        self.v = 5\n"
            "def make():\n"
            "    return C()\n"
            "print(make().v)\n"
        )
        self._run(source, b"5\n")


class CommandLineArgumentTests(unittest.TestCase):
    """A compiled program can read the arguments it was started with.

    An embedded interpreter is handed no argument vector, so `sys.argv` held
    a single entry this compiler put there and a command-line program could
    not read what it had been asked to do. The arguments are recovered from
    the operating system instead - `/proc/self/cmdline` on Linux, `_NSGetArgv`
    on macOS, `GetCommandLineW` on Windows - because the C entry point's
    signature is fixed at `int main(void)` by this compiler's own front end,
    and Windows would be left out even if it were not.
    """

    def _built(self, source: str, room: Path) -> Path:
        entry = room / "program.py"
        entry.write_text(source, encoding="utf-8")
        generated = room / "program.c"
        generated.write_text(
            python_to_capi_c(source, str(entry)), encoding="utf-8"
        )
        binary = room / "program.bin"
        compile_c_native(generated, binary, target="darwin-arm64", clean=True)
        return binary

    def test_the_arguments_arrive(self):
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("needs the host whose C-API binding is wired up")
        source = "import sys\nprint(sys.argv[1:])\n"
        with tempfile.TemporaryDirectory() as scratch:
            binary = self._built(source, Path(scratch))
            done = subprocess.run(
                [str(binary), "one", "two", "three four"],
                capture_output=True, text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr[-300:])
            self.assertEqual(
                done.stdout.strip(), "['one', 'two', 'three four']"
            )

    def test_no_arguments_is_an_empty_tail(self):
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("needs the host whose C-API binding is wired up")
        source = "import sys\nprint(sys.argv[1:], len(sys.argv))\n"
        with tempfile.TemporaryDirectory() as scratch:
            binary = self._built(source, Path(scratch))
            done = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(done.stdout.strip(), "[] 1")

    def test_a_program_that_never_asks_does_not_pay(self):
        # Recovering the arguments costs a file read on Linux and an import
        # of ctypes elsewhere. A program that does not mention `argv` should
        # carry none of it.
        quiet = python_to_capi_c("print('hello')\n", "program.py")
        self.assertNotIn("_py2bin_argv", quiet)
        asking = python_to_capi_c("import sys\nprint(sys.argv)\n", "program.py")
        self.assertIn("_py2bin_argv", asking)

    def test_from_sys_import_argv_counts_as_asking(self):
        written = python_to_capi_c(
            "from sys import argv\nprint(argv)\n", "program.py"
        )
        self.assertIn("_py2bin_argv", written)


class SyntaxErrorReportTests(unittest.TestCase):
    """A program that will not parse gets an error, not a traceback."""

    def _compile(self, source: str):
        import io
        import contextlib
        from py2bin.cli import main as cli
        with tempfile.TemporaryDirectory() as scratch:
            room = Path(scratch)
            entry = room / "bad.py"
            entry.write_text(source, encoding="utf-8")
            captured = io.StringIO()
            with contextlib.redirect_stderr(captured):
                code = cli([
                    "compile-capi", str(entry), "-o", str(room / "out"), "--clean"
                ])
            return code, captured.getvalue(), entry

    UNPARSEABLE = 'def f(\n  print("unclosed"\n'

    def test_it_names_the_file_and_the_line(self):
        code, text, entry = self._compile(self.UNPARSEABLE)
        self.assertEqual(code, 2)
        self.assertIn(str(entry), text)
        self.assertIn(":2", text)

    def test_it_shows_the_offending_line(self):
        _, text, _ = self._compile(self.UNPARSEABLE)
        self.assertIn('print("unclosed"', text)

    def test_it_is_not_a_traceback_through_py2bin(self):
        # The failure used to arrive as a Python traceback through this
        # compiler's own frames, ending at "<unknown>" - which tells a reader
        # about py2bin and nothing about the file they wrote.
        _, text, _ = self._compile("def f(\n")
        self.assertNotIn("Traceback", text)
        self.assertNotIn("<unknown>", text)
        self.assertNotIn("capi_emit.py", text)


class ShadowedBuiltinTests(unittest.TestCase):
    """A builtin the program binds is the program's, not the shortcut's."""

    _run = CApiEmitTests._run

    def test_a_program_defined_print_is_used(self):
        source = (
            "def print(*a):\n"
            "    import sys\n"
            "    sys.stdout.write('S:' + ' '.join(map(str, a)) + '\\n')\n"
            "print('hi', 1)\n"
        )
        self._run(source, b"S:hi 1\n")

    def test_a_replaced_builtin_print_is_used(self):
        # Checked at run time against what `builtins` held at start-up, so a
        # harness that captures output by replacing `print` gets its calls.
        source = (
            "import builtins\n"
            "_real = builtins.print\n"
            "def loud(*a, **k):\n"
            "    _real('LOUD:', *a, **k)\n"
            "builtins.print = loud\n"
            "print('x')\n"
        )
        self._run(source, b"LOUD: x\n")

    def test_a_shadowed_super_is_the_programs_own(self):
        source = (
            "class A:\n"
            "    def f(self): return 'A'\n"
            "class B(A):\n"
            "    def f(self):\n"
            "        super = lambda: A()\n"
            "        return 'B+' + super().f()\n"
            "print(B().f())\n"
        )
        self._run(source, b"B+A\n")

    def test_len_and_str_bound_by_the_program_are_the_programs(self):
        source = (
            "def len(x): return 'mine'\n"
            "def go():\n"
            "    str = lambda v: 'also mine'\n"
            "    return str(1)\n"
            "print(len([1, 2]), go())\n"
        )
        self._run(source, b"mine also mine\n")

    def test_len_and_str_are_not_rechecked_against_builtins(self):
        # The documented trade: `print` is verified at run time, `len` and
        # `str` are not, because the check costs a dictionary probe in the
        # innermost loops a program has. Pinned so the choice cannot drift
        # without someone deciding to change it.
        written = python_to_capi_c("print(len('ab'), str(1))\n", "program.py")
        self.assertIn("PyObject_Size(", written)
        self.assertIn("= PyObject_Str(", written)


class ProgramLocationTests(unittest.TestCase):
    """A compiled program has to know where *it* is, not where Python is."""

    _run = CApiEmitTests._run

    def test_the_program_locates_its_own_directory(self):
        # An embedded interpreter is given no argument vector, so it has
        # nothing to locate itself from and answers `sys.executable` with the
        # installation it was configured with. On Linux that is
        # `/usr/local/bin/python3.14`, wherever the program actually sits - so
        # a bundle looked for the packages it carries next to the system
        # Python, found none, and stopped on an import of something it had
        # been shipped with.
        source = (
            "import builtins, os, sys\n"
            "print(os.path.basename(builtins._py2bin_dir))\n"
        )
        with tempfile.TemporaryDirectory() as scratch:
            room = Path(scratch) / "somewhere"
            room.mkdir()
            entry = room / "prog.py"
            entry.write_text(source)
            written = python_to_capi_c(source, str(entry))
            # The location is asked of the operating system where it can be.
            self.assertIn("/proc/self/exe", written)

    def test_the_location_is_not_taken_from_sys_executable_alone(self):
        # `sys.executable` is only the program on a platform that resolves it
        # to the host binary; where it does not, it is the interpreter's own
        # installation. It may still be the last resort, but it must not be
        # the only source.
        written = python_to_capi_c("print(1)\n", "program.py")
        start = written.index("PyRun_SimpleString(\"import sys, os, builtins")
        anchor = written[start : start + 900]
        self.assertIn("proc/self/exe", anchor)
        self.assertIn("realpath", anchor)

    def test_a_compiled_program_finds_its_own_directory(self):
        # Not through `_run`: that compares the compiled output against
        # CPython running the same source, and `_py2bin_dir` is a name only a
        # compiled program has - the reference run would fail by construction.
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("needs the host whose C-API binding is wired up")
        source = (
            "import builtins, os\n"
            "print(os.path.isdir(builtins._py2bin_dir), "
            "os.path.basename(builtins._py2bin_dir))\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            generated = root / "program.c"
            generated.write_text(
                python_to_capi_c(source, str(entry)), encoding="utf-8"
            )
            binary = root / "program.bin"
            compile_c_native(generated, binary, target="darwin-arm64", clean=True)
            done = subprocess.run([str(binary)], capture_output=True)
            self.assertEqual(done.returncode, 0, done.stderr[-400:])
            # The directory the binary is in, which is the temporary one.
            self.assertEqual(
                done.stdout.decode().strip(), f"True {root.name}"
            )


class AutoFetchWithoutExcludeTests(unittest.TestCase):
    """`--auto-fetch` with no `--exclude` must not fall over."""

    def test_the_option_defaults_to_a_list(self):
        # An `append` option nobody passed is None, not an empty list, and
        # the code that reads it to decide what not to fetch iterated it. So
        # every --auto-fetch build that did not also pass --exclude stopped
        # with a TypeError - which is every build the three-question front end
        # makes, and it shipped.
        from py2bin.cli import _parser as build_parser

        parsed = build_parser().parse_args(
            ["compile-capi", "x.py", "-o", "out"]
        )
        self.assertEqual(parsed.exclude, [])
        self.assertEqual(parsed.include, [])

    def test_every_append_option_has_a_default(self):
        import argparse
        from py2bin.cli import _parser as build_parser

        missing = []

        def walk(parser, path):
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    for name, sub in action.choices.items():
                        walk(sub, f"{path} {name}".strip())
                elif action.__class__.__name__ == "_AppendAction":
                    if action.default is None:
                        missing.append(f"{path} {action.option_strings}")

        walk(build_parser(), "")
        self.assertEqual(missing, [])


class WidenedInliningTests(unittest.TestCase):
    """A helper naming a module constant, or calling another helper."""

    _run = CApiEmitTests._run

    def test_a_body_naming_a_module_constant_is_written_out(self):
        source = (
            "SCALE = 3\n"
            "def weigh(v):\n"
            "    return v * SCALE\n"
            "print(weigh(5))\n"
        )
        written = python_to_capi_c(source, "program.py")
        # One call remains: the Python-callable wrapper's own, which exists so
        # the function can still be passed around as a value. The call site
        # `weigh(5)` is gone.
        self.assertEqual(written.count("= f_weigh("), 1)
        self._run(source, b"15\n")

    def test_a_helper_calling_a_helper_collapses(self):
        source = (
            "SCALE = 3\n"
            "def weigh(v):\n"
            "    return v * SCALE\n"
            "def bump(v):\n"
            "    return weigh(v) + 1\n"
            "print(bump(5))\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertEqual(written.count("= f_bump("), 1)
        # `bump` calls `weigh`, and that call is written out inside `bump`'s
        # own body before `bump` is written out at its call site.
        self.assertEqual(written.count("= f_weigh("), 1)
        self._run(source, b"16\n")

    def test_a_name_bound_twice_is_refused(self):
        # The danger was never the module constant, it was a second one
        # elsewhere - substituting would read whichever the call site had.
        source = (
            "K = 10\n"
            "def f(a):\n"
            "    return a + K\n"
            "def go():\n"
            "    K = 1\n"
            "    return f(5) + K\n"
            "print(go(), f(5))\n"
        )
        self._run(source, b"16 15\n")

    def test_a_shadowed_builtin_in_a_body_is_refused(self):
        source = (
            "def sz(x):\n"
            "    return len(x) * 2\n"
            "def go():\n"
            "    def len(y): return 99\n"
            "    return sz([1, 2, 3])\n"
            "print(sz([1, 2, 3]), go())\n"
        )
        self._run(source, b"6 6\n")

    def test_mutual_recursion_does_not_run_away(self):
        source = (
            "def down(n):\n"
            "    return 0 if n <= 0 else down(n - 1)\n"
            "def a(x):\n"
            "    return b(x) + 1\n"
            "def b(x):\n"
            "    return a(x) - 1\n"
            "print(down(3))\n"
        )
        self._run(source, b"0\n")


class CarriedPackageLayoutTests(unittest.TestCase):
    """Where fetched packages go when there is no bundle to put them in."""

    def test_a_bundle_and_a_bare_executable_filter_alike(self):
        # One filter, two layouts. Carrying the test suites and the build
        # metadata into a bare directory but not into a `.app` would be an
        # accident of which branch ran.
        from py2bin.freezer import copy_site_packages

        with tempfile.TemporaryDirectory() as scratch:
            room = Path(scratch)
            source = room / "site"
            (source / "keep").mkdir(parents=True)
            (source / "keep" / "__init__.py").write_text("")
            (source / "pip").mkdir()
            (source / "pip" / "__init__.py").write_text("")
            (source / "thing-1.0.dist-info").mkdir()
            (source / "thing-1.0.dist-info" / "METADATA").write_text("x")
            destination = room / "beside"
            copy_site_packages(destination, (source,))
            landed = sorted(p.name for p in destination.iterdir())
            self.assertIn("keep", landed)
            self.assertNotIn("pip", landed)
            self.assertNotIn("thing-1.0.dist-info", landed)


class ConcatenatedFStringTests(unittest.TestCase):
    """Few pieces are concatenated in a chain; many are gathered and joined."""

    _run = CApiEmitTests._run

    def test_a_short_f_string_concatenates(self):
        # Three pieces: the literal, the value, the literal.
        source = "n = 42\nprint(f'value {n} here')\n"
        written = python_to_capi_c(source, "program.py")
        # The assignment form, not the extern declaration, which is always
        # emitted for every vetted entry point whether or not it is called.
        self.assertIn("= PyUnicode_Concat(", written)
        self.assertNotIn("= PyUnicode_Join(", written)
        self._run(source, b"value 42 here\n")

    def test_a_long_f_string_joins(self):
        source = "n = 1\nprint(f'{n}{n}{n}{n}{n}{n}')\n"
        written = python_to_capi_c(source, "program.py")
        self.assertIn("PyUnicode_Join", written)
        self._run(source, b"111111\n")

    def test_neither_shape_consults_a_subclass_add(self):
        # Both paths must join rather than add: `+` would run the override.
        source = (
            "class S(str):\n"
            "    def __add__(self, other):\n"
            "        return 'HIJACKED'\n"
            "class R:\n"
            "    def __repr__(self):\n"
            "        return S('r')\n"
            "x = R()\n"
            "print(f'{x!r}{x!r}c')\n"
            "print(f'{x!r}{x!r}{x!r}{x!r}{x!r}{x!r}t')\n"
        )
        self._run(source, b"rrc\nrrrrrrt\n")


class ExactContainerTests(unittest.TestCase):
    """Names whose bindings prove an exact list or dict, with no run-time guard."""

    _run = CApiEmitTests._run

    def test_append_on_a_display_bound_list_goes_direct(self):
        source = (
            "def run():\n"
            "    xs = []\n"
            "    i = 0\n"
            "    while i < 4:\n"
            "        xs.append(i * 2)\n"
            "        i = i + 1\n"
            "    return xs\n"
            "print(run())\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertNotIn("PyObject_VectorcallMethod", written.split("f_run")[1].split("static")[0])
        self._run(source, b"[0, 2, 4, 6]\n")

    def test_a_subclass_keeps_its_override(self):
        # `w = Loud()` is a call, not a display, so the name is excluded and
        # the overridden method is the one that runs.
        source = (
            "class Loud(list):\n"
            "    def append(self, v):\n"
            "        print('override')\n"
            "        list.append(self, v * 10)\n"
            "def run():\n"
            "    w = Loud()\n"
            "    w.append(3)\n"
            "    return w\n"
            "print(run())\n"
        )
        self._run(source, b"override\n[30]\n")

    def test_a_rebound_name_is_excluded(self):
        source = (
            "def run():\n"
            "    xs = []\n"
            "    xs = make()\n"
            "    try:\n"
            "        xs.append(1)\n"
            "    except AttributeError:\n"
            "        return 'AttributeError'\n"
            "def make():\n"
            "    return object()\n"
            "print(run())\n"
        )
        self._run(source, b"AttributeError\n")

    def test_dict_store_goes_direct_and_unhashable_still_raises(self):
        source = (
            "def run():\n"
            "    d = {}\n"
            "    i = 0\n"
            "    while i < 4:\n"
            "        d[i % 2] = i\n"
            "        i = i + 1\n"
            "    try:\n"
            "        d[[1]] = 2\n"
            "    except TypeError:\n"
            "        return sorted(d.items())\n"
            "print(run())\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertIn("PyDict_SetItem(", written)
        self._run(source, b"[(0, 2), (1, 3)]\n")

    def test_augmented_assignment_keeps_a_list_exact(self):
        source = (
            "def run():\n"
            "    xs = [1]\n"
            "    xs += [2, 3]\n"
            "    xs.append(4)\n"
            "    return xs\n"
            "print(run())\n"
        )
        self._run(source, b"[1, 2, 3, 4]\n")


class AppendStatementTests(unittest.TestCase):
    """`xs.append(v)` as a statement makes nothing it then throws away."""

    _run = CApiEmitTests._run

    def test_the_statement_form_emits_no_none(self):
        source = (
            "def run():\n"
            "    xs = []\n"
            "    i = 0\n"
            "    while i < 3:\n"
            "        xs.append(i)\n"
            "    "    "    i = i + 1\n"
            "    return xs\n"
            "print(run())\n"
        )
        self._run(source.replace('    "    "    ', '        '), b"[0, 1, 2]\n")

    def test_the_expression_form_still_answers_none(self):
        source = (
            "def run():\n"
            "    xs = [1]\n"
            "    kept = xs.append(2)\n"
            "    return (xs, kept)\n"
            "print(run())\n"
        )
        self._run(source, b"([1, 2], None)\n")


class JoinedStringSemanticsTests(unittest.TestCase):
    """An f-string joins; it never asks a piece's type for `__add__`."""

    _run = CApiEmitTests._run

    def test_a_subclass_add_override_is_not_consulted(self):
        # CPython's BUILD_STRING concatenates raw; going through `+` would
        # run the override, which is what the two-piece shortcut used to do.
        source = (
            "class S(str):\n"
            "    def __add__(self, other):\n"
            "        return 'HIJACKED'\n"
            "class R:\n"
            "    def __repr__(self):\n"
            "        return S('rep')\n"
            "x = R()\n"
            "print(f'{x!r}!')\n"
            "print(f'{x!r}{x!r}extra')\n"
        )
        self._run(source, b"rep!\nrepreextra\n".replace(b"repre", b"repr" + b"ep" if False else b"reprep"))


class SpecialisedRaiseTests(unittest.TestCase):
    """`raise BuiltinError(...)` without the run-time class-or-instance test."""

    _run = CApiEmitTests._run

    def test_the_usual_shapes_still_catch(self):
        source = (
            "try:\n"
            "    raise ValueError('x')\n"
            "except ValueError as e:\n"
            "    print('a', e)\n"
            "try:\n"
            "    raise ValueError\n"
            "except ValueError as e:\n"
            "    print('b', repr(e))\n"
            "try:\n"
            "    raise KeyError('k')\n"
            "except KeyError as e:\n"
            "    print('c', repr(e))\n"
        )
        self._run(source, b"a x\nb ValueError()\nc KeyError('k')\n")

    def test_the_constructor_runs_exactly_once(self):
        source = (
            "n = [0]\n"
            "class Counting(ValueError):\n"
            "    def __init__(s, *a):\n"
            "        n[0] += 1\n"
            "        super().__init__(*a)\n"
            "try:\n"
            "    raise Counting('x')\n"
            "except ValueError:\n"
            "    pass\n"
            "print('ctor ran', n[0], 'time')\n"
        )
        self._run(source, b"ctor ran 1 time\n")

    def test_a_program_class_takes_the_general_path(self):
        source = (
            "class Mine(Exception):\n"
            "    pass\n"
            "try:\n"
            "    raise Mine('own')\n"
            "except Mine as e:\n"
            "    print(e)\n"
        )
        self._run(source, b"own\n")

    def test_raise_from_keeps_its_cause(self):
        source = (
            "try:\n"
            "    raise RuntimeError('why') from KeyError('cause')\n"
            "except RuntimeError as e:\n"
            "    print(e, repr(e.__cause__))\n"
        )
        self._run(source, b"why KeyError('cause')\n")


class HoistedLengthTests(unittest.TestCase):
    """`len(name)` loaded into a machine slot, so the expression stays narrow."""

    _run = CApiEmitTests._run

    def test_an_accumulator_fed_by_len_stays_in_a_register(self):
        source = (
            "def run():\n"
            "    i = 0\n"
            "    n = 0\n"
            "    while i < 4:\n"
            "        s = 'ab' + str(i)\n"
            "        n = n + len(s)\n"
            "        i = i + 1\n"
            "    return n\n"
            "print(run())\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertNotIn("PyLong_FromLongLong(PyObject_Size", written)
        self._run(source, b"12\n")

    def test_a_loop_bounded_by_len_measures_every_iteration(self):
        # `while i < len(xs)` re-measures, so a list that grows mid-loop is
        # seen growing - hoisting must not turn the bound into a constant.
        source = (
            "def run():\n"
            "    xs = [1]\n"
            "    i = 0\n"
            "    while i < len(xs):\n"
            "        if i < 3:\n"
            "            xs.append(i)\n"
            "        i = i + 1\n"
            "    return len(xs)\n"
            "print(run())\n"
        )
        self._run(source, b"4\n")

    def test_a_side_effecting_len_runs_exactly_once(self):
        # The slow arm of the fast path re-evaluates its tree; the measurement
        # is hoisted out of it precisely so `__len__` cannot run twice.
        source = (
            "class Weird:\n"
            "    def __len__(self):\n"
            "        print('measured')\n"
            "        return 3\n"
            "def run():\n"
            "    w = Weird()\n"
            "    n = 0\n"
            "    n = n + len(w)\n"
            "    return n\n"
            "print(run())\n"
        )
        self._run(source, b"measured\n3\n")

    def test_len_of_the_unmeasurable_raises_and_is_catchable(self):
        # `PyObject_Size` answers -1 with the exception set; the shortcut
        # boxed that unchecked, so `len(5)` answered -1 and left the
        # exception for whatever ran next.
        source = (
            "def run():\n"
            "    n = 0\n"
            "    try:\n"
            "        n = n + len(5)\n"
            "    except TypeError:\n"
            "        return 'TypeError'\n"
            "    return n\n"
            "print(run())\n"
            "try:\n"
            "    print(len(5))\n"
            "except TypeError:\n"
            "    print('TypeError again')\n"
        )
        self._run(source, b"TypeError\nTypeError again\n")

    def test_a_shadowed_len_is_the_program_s_own(self):
        source = (
            "def run():\n"
            "    len = lambda x: 99\n"
            "    n = 0\n"
            "    n = n + len('ab')\n"
            "    return n\n"
            "print(run())\n"
        )
        self._run(source, b"99\n")


class BorrowedCaptureTests(unittest.TestCase):
    """A capture is borrowed from the tuple the callable holds."""

    _run = CApiEmitTests._run

    def test_a_closure_that_drops_its_own_name_mid_call(self):
        # The caller holds the callable for the length of the call, so the
        # tuple - and the borrowed capture - outlive the body even when the
        # body removes the only name the program had for it.
        source = (
            "def make():\n"
            "    k = [41]\n"
            "    def inner():\n"
            "        global f\n"
            "        f = None\n"
            "        k[0] = k[0] + 1\n"
            "        return k[0]\n"
            "    return inner\n"
            "f = make()\n"
            "print(f(), f)\n"
        )
        self._run(source, b"42 None\n")


class CountedComprehensionTests(unittest.TestCase):
    """Comprehensions over a `range`, counted in a register.

    The written-out loop paid `PyIter_Next` and an integer object per item;
    now the target is a synthetic unboxed name, the element's arithmetic runs
    in machine registers, and - with no filter - the list is made at its final
    length and filled with `PyList_SetItem`, which steals the reference.
    """

    _run = CApiEmitTests._run

    def test_the_element_runs_in_registers_and_the_list_is_preallocated(self):
        source = "print([x * 2 for x in range(5)])\n"
        written = python_to_capi_c(source, "program.py")
        self.assertIn("PyList_SetItem", written)
        self.assertIn("n__py2bin_c", written)  # the synthetic machine slot
        self._run(source, b"[0, 2, 4, 6, 8]\n")

    def test_an_identity_comprehension_is_the_constructor(self):
        source = "print([x for x in range(4)], sorted({y for y in [2, 1, 2]}))\n"
        self._run(source, b"[0, 1, 2, 3] [1, 2]\n")

    def test_negative_steps_and_empty_ranges(self):
        source = (
            "print([x for x in range(10, 2, -3)])\n"
            "print([x + 1 for x in range(3, 3)], [x for x in range(0)])\n"
        )
        self._run(source, b"[10, 7, 4]\n[] []\n")

    def test_a_filter_keeps_the_growing_list(self):
        # With an `if`, the final length is unknown, so the list grows - and
        # the answer is the same either way.
        source = "print([w for w in range(6) if w % 2 == 0])\n"
        self._run(source, b"[0, 2, 4]\n")

    def test_bounds_wider_than_a_word_decline_to_the_iterator(self):
        source = (
            "big = 2 ** 70\n"
            "print(len([x for x in range(big, big + 3)]))\n"
        )
        self._run(source, b"3\n")

    def test_an_element_that_raises_leaves_the_exception(self):
        # The preallocated list has empty slots past the failure point, and
        # tearing it down must not touch them.
        source = (
            "def boom(v):\n"
            "    if v == 2:\n"
            "        raise ValueError('mid')\n"
            "    return v\n"
            "try:\n"
            "    [boom(x) for x in range(5)]\n"
            "except ValueError as e:\n"
            "    print('caught', e)\n"
        )
        self._run(source, b"caught mid\n")

    def test_the_enclosing_name_is_untouched(self):
        source = "x = 'outer'\nprint([x * x for x in range(3)], x)\n"
        self._run(source, b"[0, 1, 4] outer\n")


class IndexedSubscriptTests(unittest.TestCase):
    """`xs[i]` with a machine integer index, without an index object."""

    _run = CApiEmitTests._run

    def test_a_sequence_is_indexed_without_boxing(self):
        source = (
            "def go():\n"
            "    xs = [10, 20, 30]\n"
            "    i = 0\n"
            "    t = 0\n"
            "    while i < 3:\n"
            "        t = t + xs[i]\n"
            "        i = i + 1\n"
            "    return t\n"
            "print(go())\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertIn("PySequence_GetItem", written)
        self._run(source, b"60\n")

    def test_a_proven_list_needs_no_protocol_test(self):
        # The bindings say it is a list, so `PySequence_Check` asks a question
        # already answered. `PySequence_GetItem` stays: `PyList_GetItem` was
        # measured slower, its borrowed reference needing an increment that is
        # an out-of-line call from here.
        source = (
            "def run():\n"
            "    xs = [10, 20, 30]\n"
            "    i = 0\n"
            "    t = 0\n"
            "    while i < 3:\n"
            "        t = t + xs[i]\n"
            "        i = i + 1\n"
            "    return t\n"
            "print(run())\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertIn("= PySequence_GetItem(", written)
        # The extern declaration always appears; a *call* must not.
        self.assertNotIn("&& PySequence_Check(", written)
        self._run(source, b"60\n")

    def test_an_unproven_container_keeps_the_test(self):
        # `xs` arrives as a parameter, so nothing here says what it is - and
        # a dict must keep the mapping lookup. The index is arithmetic so
        # that it narrows and the fast path is reached at all.
        source = (
            "def run(xs):\n"
            "    i = 0\n"
            "    i = i + 0\n"
            "    return xs[i]\n"
            "print(run([7]), run({0: 'm'}))\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertIn("&& PySequence_Check(", written)
        self._run(source, b"7 m\n")

    def test_a_dict_keeps_the_mapping_lookup(self):
        # `d[0]` is a mapping lookup and must stay one: the sequence protocol
        # would not find the key at all.
        source = (
            "def go():\n"
            "    d = {0: 'a', 1: 'b'}\n"
            "    i = 0\n"
            "    out = []\n"
            "    while i < 2:\n"
            "        out.append(d[i])\n"
            "        i = i + 1\n"
            "    return out\n"
            "print(go())\n"
        )
        self._run(source, b"['a', 'b']\n")

    def test_strings_tuples_and_negative_indices_agree(self):
        source = (
            "def go():\n"
            "    s = 'hello'\n"
            "    t = (7, 8, 9)\n"
            "    xs = [1, 2, 3]\n"
            "    one = 1\n"
            "    return (s[one], t[one], xs[-one])\n"
            "print(go())\n"
        )
        self._run(source, b"('e', 8, 3)\n")

    def test_a_getitem_with_an_effect_runs_once(self):
        # Trying the sequence protocol and retrying on failure would run the
        # program's `__getitem__` twice. It is guarded, not retried.
        source = (
            "class Odd:\n"
            "    def __getitem__(self, k):\n"
            "        print('getitem')\n"
            "        raise TypeError('nope')\n"
            "def go():\n"
            "    i = 0\n"
            "    try:\n"
            "        return Odd()[i]\n"
            "    except TypeError:\n"
            "        return 'caught'\n"
            "print(go())\n"
        )
        self._run(source, b"getitem\ncaught\n")

    def test_an_index_out_of_range_still_raises(self):
        source = (
            "def go():\n"
            "    xs = [1]\n"
            "    i = 5\n"
            "    try:\n"
            "        return xs[i]\n"
            "    except IndexError:\n"
            "        return 'IndexError'\n"
            "print(go())\n"
        )
        self._run(source, b"IndexError\n")


class InliningTests(unittest.TestCase):
    """Small module-level functions written out where they are called.

    The point is not the call saved. The register analysis reads what a name
    is assigned to decide whether it can live in one, and a value arriving
    from a call tells it nothing - so inlining is what lets the loop around it
    be narrowed. These pin that, and the refusals that keep it honest.
    """

    _run = CApiEmitTests._run

    def test_the_call_disappears_and_the_loop_narrows(self):
        source = (
            "def add(a, b):\n"
            "    return a + b\n"
            "def run():\n"
            "    t = 0\n"
            "    i = 0\n"
            "    while i < 5:\n"
            "        t = add(t, i)\n"
            "        i = i + 1\n"
            "    return t\n"
            "print(run())\n"
        )
        written = python_to_capi_c(source, "program.py")
        # The accumulator is in a register, which it could not be while the
        # value came back from a call.
        self.assertIn("long long n_t", written)
        self._run(source, b"10\n")

    def test_the_function_still_exists_as_a_value(self):
        source = (
            "def add(a, b):\n"
            "    return a + b\n"
            "g = add\n"
            "print(add(1, 2), g(3, 4), list(map(add, [1], [2])))\n"
        )
        self._run(source, b"3 7 [3]\n")

    def test_a_shadowed_name_is_not_inlined(self):
        # A nested scope with its own `add` means the call there is not this
        # function, so the whole candidate is refused.
        source = (
            "def add(a, b):\n"
            "    return a + b\n"
            "def go():\n"
            "    add = lambda x, y: 99\n"
            "    return add(1, 2)\n"
            "print(go(), add(1, 2))\n"
        )
        self._run(source, b"99 3\n")

    def test_a_body_naming_a_global_is_refused(self):
        # Substituted into a scope with its own `K`, the body would read the
        # local. Refusing every non-parameter name is what prevents it.
        source = (
            "K = 100\n"
            "def bump(a):\n"
            "    return a + K\n"
            "def go():\n"
            "    K = 1\n"
            "    return bump(5) + K\n"
            "print(go())\n"
        )
        self._run(source, b"106\n")

    def test_an_argument_that_can_have_an_effect_is_not_duplicated(self):
        # `twice` uses its parameter twice; the argument prints when it runs,
        # so a duplicated substitution would print twice.
        source = (
            "def twice(x):\n"
            "    return x + x\n"
            "def noisy():\n"
            "    print('ran')\n"
            "    return 3\n"
            "print(twice(noisy()))\n"
        )
        self._run(source, b"ran\n6\n")

    def test_arguments_keep_their_order(self):
        source = (
            "def add(a, b):\n"
            "    return a + b\n"
            "def one():\n"
            "    print('one')\n"
            "    return 1\n"
            "def two():\n"
            "    print('two')\n"
            "    return 2\n"
            "print(add(one(), two()))\n"
        )
        self._run(source, b"one\ntwo\n3\n")

    def test_a_dropped_argument_still_runs(self):
        # `first` never mentions `b`, so substituting would skip the argument
        # entirely - which must not happen when it can do something.
        source = (
            "def first(a, b):\n"
            "    return a\n"
            "def noisy():\n"
            "    print('ran')\n"
            "    return 9\n"
            "print(first(1, noisy()))\n"
        )
        self._run(source, b"ran\n1\n")

    def test_recursion_is_left_alone(self):
        source = (
            "def down(n):\n"
            "    return 0 if n <= 0 else down(n - 1)\n"
            "print(down(3))\n"
        )
        self._run(source, b"0\n")

    def test_keywords_and_spreading_take_the_call(self):
        source = (
            "def add(a, b):\n"
            "    return a + b\n"
            "pair = (3, 4)\n"
            "print(add(1, b=2), add(*pair))\n"
        )
        self._run(source, b"3 7\n")


class TextFoldingTests(unittest.TestCase):
    """Literal text joined once, and only where Python would agree."""

    _run = CApiEmitTests._run

    def test_adjacent_literals_are_folded(self):
        written = python_to_capi_c("print('a' + 'b')\n", "program.py")
        self.assertIn('"ab"', written)
        self._run("print('a' + 'b')\n", b"ab\n")

    def test_repetition_is_folded_either_way_round(self):
        self._run("print('-' * 4, 3 * 'ab')\n", b"---- ababab\n")

    def test_mismatched_types_are_left_for_run_time(self):
        # `b'a' + 'b'` is a TypeError; folding it would refuse at compile time
        # a program that Python refuses at run time, which is a different
        # program.
        source = (
            "try:\n"
            "    print(b'a' + 'b')\n"
            "except TypeError:\n"
            "    print('TypeError')\n"
        )
        self._run(source, b"TypeError\n")

    def test_an_enormous_repetition_is_left_for_run_time(self):
        written = python_to_capi_c("x = 'y' * 100000\n", "program.py")
        self.assertNotIn("y" * 5000, written)


class BuiltinLookupTests(unittest.TestCase):
    """A builtin the program names is looked up live, not cached."""

    _run = CApiEmitTests._run

    def test_rebinding_a_builtin_is_seen(self):
        # The name is interned for speed; the lookup itself stays live, so a
        # program that replaces a builtin gets the replacement.
        source = (
            "import builtins\n"
            "def shout(*a):\n"
            "    return 'replaced'\n"
            "print(sorted([2, 1]))\n"
            "builtins.sorted = shout\n"
            "print(sorted([2, 1]))\n"
        )
        self._run(source, b"[1, 2]\nreplaced\n")

    def test_a_program_may_define_its_own_len_and_str(self):
        # `len` and `str` go straight to their C entry points, which was done
        # whether or not the program had bound the name - so a module with its
        # own `len` printed the length instead of calling its own function.
        source = (
            "def len(x):\n"
            "    return 'shadowed'\n"
            "print(len([1, 2, 3]))\n"
            "def go():\n"
            "    str = lambda v: 'mine'\n"
            "    return str(5)\n"
            "print(go())\n"
        )
        self._run(source, b"shadowed\nmine\n")

    def test_the_shortcut_still_applies_when_nothing_shadows_it(self):
        written = python_to_capi_c("print(len('abc'), str(2))\n", "program.py")
        self.assertIn("PyObject_Size", written)
        self._run("print(len('abc'), str(2))\n", b"3 2\n")


class ConstantPoolTests(unittest.TestCase):
    """Literals built once at start-up rather than at every execution."""

    _run = CApiEmitTests._run

    def test_a_string_in_a_loop_is_built_once(self):
        source = (
            "def go():\n"
            "    n = 0\n"
            "    i = 0\n"
            "    while i < 3:\n"
            "        s = 'a constant'\n"
            "        n = n + len(s)\n"
            "        i = i + 1\n"
            "    return n\n"
            "print(go())\n"
        )
        written = python_to_capi_c(source, "program.py")
        self.assertEqual(written.count('PyUnicode_FromString("a constant")'), 1)
        self._run(source, b"30\n")

    def test_negative_zero_keeps_its_sign(self):
        # `-0.0 == 0.0` and the two hash alike, so a pool keyed by value gave
        # them one slot and turned every `-0.0` into `0.0`.
        source = (
            "a = -0.0\n"
            "b = 0.0\n"
            "print(a, b, str(a) == str(b))\n"
        )
        self._run(source, b"-0.0 0.0 False\n")

    def test_an_integer_and_a_float_stay_apart(self):
        source = "print(repr(1), repr(1.0))\n"
        self._run(source, b"1 1.0\n")


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

    def test_a_finally_that_itself_yields_compiles(self):
        """The cleanup is a block, and a block may suspend.

        Reached the same way from both paths - finishing and raising - so it
        can hold a `yield` of its own. What was raised waits in a name until
        the cleanup is done and is put back afterwards.
        """
        python_to_capi_c(
            "def f():\n    try:\n        yield 1\n    finally:\n        yield 2\n",
            "program.py",
        )

    def test_a_break_out_of_a_finally_region_compiles(self):
        # The cleanup runs before the jump, which is what the jump would have
        # reached had it left the ordinary way.
        python_to_capi_c(
            "def f():\n    for i in (1,):\n        try:\n            yield i\n"
            "            break\n        finally:\n            pass\n",
            "program.py",
        )

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
