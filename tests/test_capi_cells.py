"""`nonlocal`, as a cell both scopes reach through."""

import unittest

from py2bin.capi_emit import CApiEmitError, python_to_capi_c


def _compiles(source: str) -> str:
    return python_to_capi_c(source, "program.py")


class NonlocalTests(unittest.TestCase):
    def test_a_closure_can_rebind_an_enclosing_name(self):
        _compiles(
            "def outer():\n"
            "    total = 0\n"
            "    def add(n):\n"
            "        nonlocal total\n"
            "        total += n\n"
            "    add(1)\n"
            "    return total\n"
        )

    def test_a_parameter_can_be_the_one_rebound(self):
        # The cell starts from the parameter rather than empty, because a
        # parameter already holds a value when the body begins.
        c = _compiles(
            "def outer(start):\n"
            "    def bump():\n"
            "        nonlocal start\n"
            "        start = start * 2\n"
            "    bump()\n"
            "    return start\n"
        )
        self.assertIn("_py2bin_cell_start", c)

    def test_it_reaches_through_a_function_in_between(self):
        _compiles(
            "def outer():\n"
            "    n = 1\n"
            "    def middle():\n"
            "        def inner():\n"
            "            nonlocal n\n"
            "            n += 1\n"
            "        inner()\n"
            "    middle()\n"
            "    return n\n"
        )

    def test_one_declaration_naming_several_names(self):
        _compiles(
            "def outer():\n"
            "    a = 1\n"
            "    b = 2\n"
            "    def both():\n"
            "        nonlocal a, b\n"
            "        a, b = b, a\n"
            "    both()\n"
            "    return a, b\n"
        )

    def test_a_name_declared_both_ways_is_refused(self):
        with self.assertRaises(CApiEmitError) as caught:
            _compiles(
                "x = 0\n"
                "def outer():\n"
                "    x = 1\n"
                "    def inner():\n"
                "        nonlocal x\n"
                "        global x\n"
                "        x = 2\n"
                "    inner()\n"
            )
        self.assertIn("x", str(caught.exception))

    def test_a_function_without_nonlocal_is_left_alone(self):
        # Nothing should gain a cell it did not ask for.
        c = _compiles(
            "def outer():\n"
            "    total = 0\n"
            "    def read():\n"
            "        return total\n"
            "    return read()\n"
        )
        self.assertNotIn("_py2bin_cell_", c)


if __name__ == "__main__":
    unittest.main()
