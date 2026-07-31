"""A closure is not inside the region its definition sits in.

Its `return` is its own, and so is any loop it contains. Leaving through the
enclosing `finally` would set a flag and jump to a label belonging to another
C function, which the C compiler then refuses - `'_c27' is not a declared
local or parameter`, from an application compiled on a phone.
"""

import unittest

from py2bin.capi_emit import python_to_capi_c


class ClosureScopeTests(unittest.TestCase):
    def test_a_closure_returning_inside_a_finally_region(self):
        c = python_to_capi_c(
            "def f():\n"
            "    try:\n"
            "        def inner():\n"
            "            return 1\n"
            "        return inner()\n"
            "    finally:\n"
            "        pass\n",
            "program.py",
        )
        # The closure's own C function must not mention the outer clause.
        closure = c.split("_closure0")[1].split("static PyObject *")[0]
        self.assertNotIn("goto _finally", closure)

    def test_a_closure_breaking_inside_a_loop_in_a_finally_region(self):
        python_to_capi_c(
            "def f():\n"
            "    try:\n"
            "        for i in range(3):\n"
            "            pass\n"
            "        def after():\n"
            "            for j in range(2):\n"
            "                if j:\n"
                "                    return j\n"
            "            return 0\n"
            "        return after()\n"
            "    finally:\n"
            "        pass\n",
            "program.py",
        )

    def test_a_loop_inside_a_closure_inside_a_region(self):
        python_to_capi_c(
            "def f():\n"
            "    try:\n"
            "        def counter():\n"
            "            for k in range(4):\n"
            "                if k == 3:\n"
            "                    break\n"
            "            return k\n"
            "        return counter()\n"
            "    finally:\n"
            "        pass\n",
            "program.py",
        )
