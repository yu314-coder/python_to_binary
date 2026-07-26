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

_MAGIC = {"linux": b"\x7fELF", "darwin": b"\xcf\xfa\xed\xfe"}

# Objects live in the same bump arena as lists and strings, so class programs
# share the heap slice's POSIX-only support. Windows is a documented gap.
_POSIX_TARGETS = (
    "linux-x86_64",
    "linux-arm64",
    "darwin-x86_64",
    "darwin-arm64",
)
_WINDOWS_TARGETS = ("windows-x86_64", "windows-arm64")


class NativeClassTests(unittest.TestCase):
    """User-defined classes lowered to real machine code.

    Instances are heap blocks with a statically known layout; because the class
    of every instance is known at build time, method calls resolve directly and
    inline. Each accepted program is executed natively on darwin-arm64 and
    compared against CPython running the same source.
    """

    def _run(self, source: str, expected_exit: int) -> None:
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
                    f"{target} class image has a broken header",
                )

            # Instances live in the arena, which Windows gets from
            # VirtualAlloc. A PE cannot run here, so this checks the image is
            # structurally valid; the behaviour is checked by running the
            # darwin-arm64 image below, built from the same IR.
            for target in _WINDOWS_TARGETS:
                image = root / f"program-{target}.exe"
                compile_native(entry, image, target, clean=True)
                self.assertEqual(image.read_bytes()[:2], b"MZ", target)

            if not _HOST_IS_DARWIN_ARM64:
                return
            native = subprocess.run(
                [str(root / "program-darwin-arm64.bin")], capture_output=True
            )
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.returncode, expected_exit)
            # The generated machine code must agree with CPython itself.
            self.assertEqual(native.returncode, reference.returncode)
            self.assertEqual(native.stdout, reference.stdout)

    def _reject(self, source: str, fragment: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "program.bin", "darwin-arm64", clean=True)
            self.assertIn(fragment, str(caught.exception))

    # --- supported ----------------------------------------------------------

    def test_construction_attribute_and_method(self):
        self._run(
            "class Point:\n"
            "    def __init__(self, x, y):\n"
            "        self.x = x\n"
            "        self.y = y\n"
            "    def total(self):\n"
            "        return self.x + self.y\n"
            "p = Point(3, 4)\n"
            "raise SystemExit(p.total())\n",
            7,
        )

    def test_several_instances_are_independent(self):
        self._run(
            "class Point:\n"
            "    def __init__(self, x, y):\n"
            "        self.x = x\n"
            "        self.y = y\n"
            "    def total(self):\n"
            "        return self.x + self.y\n"
            "a = Point(1, 2)\n"
            "b = Point(10, 20)\n"
            "raise SystemExit(a.total() + b.total())\n",
            33,
        )

    def test_method_mutates_attribute(self):
        self._run(
            "class Counter:\n"
            "    def __init__(self, start):\n"
            "        self.value = start\n"
            "    def bump(self, by):\n"
            "        self.value = self.value + by\n"
            "        return self.value\n"
            "c = Counter(5)\n"
            "c.bump(3)\n"
            "c.bump(4)\n"
            "raise SystemExit(c.value)\n",
            12,
        )

    def test_method_calls_another_method(self):
        self._run(
            "class Box:\n"
            "    def __init__(self, w, h):\n"
            "        self.w = w\n"
            "        self.h = h\n"
            "    def area(self):\n"
            "        return self.w * self.h\n"
            "    def doubled(self):\n"
            "        return self.area() * 2\n"
            "b = Box(3, 5)\n"
            "raise SystemExit(b.doubled())\n",
            30,
        )

    def test_method_called_in_a_loop(self):
        self._run(
            "class Acc:\n"
            "    def __init__(self):\n"
            "        self.total = 0\n"
            "    def add(self, v):\n"
            "        self.total = self.total + v\n"
            "a = Acc()\n"
            "for i in range(1, 6):\n"
            "    a.add(i)\n"
            "raise SystemExit(a.total)\n",
            15,
        )

    def test_direct_attribute_read_and_write(self):
        self._run(
            "class P:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "p = P(9)\n"
            "p.x = p.x + 4\n"
            "raise SystemExit(p.x)\n",
            13,
        )

    def test_method_with_branching_control_flow(self):
        self._run(
            "class Clamp:\n"
            "    def __init__(self, v):\n"
            "        self.v = v\n"
            "    def limited(self):\n"
            "        if self.v > 10:\n"
            "            return 10\n"
            "        return self.v\n"
            "a = Clamp(4)\n"
            "b = Clamp(25)\n"
            "raise SystemExit(a.limited() + b.limited())\n",
            14,
        )

    def test_class_without_initializer(self):
        self._run(
            "class Empty:\n"
            "    def answer(self):\n"
            "        return 42\n"
            "e = Empty()\n"
            "raise SystemExit(e.answer())\n",
            42,
        )

    def test_objects_compose_with_lists_strings_and_floats(self):
        # Every implemented value type in one program, to catch heap-layout
        # interference between objects, lists, strings, and doubles.
        self._run(
            "class Stats:\n"
            "    def __init__(self, base):\n"
            "        self.base = base\n"
            "        self.count = 0\n"
            "    def observe(self, v):\n"
            "        self.count = self.count + 1\n"
            "        self.base = self.base + v\n"
            "        return self.base\n"
            "s = Stats(100)\n"
            "xs = [4, 7, 9]\n"
            "i = 0\n"
            "while i < len(xs):\n"
            "    s.observe(xs[i])\n"
            "    i = i + 1\n"
            "name = \"runs=\"\n"
            "print(name)\n"
            "ratio = 0.0\n"
            "for k in range(1, 4):\n"
            "    ratio = ratio + 0.5\n"
            "raise SystemExit(s.base + s.count + int(ratio) + xs[-1])\n",
            133,
        )

    # --- honest rejections --------------------------------------------------

    def test_inheritance_is_rejected(self):
        self._reject(
            "class A:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "class B(A):\n"
            "    def __init__(self):\n"
            "        self.y = 2\n"
            "b = B()\n"
            "raise SystemExit(b.y)\n",
            "inheritance",
        )

    def test_unknown_attribute_is_rejected(self):
        self._reject(
            "class P:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "p = P()\n"
            "raise SystemExit(p.z)\n",
            "has no native attribute",
        )

    def test_conditionally_assigned_attribute_is_rejected(self):
        # CPython raises AttributeError when __init__ skipped the assignment,
        # so a zero-filled layout slot would be a wrong answer.
        self._reject(
            "class C:\n"
            "    def __init__(self, flag):\n"
            "        self.a = 1\n"
            "        if flag:\n"
            "            self.b = 2\n"
            "c = C(0)\n"
            "raise SystemExit(c.b)\n",
            "only conditionally",
        )

    def test_attribute_first_assigned_in_a_method_is_rejected(self):
        self._reject(
            "class P:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "    def grow(self):\n"
            "        self.y = 5\n"
            "p = P()\n"
            "p.grow()\n"
            "raise SystemExit(p.x)\n",
            "has no native attribute",
        )

    def test_class_attribute_is_rejected(self):
        self._reject(
            "class P:\n"
            "    LIMIT = 5\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "p = P()\n"
            "raise SystemExit(p.x)\n",
            "only method definitions",
        )

    def test_decorated_method_is_rejected(self):
        self._reject(
            "class P:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "    @staticmethod\n"
            "    def f():\n"
            "        return 2\n"
            "p = P()\n"
            "raise SystemExit(p.x)\n",
            "cannot be decorated",
        )

    def test_recursive_method_is_rejected(self):
        self._reject(
            "class P:\n"
            "    def __init__(self):\n"
            "        self.x = 3\n"
            "    def f(self, n):\n"
            "        if n <= 0:\n"
            "            return 0\n"
            "        return self.f(n - 1)\n"
            "p = P()\n"
            "raise SystemExit(p.f(2))\n",
            "recursive",
        )

    def test_float_attribute_is_rejected(self):
        self._reject(
            "class P:\n"
            "    def __init__(self):\n"
            "        self.x = 1.5\n"
            "p = P()\n"
            "raise SystemExit(int(p.x))\n",
            "signed 64-bit integers",
        )

    def test_object_variable_cannot_change_class(self):
        self._reject(
            "class A:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "class B:\n"
            "    def __init__(self):\n"
            "        self.x = 2\n"
            "o = A()\n"
            "o = B()\n"
            "raise SystemExit(o.x)\n",
            "cannot change class",
        )

    def test_method_without_self_is_rejected(self):
        self._reject(
            "class P:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "    def f(value):\n"
            "        return value\n"
            "p = P()\n"
            "raise SystemExit(p.x)\n",
            "'self'",
        )


if __name__ == "__main__":
    unittest.main()
