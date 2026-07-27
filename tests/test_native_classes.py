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

    Single inheritance does not weaken that: a subclass repeats its base's
    fields at the front of its own layout and merges the base's methods at
    build time, and a variable is still pinned to exactly one class, so every
    call still has one statically known body.
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

    # --- single inheritance --------------------------------------------------

    def test_single_inheritance_with_super(self):
        self._run(
            "class A:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "    def scaled(self):\n"
            "        return self.x * 2\n"
            "class B(A):\n"
            "    def __init__(self, x, y):\n"
            "        super().__init__(x)\n"
            "        self.y = y\n"
            "    def total(self):\n"
            "        return self.scaled() + self.y\n"
            "n = 0\n"
            "for i in range(0, 3):\n"
            "    n += 1\n"
            "b = B(n, 4)\n"
            "raise SystemExit(b.total())\n",
            10,
        )

    def test_child_without_init_inherits_layout_and_methods(self):
        self._run(
            "class A:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "    def doubled(self):\n"
            "        return self.x * 2\n"
            "class B(A):\n"
            "    def tripled(self):\n"
            "        return self.x * 3\n"
            "n = 0\n"
            "for i in range(0, 5):\n"
            "    n += 1\n"
            "b = B(n)\n"
            "raise SystemExit(b.doubled() + b.tripled())\n",
            25,
        )

    def test_override_wins_through_an_inherited_caller(self):
        # A.report() calls self.tag(); on a B instance CPython calls B.tag, so
        # the inlined body has to resolve against the receiver's class rather
        # than the class that lexically defined report().
        self._run(
            "class A:\n"
            "    def __init__(self):\n"
            "        self.n = 1\n"
            "    def tag(self):\n"
            "        return 10\n"
            "    def report(self):\n"
            "        return self.tag() + self.n\n"
            "class B(A):\n"
            "    def tag(self):\n"
            "        return 20\n"
            "a = A()\n"
            "b = B()\n"
            "k = 0\n"
            "for i in range(0, 1):\n"
            "    k += 1\n"
            "raise SystemExit(a.report() + b.report() + k - 1)\n",
            32,
        )

    def test_parent_and_child_share_leading_offsets(self):
        self._run(
            "class A:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "    def get(self):\n"
            "        return self.x\n"
            "class B(A):\n"
            "    def __init__(self, x, y):\n"
            "        super().__init__(x)\n"
            "        self.y = y\n"
            "n = 0\n"
            "for i in range(0, 6):\n"
            "    n += 1\n"
            "a = A(n)\n"
            "b = B(n + 1, 100)\n"
            "raise SystemExit(a.get() + b.get())\n",
            13,
        )

    def test_inherited_layout_keeps_the_base_order(self):
        # B assigns the base's attributes itself, in the opposite order. If the
        # layout followed B's write order, A.diff() would return -5 on a B.
        self._run(
            "class A:\n"
            "    def __init__(self, p, q):\n"
            "        self.p = p\n"
            "        self.q = q\n"
            "    def diff(self):\n"
            "        return self.p - self.q\n"
            "class B(A):\n"
            "    def __init__(self):\n"
            "        self.q = 4\n"
            "        self.p = 9\n"
            "        self.r = 1\n"
            "n = 0\n"
            "for i in range(0, 1):\n"
            "    n += 1\n"
            "b = B()\n"
            "raise SystemExit(b.diff() + b.r + n - 1)\n",
            6,
        )

    def test_three_level_chain(self):
        self._run(
            "class A:\n"
            "    def __init__(self, a):\n"
            "        self.a = a\n"
            "class B(A):\n"
            "    def __init__(self, a, b):\n"
            "        super().__init__(a)\n"
            "        self.b = b\n"
            "class C(B):\n"
            "    def __init__(self, a, b, c):\n"
            "        super().__init__(a, b)\n"
            "        self.c = c\n"
            "    def total(self):\n"
            "        return self.a + self.b + self.c\n"
            "n = 0\n"
            "for i in range(0, 3):\n"
            "    n += 1\n"
            "c = C(1, 2, n)\n"
            "raise SystemExit(c.total())\n",
            6,
        )

    def test_grandchild_inherits_the_middle_override(self):
        self._run(
            "class A:\n"
            "    def __init__(self):\n"
            "        self.v = 0\n"
            "    def tag(self):\n"
            "        return 1\n"
            "class B(A):\n"
            "    def tag(self):\n"
            "        return 2\n"
            "class C(B):\n"
            "    def name(self):\n"
            "        return self.tag() * 10\n"
            "c = C()\n"
            "b = B()\n"
            "a = A()\n"
            "raise SystemExit(c.name() + b.tag() + a.tag())\n",
            23,
        )

    def test_grandchild_without_init_runs_the_whole_super_chain(self):
        self._run(
            "class A:\n"
            "    def __init__(self, v):\n"
            "        self.v = v\n"
            "class B(A):\n"
            "    def __init__(self, v):\n"
            "        super().__init__(v)\n"
            "        self.w = v + 1\n"
            "class C(B):\n"
            "    def sum(self):\n"
            "        return self.v + self.w\n"
            "n = 0\n"
            "for i in range(0, 7):\n"
            "    n += 1\n"
            "c = C(n)\n"
            "raise SystemExit(c.sum())\n",
            15,
        )

    def test_inherited_float_attribute(self):
        self._run(
            "class A:\n"
            "    def __init__(self, v):\n"
            "        self.r: float = v\n"
            "class B(A):\n"
            "    def __init__(self, v, n):\n"
            "        super().__init__(v)\n"
            "        self.n = n\n"
            "t = 0.0\n"
            "for i in range(0, 3):\n"
            "    t = t + 0.5\n"
            "b = B(t, 7)\n"
            "a = A(t)\n"
            "raise SystemExit(int(b.r * 2.0) + b.n + int(a.r))\n",
            11,
        )

    def test_base_without_init_accepts_a_bare_super_call(self):
        self._run(
            "class A:\n"
            "    pass\n"
            "class B(A):\n"
            "    def __init__(self, v):\n"
            "        super().__init__()\n"
            "        self.v = v\n"
            "n = 0\n"
            "for i in range(0, 9):\n"
            "    n += 1\n"
            "b = B(n)\n"
            "raise SystemExit(b.v)\n",
            9,
        )

    def test_inherited_mutator_in_a_loop(self):
        self._run(
            "class Acc:\n"
            "    def __init__(self, start):\n"
            "        self.total = start\n"
            "    def add(self, v):\n"
            "        self.total = self.total + v\n"
            "class Tagged(Acc):\n"
            "    def __init__(self, start, tag):\n"
            "        super().__init__(start)\n"
            "        self.tag = tag\n"
            "    def score(self):\n"
            "        return self.total + self.tag\n"
            "c = Tagged(0, 3)\n"
            "for i in range(1, 6):\n"
            "    c.add(i)\n"
            "raise SystemExit(c.score())\n",
            18,
        )

    def test_with_statement_over_an_inherited_context_manager(self):
        self._run(
            "class Guard:\n"
            "    def __init__(self, v):\n"
            "        self.v = v\n"
            "    def __enter__(self):\n"
            "        return self.v + 1\n"
            "    def __exit__(self, a, b, c):\n"
            "        self.v = self.v + 10\n"
            "class Loud(Guard):\n"
            "    def __init__(self, v, extra):\n"
            "        super().__init__(v)\n"
            "        self.extra = extra\n"
            "n = 0\n"
            "for i in range(0, 5):\n"
            "    n += 1\n"
            "g = Loud(n, 2)\n"
            "with g as seen:\n"
            "    total = seen\n"
            "raise SystemExit(total + g.v + g.extra)\n",
            23,
        )

    def test_subclass_composes_with_lists_strings_and_floats(self):
        self._run(
            "class Base:\n"
            "    def __init__(self, base):\n"
            "        self.base = base\n"
            "        self.count = 0\n"
            "    def observe(self, v):\n"
            "        self.count = self.count + 1\n"
            "        self.base = self.base + v\n"
            "        return self.base\n"
            "class Ratio(Base):\n"
            "    def __init__(self, base, r):\n"
            "        super().__init__(base)\n"
            "        self.r: float = r\n"
            "    def summary(self):\n"
            "        return self.base + self.count\n"
            "ratio = 0.0\n"
            "for k in range(1, 4):\n"
            "    ratio = ratio + 0.5\n"
            "s = Ratio(100, ratio)\n"
            "xs = [4, 7, 9]\n"
            "i = 0\n"
            "while i < len(xs):\n"
            "    s.observe(xs[i])\n"
            "    i = i + 1\n"
            "name = \"runs=\"\n"
            "print(name)\n"
            "raise SystemExit(s.summary() + int(s.r) + xs[-1])\n",
            133,
        )

    # --- honest rejections --------------------------------------------------

    def test_child_init_leaving_a_parent_attribute_unset_is_rejected(self):
        # CPython never runs A.__init__ here, so `x` does not exist and b.x
        # would raise AttributeError; a zero-filled slot would answer 0.
        self._reject(
            "class A:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "class B(A):\n"
            "    def __init__(self):\n"
            "        self.y = 2\n"
            "b = B()\n"
            "raise SystemExit(b.y)\n",
            "does not assign inherited attribute",
        )

    def test_multiple_bases_are_rejected(self):
        self._reject(
            "class A:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "class B:\n"
            "    def __init__(self):\n"
            "        self.y = 2\n"
            "class C(A, B):\n"
            "    pass\n"
            "c = C()\n"
            "raise SystemExit(c.x)\n",
            "single inheritance",
        )

    def test_explicit_object_alongside_a_base_is_rejected(self):
        self._reject(
            "class A:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "class B(A, object):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "b = B()\n"
            "raise SystemExit(b.x)\n",
            "single inheritance",
        )

    def test_forward_base_is_rejected(self):
        self._reject(
            "class B(A):\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "class A:\n"
            "    pass\n"
            "b = B()\n"
            "raise SystemExit(b.x)\n",
            "defined earlier",
        )

    def test_non_class_base_is_rejected(self):
        self._reject(
            "def maker():\n"
            "    return 1\n"
            "class B(maker):\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "b = B()\n"
            "raise SystemExit(b.x)\n",
            "defined earlier",
        )

    def test_variable_cannot_change_between_parent_and_child(self):
        # Static dispatch resolves from the declared class, so a name that
        # could hold either would call the wrong tag() body.
        self._reject(
            "class A:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "    def tag(self):\n"
            "        return 1\n"
            "class B(A):\n"
            "    def tag(self):\n"
            "        return 2\n"
            "o = A()\n"
            "o = B()\n"
            "raise SystemExit(o.tag())\n",
            "cannot change class",
        )

    def test_branch_cannot_pick_between_parent_and_child(self):
        self._reject(
            "class A:\n"
            "    def __init__(self, v):\n"
            "        self.x = v\n"
            "    def tag(self):\n"
            "        return 1\n"
            "class B(A):\n"
            "    def tag(self):\n"
            "        return 2\n"
            "flag = 0\n"
            "for i in range(0, 1):\n"
            "    flag += 1\n"
            "if flag:\n"
            "    o = A(1)\n"
            "else:\n"
            "    o = B(2)\n"
            "raise SystemExit(o.tag())\n",
            "cannot change class",
        )

    def test_super_outside_init_is_rejected(self):
        self._reject(
            "class A:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "    def tag(self):\n"
            "        return 1\n"
            "class B(A):\n"
            "    def tag(self):\n"
            "        return super().tag() + 1\n"
            "b = B()\n"
            "raise SystemExit(b.tag())\n",
            "super()",
        )

    def test_conditional_super_is_rejected(self):
        self._reject(
            "class A:\n"
            "    def __init__(self, v):\n"
            "        self.x = v\n"
            "class B(A):\n"
            "    def __init__(self, flag):\n"
            "        if flag:\n"
            "            super().__init__(1)\n"
            "        self.y = 2\n"
            "b = B(0)\n"
            "raise SystemExit(b.y)\n",
            "super()",
        )

    def test_two_argument_super_is_rejected(self):
        self._reject(
            "class A:\n"
            "    def __init__(self, v):\n"
            "        self.x = v\n"
            "class B(A):\n"
            "    def __init__(self, v):\n"
            "        super(B, self).__init__(v)\n"
            "b = B(3)\n"
            "raise SystemExit(b.x)\n",
            "super()",
        )

    def test_super_without_a_base_is_rejected(self):
        self._reject(
            "class A:\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.x = 1\n"
            "a = A()\n"
            "raise SystemExit(a.x)\n",
            "super()",
        )

    def test_repeated_super_is_rejected(self):
        self._reject(
            "class A:\n"
            "    def __init__(self, v):\n"
            "        self.x = v\n"
            "class B(A):\n"
            "    def __init__(self, v):\n"
            "        super().__init__(v)\n"
            "        super().__init__(v + 1)\n"
            "b = B(3)\n"
            "raise SystemExit(b.x)\n",
            "only once",
        )

    def test_inherited_attribute_changing_kind_is_rejected(self):
        self._reject(
            "class A:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "class B(A):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.x: float = 1.5\n"
            "b = B()\n"
            "raise SystemExit(int(b.x))\n",
            "one slot cannot be both",
        )

    def test_attribute_added_in_a_subclass_method_is_rejected(self):
        self._reject(
            "class A:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "class B(A):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "    def grow(self):\n"
            "        self.y = 5\n"
            "b = B()\n"
            "b.grow()\n"
            "raise SystemExit(b.x)\n",
            "has no native attribute",
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
