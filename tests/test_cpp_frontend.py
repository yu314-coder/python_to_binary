"""A subset of C++ translated into C, the way the first C++ compiler did it.

py2bin has a C compiler and no C++ one, and writing a second compiler is a
project of its own. Translating is not: a class is a struct, a member function
is a free function whose first parameter is the object, and a constructor
initialises one in place. Everything downstream sees C.

The subset is small and stated, and what falls outside it is refused by name
rather than mistranslated into C that fails somewhere the author never wrote.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from py2bin.cpp_frontend import (
    CppTranslationError,
    is_cpp,
    translate,
    translate_unity,
)


_VEC = """#include <stdio.h>
class Vec {
public:
    int x;
    int y;
    Vec(int a, int b) { x = a; y = b; }
    int sum() { return x + y; }
    int scaled(int k);
};
int Vec::scaled(int k) { return sum() * k; }
int main(void) {
    Vec v(3, 4);
    printf("%d %d %d\\n", v.x, v.sum(), v.scaled(2));
    return 0;
}
"""


class Translating(unittest.TestCase):
    def test_a_class_becomes_a_struct_and_free_functions(self) -> None:
        out = translate(_VEC, "vec.cpp")
        self.assertIn("struct Vec {", out)
        self.assertIn("static void Vec__ctor(struct Vec *this, int a, int b)", out)
        self.assertIn("static int Vec__sum(struct Vec *this)", out)
        self.assertNotIn("class Vec", out)

    def test_a_member_name_is_reached_through_this(self) -> None:
        out = translate(_VEC, "vec.cpp")
        self.assertIn("this->x = a", out)
        self.assertIn("return this->x + this->y", out)

    def test_a_bare_method_call_passes_the_object(self) -> None:
        # `sum()` inside a member is a call on `this`, and C has no such thing.
        self.assertIn("Vec__sum(this)", translate(_VEC, "vec.cpp"))

    def test_a_call_with_no_arguments_gets_no_comma(self) -> None:
        # `Vec__sum(&v, )` is not C, and a replacement string cannot see
        # whether an argument follows.
        out = translate(_VEC, "vec.cpp")
        self.assertIn("Vec__sum(&v)", out)
        self.assertNotIn(", )", out)

    def test_a_declaration_runs_the_constructor(self) -> None:
        out = translate(_VEC, "vec.cpp")
        self.assertIn("struct Vec v; Vec__ctor(&v, 3, 4);", out)

    def test_directives_stay_above_the_code(self) -> None:
        # A method that calls printf needs <stdio.h> declared before it, not
        # wherever the author happened to write the include.
        out = translate(_VEC, "vec.cpp")
        self.assertLess(out.index("#include <stdio.h>"), out.index("struct Vec"))


class Inheriting(unittest.TestCase):
    _SHAPE = """class Shape {
public:
    int w;
    Shape(int a) { w = a; }
    int width() { return w; }
};
class Box : public Shape {
public:
    int h;
    Box(int a, int b) { w = a; h = b; }
    int area() { return width() * h; }
};
int main(void) { Box b(3, 4); return b.area(); }
"""

    def test_the_base_is_embedded_first(self) -> None:
        # First, so a pointer to the derived object is a pointer to the base.
        out = translate(self._SHAPE, "s.cpp")
        self.assertIn("struct Box {\n    struct Shape __base;", out)

    def test_an_inherited_member_resolves_through_the_base(self) -> None:
        self.assertIn("this->__base.w = a", translate(self._SHAPE, "s.cpp"))

    def test_an_inherited_method_is_called_on_the_base(self) -> None:
        out = translate(self._SHAPE, "s.cpp")
        self.assertIn("Shape__width(&this->__base)", out)
        self.assertIn("Box__area(&b)", out)


class Destructors(unittest.TestCase):
    _WITH = """class R {
public:
    int n;
    R() { n = 1; }
    ~R() { n = 0; }
};
int main(void) { R r; return 0; }
"""

    def test_it_runs_before_a_return_not_after_it(self) -> None:
        """Put only at the closing brace, it sat after `return 0;`.

        Which is worse than not writing a destructor at all: the code says it
        cleans up and never does.
        """

        out = translate(self._WITH, "r.cpp")
        self.assertIn("R__dtor(&r); return 0;", out)

    def test_returning_the_object_being_destroyed_is_refused(self) -> None:
        source = self._WITH.replace("return 0;", "return r.n;")
        with self.assertRaisesRegex(CppTranslationError, "has a destructor"):
            translate(source, "r.cpp")


class Refusals(unittest.TestCase):
    def _refused(self, source: str) -> str:
        with self.assertRaises(CppTranslationError) as caught:
            translate(source, "t.cpp")
        return str(caught.exception)

    def test_each_is_named_rather_than_mistranslated(self) -> None:
        for source, expected in (
            ("template<class T> T id(T v){return v;}", "templates"),
            ("class A{public: virtual int f(){return 1;}};", "virtual"),
            ("int main(void){ throw 1; }", "exceptions"),
            ("#include <iostream>\nint main(void){return 0;}", "standard library"),
            ("class A{public:int x;};\nint main(void){A*a=new A();return 0;}", "new"),
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self._refused(source))

    def test_a_word_containing_one_is_not_one(self) -> None:
        # `newest` is not `new`, and a refusal on it would be nonsense.
        out = translate(
            "class A{public: int newest;\n A(){ newest = 1; }};\n"
            "int main(void){ A a; return a.newest; }",
            "t.cpp",
        )
        self.assertIn("this->newest = 1", out)


class SeveralFiles(unittest.TestCase):
    def test_a_class_from_a_shared_header_is_emitted_once(self) -> None:
        """Translated per file, the struct appeared once per includer.

        A struct defined twice is not C, and the error named a line in a
        header the user had written correctly.
        """

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "k.hpp").write_text(
                "#ifndef K\n#define K\nclass K {\npublic:\n int n;\n K(){n=1;}\n"
                " int get(){return n;}\n};\n#endif\n",
                encoding="utf-8",
            )
            first = root / "a.cpp"
            first.write_text('#include "k.hpp"\nint helper(void){ K k; return k.get(); }\n', encoding="utf-8")
            second = root / "main.cpp"
            second.write_text('#include "k.hpp"\nint main(void){ K k; return k.get(); }\n', encoding="utf-8")
            out = translate_unity((second, first))
            self.assertEqual(out.count("struct K {"), 1)
            self.assertEqual(out.count("static void K__ctor"), 1)


class Naming(unittest.TestCase):
    def test_the_suffixes_that_mean_cpp(self) -> None:
        self.assertTrue(is_cpp(Path("a.cpp")))
        self.assertTrue(is_cpp(Path("a.hpp")))
        self.assertFalse(is_cpp(Path("a.c")))
        self.assertFalse(is_cpp(Path("a.h")))


if __name__ == "__main__":
    unittest.main()
