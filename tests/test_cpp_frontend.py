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
    # Shape has a constructor taking nothing on purpose: a base whose only
    # constructor takes arguments needs an initialiser list, which C++ demands
    # and this subset does not read - so it is refused, and the fixture stays
    # inside what is actually accepted.
    _SHAPE = """class Shape {
public:
    int w;
    Shape() { w = 0; }
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
            ("int main(void){ throw 1; }", "exceptions"),
            ("#include <iostream>\nint main(void){return 0;}", "standard library"),
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


class Subobjects(unittest.TestCase):
    """A class holding another, and what C does not do for you.

    C++ builds the base and every class-typed member before the constructor
    body runs. C does nothing at all, so the held object read whatever was on
    the stack - `car.total()` answered 125732 where it should have answered
    104. A wrong answer, not a failure, which is the worst kind to ship.
    """

    _COMPOSED = """class Engine {
public:
    int power;
    Engine() { power = 100; }
    int rate() { return power; }
};
class Car {
public:
    Engine motor;
    int wheels;
    Car() { wheels = 4; }
    int total() { return motor.rate() + wheels; }
};
int main(void) { Car c; return c.total(); }
"""

    def test_a_held_object_is_constructed_first(self) -> None:
        out = translate(self._COMPOSED, "c.cpp")
        constructor = out[out.index("Car__ctor"):]
        self.assertIn("Engine__ctor(&this->motor);", constructor)
        self.assertLess(
            constructor.index("Engine__ctor"), constructor.index("wheels = 4")
        )

    def test_a_held_object_is_a_receiver(self) -> None:
        # `motor.rate()` is a call on a member, which qualifying alone left as
        # `this->motor.rate()` - a struct with no such field.
        self.assertIn("Engine__rate(&this->motor)", translate(self._COMPOSED, "c.cpp"))

    def test_the_class_it_holds_is_defined_first(self) -> None:
        # C needs the complete type to lay the field out, and source order is
        # not that order.
        out = translate(self._COMPOSED, "c.cpp")
        self.assertLess(out.index("struct Engine {"), out.index("struct Car {"))

    def test_a_base_needing_arguments_is_refused(self) -> None:
        source = """class Base { public: int v; Base(int a) { v = a; } };
class Derived : public Base { public: Derived() { } };
int main(void) { Derived d; return 0; }
"""
        with self.assertRaisesRegex(CppTranslationError, "initialiser list"):
            translate(source, "d.cpp")


class Arrays(unittest.TestCase):
    _BANK = """class Cell {
public:
    int n;
    Cell() { n = 5; }
    int get() { return n; }
};
int main(void) { Cell bank[3]; return bank[0].get(); }
"""

    def test_every_element_is_default_constructed(self) -> None:
        out = translate(self._BANK, "b.cpp")
        self.assertIn("Cell__ctor(&bank[", out)
        self.assertIn("for (", out)

    def test_an_element_is_a_receiver(self) -> None:
        self.assertIn("Cell__get(&bank[0])", translate(self._BANK, "b.cpp"))


class DeepInheritance(unittest.TestCase):
    _CHAIN = """class A { public: int v; A() { v = 1; } };
class B : public A { public: B() { v = 2; } };
class C : public B { public: C() { v = 3; } int deep() { return v; } };
int main(void) { C c; return c.deep(); }
"""

    def test_a_name_two_levels_up_walks_both(self) -> None:
        # One hop named a member the middle class does not have.
        self.assertIn("this->__base.__base.v", translate(self._CHAIN, "c.cpp"))


class Scopes(unittest.TestCase):
    """A function is a scope, and the file is not one.

    Rewritten as a single body, a variable declared in one function was in
    scope for every later one - and its destructor was placed at the end of
    the *last* function in the file. The compiler then named an undeclared
    variable, pointing into somebody else's function.
    """

    _TWO = """class T { public: int id; T() { id = 7; } ~T() { id = 0; } };
int first(void) { T t; int held = t.id; return held; }
int main(void) { return first(); }
"""

    def test_a_destructor_stays_in_its_own_function(self) -> None:
        out = translate(self._TWO, "t.cpp")
        first = out[out.index("int first"):out.index("int main")]
        self.assertIn("T__dtor(&t);", first)
        self.assertNotIn("T__dtor", out[out.index("int main"):])

    def test_a_class_typed_parameter_can_be_called_on(self) -> None:
        # Declared in the head and used in the body, and the body is all the
        # rewriter is handed.
        source = """class I { public: int a; I() { a = 1; } int get() { return a; } };
int viaPointer(I *p) { return p->get(); }
int main(void) { I i; return viaPointer(&i); }
"""
        self.assertIn("I__get(p)", translate(source, "p.cpp"))


class PlainStructs(unittest.TestCase):
    def test_a_bare_struct_name_is_a_type_in_c_too(self) -> None:
        """`Plain p;` is C++ and a syntax error in C.

        The struct itself is left exactly as written - it is C already - but
        the name still has to work, which is what the typedef is for.
        """

        source = """struct Plain { int f; };
class K { public: int n; K() { n = 1; } };
int main(void) { Plain p; p.f = 9; K k; return p.f + k.n; }
"""
        out = translate(source, "s.cpp")
        self.assertIn("typedef struct Plain Plain;", out)
        self.assertIn("struct Plain { int f; };", out)


class AgainstARealCompiler(unittest.TestCase):
    """Each of these was found by building the same C++ with clang++ too.

    Reading the generated C says whether it is well formed. It cannot say
    whether it means the same thing, and every one of these compiled cleanly
    while meaning something else. `tools/cpp_differential.sh` is the harness;
    clang++ is the yardstick there, never a dependency of py2bin.
    """

    def test_a_literal_is_data_and_no_name_in_it_is_a_name(self) -> None:
        """`printf("outer\\n")` became `printf("outer\\this->n")`.

        The class has a member `n`, and `\\n` contains the letter. The program
        built and printed the wrong thing.
        """

        out = translate(
            'class O { public: int n; O() { n = 2; printf("outer\\n"); } };\n'
            "int main(void) { O o; return o.n; }\n",
            "o.cpp",
        )
        self.assertIn(r'printf("outer\n")', out)
        self.assertNotIn("this->n\"", out)

    def test_a_parameter_hides_the_member_it_is_named_after(self) -> None:
        """`int mix(int n) { return this->n + n; }` answered 200, not 105."""

        out = translate(
            "class C { public: int n; C() { n = 100; }\n"
            " int mix(int n) { return this->n + n; } };\n"
            "int main(void) { C c; return c.mix(5); }\n",
            "c.cpp",
        )
        self.assertIn("return this->n + n;", out)

    def test_a_format_string_does_not_hide_a_member(self) -> None:
        # `printf("n=%d a=%d")` contains `d a=`, which reads like a
        # declaration of `a` - hiding the member from its own method.
        out = translate(
            'class T { public: int a; T() { a = 2; }\n'
            ' void show() { printf("n=%d a=%d\\n", a, a); } };\n'
            "int main(void) { T t; t.show(); return 0; }\n",
            "t.cpp",
        )
        self.assertIn("this->a, this->a", out)

    def test_an_inherited_field_resolves_from_outside_the_class(self) -> None:
        # Methods followed the base chain from the start; fields did not.
        out = translate(
            "class B { public: int v; B() { v = 1; } };\n"
            "class D : public B { public: int w; D() { w = 2; } };\n"
            "int main(void) { D d; return d.v + d.w; }\n",
            "d.cpp",
        )
        self.assertIn("d.__base.v", out)

    def test_a_nested_block_is_its_own_scope(self) -> None:
        """The destructor landed at the end of the function, not the block.

        Which is both the wrong moment and, in C, a name that is not in scope
        there at all.
        """

        out = translate(
            "class R { public: int id; R() { id = 1; } ~R() { id = 0; } };\n"
            "int main(void) { { R r; int x = r.id; } return 0; }\n",
            "r.cpp",
        )
        inner = out[out.index("{ struct R r;"):out.index("return 0;")]
        self.assertIn("R__dtor(&r);", inner)

    def test_a_pointer_member_is_a_receiver(self) -> None:
        # `a.next->get()` stopped at the first hop.
        out = translate(
            "class N { public: int v; N *next; N() { v = 0; next = 0; }\n"
            " int get() { return v; } };\n"
            "int main(void) { N a; N b; a.next = &b; return a.next->get(); }\n",
            "n.cpp",
        )
        self.assertIn("N__get(a.next)", out)

    def test_a_member_object_is_a_receiver_from_outside(self) -> None:
        out = translate(
            "class A { public: int v; A() { v = 3; } int get() { return v; } };\n"
            "class B { public: A inner; B() { } };\n"
            "int main(void) { B b; return b.inner.get(); }\n",
            "b.cpp",
        )
        self.assertIn("A__get(&b.inner)", out)

    def test_an_array_inside_a_nested_block_is_still_an_array(self) -> None:
        # Blocks were rewritten before the enclosing scope was read, so a
        # `for` body did not know the array it indexes was one.
        out = translate(
            "class E { public: int n; E() { n = 2; } int get() { return n; } };\n"
            "int main(void) { E e[3]; int i; int t = 0;\n"
            " for (i = 0; i < 3; i++) { t = t + e[i].get(); }\n"
            " return t; }\n",
            "e.cpp",
        )
        self.assertIn("E__get(&e[i])", out)


class Namespaces(unittest.TestCase):
    """A namespace is scoping, and scoping is all it can be here.

    py2bin compiles one translation unit and has no linker, so there is
    nowhere for a second `helper` to live. Flattening is therefore the whole
    of what a namespace means - `N::thing` is `thing`, and
    `using namespace N;` is nothing at all - and the one thing flattening
    cannot survive is two of the same name, which is refused rather than
    resolved by whichever came last.
    """

    def test_a_namespace_is_removed_and_its_qualifier_with_it(self) -> None:
        out = translate(
            "namespace geom {\n"
            " class Point { public: int x; Point(int a) { x = a; } int get() { return x; } };\n"
            " int twice(int n) { return n * 2; }\n"
            "}\n"
            "int main(void) { geom::Point p(3); return p.get() + geom::twice(1); }\n",
            "g.cpp",
        )
        self.assertNotIn("namespace", out)
        self.assertNotIn("geom::", out)
        self.assertIn("Point__get(&p)", out)

    def test_a_class_inside_one_is_still_a_class(self) -> None:
        # Every pass looks for classes at the top level, so the wrapper has to
        # be gone before any of them runs.
        out = translate(
            "namespace n { class K { public: int v; K() { v = 1; } }; }\n"
            "int main(void) { n::K k; return k.v; }\n",
            "k.cpp",
        )
        self.assertIn("struct K {", out)
        self.assertIn("K__ctor", out)

    def test_using_namespace_becomes_nothing(self) -> None:
        out = translate(
            "namespace u { int f(void) { return 1; } }\n"
            "using namespace u;\n"
            "int main(void) { return f(); }\n",
            "u.cpp",
        )
        self.assertNotIn("using", out)

    def test_a_nested_namespace_owns_its_own_names(self) -> None:
        """Counting them twice looked like two namespaces declaring one class."""

        out = translate(
            "namespace outer { namespace inner {\n"
            " class D { public: int v; D() { v = 1; } };\n"
            "} }\n"
            "int main(void) { outer::inner::D d; return d.v; }\n",
            "d.cpp",
        )
        self.assertIn("struct D {", out)

    def test_an_anonymous_one_needs_no_name_to_flatten(self) -> None:
        out = translate(
            "namespace { class H { public: int v; H() { v = 5; } }; }\n"
            "int main(void) { H h; return h.v; }\n",
            "h.cpp",
        )
        self.assertIn("struct H {", out)

    def test_two_namespaces_declaring_the_same_name_are_refused(self) -> None:
        with self.assertRaisesRegex(CppTranslationError, "both declare"):
            translate(
                "namespace a { int helper(void) { return 1; } }\n"
                "namespace b { int helper(void) { return 2; } }\n"
                "int main(void) { return a::helper(); }\n",
                "c.cpp",
            )

    def test_an_out_of_line_member_is_not_a_namespace_qualifier(self) -> None:
        """`::` spells both, and stripping the wrong one loses the method."""

        out = translate(
            "namespace lib { class M { public: int v; M(); int get(); };\n"
            " M::M() { v = 11; }\n int M::get() { return v; }\n}\n"
            "int main(void) { lib::M m; return m.get(); }\n",
            "m.cpp",
        )
        self.assertIn("static void M__ctor(struct M *this)", out)
        self.assertIn("static int M__get(struct M *this)", out)


class Overloads(unittest.TestCase):
    """C has one function per name, and C++ has as many as the arguments differ.

    They are told apart by how many arguments they take, which is the one
    thing a call site always shows.
    """

    def test_two_constructors_get_two_names(self) -> None:
        out = translate(
            "class N{public:int v;\n N(){v=0;}\n N(int a){v=a;}\n};\n"
            "int main(void){ N x; N y(5); return y.v; }",
            "t.cpp",
        )
        self.assertIn("N__ctor__0(struct N *this)", out)
        self.assertIn("N__ctor__1(struct N *this, int a)", out)
        # And the call sites pick by count, not by order of declaration.
        self.assertIn("N__ctor__0(&x)", out)
        self.assertIn("N__ctor__1(&y, 5)", out)

    def test_a_method_alone_keeps_its_plain_name(self) -> None:
        # A suffix on every method would make the C unreadable for no gain.
        out = translate(
            "class N{public:int v;\n N(){v=0;}\n int get(){return v;}};\n"
            "int main(void){ N x; return x.get(); }",
            "t.cpp",
        )
        self.assertIn("N__get(struct N *this)", out)
        self.assertNotIn("N__get__0", out)


class Heap(unittest.TestCase):
    def test_new_becomes_a_function_that_allocates_then_constructs(self) -> None:
        out = translate(
            "class N{public:int v;\n N(int a){v=a;}};\n"
            "int main(void){ N *p = new N(3); delete p; return 0; }",
            "t.cpp",
        )
        self.assertIn("malloc(sizeof(struct N))", out)
        self.assertIn("N__delete(p);", out)
        # The allocator comes with it, so `new` works without <stdlib.h>.
        self.assertIn("#include <stdlib.h>", out)

    def test_a_file_that_never_allocates_gets_no_allocator(self) -> None:
        out = translate(
            "class N{public:int v;\n N(){v=1;}};\nint main(void){ N a; return a.v; }",
            "t.cpp",
        )
        self.assertNotIn("stdlib.h", out)
        self.assertNotIn("N__new", out)

    def test_delete_array_runs_one_destructor_per_element(self) -> None:
        out = translate(
            "class N{public:int v;\n N(){v=1;}\n ~N(){v=0;}};\n"
            "int main(void){ N *p = new N[4]; delete[] p; return 0; }",
            "t.cpp",
        )
        # The count is written in front of the block, because C++ asks for
        # every element to be destroyed and nothing else records how many.
        self.assertIn("*(unsigned long *)__block = __n;", out)
        self.assertIn("N__delete_array(p);", out)


class Virtual(unittest.TestCase):
    def test_a_polymorphic_class_carries_a_table_pointer(self) -> None:
        out = translate(
            "class A{public: virtual int f(){return 1;}};\n"
            "class B : public A{public: int f(){return 2;}};\n"
            "int main(void){ B b; A *p = &b; return p->f(); }",
            "t.cpp",
        )
        self.assertIn("void **__vptr;", out)
        self.assertIn("A__vtable", out)
        self.assertIn("B__vtable", out)
        # The call reads the slot rather than naming a function.
        self.assertIn("__vptr[0]", out)

    def test_the_derived_table_keeps_the_base_ordering(self) -> None:
        """A base pointer reads slot 1 expecting the base's second virtual.

        A derived class may add to the table and may replace what is in it,
        but may not move anything: the caller holding a base pointer has no
        way to know it is looking at a derived object.
        """

        out = translate(
            "class A{public: virtual int f(){return 1;} virtual int g(){return 2;}};\n"
            "class B : public A{public: int g(){return 3;} virtual int h(){return 4;}};\n"
            "int main(void){ B b; return b.g(); }",
            "t.cpp",
        )
        table = [line for line in out.splitlines() if "B__vtable" in line][0]
        self.assertIn("A__f", table.split("{")[1].split(",")[0])
        self.assertIn("B__g", table.split(",")[1])
        self.assertIn("B__h", table.split(",")[2])

    def test_a_non_polymorphic_class_carries_nothing(self) -> None:
        out = translate(
            "class A{public:int x;\n A(){x=1;}\n int f(){return x;}};\n"
            "int main(void){ A a; return a.f(); }",
            "t.cpp",
        )
        self.assertNotIn("__vptr", out)
        self.assertNotIn("vtable", out)

    def test_only_the_root_stores_the_pointer(self) -> None:
        # Two pointers in one object would leave the wrong one at the front.
        out = translate(
            "class A{public: virtual int f(){return 1;}};\n"
            "class B : public A{public: int f(){return 2;}};\n"
            "class C : public B{public: int f(){return 3;}};\n"
            "int main(void){ C c; return c.f(); }",
            "t.cpp",
        )
        self.assertEqual(out.count("void **__vptr;"), 1)


class References(unittest.TestCase):
    def test_a_reference_parameter_becomes_a_pointer_on_both_sides(self) -> None:
        out = translate(
            "class B{public:int v;\n B(){v=1;}};\n"
            "void bump(B &b){ b.v = b.v + 1; }\n"
            "int main(void){ B a; bump(a); return a.v; }",
            "t.cpp",
        )
        self.assertIn("void bump(struct B *b)", out)
        self.assertIn("b->v = b->v + 1", out)
        self.assertIn("bump(&a)", out)

    def test_a_reference_to_a_number_is_dereferenced_at_each_use(self) -> None:
        out = translate(
            "void swap(int &a, int &b){ int t = a; a = b; b = t; }\n"
            "int main(void){ int x = 1, y = 2; swap(x, y); return x; }",
            "t.cpp",
        )
        self.assertIn("void swap(int *a, int *b)", out)
        self.assertIn("(*a) = (*b);", out)
        self.assertIn("swap(&x, &y)", out)

    def test_the_binding_itself_is_not_dereferenced(self) -> None:
        """`int &r = v;` declares the pointer; every other use follows it."""

        out = translate(
            "int main(void){ int v = 1; int &r = v; r = r + 1; return v; }",
            "t.cpp",
        )
        self.assertIn("int *r = &(v);", out)
        self.assertIn("(*r) = (*r) + 1;", out)
        self.assertNotIn("(*r) = &", out)


class Templates(unittest.TestCase):
    """A template is a pattern, so it becomes one copy per set of arguments.

    The copies are named after what they were made from, which is what makes
    the C readable: `Box__int` rather than a hash.
    """

    def test_a_class_template_becomes_one_class_per_argument(self) -> None:
        out = translate(
            "template<typename T>\nclass Box{public: T v;\n Box(){v=0;}\n"
            " T get(){return v;}};\n"
            "int main(void){ Box<int> a; Box<double> b; return a.get(); }",
            "t.cpp",
        )
        self.assertIn("struct Box__int", out)
        self.assertIn("struct Box__double", out)
        self.assertIn("int Box__int__get(struct Box__int *this)", out)
        self.assertIn("double Box__double__get(struct Box__double *this)", out)
        # The pattern itself is not code and must not survive into the C.
        self.assertNotIn("template", out)

    def test_one_copy_however_many_times_it_is_asked_for(self) -> None:
        out = translate(
            "template<typename T>\nclass Box{public: T v;\n Box(){v=0;}};\n"
            "int main(void){ Box<int> a; Box<int> b; Box<int> c; return 0; }",
            "t.cpp",
        )
        self.assertEqual(out.count("struct Box__int {"), 1)

    def test_a_function_template_deduces_from_a_literal(self) -> None:
        out = translate(
            "template<typename T>\nT twice(T v){ return v + v; }\n"
            "int main(void){ return twice(5) + (int)twice(2.5); }",
            "t.cpp",
        )
        self.assertIn("int twice__int(int v)", out)
        self.assertIn("double twice__double(double v)", out)
        self.assertIn("twice__int(5)", out)

    def test_a_call_it_cannot_read_is_refused_with_the_spelling_that_works(
        self,
    ) -> None:
        """Guessing here would compile the wrong copy and run.

        Deduction covers literals and variables it can see declared. Anything
        else is refused, and the message says to write the argument out.
        """

        with self.assertRaises(CppTranslationError) as caught:
            translate(
                "template<typename T>\nT twice(T v){ return v + v; }\n"
                "int pick(void);\n"
                "int main(void){ return twice(pick()); }",
                "t.cpp",
            )
        self.assertIn("twice<type>", str(caught.exception))
