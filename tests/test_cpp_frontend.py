"""A subset of C++ translated into C, the way the first C++ compiler did it.

py2bin has a C compiler and no C++ one, and writing a second compiler is a
project of its own. Translating is not: a class is a struct, a member function
is a free function whose first parameter is the object, and a constructor
initialises one in place. Everything downstream sees C.

The subset is small and stated, and what falls outside it is refused by name
rather than mistranslated into C that fails somewhere the author never wrote.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from py2bin.c_frontend import CCompileError
from py2bin.c_native import compile_c_native
from py2bin.cpp_frontend import (
    CppTranslationError,
    inline_local_includes,
    is_cpp,
    translate,
    translate_unity,
)


def with_headers(source: str) -> str:
    """Translate source that includes one of py2bin's own C++ headers.

    `translate` is handed text that already has its includes resolved;
    `inline_local_includes` is what resolves them, and what supplies
    <vector>, <algorithm> and the rest. A test that calls `translate`
    directly on a file with `#include <algorithm>` is testing neither.
    """

    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "t.cpp"
        path.write_text(source, encoding="utf-8")
        return translate(
            inline_local_includes(path, [], set(), set()), str(path)
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

    def test_nothing_is_on_the_list_any_more(self) -> None:
        """The list was how the subset said what it was not.

        Every entry that was on it - templates, virtual functions, the heap,
        exceptions, the standard library - is something it does now. It is
        kept empty rather than deleted because a construct found unhandled
        belongs back on it: a refusal that names the feature is a better
        message than whatever the C compiler makes of the wreckage.
        """

        from py2bin.cpp_frontend import _REFUSED

        self.assertEqual(_REFUSED, ())

    def test_a_word_containing_one_is_not_one(self) -> None:
        # `newest` is not `new`, and a refusal on it would be nonsense.
        out = translate(
            "class A{public: int newest;\n A(){ newest = 1; }};\n"
            "int main(void){ A a; return a.newest; }",
            "t.cpp",
        )
        self.assertIn("this->newest = 1", out)

    def test_the_pragma_operator_is_left_for_the_preprocessor(self) -> None:
        """`_Pragma` is a directive in disguise, not C++ this translates.

        The preprocessor runs after this and reads it. What matters here is
        that nothing on the way past rewrites it, because the string it
        carries is a directive and not an expression.
        """

        out = translate(
            '#define HINT(x) _Pragma(#x)\nHINT(acme unroll)\n'
            '_Pragma("acme hint")\nint main(void){ return 0; }\n',
            "t.cpp",
        )
        self.assertIn('_Pragma(#x)', out)
        self.assertIn('_Pragma("acme hint")', out)


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

    def test_one_file_spelled_two_ways_is_pasted_once(self) -> None:
        """`"Foo.h"` and `"foo.h"` are one file where case does not count.

        The pasted-once check compared the settled path, which says nothing
        about case, so both includes were read and the translator was handed
        two copies of every class in the header. That came out as `two
        definitions of foo_value() take the same arguments`, pointing at the
        header rather than at the second include - and an include guard did
        not save it, because this pass runs before the preprocessor and has
        no `#ifndef`.
        """

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "Foo.h").write_text(
                "#pragma once\nclass Foo { public: int n; Foo(){n=1;} };\n"
                "inline int foo_value(Foo f) { return f.n; }\n",
                encoding="utf-8",
            )
            if not (root / "foo.h").is_file():
                self.skipTest("this filesystem tells Foo.h and foo.h apart")
            source = root / "main.cpp"
            source.write_text(
                '#include "Foo.h"\n#include "foo.h"\n'
                "int main(void){ Foo f; return foo_value(f); }\n",
                encoding="utf-8",
            )
            pasted = inline_local_includes(source, [], set(), set())
            self.assertEqual(pasted.count("class Foo"), 1)
            # And the whole way through: two copies of a free function is
            # what the translator refused, and the message named the header.
            translate(pasted, str(source))

    def test_a_header_reached_through_a_symlink_is_pasted_once(self) -> None:
        """The other half of the same question: two names, one file.

        A settled path does follow a symlink, so this one worked already -
        it is here so that asking the filesystem for identity keeps answering
        it, whichever way the identity is asked for.
        """

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "real.hpp").write_text(
                "#pragma once\nclass R { public: int n; R(){n=2;} };\n"
                "inline int r_value(R r) { return r.n; }\n",
                encoding="utf-8",
            )
            try:
                (root / "alias.hpp").symlink_to("real.hpp")
            except (OSError, NotImplementedError):
                self.skipTest("this filesystem has no symlinks")
            source = root / "main.cpp"
            source.write_text(
                '#include "real.hpp"\n#include "./alias.hpp"\n'
                "int main(void){ R r; return r_value(r); }\n",
                encoding="utf-8",
            )
            pasted = inline_local_includes(source, [], set(), set())
            self.assertEqual(pasted.count("class R"), 1)
            translate(pasted, str(source))

    def test_the_same_header_quoted_and_angled_is_pasted_once(self) -> None:
        """Which brackets an include wears says where to look, not what.

        Found on the search path either way, it is the same file, and pasting
        it a second time would define the class twice.
        """

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "shared.hpp").write_text(
                "#pragma once\nclass S { public: int n; S(){n=3;} };\n"
                "inline int s_value(S s) { return s.n; }\n",
                encoding="utf-8",
            )
            source = root / "main.cpp"
            source.write_text(
                '#include "shared.hpp"\n#include <shared.hpp>\n'
                "int main(void){ S s; return s_value(s); }\n",
                encoding="utf-8",
            )
            pasted = inline_local_includes(source, (str(root),), set(), set())
            self.assertEqual(pasted.count("class S"), 1)
            translate(pasted, str(source))


class WhereItIsWritten(unittest.TestCase):
    """`__LINE__` and `__FILE__` against the file the user opened.

    The preprocessor runs after the paste and after the classes have been
    rewritten, so by then the line a token sits on is a line of the joined
    unit and the only file name left is the one the build started from.
    Both are answered while each file is still its own text.
    """

    def test_a_header_answers_its_own_line_and_its_own_name(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            header = root / "k.hpp"
            header.write_text(
                "#pragma once\n"
                "// a line\n"
                "inline int written_at(void){ return __LINE__; }\n"
                "inline const char *written_in(void){ return __FILE__; }\n",
                encoding="utf-8",
            )
            entry = root / "main.cpp"
            entry.write_text(
                '#include "k.hpp"\n'
                "int main(void){ return written_at() + __LINE__; }\n",
                encoding="utf-8",
            )
            out = inline_local_includes(entry, (), set(), set())
            self.assertIn("return 3; }", out)
            self.assertIn(f'return "{header}"; }}', out)
            self.assertIn("written_at() + 2;", out)

    def test_the_class_rewriting_does_not_move_it(self) -> None:
        """A method body is hoisted out of its class and lands elsewhere.

        Answered before that happens, so the number goes with the statement
        rather than staying behind at the line the statement left.
        """

        out = with_headers(
            "class K {\npublic:\n int asked(){ return __LINE__; }\n};\n"
            "int main(void){ K k; return k.asked(); }\n"
        )
        self.assertIn("return 3;", out)
        self.assertNotIn("__LINE__", out)

    def test_a_literal_that_spells_one_is_left_alone(self) -> None:
        out = with_headers('const char *s = "__LINE__ __FILE__";\n')
        self.assertIn('"__LINE__ __FILE__"', out)

    def test_a_macro_body_that_names_one_is_left_for_the_preprocessor(self) -> None:
        """Its value is the line of the *call*, which nothing here can know.

        Calls are expanded by the preprocessor, long after this text has been
        rewritten, so the line it reports is one in the translated unit and
        not the one the macro was called on. That is imprecise and it is left
        that way on purpose: the number goes in a log message, and refusing
        to build every program that owns a LOG macro is a worse answer than
        an approximate line in one. What is written in code, below, is exact.
        """

        with tempfile.TemporaryDirectory() as scratch:
            entry = Path(scratch) / "main.cpp"
            entry.write_text(
                "#include <cstdio>\n"
                "\n"
                '#define WHERE() printf("%d\\n", __LINE__)\n'
                "int main(void){ WHERE(); return 0; }\n",
                encoding="utf-8",
            )
            out = inline_local_includes(entry, (), set(), set())
            # Untouched in the directive, so the preprocessor still has one
            # to answer; the call site keeps naming the macro either way.
            self.assertIn("__LINE__", out)
            self.assertIn("WHERE()", out)


class Naming(unittest.TestCase):
    def test_the_suffixes_that_mean_cpp(self) -> None:
        self.assertTrue(is_cpp(Path("a.cpp")))
        self.assertTrue(is_cpp(Path("a.hpp")))
        self.assertFalse(is_cpp(Path("a.c")))
        self.assertFalse(is_cpp(Path("a.h")))


class ComVtableLayout(unittest.TestCase):
    """A COM interface's layout is its ABI, and nothing reports getting it wrong.

    A call through a vtable is a load and a branch: a method one slot away
    from where the object actually has it jumps into whatever is next, on an
    object somebody else built, in a process that was working a moment ago.
    """

    _INTERFACE = """#include <unknwn.h>
class IThing : public IUnknown {
public:
    virtual HRESULT DoIt(int n) = 0;
    virtual HRESULT Twice(int n) = 0;
};
class Thing : public IThing {
public:
    HRESULT QueryInterface(REFIID riid, void **o) { *o = (void *)this; return S_OK; }
    unsigned long AddRef() { return 1; }
    unsigned long Release() { return 0; }
    HRESULT DoIt(int n) { return (HRESULT)n; }
    HRESULT Twice(int n) { return (HRESULT)n; }
};
int main() { Thing t; return (int)t.DoIt(0); }
"""

    def _table(self, out: str, owner: str) -> "list[str]":
        found = re.search(
            rf"static void \*{owner}__vtable\[(\d+)\] = \{{([^}}]*)\}}", out
        )
        self.assertIsNotNone(found, f"no vtable emitted for {owner}")
        slots = [entry.strip() for entry in found.group(2).split(",")]
        self.assertEqual(len(slots), int(found.group(1)))
        return slots

    def test_iunknown_is_three_slots_and_not_four(self):
        """py2bin's own <unknwn.h> once gave IUnknown a virtual destructor,
        which is a fourth entry - and pushed every derived interface's methods
        one slot down from where COM puts them."""

        out = with_headers(self._INTERFACE)
        self.assertEqual(len(self._table(out, "IUnknown")), 3)

    def test_a_derived_interface_starts_at_slot_three(self):
        out = with_headers(self._INTERFACE)
        slots = self._table(out, "Thing")
        self.assertEqual(len(slots), 5)
        self.assertIn("QueryInterface", slots[0])
        self.assertIn("AddRef", slots[1])
        self.assertIn("Release", slots[2])
        self.assertIn("DoIt", slots[3])
        self.assertIn("Twice", slots[4])


class PartialSpecialisation(unittest.TestCase):
    """A class template written again for a shape of argument.

    Which copy a use gets is decided by which pattern is narrower, and
    getting that wrong is not a build failure - it is a different answer.
    """

    _SHAPES = """#include <stdio.h>
template <class T> struct Name { static const int id = 0; };
template <> struct Name<int> { static const int id = 1; };
template <class T> struct Name<T *> { static const int id = 2; };
template <class T> struct Name<T **> { static const int id = 3; };
int main() {
    return Name<double>::id + Name<int>::id
         + Name<double *>::id + Name<double **>::id;
}
"""

    def test_the_narrower_pattern_wins(self):
        """A copy is made for each use, and each use gets the narrowest
        pattern that fits it - the general one only where none does."""

        out = with_headers(self._SHAPES)
        self.assertIn("Name__double__id = 0", out)
        self.assertIn("Name__int__id = 1", out)
        self.assertIn("Name__double_p__id = 2", out)
        self.assertIn("Name__double_p_p__id = 3", out)

    def test_a_deeper_pattern_beats_a_shallower_one(self):
        """`T **` and `T *` both fit `int **`; C++ takes the first."""

        from py2bin.cpp_frontend import _how_narrow, _narrower_copy

        parameters = {"T"}
        self.assertGreater(
            _how_narrow(["T **"], parameters), _how_narrow(["T *"], parameters)
        )
        self.assertGreater(
            _how_narrow(["int"], parameters), _how_narrow(["T **"], parameters)
        )
        # `<T, T>` says the two arguments are the same, which `<T, U>` does not.
        self.assertGreater(
            _how_narrow(["T", "T"], parameters),
            _how_narrow(["T", "U"], {"T", "U"}),
        )

    def test_a_parameter_named_twice_has_to_match_twice(self):
        from py2bin.cpp_frontend import _narrower_copy

        entries = [([("T", True, False)], ["T", "T"], "{ int a; }", "struct")]
        self.assertIsNotNone(_narrower_copy(entries, ["int", "int"]))
        self.assertIsNone(_narrower_copy(entries, ["int", "char"]))

    def test_a_shape_that_does_not_fit_is_not_used(self):
        from py2bin.cpp_frontend import _narrower_copy

        entries = [([("T", True, False)], ["T *"], "{ int a; }", "struct")]
        self.assertIsNotNone(_narrower_copy(entries, ["int *"]))
        self.assertIsNone(_narrower_copy(entries, ["int"]))

    def test_two_classes_may_each_have_a_member_called_value(self):
        """Keyed by the member's name alone, the second took the first's -
        and a traits header is nothing but classes with a `value`."""

        out = with_headers(
            "struct A { static const int value = 1; };\n"
            "struct B { static const int value = 2; };\n"
            "int main() { return A::value + B::value; }\n"
        )
        self.assertIn("A__value", out)
        self.assertIn("B__value", out)
        self.assertNotIn("A::value", out)
        self.assertNotIn("B::value", out)

    def test_a_class_with_only_static_members_is_still_a_type(self):
        """C has no empty struct, and a traits class has no data at all."""

        out = with_headers(
            "struct A { static const int value = 1; };\n"
            "int main() { return A::value; }\n"
        )
        self.assertIn("__empty", out)


class PacksAndGuards(unittest.TestCase):
    """A parameter standing for however many, and a return type that decides
    whether the function is a candidate at all."""

    def test_a_pack_is_counted(self):
        out = with_headers(
            "template <class... Ts> struct arity "
            "{ static const int value = sizeof...(Ts); };\n"
            "int main() { return arity<int, char, double>::value; }\n"
        )
        self.assertIn("arity__int_char_double__value = 3", out)

    def test_a_pack_may_stand_for_nothing(self):
        out = with_headers(
            "template <class... Ts> struct arity "
            "{ static const int value = sizeof...(Ts); };\n"
            "int main() { return arity<>::value; }\n"
        )
        self.assertIn("= 0", out)

    def test_a_recursive_pack_stops_on_the_ordinary_function(self):
        """C++ prefers an ordinary function to a copy of a template, which is
        what ends the recursion. Renamed wholesale, each copy called itself
        with one argument fewer than it takes."""

        out = with_headers(
            "static int total(int a) { return a; }\n"
            "template <class... Rest> static int total(int a, Rest... rest) "
            "{ return a + total(rest...); }\n"
            "int main() { return total(1, 2, 3); }\n"
        )
        self.assertIn("total__int_int(", out)
        self.assertIn("total__int(", out)
        # The two-argument copy calls the one-argument copy, not itself.
        two = out[out.index("total__int_int("):]
        self.assertIn("total__int(", two[: two.index("}")])

    def test_a_guard_that_says_no_takes_the_function_out_of_the_running(self):
        out = with_headers(
            "template <bool B, class T> struct enable_if {};\n"
            "template <class T> struct enable_if<true, T> { typedef T type; };\n"
            "template <class T> struct is_pointer "
            "{ static const bool value = false; };\n"
            "template <class T> struct is_pointer<T *> "
            "{ static const bool value = true; };\n"
            "template <class T> typename enable_if<is_pointer<T>::value, int>::type"
            " kind(T v) { return 1; }\n"
            "template <class T> typename enable_if<!is_pointer<T>::value, int>::type"
            " kind(T v) { return 2; }\n"
            "int main() { int n = 5; return kind(&n) + kind(n); }\n"
        )
        # One copy per call, each from the pattern whose guard held.
        self.assertIn("kind__int_p", out)
        self.assertIn("kind__int", out)
        self.assertNotIn("enable_if", out.split("int main")[-1])

    def test_a_boolean_has_one_spelling(self):
        """`true` and `1` are the same answer, and a copy is named after it -
        two spellings made two copies of the same class."""

        from py2bin.cpp_frontend import _settled

        made = {"is_pointer__int": "struct is_pointer__int "
                                   "{ static const bool value = false; };"}
        self.assertEqual(_settled("is_pointer<int>::value", made), "0")
        self.assertEqual(_settled("!is_pointer<int>::value", made), "1")
        self.assertEqual(_settled("true", made), "1")
        self.assertEqual(_settled("!false", made), "1")
        self.assertIsNone(_settled("is_pointer<char>::value", made))


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


class OpaqueTypes(unittest.TestCase):
    """A name declared here and defined somewhere this file never sees.

    The PIMPL spelling, and every C handle a header hands out. Only pointers
    to it are ever written, which is all C++ asks for - and all C asks for
    too, once it has been told the name is a type at all.
    """

    def test_a_type_never_defined_here_keeps_its_name(self) -> None:
        """`Session *inside;` reached the C front end as a declaration of
        nothing, because the forward declaration that named the type had been
        dropped along with the ones that had a body below them."""

        out = translate(
            "class Session;\n"
            "struct Door { Session *inside; int tag; };\n"
            "int main(void) { Door d; d.inside = 0; d.tag = 1; return d.tag; }\n",
            "d.cpp",
        )
        self.assertIn("typedef struct Session Session;", out)
        self.assertNotIn("struct Session {", out)

    def test_a_forward_declaration_of_a_class_defined_here_still_goes(self) -> None:
        """The typedefs this emits already name it, so the line has no work."""

        out = translate(
            "class Later;\n"
            "class Later { public: int v; Later() { v = 4; } };\n"
            "int main(void) { Later l; return l.v; }\n",
            "l.cpp",
        )
        self.assertEqual(out.count("typedef struct Later Later;"), 1)

    def test_a_forward_declared_union_keeps_the_word_union(self) -> None:
        out = translate(
            "union Cell;\n"
            "struct Slot { Cell *held; };\n"
            "int main(void) { Slot s; s.held = 0; return 0; }\n",
            "c.cpp",
        )
        self.assertIn("typedef union Cell Cell;", out)

    def test_taking_it_apart_is_refused_rather_than_guessed(self) -> None:
        """Naming the type is not the same as knowing its shape. A `sizeof`
        or a member reach the C front end, which has the answer already."""

        with tempfile.TemporaryDirectory() as scratch:
            for body, complaint in (
                ("int main(void) { return (int)sizeof(Impl); }", "complete type"),
                ("int peek(Impl *p) { return p->v; }\n"
                 "int main(void) { return peek(0); }", "incomplete"),
                ("int main(void) { Impl one; return (int)(long)&one; }",
                 "incomplete"),
            ):
                source = Path(scratch) / "o.cpp"
                source.write_text(f"struct Impl;\n{body}\n", encoding="utf-8")
                with self.subTest(body=body):
                    with self.assertRaisesRegex(CCompileError, complaint):
                        compile_c_native(
                            source,
                            Path(scratch) / "o.bin",
                            target="darwin-arm64",
                            clean=True,
                        )


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

    def test_two_namespaces_with_a_variable_of_one_name_are_refused(self) -> None:
        """The one clash the C front end below cannot see for itself.

        `int count;` in one and `int count = 5;` in the other are two
        variables in C++ and, once the wrappers are gone, a tentative
        definition and its definition in C - which is one variable. The
        program built and wrote both namespaces' state into the same word.
        """

        with self.assertRaisesRegex(CppTranslationError, "both declare 'count'"):
            translate(
                "namespace a { int count; }\n"
                "namespace b { int count = 5; }\n"
                "int main(void) { a::count = 1; return a::count + b::count; }\n",
                "v.cpp",
            )

    def test_a_variable_clashing_with_one_outside_is_refused(self) -> None:
        with self.assertRaisesRegex(
            CppTranslationError, "outside every namespace"
        ):
            translate(
                "int count;\n"
                "namespace a { int count = 5; }\n"
                "int main(void) { count = 1; return count + a::count; }\n",
                "g.cpp",
            )

    def test_a_local_inside_one_is_not_the_namespaces_own_name(self) -> None:
        """What a function calls its locals is nobody else's business.

        Read together with the namespace's own declarations, two namespaces
        whose functions each had an `int n;` looked like a clash, and a
        program clang++ builds was refused.
        """

        out = translate(
            "namespace a { int base = 1; int one(void) { int n = 2; return n; } }\n"
            "namespace b { int other = 3; int two(void) { int n = 4; return n; } }\n"
            "int main(void) { return a::one() + b::two() + a::base + b::other; }\n",
            "n.cpp",
        )
        self.assertIn("int base = 1;", out)
        self.assertIn("int other = 3;", out)

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
                "int main(void){ return twice(pick()); }",
                "t.cpp",
            )
        self.assertIn("twice<type>", str(caught.exception))

    def test_a_prototype_says_what_a_call_answers(self) -> None:
        """A declaration says the same thing a definition does.

        `int pick(void);` is how C++ writes a function whose body is
        elsewhere, and deduction reads it the same way it reads a definition -
        it used to refuse this, which is not something C++ does.
        """

        out = translate(
            "template<typename T>\nT twice(T v){ return v + v; }\n"
            "int pick(void);\n"
            "int main(void){ return twice(pick()); }",
            "t.cpp",
        )
        self.assertIn("twice__int", out)


class ComHeaders(unittest.TestCase):
    """py2bin's own `<unknwn.h>`, because nobody publishes one.

    Every open implementation of the Windows API generates `unknwn.h` from an
    `.idl` at build time, and the vendor's own ships inside a toolchain - so
    there is no file to fetch. What COM *is* is a struct whose first member
    points at a table of function pointers, which is what py2bin lays a class
    with virtual methods out as, so the header says that in C++.
    """

    def test_an_interface_derives_from_it_and_dispatches(self) -> None:
        out = with_headers(
            "#include <unknwn.h>\n"
            "class ICounter : public IUnknown {\n"
            "public: virtual HRESULT Next(int *out) = 0; };\n"
            "class Counter : public ICounter {\n"
            "public: unsigned long refs; int value;\n"
            "  Counter() { refs = 1; value = 0; }\n"
            "  HRESULT QueryInterface(REFIID riid, void **object)"
            " { *object = this; return S_OK; }\n"
            "  unsigned long AddRef() { refs = refs + 1; return refs; }\n"
            "  unsigned long Release() { refs = refs - 1; return refs; }\n"
            "  HRESULT Next(int *out) { value = value + 1; *out = value;"
            " return S_OK; } };\n"
            "int main(){ Counter c; ICounter *p = &c; int got = 0;\n"
            "  p->Next(&got); return got; }\n"
        )
        # The interface's own table, and a call that goes through it.
        self.assertIn("Counter__vtable", out)
        self.assertIn("__vptr", out)
        # HRESULT and GUID are asked for rather than written out again: they
        # are <wtypes.h>'s, and py2bin's own <unknwn.h> declaring them too
        # made the same names against two different structs for any program
        # reaching both. A guard would not answer it - the translator moves
        # every directive to the top, so a `#ifndef` around a declaration
        # ends up above what it was meant to guard.
        self.assertIn("#include <wtypes.h>", out)
        self.assertNotIn("typedef long HRESULT;", out)
        self.assertLess(out.index("#include <wtypes.h>"), out.index("HRESULT Counter"))

    def test_the_header_is_pasted_before_the_translation(self) -> None:
        """It has to be: the translator reads classes, and the C preprocessor
        runs after it - so a class declared in a preprocessor header would be
        invisible to the very pass that needs it.
        """

        from py2bin.cpp_frontend import _BUILTIN_CPP_HEADERS

        self.assertIn("unknwn.h", _BUILTIN_CPP_HEADERS)
        self.assertIn("class IUnknown", _BUILTIN_CPP_HEADERS["unknwn.h"])


class OverloadsByType(unittest.TestCase):
    """Where two overloads take the same number, the types decide.

    The type of a literal is what it is written as; the type of a name is
    read from where it was declared. Anything else is refused, because the
    alternative is compiling the wrong one and running.
    """

    def test_the_same_count_is_told_apart_by_the_types(self) -> None:
        out = translate(
            "class L{public:int n;\n L(){n=0;}\n"
            " int show(int v){return v;}\n"
            " int show(const char *s){return s[0];}};\n"
            'int main(void){ L l; return l.show(1) + l.show("a"); }',
            "t.cpp",
        )
        self.assertIn("L__show__1__int(struct L *this, int v)", out)
        self.assertIn("L__show__1__char_p(struct L *this, const char *s)", out)
        self.assertIn("L__show__1__int(&l, 1)", out)
        self.assertIn('L__show__1__char_p(&l, "a")', out)

    def test_a_name_is_looked_up_where_it_was_declared(self) -> None:
        out = translate(
            "class L{public:int n;\n L(){n=0;}\n"
            " int show(int v){return v;}\n"
            " int show(double v){return (int)v;}};\n"
            "int main(void){ L l; double d = 1.5; return l.show(d); }",
            "t.cpp",
        )
        self.assertIn("L__show__1__double(&l, d)", out)

    def test_one_it_cannot_choose_between_is_refused(self) -> None:
        with self.assertRaises(CppTranslationError) as caught:
            translate(
                "class L{public:int n;\n L(){n=0;}\n"
                " int show(int v){return v;}\n"
                " int show(double v){return (int)v;}};\n"
                "int main(void){ L l; return l.show(pick()); }",
                "t.cpp",
            )
        self.assertIn("cannot tell which is meant", str(caught.exception))

    def test_a_prototype_is_enough_to_choose_between_them(self) -> None:
        out = translate(
            "int pick(void);\n"
            "class L{public:int n;\n L(){n=0;}\n"
            " int show(int v){return v;}\n"
            " int show(double v){return (int)v;}};\n"
            "int main(void){ L l; return l.show(pick()); }",
            "t.cpp",
        )
        self.assertIn("L__show__1__int", out)


class FileScopeObjects(unittest.TestCase):
    def test_a_global_object_is_in_scope_in_every_function(self) -> None:
        out = translate(
            "class C{public:int n;\n C(){n=0;}\n int bump(){n=n+1;return n;}};\n"
            "C shared;\n"
            "int use(void){ return shared.bump(); }\n"
            "int main(void){ return use(); }",
            "t.cpp",
        )
        self.assertIn("C__bump(&shared)", out)

    def test_its_constructor_runs_before_anything_else(self) -> None:
        """C++ builds them before `main`, and C has no place to put that.

        The first thing `main` does is what C++ had already done, which is the
        closest a translation to C can get without inventing a startup hook.
        """

        out = translate(
            "class C{public:int n;\n C(){n=7;}};\nC shared;\n"
            "int main(void){ return shared.n; }",
            "t.cpp",
        )
        entry = out[out.index("int main"):]
        self.assertIn("C__ctor(&shared);", entry.split("\n")[0])


class Streams(unittest.TestCase):
    def test_a_chain_becomes_one_call_around_another(self) -> None:
        """`cout << a << b` is two calls, and the second is called on the first.

        The generic operator rewriter wants a name on the left, and after the
        first `<<` there is a call there instead - so the chain is read whole.
        """

        out = translate(
            'class S{public:int f;\n S(){f=1;}\n S &operator<<(int v){return *this;}\n'
            ' S &operator<<(const char *v){return *this;}};\n'
            'S out;\nint main(void){ out << "a" << 2; return 0; }',
            "t.cpp",
        )
        self.assertIn("S__op_shl__1__char_p(&out,", out)
        self.assertIn("S__op_shl__1__int(", out)
        # The inner call's result is the outer call's receiver: each `<<`
        # hands the stream back as the address its reference became, so the
        # next one takes it directly and nothing is dereferenced in between.
        self.assertIn('S__op_shl__1__int(S__op_shl__1__char_p(&out, "a"), 2)', out)


class Exceptions(unittest.TestCase):
    """No unwinder, so the propagation is written out where it happens.

    A `throw` sets a flag and returns; every caller tests the flag right
    after the call. That is exact as long as the test lands where the call
    did, which is why a statement holding one is split.
    """

    def test_a_throw_sets_the_flag_and_leaves(self) -> None:
        out = translate(
            "int risky(int n){ if (n < 0) throw 7; return n; }\n"
            "int main(void){ try { return risky(-1); } catch (int e) { return e; } }",
            "t.cpp",
        )
        self.assertIn("__py2bin_thrown = 1", out)
        self.assertIn("__py2bin_in_flight = (long)(7)", out)
        # And the caller looks, immediately after the call.
        self.assertIn("if (__py2bin_thrown)", out)

    def test_the_handler_is_reached_by_a_jump_from_where_it_happened(self) -> None:
        out = translate(
            "int risky(int n){ if (n < 0) throw 7; return n; }\n"
            "int main(void){ try { risky(-1); } catch (int e) { return e; } return 0; }",
            "t.cpp",
        )
        self.assertIn("goto __py2bin_catch_1;", out)
        self.assertIn("__py2bin_catch_1: ;", out)
        self.assertIn("int e = (int)__py2bin_in_flight;", out)
        # The handler is jumped over when nothing was thrown.
        self.assertIn("goto __py2bin_done_1;", out)

    def test_a_file_that_never_throws_declares_no_state(self) -> None:
        out = translate("int main(void){ return 0; }", "t.cpp")
        self.assertNotIn("__py2bin_thrown", out)

    def test_a_call_behind_a_short_circuit_is_refused(self) -> None:
        """Splitting it would run it when C++ says it must not.

        `a() || b()` runs `b` only if `a` was false. Lifting `b` to its own
        statement runs it always, which is a different program - so this says
        so instead of quietly writing one.
        """

        with self.assertRaises(CppTranslationError) as caught:
            translate(
                "int risky(int n){ if (n < 0) throw 1; return n; }\n"
                "int main(void){ if (1 || risky(-1)) { return 0; } return 1; }",
                "t.cpp",
            )
        self.assertIn("&&", str(caught.exception))


class Destructors(unittest.TestCase):
    def test_a_return_inside_a_block_destroys_the_scopes_around_it(self) -> None:
        """It leaves the function, not just the block it is written in.

        Only the block's own objects were destroyed, so an object made in the
        function and returned from inside an `if` was never destroyed at all.
        """

        out = translate(
            "class R{public:int n;\n R(){n=1;}\n ~R(){n=0;}};\n"
            "int f(int k){ R r; if (k) { return 1; } return 2; }",
            "t.cpp",
        )
        body = out[out.index("int f("):]
        first = body[: body.index("return 1;")]
        self.assertIn("R__dtor(&r);", first)


class ThrownObjects(unittest.TestCase):
    def test_an_object_is_copied_to_the_heap_and_the_flag_carries_its_address(
        self,
    ) -> None:
        """A word cannot hold a struct, and the frame that threw it is gone.

        So a copy outlives the frame, which is what an exception object is.
        """

        out = translate(
            "class Err{public:int code;\n Err(){code=5;}};\n"
            "int f(void){ Err e; throw e; }\n"
            "int main(void){ try { f(); } catch (Err e) { return e.code; } return 0; }",
            "t.cpp",
        )
        self.assertIn("malloc(sizeof(struct Err))", out)
        self.assertIn("__py2bin_in_flight = (long)__py2bin_raised", out)
        # And the handler gets the copy back, declared then assigned because
        # py2bin's C takes `o = *p;` and not `struct V o = *p;`.
        self.assertIn("__py2bin_in_flight;", out)

    def test_a_number_still_goes_in_the_word_itself(self) -> None:
        out = translate(
            "int f(void){ throw 7; }\n"
            "int main(void){ try { f(); } catch (int e) { return e; } return 0; }",
            "t.cpp",
        )
        self.assertIn("__py2bin_in_flight = (long)(7)", out)
        self.assertNotIn("malloc", out)


class NestedTry(unittest.TestCase):
    def test_a_try_inside_a_try_gets_its_own_handler(self) -> None:
        out = translate(
            "int f(int n){ if (n < 0) throw 1; return n; }\n"
            "int main(void){\n"
            "  try { try { f(-1); } catch (int e) { throw 2; } }\n"
            "  catch (int e) { return e; }\n"
            "  return 0; }",
            "t.cpp",
        )
        self.assertIn("__py2bin_catch_1", out)
        self.assertIn("__py2bin_catch_2", out)

    def test_a_throw_in_a_handler_goes_outward(self) -> None:
        """A handler is not inside its own try, so it cannot catch itself."""

        out = translate(
            "int f(int n){ if (n < 0) throw 1; return n; }\n"
            "int main(void){\n"
            "  try { try { f(-1); } catch (int a) { throw 2; } }\n"
            "  catch (int b) { return b; }\n"
            "  return 0; }",
            "t.cpp",
        )
        inner = out[out.index("int a = "):]
        # The rethrow jumps to the outer handler, not back to the inner one.
        self.assertIn("goto __py2bin_catch_1;", inner.split("__py2bin_done_1")[0])


class Uncaught(unittest.TestCase):
    def test_one_that_reaches_the_end_of_main_stops_the_program(self) -> None:
        """C++ aborts. py2bin cannot raise a signal, so it exits non-zero.

        Running on as though nothing had happened - which is what a plain
        `return 0;` would do - is the one answer that is definitely wrong.
        """

        from py2bin.cpp_frontend import UNCAUGHT_STATUS

        out = translate(
            "int f(void){ throw 9; }\nint main(void){ f(); return 0; }", "t.cpp"
        )
        self.assertIn(f"return {UNCAUGHT_STATUS};", out)
        self.assertNotEqual(UNCAUGHT_STATUS, 0)


class Algorithm(unittest.TestCase):
    """<algorithm> is templates over pointers, which is what a contiguous
    iterator is. Its calls are also the hardest deduction in the language:
    `sort(v.begin(), v.end())` says nothing about the type anywhere."""

    def test_a_call_is_deduced_through_a_member_call(self) -> None:
        out = with_headers(
            "#include <vector>\n#include <algorithm>\n"
            "int main(void){ std::vector<int> v; v.push_back(1);"
            " std::sort(v.begin(), v.end()); return v[0]; }"
        )
        self.assertIn("sort__int(", out)
        self.assertNotIn("sort__int_p", out)

    def test_an_array_deduces_as_a_pointer_to_its_element(self) -> None:
        """`sort(raw, raw + 8)` is how half of all uses are written."""

        out = with_headers(
            "#include <algorithm>\n"
            "int main(void){ int raw[4]; std::sort(raw, raw + 4); return raw[0]; }",
        )
        self.assertIn("sort__int(", out)

    def test_a_template_calling_another_gets_both_copies(self) -> None:
        """`sort` calls `__sift`, from inside a copy the file never wrote.

        The expansion used to scan only the file, so a call inside a copy was
        left naming the pattern - which is not a function.
        """

        out = with_headers(
            "#include <algorithm>\n"
            "int main(void){ double d[4]; std::sort(d, d + 4); return 0; }",
        )
        self.assertIn("__sift__double(", out)
        self.assertNotIn("__sift__int", out)


class ThrownTemporaries(unittest.TestCase):
    def test_a_temporary_built_in_the_throw_is_allocated_and_constructed(
        self,
    ) -> None:
        """`throw std::runtime_error("x")` is how exceptions are thrown.

        `new` is exactly allocate-then-construct, which is what an exception
        object has to be: a copy that outlives the frame that threw it.
        """

        out = with_headers(
            "#include <stdexcept>\n"
            "int f(void){ throw std::runtime_error(\"bad\"); }\n"
            "int main(void){ try { f(); } catch (std::exception &e) "
            "{ return e.what()[0]; } return 0; }",
        )
        self.assertIn("runtime_error__new", out)
        self.assertIn("__py2bin_in_flight = (long)__py2bin_raised", out)

    def test_catching_by_reference_binds_rather_than_copies(self) -> None:
        out = with_headers(
            "#include <stdexcept>\n"
            "int f(void){ throw std::runtime_error(\"bad\"); }\n"
            "int main(void){ try { f(); } catch (std::exception &e) "
            "{ return e.what()[0]; } return 0; }",
        )
        # A pointer to what is in flight, and the virtual call through it.
        self.assertIn("struct exception *e = &(*(exception *)__py2bin_in_flight)", out)

    def test_catching_a_base_by_value_is_refused(self) -> None:
        """C++ slices it; py2bin's copy would not. Say so rather than differ."""

        with self.assertRaises(CppTranslationError) as caught:
            with_headers(
                "#include <stdexcept>\n"
                "int f(void){ throw std::runtime_error(\"bad\"); }\n"
                "int main(void){ try { f(); } catch (std::exception e) "
                "{ return 1; } return 0; }",
            )
        self.assertIn("&e", str(caught.exception))

    def test_a_class_nothing_derives_from_may_be_caught_by_value(self) -> None:
        # Slicing a T to a T loses nothing, so the refusal above must not
        # fire on every catch of a class that happens to have a base.
        out = with_headers(
            "#include <stdexcept>\n"
            "int f(void){ throw std::out_of_range(\"z\"); }\n"
            "int main(void){ try { f(); } catch (std::out_of_range e) "
            "{ return 1; } return 0; }",
        )
        self.assertIn("out_of_range e;", out)


class StatementOrder(unittest.TestCase):
    def test_a_hoisted_call_stays_after_the_try_it_follows(self) -> None:
        """The try's placeholder needed a `;` to be a statement of its own.

        Without it the try and the statement after it were one statement, so
        the call hoisted out of that statement was placed before the try and
        ran too early - visibly, in the order things printed.
        """

        out = translate(
            "int risky(int n){ if (n < 0) throw 1; return n; }\n"
            "int main(void){\n"
            "  try { risky(-1); } catch (int e) { return 1; }\n"
            "  return risky(7); }",
            "t.cpp",
        )
        body = out[out.index("int main"):]
        self.assertLess(
            body.index("__py2bin_catch_1"),
            body.index("= risky(7)"),
            "the call after the try was hoisted in front of it",
        )


class Filesystem(unittest.TestCase):
    """`path` is string work; the queries go to the platform.

    The split is not cosmetic: `#ifdef` is read by the C preprocessor, and the
    C++ translator runs before it - so a conditional written in a C++ header
    is still there when the translator reads the file, and it sees both
    branches. The platform half lives in a C header for that reason.
    """

    def test_the_platform_half_is_reached_through_c(self) -> None:
        out = with_headers(
            "#include <filesystem>\n"
            "int main(void){ std::filesystem::path p(\"/tmp\");"
            " return std::filesystem::exists(p); }"
        )
        self.assertIn("__py2bin_fs_exists", out)
        # And no conditional survives into the C++ the translator reads.
        self.assertNotIn("#ifdef _WIN32", out.split("int main")[0].split("path")[0])

    def test_a_namespace_alias_is_another_name_for_one(self) -> None:
        out = translate(
            "namespace deep { class T { public: int v; T(){v=3;} }; }\n"
            "namespace d = deep;\n"
            "int main(void){ d::T t; return t.v; }",
            "t.cpp",
        )
        self.assertIn("struct T t;", out)
        self.assertNotIn("namespace", out)


class ByValueFreeFunctions(unittest.TestCase):
    def test_a_class_taken_by_value_becomes_a_pointer_and_a_copy(self) -> None:
        """Methods did this from the start; free functions did not.

        `int twice(V v)` declared a struct parameter this backend cannot
        pass, and the call was refused with a type error rather than the
        missing feature it was.
        """

        out = translate(
            "class V{public:int n;\n V(){n=5;}\n int get(){return n;}};\n"
            "int twice(V v){ return v.get() * 2; }\n"
            "int main(void){ V a; return twice(a); }",
            "t.cpp",
        )
        self.assertIn("int twice(struct V *__by_value_v)", out)
        self.assertIn("v = *__by_value_v;", out)
        self.assertIn("twice(&a)", out)


class ChainedValueReturns(unittest.TestCase):
    def test_a_call_on_a_returned_object_gets_a_temporary(self) -> None:
        """A value return writes through a hidden pointer and yields nothing.

        So its result is not something anything can be called on until there
        is somewhere for it to live. C++ calls that a temporary; this writes
        the temporary out, in the C++, before anything else reads the body.
        """

        out = translate(
            "class S{public:int n;\n S(){n=1;}\n S twin(){ S o; o.n=n*2; return o; }\n"
            " int get(){return n;}};\n"
            "int main(void){ S a; return a.twin().get(); }",
            "t.cpp",
        )
        self.assertIn("__py2bin_value_1", out)
        self.assertIn("S__twin(&a, &__py2bin_value_1)", out)
        self.assertIn("S__get(&__py2bin_value_1)", out)

    def test_the_declaration_form_still_needs_no_temporary(self) -> None:
        # `S c = a.twin();` already hands the callee the caller's own space.
        out = translate(
            "class S{public:int n;\n S(){n=1;}\n S twin(){ S o; return o; }};\n"
            "int main(void){ S a; S c = a.twin(); return c.n; }",
            "t.cpp",
        )
        self.assertNotIn("__py2bin_value", out)
        self.assertIn("struct S c; S__twin(&a, &c);", out)


class TemplateNamesakes(unittest.TestCase):
    def test_a_method_named_like_a_template_is_not_a_call_to_it(self) -> None:
        """<string> has a `find` method and <algorithm> has a `find` template.

        The method's own head reads exactly like a call to the template until
        you notice what follows the parentheses - a body, which no call has.
        """

        out = with_headers(
            "#include <string>\n#include <algorithm>\n"
            "int main(void){ std::string s; s.assign(\"abc\");"
            " return s.find('b'); }"
        )
        self.assertIn("string__find", out)


class CallOperator(unittest.TestCase):
    def test_operator_call_is_parsed_and_reached(self) -> None:
        """`int operator()(int x)` has two parameter lists as far as `find`
        is concerned, and the first one is the operator's own name."""

        out = translate(
            "class D{public: int operator()(int x){ return x*2; }};\n"
            "int main(void){ D d; return d(5); }",
            "t.cpp",
        )
        self.assertIn("D__op_call(struct D *this, int x)", out)
        self.assertIn("D__op_call(&d, 5)", out)


class Lambdas(unittest.TestCase):
    """A lambda is a class with a call operator and a member per capture.

    That is not an analogy - it is what the standard says one is, and writing
    it out is all this does.
    """

    def test_one_becomes_a_class_and_an_object_of_it(self) -> None:
        out = translate(
            "int main(void){ auto f = [](int x){ return x + 1; };"
            " return f(2); }",
            "t.cpp",
        )
        self.assertIn("__py2bin_lambda_1__op_call", out)
        # `auto f = <lambda>` names the object itself; nothing is copied.
        self.assertIn("struct __py2bin_lambda_1 f;", out)

    def test_a_capture_becomes_a_member_set_where_it_is_made(self) -> None:
        out = translate(
            "int main(void){ int base = 10;"
            " auto f = [base](int x){ return x + base; }; return f(1); }",
            "t.cpp",
        )
        self.assertIn("f.base = base;", out)

    def test_the_return_type_is_read_from_the_return(self) -> None:
        out = translate(
            "int main(void){ auto f = [](double x){ return x * 2.0; };"
            " return (int)f(1.5); }",
            "t.cpp",
        )
        self.assertIn("double __py2bin_lambda_1__op_call", out)

    def test_a_reference_capture_is_the_address_and_is_followed(self) -> None:
        """Which is what a reference is, once it is written out."""

        out = translate(
            "class H{public:int n;\n H(){n=0;}\n void put(int v){n+=v;}};\n"
            "int main(void){ H h; auto f = [&h](int v){ h.put(v); };"
            " f(2); return h.n; }",
            "t.cpp",
        )
        self.assertIn("H * h;", out)       # the member holds the address
        self.assertIn("f.h = &h;", out)    # and is given it where it is made

    def test_capturing_everything_takes_what_the_body_uses(self) -> None:
        """`[&]` and `[=]` capture what C++ says they do: what is mentioned.

        Read from the body rather than guessed at - a name the enclosing
        scope does not declare is a global or a function and needs nothing.
        """

        out = translate(
            "int main(void){ int a = 2, total = 0;"
            " auto f = [&](int n){ total += a * n; }; f(3); return total; }",
            "t.cpp",
        )
        self.assertIn("f.total = &total;", out)
        self.assertIn("f.a = &a;", out)

    def test_a_member_named_like_a_lambda_head_is_not_one(self) -> None:
        """`operator[](int i) {` reads exactly like `[](int i) {`.

        Stopping at the first one meant a member in a supplied header hid
        every real lambda after it in the file.
        """

        out = with_headers(
            "#include <string>\n"
            "int main(void){ std::string s; s.assign(\"ab\");"
            " auto f = [](int x){ return x; }; return f(s.size()); }"
        )
        self.assertIn("__py2bin_lambda_1", out)


class OverloadedTemplates(unittest.TestCase):
    def test_two_templates_of_one_name_both_survive(self) -> None:
        """`sort(first, last)` and `sort(first, last, less_than)`.

        Keyed by the name alone, the second replaced the first and every call
        to the two-argument form became unreadable.
        """

        out = with_headers(
            "#include <algorithm>\n"
            "int main(void){ int a[3]; a[0]=2; a[1]=1; a[2]=3;\n"
            "  std::sort(a, a + 3);\n"
            "  auto down = [](int x, int y){ return x > y; };\n"
            "  std::sort(a, a + 3, down);\n"
            "  return a[0]; }"
        )
        self.assertIn("sort__int(", out)
        self.assertIn("__sift_by__int", out)


class Corpus(unittest.TestCase):
    """The corpus is the thing that catches what the unit tests do not.

    Every construct here got in because something broke on it, and a corpus
    that quietly shrinks stops being that. These do not run the programs -
    `tools/cpp_sweep.sh` does, against clang++ and across every target - they
    check that what it runs is still there.
    """

    def _corpus(self) -> "list[Path]":
        root = Path(__file__).resolve().parents[1] / "tools" / "cpp_corpus"
        return sorted(root.glob("*.cpp"))

    def test_every_program_still_translates(self) -> None:
        """Not what they mean - the sweep asks that - but that none is broken.

        A corpus program that stops translating is a regression whether or
        not anyone runs the sweep that day, and this runs on every change.
        """

        for source in self._corpus():
            with self.subTest(program=source.stem):
                translate(
                    inline_local_includes(source, [], set(), set()), str(source)
                )

    def test_the_corpus_covers_the_ground_it_is_meant_to(self) -> None:
        names = {source.stem for source in self._corpus()}
        for wanted in (
            "virtual_basic", "template_class", "references", "heap_new",
            "exception_basic", "lambda_captures", "std_vector", "std_string",
            "iostream_hello", "algorithm_sort", "filesystem_paths",
            "enum_plain", "static_member", "range_for", "init_list",
            "nested_class", "default_args", "casts", "operators_all",
        ):
            self.assertIn(wanted, names, f"{wanted} has gone from the corpus")
        self.assertGreater(len(names), 100)

    def test_a_supplied_header_has_its_own_includes_read(self) -> None:
        """Somebody's own `fstream` may include `<string>`, and often does.

        A header spelled without an extension is a C++ one, and reading C++ is
        what this stage is for - so it is pasted here and its own includes are
        resolved here too. Left for the preprocessor below, the `<string>`
        inside it reached a C compiler that ships no such header, and the
        classes the header declared were never translated at all.
        """

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "headers").mkdir()
            (root / "headers" / "fstream").write_text(
                "#include <string>\n"
                "namespace std { class ofstream {\n"
                "public: int written; ofstream() : written(0) {}\n"
                "void open(const std::string &name)"
                " { written = (int)name.size(); } }; }\n"
            )
            source = root / "main.cpp"
            source.write_text(
                "#include <fstream>\n#include <string>\n"
                "int main() { std::ofstream f;"
                " f.open(std::string(\"ab\")); return f.written; }\n"
            )
            written = translate(
                inline_local_includes(
                    source, (str(root / "headers"),), set(), set()
                ),
                str(source),
            )
        self.assertNotIn("#include <string>", written)
        self.assertIn("ofstream", written)

    def test_the_sweep_is_where_the_readme_says(self) -> None:
        root = Path(__file__).resolve().parents[1] / "tools"
        for name in ("cpp_sweep.sh", "cpp_differential.sh"):
            self.assertTrue((root / name).exists(), f"{name} is missing")


class MoreOfTheLanguage(unittest.TestCase):
    """What the sweep found once it stopped waiting to be told."""

    def test_a_multi_declarator_statement_declares_all_of_them(self) -> None:
        """`int a = 1, b = 2;` puts the type in front of the first only.

        A pattern looking for one type and one name never sees the rest, so
        `b` had no type at all - which is what stopped `[&]` from finding
        what to capture.
        """

        from py2bin.cpp_frontend import _declared_here

        found = _declared_here("int a = 1, b = 2; double *p = 0; char buf[8];")
        self.assertEqual(found.get("a"), "int")
        self.assertEqual(found.get("b"), "int")
        self.assertEqual(found.get("p"), "double *")
        self.assertEqual(found.get("buf"), "char *")

    def test_a_static_member_function_takes_no_object(self) -> None:
        out = translate(
            "class M{public: static int twice(int n){return n*2;} int k; M(){k=1;}};\n"
            "int main(void){ return M::twice(21); }",
            "t.cpp",
        )
        self.assertIn("M__twice(int n)", out)
        self.assertNotIn("M__twice(struct M", out)
        self.assertIn("M__twice(21)", out)

    def test_a_function_pointer_member_keeps_c_s_spelling(self) -> None:
        """`int (*op)(int)` puts the name inside the parentheses.

        Which is also why it was mistaken for a method declared here and
        defined elsewhere - both have parentheses, and only where the name
        sits tells them apart.
        """

        out = translate(
            "class H{public: int (*op)(int); H(){op=0;} int use(int n){return op(n);}};\n"
            "int main(void){ H h; return 0; }",
            "t.cpp",
        )
        self.assertIn("(*op)(int);", out)
        self.assertIn("this->op(n)", out)

    def test_a_free_function_may_return_a_class(self) -> None:
        """The same hidden pointer a method gets, for the same reason."""

        out = translate(
            "class R{public:int a;R(){a=0;}};\n"
            "R make(int x){ R r; r.a = x; return r; }\n"
            "int main(void){ R r = make(4); return r.a; }",
            "t.cpp",
        )
        self.assertIn("void make(struct R *__ret,", out)
        self.assertIn("make(&r, 4)", out)

    def test_a_derived_object_upcasts_at_a_reference_parameter(self) -> None:
        out = translate(
            "class S{public: virtual int id(){return 1;} virtual ~S(){}};\n"
            "class T : public S{public: int id(){return 2;}};\n"
            "int ask(S &s){ return s.id(); }\n"
            # Assigned rather than returned: returning an object that has a
            # destructor is refused, and says so.
            "int main(void){ T t; int n = ask(t); return n; }",
            "t.cpp",
        )
        self.assertIn("ask((struct S *)&t)", out)

    def test_a_class_pointer_member_is_already_the_address(self) -> None:
        """Taking one gave `&this->p`, which is a `T **`."""

        out = translate(
            "class A{public:int n;A(){n=1;} int get(){return n;}};\n"
            "class B{public: A *inner; B(){inner=0;} int use(){ return inner->get(); }};\n"
            "int main(void){ B b; return 0; }",
            "t.cpp",
        )
        self.assertIn("A__get(this->inner)", out)


class WhatTheHuntFound(unittest.TestCase):
    """Bugs a fan-out of agents found by probing, not by being told.

    Every one of these was reachable from ordinary C++ and none was in the
    corpus, which is the point: a corpus catches what it already knows.
    """

    def test_a_class_holding_another_constructs_it(self) -> None:
        """C++ writes the default constructor a class did not.

        Subobject construction was only put into a constructor the author
        wrote, so `class Outer { Inner in; };` left `in` as whatever was on
        the stack - a program that runs and is wrong, which is worse than one
        that is refused.
        """

        out = translate(
            "class Inner{public:int v;Inner(){v=5;}};\n"
            "class Outer{public: Inner in;};\n"
            "int main(){ Outer o; return o.in.v; }",
            "t.cpp",
        )
        self.assertIn("Inner__ctor(&this->in);", out)
        self.assertIn("Outer__ctor(&o);", out)

    def test_a_copy_runs_the_copy_constructor(self) -> None:
        """A bitwise copy of an object that owns something makes two owners."""

        source = (
            "class H{public:int n;H(int v){n=v;}H(const H &o){n=o.n+100;}};\n"
            "int use(H h){ return h.n; }\n"
            "int main(){ H a(3); return use(a); }"
        )
        out = translate(source, "t.cpp")
        self.assertIn("(&h, __by_value_h);", out)   # the copy constructor

    def test_two_members_in_one_declaration_are_both_members(self) -> None:
        """`int x, y;` - the type is in front of the first only.

        The struct was laid out correctly; the names simply were not in the
        set that drives implicit `this`, so a class could not reach its own
        `x` from its own methods.
        """

        out = translate(
            "struct P{ int x, y; void set(){ x = 1; y = 2; } };\n"
            "int main(){ P p; p.set(); return p.x; }",
            "t.cpp",
        )
        self.assertIn("this->x = 1", out)
        self.assertIn("this->y = 2", out)

    def test_an_array_of_pointers_is_still_one_inside_a_block(self) -> None:
        """The identical statement translated without the braces."""

        out = translate(
            "class I{public:int n;I(){n=5;}int get(){return n;}};\n"
            "int main(){ I a; I *all[1]; all[0]=&a;\n"
            "  for (int i=0;i<1;i++) { return all[i]->get(); } return 0; }",
            "t.cpp",
        )
        self.assertIn("I__get(all[i])", out)

    def test_a_default_argument_on_a_member_reaches_the_call(self) -> None:
        out = translate(
            "class C{public:int n;C(){n=0;}void bump(int by = 1){n+=by;}};\n"
            "int main(){ C c; c.bump(); c.bump(5); return c.n; }",
            "t.cpp",
        )
        self.assertIn("C__bump(&c, 1)", out)
        self.assertIn("C__bump(&c, 5)", out)

    def test_an_out_of_line_const_method_is_translated(self) -> None:
        """`const` after the parameters had no slot in the pattern, so the
        definition stayed as C++ while the call site was rewritten."""

        out = translate(
            "class B{public:B(int w){w_=w;} int width() const; private: int w_;};\n"
            "int B::width() const { return w_; }\n"
            "int main(){ B b(7); return b.width(); }",
            "t.cpp",
        )
        self.assertIn("B__width(struct B *this)", out)
        self.assertNotIn("B::width", out)

    def test_a_derived_pointer_upcasts_where_a_base_is_wanted(self) -> None:
        out = translate(
            "class S{public: virtual int id(){return 1;} virtual ~S(){}};\n"
            "class T : public S{public: int id(){return 2;}};\n"
            "int ask(S *s){ return s->id(); }\n"
            "S *pick(T *t){ return t; }\n"
            # Assigned rather than returned: returning an object with a
            # destructor is refused, and says so.
            "int main(){ T t; int n = ask(&t); return n; }",
            "t.cpp",
        )
        self.assertIn("return (struct S *)t;", out)
        self.assertIn("ask((struct S *)&t)", out)

    def test_copy_initialising_from_another_object(self) -> None:
        out = translate(
            "class B{public:int n;B(int v){n=v;}};\n"
            "int main(){ B a(3); B b = a; return b.n; }",
            "t.cpp",
        )
        self.assertIn("struct B b;", out)
        self.assertNotIn("struct B b = a", out)

    def test_a_temporary_becomes_an_object_with_a_name(self) -> None:
        """C has no expression that constructs anything."""

        out = translate(
            "class V{public:int n;V(int x){n=x;}};\n"
            "int take(V v){ return v.n; }\n"
            "int main(){ V t = V(5); return take(V(6)) + t.n; }",
            "t.cpp",
        )
        # The declaration's own object is the temporary; the argument needs
        # one of its own.
        self.assertIn("V__ctor(&t, 5)", out)
        self.assertIn("__py2bin_temp_", out)


class CapturingThis(unittest.TestCase):
    """`[this]` is how a callback inside a class is written.

    It was refused on the grounds that the enclosing class's type was not
    known - the lambda is written out before any class is read. The name was
    all that was missing, and the text has it.
    """

    def test_the_object_becomes_a_member_of_the_closure(self) -> None:
        out = translate(
            "class V{public:int n;\n V(){n=0;}\n void bump(){n=n+1;}\n"
            " int go(){ auto f = [this](){ bump(); }; f(); return n; }};\n"
            "int main(){ V v; return v.go(); }",
            "t.cpp",
        )
        self.assertIn("__py2bin_self", out)
        # Given the object where the closure is made...
        self.assertIn("__py2bin_self = this;", out)
        # ...and the body reaches the class through it.
        self.assertIn("V__bump(", out)

    def test_a_bare_member_name_goes_through_it_too(self) -> None:
        """Which is the whole of what capturing `this` buys."""

        out = translate(
            "class V{public:int n;\n V(){n=0;}\n"
            " int go(){ auto f = [this](){ n = n + 1; }; f(); return n; }};\n"
            "int main(){ V v; return v.go(); }",
            "t.cpp",
        )
        self.assertIn("__py2bin_self->n", out)

    def test_capture_defaults_take_the_object_as_well(self) -> None:
        """`[=]` and `[&]` inside a member function capture `this` in C++.

        Without it the body named a method that exists on no class the
        closure knows about.
        """

        for spelled in ("=", "&"):
            with self.subTest(capture=spelled):
                out = translate(
                    "class V{public:int n;\n V(){n=0;}\n void bump(){n=n+1;}\n"
                    f" int go(){{ auto f = [{spelled}](){{ bump(); }}; f();"
                    " return n; }};\n"
                    "int main(){ V v; return v.go(); }",
                    "t.cpp",
                )
                self.assertIn("__py2bin_self = this;", out)

    def test_a_capture_default_that_needs_nothing_takes_nothing(self) -> None:
        # A closure that never reaches the object should not hold one.
        out = translate(
            "class V{public:int n;\n V(){n=0;}\n"
            " int go(){ int k = 2; auto f = [=](int x){ return x * k; };"
            " return f(3); }};\n"
            "int main(){ V v; return v.go(); }",
            "t.cpp",
        )
        self.assertNotIn("__py2bin_self", out)

    def test_capturing_this_outside_a_class_says_so(self) -> None:
        with self.assertRaises(CppTranslationError) as caught:
            translate(
                "int main(){ auto f = [this](){ return 1; }; return f(); }",
                "t.cpp",
            )
        self.assertIn("outside any class", str(caught.exception))


class SecondHuntPass(unittest.TestCase):
    """Seven more of the fan-out's findings."""

    def test_delete_is_a_word_and_not_a_prefix(self) -> None:
        """`deleteAll()` became a call to `free`.

        The word boundary is the whole fix, and the failure it caused was an
        ordinary method silently turning into a deallocation.
        """

        out = translate(
            "class W{public:int deleted;int deleteLater;\n W(){deleted=0;deleteLater=3;}\n"
            " int deleteAll(){ deleted=1; return deleteLater; }};\n"
            "int main(){ W w; return w.deleteAll(); }",
            "t.cpp",
        )
        self.assertIn("W__deleteAll", out)
        self.assertNotIn("free(", out)

    def test_friend_asks_for_what_is_already_given(self) -> None:
        out = translate(
            "class P;\nclass O{ friend class P; public: int secret; O(){secret=42;} };\n"
            "class P{public: int read(O &o){ return o.secret; }};\n"
            "int main(){ O o; P p; return p.read(o); }",
            "t.cpp",
        )
        self.assertNotIn("friend", out)

    def test_the_address_of_a_reference_is_the_object(self) -> None:
        """`this == &other` in an assignment operator.

        A reference is a pointer here, so taking its address gave a `T **`.
        """

        out = translate(
            "class C{public:int n;C(){n=1;}\n"
            " C &operator=(const C &o){ if (this == &o) return *this; n=o.n; return *this; }};\n"
            "int main(){ C a; C b; a.n=7; b=a; return b.n; }",
            "t.cpp",
        )
        self.assertIn("this == o", out)

    def test_a_base_named_in_the_initialiser_list_is_constructed_with_it(
        self,
    ) -> None:
        """And *before* the derived class installs its own table.

        C++ builds an object base-first and its type changes as it goes;
        installing the table first let the base's constructor set it back, so
        every virtual call answered as the base.
        """

        out = translate(
            "class B{public:int n;B(int v){n=v;}virtual int show(){return n;}virtual ~B(){}};\n"
            "class S : public B{public:S(int v) : B(v*2) {} int show(){return B__NONE;}};\n".replace(
                "B__NONE", "B::show() + 1"
            )
            # Assigned first: returning an object with a destructor is refused.
            + "int main(){ S s(5); int n = s.show(); return n; }",
            "t.cpp",
        )
        body = out[out.index("S__ctor"):]
        first = body[: body.index("}")]
        self.assertLess(
            first.index("B__ctor"), first.index("__vptr"),
            "the table was installed before the base was constructed",
        )

    def test_a_base_implementation_can_be_named(self) -> None:
        """`Base::show()` is the one call a virtual does *not* dispatch."""

        out = translate(
            "class B{public:int n;B(){n=1;}virtual int show(){return n;}virtual ~B(){}};\n"
            "class S : public B{public: int show(){ return B::show() + 1; }};\n"
            "int main(){ S s; int n = s.show(); return n; }",
            "t.cpp",
        )
        self.assertIn("B__show(&this->__base)", out)

    def test_an_enum_declared_in_a_class_becomes_a_type_of_its_own(self) -> None:
        out = translate(
            "class M{public: enum Mode { Off, On }; Mode m; M(){m=On;}\n"
            " int get(){return (int)m;}};\n"
            "int main(){ M x; return x.get() + (int)M::Off; }",
            "t.cpp",
        )
        self.assertIn("enum Mode { Off, On };", out)
        # And above the struct that holds one, since C reads top to bottom.
        self.assertLess(out.index("enum Mode"), out.index("struct M {"))

    def test_a_static_const_given_its_value_in_the_class(self) -> None:
        out = translate(
            "class L{public: static const int limit = 10; int n; L(){n=0;}\n"
            " int room(){ return limit - n; }};\n"
            "int main(){ L l; return l.room() + L::limit; }",
            "t.cpp",
        )
        self.assertIn("L__limit = 10;", out)

    def test_a_grandparent_virtual_through_the_middle_class(self) -> None:
        """The table is read from the object, not from a base subobject.

        Applying the base adjustment to the receiver *and* to the path that
        finds the table counted it twice, and `A` has no `__base`.
        """

        out = translate(
            "class A{public: virtual int id(){return 1;} virtual ~A(){}};\n"
            "class B : public A{public:};\n"
            "class C : public B{public: int id(){return 3;}};\n"
            "int main(){ C c; B *p = (B *)&c; return p->id(); }",
            "t.cpp",
        )
        self.assertNotIn("__base.__base.__vptr", out.split("int main")[1])


class OutOfLineAndLiterals(unittest.TestCase):
    """What a file that embeds a web page runs into."""

    def test_a_lambda_in_an_out_of_line_method_finds_its_class(self) -> None:
        """`int Bridge::run() { ... }` has no class braces around it.

        The name says which class, which is the only thing the capture
        needed - and a method defined outside its class is where a callback
        is usually written.
        """

        out = translate(
            "class B{public:int n;B();void note();int run();};\n"
            "B::B(){n=0;}\n"
            "void B::note(){n=n+1;}\n"
            "int B::run(){ auto f = [this](){ note(); }; f(); return n; }\n"
            "int main(){ B b; return b.run(); }",
            "t.cpp",
        )
        self.assertIn("__py2bin_self = this;", out)
        self.assertIn("B__note(", out)

    def test_a_brace_inside_a_literal_is_not_a_brace(self) -> None:
        """A file embedding HTML or JavaScript is full of them.

        Counting braces inside strings puts every class body's end in the
        wrong place, and the text looks perfectly fine afterwards.
        """

        from py2bin.cpp_frontend import _matching

        text = 'class V { const char *p() { return "a{b"; } int n; };'
        # Without literal handling the stray `{` makes this run off the end.
        self.assertEqual(text[_matching(text, text.index("{")) - 1], "}")

    def test_a_class_holding_a_braced_literal_still_translates(self) -> None:
        out = translate(
            'class V{public:int n;V(){n=0;}\n'
            ' const char *page(){ return "<b style=\\"a{b}\\">{}</b>"; }\n'
            ' int go(){ auto f = [this](){ n = n + 1; }; f(); return n; }};\n'
            "int main(){ V v; return v.go(); }",
            "t.cpp",
        )
        self.assertIn("__py2bin_self", out)


class WrlCallbacks(unittest.TestCase):
    """`Microsoft::WRL::Callback<I>(lambda)` written out as the class it is.

    This is how every WebView2 program hands a completion handler to the
    loader, and the loader calls straight into the table the object carries.
    Nothing in the corpus reaches it - a COM interface needs a platform header
    - so it is checked here instead, and a shape that used to be refused is
    checked alongside the one that was not.
    """

    interfaces = """
    struct IUnknown {
        virtual HRESULT QueryInterface(REFIID, void **) = 0;
        virtual unsigned long AddRef() = 0;
        virtual unsigned long Release() = 0;
    };
    struct IDone : public IUnknown {
        virtual HRESULT Invoke(HRESULT, int) = 0;
    };
    """

    def translated(self, body: str) -> str:
        return translate(
            "typedef int HRESULT; typedef int REFIID;\n"
            "#define S_OK 0\n"
            "#define E_POINTER 1\n"
            + self.interfaces
            + body
        )

    def test_a_callback_that_carries_the_enclosing_object(self):
        out = self.translated(
            "struct Host {\n"
            "    int seen;\n"
            "    IDone *make() {\n"
            "        return Microsoft::WRL::Callback<IDone>(\n"
            "            [this](HRESULT code, int value) -> HRESULT {\n"
            "                seen = value; return code;\n"
            "            }).Get();\n"
            "    }\n"
            "};\n"
        )
        # The object holds the host it was written in, and reaches its member
        # through it rather than through a `this` that is now somebody else.
        self.assertIn("Host * __py2bin_self", out.replace("  ", " "))
        self.assertRegex(out, r"__py2bin_callback_\d+__vtable\[4\]")

    def test_a_callback_that_carries_nothing(self):
        # `[]` is ordinary C++ and used to be refused: py2bin always wrote a
        # member for the enclosing object, so a lambda with nothing to carry
        # had nowhere to be written and no class to be written in.
        out = self.translated(
            "IDone *make() {\n"
            "    return Microsoft::WRL::Callback<IDone>(\n"
            "        [](HRESULT code, int value) -> HRESULT {\n"
            "            return value == 0 ? code : S_OK;\n"
            "        }).Get();\n"
            "}\n"
        )
        self.assertRegex(out, r"__py2bin_callback_\d+__vtable\[4\]")
        self.assertNotIn("__py2bin_self", out)
        # And its constructor asks for nothing, because there is nothing to
        # hand it.
        self.assertRegex(out, r"__py2bin_callback_\d+__ctor\(struct \w+ \*this\)")

    def test_the_table_is_in_the_order_the_abi_says(self):
        # QueryInterface, AddRef, Release, then the interface's own method.
        # The order is the ABI: the loader reads slot 3 to invoke.
        out = self.translated(
            "IDone *make() {\n"
            "    return Microsoft::WRL::Callback<IDone>(\n"
            "        [](HRESULT code, int value) -> HRESULT { return code; }\n"
            "    ).Get();\n"
            "}\n"
        )
        table = re.search(r"__py2bin_callback_\d+__vtable\[4\] = \{([^}]*)\}", out)
        self.assertIsNotNone(table)
        slots = [one.strip() for one in table.group(1).split(",")]
        for wanted, slot in zip(
            ("QueryInterface", "AddRef", "Release", "Invoke"), slots
        ):
            self.assertTrue(slot.endswith(f"__{wanted}"), f"{slot} is not {wanted}")

    def test_a_capture_it_cannot_write_is_still_refused(self):
        with self.assertRaises(CppTranslationError) as raised:
            self.translated(
                "IDone *make(int held) {\n"
                "    return Microsoft::WRL::Callback<IDone>(\n"
                "        [held](HRESULT code, int value) -> HRESULT { return held; }\n"
                "    ).Get();\n"
                "}\n"
            )
        self.assertIn("captures held", str(raised.exception))
