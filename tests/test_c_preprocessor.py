"""py2bin's C preprocessor: what it expands, and what it refuses.

Every expectation below is derived by hand from what C11 clause 6.10 requires,
not from another compiler -- py2bin has no toolchain to compare against. On a
darwin-arm64 host the programs are also built and RUN, and their real stdout is
checked, because a preprocessor that produces plausible-looking tokens can still
produce a program that computes the wrong answer.
"""

from __future__ import annotations

import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

from py2bin.c_frontend import CCompileError, compile_c_to_ir
from py2bin.c_native import compile_c_native
from py2bin.c_preprocessor import preprocess


_HOST_IS_DARWIN_ARM64 = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)


def expand(source: str, **options) -> str:
    """The preprocessed token stream, written back out as readable text."""

    pieces = []
    for token in preprocess(source, "t.c", target="darwin-arm64", **options):
        if token.kind == "eof":
            break
        if token.kind == "string":
            pieces.append('"' + token.value.decode("latin-1") + '"')
        else:
            pieces.append(str(token.value))
    return " ".join(pieces)


class PreprocessorTestCase(unittest.TestCase):
    def build(
        self,
        source: str,
        headers: dict[str, str] | None = None,
        include_dirs: tuple[str, ...] = (),
        defines: tuple[str, ...] = (),
        target: str = "darwin-arm64",
    ) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # /var is a symlink to /private/var on macOS and py2bin reports
        # resolved paths, so compare against resolved ones.
        root = Path(directory.name).resolve()
        for name, text in (headers or {}).items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        entry = root / "program.c"
        entry.write_text(source, encoding="utf-8")
        artifact = root / "program.bin"
        compile_c_native(
            entry,
            artifact,
            target=target,
            clean=True,
            include_dirs=tuple(str(root / item) for item in include_dirs),
            defines=defines,
        )
        return artifact

    def run_c(self, source: str, stdout: str | None = None, **options) -> None:
        artifact = self.build(source, **options)
        if not _HOST_IS_DARWIN_ARM64:
            return
        result = subprocess.run([str(artifact)], capture_output=True, text=True)
        if stdout is not None:
            self.assertEqual(result.stdout, stdout)

    def reject(self, source: str, expected: str) -> None:
        with self.assertRaises(CCompileError) as caught:
            compile_c_to_ir(source, "reject.c", "darwin-arm64")
        self.assertRegex(str(caught.exception), expected)
        # A rejection is only useful if it says where.
        self.assertRegex(str(caught.exception), r"[\w<>.]+:\d+:\d+: ")


class ExpansionTests(PreprocessorTestCase):
    """Macro replacement, straight from C11 6.10.3."""

    def test_object_and_function_like_macros_reach_the_running_program(self):
        self.run_c(
            """
#include <stdio.h>
#define WIDTH 8
#define HEIGHT (WIDTH / 2)
#define AREA(w, h) ((w) * (h))
int main(void) {
    printf("%d %d %d\\n", WIDTH, HEIGHT, AREA(WIDTH, HEIGHT));
    return 0;
}
""",
            stdout="8 4 32\n",
        )

    def test_an_argument_is_substituted_exactly_as_many_times_as_it_is_written(self):
        # This is the whole contract of a macro: C substitutes the argument
        # TEXTUALLY, so one use is one evaluation and two uses are two. py2bin
        # has shipped six wrong-answer bugs from lowering one written value
        # twice, and a preprocessor is the one place where writing it twice is
        # correct -- so both directions are counted at run time.
        self.run_c(
            """
#include <stdio.h>
#define ONCE(x) (x)
#define TWICE(x) ((x) + (x))
#define PICK(c, a, b) ((c) ? (a) : (b))
int calls = 0;
int bump(void) { calls = calls + 1; return 3; }
int main(void) {
    calls = 0;
    int a = ONCE(bump());
    printf("%d %d\\n", a, calls);
    calls = 0;
    int b = TWICE(bump());
    printf("%d %d\\n", b, calls);
    calls = 0;
    int c = PICK(1, bump(), bump());
    printf("%d %d\\n", c, calls);
    return 0;
}
""",
            stdout="3 1\n6 2\n3 1\n",
        )

    @unittest.skipUnless(
        platform.system() == "Darwin", "otool is a macOS developer tool"
    )
    def test_the_machine_code_holds_one_call_per_written_call(self):
        # The run-time count above cannot see a call that is emitted but never
        # reached, so count the call instructions too. py2bin emits printf
        # inline, so every 'bl' in this program is one of the five calls to
        # bump() that the three macros above expand to: 1 + 2 + 2.
        artifact = self.build(
            """
#include <stdio.h>
#define ONCE(x) (x)
#define TWICE(x) ((x) + (x))
#define PICK(c, a, b) ((c) ? (a) : (b))
int calls = 0;
int bump(void) { calls = calls + 1; return 3; }
int main(void) {
    int a = ONCE(bump());
    int b = TWICE(bump());
    int c = PICK(1, bump(), bump());
    printf("%d %d %d\\n", a, b, c);
    return 0;
}
"""
        )
        text = subprocess.run(
            ["otool", "-tvV", str(artifact)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(sum(1 for line in text.splitlines() if "\tbl\t" in line), 5)

    def test_a_macro_cannot_expand_into_itself(self):
        # C11 6.10.3.4p2: a macro name found while its own replacement is being
        # rescanned is not replaced again, which is what makes these terminate.
        self.assertEqual(expand("#define A B\n#define B A\nA B\n"), "A B")
        self.assertEqual(expand("#define F(x) F(x)\nF(1)\n"), "F ( 1 )")
        self.assertEqual(
            expand("#define A(x) A(x)\nA(A(1))\n"), "A ( A ( 1 ) )"
        )

    def test_a_token_that_came_out_of_a_macro_stays_out_of_it(self):
        # The classic 'painted blue' case: the argument 'foo' is expanded on
        # its own, where it is a function-like name with no '(' after it, so it
        # survives -- and it must not be invoked by the '(2)' that follows the
        # expansion either, because it carries foo's own name in its hide set.
        self.assertEqual(
            expand("#define foo(x) bar x\nfoo(foo) (2)\n"), "bar foo ( 2 )"
        )
        self.assertEqual(expand("#define X X + 1\nX\n"), "X + 1")
        self.assertEqual(
            expand("#define F(x) G(x)\n#define G(x) F(x)\nF(1)\n"), "F ( 1 )"
        )
        # A chain of macros that does terminate still expands the whole way.
        self.assertEqual(
            expand("#define f(x) x\n#define g f\n#define h g(2)\nh\n"), "2"
        )

    def test_a_function_like_name_is_examined_before_what_follows_it(self):
        # Q is not called here: the token after it is P, not '('. Expanding P
        # first and then letting Q take the parenthesis would be wrong.
        self.assertEqual(
            expand("#define P (1+2)\n#define Q(x) x\nQ P\n"), "Q ( 1 + 2 )"
        )

    def test_a_function_like_macro_without_a_call_is_left_alone(self):
        self.assertEqual(expand("#define f(x) x + 1\nf\n"), "f")
        self.assertEqual(expand("#define f(x) x + 1\nf (2)\n"), "2 + 1")
        # The '(' may come from anywhere, including a line further on.
        self.assertEqual(expand("#define f(x) x + 1\nf\n(2)\n"), "2 + 1")

    def test_an_argument_is_expanded_before_it_is_substituted(self):
        self.assertEqual(
            expand("#define ID(x) x\n#define ONE 1\nID(ONE)\n"), "1"
        )
        self.assertEqual(expand("#define ID(x) x\nID(ID(ID(7)))\n"), "7")
        # ... but an empty argument substitutes nothing at all.
        self.assertEqual(expand("#define EMPTY\n#define f(x) [x]\nf(EMPTY)\n"), "[ ]")
        self.assertEqual(expand("#define g(a, b) a b\ng(,)\n"), "")

    def test_stringify_reproduces_the_argument_as_it_was_written(self):
        # C11 6.10.3.2p2: the spelling, with runs of whitespace squeezed to one
        # space, the leading and trailing whitespace deleted, and a backslash
        # inserted before each " and \ inside a literal. What the program ends
        # up holding is therefore the argument exactly as it was typed: the
        # inserted backslashes are removed again when the new string literal is
        # lexed, so 'a "b\n" 'c'' comes back with its \n still two characters.
        self.assertEqual(
            expand("#define S(x) #x\nS(  a   \"b\\n\"  'c' )\n"),
            '"a "b\\n" \'c\'"',
        )
        # An unexpanded argument is stringified; the two-step idiom expands it.
        self.assertEqual(
            expand("#define S(x) #x\n#define XS(x) S(x)\n#define N 4\nS(N) XS(N)\n"),
            '"N" "4"',
        )

    def test_paste_joins_two_tokens_into_one(self):
        self.assertEqual(expand("#define P(a, b) a ## b\nP(1, 2)\n"), "12")
        self.assertEqual(expand("#define P(a, b) a ## b\nP(x, y)\n"), "xy")
        self.assertEqual(expand("#define P(a, b) a ## b\nP(+, +)\n"), "++")
        # An empty operand leaves a placemarker, which disappears (6.10.3.3p2).
        self.assertEqual(expand("#define P(a, b) a ## b\nP(, )z\n"), "z")
        # Pasting happens before the result is rescanned for more macros.
        self.assertEqual(
            expand(
                "#define CAT(a, b) a##b\n#define CAT2(a, b) CAT(a, b)\n"
                "CAT2(CAT(1, 2), 3)\n"
            ),
            "123",
        )

    def test_the_standards_own_example_of_hash_and_paste(self):
        # C11 6.10.3.5 EXAMPLE 4, checked against the expansion the standard
        # prints there.
        source = (
            "#define str(s) # s\n"
            "#define xstr(s) str(s)\n"
            '#define debug(s, t) printf("x" # s "= %d, x" # t "= %s", x ## s, x ## t)\n'
            "debug(1, 2);\n"
        )
        self.assertEqual(
            expand(source),
            'printf ( "x" "1" "= %d, x" "2" "= %s" , x1 , x2 ) ;',
        )

    def test_the_standards_hardest_worked_example(self):
        # C11 6.10.3.5 EXAMPLE 3, with the expansion the standard prints. This
        # is the whole algorithm at once: rescanning, an argument expanded
        # before it is substituted, a macro that is not replaced inside its own
        # expansion, an invocation whose '(' comes from the source rather than
        # from the replacement list, and empty arguments either side of ##.
        source = """
#define x 3
#define f(a) f(x * (a))
#undef x
#define x 2
#define g f
#define z z[0]
#define h g(~
#define m(a) a(w)
#define w 0,1
#define t(a) a
#define p() int
#define q(x) x
#define r(x,y) x ## y
#define str(x) # x
f(y+1) + f(f(z)) % t(t(g)(0) + t)(1);
g(x+(3,4)-w) | h 5) & m
(f)^m(m);
p() i[q()] = { q(1), r(2,3), r(4,), r(,5), r(,) };
char c[2][6] = { str(hello), str() };
"""
        self.assertEqual(
            expand(source),
            "f ( 2 * ( y + 1 ) ) + f ( 2 * ( f ( 2 * ( z [ 0 ] ) ) ) ) "
            "% f ( 2 * ( 0 ) ) + t ( 1 ) ; "
            "f ( 2 * ( 2 + ( 3 , 4 ) - 0 , 1 ) ) | f ( 2 * ( ~ 5 ) ) "
            "& f ( 2 * ( 0 , 1 ) ) ^ m ( 0 , 1 ) ; "
            "int i [ ] = { 1 , 23 , 4 , 5 , } ; "
            'char c [ 2 ] [ 6 ] = { "hello" , "" } ;',
        )

    def test_the_standards_example_of_pasting_empty_arguments(self):
        # C11 6.10.3.5 EXAMPLE 5.
        self.assertEqual(
            expand(
                "#define t(x,y,z) x ## y ## z\n"
                "int j[] = { t(1,2,3), t(,4,5), t(6,,7), t(8,9,),\n"
                "t(10,,), t(,11,), t(,,12), t(,,) };\n"
            ),
            "int j [ ] = { 123 , 45 , 67 , 89 , 10 , 11 , 12 , } ;",
        )

    def test_the_standards_example_of_pasting_two_hashes(self):
        # C11 6.10.3.3 EXAMPLE. The space in the result is the point: what
        # replaces a macro name inherits the whitespace in front of that name,
        # so the pasted '##' is stringified with the space that preceded
        # hash_hash.
        self.assertEqual(
            expand(
                "#define hash_hash # ## #\n"
                "#define mkstr(a) # a\n"
                "#define in_between(a) mkstr(a)\n"
                "#define join(c, d) in_between(c hash_hash d)\n"
                "char p[] = join(x, y);\n"
            ),
            'char p [ ] = "x ## y" ;',
        )

    def test_paste_builds_a_real_function_name_that_runs(self):
        self.run_c(
            """
#include <stdio.h>
#define NAME(a, b) a ## b
#define STR(x) #x
int NAME(add, two)(int n) { return n + 2; }
int main(void) {
    printf("%d %s\\n", addtwo(40), STR(add two));
    return 0;
}
""",
            stdout="42 add two\n",
        )

    def test_variable_arguments(self):
        self.assertEqual(
            expand("#define V(...) __VA_ARGS__\nV(1, 2, 3)\n"), "1 , 2 , 3"
        )
        self.assertEqual(
            expand("#define V(a, ...) a - __VA_ARGS__\nV(1, 2, 3)\n"), "1 - 2 , 3"
        )
        self.assertEqual(expand("#define V(a, ...) a __VA_ARGS__\nV(1)\n"), "1")
        self.run_c(
            """
#include <stdio.h>
#define SAY(fmt, ...) printf(fmt, __VA_ARGS__)
int main(void) {
    SAY("%d-%d\\n", 10, 20);
    return 0;
}
""",
            stdout="10-20\n",
        )

    def test_a_macro_may_be_defined_again_only_identically(self):
        self.assertEqual(expand("#define X 1\n#define X 1\nX\n"), "1")
        self.reject(
            "#define X 1\n#define X 2\nint main(void) { return X; }\n",
            "redefined with a different replacement list",
        )
        self.assertEqual(expand("#define X 1\n#undef X\n#define X 2\nX\n"), "2")
        self.assertEqual(expand("#define X 1\n#undef X\nX\n"), "X")


class TableTests(PreprocessorTestCase):
    """One table expanded several ways: what macros are really used for."""

    def test_an_x_macro_table_builds_functions_names_and_a_sum(self):
        # The X-macro idiom leans on almost everything at once: a multi-line
        # continued list, a macro redefined after #undef, stringification, and
        # adjacent string literals joining afterwards. 1 + 2 + 4 = 7.
        self.run_c(
            """
#include <stdio.h>
#define COLORS      \\
    X(red,   1)     \\
    X(green, 2)     \\
    X(blue,  4)

#define X(name, value) static int name(void) { return value; }
COLORS
#undef X

#define X(name, value) + name()
static int total(void) { return 0 COLORS; }
#undef X

#define X(name, value) #name ","
static const char *names = COLORS "";
#undef X

int main(void) {
    printf("%d %s\\n", total(), names);
    return 0;
}
""",
            stdout="7 red,green,blue,\n",
        )

    def test_the_same_program_preprocesses_the_same_way_for_every_target(self):
        # Nothing in the preprocessor is target-specific except the __py2bin
        # macros, so the six encoders must all see the same 30.
        source = """
#define A 5
#define B 6
#define MUL(x, y) ((x) * (y))
#if MUL(A, B) == 30
#define RESULT MUL(A, B)
#else
#define RESULT 0
#endif
int main(void) { return RESULT; }
"""
        for target in ("linux-x86_64", "linux-arm64", "darwin-x86_64",
                       "darwin-arm64", "windows-x86_64", "windows-arm64"):
            with self.subTest(target=target):
                artifact = self.build(source, target=target)
                self.assertTrue(artifact.is_file())
        if _HOST_IS_DARWIN_ARM64:
            artifact = self.build(source)
            self.assertEqual(subprocess.run([str(artifact)]).returncode, 30)


class LayoutTests(PreprocessorTestCase):
    """Translation phases 1 to 3: continued lines and comments."""

    def test_a_backslash_newline_continues_anything(self):
        self.assertEqual(expand("#define LONG 1 + \\\n 2\nLONG\n"), "1 + 2")
        self.run_c(
            """
#include <stdio.h>
#define SUM(a, b) \\
    ((a) + \\
     (b))
int main(void) {
    printf("%d\\n", SUM(20, 22));
    return 0;
}
""",
            stdout="42\n",
        )

    def test_a_comment_becomes_one_space(self):
        self.assertEqual(expand("int/**/x;\n"), "int x ;")
        # A block comment that spans lines takes the newlines with it, so a
        # directive really does continue through one.
        self.assertEqual(expand("#define M /* here\n */ 9\nM\n"), "9")
        # ... and neither comment marker means anything inside a literal.
        self.assertEqual(expand('char *s = "a/*b*/c";\n'), 'char * s = "a/*b*/c" ;')

    def test_a_macro_call_may_span_lines(self):
        self.assertEqual(expand("#define F(a, b) a + b\nF(1,\n2)\n"), "1 + 2")


class ConditionalTests(PreprocessorTestCase):
    """#if and its family, and the constant expressions they evaluate."""

    def test_the_branches_that_are_taken(self):
        self.assertEqual(expand("#if 1\na\n#else\nb\n#endif\n"), "a")
        self.assertEqual(expand("#if 0\na\n#else\nb\n#endif\n"), "b")
        self.assertEqual(
            expand("#if 0\na\n#elif 1\nb\n#elif 1\nc\n#else\nd\n#endif\n"), "b"
        )
        self.assertEqual(
            expand("#if 1\n#if 0\na\n#else\nb\n#endif\n#endif\n"), "b"
        )
        # A group inside one that was not taken stays dark whatever it says.
        self.assertEqual(
            expand("#if 0\n#if 1\na\n#else\nb\n#endif\n#endif\nz\n"), "z"
        )
        self.assertEqual(
            expand("#if 0\n#if 0\na\n#elif 1\nb\n#else\nc\n#endif\n#endif\nz\n"), "z"
        )
        self.assertEqual(
            expand("#if 0\na\n#elif 0\nb\n#elif 1\nc\n#else\nd\n#endif\n"), "c"
        )
        self.assertEqual(
            expand("#if 1\na\n#elif 1\nb\n#else\nc\n#endif\n"), "a"
        )
        self.assertEqual(expand("#define X\n#ifdef X\na\n#endif\n"), "a")
        self.assertEqual(expand("#ifndef X\na\n#endif\n"), "a")
        self.assertEqual(
            expand("#define X 0\n#if defined X && !defined(Y)\na\n#endif\n"), "a"
        )

    def test_a_skipped_group_is_not_interpreted_at_all(self):
        # Only the nesting matters inside a group that is not taken: what is in
        # it need not even be valid C.
        self.assertEqual(
            expand("#if 0\nthis ## is @@ nonsense\n#if x(\n#endif\n#endif\nz\n"), "z"
        )

    def test_the_expression_is_evaluated_the_way_c_requires(self):
        for expression, taken in (
            ("(2 + 3) * 4 == 20", True),
            ("(1 << 3) == 8", True),
            ("7 / 2 == 3 && -7 / 2 == -3 && -7 % 2 == -1", True),  # toward zero
            ("-1 < 0u", False),  # the usual conversions make -1 huge
            ("0x8000000000000000 > 0", True),  # too big for signed, so unsigned
            ("'A' == 65", True),
            ("UNDEFINED_NAME == 0", True),  # 6.10.1p4: what is left becomes 0
            ("1 ? 2 : 0", True),
            ("~0 == -1", True),
        ):
            with self.subTest(expression=expression):
                source = f"#if {expression}\nyes\n#else\nno\n#endif\n"
                self.assertEqual(expand(source), "yes" if taken else "no")

    def test_an_operand_that_is_not_evaluated_is_not_diagnosed(self):
        # A constant expression may contain anything in a branch it does not
        # take, so the division by zero here must not be reported.
        self.assertEqual(expand("#if 0 && 1 / 0\na\n#else\nb\n#endif\n"), "b")
        self.assertEqual(expand("#if 1 || 1 / 0\na\n#endif\n"), "a")
        self.assertEqual(expand("#if 1 ? 2 : 1 / 0\na\n#endif\n"), "a")

    def test_conditionals_choose_what_the_program_does(self):
        self.run_c(
            """
#include <stdio.h>
#define LEVEL 2
int main(void) {
#if LEVEL == 1
    printf("one\\n");
#elif LEVEL == 2
    printf("two\\n");
#else
    printf("other\\n");
#endif
    return 0;
}
""",
            stdout="two\n",
        )

    def test_a_macro_defined_on_the_command_line(self):
        self.assertEqual(
            expand("#if defined(K) && K == 7\na\n#endif\n", defines=("K=7",)), "a"
        )
        self.assertEqual(expand("#ifdef K\na\n#endif\n", defines=("K",)), "a")
        self.run_c(
            """
#include <stdio.h>
int main(void) {
#ifdef QUIET
    return 0;
#else
    printf("loud\\n");
    return 0;
#endif
}
""",
            stdout="",
            defines=("QUIET",),
        )


class IncludeTests(PreprocessorTestCase):
    """#include, the search for a file, and being included twice."""

    def test_a_quoted_header_is_found_beside_the_file_that_included_it(self):
        self.run_c(
            """
#include <stdio.h>
#include "shape.h"
#include "shape.h"
int main(void) {
    printf("%d %d\\n", SIDES, area(3));
    return 0;
}
""",
            headers={
                "shape.h": """
#ifndef SHAPE_H
#define SHAPE_H
#define SIDES 4
static int area(int side) { return side * side; }
#endif
"""
            },
            stdout="4 9\n",
        )

    def test_an_angled_header_is_found_along_the_search_path(self):
        self.run_c(
            """
#include <stdio.h>
#include <depth/deep.h>
int main(void) {
    printf("%d\\n", DEEP + NESTED);
    return 0;
}
""",
            headers={
                "vendor/depth/deep.h": '#define DEEP 40\n#include "more.h"\n',
                "vendor/depth/more.h": "#pragma once\n#define NESTED 2\n",
            },
            include_dirs=("vendor",),
            stdout="42\n",
        )

    def test_pragma_once_and_an_include_guard_both_stop_a_second_read(self):
        # Reading either of these twice would declare 'twice' twice, which the
        # C front end rejects -- so compiling at all is the check.
        self.run_c(
            """
#include <stdio.h>
#include "guard.h"
#include "guard.h"
#include "once.h"
#include "once.h"
int main(void) {
    printf("%d\\n", guarded(1) + onced(1));
    return 0;
}
""",
            headers={
                "guard.h": "#ifndef GUARD_H\n#define GUARD_H\n"
                "static int guarded(int n) { return n + 1; }\n#endif\n",
                "once.h": "#pragma once\n"
                "static int onced(int n) { return n + 10; }\n",
            },
            stdout="13\n",
        )

    def test_a_header_name_may_come_from_a_macro(self):
        self.run_c(
            """
#include <stdio.h>
#define WHICH "chosen.h"
#include WHICH
int main(void) {
    printf("%d\\n", CHOSEN);
    return 0;
}
""",
            headers={"chosen.h": "#define CHOSEN 5\n"},
            stdout="5\n",
        )

    def test_a_diagnostic_names_the_header_the_mistake_is_in(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # /var is a symlink to /private/var on macOS and py2bin reports
        # resolved paths, so compare against resolved ones.
        root = Path(directory.name).resolve()
        (root / "bad.h").write_text(
            "static int bad(void) { return nowhere; }\n", encoding="utf-8"
        )
        entry = root / "program.c"
        entry.write_text(
            '#include "bad.h"\nint main(void) { return bad(); }\n', encoding="utf-8"
        )
        with self.assertRaises(CCompileError) as caught:
            compile_c_native(entry, root / "out.bin", target="darwin-arm64", clean=True)
        self.assertEqual(caught.exception.filename, str(root / "bad.h"))
        self.assertEqual(caught.exception.line, 1)


class BuiltInHeaderTests(PreprocessorTestCase):
    """The standard headers py2bin serves out of its own pocket."""

    def test_limits_and_stdint_carry_the_values_this_dialect_really_has(self):
        self.run_c(
            """
#include <stdio.h>
#include <limits.h>
#include <stdint.h>
int main(void) {
    printf("%d %d %d\\n", CHAR_BIT, INT_MAX, INT_MIN);
    printf("%u %llu\\n", UINT_MAX, ULLONG_MAX);
    printf("%d %lld\\n", INT8_MAX, INT64_MAX);
    printf("%d\\n", SCHAR_MIN + UCHAR_MAX);
    return 0;
}
""",
            # INT_MIN prints as -2147483648, and SCHAR_MIN + UCHAR_MAX is
            # -128 + 255 = 127 after both promote to int.
            stdout="8 2147483647 -2147483648\n"
            "4294967295 18446744073709551615\n"
            "127 9223372036854775807\n"
            "127\n",
        )

    def test_stdbool_gives_the_real_boolean_type(self):
        self.run_c(
            """
#include <stdio.h>
#include <stdbool.h>
int main(void) {
    bool yes = true;
    bool no = false;
    /* _Bool stores 0 or 1 whatever is assigned to it. */
    bool narrowed = 256;
    printf("%d %d %d\\n", yes, no, narrowed);
    return 0;
}
""",
            stdout="1 0 1\n",
        )

    def test_inttypes_format_macros_paste_onto_the_format_string(self):
        # The PRI macros only work because adjacent string literals are joined
        # after expansion, which is translation phase 6.
        self.run_c(
            """
#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>
int main(void) {
    int64_t big = 9223372036854775807LL;
    printf("%" PRId64 "\\n", big);
    return 0;
}
""",
            stdout="9223372036854775807\n",
        )

    def test_null_is_a_null_pointer(self):
        self.run_c(
            """
#include <stdio.h>
#include <stddef.h>
int main(void) {
    int n = 7;
    int *p = NULL;
    p = &n;
    printf("%d %d\\n", p == NULL, *p);
    return 0;
}
""",
            stdout="0 7\n",
        )


class PredefinedTests(PreprocessorTestCase):
    def test_the_predefined_macros(self):
        self.assertEqual(expand("__LINE__\n\n__LINE__\n"), "1 3")
        self.assertEqual(expand("__FILE__\n"), '"t.c"')
        self.assertEqual(expand("__STDC__ __STDC_VERSION__ __STDC_HOSTED__\n"), "1 201112 0")
        self.assertEqual(expand("__py2bin__ __py2bin_target__\n"), '1 "darwin-arm64"')
        self.assertEqual(
            expand("#if defined(__py2bin_arm64__) && defined(__py2bin_darwin__)\na\n#endif\n"),
            "a",
        )

    def test_the_date_and_time_are_fixed_so_a_build_is_reproducible(self):
        # C11 6.10.8.1 allows an implementation-defined constant when the date
        # of translation is not available, and py2bin declares it is not: the
        # same source has to compile to the same bytes.
        self.assertEqual(expand("__DATE__ __TIME__\n"), '"Jan  1 1970" "00:00:00"')

    def test_line_is_where_the_macro_was_used(self):
        self.assertEqual(expand("#define HERE __LINE__\n\n\nHERE\n"), "4")

    def test_a_program_can_print_where_it_was_compiled_from(self):
        self.run_c(
            """
#include <stdio.h>
#define TRACE printf("line %d\\n", __LINE__)
int main(void) {
    TRACE;
    TRACE;
    return 0;
}
""",
            stdout="line 5\nline 6\n",
        )


class RejectionTests(PreprocessorTestCase):
    """What py2bin will not guess at."""

    def test_directives_it_does_not_implement(self):
        self.reject("#line 5\nint main(void) { return 0; }\n", "#line is not implemented")
        self.reject(
            "#warning hmm\nint main(void) { return 0; }\n", "#warning is not implemented"
        )
        self.reject(
            "#pragma pack(1)\nint main(void) { return 0; }\n",
            "the only #pragma py2bin implements is 'once'",
        )
        self.reject(
            "#unheard_of\nint main(void) { return 0; }\n",
            "unknown preprocessing directive",
        )

    def test_error_stops_the_compilation_with_its_message(self):
        self.reject(
            "#if 1\n#error this target is not supported\n#endif\n",
            "#error this target is not supported",
        )

    def test_a_header_it_cannot_find(self):
        self.reject(
            '#include "nowhere.h"\nint main(void) { return 0; }\n',
            "cannot find the header",
        )
        self.reject(
            "#include <sys/socket.h>\nint main(void) { return 0; }\n",
            "cannot find the header",
        )

    def test_unbalanced_conditionals(self):
        self.reject("#if 1\nint main(void) { return 0; }\n", "never closed with #endif")
        self.reject("#endif\nint main(void) { return 0; }\n", "without a matching #if")
        self.reject("#else\nint main(void) { return 0; }\n", "without a matching #if")
        self.reject(
            "#if 1\n#else\n#else\n#endif\nint main(void) { return 0; }\n",
            "#else after #else",
        )
        self.reject(
            "#if 1\n#else\n#elif 1\n#endif\nint main(void) { return 0; }\n",
            "#elif after #else",
        )

    def test_bad_if_expressions(self):
        self.reject("#if 1 / 0\n#endif\nint main(void) { return 0; }\n", "division by zero")
        self.reject(
            "#if 1.5\n#endif\nint main(void) { return 0; }\n",
            "not an integer",
        )
        self.reject(
            '#if "text"\n#endif\nint main(void) { return 0; }\n',
            "unexpected",
        )
        self.reject("#if 1 << 64\n#endif\nint main(void) { return 0; }\n", "undefined in C")
        self.reject("#if (1\n#endif\nint main(void) { return 0; }\n", r"has no '\)'")
        self.reject("#if\n#endif\nint main(void) { return 0; }\n", "needs an expression")
        self.reject(
            "#define D defined\n#if D(X)\n#endif\nint main(void) { return 0; }\n",
            "came out of a macro expansion",
        )

    def test_an_if_expression_it_cannot_nest_far_enough_to_evaluate(self):
        # C11 5.2.4.1 asks a compiler to manage 63 levels of parentheses;
        # py2bin manages more than that and then says what happened rather than
        # dying of recursion.
        deep = "(" * 90 + "1" + ")" * 90
        self.assertEqual(expand(f"#if {deep}\nyes\n#endif\n"), "yes")
        self.reject(
            "#if " + "(" * 500 + "1" + ")" * 500 + "\n#endif\nint main(void){return 0;}\n",
            "nests more than",
        )

    def test_a_stringify_that_would_not_be_a_string_literal(self):
        self.reject(
            "#define S(x) #x\nchar *s = S(\\);\nint main(void) { return 0; }\n",
            "does not make a valid string literal",
        )

    def test_bad_macro_definitions(self):
        self.reject("#define\nint main(void) { return 0; }\n", "needs a macro name")
        self.reject(
            "#define F(a) ## a\nint main(void) { return 0; }\n",
            "cannot begin a replacement list",
        )
        self.reject(
            "#define F(a) a ##\nint main(void) { return 0; }\n",
            "cannot end a replacement list",
        )
        self.reject(
            "#define F(a) #b\nint main(void) { return 0; }\n",
            "must be followed by a parameter name",
        )
        self.reject(
            "#define F(a, a) a\nint main(void) { return 0; }\n", "is named twice"
        )
        self.reject(
            "#define F(a, ...) x, ## __VA_ARGS__\nint main(void) { return 0; }\n",
            "GNU extension",
        )
        self.reject(
            "#define __LINE__ 9\nint main(void) { return 0; }\n",
            "predefined and cannot be redefined",
        )
        self.reject(
            "#define F(a) __VA_ARGS__\nint main(void) { return 0; }\n",
            "means something only in the replacement list",
        )

    def test_bad_macro_calls(self):
        self.reject(
            "#define F(a, b) a\nint main(void) { return F(1); }\n",
            r"takes 2 argument\(s\) but 1 were given",
        )
        self.reject(
            "#define F(a) a\nint main(void) { return F(1; }\n", "is not closed"
        )
        self.reject(
            "#define P(a, b) a ## b\nint p = P(+, -);\nint main(void) { return 0; }\n",
            "does not make a single preprocessing token",
        )

    def test_a_stray_preprocessing_operator(self):
        self.reject("int a = 1; # 2\nint main(void) { return 0; }\n", "means nothing in C")
        # A '#' that is not a directive and not inside a #define is not an
        # operator either: it reaches the C lexer, which has no use for it.
        self.reject(
            "int main(void) { return # 1; }\n", "means nothing in C"
        )


if __name__ == "__main__":
    unittest.main()
