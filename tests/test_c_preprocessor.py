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
from py2bin.c_preprocessor import PPToken, Preprocessor, preprocess


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

    def scratch(self) -> Path:
        """An empty directory for a test that lays out its own files.

        `build` writes the headers from a dict, which cannot say how a file
        got where it is - and a symlink, or a second spelling of a name this
        filesystem already answers to, is exactly that question.
        """

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name).resolve()

    def run_in(self, root: Path, source: str, stdout: str | None = None) -> None:
        entry = root / "program.c"
        entry.write_text(source, encoding="utf-8")
        artifact = root / "program.bin"
        compile_c_native(entry, artifact, target="darwin-arm64", clean=True)
        if not _HOST_IS_DARWIN_ARM64:
            return
        result = subprocess.run([str(artifact)], capture_output=True, text=True)
        if stdout is not None:
            self.assertEqual(result.stdout, stdout)

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

    def test_prose_in_a_skipped_group_is_not_lexed_as_code(self) -> None:
        """A skipped group is read for its directives and for nothing else.

        `this doesn't compile` under an `#if 0` is how a header explains
        what it switched off, and the apostrophe was reported as an
        unterminated character constant before the conditional was even
        read. clang reads a quote nothing on its line closes as a stray
        character and moves on; in a group that is compiled it is still the
        error it always was, reported where it stands.
        """

        self.assertEqual(
            expand("#if 0\nThis block doesn't compile.\n\"nor this\n#endif\nz\n"),
            "z",
        )
        self.assertEqual(
            expand("#ifdef NEVER\nIt isn't compiled either.\n#endif\nz\n"), "z"
        )
        self.reject("int main(void) { char c = 'a; return 0; }\n", "unterminated character constant")
        self.reject('int main(void) { const char *s = "a; return 0; }\n', "unterminated string literal")
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

    def test_pragma_once_holds_when_one_file_is_spelled_two_ways(self):
        """`"Once.h"` and `"once.h"` name one file where case does not count.

        The header read twice was remembered by its settled path, and a
        settled path keeps whatever case was written - so on this filesystem
        `#pragma once` saw two files, read both, and the C front end reported
        `onced` defined twice in a header that defines it once.
        """

        root = self.scratch()
        (root / "Once.h").write_text(
            "#pragma once\nstatic int onced(int n) { return n + 10; }\n",
            encoding="utf-8",
        )
        if not (root / "once.h").is_file():
            self.skipTest("this filesystem tells Once.h and once.h apart")
        self.run_in(
            root,
            '#include <stdio.h>\n#include "Once.h"\n#include "once.h"\n'
            'int main(void) { printf("%d\\n", onced(3)); return 0; }\n',
            stdout="13\n",
        )

    def test_pragma_once_holds_through_a_symlink_and_a_dotted_path(self):
        """Two more spellings of one file, and the same question about them.

        A settled path already answers these, and they are here so that the
        move to asking the filesystem for identity keeps answering them.
        """

        root = self.scratch()
        (root / "sub").mkdir()
        (root / "sub" / "once.h").write_text(
            "#pragma once\nstatic int onced(int n) { return n + 10; }\n",
            encoding="utf-8",
        )
        try:
            (root / "alias.h").symlink_to(Path("sub") / "once.h")
        except (OSError, NotImplementedError):
            self.skipTest("this filesystem has no symlinks")
        self.run_in(
            root,
            "#include <stdio.h>\n"
            '#include "sub/once.h"\n#include "./sub//once.h"\n'
            '#include "alias.h"\n'
            'int main(void) { printf("%d\\n", onced(3)); return 0; }\n',
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
            "#unheard_of\nint main(void) { return 0; }\n",
            "unknown preprocessing directive",
        )

    def test_a_warning_is_reported_and_the_build_carries_on(self):
        """Refusing it stopped a build over the one directive whose whole
        purpose is not to stop one - and a published header set writes one at
        the top of each header it has superseded, so nine of the 1350 headers
        in one of them stopped there and nowhere else."""

        import contextlib
        import io

        said = io.StringIO()
        with contextlib.redirect_stderr(said):
            self.assertEqual(
                expand("#warning this header moved\nkept\n"), "kept"
            )
        self.assertIn("warning: t.c:1:2: this header moved", said.getvalue())

    def test_a_pragma_that_would_move_a_member_is_refused_by_name(self):
        """The ones ignoring would get wrong, rather than the ones it knows.

        C says an implementation ignores a pragma it does not recognise, and
        py2bin does - but not these. Each of them says where a member sits,
        which definition a name reaches, or which section something lands
        in, and a program built as if it were not written is a different
        program with nothing to say so.
        """

        for spelled, changes in (
            ("#pragma ms_struct on", "another ABI's rules"),
            ("#pragma pointers_to_members(full_generality)", "pointer to a member"),
            ("#pragma scalar_storage_order big-endian", "other byte order"),
            ("#pragma weak helper", "makes a definition weak"),
            ("#pragma data_seg(\".shared\")", "section of their own"),
        ):
            with self.subTest(spelled=spelled):
                self.reject(
                    f"{spelled}\nint main(void) {{ return 0; }}\n", changes
                )

    def test_an_error_says_which_branches_it_fell_through(self):
        """A header that stops at the end of a chain is saying what it wanted.

        Naming the branches is the whole answer: it says which conditions
        would have had to hold, and lets the author see whether any of them
        is one they can arrange. Guessing at a flag instead got it wrong -
        `#error You must define NtCurrentTeb()` cannot be answered with a
        `-D` of that name, because the chain never tests for it.
        """

        with self.assertRaises(CCompileError) as caught:
            preprocess(
                "#ifdef WINE_UNIX_LIB\n"
                "#elif defined(__i386__) && defined(__GNUC__)\n"
                "#elif defined(__x86_64__) && defined(_MSC_VER)\n"
                "#elif !defined(RC_INVOKED)\n"
                "# error You must define NtCurrentTeb() for your architecture\n"
                "#endif\n",
                "t.c",
                target="windows-x86_64",
            )
        message = str(caught.exception)
        self.assertIn("You must define NtCurrentTeb()", message)
        self.assertIn("none of these held", message)
        self.assertIn("#ifdef WINE_UNIX_LIB", message)
        self.assertIn("defined(__x86_64__) && defined(_MSC_VER)", message)
        # The branch that *was* taken is not among them: it is the one that
        # led here, not one that failed.
        self.assertNotIn("RC_INVOKED", message)

    def test_a_branch_that_held_is_not_reported_as_a_failure(self):
        # Nothing falls through here, so there is nothing to explain.
        tokens = preprocess(
            "#if 1\nint main(void) { return 0; }\n"
            "#else\n# error unreachable\n#endif\n",
            "t.c",
        )
        self.assertTrue(any(token.value == "main" for token in tokens))

    def test_a_header_of_its_own_may_define_null_first(self):
        """C says a redefinition has to be identical, and both spellings are
        valid null pointer constants - so py2bin's own headers must not fight
        a vendored one over it. Whichever got there first keeps it.
        """

        for spelled in ("0", "((void *)0)", "0L"):
            with self.subTest(spelled=spelled):
                tokens = preprocess(
                    f"#define NULL {spelled}\n#include <stddef.h>\n"
                    "#include <stdio.h>\n"
                    "int main(void) { return NULL == 0; }\n",
                    "t.c",
                )
                self.assertTrue(any(token.value == "main" for token in tokens))

    def test_a_pragma_that_says_nothing_about_the_program_is_accepted(self):
        """Diagnostics, folding, linking, and whatever else: none change the C.

        C says an implementation ignores a pragma it does not recognise, and
        the set of pragmas a compiler can be told about is unbounded, so this
        is the default and not the exception. Naming the ones py2bin knew and
        refusing the rest stopped ordinary portable source on its first line
        - `#pragma unroll`, `#pragma ivdep` and every vendor's own hint.

        `#pragma region section` is here because the second word of a pragma
        belongs to that pragma: reading it as a name of its own would refuse
        a fold marker for being called `section`.
        """

        for spelled in (
            "#pragma warning( disable: 4049 )",
            "#pragma region setup",
            "#pragma region section",
            "#pragma endregion",
            "#pragma GCC diagnostic ignored \"-Wunused\"",
            "#pragma clang diagnostic push",
            "#pragma comment(lib, \"user32.lib\")",
            "#pragma message(\"building\")",
            "#pragma unroll 4",
            "#pragma ivdep",
            "#pragma acme vectorize always",
            "#pragma",
        ):
            with self.subTest(spelled=spelled):
                self.assertIn(
                    "main",
                    " ".join(
                        token.value
                        for token in preprocess(
                            f"{spelled}\nint main(void) {{ return 0; }}\n", "t.c"
                        )
                        if isinstance(token.value, str)
                    ),
                )

    def test_the_pragma_operator_is_read_where_the_macros_are(self):
        """`_Pragma("...")` is the only way a macro can write a pragma.

        A directive is not a token, so nothing a macro expands to can be a
        `#pragma`; C99 gave the same thing an operator spelling for that
        reason. It is read during macro replacement because the string is
        usually built by `#` out of the macro's own argument, and there is
        nothing to read until then.
        """

        self.assertEqual(
            expand(
                '#define DO_PRAGMA(x) _Pragma(#x)\n'
                'DO_PRAGMA(GCC diagnostic ignored "-Wunused")\n'
                '_Pragma("acme hint")\n'
                "int main(void) { return 0; }\n"
            ),
            "int main ( void ) { return 0 ; }",
        )

    def test_the_pragma_operator_undoes_the_escapes_in_its_string(self):
        """C11 6.10.9: the prefix and quotes go, `\\"` becomes a quote and
        `\\\\` becomes one backslash, and what is left is read as the tokens
        that would have followed a `#pragma`.
        """

        at = PPToken("identifier", "_Pragma", 1, 1, "t.c", False)
        literal = PPToken("string", r'L"message(\"a\\b\")"', 1, 9, "t.c", False)
        self.assertEqual(
            [one.spelling for one in Preprocessor._destringized(literal, at)],
            ["message", "(", r'"a\b"', ")"],
        )
        # And put back where the operator was written, because the position
        # inside a string is one nobody can point at.
        self.assertEqual(
            {(one.line, one.column) for one in Preprocessor._destringized(literal, at)},
            {(1, 1)},
        )

    def test_a_pack_written_as_the_operator_still_packs(self):
        """The one pragma that means something here, spelled the macro way.

        A header that wants a packed struct behind a name of its own has no
        other way to write it, and reading the operator and then dropping
        what it said would lay the struct out with the padding the pragma
        was there to remove.
        """

        self.run_c(
            """
#include <stdio.h>
#define BEGIN_PACKED _Pragma("pack(push, 1)")
#define END_PACKED   _Pragma("pack(pop)")

BEGIN_PACKED
struct Wire { char tag; int value; };
END_PACKED
struct Loose { char tag; int value; };

int main(void) {
    printf("%d %d\\n", (int)sizeof(struct Wire), (int)sizeof(struct Loose));
    return 0;
}
""",
            stdout="5 8\n",
        )

    def test_a_pack_written_as_the_operator_is_refused_out_of_cpp(self):
        """The same pack, refused when the C came from the C++ translator.

        That translator reads a pack out of the text - before any macro has
        been replaced - so a class can carry its packing to wherever the
        struct is written out, and both the class and the pragma above it
        move. By the time the string exists to be read here the classes have
        been laid out, unpacked, and honouring the pack now would pack
        whatever happened to land below instead. It printed `8 8` where
        clang++ printed `5 8` and said nothing.

        Only out of C++: in C nothing moves, and the test above builds and
        runs exactly this program.
        """

        for spelled in (
            '_Pragma("pack(push, 1)")\n',
            '#define PACKED _Pragma("pack(1)")\nPACKED\n',
            '#define DO_PRAGMA(x) _Pragma(#x)\nDO_PRAGMA(pack(1))\n',
        ):
            with self.subTest(spelled=spelled):
                with self.assertRaises(CCompileError) as caught:
                    compile_c_to_ir(
                        f"{spelled}struct W {{ char a; int b; }};\n"
                        "int main(void) { return 0; }\n",
                        "t.cpp",
                        "darwin-arm64",
                        cplusplus=True,
                    )
                self.assertIn("write it as `#pragma pack", str(caught.exception))

    def test_the_operator_carries_the_same_refusals_the_directive_does(self):
        self.reject(
            '_Pragma("ms_struct on")\nint main(void) { return 0; }\n',
            "another ABI's rules",
        )

    def test_the_operator_needs_a_parenthesised_string(self):
        self.reject(
            "_Pragma(hint)\nint main(void) { return 0; }\n",
            "_Pragma takes a string literal",
        )
        self.reject(
            '_Pragma "hint"\nint main(void) { return 0; }\n',
            "_Pragma takes a parenthesised string literal",
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


class WindowsHeaderPieceTests(PreprocessorTestCase):
    """The SDK splits <windows.h> up; every piece of it is py2bin's own."""

    def compile_for_windows(self, source: str, target: str = "windows-x86_64") -> None:
        with tempfile.TemporaryDirectory() as directory:
            where = Path(directory) / "piece.c"
            where.write_text(source, newline="\n")
            compile_c_native(where, Path(directory) / "piece.exe", target=target)

    def test_winnt_h_is_served_rather_than_fetched(self):
        """Wine's is the copy a fetch finds, and it stops at

            winnt.h:2638: #error You must define NtCurrentTeb() for your architecture

        because every branch of that chain wants GCC or MSVC paired with an
        architecture, and each of those bodies wants inline assembly or an
        MSVC intrinsic. py2bin is neither compiler, so it brings its own.
        """

        self.compile_for_windows(
            """
#include <winnt.h>
int main(void) {
    LARGE_INTEGER n;
    n.QuadPart = 42;
    return (int)n.QuadPart - 42;
}
"""
        )

    def test_every_piece_names_the_same_header_once(self):
        """Read twice under two names, the text would redefine everything."""

        for target in ("windows-x86_64", "windows-arm64"):
            with self.subTest(target=target):
                self.compile_for_windows(
                    """
#include <windows.h>
#include <winnt.h>
#include <windef.h>
#include <minwindef.h>
#include <minwinbase.h>
#include <winbase.h>
#include <winuser.h>
#include <basetsd.h>
int main(void) {
    LUID id;
    id.LowPart = 1;
    id.HighPart = 2;
    FILETIME stamp;
    stamp.dwLowDateTime = 0;
    stamp.dwHighDateTime = 0;
    BOOLEAN ok = TRUE;
    UCHAR c = 3;
    PVOID nothing = (PVOID)0;
    return (int)(id.LowPart + id.HighPart + ok + c) - 7 + (nothing != 0);
}
""",
                    target,
                )

    def test_a_generated_com_header_takes_its_c_branch(self):
        """MIDL output declares an interface twice - once as C++ classes,
        once as a table of function pointers - and chooses with

            #if defined(__cplusplus) && !defined(CINTERFACE)

        py2bin defines no __cplusplus, so the second is the branch taken, and
        the second is the one it can compile: a COM object IS that table."""

        self.compile_for_windows(
            """
#include <windows.h>
#include <unknwn.h>
#include <objidl.h>
#include <oaidl.h>
#include <EventToken.h>

#if defined(__cplusplus) && !defined(CINTERFACE)
#error py2bin must not present itself as a C++ compiler to a generated header
#endif

int main(void) {
    IStream *stream = (IStream *)0;
    IUnknown *root = (IUnknown *)0;
    VARIANT value;
    EventRegistrationToken token;
    value.vt = VT_I8;
    value.llVal = 7;
    token.value = 3;
    if (stream != (IStream *)0) { stream->lpVtbl->Release(stream); }
    if (root != (IUnknown *)0) { root->lpVtbl->AddRef(root); }
    return (int)(value.llVal + token.value) - 10;
}
"""
        )

    def test_the_com_tables_are_the_right_depth(self):
        """A slot out of place is a call to a different function, and a
        vtable call is a load and a branch, so nothing would report it."""

        from py2bin.c_preprocessor import _OBJIDL_H, _UNKNWN_H

        root = _UNKNWN_H[_UNKNWN_H.index("IUnknownVtbl {"):]
        self.assertEqual(root[: root.index("}")].count(");"), 3)
        for owner, slots in (("ISequentialStreamVtbl {", 5), ("IStreamVtbl {", 14)):
            table = _OBJIDL_H[_OBJIDL_H.index(owner):]
            self.assertEqual(
                table[: table.index("} ")].count(");"),
                slots,
                f"{owner} is not {slots} slots deep",
            )

    def test_a_header_may_ask_what_this_compiler_has(self):
        """`__has_feature(x)` and its family take an argument, so leaving each
        to the rule that turns an unknown identifier into 0 left the `(`
        behind - which stopped libc++'s <type_traits> on its first line of
        feature detection, at

            #if __has_feature(is_union) || (_GNUC_VER >= 403)
        """

        self.assertEqual(
            expand(
                "#if __has_feature(is_union) || (_GNUC_VER >= 403)\n"
                "no\n"
                "#else\n"
                "yes\n"
                "#endif\n"
            ),
            "yes",
        )
        for asking in (
            "__has_builtin(__builtin_expect)",
            "__has_attribute(always_inline)",
            "__has_cpp_attribute(nodiscard)",
            "__has_extension(c_atomic)",
            "__has_declspec_attribute(dllimport)",
            "__has_keyword(constexpr)",
        ):
            with self.subTest(asking=asking):
                self.assertEqual(
                    expand(f"#if {asking}\nno\n#else\nyes\n#endif\n"), "yes"
                )

    def test_has_include_is_answered_by_looking(self):
        self.assertEqual(
            expand("#if __has_include(<stdio.h>)\nyes\n#else\nno\n#endif\n"),
            "yes",
        )
        self.assertEqual(
            expand(
                "#if __has_include(<nowhere_at_all.h>)\nno\n#else\nyes\n#endif\n"
            ),
            "yes",
        )

    def test_the_header_a_set_generates_is_py2bin_s_own(self):
        """mingw-w64 writes `#include <_mingw.h>` at the top of every header
        it has, and that is the one file in the set which does not exist:
        it is made from `_mingw.h.in` by a configure step. What it holds is a
        description of the compiler reading it, so py2bin is the one that
        knows the answers."""

        self.compile_for_windows(
            """
#include <_mingw.h>
#include <stdio.h>
struct sized { __LONG32 a; __int64 b; __int32 c; };
int main(void) {
    struct sized s;
    s.a = 1;
    s.b = 2;
    s.c = 3;
    _CRT_UNUSED(s);
    return (int)(s.a + s.b + s.c) - 6;
}
"""
        )

    def test_a_fetched_header_is_written_in_the_sets_own_spellings(self):
        """A fetched set reaches the core through <windef.h> or
        <minwindef.h> - which are py2bin's <windows.h> - and then writes the
        rest of itself in the words that core would have given it: WINBOOL
        for a BOOL, VOID for void, __C89_NAMELESS before an anonymous member,
        WINBASEAPI before an import, DECLARE_HANDLE for a handle. Each was a
        name nothing had defined, and the parser stopped on it as if it were
        a type. Over one published set of 1350 headers, WINBOOL alone is
        where 290 of them stopped and `__C89_NAMELESS` another 137.
        """

        with tempfile.TemporaryDirectory() as directory:
            where = Path(directory)
            (where / "vendor.h").write_text(
                """
#include <minwindef.h>

#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
DECLARE_HANDLE(HTHING);

typedef struct _THING {
    WINBOOL live;
    LCID locale;
    __C89_NAMELESS union {
        DWORD whole;
        WORD half[2];
    } u;
} THING, *PTHING;

WINBASEAPI VOID WINAPI ThingReset(HTHING handle);

FORCEINLINE WINBOOL ThingLive(const THING *thing) { return thing->live; }
#endif
""",
                newline="\n",
            )
            (where / "piece.c").write_text(
                """
#include "vendor.h"
int main(void) {
    THING thing;
    HTHING handle = (HTHING)0;
    thing.live = TRUE;
    thing.locale = 1033;
    thing.u.whole = 0x00020001;
    return (int)(ThingLive(&thing) + thing.u.half[1] + (handle != 0)) - 3;
}
""",
                newline="\n",
            )
            compile_c_native(
                where / "piece.c", where / "piece.exe", target="windows-x86_64"
            )

    def test_what_py2bin_supplies_is_a_default(self):
        """py2bin's own header says what `S_OK` is so a program that never
        reaches a platform set still has it. A program that does reach one
        has the real thing, and every set spells these differently - so the
        real one wins rather than being reported as a clash."""

        # Redefining one is not a clash, and the second definition is the
        # one that stands.
        self.compile_for_windows(
            """
#include <windows.h>
#define E_OUTOFMEMORY _HRESULT_TYPEDEF_(0x8007000E)
#define _HRESULT_TYPEDEF_(x) ((HRESULT)x)
int main(void) { return (int)(E_OUTOFMEMORY == E_OUTOFMEMORY) - 1; }
"""
        )

    def test_a_default_loses_whichever_order_the_two_arrive_in(self):
        """A set reaches py2bin's <windows.h> through one of its own headers
        as often as the other way round: <apisetcconv.h> defines WINBASEAPI
        as DECLSPEC_IMPORT and then includes a piece of <windows.h>, which is
        py2bin's - and py2bin's, defining it as nothing, reported a clash
        against the set's own word for its own thing."""

        self.compile_for_windows(
            """
#define WINAPI __stdcall
#define WINBASEAPI __declspec(dllimport)
#include <windows.h>
WINBASEAPI DWORD WINAPI TheirOwnEntryPoint(void);
int main(void) { return 0; }
"""
        )

    def test_a_set_may_declare_an_entry_point_py2bin_declares_too(self):
        """C allows a function to be declared twice, and a program that
        includes <windows.h> and then a piece of a fetched set gets exactly
        that: py2bin's header bound `CloseHandle` to the ABI it emits a call
        for, and <handleapi.h> declares the same function again. The second
        declaration is checked against the same table rather than refused for
        existing - and one that disagrees is still refused, by the
        disagreement."""

        self.compile_for_windows(
            """
#include <windows.h>
WINBASEAPI BOOL WINAPI CloseHandle(HANDLE hObject);
WINBASEAPI BOOL WINAPI SetConsoleCP(UINT wCodePageID);
int main(void) { return 0; }
"""
        )
        with self.assertRaises(CCompileError) as caught:
            compile_c_to_ir(
                "#include <windows.h>\n"
                "BOOL WINAPI CloseHandle(HANDLE a, HANDLE b);\n"
                "int main(void) { return 0; }\n",
                "reject.c",
                "windows-x86_64",
            )
        self.assertRegex(str(caught.exception), "vetted adapter ABI takes")

    def test_the_structs_this_header_adds_are_laid_out_as_windows_lays_them(self):
        """Every number here was checked against the published headers rather
        than remembered: `OVERLAPPED` is 32 bytes because the union is one
        name for two DWORDs and another for a pointer over the same eight,
        and moving `hEvent` off 24 would hand the kernel a different struct
        from the one it fills in."""

        from py2bin.c_frontend import Parser
        from py2bin.c_preprocessor import preprocess

        for target in ("windows-x86_64", "windows-arm64"):
            with self.subTest(target=target):
                parser = Parser(
                    list(preprocess("#include <windows.h>\n", "t.c", target=target)),
                    "t.c",
                    target,
                )
                parser.translation_unit()
                for tag, size in (
                    ("_SECURITY_ATTRIBUTES", 24),
                    ("_OVERLAPPED", 32),
                    ("_SYSTEMTIME", 16),
                    ("_LIST_ENTRY", 16),
                    ("tagPOINTS", 4),
                    ("tagSIZE", 8),
                    ("_RECTL", 16),
                ):
                    self.assertEqual(parser.struct_tags[tag].size, size, tag)
                overlapped = parser.struct_tags["_OVERLAPPED"]
                for name, offset in (
                    ("Internal", 0),
                    ("InternalHigh", 8),
                    ("Offset", 16),
                    ("OffsetHigh", 20),
                    ("Pointer", 16),
                    ("hEvent", 24),
                ):
                    self.assertEqual(overlapped.member(name).offset, offset, name)

    def test_two_definitions_of_a_macro_nobody_supplied_still_clash(self):
        self.reject(
            "#define OURS 1\n#define OURS 2\nint main(void) { return OURS; }\n",
            "redefined with a different replacement list",
        )

    def test_windows_is_llp64(self):
        """A negative array length is an error, so this builds only where the
        widths are the ones the data model gives. py2bin was LP64 on every
        target, on the reasoning that it shared no layout with a platform C
        library - which stopped being true the day it compiled a vendor's
        header."""

        self.compile_for_windows(
            """
#include <stdint.h>
static int windows_long_is_four[sizeof(long) == 4 ? 1 : -1];
static int windows_size_t_is_eight[sizeof(size_t) == 8 ? 1 : -1];
static int windows_pointer_is_eight[sizeof(void *) == 8 ? 1 : -1];
static int windows_intptr_is_eight[sizeof(intptr_t) == 8 ? 1 : -1];
static int windows_long_long_is_eight[sizeof(long long) == 8 ? 1 : -1];
int main(void) { return 0; }
"""
        )

    def test_everywhere_else_is_lp64(self):
        source = """
static int elsewhere_long_is_eight[sizeof(long) == 8 ? 1 : -1];
int main(void) { return 0; }
"""
        for target in ("linux-x86_64", "darwin-arm64"):
            with self.subTest(target=target):
                with tempfile.TemporaryDirectory() as directory:
                    where = Path(directory) / "model.c"
                    where.write_text(source, newline="\n")
                    compile_c_native(
                        where, Path(directory) / "model", target=target
                    )

    def test_the_ole_structs_are_laid_out_as_the_platform_lays_them(self):
        """Every size and offset here was computed from the same fields at
        the widths Windows gives them, not remembered. A struct's whole worth
        is that each member sits where the platform puts it: `LONG lindex` is
        four bytes there, and eight would move `tymed` and make the struct the
        wrong size to hand to anything."""

        from py2bin.c_frontend import Parser
        from py2bin.c_preprocessor import preprocess

        source = "#include <windows.h>\n#include <objidl.h>\nint main(void){return 0;}\n"
        for target in ("windows-x86_64", "windows-arm64"):
            with self.subTest(target=target):
                parser = Parser(
                    list(preprocess(source, "t.c", target=target)), "t.c", target
                )
                parser.translation_unit()
                for tag, size in (
                    ("tagFORMATETC", 32),
                    ("tagSTGMEDIUM", 24),
                    ("tagDVTARGETDEVICE", 16),
                    ("tagSTATDATA", 56),
                ):
                    self.assertEqual(parser.struct_tags[tag].size, size, tag)
                spelled = parser.struct_tags["tagFORMATETC"]
                for name, offset in (
                    ("cfFormat", 0),
                    ("ptd", 8),
                    ("dwAspect", 16),
                    ("lindex", 20),
                    ("tymed", 24),
                ):
                    self.assertEqual(spelled.member(name).offset, offset, name)
                medium = parser.struct_tags["tagSTGMEDIUM"]
                # Reached through the unnamed union, which is what makes them
                # members of this struct at all.
                for name, offset in (
                    ("tymed", 0),
                    ("hGlobal", 8),
                    ("pstm", 8),
                    ("pstg", 8),
                    ("pUnkForRelease", 16),
                ):
                    self.assertEqual(medium.member(name).offset, offset, name)


    def test_automation_types_are_laid_out_as_the_sdk_lays_them(self):
        """VARIANT is twenty-four bytes on 64-bit Windows, not sixteen.

        `Invoke` reads `rgvarg[1]` twenty-four bytes along; written as a tag
        and eight bytes of value it was sixteen, and the second argument was
        read from the middle of the first. CY, DECIMAL, BLOB, CLIPDATA and
        SAFEARRAY are the shapes <wtypes.h> gives them."""

        from py2bin.c_frontend import Parser
        from py2bin.c_preprocessor import preprocess

        source = "#include <windows.h>\n#include <oaidl.h>\nint main(void){return 0;}\n"
        for target in ("windows-x86_64", "windows-arm64"):
            with self.subTest(target=target):
                parser = Parser(
                    list(preprocess(source, "t.c", target=target)), "t.c", target
                )
                parser.translation_unit()
                for tag, size in (
                    ("tagVARIANT", 24),
                    ("__py2bin_DISPPARAMS", 24),
                    ("__py2bin_EXCEPINFO", 64),
                    ("tagCY", 8),
                    ("tagDEC", 16),
                    ("tagBLOB", 16),
                    ("tagCLIPDATA", 16),
                    ("tagSAFEARRAYBOUND", 8),
                    ("tagSAFEARRAY", 32),
                ):
                    self.assertEqual(parser.struct_tags[tag].size, size, tag)
                variant = parser.struct_tags["tagVARIANT"]
                # Reached through the unnamed union and struct, as the SDK
                # spells them without NONAMELESSUNION.
                for name, offset in (
                    ("vt", 0),
                    ("lVal", 8),
                    ("bstrVal", 8),
                    ("pvRecord", 8),
                    ("pRecInfo", 16),
                    ("decVal", 0),
                ):
                    self.assertEqual(variant.member(name).offset, offset, name)
                given = parser.struct_tags["__py2bin_DISPPARAMS"]
                self.assertEqual(given.member("cArgs").offset, 16)
                self.assertEqual(given.member("cNamedArgs").offset, 20)
                failed = parser.struct_tags["__py2bin_EXCEPINFO"]
                self.assertEqual(failed.member("scode").offset, 56)
    def test_a_generated_header_can_be_read_the_cpp_way(self):
        """MIDL declares each interface twice and picks with

            #if defined(__cplusplus) && !defined(CINTERFACE)

        The translator runs before the preprocessor and has no `#if`, so a
        header like that used to be left to the preprocessor - which took the
        C branch, and a program calling an interface the C++ way was told the
        struct had no such member. The preprocessor now runs first for that
        header alone, with `__cplusplus` defined, and hands the translator
        the one branch a C++ compiler would have been given."""

        from py2bin.c_preprocessor import as_cplusplus

        with tempfile.TemporaryDirectory() as directory:
            where = Path(directory)
            (where / "chooses.h").write_text(
                "#if defined(__cplusplus) && !defined(CINTERFACE)\n"
                "struct IThing { virtual int Poke(int n) = 0; };\n"
                "#else\n"
                "typedef struct IThingVtbl { int (*Poke)(void *, int); } IThingVtbl;\n"
                "struct IThing { const IThingVtbl *lpVtbl; };\n"
                "#endif\n",
                newline="\n",
            )
            supplied: "set[str]" = set()
            text = as_cplusplus(
                "chooses.h", where, (str(where),), "windows-x86_64", supplied,
                set(),
            )
            self.assertIn("virtual", text)
            self.assertNotIn("lpVtbl", text)

    def test_what_one_run_supplied_the_next_is_told(self):
        """The preprocessed branch carries py2bin's own headers expanded into
        it. Read again by the run that reads the rest of the program, every
        struct in them would be defined twice."""

        self.compile_for_windows(
            """
#pragma py2bin supplied "wtypes.h"
typedef struct _GUID { int mine; } GUID;
int main(void) { GUID g; g.mine = 1; return g.mine - 1; }
"""
        )

    def test_sys_types_h_agrees_with_the_data_model(self):
        """`ssize_t` is as wide as a pointer, and a Windows `long` is not.

        The header wrote `typedef long ssize_t;` on every target. The
        compiler already knows the name at the pointer's width, so on a
        Windows target the two disagreed and including <sys/types.h> at all
        failed with "'ssize_t' is already a different type" - the header was
        unreachable on two of the six machines.
        """

        for target in ("windows-x86_64", "windows-arm64",
                       "darwin-arm64", "linux-x86_64"):
            with self.subTest(target=target):
                self.compile_for_windows(
                    """
#include <sys/types.h>
int main(void) {
    ssize_t held = -3;
    off_t place = 4096;
    return (int)(held + 3) + (int)(place - 4096)
         + (int)(sizeof(ssize_t) - sizeof(void *));
}
""",
                    target,
                )

    def test_a_piece_still_says_it_is_for_windows(self):
        with self.assertRaises(Exception) as caught:
            self.compile_for_windows(
                "#include <winnt.h>\nint main(void) { return 0; }\n",
                "linux-x86_64",
            )
        self.assertIn("is for Windows targets", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
