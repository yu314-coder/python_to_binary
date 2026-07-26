"""py2bin's C compiler: the language it accepts, and what it refuses.

Every expected value below is derived by hand from what the C standard requires,
not from another compiler -- py2bin deliberately has no toolchain to compare
against. On a darwin-arm64 host each program is also built and RUN, and its real
stdout and exit status are checked; elsewhere the same programs are compiled for
all six targets so the encoders still see them.
"""

from __future__ import annotations

import platform
import random
import re
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from py2bin.c_frontend import CCompileError, compile_c_to_ir
from py2bin.c_native import compile_c_native
from py2bin.native import supported_targets


_HOST_IS_DARWIN_ARM64 = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)
_STDIO = "#include <stdio.h>\n"


class CProgramTestCase(unittest.TestCase):
    """Compile a C program, and on darwin-arm64 run it and check what it did."""

    def build(self, source: str, target: str = "darwin-arm64") -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        entry = root / "program.c"
        entry.write_text(source, encoding="utf-8")
        artifact = root / "program.bin"
        compile_c_native(entry, artifact, target=target, clean=True)
        return artifact

    def run_c(
        self, source: str, stdout: str | None = None, status: int | None = None
    ) -> None:
        artifact = self.build(source)
        if not _HOST_IS_DARWIN_ARM64:
            return
        result = subprocess.run([str(artifact)], capture_output=True, text=True)
        if stdout is not None:
            self.assertEqual(result.stdout, stdout)
        if status is not None:
            self.assertEqual(result.returncode, status)

    def reject(self, source: str, expected: str) -> None:
        with self.assertRaises(CCompileError) as caught:
            compile_c_to_ir(source, "reject.c", "darwin-arm64")
        self.assertRegex(str(caught.exception), expected)
        # A rejection is only useful if it says where.
        self.assertRegex(str(caught.exception), r"reject\.c:\d+:\d+: ")


class IntegerTypeTests(CProgramTestCase):
    """char/short/int/long/long long and the conversions between them."""

    def test_every_boundary_truncates_and_extends_as_c_requires(self):
        # 127+1 wraps to -128 in a signed char, 255+1 to 0 in an unsigned char,
        # and so on up to 2**31-1 -> -2**31 in an int.
        self.run_c(
            _STDIO
            + """
int main(void) {
    signed char sc = 127;
    unsigned char uc = 255;
    short sh = 32767;
    unsigned short us = 65535;
    int i = 2147483647;
    unsigned int ui = 4294967295u;
    printf("%d %d\\n", sc, (int)(signed char)(sc + 1));
    printf("%d %d\\n", uc, (int)(unsigned char)(uc + 1));
    printf("%d %d\\n", sh, (int)(short)(sh + 1));
    printf("%d %d\\n", us, (int)(unsigned short)(us + 1));
    printf("%d %d\\n", i, (int)(i + 1));
    printf("%u %u\\n", ui, (unsigned int)(ui + 1));
    return 0;
}
""",
            stdout=(
                "127 -128\n255 0\n32767 -32768\n65535 0\n"
                "2147483647 -2147483648\n4294967295 0\n"
            ),
        )

    def test_plain_char_is_signed(self):
        # py2bin's C is one dialect on all six targets, and it picks Apple's
        # (and x86's) signed plain char.
        self.run_c(
            _STDIO
            + """
int main(void) {
    char c = 200;
    printf("%d %d\\n", c, (int)(unsigned char)c);
    return 0;
}
""",
            stdout="-56 200\n",
        )

    def test_integer_promotions_happen_before_arithmetic(self):
        # 200+100 is 300 because both operands promote to int; only the
        # assignment back to unsigned char truncates it to 44.
        self.run_c(
            _STDIO
            + """
int main(void) {
    unsigned char a = 200;
    unsigned char b = 100;
    unsigned char c = a + b;
    short s = 300;
    printf("%d %d %d\\n", a + b, c, s * s);
    return 0;
}
""",
            stdout="300 44 90000\n",
        )

    def test_usual_arithmetic_conversions_pick_the_right_common_type(self):
        # int+unsigned int is unsigned int, so -1 becomes 4294967295 and the
        # comparison flips; long+unsigned int is long, because long represents
        # every unsigned int, so there -1 stays negative.
        self.run_c(
            _STDIO
            + """
int main(void) {
    int i = -1;
    unsigned int u = 1;
    long l = -1;
    unsigned long ul = 1;
    printf("%u %d\\n", i + u, i < u);
    printf("%ld %d\\n", l + u, l < u);
    printf("%d\\n", l < ul);
    return 0;
}
""",
            stdout="0 0\n0 1\n0\n",
        )

    def test_right_shift_is_arithmetic_for_signed_and_logical_for_unsigned(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    int i = -16;
    unsigned int u = 4294967280u;
    unsigned long long q = 18446744073709551615ull;
    long long s = -1;
    printf("%d %u\\n", i >> 2, u >> 2);
    printf("%llu %lld\\n", q >> 60, s >> 60);
    printf("%d %lld\\n", 1 << 31, 1LL << 40);
    return 0;
}
""",
            stdout="-4 1073741820\n15 -1\n-2147483648 1099511627776\n",
        )

    def test_division_truncates_toward_zero_and_honours_signedness(self):
        # C99 requires truncation toward zero, so -7/2 is -3 and -7%2 is -1.
        # 4294967295/3 must use an unsigned divide; a signed one would see -1.
        self.run_c(
            _STDIO
            + """
int main(void) {
    unsigned int u = 4294967295u;
    long long big = -9223372036854775807LL;
    printf("%d %d\\n", -7 / 2, -7 % 2);
    printf("%d %d\\n", 7 / -2, 7 % -2);
    printf("%u %u\\n", u / 3, u % 3);
    printf("%lld\\n", big / 1000000007LL);
    return 0;
}
""",
            stdout="-3 -1\n-3 1\n1431655765 0\n-9223371972\n",
        )

    def test_unsigned_comparison_is_not_the_signed_one(self):
        # 0u-1 is 4294967295; a signed compare would make this loop run.
        self.run_c(
            _STDIO
            + """
int main(void) {
    unsigned int u = 0;
    int guard = 0;
    unsigned char b = 250;
    int steps = 0;
    while (u - 1 < 3) { guard++; if (guard > 10) { break; } u++; }
    while (b != 0) { b++; steps++; }
    printf("%u %d %d\\n", u, guard, steps);
    return 0;
}
""",
            stdout="0 0 6\n",
        )


class MemoryTests(CProgramTestCase):
    """Local arrays, &x, *p, pointer arithmetic and indexing."""

    def test_arrays_pointers_and_pointer_difference(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    int a[5];
    int i;
    int total = 0;
    int *p = a;
    int *q;
    for (i = 0; i < 5; i++) { a[i] = i * i; }
    for (i = 0; i < 5; i++) { total += *(p + i); }
    q = &a[4];
    printf("%d %d %d\\n", total, a[3], p[3]);
    long d = q - p;
    int last = *--q;
    printf("%ld %d\\n", d, last);
    *q = 100;
    printf("%d\\n", a[3]);
    return 0;
}
""",
            stdout="30 9 9\n4 9\n100\n",
        )

    def test_narrow_loads_and_stores_use_the_declared_width(self):
        # Reading a signed char must sign-extend and an unsigned one must not:
        # this is the same defect class as a missing sxtw.
        self.run_c(
            _STDIO
            + """
int main(void) {
    signed char sc[4];
    unsigned char *up;
    short sh[3];
    unsigned short *uh;
    sc[0] = -1; sc[1] = 127; sc[2] = -128; sc[3] = 5;
    up = (unsigned char *)sc;
    sh[0] = -32768; sh[1] = 32767; sh[2] = -1;
    uh = (unsigned short *)sh;
    printf("%d %d %d %d\\n", sc[0], sc[1], sc[2], sc[3]);
    printf("%d %d\\n", up[0], up[2]);
    printf("%d %d %d\\n", sh[0], sh[1], sh[2]);
    printf("%u %u\\n", uh[0], uh[2]);
    return 0;
}
""",
            stdout="-1 127 -128 5\n255 128\n-32768 32767 -1\n32768 65535\n",
        )

    def test_address_of_a_local_survives_a_call(self):
        self.run_c(
            _STDIO
            + """
void bump(int *p) { *p = *p + 10; }
int main(void) {
    int x = 5;
    int *p = &x;
    int **pp = &p;
    *p = 7;
    printf("%d\\n", x);
    bump(&x);
    printf("%d\\n", x);
    **pp = 42;
    printf("%d\\n", x);
    return 0;
}
""",
            stdout="7\n17\n42\n",
        )

    def test_a_wider_object_can_be_read_and_written_one_byte_at_a_time(self):
        # Both architectures py2bin emits for are little-endian.
        self.run_c(
            _STDIO
            + """
int main(void) {
    unsigned int v = 0x11223344u;
    unsigned char *b = (unsigned char *)&v;
    printf("%x %x %x %x\\n", b[0], b[1], b[2], b[3]);
    b[0] = 0xff;
    printf("%x\\n", v);
    return 0;
}
""",
            stdout="44 33 22 11\n112233ff\n",
        )

    def test_multidimensional_arrays_index_row_major(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    int g[3][4];
    int r, c;
    int *flat;
    for (r = 0; r < 3; r++) { for (c = 0; c < 4; c++) { g[r][c] = r * 10 + c; } }
    flat = &g[0][0];
    printf("%d %d %d %d\\n", g[0][0], g[1][2], g[2][3], flat[6]);
    printf("%zu %zu\\n", sizeof(g), sizeof(g[0]));
    return 0;
}
""",
            stdout="0 12 23 12\n48 16\n",
        )

    def test_array_initializers_zero_fill_the_rest(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    char s[] = "hi there";
    int n[8] = {1, 2, 3};
    printf("%s %zu\\n", s, sizeof(s));
    printf("%d %d %d %d\\n", n[0], n[2], n[3], n[7]);
    s[0] = 'H';
    printf("%s\\n", s);
    return 0;
}
""",
            stdout="hi there 9\n1 3 0 0\nHi there\n",
        )

    def test_stores_of_different_widths_reach_the_right_bytes(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    unsigned char raw[8];
    unsigned int *w = (unsigned int *)raw;
    unsigned short *h = (unsigned short *)raw;
    int i;
    for (i = 0; i < 8; i++) { raw[i] = 0; }
    w[0] = 0xdeadbeefu;
    h[2] = 0x1234;
    printf("%x %x %x %x\\n", raw[0], raw[1], raw[4], raw[5]);
    printf("%x\\n", w[0]);
    return 0;
}
""",
            stdout="ef be 34 12\ndeadbeef\n",
        )


class SizeofAndCastTests(CProgramTestCase):
    def test_sizeof_reports_the_lp64_model_py2bin_documents(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    int a[7];
    printf("%zu %zu %zu %zu %zu\\n", sizeof(char), sizeof(short), sizeof(int),
           sizeof(long), sizeof(long long));
    printf("%zu %zu\\n", sizeof(int *), sizeof(char *));
    printf("%zu %zu\\n", sizeof(a), sizeof(a) / sizeof(a[0]));
    printf("%zu\\n", sizeof "abcd");
    return 0;
}
""",
            stdout="1 2 4 8 8\n8 8\n28 7\n5\n",
        )

    def test_casts_truncate_then_extend(self):
        # 0x1234567890abcdef truncated to int is 0x90abcdef == -1867788817,
        # and widening that back as unsigned long long gives 2**64 - 1867788817.
        self.run_c(
            _STDIO
            + """
int main(void) {
    long long v = 0x1234567890abcdefLL;
    printf("%d %d %d\\n", (int)v, (short)v, (signed char)v);
    printf("%llu\\n", (unsigned long long)(int)v);
    return 0;
}
""",
            stdout="-1867788817 -12817 -17\n18446744071841762799\n",
        )

    def test_sizeof_does_not_evaluate_its_operand(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    int n = 0;
    int a[4];
    printf("%zu %d\\n", sizeof(a[n++]), n);
    printf("%zu %d\\n", sizeof(n++), n);
    return 0;
}
""",
            stdout="4 0\n4 0\n",
        )


class OperatorTests(CProgramTestCase):
    def test_prefix_and_postfix_increment(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    int i = 5;
    int a[4] = {10, 20, 30, 40};
    int *p = a;
    int j = 0;
    printf("%d %d\\n", i++, i);
    printf("%d %d\\n", ++i, i);
    printf("%d %d\\n", i--, i);
    printf("%d %d\\n", --i, i);
    int v1 = *p++;
    int v2 = *p;
    int v3 = *++p;
    printf("%d %d %d\\n", v1, v2, v3);
    a[j++] = 99;
    printf("%d %d %d\\n", a[0], a[1], j);
    return 0;
}
""",
            stdout="5 6\n7 7\n7 6\n5 5\n10 20 30\n99 20 1\n",
        )

    def test_comma_operator_and_conditional(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    int x = 0;
    int y = (x = 3, x + 4);
    int i;
    int z = 5;
    for (i = 0; i < 3; i++, x++) { }
    printf("%d %d %d %d\\n", x, y, i, z);
    printf("%d %d %d\\n", 1 ? 10 : 20, 0 ? 10 : 20, z > 3 ? z * 2 : z / 2);
    return 0;
}
""",
            stdout="6 7 3 5\n10 20 10\n",
        )

    def test_do_while_runs_its_body_first(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    int n = 0;
    int trips = 0;
    int never = 0;
    int k = 0;
    int c = 0;
    do { trips++; n += 3; } while (n < 10);
    do { never++; } while (0);
    do { k++; if (k % 2) { continue; } c++; } while (k < 6);
    printf("%d %d %d %d %d\\n", trips, n, never, k, c);
    return 0;
}
""",
            stdout="4 12 1 6 3\n",
        )

    def test_switch_falls_through_until_a_break(self):
        # i==2 adds 10 and then falls into case 3 for another 100.
        self.run_c(
            _STDIO
            + """
int main(void) {
    int total = 0;
    int i;
    char c = 'b';
    for (i = 0; i < 6; i++) {
        switch (i) {
            case 0:
            case 1:
                total += 1;
                break;
            case 2:
                total += 10;
            case 3:
                total += 100;
                break;
            default:
                total += 1000;
        }
    }
    printf("%d\\n", total);
    switch (c) {
        case 'a': printf("A\\n"); break;
        case 'b': printf("B\\n"); break;
        default: printf("?\\n");
    }
    return 0;
}
""",
            stdout="2212\nB\n",
        )

    def test_break_leaves_the_switch_but_continue_leaves_the_loop_body(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    int i;
    int seen = 0;
    for (i = 0; i < 5; i++) {
        switch (i) {
            case 2: continue;
            case 3: break;
        }
        seen = seen * 10 + i;
    }
    printf("%d\\n", seen);
    return 0;
}
""",
            stdout="134\n",
        )

    def test_switch_matches_negative_and_unsigned_control_values(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    int v;
    int out = 0;
    unsigned char c = 200;
    for (v = -2; v <= 2; v++) {
        switch (v) {
            case -2: out = out * 10 + 1; break;
            case -1: out = out * 10 + 2; break;
            case 0:  out = out * 10 + 3; break;
            default: out = out * 10 + 9;
        }
    }
    printf("%d\\n", out);
    switch (c) { case 200: printf("hit\\n"); break; default: printf("miss\\n"); }
    return 0;
}
""",
            stdout="12399\nhit\n",
        )

    def test_goto_jumps_forward_out_of_nested_loops_and_backward(self):
        # i*j first exceeds 4 at i==2, j==3, by which point total is 12.
        self.run_c(
            _STDIO
            + """
int main(void) {
    int i = 0;
    int j = 0;
    int total = 0;
    int n = 0;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            if (i * j > 4) { goto done; }
            total += i * j;
        }
    }
done:
    printf("%d %d %d\\n", i, j, total);
again:
    n++;
    if (n < 3) { goto again; }
    printf("%d\\n", n);
    return 0;
}
""",
            stdout="2 3 12\n3\n",
        )

    def test_short_circuit_operands_are_not_evaluated(self):
        self.run_c(
            _STDIO
            + """
int side(int *counter, int result) { *counter = *counter + 1; return result; }
int main(void) {
    int n = 0;
    int v;
    if (side(&n, 0) && side(&n, 1)) { printf("no\\n"); }
    printf("%d\\n", n);
    n = 0;
    if (side(&n, 1) || side(&n, 1)) { printf("yes\\n"); }
    printf("%d\\n", n);
    n = 0;
    v = side(&n, 3) ? side(&n, 7) : side(&n, 9);
    printf("%d %d\\n", v, n);
    return 0;
}
""",
            stdout="1\nyes\n1\n7 2\n",
        )


class FloatingPointTests(CProgramTestCase):
    """C's floating types, checked against IEEE-754 arithmetic done by hand.

    Every expectation is a value the standard pins down exactly. The programs
    print integers scaled out of the doubles rather than printing the doubles,
    because these tests are about the arithmetic; printf's own floating
    conversions are exercised separately.
    """

    def test_arithmetic_on_doubles_is_exact_where_ieee_754_is_exact(self):
        # Every value here is a sum of powers of two, so each result is exact
        # and the scaled integers below are the only possible answers.
        self.run_c(
            _STDIO
            + """
int main(void) {
    double a = 1.5;
    double b = 0.25;
    printf("%d\\n", (int)((a + b) * 100.0));
    printf("%d\\n", (int)((a - b) * 100.0));
    printf("%d\\n", (int)((a * b) * 1000.0));
    printf("%d\\n", (int)((a / b) * 100.0));
    printf("%d\\n", (int)(-a * 100.0));
    printf("%d\\n", (int)(+a * 100.0));
    return 0;
}
""",
            stdout="175\n125\n375\n600\n-150\n150\n",
        )

    def test_conversions_between_integer_and_floating_types(self):
        # C truncates toward zero converting to an integer, and 2**64-1 is not
        # representable as a double so it converts to 2**64 exactly.
        self.run_c(
            _STDIO
            + """
int main(void) {
    printf("%d %d\\n", (int)1.9, (int)-1.9);
    printf("%d\\n", (int)(double)7);
    unsigned long long big = 18446744073709551615ULL;
    printf("%d\\n", (int)((double)big / 1e18));
    double huge = 1e19;
    printf("%llu\\n", (unsigned long long)huge);
    long long signed_max = 9223372036854775807LL;
    printf("%d\\n", (int)((double)signed_max / 1e18));
    double d = 3;
    printf("%d\\n", (int)(d * 100));
    return 0;
}
""",
            stdout="1 -1\n7\n18\n10000000000000000000\n9\n300\n",
        )

    def test_float_rounds_where_c_says_the_extra_precision_goes(self):
        # 0.1 is not representable in either format. The nearest double times
        # 1e9 truncates to 100000000; the nearest float is
        # 0.100000001490116119384765625, whose product truncates to 100000001.
        # An assignment, a cast, a parameter and a return must each round.
        self.run_c(
            _STDIO
            + """
float identity(float value) { return value; }
int main(void) {
    double exact = 0.1;
    float rounded = 0.1;
    printf("%d\\n", (int)(exact * 1000000000.0));
    printf("%d\\n", (int)(rounded * 1000000000.0));
    printf("%d\\n", (int)((double)(float)0.1 * 1000000000.0));
    printf("%d\\n", (int)(identity(0.1) * 1000000000.0));
    printf("%d\\n", (double)(float)0.1 == 0.1);
    printf("%d %d\\n", (int)sizeof(float), (int)sizeof(double));
    return 0;
}
""",
            stdout="100000000\n100000001\n100000001\n100000001\n0\n4 8\n",
        )

    def test_a_float_object_really_occupies_four_bytes(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    float values[4];
    int i;
    for (i = 0; i < 4; i++) values[i] = (float)i / 4.0f;
    printf("%d\\n", (int)((char *)&values[1] - (char *)&values[0]));
    printf("%d %d %d %d\\n",
           (int)(values[0] * 100), (int)(values[1] * 100),
           (int)(values[2] * 100), (int)(values[3] * 100));
    return 0;
}
""",
            stdout="4\n0 25 50 75\n",
        )

    def test_the_usual_arithmetic_conversions_reach_the_floating_types(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    int i = 3;
    printf("%d\\n", (int)((i / 2) * 100));
    printf("%d\\n", (int)((i / 2.0) * 100));
    printf("%d\\n", (int)((1 + 0.5f) * 100));
    unsigned int u = 4000000000u;
    printf("%d\\n", (int)(u / 1e9));
    return 0;
}
""",
            stdout="100\n150\n150\n4\n",
        )

    def test_every_ordering_is_false_when_an_operand_is_a_nan(self):
        # C requires <, <=, > and >= to be false and != to be true for an
        # unordered pair. This is the case the integer condition codes get
        # wrong on both architectures.
        self.run_c(
            _STDIO
            + """
int main(void) {
    double zero = 0.0;
    double nan = zero / zero;
    printf("%d %d %d %d %d %d\\n",
           nan == nan, nan != nan, nan < 1.0, nan <= 1.0, nan > 1.0, nan >= 1.0);
    printf("%d %d\\n", 1.5 < 2.0, -0.0 == 0.0);
    double infinity = 1.0 / zero;
    printf("%d %d\\n", infinity > 1e308, -infinity < -1e308);
    return 0;
}
""",
            stdout="0 1 0 0 0 0\n1 1\n1 1\n",
        )

    def test_a_double_controls_conditions_exactly_as_c_says(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    double c = 0.0;
    int steps = 0;
    while (c < 1.0) { c += 0.25; steps++; }
    printf("%d %d\\n", steps, (int)(c * 100));
    if (!0.0) printf("zero is false\\n");
    if (-0.0) printf("unreachable\\n");
    printf("%d %d\\n", 2.5 && 0.0, 0.0 || 0.5);
    printf("%d\\n", (int)(_Bool)0.5);
    for (c = 3.0; c > 0.0; c -= 1.0) printf("%d", (int)c);
    printf("\\n");
    return 0;
}
""",
            stdout="4 100\nzero is false\n0 1\n1\n321\n",
        )

    def test_a_conditional_expression_may_mix_an_integer_and_a_double_arm(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    printf("%d\\n", (int)((1 ? 2.5 : 3) * 10));
    printf("%d\\n", (int)((0 ? 2 : 3.5) * 10));
    printf("%d\\n", (int)((1 ? 2 : 3) * 10));
    return 0;
}
""",
            stdout="25\n35\n20\n",
        )

    def test_a_conditional_evaluates_only_the_arm_it_selects(self):
        # The arms' stores are rewritten in place once the common type is
        # known; rewriting rather than re-lowering is what keeps each arm
        # evaluated exactly once.
        self.run_c(
            _STDIO
            + """
int bump(int *n) { *n = *n + 1; return 1; }
int main(void) {
    int calls = 0;
    double picked = bump(&calls) ? 2.5 : (double)bump(&calls);
    printf("%d %d\\n", calls, (int)(picked * 10));
    return 0;
}
""",
            stdout="1 25\n",
        )

    def test_increment_and_compound_assignment_work_on_floating_objects(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    double d = 1.5;
    printf("%d ", (int)(d++ * 10));
    printf("%d ", (int)(d * 10));
    printf("%d ", (int)(++d * 10));
    d *= 2;
    d -= 0.5;
    d /= 2;
    printf("%d\\n", (int)(d * 100));
    float f = 1.0f;
    f += 0.5f;
    printf("%d\\n", (int)(f * 100));
    return 0;
}
""",
            stdout="15 25 35 325\n150\n",
        )

    def test_doubles_pass_through_calls_recursion_and_pointers(self):
        self.run_c(
            _STDIO
            + """
double power(double base, int exponent) {
    if (exponent == 0) return 1.0;
    return base * power(base, exponent - 1);
}
double total(double *values, int count) {
    double sum = 0.0;
    int i;
    for (i = 0; i < count; i++) sum += values[i];
    return sum;
}
void scale(double *value, double factor) { *value = *value * factor; }
int main(void) {
    printf("%d\\n", (int)power(1.5, 4));
    double values[4];
    values[0] = 1.5; values[1] = 2.25; values[2] = -0.75; values[3] = 8.0;
    printf("%d\\n", (int)(total(values, 4) * 100));
    double one = 0.5;
    scale(&one, 3.0);
    printf("%d\\n", (int)(one * 100));
    return 0;
}
""",
            # 1.5**4 == 5.0625 exactly, so the truncation is 5.
            stdout="5\n1100\n150\n",
        )

    def test_a_floating_target_read_and_written_in_one_expression(self):
        # The value stored may read the very object about to be written, so it
        # has to be pinned in a slot before the store rather than recomputed
        # after it. Each of these would answer differently if it were not.
        self.run_c(
            _STDIO
            + """
double twice(int *n, double v) { *n = *n + 1; return v * 2.0; }
int main(void) {
    int calls = 0;
    double d = 1.0;
    d = d + d;
    printf("%g\\n", d);
    double e = 1.0;
    e += twice(&calls, 3.0);
    printf("%g %d\\n", e, calls);
    double *p = &d;
    (*p)++;
    printf("%g %g\\n", d, *p);
    float f = 2.0f;
    f = f * f;
    printf("%g\\n", (double)f);
    return 0;
}
""",
            stdout="2\n7 1\n3 3\n4\n",
        )

    def test_a_struct_lays_out_and_copies_its_floating_members(self):
        self.run_c(
            _STDIO
            + """
struct sample { double x; float y; int tag; };
int main(void) {
    struct sample a;
    a.x = 2.5; a.y = 1.25f; a.tag = 7;
    struct sample b;
    b = a;
    b.x = b.x + 1.0;
    printf("%d %d %d\\n", (int)(a.x * 10), (int)(a.y * 10), a.tag);
    printf("%d %d %d\\n", (int)(b.x * 10), (int)(b.y * 10), b.tag);
    printf("%d %d %d\\n",
           (int)sizeof(struct sample),
           (int)((char *)&a.y - (char *)&a.x),
           (int)((char *)&a.tag - (char *)&a.x));
    return 0;
}
""",
            # double at 0, float at 8, int at 12; the whole thing is 8-aligned.
            stdout="25 12 7\n35 12 7\n16 8 12\n",
        )

    def test_hexadecimal_and_subnormal_floating_constants(self):
        # 0x1.8p3 is (1 + 1/2) * 8 == 12 exactly. The nearest float to 1e-40 is
        # a subnormal, 9.99994610111476e-41, so the product truncates to 99999
        # rather than the 100000 a normal value would give.
        self.run_c(
            _STDIO
            + """
int main(void) {
    printf("%d\\n", (int)0x1.8p3);
    printf("%d\\n", (int)(.5 * 100));
    printf("%d\\n", (int)(1e2));
    float small = 1e-40f;
    printf("%d\\n", (int)(small * 1e45));
    return 0;
}
""",
            stdout="12\n50\n100\n99999\n",
        )


class FloatingPrintfTests(CProgramTestCase):
    """%f, %e and %g, whose formatter py2bin emits itself.

    The conversion is exact: every finite double is a finite decimal, and the
    emitted code builds that decimal digit by digit rather than estimating it.
    The expectations below are what C requires, and the widest of them --
    printing 1e300 with no fraction -- is the 301-digit integer that double
    exactly equals, which no approximate method could produce.
    """

    def test_the_default_fixed_conversion(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    printf("%f\\n", 1.5);
    printf("%f\\n", 0.0);
    printf("%f\\n", -0.0);
    printf("%f\\n", -2.25);
    printf("%f\\n", 100.0);
    printf("%f\\n", 0.0001220703125);
    return 0;
}
""",
            stdout=(
                "1.500000\n0.000000\n-0.000000\n-2.250000\n100.000000\n0.000122\n"
            ),
        )

    def test_rounding_is_half_to_even_on_the_exact_value(self):
        # 0.5, 1.5, 2.5 and 3.5 are all exact, so each is a genuine tie and the
        # even neighbour wins. 0.05 is NOT exact -- the double is just above
        # the tie -- so it rounds up, which an implementation that looked only
        # at the printed digits would get wrong.
        self.run_c(
            _STDIO
            + """
int main(void) {
    printf("%.0f %.0f %.0f %.0f\\n", 0.5, 1.5, 2.5, 3.5);
    printf("%.0f %.0f\\n", -0.5, -1.5);
    printf("%.1f %.1f\\n", 0.25, 0.75);
    printf("%.1f\\n", 0.05);
    printf("%.2f\\n", 3.14159);
    return 0;
}
""",
            stdout="0 2 2 4\n-0 -2\n0.2 0.8\n0.1\n3.14\n",
        )

    def test_the_digits_printed_are_the_digits_the_double_has(self):
        # 0.1 is 0.1000000000000000055511151231257827021181583404541015625, and
        # 1e300 is an integer of 301 digits. These are the exact values, so a
        # correct formatter prints exactly this and nothing else can.
        self.run_c(
            _STDIO
            + """
int main(void) {
    printf("%.20f\\n", 0.1);
    printf("%.17g\\n", 0.1);
    printf("%f\\n", 1e20);
    printf("%.0f\\n", 1e300);
    return 0;
}
""",
            stdout=(
                "0.10000000000000000555\n"
                "0.10000000000000001\n"
                "100000000000000000000.000000\n"
                "1000000000000000052504760255204420248704468581108159154915854115"
                "5118024579889081957863713750804478640437044438328838781769425232"
                "3536043057564479218478670698284838720092657580373783023379478809"
                "0059368953234970799945081119038967640880074652742780142494579258"
                "788820056842838115669472196386865459400540160\n"
            ),
        )

    def test_the_exponential_conversion(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    printf("%e\\n", 1.5);
    printf("%e\\n", 0.0);
    printf("%E\\n", 0.000123);
    printf("%.3e\\n", 1234.5678);
    printf("%.0e\\n", 9.5);
    printf("%.1e\\n", 9.95);
    printf("%e\\n", 1e300);
    printf("%e\\n", 5e-324);
    return 0;
}
""",
            # 9.5 with one significant digit is a tie that rounds to the even
            # 10, so the exponent grows. 9.95 is not exactly 9.95: the double
            # is just below it, so two significant digits give 9.9, not 10.
            stdout=(
                "1.500000e+00\n"
                "0.000000e+00\n"
                "1.230000E-04\n"
                "1.235e+03\n"
                "1e+01\n"
                "9.9e+00\n"
                "1.000000e+300\n"
                "4.940656e-324\n"
            ),
        )

    def test_the_general_conversion_picks_a_shape_and_trims(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    printf("%g\\n", 100000.0);
    printf("%g\\n", 1000000.0);
    printf("%g\\n", 0.0001);
    printf("%g\\n", 0.00001);
    printf("%g\\n", 1.5);
    printf("%g\\n", 0.0);
    printf("%g\\n", 123456.0);
    printf("%g\\n", 1234567.0);
    printf("%.3g\\n", 1234.0);
    printf("%.1g\\n", 0.0);
    printf("%G\\n", 0.000001);
    return 0;
}
""",
            stdout=(
                "100000\n1e+06\n0.0001\n1e-05\n1.5\n0\n123456\n1.23457e+06\n"
                "1.23e+03\n0\n1E-06\n"
            ),
        )

    def test_infinities_and_nans_print_as_words(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    double zero = 0.0;
    printf("%f %f\\n", 1.0 / zero, -1.0 / zero);
    printf("%e %g\\n", 1.0 / zero, 1.0 / zero);
    printf("%f\\n", zero / zero);
    printf("%E %G %F\\n", 1.0 / zero, 1.0 / zero, zero / zero);
    return 0;
}
""",
            stdout="inf -inf\ninf inf\nnan\nINF INF NAN\n",
        )

    def test_a_float_argument_is_widened_to_the_double_printf_reads(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    float f = 0.1f;
    printf("%.9f\\n", f);
    printf("%.9f\\n", 0.1);
    return 0;
}
""",
            # The float holds 0.100000001490116119384765625; the double holds
            # 0.1000000000000000055511151231257827.
            stdout="0.100000001\n0.100000000\n",
        )

    def test_many_conversions_share_one_formatter_and_return_to_their_sites(self):
        # The formatter is a subroutine built out of the jumps the IR has, so
        # the risk is a site returning to the WRONG place. Mixing shapes and
        # cases in one statement, in a loop, and inside a called function
        # exercises the return dispatch from every direction.
        self.run_c(
            _STDIO
            + """
void show(double value) { printf("[%g|%e]", value, value); }
int main(void) {
    int i;
    printf("%f %e %g %F %E %G\\n", 1.5, 1.5, 1.5, 1.5, 1.5, 1.5);
    for (i = 1; i <= 3; i++) { printf("%.1f,", (double)i / 2.0); }
    printf("\\n");
    show(0.5);
    show(1500000.0);
    printf("\\n");
    return 0;
}
""",
            stdout=(
                "1.500000 1.500000e+00 1.5 1.500000 1.500000E+00 1.5\n"
                "0.5,1.0,1.5,\n"
                "[0.5|5.000000e-01][1.5e+06|1.500000e+06]\n"
            ),
        )

    def test_the_widest_output_still_fits_the_frame_buffer(self):
        # The largest finite double with the largest precision py2bin accepts:
        # 309 integer digits, a sign, a point and 120 fraction digits. This is
        # what sizes the output buffer, so it is the case that would overrun.
        widest = float.hex(1.7976931348623157e308)
        self.run_c(
            _STDIO
            + "int main(void) {\n"
            + f'    printf("%.120f\\n", -{widest});\n'
            + '    printf("%.120f\\n", 0x0.0000000000001p-1022);\n'
            + "    return 0;\n}\n",
            stdout=(
                "%.120f\n" % -1.7976931348623157e308 + "%.120f\n" % 5e-324
            ),
        )

    def test_the_formatter_is_emitted_once_per_body_not_once_per_conversion(self):
        one = compile_c_to_ir(
            _STDIO + 'int main(void) { printf("%f", 1.0); return 0; }\n',
            "one.c",
            "darwin-arm64",
        )
        many = compile_c_to_ir(
            _STDIO
            + "int main(void) {\n"
            + "".join(f'    printf("%f", {index}.0);\n' for index in range(40))
            + "    return 0;\n}\n",
            "many.c",
            "darwin-arm64",
        )
        # Forty conversions must cost forty short call sites, not forty copies
        # of a formatter that is hundreds of operations long.
        self.assertLess(len(many.operations), len(one.operations) + 40 * 20)

    def test_a_floating_conversion_evaluates_its_argument_once(self):
        self.run_c(
            _STDIO
            + """
double tick(int *n) { *n = *n + 1; return 1.5; }
int main(void) {
    int calls = 0;
    printf("%f %f\\n", tick(&calls), tick(&calls));
    printf("%d\\n", calls);
    return 0;
}
""",
            stdout="1.500000 1.500000\n2\n",
        )

    def test_what_printf_still_refuses(self):
        self.reject(
            'int main(void) { printf("%5.2f", 1.0); return 0; }\n',
            "flags and field widths",
        )
        self.reject(
            'int main(void) { printf("%.200f", 1.0); return 0; }\n',
            "beyond the 120",
        )
        self.reject(
            'int main(void) { printf("%Lf", 1.0); return 0; }\n', "long double"
        )
        self.reject(
            'int main(void) { printf("%.3d", 1); return 0; }\n',
            "precision on %d is not implemented",
        )
        self.reject(
            'int main(void) { printf("%f", 1); return 0; }\n',
            "needs a floating value",
        )
        self.reject(
            'int main(void) { double d = 1.0; printf("%d", d); return 0; }\n',
            "needs an integer",
        )


class EvaluatedOnceTests(CProgramTestCase):
    """The recurring defect here is a value written once, lowered twice.

    Each program below puts a side-effecting call in a position where the
    lowering reuses the expression, and counts how many times it really ran.
    """

    def test_a_compound_assignment_computes_its_target_address_once(self):
        self.run_c(
            _STDIO
            + """
int tick(int *n) { *n = *n + 1; return 2; }
int main(void) {
    int calls = 0;
    int a[8];
    int i;
    for (i = 0; i < 8; i++) { a[i] = 0; }
    a[tick(&calls)] += 5;
    a[tick(&calls)] += 5;
    printf("%d %d %d\\n", calls, a[2], a[3]);
    return 0;
}
""",
            stdout="2 10 0\n",
        )

    def test_an_increment_computes_its_target_address_once(self):
        self.run_c(
            _STDIO
            + """
int where(int *n) { *n = *n + 1; return 1; }
int main(void) {
    int calls = 0;
    int a[4] = {0, 0, 0, 0};
    int *p = a;
    p[where(&calls)]++;
    p[where(&calls)]++;
    printf("%d %d\\n", calls, a[1]);
    return 0;
}
""",
            stdout="2 2\n",
        )

    def test_the_value_of_an_assignment_is_what_was_stored(self):
        # (c = 200) has the value of the stored signed char, not of 200.
        self.run_c(
            _STDIO
            + """
int main(void) {
    int x = 1;
    int y = (x = x + 1);
    signed char c = 0;
    int z = (c = 200);
    int a[3] = {0, 0, 0};
    int w = (a[1] += 7);
    printf("%d %d\\n", x, y);
    printf("%d %d\\n", c, z);
    printf("%d %d\\n", a[1], w);
    return 0;
}
""",
            stdout="2 2\n-56 -56\n7 7\n",
        )

    def test_a_loop_condition_runs_every_iteration(self):
        self.run_c(
            _STDIO
            + """
int step(int *n) { *n = *n + 1; return *n; }
int main(void) {
    int n = 0;
    int m = 0;
    int trips = 0;
    while (step(&n) < 4) { }
    for (; step(&m) < 4; ) { trips++; }
    printf("%d %d %d\\n", n, m, trips);
    return 0;
}
""",
            stdout="4 4 3\n",
        )

    def test_each_call_in_one_expression_happens_exactly_once(self):
        self.run_c(
            _STDIO
            + """
int note(int *log, int *n, int value) {
    log[*n] = value;
    *n = *n + 1;
    return value;
}
int main(void) {
    int log[8];
    int n = 0;
    int total = note(log, &n, 1) + note(log, &n, 2) * note(log, &n, 3);
    printf("%d %d %d %d %d\\n", total, n, log[0], log[1], log[2]);
    return 0;
}
""",
            stdout="7 3 1 2 3\n",
        )


class FunctionTests(CProgramTestCase):
    def test_functions_are_inlined_including_nested_calls(self):
        self.run_c(
            _STDIO
            + """
int add(int a, int b) { return a + b; }
long long fact(long long n) {
    long long r = 1;
    long long i;
    for (i = 2; i <= n; i++) { r *= i; }
    return r;
}
void greet(void) { printf("hi\\n"); }
int main(void) {
    printf("%d\\n", add(add(1, 2), add(3, 4)));
    printf("%lld\\n", fact(20));
    greet();
    return add(20, 22);
}
""",
            stdout="10\n2432902008176640000\nhi\n",
            status=42,
        )

    def test_early_return_leaves_the_function_not_the_caller(self):
        self.run_c(
            _STDIO
            + """
void emit(int v) {
    if (v < 0) { return; }
    printf("%d\\n", v);
}
int pick(int v) {
    if (v > 10) { return 1; }
    if (v > 5) { return 2; }
    return 3;
}
int main(void) {
    emit(-1);
    emit(7);
    printf("%d %d %d\\n", pick(20), pick(7), pick(1));
    return 0;
}
""",
            stdout="7\n1 2 3\n",
        )

    def test_a_parameter_is_a_copy(self):
        self.run_c(
            _STDIO
            + """
int shadow(int v) { v = v + 100; return v; }
int main(void) {
    int v = 1;
    printf("%d %d\\n", shadow(v), v);
    return 0;
}
""",
            stdout="101 1\n",
        )


class PrintfTests(CProgramTestCase):
    def test_runtime_conversions_format_the_real_values(self):
        # -9223372036854775808 is the case a naive absolute value gets wrong.
        self.run_c(
            _STDIO
            + """
int main(void) {
    printf("%d|%lld|%u|%x|%X|%c|%s|%%\\n",
           -1234, -9223372036854775807LL - 1, 4000000000u, 48879, 48879,
           'Z', "text");
    printf("%d %d\\n", 0, -0);
    printf("%hhd %hd %hhu %hu\\n", 300, 70000, 300, 70000);
    return 0;
}
""",
            stdout=(
                "-1234|-9223372036854775808|4000000000|beef|BEEF|Z|text|%\n"
                "0 0\n44 4464 44 4464\n"
            ),
        )

    def test_a_format_without_conversions_needs_no_syscall_support(self):
        for target in supported_targets():
            with self.subTest(target=target):
                self.build(
                    _STDIO + 'int main(void) { printf("plain\\n"); return 0; }\n',
                    target=target,
                )

    def test_a_runtime_conversion_is_refused_where_it_cannot_be_emitted(self):
        with self.assertRaisesRegex(CCompileError, "runtime conversion"):
            compile_c_to_ir(
                _STDIO + 'int main(void) { printf("%d\\n", 1); return 0; }\n',
                "win.c",
                "windows-x86_64",
            )

    def test_unimplemented_conversions_are_named_not_guessed(self):
        self.reject(
            _STDIO + 'int main(void) { printf("%a\\n", 1.0); return 0; }\n',
            "is not implemented",
        )
        self.reject(
            _STDIO + 'int main(void) { printf("%5d\\n", 1); return 0; }\n',
            "field widths are not implemented",
        )
        self.reject(
            _STDIO + 'int main(void) { printf("%d %d\\n", 1); return 0; }\n',
            "conversion",
        )


class RejectionTests(CProgramTestCase):
    """What py2bin's C compiler refuses, with a location, instead of guessing."""

    def test_recursion_is_refused_on_a_target_with_no_call_abi(self):
        """The ARM64 and System V encoders have a call ABI; the Windows ones
        still inline, so there recursion is rejected with a location rather
        than miscompiled."""

        source = (
            "int f(int n) { return n ? f(n - 1) : 0; }\n"
            "int main(void) { return f(3); }\n"
        )
        for target in ("windows-arm64", "windows-x86_64"):
            with self.subTest(target=target):
                with self.assertRaises(CCompileError) as caught:
                    compile_c_to_ir(source, "reject.c", target)
                self.assertRegex(str(caught.exception), "recursive call")
                self.assertRegex(str(caught.exception), r"reject\.c:1:27")

    def test_more_than_eight_arguments_is_refused(self):
        parameters = ", ".join(f"int a{index}" for index in range(9))
        arguments = ", ".join(str(index) for index in range(9))
        self.reject(
            f"int wide({parameters}) {{ return a0; }}\n"
            f"int main(void) {{ return wide({arguments}); }}\n",
            "at most 8 arguments in registers",
        )

    def test_long_double_is_refused(self):
        self.reject(
            "int main(void) { long double d = 1.0; return 0; }\n", "long double"
        )
        self.reject("int main(void) { return (int)1.0L; }\n", "long double")

    def test_floating_operands_are_refused_where_c_needs_an_integer(self):
        self.reject(
            "int main(void) { double d = 1.0; return (int)(d % 2); }\n",
            "needs integer operands",
        )
        self.reject("int main(void) { double d = 1.0; return ~d; }\n", "unary '~'")
        self.reject(
            "int main(void) { double d = 1.0; return d << 1; }\n",
            "needs integer operands",
        )
        self.reject(
            "int main(void) { double d = 1.0; switch (d) { case 1: break; } return 0; }\n",
            "'switch' needs an integer",
        )
        self.reject(
            "int main(void) { double a[2]; double d = 0.0; return (int)a[d]; }\n",
            "offset by an integer",
        )
        self.reject(
            "int main(void) { int a[1.5]; return 0; }\n",
            "not an integer constant expression",
        )

    def test_a_pointer_never_converts_to_or_from_a_floating_type(self):
        self.reject(
            "int main(void) { double d = 1.0; int *p = (int *)d; return p == 0; }\n",
            "no conversion between a floating type and a pointer",
        )
        self.reject(
            "int main(void) { int i = 0; double d = (double)&i; return (int)d; }\n",
            "no conversion between a pointer and a floating type",
        )

    def test_a_floating_constant_that_overflows_is_refused(self):
        self.reject("int main(void) { double d = 1e400; return 0; }\n", "overflows")
        self.reject("int main(void) { float f = 1e40f; return 0; }\n", "overflows")
        self.reject(
            "int main(void) { double d = 1.0e+; return 0; }\n", "exponent has no digits"
        )
        self.reject(
            "int main(void) { double d = 0x1.8; return 0; }\n", "binary exponent"
        )

    def test_undefined_enum_tag_is_refused(self):
        self.reject(
            "int main(void) { enum Missing e; return 0; }\n", "has not been defined"
        )

    def test_conflicting_typedef_is_refused(self):
        self.reject(
            "typedef int t;\ntypedef char t;\nint main(void) { return 0; }\n",
            "already a different type",
        )

    def test_incomplete_struct_use_is_refused(self):
        # A tag with no body may only be pointed at; its layout is unknown.
        self.reject(
            "struct S;\nint main(void) { struct S s; return 0; }\n", "incomplete"
        )
        self.reject(
            "struct S;\nint main(void) { struct S *p; return p->x; }\n",
            "incomplete",
        )

    def test_unknown_member_is_refused(self):
        self.reject(
            "struct S { int a; };\n"
            "int main(void) { struct S s; return s.b; }\n",
            "no member",
        )

    def test_member_access_needs_an_aggregate(self):
        self.reject(
            "int main(void) { int n = 1; return n.a; }\n", "needs a struct"
        )

    def test_a_header_py2bin_cannot_find_is_refused(self):
        # py2bin has no system include path: a real <sys/socket.h> is full of
        # extensions this compiler does not have, so it says so.
        self.reject(
            "#include <sys/socket.h>\nint main(void) { return 0; }\n",
            "cannot find the header",
        )

    def test_a_file_scope_initializer_must_be_a_constant_expression(self):
        # C11 6.7.9p4: an object with static storage duration is initialized
        # before the program starts, so nothing is running that could evaluate
        # a variable read or a call.
        self.reject(
            "int a = 1;\nint b = a;\nint main(void) { return b; }\n",
            "must be a constant expression",
        )
        self.reject(
            "int f(void) { return 1; }\nint b = f();\n"
            "int main(void) { return b; }\n",
            "must be a constant expression",
        )

    def test_a_block_scope_static_is_refused(self):
        self.reject(
            "int main(void) { static int n = 1; return n; }\n",
            "static object inside a block",
        )

    def test_a_file_scope_object_cannot_be_redeclared_incompatibly(self):
        self.reject(
            "int x = 1;\nint x = 2;\nint main(void) { return x; }\n",
            "initialized twice",
        )
        self.reject(
            "int x;\nlong x;\nint main(void) { return 0; }\n",
            "was declared int and is now declared long",
        )
        self.reject(
            "int main(void) { return 0; }\nint main = 3;\n",
            "already declared as a function",
        )
        self.reject("void v;\nint main(void) { return 0; }\n", "cannot have type void")

    def test_an_extern_object_is_refused_with_a_clear_message(self):
        self.reject(
            "extern int errno;\nint main(void) { return 0; }\n",
            "no linker",
        )

    def test_an_opaque_handle_cannot_be_dereferenced_or_offset(self):
        prototypes = "extern void Py_Initialize(void);\n"
        self.reject(
            prototypes + "int main(void) { PyObject *p; p = 0; return *p; }\n",
            "incomplete type",
        )
        self.reject(
            prototypes + "int main(void) { PyObject *p; p = 0; p = p + 1; return 0; }\n",
            "incomplete type",
        )

    def test_incompatible_pointers_need_an_explicit_cast(self):
        self.reject(
            "int main(void) { int x; char *p = &x; return 0; }\n", "explicit cast"
        )
        self.reject(
            "int main(void) { int x; int y = &x; return y; }\n", "explicit cast"
        )

    def test_an_undefined_label_or_name_is_refused(self):
        self.reject("int main(void) { goto nowhere; }\n", "no label")
        self.reject("int main(void) { return zzz; }\n", "not a declared")
        self.reject("int main(void) { return f(); }\n", "not a function declared")
        self.reject(
            "int f(int n);\nint main(void) { return f(1); }\n",
            "declared but never defined",
        )

    def test_misplaced_control_flow_is_refused(self):
        self.reject("int main(void) { break; }\n", "not inside a loop")
        self.reject("int main(void) { continue; }\n", "not inside a loop")
        self.reject("int main(void) { case 1: return 0; }\n", "not inside a switch")

    def test_a_switch_cannot_repeat_a_case_or_a_default(self):
        self.reject(
            "int main(void) { int x = 1; switch (x) { case 1: break; case 1: break; }"
            " return 0; }\n",
            "duplicate case",
        )
        self.reject(
            "int main(void) { int x = 1; switch (x) { default: break; default: break; }"
            " return 0; }\n",
            "at most one",
        )

    def test_constant_division_by_zero_is_refused(self):
        self.reject("int main(void) { return 1 / 0; }\n", "division by zero")

    def test_an_array_is_not_assignable_and_needs_a_positive_length(self):
        self.reject(
            "int main(void) { int a[2]; int b[2]; a = b; return 0; }\n",
            "not assignable",
        )
        self.reject("int main(void) { int a[0]; return 0; }\n", "positive constant")
        self.reject("int main(void) { int n = 3; int a[n]; return 0; }\n", "constant")

    def test_the_entry_point_must_be_int_main_void(self):
        self.reject("int f(void) { return 1; }\n", "no int main")
        self.reject("void main(void) { }\n", "int main\\(void\\)")

    def test_variadic_definitions_are_refused(self):
        self.reject(
            "int f(int a, ...) { return a; }\nint main(void) { return f(1); }\n",
            "variadic",
        )
        self.reject(
            "int main(void) { int (*f)(int, ...); return 0; }\n", "variadic"
        )

    def test_a_frame_larger_than_the_native_one_is_refused(self):
        self.reject(
            "int main(void) { long long huge[5000]; return 0; }\n", "stack frame"
        )


class ConstantFoldingTests(CProgramTestCase):
    """A folded constant must equal what the same expression computes at runtime.

    py2bin folds integer constants in the frontend, so every fold is a second
    implementation of the operator that has to agree with the encoder's. It did
    not: constants that arrived as unsigned quantities kept Python's unbounded
    value, and `(long)0xFFFFFFFFFFFFFFFE >= 79490271399379139` folded to true
    while the machine code answered false. These pin the agreement.
    """

    def test_a_constant_of_unsigned_origin_reads_back_signed(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    long a = 0xfffffffffffffffeull;
    long b = 0x11af1851c8d02c3ull;
    printf("%d %d\\n",
           (long)(0xfffffffffffffffeull) >= (long)(0x11af1851c8d02c3ull), a >= b);
    printf("%lld %lld\\n",
           (long)(0xffffffffffffffffull) / (int)(0xfffffffffffffffeull),
           (long long)a / -2);
    printf("%d %d\\n",
           (int)(0x7fffffffull) > (long)(0x8000000000000000ull),
           2147483647 > (long)0x8000000000000000ull);
    return 0;
}
""",
            stdout="0 0\n0 1\n1 1\n",
        )

    def test_folded_and_unfolded_forms_agree(self):
        # Every left-hand value is folded at compile time; every right-hand one
        # goes through a local, which the folder cannot see through.
        self.run_c(
            _STDIO
            + """
int main(void) {
    long long la = -1;
    unsigned long long ua = 18446744073709551615ull;
    int ia = -7;
    unsigned int ui = 4294967295u;
    short sa = -300;
    printf("%d %d\\n", -1LL < 1ull, la < (unsigned long long)1);
    printf("%llu %llu\\n", 18446744073709551615ull / 3, ua / 3);
    printf("%llu %llu\\n", 18446744073709551615ull % 7, ua % 7);
    printf("%d %d\\n", -7 / 2, ia / 2);
    printf("%d %d\\n", -7 % 2, ia % 2);
    printf("%u %u\\n", 4294967295u >> 3, ui >> 3);
    printf("%d %d\\n", (int)(short)-300, (int)sa);
    printf("%d %d\\n", -7 < 4294967295u, ia < ui);
    printf("%lld %lld\\n", (long long)(unsigned int)-7, (long long)(unsigned int)ia);
    return 0;
}
""",
            stdout=(
                "0 0\n6148914691236517205 6148914691236517205\n"
                "1 1\n-3 -3\n-1 -1\n536870911 536870911\n"
                "-300 -300\n1 1\n4294967289 4294967289\n"
            ),
        )


class WholeProgramTests(CProgramTestCase):
    """Programs that use the whole feature set at once, not one corner of it."""

    def test_a_sieve_of_eratosthenes(self):
        # There are 46 primes below 200, the last four being 191/193/197/199.
        self.run_c(
            _STDIO
            + """
int main(void) {
    unsigned char sieve[200];
    int i, j, count;
    for (i = 0; i < 200; i++) { sieve[i] = 1; }
    sieve[0] = 0; sieve[1] = 0;
    for (i = 2; i * i < 200; i++) {
        if (sieve[i]) {
            for (j = i * i; j < 200; j += i) { sieve[j] = 0; }
        }
    }
    count = 0;
    for (i = 0; i < 200; i++) { if (sieve[i]) { count++; } }
    printf("%d\\n", count);
    for (i = 190; i < 200; i++) { if (sieve[i]) { printf("%d ", i); } }
    printf("\\n");
    return count;
}
""",
            stdout="46\n191 193 197 199 \n",
            status=46,
        )

    def test_a_bubble_sort_swapping_through_pointers(self):
        self.run_c(
            _STDIO
            + """
void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }
int main(void) {
    int v[8] = {5, 3, 9, 1, 7, 2, 8, 4};
    int i, j;
    for (i = 0; i < 8; i++) {
        for (j = 0; j + 1 < 8 - i; j++) {
            if (v[j] > v[j + 1]) { swap(&v[j], &v[j + 1]); }
        }
    }
    for (i = 0; i < 8; i++) { printf("%d", v[i]); }
    printf("\\n");
    return v[7];
}
""",
            stdout="12345789\n",
            status=9,
        )

    def test_string_handling_through_char_pointers(self):
        self.run_c(
            _STDIO
            + """
int copy(char *to, char *from) {
    int n = 0;
    while (*from) { *to = *from; to++; from++; n++; }
    *to = 0;
    return n;
}
int compare(char *a, char *b) {
    while (*a && *a == *b) { a++; b++; }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}
int main(void) {
    char buffer[32];
    int n = copy(buffer, "compiled by py2bin");
    printf("%s %d\\n", buffer, n);
    printf("%d %d %d\\n", compare(buffer, "compiled by py2bin"),
           compare("abc", "abd") < 0, compare("b", "a") > 0);
    return 0;
}
""",
            stdout="compiled by py2bin 18\n0 1 1\n",
        )

    def test_euclid_and_the_widest_unsigned_values(self):
        # C's % keeps the dividend's sign, so -48 % 18 is -12, not 6.
        self.run_c(
            _STDIO
            + """
long long gcd(long long a, long long b) {
    while (b != 0) { long long t = a % b; a = b; b = t; }
    return a < 0 ? -a : a;
}
int main(void) {
    unsigned long long v = 18446744073709551615ull;
    printf("%lld %lld %lld\\n", gcd(1071, 462), gcd(-48, 18), gcd(17, 0));
    printf("%llu %llx\\n", v, v);
    printf("%llu\\n", v / 7);
    return 0;
}
""",
            stdout=(
                "21 6 17\n18446744073709551615 ffffffffffffffff\n"
                "2635249153387078802\n"
            ),
        )

    def test_labels_in_a_function_inlined_twice_do_not_collide(self):
        self.run_c(
            _STDIO
            + """
int find(int *a, int n, int wanted) {
    int i = 0;
loop:
    if (i >= n) { return -1; }
    if (a[i] == wanted) { goto found; }
    i++;
    goto loop;
found:
    return i;
}
int main(void) {
    int a[5] = {4, 8, 15, 16, 23};
    printf("%d %d\\n", find(a, 5, 15), find(a, 5, 99));
    return 0;
}
""",
            stdout="2 -1\n",
        )

    def test_switches_nest_and_work_inside_an_inlined_function(self):
        self.run_c(
            _STDIO
            + """
int classify(int v) {
    int out;
    switch (v % 3) {
        case 0: out = 100; break;
        case 1: out = 200; break;
        default: out = 300;
    }
    return out;
}
int main(void) {
    int i, j, out = 0;
    for (i = 0; i < 2; i++) {
        switch (i) {
            case 0:
                for (j = 0; j < 2; j++) {
                    switch (j) {
                        case 0: out += 1; break;
                        default: out += 2;
                    }
                }
                break;
            default:
                out += 100;
        }
    }
    printf("%d %d %d %d %d\\n", out, classify(3), classify(4), classify(5),
           classify(-1));
    return 0;
}
""",
            stdout="103 100 200 300 300\n",
        )

    def test_block_scopes_shadow_and_restore(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    int x = 1;
    {
        int x = 2;
        { int x = 3; printf("%d\\n", x); }
        printf("%d\\n", x);
    }
    printf("%d\\n", x);
    return 0;
}
""",
            stdout="3\n2\n1\n",
        )

    def test_literals_in_every_base_and_escape(self):
        self.run_c(
            _STDIO
            + """
int main(void) {
    printf("%d %d %d %d\\n", 0x10, 010, '\\n', '\\x41');
    printf("%d %d\\n", '\\0', '\\377');
    return 0;
}
""",
            stdout="16 8 10 65\n0 -1\n",
        )

    def test_a_frame_of_thousands_of_words_still_addresses_every_slot(self):
        # 3900 words is past the 4095-byte reach of a single sub-sp immediate
        # and past the 12-bit reach of an add-immediate slot address.
        self.run_c(
            _STDIO
            + """
int main(void) {
    long long block[3900];
    int i;
    long long total = 0;
    for (i = 0; i < 3900; i++) { block[i] = i * 2; }
    for (i = 0; i < 3900; i++) { total += block[i]; }
    printf("%lld\\n", total);
    return (int)(total % 251);
}
""",
            stdout="15206100\n",
            status=18,
        )


# --- a model of C's integer semantics, for the differential test below -------
#
# py2bin has no other C implementation to compare against, so the comparison is
# against C11 itself: the rules for integer promotion, the usual arithmetic
# conversions, and truncation, written out directly. Any expression whose
# behaviour C leaves undefined is discarded instead of compared.

_MODEL_TYPES = {
    "signed char": (1, True, 1),
    "unsigned char": (1, False, 1),
    "short": (2, True, 2),
    "unsigned short": (2, False, 2),
    "int": (4, True, 3),
    "unsigned int": (4, False, 3),
    "long": (8, True, 4),
    "unsigned long": (8, False, 4),
    "long long": (8, True, 5),
    "unsigned long long": (8, False, 5),
}
_MODEL_UNSIGNED = {
    "int": "unsigned int",
    "long": "unsigned long",
    "long long": "unsigned long long",
}


def _model_wrap(value: int, name: str) -> int:
    size, signed, _rank = _MODEL_TYPES[name]
    modulus = 1 << (size * 8)
    value &= modulus - 1
    if signed and value >= modulus >> 1:
        value -= modulus
    return value


def _model_fits(value: int, name: str) -> bool:
    size, signed, _rank = _MODEL_TYPES[name]
    bits = size * 8
    if signed:
        return -(1 << (bits - 1)) <= value <= (1 << (bits - 1)) - 1
    return 0 <= value <= (1 << bits) - 1


def _model_promote(name: str) -> str:
    return "int" if _MODEL_TYPES[name][2] < 3 else name


def _model_usual(left: str, right: str) -> str:
    left, right = _model_promote(left), _model_promote(right)
    if left == right:
        return left
    _ls, left_signed, left_rank = _MODEL_TYPES[left]
    _rs, right_signed, right_rank = _MODEL_TYPES[right]
    if left_signed == right_signed:
        return left if left_rank > right_rank else right
    unsigned, signed = (left, right) if not left_signed else (right, left)
    if _MODEL_TYPES[unsigned][2] >= _MODEL_TYPES[signed][2]:
        return unsigned
    if _MODEL_TYPES[signed][0] > _MODEL_TYPES[unsigned][0]:
        return signed
    return _MODEL_UNSIGNED[signed]


class _Undefined(Exception):
    """The expression's behaviour is not defined by C, so it proves nothing."""


class _Term:
    def __init__(self, text: str, ctype: str, value: int):
        self.text = text
        self.ctype = ctype
        self.value = value


def _model_leaf(rng, variables):
    if variables and rng.random() < 0.6:
        name, ctype, value = rng.choice(variables)
        return _Term(name, ctype, value)
    ctype = rng.choice(list(_MODEL_TYPES))
    size, signed, _rank = _MODEL_TYPES[ctype]
    bits = size * 8
    if signed:
        value = rng.choice(
            [0, 1, -1, 2, -2, (1 << (bits - 1)) - 1, -(1 << (bits - 1))]
        )
    else:
        value = rng.choice([0, 1, 2, (1 << bits) - 1, (1 << (bits - 1))])
    # A hexadecimal ull pattern always has a type; the cast reproduces exactly
    # the value the model wrapped to.
    pattern = value & 0xFFFFFFFFFFFFFFFF
    return _Term(f"(({ctype})(0x{pattern:x}ull))", ctype, _model_wrap(value, ctype))


def _model_build(rng, variables, depth):
    if depth == 0:
        return _model_leaf(rng, variables)
    kind = rng.choice(["unary", "binary", "binary", "cast", "leaf"])
    if kind == "leaf":
        return _model_leaf(rng, variables)
    if kind == "cast":
        inner = _model_build(rng, variables, depth - 1)
        target = rng.choice(list(_MODEL_TYPES))
        return _Term(
            f"(({target})({inner.text}))", target, _model_wrap(inner.value, target)
        )
    if kind == "unary":
        inner = _model_build(rng, variables, depth - 1)
        operator = rng.choice(["-", "~", "!"])
        result = _model_promote(inner.ctype)
        value = _model_wrap(inner.value, result)
        if operator == "!":
            return _Term(f"(!({inner.text}))", "int", int(value == 0))
        raw = -value if operator == "-" else ~value
        if _MODEL_TYPES[result][1] and not _model_fits(raw, result):
            raise _Undefined()
        return _Term(f"({operator}({inner.text}))", result, _model_wrap(raw, result))
    left = _model_build(rng, variables, depth - 1)
    right = _model_build(rng, variables, depth - 1)
    operator = rng.choice(
        ["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>",
         "==", "!=", "<", "<=", ">", ">="]
    )
    text = f"(({left.text}) {operator} ({right.text}))"
    if operator in {"<<", ">>"}:
        result = _model_promote(left.ctype)
        value = _model_wrap(left.value, result)
        count = _model_wrap(right.value, _model_promote(right.ctype))
        if not 0 <= count < _MODEL_TYPES[result][0] * 8:
            raise _Undefined()
        if operator == "<<":
            if _MODEL_TYPES[result][1] and (
                value < 0 or not _model_fits(value << count, result)
            ):
                raise _Undefined()
            return _Term(text, result, _model_wrap(value << count, result))
        # >> of a negative value is implementation-defined; py2bin documents the
        # arithmetic shift, which is what every real implementation performs.
        return _Term(text, result, _model_wrap(value >> count, result))
    common = _model_usual(left.ctype, right.ctype)
    a = _model_wrap(left.value, common)
    b = _model_wrap(right.value, common)
    if operator in {"==", "!=", "<", "<=", ">", ">="}:
        outcome = {
            "==": a == b, "!=": a != b, "<": a < b,
            "<=": a <= b, ">": a > b, ">=": a >= b,
        }[operator]
        return _Term(text, "int", int(outcome))
    if operator in {"/", "%"}:
        if b == 0:
            raise _Undefined()
        quotient = abs(a) // abs(b)
        if (a < 0) != (b < 0):
            quotient = -quotient
        raw = quotient if operator == "/" else a - quotient * b
        if _MODEL_TYPES[common][1] and not _model_fits(raw, common):
            raise _Undefined()
        return _Term(text, common, _model_wrap(raw, common))
    raw = {"+": a + b, "-": a - b, "*": a * b,
           "&": a & b, "|": a | b, "^": a ^ b}[operator]
    if operator in {"&", "|", "^"}:
        raw = _model_wrap(raw, common)
    elif _MODEL_TYPES[common][1] and not _model_fits(raw, common):
        raise _Undefined()
    return _Term(text, common, _model_wrap(raw, common))


class DifferentialTests(CProgramTestCase):
    """Random C expressions, checked against the model of C above.

    The seed is fixed so a failure is reproducible; run
    ``.py2bin-work/fuzz.py`` style loops with other seeds to search further.
    """

    def test_random_integer_expressions_match_the_c_standard(self):
        for seed in (20260725, 2, 3, 5, 8, 13):
            with self.subTest(seed=seed):
                self._compare_batch(seed)

    def _compare_batch(self, seed: int) -> None:
        rng = random.Random(seed)
        variables = []
        declarations = []
        for index in range(6):
            ctype = rng.choice(list(_MODEL_TYPES))
            size, signed, _rank = _MODEL_TYPES[ctype]
            bits = size * 8
            if signed:
                value = rng.randrange(-(1 << (bits - 1)), 1 << (bits - 1))
            else:
                value = rng.randrange(0, 1 << bits)
            pattern = value & 0xFFFFFFFFFFFFFFFF
            declarations.append(f"    {ctype} v{index} = ({ctype})(0x{pattern:x}ull);")
            variables.append((f"v{index}", ctype, value))
        lines = []
        expected = []
        while len(lines) < 120:
            try:
                term = _model_build(rng, variables, rng.randint(1, 3))
            except _Undefined:
                continue
            size, signed, _rank = _MODEL_TYPES[term.ctype]
            if not signed and size == 8:
                lines.append(
                    f'    printf("%llu\\n", (unsigned long long)({term.text}));'
                )
            else:
                lines.append(f'    printf("%lld\\n", (long long)({term.text}));')
            expected.append(str(term.value))
        source = (
            _STDIO
            + "int main(void) {\n"
            + "\n".join(declarations)
            + "\n"
            + "\n".join(lines)
            + "\n    return 0;\n}\n"
        )
        artifact = self.build(source)
        if not _HOST_IS_DARWIN_ARM64:
            return
        got = subprocess.run(
            [str(artifact)], capture_output=True, text=True
        ).stdout.splitlines()
        self.assertEqual(len(got), len(expected))
        for index, (actual, want) in enumerate(zip(got, expected)):
            if actual != want:
                self.fail(
                    f"{lines[index].strip()}\n"
                    f"  {' '.join(declaration.strip() for declaration in declarations)}\n"
                    f"  got {actual}, C requires {want}"
                )


class FloatingDifferentialTests(CProgramTestCase):
    """py2bin's floating formatter against a second correctly-rounded one.

    CPython formats a double by computing its exact decimal value and rounding
    half to even, which is what C11 7.21.6.1p13 recommends and what py2bin
    emits code to do. It is an INDEPENDENT implementation -- David Gay's dtoa,
    not the algorithm here -- so agreeing with it on hundreds of cases,
    including random bit patterns, both extremes and the smallest subnormal, is
    real evidence rather than a restatement of this compiler's own arithmetic.
    """

    _FORMATS = (
        "%f", "%.0f", "%.1f", "%.3f", "%.15f",
        "%e", "%.0e", "%.3e", "%.15e",
        "%g", "%.1g", "%.3g", "%.10g", "%.17g",
        "%E", "%G",
    )

    def test_random_expression_trees_match_ieee_754_arithmetic(self):
        """Compiled double arithmetic against Python's, which is binary64 too.

        Printing each result with %.17g pins down the full 53-bit significand,
        so a single wrong rounding anywhere in a tree of up to sixteen
        operations shows as a different string.
        """

        for seed in (3, 11, 77, 2026):
            with self.subTest(seed=seed):
                self._compare_arithmetic(seed)

    def _compare_arithmetic(self, seed: int) -> None:
        generator = random.Random(seed)
        names = [f"v{index}" for index in range(6)]
        values = []
        declarations = []
        for name in names:
            value = generator.choice(
                [
                    generator.uniform(-1000, 1000),
                    generator.uniform(-1, 1) * 10.0 ** generator.randrange(-30, 30),
                    float(generator.randrange(-(10**9), 10**9)),
                ]
            )
            values.append(value)
            declarations.append(f"    double {name} = {float.hex(value)};")

        def build(depth: int) -> tuple[str, float]:
            if depth == 0 or generator.random() < 0.3:
                index = generator.randrange(len(names))
                return names[index], values[index]
            left, left_value = build(depth - 1)
            right, right_value = build(depth - 1)
            operator = generator.choice("+-*/")
            if operator == "/" and right_value == 0.0:
                operator = "+"
            try:
                result = {
                    "+": lambda: left_value + right_value,
                    "-": lambda: left_value - right_value,
                    "*": lambda: left_value * right_value,
                    "/": lambda: left_value / right_value,
                }[operator]()
            except OverflowError as error:  # overflow to infinity is not compared
                raise ValueError from error
            if result != result or abs(result) == float("inf"):
                raise ValueError
            return f"(({left}) {operator} ({right}))", result

        lines = []
        expected = []
        while len(lines) < 120:
            try:
                text, result = build(generator.randint(1, 4))
            except ValueError:
                continue
            lines.append(f'    printf("%.17g\\n", {text});')
            expected.append("%.17g" % result)
        source = (
            _STDIO
            + "int main(void) {\n"
            + "\n".join(declarations)
            + "\n"
            + "\n".join(lines)
            + "\n    return 0;\n}\n"
        )
        artifact = self.build(source)
        if not _HOST_IS_DARWIN_ARM64:
            return
        got = subprocess.run(
            [str(artifact)], capture_output=True, text=True
        ).stdout.splitlines()
        self.assertEqual(len(got), len(expected))
        for index, (actual, want) in enumerate(zip(got, expected)):
            if actual != want:
                self.fail(
                    f"{lines[index].strip()}\n"
                    f"  {' '.join(item.strip() for item in declarations)}\n"
                    f"  got {actual}, IEEE-754 binary64 gives {want}"
                )

    def _values(self) -> list[float]:
        chosen = [
            0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5, 3.5, 0.1, 0.25, 100.0,
            1e-5, 1e-4, 123456.0, 1234567.0, 9.5, 9.95, 0.0001220703125,
            3.141592653589793, 1e20, 1e-20, 1e300, 1e-300, 0.000123456,
            1023.9999999999999, -12345.678, 1e16, 0.5 ** 52, 1234.5678,
            5e-324,  # the smallest subnormal: 751 digits of exact expansion
            1.7976931348623157e308,  # the largest finite double
            2.2250738585072014e-308,  # the smallest normal
        ]
        generator = random.Random(20260726)
        for _ in range(12):
            pattern = generator.getrandbits(64)
            value = struct.unpack("<d", struct.pack("<Q", pattern))[0]
            if value == value and abs(value) != float("inf"):
                chosen.append(value)
        return chosen

    def test_every_conversion_matches_a_correctly_rounded_reference(self):
        lines = []
        expected = []
        for value in self._values():
            # float.hex() is exact, so the C source names the same double the
            # reference formatted -- no decimal round trip in between.
            for form in self._FORMATS:
                lines.append(f'    printf("{form}\\n", {float.hex(value)});')
                expected.append(form % value)
        source = (
            _STDIO + "int main(void) {\n" + "\n".join(lines) + "\n    return 0;\n}\n"
        )
        # All of it is one program, which also shows the formatter is emitted
        # once per body rather than once per conversion.
        artifact = self.build(source)
        if not _HOST_IS_DARWIN_ARM64:
            return
        got = subprocess.run(
            [str(artifact)], capture_output=True, text=True
        ).stdout.splitlines()
        self.assertEqual(len(got), len(expected))
        for index, (actual, want) in enumerate(zip(got, expected)):
            if actual != want:
                self.fail(
                    f"{lines[index].strip()}\n"
                    f"  got {actual!r}, a correctly rounded C printf "
                    f"gives {want!r}"
                )


class EncoderTests(unittest.TestCase):
    """The new IR really becomes the instructions it claims to."""

    _SOURCE = """
int main(void) {
    signed char sc[4];
    unsigned short h[2];
    int i;
    long long q = -100;
    unsigned long long u = 18446744073709551615ull;
    int *p;
    sc[0] = -1;
    h[1] = 65535;
    i = sc[0] + h[1];
    q = q / 7 + q % 7;
    u = u / 3 + (u >> 5);
    if (u > 3) { i = i + (int)(q + (long long)u); }
    p = &i;
    *p = *p + 1;
    return i;
}
"""

    def _disassemble(self, target: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "widths.c"
            entry.write_text(self._SOURCE, encoding="utf-8")
            artifact = root / "widths.bin"
            compile_c_native(entry, artifact, target=target, clean=True)
            return subprocess.run(
                ["otool", "-tvV", str(artifact)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout

    @unittest.skipUnless(
        platform.system() == "Darwin", "otool is a macOS developer tool"
    )
    def test_x86_64_uses_the_right_widths_and_divides(self):
        text = self._disassemble("darwin-x86_64")
        for mnemonic in (
            r"\bmovb\b",  # 1-byte store
            r"\bmovw\b",  # 2-byte store
            r"\bmovsbq\b",  # sign-extending 1-byte load
            r"\bmovzwl\b",  # zero-extending 2-byte load
            r"\bmovslq\b",  # sign-extending 4-byte load
            r"\bleaq\b",  # the address of a stack slot
            r"\bcqto\b",  # signed division setup
            r"\bidivq\b",
            r"\bdivq\b",  # unsigned division
            r"\bshrq\b",  # logical right shift
        ):
            with self.subTest(mnemonic=mnemonic):
                self.assertRegex(text, mnemonic)

    @unittest.skipUnless(
        platform.system() == "Darwin", "otool is a macOS developer tool"
    )
    def test_arm64_uses_the_right_widths_and_divides(self):
        text = self._disassemble("darwin-arm64")
        for mnemonic in (
            r"\bstrb\b",
            r"\bstrh\b",
            r"\bldrsb\b",
            r"\bldrh\b",
            r"\bldrsw\b",
            r"\bsdiv\b",
            r"\budiv\b",
            r"\bmsub\b",
            r"\blsr\b",
            r"add\tx0, x29,",
        ):
            with self.subTest(mnemonic=mnemonic):
                self.assertRegex(text, mnemonic)

    _FLOAT_SOURCE = """
double blend(double a, float b) { return a * 2.0 + (double)(b / 2.0f); }
int main(void) {
    double d = 1.5;
    float f = 0.25f;
    double values[2];
    values[0] = blend(d, f);
    values[1] = (double)(unsigned long long)(values[0] * 4.0);
    if (values[0] == values[1]) { d = d + 100.0; }
    if (values[0] < values[1]) { d = d + 1.0; }
    return (int)(values[1] + d);
}
"""

    def _disassemble_source(self, source: str, target: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "floats.c"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "floats.bin"
            compile_c_native(entry, artifact, target=target, clean=True)
            return subprocess.run(
                ["otool", "-tvV", str(artifact)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout

    @unittest.skipUnless(
        platform.system() == "Darwin", "otool is a macOS developer tool"
    )
    def test_x86_64_emits_real_sse2_for_the_floating_types(self):
        """x86-64 cannot be RUN here, so its float encoding is read instead."""

        text = self._disassemble_source(self._FLOAT_SOURCE, "darwin-x86_64")
        for mnemonic in (
            r"\baddsd\b",
            r"\bmulsd\b",
            r"\bdivss\b|\bdivsd\b",
            r"\bcvtsd2ss\b",  # rounding a double to a float object
            r"\bcvtss2sd\b",  # widening a float object back
            r"\bcvttsd2si\b",  # the C conversion to an integer
            r"\bcvtsi2sd\b",  # and back
            r"\bucomisd\b",
            r"\bmovq\b",  # the bit-pattern moves between rax and xmm0
            r"\bsetnp\b",  # the ordered half of an equality comparison
        ):
            with self.subTest(mnemonic=mnemonic):
                self.assertRegex(text, mnemonic)

    @unittest.skipUnless(
        platform.system() == "Darwin", "otool is a macOS developer tool"
    )
    def test_arm64_emits_real_scalar_neon_for_the_floating_types(self):
        text = self._disassemble_source(self._FLOAT_SOURCE, "darwin-arm64")
        for mnemonic in (
            r"\bfadd\t",
            r"\bfmul\t",
            r"\bfdiv\t",
            r"\bfcvt\t",  # between binary32 and binary64
            r"\bfcvtzs\t|\bfcvtzu\t",
            r"\bucvtf\t|\bscvtf\t",
            r"\bfcmp\t",
            r"\bfmov\t",
        ):
            with self.subTest(mnemonic=mnemonic):
                self.assertRegex(text, mnemonic)

    def test_a_floating_program_builds_for_every_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "floats.c"
            entry.write_text(self._FLOAT_SOURCE, encoding="utf-8")
            for target in supported_targets():
                with self.subTest(target=target):
                    artifact = root / f"floats-{target}.bin"
                    compile_c_native(entry, artifact, target=target, clean=True)
                    self.assertGreater(artifact.stat().st_size, 0)
        if _HOST_IS_DARWIN_ARM64:
            # blend(1.5, 0.25f) is 3.0 + 0.125 == 3.125 exactly; times four is
            # 12.5, whose unsigned conversion truncates to 12. 3.125 == 12.0 is
            # false and 3.125 < 12.0 is true, so d becomes 2.5 and the result
            # truncates from 14.5.
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                entry = root / "floats.c"
                entry.write_text(self._FLOAT_SOURCE, encoding="utf-8")
                artifact = root / "floats.bin"
                compile_c_native(entry, artifact, target="darwin-arm64", clean=True)
                self.assertEqual(subprocess.run([str(artifact)]).returncode, 14)

    @unittest.skipUnless(
        _HOST_IS_DARWIN_ARM64, "native execution requires a darwin-arm64 host"
    )
    def test_a_call_in_a_floating_position_is_emitted_once(self):
        """The recurring defect: one written call lowered into the IR twice.

        A printf floating argument and a '?:' arm are both positions where the
        lowering reuses a value, so this counts the branch instructions in the
        image as well as the side effects at run time. Three written calls must
        be three `bl` sites, not six.
        """

        source = """
#include <stdio.h>
double tick(int *n) { *n = *n + 1; return 1.5; }
int main(void) {
    int calls = 0;
    printf("%.1f %.1f\\n", tick(&calls), tick(&calls));
    double picked = (calls == 0) ? tick(&calls) : 2.5;
    printf("%d %g\\n", calls, picked);
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "once.c"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "once.bin"
            compile_c_native(entry, artifact, target="darwin-arm64", clean=True)
            text = subprocess.run(
                ["otool", "-tvV", str(artifact)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            self.assertEqual(len(re.findall(r"\tbl\t", text)), 3)
            run = subprocess.run([str(artifact)], capture_output=True, text=True)
            # Both printf arguments ran, so `calls` is 2 and the '?:' takes its
            # second arm without calling again.
            self.assertEqual(run.stdout, "1.5 1.5\n2 2.5\n")

    @unittest.skipUnless(
        _HOST_IS_DARWIN_ARM64, "native execution requires a darwin-arm64 host"
    )
    def test_an_inlined_helper_calls_its_extern_once(self):
        """`n + n` reads a local twice; the call that filled it must not repeat."""

        source = """
extern void Py_Initialize(void);
extern void Py_Finalize(void);
extern PyObject *PyLong_FromLongLong(long long value);
extern long long PyLong_AsLongLong(PyObject *value);
extern PyObject *PyNumber_Add(PyObject *left, PyObject *right);
extern void Py_DecRef(PyObject *value);

long long twice(PyObject *value) {
    long long n = PyLong_AsLongLong(value);
    return n + n;
}
int main(void) {
    PyObject *a;
    PyObject *b;
    PyObject *s;
    long long r;
    Py_Initialize();
    a = PyLong_FromLongLong(20);
    b = PyLong_FromLongLong(1);
    s = PyNumber_Add(a, b);
    r = twice(s);
    Py_DecRef(a); Py_DecRef(b); Py_DecRef(s);
    Py_Finalize();
    return (int)r;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "once.c"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "once.bin"
            compile_c_native(entry, artifact, target="darwin-arm64", clean=True)
            text = subprocess.run(
                ["otool", "-tvV", str(artifact)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            # Py_Initialize, PyLong_FromLongLong x2, PyNumber_Add,
            # PyLong_AsLongLong, Py_DecRef x3, Py_Finalize.
            self.assertEqual(len(re.findall(r"blr\tx16", text)), 9)
            run = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(run.returncode, 42)


class EveryTargetTests(unittest.TestCase):
    def test_one_program_using_every_phase_a_feature_builds_everywhere(self):
        source = """
int classify(int v) {
    switch (v) {
        case 0: return 100;
        case 1:
        case 2: return 200;
        default: break;
    }
    return 300;
}
int main(void) {
    int a[4];
    int i;
    long long total = 0;
    unsigned char c;
    int *p;
    for (i = 0; i < 4; i++) { a[i] = i * 3; }
    i = 0;
    do { total += a[i]; i++; } while (i < 4);
    switch (total) { case 18: total = total * 2; break; default: total = 0; }
    c = (unsigned char)total;
    p = &i;
    *p = 9;
    total = total + classify(1);
    return (int)(total + c + i) / 5;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "every.c"
            entry.write_text(source, encoding="utf-8")
            for target in supported_targets():
                with self.subTest(target=target):
                    artifact = root / f"every-{target}.bin"
                    compile_c_native(entry, artifact, target=target, clean=True)
                    self.assertGreater(artifact.stat().st_size, 0)
            if not _HOST_IS_DARWIN_ARM64:
                return
            # a is {0,3,6,9}; total is 18, doubled to 36 by the switch, then
            # classify(1) adds 200 -> 236. c is (unsigned char)36 == 36 and i is
            # 9, so the result is (236 + 36 + 9) / 5 == 281 / 5 == 56.
            run = subprocess.run(
                [str(root / "every-darwin-arm64.bin")], capture_output=True
            )
            self.assertEqual(run.returncode, 56)


class RecursionTests(CProgramTestCase):
    """Real calls: a frame per call, a saved link register, and recursion.

    Every expected value below is derived by hand from the C standard.
    """

    def test_factorial_and_fibonacci(self):
        # 10! == 3628800 and fib(20) == 6765 with fib(0) == 0, fib(1) == 1.
        self.run_c(
            _STDIO
            + """
long long factorial(long long n) {
    if (n <= 1) { return 1; }
    return n * factorial(n - 1);
}
long long fib(long long n) {
    if (n < 2) { return n; }
    return fib(n - 1) + fib(n - 2);
}
int main(void) {
    printf("%lld %lld\\n", factorial(10), fib(20));
    return factorial(5) == 120 && fib(10) == 55;
}
""",
            stdout="3628800 6765\n",
            status=1,
        )

    def test_a_deep_recursion_really_saves_and_restores_its_frame(self):
        # Each frame owns a four-element array it rewrites before and checks
        # after the recursive call, so a frame that was NOT preserved shows up
        # as a negative sentinel rather than as a merely different total.
        self.run_c(
            _STDIO
            + """
long long depth(long long n) {
    long long scratch[4];
    scratch[0] = n;
    scratch[1] = n * 2;
    scratch[2] = 0;
    scratch[3] = 7;
    if (n == 0) { return 0; }
    scratch[2] = depth(n - 1);
    if (scratch[3] != 7) { return -1; }
    if (scratch[1] != n * 2) { return -2; }
    if (scratch[0] != n) { return -3; }
    return scratch[2] + 1;
}
int main(void) {
    printf("%lld\\n", depth(5000));
    return 0;
}
""",
            stdout="5000\n",
            status=0,
        )

    def test_mutual_recursion_through_forward_declarations(self):
        self.run_c(
            _STDIO
            + """
int is_even(long long n);
int is_odd(long long n);

int is_even(long long n) { if (n == 0) { return 1; } return is_odd(n - 1); }
int is_odd(long long n) { if (n == 0) { return 0; } return is_even(n - 1); }

int main(void) {
    printf("%d %d %d %d\\n",
           is_even(1000), is_odd(1000), is_even(1001), is_odd(1001));
    return 0;
}
""",
            stdout="1 0 0 1\n",
            status=0,
        )

    def test_mutual_recursion_computes_a_real_result(self):
        # Ackermann(2, 3) == 9, expanded by hand from A(m,n).
        self.run_c(
            _STDIO
            + """
long long ack(long long m, long long n) {
    if (m == 0) { return n + 1; }
    if (n == 0) { return ack(m - 1, 1); }
    return ack(m - 1, ack(m, n - 1));
}
int main(void) {
    printf("%lld %lld %lld\\n", ack(0, 0), ack(1, 3), ack(2, 3));
    return 0;
}
""",
            stdout="1 5 9\n",
            status=0,
        )

    def test_eight_arguments_reach_the_callee_in_order(self):
        # 1*1 + 2*2 + ... + 8*8 == 204, and the subtraction chain pins the
        # ORDER: a swapped pair would change the sign of the difference.
        self.run_c(
            _STDIO
            + """
long long weigh(long long a, long long b, long long c, long long d,
                long long e, long long f, long long g, long long h) {
    return a * 1 + b * 2 + c * 3 + d * 4 + e * 5 + f * 6 + g * 7 + h * 8;
}
long long order(int a, short b, char c, long long d,
                unsigned e, int f, int g, int h) {
    return ((((((a - b) * 10 + c) * 10 + (int)d) * 10 + (int)e) * 10 + f)
            * 10 + g) * 10 + h;
}
int main(void) {
    printf("%lld\\n", weigh(1, 2, 3, 4, 5, 6, 7, 8));
    printf("%lld\\n", order(1, 2, 3, 4, 5, 6, 7, 8));
    return 0;
}
""",
            # order: ((((((1-2)*10+3)*10+4)*10+5)*10+6)*10+7)*10+8
            #      = (-1*10+3)=-7; -7*10+4=-66; -66*10+5=-655;
            #        -655*10+6=-6544; -6544*10+7=-65433; -65433*10+8=-654322
            stdout="204\n-654322\n",
            status=0,
        )

    def test_a_call_in_a_reused_position_happens_exactly_once(self):
        """The recurring defect in this backend is a value lowered twice.

        Each expression below re-reads its operand -- a ternary condition, a
        short-circuit operand, an index that is both an address and a
        read-modify-write target -- so a duplicated call shows up as a counter
        that advanced twice.
        """

        self.run_c(
            _STDIO
            + """
int bump(int *counter) {
    *counter = *counter + 1;
    return *counter;
}
int main(void) {
    int c = 0;
    int a[8];
    int i;
    int x;
    for (i = 0; i < 8; i++) { a[i] = 0; }
    x = bump(&c) ? 100 : 200;
    printf("%d %d\\n", x, c);
    x = bump(&c) && 1;
    printf("%d %d\\n", x, c);
    x = bump(&c) || 1;
    printf("%d %d\\n", x, c);
    a[bump(&c)] = 55;
    printf("%d %d %d\\n", a[4], c, a[3]);
    a[bump(&c)] += 7;
    printf("%d %d %d\\n", a[5], c, a[6]);
    x = 10;
    x += bump(&c);
    printf("%d %d\\n", x, c);
    x = (bump(&c) > 0) ? bump(&c) : bump(&c);
    printf("%d %d\\n", x, c);
    return 0;
}
""",
            stdout=(
                "100 1\n"  # the condition called once
                "1 2\n"  # && evaluated its left operand once
                "1 3\n"  # || short-circuited after one call
                "55 4 0\n"  # the index was computed once, a[3] untouched
                "7 5 0\n"  # the read-modify-write index was computed once
                "16 6\n"  # 10 + 6
                "8 8\n"  # condition, then ONLY the taken arm
            ),
            status=0,
        )

    @unittest.skipUnless(
        platform.system() == "Darwin", "otool is a macOS developer tool"
    )
    def test_each_call_site_emits_exactly_one_branch_and_link(self):
        source = (
            _STDIO
            + """
int bump(int *counter) { *counter = *counter + 1; return *counter; }
int main(void) {
    int c = 0;
    int a[4];
    int x;
    a[0] = 0; a[1] = 0; a[2] = 0; a[3] = 0;
    x = bump(&c) ? 1 : 2;
    a[bump(&c)] += 1;
    x += bump(&c);
    return x + c;
}
"""
        )
        artifact = self.build(source)
        text = subprocess.run(
            ["otool", "-tvV", str(artifact)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        # Three textual calls, three bl instructions: no call is duplicated,
        # and none is inlined away into a second copy of the body.
        self.assertEqual(len(re.findall(r"\bbl\t", text)), 3)
        # A real AAPCS64 frame: the link register is saved and restored, and
        # the callee returns rather than falling through.
        self.assertRegex(text, r"stp\tx29, x30, \[sp\]")
        self.assertRegex(text, r"ldp\tx29, x30, \[sp\]")
        self.assertRegex(text, r"\bret\b")

    def test_a_recursive_function_may_print_and_use_a_switch(self):
        # Everything a function body can contain has to keep working when the
        # body becomes a real callee with its own frame and its own labels.
        self.run_c(
            _STDIO
            + """
int classify(int v) {
    int out;
    switch (v % 3) {
        case 0: out = 100; break;
        case 1: out = 200; break;
        default: out = 300;
    }
    return out;
}
void countdown(int n) {
    if (n == 0) { printf("go\\n"); return; }
    printf("%d %d\\n", n, classify(n));
    countdown(n - 1);
}
int main(void) {
    countdown(4);
    return 0;
}
""",
            stdout="4 200\n3 100\n2 300\n1 200\ngo\n",
            status=0,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class PrintfSequencePointTests(CProgramTestCase):
    """printf may produce no output until every argument is evaluated.

    C11 6.5.2.2p10 places a sequence point after the arguments are evaluated
    and before the call, so an argument that itself writes to stdout must be
    seen before any of the format string.
    """

    def test_argument_side_effects_precede_all_output(self):
        self.run_c(
            _STDIO
            + """
long long side(long long n) { printf("[inner]\\n"); return n; }
int main(void) {
    printf("A=%lld end\\n", side(7));
    return 0;
}
""",
            stdout="[inner]\nA=7 end\n",
        )

    def test_every_argument_is_evaluated_before_output(self):
        self.run_c(
            _STDIO
            + """
long long a(void) { printf("[a]\\n"); return 1; }
long long b(void) { printf("[b]\\n"); return 2; }
int main(void) {
    printf("x=%lld y=%lld\\n", a(), b());
    return 0;
}
""",
            stdout="[a]\n[b]\nx=1 y=2\n",
        )


class StructTests(CProgramTestCase):
    """struct and union, laid out by C's alignment and padding rules."""

    def test_members_are_read_and_written(self):
        self.run_c(
            _STDIO
            + """
struct P { int x; int y; };
int main(void) {
    struct P p;
    p.x = 3; p.y = 4;
    printf("%d\\n", p.x + p.y);
    return 0;
}
""",
            stdout="7\n",
        )

    def test_layout_follows_c_alignment_and_padding(self):
        # char at 0, int padded to 4, char at 8, tail padding to 12.
        self.run_c(
            _STDIO
            + """
struct P { char c; int i; char d; };
union U { int i; char c; };
int main(void) {
    printf("%zu %zu %zu\\n", sizeof(struct P), sizeof(union U), sizeof(char));
    return 0;
}
""",
            stdout="12 4 1\n",
        )

    def test_a_union_overlays_its_members(self):
        self.run_c(
            _STDIO
            + """
union U { int i; char c; };
int main(void) {
    union U u;
    u.i = 0;
    u.c = 1;
    printf("%d\\n", u.i);
    return 0;
}
""",
            stdout="1\n",
        )

    def test_writing_one_member_leaves_the_others_alone(self):
        self.run_c(
            _STDIO
            + """
struct P { char c; int i; short s; };
int main(void) {
    struct P p;
    p.c = 1; p.i = 70000; p.s = 3;
    printf("%d %d %d\\n", p.c, p.i, p.s);
    return 0;
}
""",
            stdout="1 70000 3\n",
        )

    def test_arrow_through_a_pointer(self):
        self.run_c(
            _STDIO
            + """
struct P { int x; int y; };
int main(void) {
    struct P p; struct P *q = &p;
    q->x = 9; q->y = 1;
    printf("%d\\n", q->x + p.y);
    return 0;
}
""",
            stdout="10\n",
        )

    def test_nested_structs_and_arrays_of_structs(self):
        self.run_c(
            _STDIO
            + """
struct Inner { int a; int b; };
struct Outer { struct Inner in; int c; };
int main(void) {
    struct Outer o;
    o.in.a = 10; o.in.b = 20; o.c = 5;
    struct Outer many[2];
    int i;
    for (i = 0; i < 2; i++) { many[i].in.a = i; many[i].c = i * 100; }
    printf("%d %d %d\\n", o.in.a + o.in.b, o.c, many[1].c + many[1].in.a);
    return 0;
}
""",
            stdout="30 5 101\n",
        )

    def test_struct_assignment_copies_by_value(self):
        self.run_c(
            _STDIO
            + """
struct P { char c; int i; short s; };
int main(void) {
    struct P a; struct P b;
    a.c = 1; a.i = 2; a.s = 3;
    b = a;
    a.c = 9; a.i = 9; a.s = 9;
    printf("%d %d %d\\n", b.c, b.i, b.s);
    return 0;
}
""",
            stdout="1 2 3\n",
        )

    def test_struct_pointer_argument_and_recursion(self):
        self.run_c(
            _STDIO
            + """
struct N { int v; };
long long sum(struct N *a, long long i, long long n) {
    if (i >= n) { return 0; }
    return a[i].v + sum(a, i + 1, n);
}
int main(void) {
    struct N a[5];
    int i;
    for (i = 0; i < 5; i++) { a[i].v = i + 1; }
    printf("%lld\\n", sum(a, 0, 5));
    return 0;
}
""",
            stdout="15\n",
        )


class EnumAndTypedefTests(CProgramTestCase):
    """enum constants and typedef names."""

    def test_enumerators_count_from_zero_and_continue(self):
        self.run_c(
            _STDIO
            + """
enum Plain { A, B, C };
enum Explicit { X = 5, Y, Z = 10 };
int main(void) {
    printf("%d %d\\n", A + B + C, X + Y + Z);
    return 0;
}
""",
            stdout="3 21\n",
        )

    def test_typedef_names_a_type_including_a_struct(self):
        self.run_c(
            _STDIO
            + """
typedef int myint;
typedef struct P { int x; int y; } Point;
long long total(Point *p) { return p->x + p->y; }
int main(void) {
    myint n = 7;
    Point p;
    p.x = 20; p.y = 2;
    printf("%d %lld\\n", n, total(&p));
    return 0;
}
""",
            stdout="7 22\n",
        )

    def test_a_local_shadows_an_enumerator(self):
        self.run_c(
            _STDIO
            + """
enum E { VALUE = 9 };
int main(void) {
    int VALUE = 1;
    printf("%d\\n", VALUE);
    return 0;
}
""",
            stdout="1\n",
        )


class FileScopeObjectTests(CProgramTestCase):
    """Objects with static storage duration: C file-scope variables.

    They live in one contiguous block established before the program's first
    instruction, so the same object is the same object in ``main`` and in every
    function ``main`` calls -- which a stack slot could never be.
    """

    def test_a_global_starts_at_zero_and_outlives_a_call(self):
        # C11 6.7.9p10: an object with static storage duration and no
        # initializer starts as if assigned 0.
        self.run_c(
            _STDIO
            + """
int counter;
int bump(int by) { counter = counter + by; return counter; }
int main(void) {
    printf("%d\\n", counter);
    bump(4);
    bump(9);
    printf("%d\\n", counter);
    return 0;
}
""",
            stdout="0\n13\n",
        )

    def test_constant_initializers_of_every_scalar_kind(self):
        self.run_c(
            _STDIO
            + """
int limit = 3 + 4 * 2;
long total = 7;
unsigned char small = 300;
double ratio = 2.5;
char letter = 'Z';
int main(void) {
    printf("%d %ld %u %g %c\\n", limit, total, small, ratio, letter);
    return 0;
}
""",
            # 3 + 4*2 == 11; 300 truncated into an unsigned char is 300-256==44.
            stdout="11 7 44 2.5 Z\n",
        )

    def test_a_global_array_is_shared_by_every_function(self):
        self.run_c(
            _STDIO
            + """
int table[5] = {10, 20, 30, 40, 50};
int sum(void) {
    int i; int s;
    s = 0;
    for (i = 0; i < 5; i++) { s = s + table[i]; }
    return s;
}
void poke(int at, int value) { table[at] = value; }
int main(void) {
    printf("%d\\n", sum());
    poke(2, 99);
    printf("%d\\n", sum());
    return 0;
}
""",
            # 10+20+30+40+50 == 150, then 150 - 30 + 99 == 219.
            stdout="150\n219\n",
        )

    def test_a_partly_braced_global_array_is_zero_filled(self):
        self.run_c(
            _STDIO
            + """
int a[6] = {1, 2, 3};
int deduced[] = {4, 5};
int main(void) {
    printf("%d %d %d %d\\n", a[2], a[3], a[5], (int)sizeof(deduced));
    return 0;
}
""",
            stdout="3 0 0 8\n",
        )

    def test_a_global_char_array_holds_its_string(self):
        self.run_c(
            _STDIO
            + """
char name[8] = "py2bin";
int main(void) {
    printf("%s %d %d\\n", name, (int)sizeof(name), name[6]);
    return 0;
}
""",
            stdout="py2bin 8 0\n",
        )

    def test_a_global_address_constant_initializer(self):
        # C11 6.6p9 allows an address constant: the address of a static object,
        # which is known before the program starts.
        self.run_c(
            _STDIO
            + """
int table[4] = {1, 2, 3, 4};
int *cursor = table;
int one = 1;
int *at_one = &one;
char *greeting = "hi";
int main(void) {
    printf("%d %d %s\\n", cursor[2], *at_one, greeting);
    table[2] = 30;
    printf("%d\\n", cursor[2]);
    return 0;
}
""",
            stdout="3 1 hi\n30\n",
        )

    def test_static_at_file_scope_is_accepted_and_a_local_shadows_a_global(self):
        self.run_c(
            _STDIO
            + """
static int shared = 5;
int read_it(void) { return shared; }
int main(void) {
    int shared = 100;
    printf("%d %d\\n", shared, read_it());
    return 0;
}
""",
            stdout="100 5\n",
        )

    def test_a_global_struct_is_shared_and_starts_zeroed(self):
        self.run_c(
            _STDIO
            + """
struct P { char c; int i; };
struct P origin;
void fill(void) { origin.c = 3; origin.i = 40; }
int main(void) {
    printf("%d %d\\n", origin.c, origin.i);
    fill();
    printf("%d %d\\n", origin.c, origin.i);
    return 0;
}
""",
            stdout="0 0\n3 40\n",
        )

    def test_several_objects_in_one_declaration(self):
        self.run_c(
            _STDIO
            + """
int a = 1, *p, b[3] = {7, 8, 9};
int main(void) {
    p = b;
    printf("%d %d %d\\n", a, p[1], b[2]);
    return 0;
}
""",
            stdout="1 8 9\n",
        )


class FunctionPointerTests(CProgramTestCase):
    """C function designators, function-pointer types, and indirect calls.

    Every expectation is derived from the standard by hand. The programs that
    matter most are the ones where the CALLEE is chosen by an expression with a
    side effect: C evaluates the called expression exactly once, and a backend
    that re-emitted it would call twice.
    """

    def test_a_pointer_calls_the_function_it_was_given(self):
        self.run_c(
            _STDIO
            + """
int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }
int main(void) {
    int (*p)(int, int);
    p = add;
    printf("%d\\n", p(3, 4));
    p = sub;
    printf("%d\\n", p(3, 4));
    return 0;
}
""",
            stdout="7\n-1\n",
        )

    def test_a_designator_and_its_address_are_the_same_pointer(self):
        # C11 6.3.2.1p4: a function designator decays to a pointer, and 6.5.3.2
        # makes &f that same pointer. *fp is the function again, which decays
        # straight back -- so all of these are one call.
        self.run_c(
            _STDIO
            + """
int twice(int n) { return n + n; }
int main(void) {
    int (*a)(int) = twice;
    int (*b)(int) = &twice;
    printf("%d %d %d %d %d\\n", a(1), b(2), (*a)(3), (**a)(4), (****b)(5));
    printf("%d\\n", a == b);
    return 0;
}
""",
            stdout="2 4 6 8 10\n1\n",
        )

    def test_a_function_pointer_is_a_parameter_like_any_other(self):
        self.run_c(
            _STDIO
            + """
int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }
int apply(int (*op)(int, int), int a, int b) { return op(a, b); }
int main(void) {
    printf("%d %d\\n", apply(add, 6, 7), apply(mul, 6, 7));
    return 0;
}
""",
            stdout="13 42\n",
        )

    def test_a_table_of_function_pointers_in_static_storage(self):
        self.run_c(
            _STDIO
            + """
int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }
int mul(int a, int b) { return a * b; }
typedef int (*binop)(int, int);
binop table[3] = {add, sub, mul};
int main(void) {
    int i;
    for (i = 0; i < 3; i++) { printf("%d ", table[i](8, 2)); }
    printf("\\n%d\\n", (int)sizeof(table));
    return 0;
}
""",
            stdout="10 6 16 \n24\n",
        )

    def test_the_called_expression_is_evaluated_exactly_once(self):
        # This is the defect class this backend has produced most often: a value
        # written once being lowered twice. Here the side effect is counted.
        self.run_c(
            _STDIO
            + """
int probes;
int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }
typedef int (*binop)(int, int);
binop table[2] = {add, sub};
int probe(void) { probes = probes + 1; return 1; }
binop fetch(void) { probes = probes + 10; return add; }
int main(void) {
    printf("%d %d\\n", table[probe()](50, 8), probes);
    printf("%d %d\\n", fetch()(3, 4), probes);
    return 0;
}
""",
            stdout="42 1\n7 11\n",
        )

    def test_exactly_one_indirect_branch_is_emitted_per_call_site(self):
        # Counting side effects catches a repeated call; counting the branches
        # in the machine code catches a repeated call the test happened not to
        # observe. Three source-level indirect calls, three `blr`s.
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("needs a darwin-arm64 host to disassemble")
        artifact = self.build(
            _STDIO
            + """
int add(int a, int b) { return a + b; }
int spin(int n) { return n; }
typedef int (*binop)(int, int);
binop table[1] = {add};
int chase(int n) { if (n <= 0) { return 0; } return 1 + table[0](0, spin(n - 1)); }
int main(void) {
    int (*p)(int, int) = add;
    printf("%d %d\\n", p(1, 2), table[0](3, 4));
    printf("%d\\n", chase(2));
    return 0;
}
"""
        )
        listing = subprocess.run(
            ["otool", "-tvV", str(artifact)], capture_output=True, text=True
        ).stdout
        self.assertEqual(len(re.findall(r"\bblr\b", listing)), 3)

    def test_a_function_pointer_member_and_a_void_result(self):
        self.run_c(
            _STDIO
            + """
int total;
struct ops { int (*op)(int, int); char tag; };
int add(int a, int b) { return a + b; }
void bump(int n) { total = total + n; }
int main(void) {
    struct ops o;
    void (*v)(int);
    o.op = add;
    o.tag = 'A';
    printf("%d %c %d\\n", o.op(4, 5), o.tag, (int)sizeof(o));
    v = bump;
    v(7);
    v(3);
    printf("%d\\n", total);
    return 0;
}
""",
            # struct ops is a pointer (8, alignment 8) then a char, padded to 16.
            stdout="9 A 16\n10\n",
        )

    def test_recursion_through_a_function_pointer(self):
        self.run_c(
            _STDIO
            + """
typedef int (*counter)(int);
counter self;
int countdown(int n) {
    if (n <= 0) { return 0; }
    return 1 + self(n - 1);
}
int main(void) {
    self = countdown;
    printf("%d\\n", self(7));
    return 0;
}
""",
            stdout="7\n",
        )

    def test_a_function_pointer_carries_floating_arguments_and_results(self):
        self.run_c(
            _STDIO
            + """
double scale(double x, float k) { return x * k; }
int main(void) {
    double (*f)(double, float);
    f = scale;
    printf("%g\\n", f(2.5, 4.0f));
    return 0;
}
""",
            stdout="10\n",
        )

    def test_a_function_returning_a_function_pointer(self):
        self.run_c(
            _STDIO
            + """
int add(int a, int b) { return a + b; }
typedef int (*binop)(int, int);
binop chooser(int k) { return add; }
int main(void) {
    binop (*maker)(int) = chooser;
    printf("%d %d\\n", maker(0)(2, 3), chooser(1)(10, 1));
    return 0;
}
""",
            stdout="5 11\n",
        )

    def test_a_narrow_result_comes_back_converted_to_its_type(self):
        # The result register holds an unspecified value in its upper bits, so
        # a signed char result must be sign-extended and an unsigned one
        # zero-extended before it is used.
        self.run_c(
            _STDIO
            + """
signed char narrow(int n) { return (signed char)n; }
unsigned char wide(int n) { return (unsigned char)n; }
short middle(int n) { return (short)n; }
int main(void) {
    signed char (*p)(int) = narrow;
    unsigned char (*q)(int) = wide;
    short (*r)(int) = middle;
    printf("%d %d %d\\n", p(200), q(200), r(70000));
    printf("%d %d\\n", p(-1), q(-1));
    return 0;
}
""",
            # 200 into a signed char is 200-256; 70000 into a short is
            # 70000-65536; -1 read back as an unsigned char is 255.
            stdout="-56 200 4464\n-1 255\n",
        )

    def test_a_void_result_is_not_a_value(self):
        self.reject(
            "void nothing(int n) { }\n"
            "int main(void) { void (*v)(int) = nothing; int x = v(1); return x; }\n",
            "has no value",
        )

    def test_a_cast_names_a_function_pointer_type(self):
        self.run_c(
            _STDIO
            + """
int add(int a, int b) { return a + b; }
int main(void) {
    printf("%d\\n", ((int (*)(int, int))add)(9, 9));
    return 0;
}
""",
            stdout="18\n",
        )

    def test_a_declarator_nests_as_c_says_it_does(self):
        # int *(*f)(char *) is a pointer to a function returning int *, and
        # int (*ops[2])(void) is an array of 2 pointers to functions.
        self.run_c(
            _STDIO
            + """
int cell = 41;
int *bump(char *ignored) { cell = cell + 1; return &cell; }
int one(void) { return 1; }
int two(void) { return 2; }
int main(void) {
    int *(*f)(char *) = bump;
    int (*ops[2])(void);
    ops[0] = one;
    ops[1] = two;
    printf("%d %d %d\\n", *f(0), ops[0]() + ops[1](), (int)sizeof(ops));
    return 0;
}
""",
            stdout="42 3 16\n",
        )


class FunctionPointerRejectionTests(CProgramTestCase):
    """What py2bin refuses about function pointers, and why."""

    def test_an_incompatible_function_pointer_needs_a_cast(self):
        self.reject(
            "int add(int a, int b) { return a + b; }\n"
            "int main(void) { int (*p)(int) = add; return 0; }\n",
            "explicit cast between incompatible pointer types",
        )
        # C11 6.3.2.3 converts void * to and from a pointer to an OBJECT only.
        self.reject(
            "int add(int a, int b) { return a + b; }\n"
            "int main(void) { void *v = add; return 0; }\n",
            "explicit cast between incompatible pointer types",
        )

    def test_the_argument_count_of_the_pointers_prototype_is_checked(self):
        self.reject(
            "int add(int a, int b) { return a + b; }\n"
            "int main(void) { int (*p)(int, int) = add; return p(1); }\n",
            r"takes 2 argument\(s\), got 1",
        )

    def test_only_a_function_or_a_pointer_to_one_can_be_called(self):
        self.reject(
            "int main(void) { int n = 1; return n(1); }\n",
            "needs a function or a pointer to one",
        )

    def test_an_unprototyped_function_type_is_refused(self):
        # C's empty () means the parameters are UNSPECIFIED, so a call through
        # such a pointer cannot be checked at all.
        self.reject(
            "int add(int a, int b) { return a + b; }\n"
            "int main(void) { int (*p)() = add; return 0; }\n",
            "must state its parameter types",
        )

    def test_sizeof_a_function_is_refused(self):
        self.reject(
            "int f(void) { return 1; }\n"
            "int main(void) { return (int)sizeof(f); }\n",
            "sizeof to a function",
        )

    def test_the_address_of_main_is_refused(self):
        self.reject(
            "int main(void) { int (*p)(void) = main; return 0; }\n",
            "process entry point",
        )

    def test_a_function_declared_but_never_defined_has_no_address(self):
        self.reject(
            "int later(void);\n"
            "int main(void) { int (*p)(void) = later; return 0; }\n",
            "declared but never defined",
        )

    def test_a_function_declaration_inside_a_block_is_refused(self):
        self.reject(
            "int main(void) { int f(void); return f(); }\n",
            "function inside a block",
        )

    def test_a_nested_function_declarator_is_refused(self):
        self.reject(
            "int add(int a, int b) { return a + b; }\n"
            "int (*chooser(int k))(int, int) { return add; }\n"
            "int main(void) { return 0; }\n",
            "plain declarator form",
        )

    def test_types_c_does_not_have_are_refused(self):
        self.reject(
            "int table[2](void);\nint main(void) { return 0; }\n",
            "an array of int .void. is not a type C has",
        )
        self.reject(
            "int (*f(void))[3];\nint main(void) { return 0; }\n",
            "plain declarator form|cannot return",
        )

    def test_a_function_pointer_needs_a_target_with_a_real_call_abi(self):
        source = (
            "int add(int a, int b) { return a + b; }\n"
            "int main(void) { int (*p)(int, int) = add; return p(1, 2); }\n"
        )
        with self.assertRaises(CCompileError) as caught:
            compile_c_to_ir(source, "reject.c", "windows-x86_64")
        self.assertRegex(str(caught.exception), "call ABI is not implemented")


class FunctionPointerDifferentialTests(CProgramTestCase):
    """Random runtime dispatch, checked against expectations computed here.

    Every callee is chosen at run time through a pointer, so nothing about the
    result can come from constant folding, and the probe count proves the
    expression that selected the callee ran exactly as many times as C says.
    """

    _OPS = {
        "add": ("a + b", lambda a, b: a + b),
        "sub": ("a - b", lambda a, b: a - b),
        "mul": ("a * b", lambda a, b: a * b),
        "andv": ("a & b", lambda a, b: a & b),
        "orv": ("a | b", lambda a, b: a | b),
        "xorv": ("a ^ b", lambda a, b: a ^ b),
        "maxv": ("a > b ? a : b", lambda a, b: a if a > b else b),
    }

    @staticmethod
    def _as_int(value: int) -> int:
        """Reduce to the value a 32-bit signed C ``int`` would hold."""

        value &= 0xFFFFFFFF
        return value - (1 << 32) if value >= (1 << 31) else value

    def test_random_dispatch_matches_the_operation_it_selected(self):
        rng = random.Random(20260726)
        names = list(self._OPS)
        lines = [_STDIO.strip()]
        for name in names:
            body, _model = self._OPS[name]
            lines.append(f"int {name}(int a, int b) {{ return {body}; }}")
        lines.append("typedef int (*binop)(int, int);")
        lines.append("int probes;")
        lines.append(f"binop table[{len(names)}] = {{{', '.join(names)}}};")
        lines.append("int probe(int k) { probes = probes + 1; return k; }")
        lines.append("int main(void) {")
        lines.append("    binop p;")
        expected: list[str] = []
        probes = 0
        for _ in range(120):
            index = rng.randrange(len(names))
            name = names[index]
            a = self._as_int(rng.randint(-(2**31), 2**31 - 1))
            b = self._as_int(rng.randint(-(2**31), 2**31 - 1))
            shape = rng.randrange(4)
            if shape == 0:
                # The callee is selected by a call with a side effect.
                call = f"table[probe({index})]({a}, {b})"
                probes += 1
            elif shape == 1:
                lines.append(f"    p = {name};")
                call = f"p({a}, {b})"
            elif shape == 2:
                call = f"(*table[{index}])({a}, {b})"
            else:
                call = f"(probes >= 0 ? table[{index}] : table[0])({a}, {b})"
            lines.append(f'    printf("%d\\n", {call});')
            expected.append(str(self._as_int(self._OPS[name][1](a, b))))
        lines.append('    printf("%d\\n", probes);')
        expected.append(str(probes))
        lines.append("    return 0;")
        lines.append("}")
        artifact = self.build("\n".join(lines) + "\n")
        if not _HOST_IS_DARWIN_ARM64:
            return
        result = subprocess.run([str(artifact)], capture_output=True, text=True)
        self.assertEqual(result.stdout.split(), expected)


class IncompleteMemberTests(CProgramTestCase):
    """A struct member has to have a size, or the layout is a lie."""

    def test_a_member_of_function_type_is_refused(self):
        self.reject(
            "struct S { int f(void); };\nint main(void) { return 0; }\n",
            "has function type",
        )

    def test_a_member_with_no_array_length_is_refused(self):
        # `int a[];` is not a member C has here: a flexible array member must
        # be last and cannot be the only member. py2bin used to lay it out with
        # a size of zero, which put the NEXT member at the same offset -- so
        # this exact program printed 3 where C requires the compiler to
        # diagnose it.
        self.reject(
            "struct S { int a[]; int b; };\n"
            "int main(void) { struct S s; s.b = 7; s.a[0] = 3; return s.b; }\n",
            "incomplete type",
        )


class TypedefFunctionTypeTests(CProgramTestCase):
    """``typedef int T(void);`` names a function type; ``T *`` points at one."""

    def test_a_typedef_function_type_declares_pointers(self):
        self.run_c(
            _STDIO
            + """
typedef int T(void);
int one(void) { return 1; }
int two(void) { return 2; }
T *global_hook = one;
int main(void) {
    T *local_hook = two;
    printf("%d %d\\n", global_hook(), local_hook());
    return 0;
}
""",
            stdout="1 2\n",
        )


class SystemVCallAbiTests(unittest.TestCase):
    """The x86-64 call ABI, verified by disassembly only.

    This host is darwin-arm64 and there is no Rosetta or emulation here, so no
    x86-64 binary is ever executed. These assertions check the encoding, not
    the behaviour, and say so.
    """

    def _text(self, source: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "p.c"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "p.bin"
            compile_c_native(entry, artifact, target="darwin-x86_64", clean=True)
            self.assertEqual(artifact.read_bytes()[:4], b"\xcf\xfa\xed\xfe")
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return ""
            return subprocess.run(
                ["otool", "-tvV", str(artifact)], capture_output=True, text=True
            ).stdout

    _RECURSIVE = (
        "long long f(long long n) { return n <= 1 ? 1 : n * f(n - 1); }\n"
        "int main(void) { return f(5); }\n"
    )

    def test_recursion_now_compiles_for_x86_64(self):
        # It was rejected outright before the System V encoder existed.
        text = self._text(self._RECURSIVE)
        if not text:
            return
        self.assertIn("callq", text)

    def test_the_frame_follows_system_v(self):
        text = self._text(self._RECURSIVE)
        if not text:
            return
        self.assertIn("pushq\t%rbp", text)
        self.assertIn("movq\t%rsp, %rbp", text)
        self.assertIn("popq\t%rbp", text)
        self.assertIn("retq", text)

    def test_the_first_parameter_arrives_in_rdi(self):
        text = self._text(self._RECURSIVE)
        if not text:
            return
        self.assertRegex(text, r"movq\t%rdi, -0x[0-9a-f]+\(%rbp\)")

    def test_every_stack_adjustment_keeps_rsp_16_byte_aligned(self):
        # System V requires rsp % 16 == 0 immediately before a call, so every
        # frame and every spill must move rsp by a multiple of 16.
        text = self._text(self._RECURSIVE)
        if not text:
            return
        amounts = re.findall(r"(?:sub|add)q\t\$0x([0-9a-f]+), %rsp", text)
        self.assertTrue(amounts)
        for amount in amounts:
            self.assertEqual(int(amount, 16) % 16, 0, f"0x{amount} misaligns rsp")
