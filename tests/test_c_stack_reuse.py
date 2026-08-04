"""A function's frame is sized by what it needs at once, not by its length.

Every intermediate the C lowering needed used to take a stack slot and never
give it back, so the frame grew with the number of statements in a function
rather than with how much of it was live. A single `def` of about eighteen
hundred statements reached the 512 KB budget and the build was refused - which
is a limit generated code walks into without doing anything unusual.

Slots taken for a statement's temporaries are handed back when it finishes.
The two ways that goes wrong both have a test here: a frame built from the
count that is left rather than from the high-water mark, and a slot handed
back that something still holds.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from py2bin.c_native import compile_c_native
from py2bin.capi_emit import python_to_capi_c

_HOST_IS_DARWIN_ARM64 = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)


def _build_and_run(source: str) -> str:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        entry = root / "program.c"
        entry.write_text(source, encoding="utf-8")
        binary = root / "program.bin"
        compile_c_native(entry, binary, target="darwin-arm64", clean=True)
        if not _HOST_IS_DARWIN_ARM64:
            return ""
        done = subprocess.run([str(binary)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-400:]
        return done.stdout


def _long_python_function(statements: int) -> tuple[str, str]:
    """A Python `def` of many statements, and what running it should print.

    Written in Python rather than C on purpose. A C statement of this shape
    costs a handful of slots, so even twenty-four thousand of them stayed
    inside the old budget - the first version of this test passed against the
    unfixed compiler, which is the only thing worse than no test. One Python
    statement becomes a dozen lines of C driving the interpreter and cost
    around thirty-six slots, which is why 1,800 of them was the ceiling.
    """

    lines = ["def main():", "    t = 1"]
    for n in range(statements):
        lines.append(f"    t = (t * {n % 7 + 2} + {n % 13}) % 1000003")
    lines += ["    print(t)", "main()"]
    t = 1
    for n in range(statements):
        t = (t * (n % 7 + 2) + n % 13) % 1000003
    return "\n".join(lines) + "\n", f"{t}\n"


@pytest.mark.parametrize("statements", [2500, 9000])
def test_a_function_longer_than_the_old_ceiling_compiles(statements):
    # 1,800 was the old refusal point; both of these are past it. The answer
    # is checked rather than only the exit status, because a frame sized from
    # the slots still outstanding instead of from the high-water mark would
    # build happily and then write through offsets it does not own.
    program, expected = _long_python_function(statements)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        entry = root / "program.py"
        entry.write_text(program, encoding="utf-8")
        generated = root / "program.c"
        generated.write_text(
            python_to_capi_c(program, str(entry)), encoding="utf-8"
        )
        if not _HOST_IS_DARWIN_ARM64:
            return
        binary = root / "program.bin"
        compile_c_native(generated, binary, target="darwin-arm64", clean=True)
        done = subprocess.run([str(binary)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-400:]
        assert done.stdout == expected


def test_the_float_formatter_keeps_its_scratch_across_statements():
    # The scratch the float formatter works in is taken by whichever statement
    # first prints a double and read by every statement after it. Handing it
    # back with that statement's temporaries put later output through slots
    # something else had taken - it did not print wrong numbers, it exhausted
    # the heap. Two prints separated by other work is the shape that catches
    # it; one print cannot.
    source = (
        "#include <stdio.h>\n"
        "int main(void) {\n"
        "    double t = 1.0;\n"
        "    long long n = 0;\n"
        "    printf(\"%.4f\\n\", t);\n"
        "    for (int i = 0; i < 40; i++) { t = t * 1.5 + 0.25; n += i; }\n"
        "    printf(\"%.4f %lld\\n\", t, n);\n"
        "    for (int i = 0; i < 10; i++) { t = t / 2.0; }\n"
        "    printf(\"%.4f\\n\", t);\n"
        "    return 0;\n"
        "}\n"
    )
    answer = _build_and_run(source)
    if _HOST_IS_DARWIN_ARM64:
        assert answer.splitlines()[0] == "1.0000"
        assert len(answer.splitlines()) == 3


def test_a_loops_condition_survives_its_own_body():
    # The reclaiming rule is per statement, which is what makes it safe here.
    # A `while` evaluates its condition into slots taken before the body is
    # lowered; the body's statements mark above those, so they cannot take
    # them. If they could, the loop would come back to a condition holding
    # whatever the body left there.
    source = (
        "#include <stdio.h>\n"
        "int main(void) {\n"
        "    long long i = 0, t = 0;\n"
        "    while (i * 2 + 1 < 201) {\n"
        "        long long a = i % 5, b = i % 7;\n"
        "        t += (a + 1) * (b + 2) % 97;\n"
        "        i++;\n"
        "    }\n"
        '    printf("%lld %lld\\n", i, t);\n'
        "    return 0;\n"
        "}\n"
    )
    answer = _build_and_run(source)
    if _HOST_IS_DARWIN_ARM64:
        i = t = 0
        while i * 2 + 1 < 201:
            t += (i % 5 + 1) * (i % 7 + 2) % 97
            i += 1
        assert answer == f"{i} {t}\n"


def test_a_local_declared_mid_function_is_not_reclaimed():
    # Declarations raise the floor reclaiming stops at. Without that, a local
    # declared halfway down would be handed back with the temporaries of the
    # statement that declared it, and read afterwards as whatever landed
    # there next.
    lines = ["#include <stdio.h>", "int main(void) {", "    long long t = 0;"]
    for n in range(40):
        lines.append(f"    long long keep_{n} = {n} * 3 + 1;")
        lines.append(f"    t = (t + keep_{n} * {n % 5 + 1}) % 99991;")
    lines.append("    long long sum = 0;")
    for n in range(40):
        lines.append(f"    sum += keep_{n};")
    lines += ['    printf("%lld %lld\\n", t, sum);', "    return 0;", "}"]
    answer = _build_and_run("\n".join(lines) + "\n")
    if _HOST_IS_DARWIN_ARM64:
        t = 0
        for n in range(40):
            t = (t + (n * 3 + 1) * (n % 5 + 1)) % 99991
        assert answer == f"{t} {sum(n * 3 + 1 for n in range(40))}\n"
