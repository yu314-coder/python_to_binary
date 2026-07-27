import ast
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest

from py2bin.c_native import (
    CNativeCompileError,
    c_to_python_source,
    compile_c_native,
    compile_python_via_c,
    parse_canonical_c,
)
from py2bin.cli import main
from py2bin.native import supported_targets


class CanonicalCNativeTests(unittest.TestCase):
    def test_python_to_c_to_every_machine_code_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sum_to.py"
            source.write_text(
                "def sum_to(limit: int) -> int:\n"
                "    total = 0\n"
                "    for value in range(limit + 1):\n"
                "        total += value\n"
                "    return total\n"
                "raise SystemExit(sum_to(9))\n",
                encoding="utf-8",
            )
            results = {}
            for target in supported_targets():
                extension = ".exe" if target.startswith("windows-") else ".bin"
                bridge = compile_python_via_c(
                    source,
                    root / f"sum_to-{target}{extension}",
                    target=target,
                )
                results[target] = bridge.native
                self.assertIn("long long sum_to", bridge.c_source)
                self.assertIn("raise SystemExit(sum_to(9))", bridge.reconstructed_python)
            self.assertEqual(set(results), set(supported_targets()))
            self.assertEqual(results["windows-x86_64"].artifact.read_bytes()[:2], b"MZ")
            self.assertEqual(results["windows-arm64"].artifact.read_bytes()[:2], b"MZ")
            self.assertEqual(results["linux-x86_64"].artifact.read_bytes()[:4], b"\x7fELF")
            self.assertEqual(results["linux-arm64"].artifact.read_bytes()[:4], b"\x7fELF")
            self.assertEqual(results["darwin-x86_64"].artifact.read_bytes()[:4], b"\xcf\xfa\xed\xfe")
            self.assertEqual(results["darwin-arm64"].artifact.read_bytes()[:4], b"\xcf\xfa\xed\xfe")
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                run = subprocess.run([str(results["darwin-arm64"].artifact)])
                self.assertEqual(run.returncode, 45)

    def test_direct_c_is_parsed_and_compiled_without_a_toolchain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "answer.c"
            source.write_text(
                "long long twice(long long value) {\n"
                "    return value * 2;\n"
                "}\n"
                "int main(void) {\n"
                "    long long answer;\n"
                "    answer = twice(21);\n"
                "    return answer;\n"
                "}\n",
                encoding="utf-8",
            )
            result = compile_c_native(
                source,
                root / "answer",
                target="darwin-arm64",
            )
            self.assertGreater(result.operations, 0)
            artifact = result.artifact.read_bytes()
            self.assertEqual(artifact[:4], b"\xcf\xfa\xed\xfe")
            self.assertNotIn(b"python", artifact.lower())
            self.assertNotIn(b"cpython", artifact.lower())
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                run = subprocess.run([str(result.artifact)])
                self.assertEqual(run.returncode, 42)

    def test_generated_literal_printf_round_trips(self):
        c_source = (
            "#include <stdio.h>\n"
            "int main(void) {\n"
            '    printf("100%% native\\n");\n'
            "    return 0;\n"
            "}\n"
        )
        reconstructed = c_to_python_source(c_source, "literal.c")
        self.assertIn("print('100% native')", reconstructed)

    def test_general_c_types_and_pointers_are_rejected(self):
        for source in (
            "double helper(void) { return 1; }\nint main(void) { return 0; }\n",
            "long long helper(long long *value) { return 1; }\n"
            "int main(void) { return 0; }\n",
        ):
            with self.subTest(source=source):
                with self.assertRaises(CNativeCompileError):
                    c_to_python_source(source, "general.c")

    def test_division_and_noninteger_printf_are_rejected(self):
        with self.assertRaisesRegex(
            CNativeCompileError,
            "division and modulo are not implemented",
        ):
            c_to_python_source(
                "int main(void) { return 8 / 2; }\n",
                "divide.c",
            )
        with self.assertRaisesRegex(
            CNativeCompileError,
            "only %% and compile-time integer %lld",
        ):
            c_to_python_source(
                '#include <stdio.h>\nint main(void) { printf("%g\\n", 1); return 0; }\n',
                "format.c",
            )

    def test_generated_compile_time_integer_printf_round_trips(self):
        c_source = (
            "#include <stdio.h>\n"
            "int main(void) {\n"
            "    long long answer;\n"
            "    answer = 6 * 7;\n"
            '    printf("answer: %lld\\n", answer);\n'
            "    return 0;\n"
            "}\n"
        )
        reconstructed = c_to_python_source(c_source, "integer-format.c")
        self.assertIn("f'answer: {answer}'", reconstructed)

    def test_noncanonical_for_loop_is_rejected(self):
        with self.assertRaisesRegex(CNativeCompileError, "canonical signed-step"):
            c_to_python_source(
                "int main(void) {\n"
                "    long long i;\n"
                "    for (i = 0; i < 3; i += 1) { ; }\n"
                "    return 0;\n"
                "}\n",
                "loop.c",
            )

    def test_cli_retains_the_c_it_really_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "status.py"
            source.write_text("raise SystemExit(7)\n", encoding="utf-8")
            c_output = root / "status.c"
            executable = root / "status.exe"
            status = main(
                [
                    "compile-via-c",
                    str(source),
                    "--target",
                    "windows-x86_64",
                    "--c-output",
                    str(c_output),
                    "-o",
                    str(executable),
                ]
            )
            self.assertEqual(status, 0)
            self.assertTrue(c_output.is_file())
            self.assertIn("return 7;", c_output.read_text(encoding="utf-8"))
            self.assertEqual(executable.read_bytes()[:2], b"MZ")


if __name__ == "__main__":
    unittest.main()


class CApiDialectTests(unittest.TestCase):
    """The generated C dialect that drives an embedded CPython.

    Because py2bin emits this C itself, it declares every external symbol with
    an explicit prototype instead of including <Python.h>. That is what removes
    the need for a C preprocessor and for parsing CPython's macro-heavy
    headers, while still producing real dyld-bound calls.
    """

    _SOURCE = (
        "extern void Py_Initialize(void);\n"
        "extern int PyRun_SimpleString(char *source);\n"
        "extern void Py_Finalize(void);\n"
        "\n"
        "int main(void)\n"
        "{\n"
        "    Py_Initialize();\n"
        "    PyRun_SimpleString(\"print('embedded')\");\n"
        "    Py_Finalize();\n"
        "    return 0;\n"
        "}\n"
    )

    def test_extern_prototypes_become_adapter_abi_imports(self):
        tree = parse_canonical_c(self._SOURCE, "capi.c")
        imports = [
            node for node in tree.body if isinstance(node, ast.ImportFrom)
        ]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "py2bin.cabi")
        self.assertEqual(
            sorted(alias.name for alias in imports[0].names),
            ["PyRun_SimpleString", "Py_Finalize", "Py_Initialize"],
        )

    def test_unvetted_external_symbol_is_rejected(self):
        with self.assertRaises(CNativeCompileError) as caught:
            parse_canonical_c(
                "extern int system(char *command);\nint main(void) { return 0; }\n",
                "bad.c",
            )
        self.assertIn("vetted adapter ABI", str(caught.exception))

    def test_dereferenceable_pointers_are_still_rejected(self):
        # Only opaque handles may be pointers; a long long * would need loads
        # and stores through an arbitrary address, which this backend has not
        # implemented.
        with self.assertRaises(CNativeCompileError):
            parse_canonical_c(
                "long long helper(long long *value) { return 1; }\n"
                "int main(void) { return 0; }\n",
                "bad.c",
            )

    def test_embedded_cpython_binary_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "capi.c"
            source.write_text(self._SOURCE, encoding="utf-8")
            artifact = root / "capi.bin"
            compile_c_native(source, artifact, target="darwin-arm64", clean=True)
            self.assertEqual(artifact.read_bytes()[:4], b"\xcf\xfa\xed\xfe")
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            run = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn(b"embedded", run.stdout)


class CForLoopSemanticsTests(unittest.TestCase):
    """A C `for` is not a Python `for`, and must not share its lowering.

    C evaluates the initializer even when the body never runs, leaves the
    counter at the first value that fails the test, and lets the body affect
    iteration. `for x in range(...)` does none of those.
    """

    def _exit(self, body: str) -> int:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "loop.c"
            source.write_text(body, encoding="utf-8")
            artifact = root / "loop.bin"
            compile_c_native(source, artifact, target="darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return -1
            return subprocess.run([str(artifact)], capture_output=True).returncode

    def test_counter_stops_at_the_first_failing_value(self):
        result = self._exit(
            "int main(void)\n{\n"
            "    long long i;\n"
            "    for (i = 0; (1) >= 0 ? i < 5 : i > 5; i += 1) { }\n"
            "    return i;\n}\n"
        )
        if result >= 0:
            self.assertEqual(result, 5)  # C: 5, Python's range would give 4

    def test_initializer_runs_even_for_an_empty_range(self):
        result = self._exit(
            "int main(void)\n{\n"
            "    long long j = 9;\n"
            "    for (j = 0; (1) >= 0 ? j < 0 : j > 0; j += 1) { }\n"
            "    return j;\n}\n"
        )
        if result >= 0:
            self.assertEqual(result, 0)  # C: 0, Python would leave 9

    def test_body_may_advance_the_counter(self):
        result = self._exit(
            "int main(void)\n{\n"
            "    long long i;\n    long long trips = 0;\n"
            "    for (i = 0; (1) >= 0 ? i < 10 : i > 10; i += 1)"
            " { trips += 1; i += 1; }\n"
            "    return trips;\n}\n"
        )
        if result >= 0:
            self.assertEqual(result, 5)  # C: 5, Python's range would give 10

    def test_continue_still_advances_the_counter(self):
        result = self._exit(
            "int main(void)\n{\n"
            "    long long i;\n    long long n = 0;\n"
            "    for (i = 0; (1) >= 0 ? i < 6 : i > 6; i += 1)"
            " { if (i == 2) { continue; } n += 1; }\n"
            "    return n * 10 + i;\n}\n"
        )
        if result >= 0:
            self.assertEqual(result, 56)


class PythonViaCTests(unittest.TestCase):
    """Python -> generated C -> py2bin's C compiler -> machine code.

    The generated C now goes through the real C front end, so this path
    inherits what that compiler can do rather than the narrow shape the
    canonical-C bridge was written for. Each program is run natively and
    compared against CPython running the same source.
    """

    def _matches_cpython(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "p.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "p.bin"
            compile_python_via_c(
                entry, artifact, target="darwin-arm64", clean=True
            )
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            native = subprocess.run([str(artifact)], capture_output=True)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            self.assertEqual(native.returncode, reference.returncode)
            self.assertEqual(native.stdout, reference.stdout)

    def test_integer_division_and_modulo(self):
        self._matches_cpython(
            "a = 0\n"
            "for i in range(1, 20):\n"
            "    a += (i * 7) // 3 + (i * 7) % 3\n"
            "raise SystemExit(a % 251)\n"
        )

    def test_floating_arithmetic(self):
        self._matches_cpython(
            "x = 0.0\n"
            "for i in range(1, 11):\n"
            "    x = x + 1.0 / float(i)\n"
            "raise SystemExit(int(x * 100.0))\n"
        )

    def test_a_function_with_a_loop(self):
        self._matches_cpython(
            "def gcd(a: int, b: int) -> int:\n"
            "    while b != 0:\n"
            "        t = b\n"
            "        b = a % b\n"
            "        a = t\n"
            "    return a\n"
            "raise SystemExit(gcd(1071, 462))\n"
        )

    def test_boolean_operators_and_elif(self):
        self._matches_cpython(
            "n = 0\n"
            "for i in range(1, 30):\n"
            "    if i % 3 == 0 and i % 5 == 0:\n"
            "        n += 10\n"
            "    elif i % 3 == 0 or i % 5 == 0:\n"
            "        n += 1\n"
            "raise SystemExit(n)\n"
        )
