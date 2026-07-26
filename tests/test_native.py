from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest

import struct

from py2bin.native import NativeCompileError, compile_all, compile_native, resolve_target
from py2bin.native.arm64 import encode_darwin as encode_darwin_arm64
from py2bin.native.compiler import compile_native_module
from py2bin.native.ir import IntConstant, Module
from py2bin.cli import main


class NativeCompilerTests(unittest.TestCase):
    def test_resolves_os_and_architecture_aliases(self):
        self.assertEqual(resolve_target("macos", "aarch64"), "darwin-arm64")
        self.assertEqual(resolve_target("windows", "x64"), "windows-x86_64")
        self.assertEqual(resolve_target("linux", "arm64"), "linux-arm64")

    def test_compile_all_cross_targets_without_toolchain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello.py"
            source.write_text("print('everywhere')\n", encoding="utf-8")
            results = compile_all(source, root / "dist")
            self.assertEqual(len(results), 6)
            by_target = {result.target: result.artifact for result in results}
            self.assertEqual(by_target["windows-x86_64"].read_bytes()[:2], b"MZ")
            self.assertEqual(by_target["windows-arm64"].read_bytes()[:2], b"MZ")
            self.assertEqual(by_target["linux-arm64"].read_bytes()[:4], b"\x7fELF")
            self.assertEqual(by_target["darwin-arm64"].read_bytes()[:4], b"\xcf\xfa\xed\xfe")

    def test_runtime_integer_loops_cross_compile_for_every_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sum.py"
            source.write_text(
                "total = 0\n"
                "for value in range(1, 11):\n"
                "    total += value\n"
                "raise SystemExit(total)\n",
                encoding="utf-8",
            )
            results = compile_all(source, root / "dist")
            self.assertEqual({result.target for result in results}, {
                "linux-x86_64",
                "linux-arm64",
                "darwin-x86_64",
                "darwin-arm64",
                "windows-x86_64",
                "windows-arm64",
            })
            self.assertTrue(all(result.operations > 2 for result in results))

    def test_imported_pure_integer_functions_inline_for_every_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "helper.py"
            helper.write_text(
                "BIAS = 3\n"
                "def twice(value: int) -> int:\n"
                "    doubled = value * 2\n"
                "    return doubled\n"
                "def affine(value: int) -> int:\n"
                "    adjusted = twice(value) + BIAS\n"
                "    if adjusted < 9:\n"
                "        return adjusted + 1\n"
                "    return adjusted - 1\n",
                encoding="utf-8",
            )
            source = root / "main.py"
            source.write_text(
                "from helper import affine as transform\n"
                "total = 0\n"
                "for value in range(1, 5):\n"
                "    total += transform(value)\n"
                "raise SystemExit(total)\n",
                encoding="utf-8",
            )
            results = compile_all(
                source,
                root / "dist",
                source_roots=(root,),
            )
            manual = root / "manual.py"
            manual.write_text(
                "total = 0\n"
                "for value in range(1, 5):\n"
                "    total += (value * 2 + 3 + 1 "
                "if value * 2 + 3 < 9 else value * 2 + 3 - 1)\n"
                "raise SystemExit(total)\n",
                encoding="utf-8",
            )
            manual_results = compile_all(manual, root / "manual-dist")
            self.assertEqual(len(results), 6)
            self.assertTrue(all(result.operations > 2 for result in results))
            by_target = {result.target: result.artifact for result in results}
            manual_by_target = {
                result.target: result.artifact for result in manual_results
            }
            for target, artifact in by_target.items():
                self.assertEqual(
                    artifact.read_bytes(),
                    manual_by_target[target].read_bytes(),
                    f"{target} retained function-call overhead",
                )
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                run = subprocess.run([str(by_target["darwin-arm64"])])
                self.assertEqual(run.returncode, 32)

    def test_same_module_pure_integer_function_is_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "square.py"
            source.write_text(
                "def square(value: int) -> int:\n"
                "    result = value * value\n"
                "    return result\n"
                "answer = square(7)\n"
                "raise SystemExit(answer)\n",
                encoding="utf-8",
            )
            result = compile_native(source, root / "square", "darwin-arm64")
            self.assertGreater(result.operations, 1)
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                run = subprocess.run([str(result.artifact)])
                self.assertEqual(run.returncode, 49)

    def test_native_void_procedure_with_bare_return_cross_compiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "procedure.py"
            source.write_text(
                "def announce(flag: int, repeat: int = 2) -> None:\n"
                "    for index in range(repeat):\n"
                "        if flag:\n"
                "            print('on')\n"
                "        else:\n"
                "            print('off')\n"
                "    if not flag:\n"
                "        return\n"
                "    print('done')\n"
                "announce(1)\n"
                "raise SystemExit(7)\n",
                encoding="utf-8",
            )

            results = compile_all(source, root / "dist")

            self.assertEqual(len(results), 6)
            self.assertTrue(all(result.operations > 2 for result in results))
            by_target = {result.target: result.artifact for result in results}
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                run = subprocess.run(
                    [str(by_target["darwin-arm64"])],
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(run.returncode, 7)
                self.assertEqual(run.stdout, b"on\non\ndone\n")

    def test_static_complex_annotations_are_erased_without_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "annotations.py"
            source.write_text(
                "def notify(value: list[str] | None) -> None:\n"
                "    print('notified')\n"
                "notify(3)\n",
                encoding="utf-8",
            )

            result = compile_native(
                source,
                root / "annotations",
                "darwin-arm64",
            )

            self.assertGreater(result.operations, 1)

    def test_native_procedure_cannot_be_used_as_integer_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad_procedure.py"
            source.write_text(
                "def notify() -> None:\n"
                "    print('called')\n"
                "answer = notify()\n"
                "raise SystemExit(answer)\n",
                encoding="utf-8",
            )
            output = root / "bad"

            with self.assertRaisesRegex(
                NativeCompileError,
                "procedure notify.*does not produce a value",
            ):
                compile_native(source, output, "darwin-arm64")
            self.assertFalse(output.exists())

    def test_native_function_boolean_logic_and_chained_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "logic.py"
            source.write_text(
                "def classify(value: int) -> int:\n"
                "    inside = 2 <= value < 5 and value != 3\n"
                "    return 10 if inside or value == 7 else 1\n"
                "total = 0\n"
                "for value in range(1, 8):\n"
                "    total += classify(value)\n"
                "raise SystemExit(total)\n",
                encoding="utf-8",
            )
            results = compile_all(source, root / "logic-dist")
            self.assertEqual(len(results), 6)
            by_target = {result.target: result.artifact for result in results}
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                run = subprocess.run([str(by_target["darwin-arm64"])])
                self.assertEqual(run.returncode, 34)

    def test_relative_local_function_import_is_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "helper.py").write_text(
                "def triple(value: int) -> int:\n"
                "    return value * 3\n",
                encoding="utf-8",
            )
            source = package / "main.py"
            source.write_text(
                "from .helper import triple\n"
                "raise SystemExit(triple(8))\n",
                encoding="utf-8",
            )
            result = compile_native(
                source,
                root / "relative",
                "darwin-arm64",
                source_roots=(root,),
            )
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                run = subprocess.run([str(result.artifact)])
                self.assertEqual(run.returncode, 24)

    def test_recursive_native_function_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recursive.py"
            source.write_text(
                "def recurse(value: int) -> int:\n"
                "    return recurse(value)\n"
                "raise SystemExit(recurse(1))\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                NativeCompileError,
                "recursive native function",
            ):
                compile_native(source, root / "recursive", "linux-x86_64")

    def test_executable_function_annotation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "annotation.py"
            source.write_text(
                "def unsafe(value: print('annotation side effect')) -> int:\n"
                "    return value\n"
                "raise SystemExit(unsafe(1))\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                NativeCompileError,
                "annotation has runtime behavior",
            ):
                compile_native(source, root / "annotation", "linux-x86_64")

    def test_large_straight_line_function_uses_imperative_ir_without_ast_blowup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "oversized.py"
            assignments = "".join(
                "    value = value + value\n"
                for _ in range(20)
            )
            source.write_text(
                "def expand(value: int) -> int:\n"
                f"{assignments}"
                "    return value\n"
                "raise SystemExit(expand(1))\n",
                encoding="utf-8",
            )
            result = compile_native(
                source,
                root / "oversized",
                "linux-x86_64",
            )
            self.assertGreater(result.operations, 20)

    def test_function_loops_early_return_and_mutable_branches_are_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "imperative.py"
            source.write_text(
                "def accumulate(limit: int) -> int:\n"
                "    total = 0\n"
                "    for value in range(limit):\n"
                "        if value == 4:\n"
                "            continue\n"
                "        total += value\n"
                "        if total >= 12:\n"
                "            return total\n"
                "    return total\n"
                "raise SystemExit(accumulate(10))\n",
                encoding="utf-8",
            )
            results = compile_all(source, root / "imperative-dist")
            self.assertEqual(len(results), 6)
            self.assertTrue(all(result.operations > 10 for result in results))
            by_target = {result.target: result.artifact for result in results}
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                run = subprocess.run([str(by_target["darwin-arm64"])])
                self.assertEqual(run.returncode, 17)

    def test_function_integer_defaults_and_named_arguments_are_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "defaults.py"
            source.write_text(
                "def weighted(value: int, scale: int = 2, bias: int = 3) -> int:\n"
                "    return value * scale + bias\n"
                "def aggregate(limit: int, scale: int = 2) -> int:\n"
                "    total = 0\n"
                "    for value in range(limit):\n"
                "        total += weighted(value, bias=1, scale=scale)\n"
                "    return total\n"
                "raise SystemExit(aggregate(4, scale=3))\n",
                encoding="utf-8",
            )
            results = compile_all(source, root / "default-dist")
            self.assertEqual(len(results), 6)
            by_target = {result.target: result.artifact for result in results}
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                run = subprocess.run([str(by_target["darwin-arm64"])])
                self.assertEqual(run.returncode, 22)

    def test_function_non_integer_default_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad_default.py"
            source.write_text(
                "def choose(value: int = None) -> int:\n"
                "    return value\n"
                "raise SystemExit(choose())\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                NativeCompileError,
                "defaults must be compile-time int/bool",
            ):
                compile_native(source, root / "bad_default", "linux-x86_64")

    def test_nested_local_modules_inline_imperative_functions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "numbers"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "core.py").write_text(
                "def sum_to(limit: int) -> int:\n"
                "    total = 0\n"
                "    for value in range(limit + 1):\n"
                "        total += value\n"
                "    return total\n",
                encoding="utf-8",
            )
            (package / "facade.py").write_text(
                "from .core import sum_to\n"
                "def answer(value: int) -> int:\n"
                "    return sum_to(value)\n",
                encoding="utf-8",
            )
            source = root / "main.py"
            source.write_text(
                "from numbers.facade import answer\n"
                "raise SystemExit(answer(9))\n",
                encoding="utf-8",
            )
            result = compile_native(
                source,
                root / "nested",
                "darwin-arm64",
                source_roots=(root,),
            )
            self.assertGreater(result.operations, 8)
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                run = subprocess.run([str(result.artifact)])
                self.assertEqual(run.returncode, 45)

    def test_imported_function_module_with_side_effect_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "helper.py").write_text(
                "print('module side effect')\n"
                "def answer() -> int:\n"
                "    return 42\n",
                encoding="utf-8",
            )
            source = root / "main.py"
            source.write_text(
                "from helper import answer\n"
                "raise SystemExit(answer())\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                NativeCompileError,
                "executable top-level",
            ):
                compile_native(
                    source,
                    root / "main",
                    "linux-x86_64",
                    source_roots=(root,),
                )

    def test_native_cli_dispatches_without_bundle_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello.py"
            source.write_text("print('cli')\n", encoding="utf-8")
            status = main(
                ["compile", str(source), "--output", str(root / "hello"), "--target", "linux-x86_64"]
            )
            self.assertEqual(status, 0)
            self.assertTrue((root / "hello").exists())

    def test_native_cli_uses_source_root_for_local_function(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "helper.py").write_text(
                "def answer() -> int:\n"
                "    return 42\n",
                encoding="utf-8",
            )
            source = root / "main.py"
            source.write_text(
                "from helper import answer\n"
                "raise SystemExit(answer())\n",
                encoding="utf-8",
            )
            output = root / "main.exe"
            status = main(
                [
                    "compile",
                    str(source),
                    "--source-root",
                    str(root),
                    "--output",
                    str(output),
                    "--target",
                    "windows-x86_64",
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(output.read_bytes()[:2], b"MZ")

    def test_writes_elf_without_assembler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello.py"
            source.write_text("print('native', 6 * 7)\n", encoding="utf-8")
            result = compile_native(source, root / "hello", "linux-x86_64")
            self.assertEqual(result.artifact.read_bytes()[:4], b"\x7fELF")
            self.assertEqual(result.operations, 2)

    def test_writes_macho_without_assembler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello.py"
            source.write_text("name = 'native'\nprint(f'hello {name}')\n", encoding="utf-8")
            result = compile_native(source, root / "hello", "darwin-x86_64")
            self.assertEqual(result.artifact.read_bytes()[:4], b"\xcf\xfa\xed\xfe")

    def test_writes_windows_exe_without_assembler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello.py"
            source.write_text("print('hello windows')\n", encoding="utf-8")
            result = compile_native(source, root / "hello.exe", "windows-x86_64")
            image = result.artifact.read_bytes()
            self.assertEqual(image[:2], b"MZ")
            pe_offset = int.from_bytes(image[0x3C:0x40], "little")
            self.assertEqual(image[pe_offset:pe_offset + 4], b"PE\0\0")

    def test_writes_windows_arm64_exe_without_assembler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello.py"
            source.write_text("print('hello windows arm64')\n", encoding="utf-8")
            result = compile_native(source, root / "hello.exe", "windows-arm64")
            image = result.artifact.read_bytes()
            pe_offset = int.from_bytes(image[0x3C:0x40], "little")
            self.assertEqual(image[pe_offset:pe_offset + 4], b"PE\0\0")
            self.assertEqual(int.from_bytes(image[pe_offset + 4:pe_offset + 6], "little"), 0xAA64)

    def test_cli_selects_os_and_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello.py"
            source.write_text("print('selected')\n", encoding="utf-8")
            output = root / "selected.exe"
            status = main(
                [
                    "compile",
                    str(source),
                    "--output",
                    str(output),
                    "--os",
                    "windows",
                    "--arch",
                    "arm64",
                ]
            )
            self.assertEqual(status, 0)
            image = output.read_bytes()
            pe_offset = int.from_bytes(image[0x3C:0x40], "little")
            self.assertEqual(int.from_bytes(image[pe_offset + 4:pe_offset + 6], "little"), 0xAA64)

    def test_writes_linux_arm64_without_assembler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello.py"
            source.write_text("print('hello arm64')\n", encoding="utf-8")
            result = compile_native(source, root / "hello", "linux-arm64")
            image = result.artifact.read_bytes()
            self.assertEqual(image[:4], b"\x7fELF")
            self.assertEqual(int.from_bytes(image[18:20], "little"), 183)

    def test_writes_macho_arm64_without_assembler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello.py"
            source.write_text("print('hello mac arm64')\n", encoding="utf-8")
            result = compile_native(source, root / "hello", "darwin-arm64")
            image = result.artifact.read_bytes()
            self.assertEqual(image[:4], b"\xcf\xfa\xed\xfe")
            self.assertEqual(int.from_bytes(image[4:8], "little"), 0x0100000C)
            self.assertIn((0x80000022).to_bytes(4, "little"), image[:0x4000])

    def test_writes_native_macos_app(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello.py"
            source.write_text("print('app')\n", encoding="utf-8")
            result = compile_native(source, root / "Hello", "darwin-arm64", app=True)
            self.assertEqual(result.artifact.suffix, ".app")
            executable = result.artifact / "Contents" / "MacOS" / "hello"
            self.assertEqual(executable.read_bytes()[:4], b"\xcf\xfa\xed\xfe")
            self.assertTrue((result.artifact / "Contents" / "Info.plist").exists())

    def test_rejects_dynamic_python_instead_of_miscompiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.py"
            source.write_text("import torch\n", encoding="utf-8")
            with self.assertRaisesRegex(NativeCompileError, "native subset"):
                compile_native(source, root / "bad", "linux-x86_64")

    @unittest.skipUnless(
        platform.system() == "Darwin" and platform.machine() == "x86_64",
        "x86-64 Mach-O execution requires a native x86-64 Mac",
    )
    def test_generated_macho_runs_without_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello.py"
            source.write_text("print('hardwritten-ok')\n", encoding="utf-8")
            result = compile_native(source, root / "hello", "darwin-x86_64")
            run = subprocess.run([str(result.artifact)], capture_output=True, text=True, check=True)
            self.assertEqual(run.stdout.strip(), "hardwritten-ok")

    @unittest.skipUnless(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "native arm64 Mach-O requires Apple Silicon",
    )
    def test_generated_signed_arm64_macho_runs_without_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello.py"
            source.write_text("print('arm64-hardwritten-ok')\n", encoding="utf-8")
            result = compile_native(source, root / "hello", "darwin-arm64")
            run = subprocess.run([str(result.artifact)], capture_output=True, text=True, check=True)
            self.assertEqual(run.stdout.strip(), "arm64-hardwritten-ok")

    @unittest.skipUnless(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "native arm64 Mach-O requires Apple Silicon",
    )
    def test_runtime_integer_control_flow_runs_as_machine_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "runtime_sum.py"
            source.write_text(
                "total = 0\n"
                "value = 1\n"
                "while value <= 10:\n"
                "    total += value\n"
                "    value += 1\n"
                "if total == 55:\n"
                "    raise SystemExit(total)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            result = compile_native(source, root / "runtime_sum", "darwin-arm64")
            run = subprocess.run([str(result.artifact)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 55, run.stderr)


if __name__ == "__main__":
    unittest.main()


class ForLoopVariableSemanticsTests(unittest.TestCase):
    """`for x in range(...)` must leave x exactly as CPython leaves it.

    Advancing the user's variable directly would leave it holding `stop` after
    the loop, and would clobber it even when the range is empty.
    """

    def _exit(self, source: str) -> tuple[int, int]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "loop.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "loop.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            ).returncode
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return reference, reference
            native = subprocess.run([str(artifact)], capture_output=True).returncode
            return reference, native

    def test_counter_keeps_its_last_value(self):
        reference, native = self._exit(
            "for i in range(1, 5):\n    pass\nraise SystemExit(i)\n"
        )
        self.assertEqual(native, reference)
        self.assertEqual(native, 4)

    def test_empty_range_leaves_the_variable_untouched(self):
        reference, native = self._exit(
            "i = 99\nfor i in range(3, 3):\n    pass\nraise SystemExit(i)\n"
        )
        self.assertEqual(native, reference)
        self.assertEqual(native, 99)

    def test_step_and_negative_step_keep_the_last_value(self):
        for source, expected in (
            ("for i in range(0, 10, 3):\n    pass\nraise SystemExit(i)\n", 9),
            ("for i in range(10, 0, -2):\n    pass\nraise SystemExit(i)\n", 2),
        ):
            reference, native = self._exit(source)
            self.assertEqual(native, reference, source)
            self.assertEqual(native, expected, source)

    def test_break_keeps_the_value_at_the_break(self):
        reference, native = self._exit(
            "for i in range(0, 10):\n"
            "    if i == 4:\n"
            "        break\n"
            "raise SystemExit(i)\n"
        )
        self.assertEqual(native, reference)
        self.assertEqual(native, 4)


class UnboundLoopVariableTests(unittest.TestCase):
    """A loop variable may be unbound if the range can be empty.

    CPython raises UnboundLocalError there. The native slot would hold an
    unrelated value, so the read is rejected rather than miscompiled.
    """

    def _compile(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "loop.py"
            entry.write_text(source, encoding="utf-8")
            compile_native(entry, root / "loop.bin", "darwin-arm64", clean=True)

    def test_possibly_unbound_read_is_rejected(self):
        with self.assertRaises(NativeCompileError) as caught:
            self._compile(
                "for g in range(0, 0):\n    pass\nraise SystemExit(g)\n"
            )
        self.assertIn("unbound", str(caught.exception))

    def test_provably_non_empty_range_binds_the_name(self):
        self._compile("for i in range(1, 5):\n    pass\nraise SystemExit(i)\n")

    def test_name_bound_before_the_loop_stays_readable(self):
        self._compile(
            "i = 99\nfor i in range(0, 0):\n    pass\nraise SystemExit(i)\n"
        )

    def test_loop_variable_is_usable_inside_the_body(self):
        self._compile(
            "t = 0\nfor i in range(1, 5):\n    t += i\nraise SystemExit(t)\n"
        )


class OperandEvaluatedOnceTests(unittest.TestCase):
    """An operand named once in the source must be evaluated once.

    The backends re-emit an expression tree at every occurrence, so any value
    reused by a lowering has to be held in a slot instead.
    """

    def _run(self, source: str, expected: int) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "p.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "p.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            ).returncode
            self.assertEqual(reference, expected, "test expectation is wrong")
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            native = subprocess.run([str(artifact)], capture_output=True).returncode
            self.assertEqual(native, reference)

    def test_chained_comparison_matches_cpython(self):
        for source, expected in (
            ("n = 0\nfor k in range(1, 4):\n    n += k\n"
             "raise SystemExit(1 if 0 <= n < 10 else 2)\n", 1),
            ("n = 0\nfor k in range(1, 20):\n    n += k\n"
             "raise SystemExit(1 if 0 <= n < 10 else 2)\n", 2),
            ("a = 0\nfor k in range(1, 3):\n    a += k\n"
             "raise SystemExit(1 if 0 < a < 5 < 100 else 2)\n", 1),
        ):
            self._run(source, expected)

    def test_self_referential_assignment_keeps_the_previous_value(self):
        # Promoting a constant name to a runtime value must write the old
        # constant into its slot first: the right-hand side reads the name.
        self._run("x = 5\nfor k in range(1, 3):\n    x = x + k\nraise SystemExit(x)\n", 8)

    def test_self_referential_float_assignment(self):
        self._run(
            "y = 1.5\nfor k in range(1, 3):\n    y = y + 0.5\n"
            "raise SystemExit(int(y * 10))\n",
            25,
        )


class Arm64CallAbiTests(unittest.TestCase):
    """The call ABI at the encoder level, independent of any front end."""

    @staticmethod
    def _module(argument_count: int, callee_slots: int = 1):
        from py2bin.native.ir import (
            Call,
            ExitValue,
            Function,
            IntConstant,
            IntLoad,
            Return,
            Store,
        )

        call = Call(
            "callee", tuple(IntConstant(index + 1) for index in range(argument_count))
        )
        return Module(
            [Store(0, call), ExitValue(IntLoad(0))],
            1,
            [
                Function(
                    "callee",
                    argument_count,
                    max(callee_slots, argument_count),
                    [Return(IntConstant(7))],
                )
            ],
        )

    def _words(self, code: bytes) -> list[int]:
        return list(struct.unpack(f"<{len(code) // 4}I", code[: len(code) // 4 * 4]))

    def test_arguments_land_in_x0_through_x7(self):
        # Each argument is loaded from its own spill cell into its register.
        # The cells are read at an offset now that a memory argument area may
        # sit below them, so check the registers, not the order.
        words = self._words(encode_darwin_arm64(self._module(8), 0x100004000))
        loads = [word for word in words if word & 0xFFC003E0 == 0xF94003E0]
        self.assertEqual(sorted({word & 7 for word in loads}), list(range(8)))

    def test_arguments_past_the_eighth_go_in_the_memory_area(self):
        # AAPCS64 passes the ninth argument onward at [sp], [sp+8], ... The
        # encoder copies them there through x9, so a ten-argument call must
        # contain stores of x9 that an eight-argument call does not.
        few = self._words(encode_darwin_arm64(self._module(8), 0x100004000))
        many = self._words(encode_darwin_arm64(self._module(10), 0x100004000))
        stores = lambda words: sum(
            1 for word in words if word & 0xFFC003FF == 0xF90003E9
        )
        self.assertEqual(stores(few), 0)
        self.assertEqual(stores(many), 2)

    def test_the_branch_targets_the_callee_and_the_stack_is_aligned(self):
        words = self._words(encode_darwin_arm64(self._module(3), 0x100004000))
        branches = [
            (index, word)
            for index, word in enumerate(words)
            if word & 0xFC000000 == 0x94000000
        ]
        self.assertEqual(len(branches), 1)
        index, word = branches[0]
        offset = word & 0x03FFFFFF
        if offset & 0x02000000:
            offset -= 1 << 26
        target = index + offset
        # The callee starts with its prologue: a frame, then the saved pair.
        self.assertEqual(words[target + 1], 0xA9007BFD)  # stp x29, x30, [sp]
        self.assertEqual(words[target + 2], 0x910003FD)  # mov x29, sp
        # SP must be a MULTIPLE OF 16 at the branch, which AAPCS64 requires.
        # It is not back at zero: the argument spill cells and the memory
        # argument area stay allocated across the call and are released after
        # it, so the test checks alignment rather than depth.
        depth = 0
        for word in words[2:index]:  # words[0:2] are the entry point's own frame
            if word & 0xFFC003FF == 0xD10003FF:  # sub sp, sp, #imm
                depth -= (word >> 10) & 0xFFF
            elif word & 0xFFC003FF == 0x910003FF:  # add sp, sp, #imm
                depth += (word >> 10) & 0xFFF
        self.assertEqual(depth % 16, 0)
        self.assertEqual(depth % 16, 0, "sp must be 16-byte aligned at the call")

    def test_the_callee_saves_and_restores_the_frame_and_returns(self):
        module = self._module(1, callee_slots=4)
        words = self._words(encode_darwin_arm64(module, 0x100004000))
        self.assertEqual(words[-1], 0xD65F03C0)  # ret
        self.assertEqual(words[-2], 0x910003FF | (48 << 10))  # add sp, sp, #48
        self.assertEqual(words[-3], 0xA9407BFD)  # ldp x29, x30, [sp]
        self.assertEqual(words[-4], 0x910003BF)  # mov sp, x29

    def test_a_call_to_an_undefined_function_is_refused(self):
        from py2bin.native.ir import Call, ExitValue, Store

        module = Module([Store(0, Call("missing")), ExitValue(IntConstant(0))], 1)
        with self.assertRaisesRegex(ValueError, "undefined native IR function"):
            encode_darwin_arm64(module, 0x100004000)

    def test_a_return_outside_a_function_body_is_refused(self):
        from py2bin.native.ir import Return

        module = Module([Return(IntConstant(1))], 0)
        with self.assertRaisesRegex(ValueError, "only legal inside a Function"):
            encode_darwin_arm64(module, 0x100004000)


class Arm64StaticStorageTests(unittest.TestCase):
    """``Module.static_bytes`` and ``GlobalAddress`` at the encoder level.

    Static storage is what a C file-scope variable needs: one object that
    outlives every stack frame and that the entry point and every ``Function``
    body reach identically. The block is an anonymous mapping whose base sits
    in X28 for the whole run.
    """

    @staticmethod
    def _module():
        from py2bin.native.ir import (
            Call,
            ExitValue,
            Function,
            FunctionAddress,
            GlobalAddress,
            HeapLoad,
            HeapStore,
            IndirectCall,
            IntBinary,
            IntLoad,
            Return,
            Store,
        )

        bump = Function(
            "bump",
            0,
            1,
            [
                HeapStore(
                    GlobalAddress(0),
                    IntBinary(
                        "add", HeapLoad(GlobalAddress(0), 8, True), IntConstant(100)
                    ),
                    8,
                ),
                Return(IntConstant(0)),
            ],
        )
        return Module(
            [
                HeapStore(GlobalAddress(0), IntConstant(23), 8),
                Store(0, Call("bump", ())),
                Store(1, FunctionAddress("bump")),
                Store(2, IndirectCall(IntLoad(1), ())),
                ExitValue(HeapLoad(GlobalAddress(0), 8, True)),
            ],
            4,
            [bump],
            static_bytes=4096,
        )

    def test_the_entry_prologue_maps_the_block_and_keeps_it_in_x28(self):
        code = encode_darwin_arm64(self._module(), 0x100004000)
        words = list(struct.unpack(f"<{len(code) // 4}I", code[: len(code) // 4 * 4]))
        # mov x28, x0 establishes the base exactly once, in the entry prologue.
        self.assertEqual(words.count(0xAA0003FC), 1)
        # add x0, x28, #0 is how a GlobalAddress(0) is materialized.
        self.assertIn(0x91000380, words)
        # Nothing else ever writes X28, so the base survives every call.
        for word in words:
            destination = word & 0x1F
            if destination == 28:
                self.assertEqual(word, 0xAA0003FC)

    @unittest.skipUnless(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "runs the produced arm64 Mach-O natively",
    )
    def test_the_entry_point_and_a_function_share_one_static_object(self):
        from py2bin.native.formats.macho import write_macho_arm64

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "static.bin"
            artifact.write_bytes(
                write_macho_arm64(encode_darwin_arm64(self._module(), 0x100004000))
            )
            artifact.chmod(0o755)
            # 23, then +100 through a direct call, then +100 through a call made
            # via the function's address held in a stack slot.
            self.assertEqual(subprocess.run([str(artifact)]).returncode, 223)

    def test_targets_without_a_static_base_reject_the_module(self):
        for target in (
            "windows-x86_64",
        ):
            with self.subTest(target=target):
                from py2bin.native.ir import ExitValue, GlobalAddress, HeapLoad

                # No Function, so this reaches the static-storage check rather
                # than the call-ABI one that precedes it.
                module = Module(
                    [ExitValue(HeapLoad(GlobalAddress(0), 8, True))],
                    0,
                    static_bytes=4096,
                )
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(
                        NativeCompileError, "static storage .*only supported"
                    ):
                        compile_native_module(
                            Path("static.py"),
                            module,
                            Path(directory) / "static.bin",
                            target=target,
                            clean=True,
                        )

    def test_the_address_of_an_undefined_function_is_refused(self):
        from py2bin.native.ir import ExitValue, FunctionAddress, Store

        module = Module(
            [Store(0, FunctionAddress("missing")), ExitValue(IntConstant(0))], 1
        )
        with self.assertRaisesRegex(ValueError, "undefined native IR function"):
            encode_darwin_arm64(module, 0x100004000)


class WindowsArm64CallAbiTests(unittest.TestCase):
    """The Windows ARM64 call ABI, verified by decoding the instructions.

    Windows binaries cannot be executed on this host and no emulator is used,
    so these assertions check the encoding, not the behaviour.
    """

    _SOURCE = (
        "long long f(long long n) { return n <= 1 ? 1 : n * f(n - 1); }\n"
        "int main(void) { return f(5); }\n"
    )

    def _words(self):
        from py2bin.c_frontend import compile_c_to_ir
        from py2bin.native.arm64 import encode_windows

        module = compile_c_to_ir(self._SOURCE, "f.c", "windows-arm64")
        code = encode_windows(
            module,
            0x1000,
            {"GetStdHandle": 0x2000, "WriteFile": 0x2008, "ExitProcess": 0x2010},
        )
        return struct.unpack(f"<{len(code) // 4}I", code)

    def test_a_recursive_program_encodes(self):
        words = self._words()
        branches = [w for w in words if (w >> 26) == 0x25]  # bl
        # One call from main, one recursive call inside f.
        self.assertEqual(len(branches), 2)

    def test_each_body_saves_and_restores_the_frame(self):
        words = self._words()
        self.assertIn(0xA9007BFD, words)  # stp x29, x30, [sp]
        self.assertIn(0xA9407BFD, words)  # ldp x29, x30, [sp]
        self.assertIn(0xD65F03C0, words)  # ret

    def test_the_recursive_call_targets_the_body(self):
        words = self._words()
        entries = [i for i, w in enumerate(words) if w == 0xA9007BFD]
        self.assertTrue(entries)
        body = entries[0] - 1  # the frame allocation precedes the save
        for index, word in enumerate(words):
            if (word >> 26) == 0x25:
                distance = ((word & 0x3FFFFFF) ^ 0x2000000) - 0x2000000
                self.assertEqual(index + distance, body)

    def test_a_windows_program_still_builds_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "f.c"
            entry.write_text(self._SOURCE, encoding="utf-8")
            artifact = root / "f.exe"
            from py2bin.c_native import compile_c_native

            compile_c_native(entry, artifact, target="windows-arm64", clean=True)
            image = artifact.read_bytes()
            self.assertEqual(image[:2], b"MZ")
            offset = struct.unpack("<I", image[0x3C:0x40])[0]
            self.assertEqual(struct.unpack("<H", image[offset + 4 : offset + 6])[0], 0xAA64)
