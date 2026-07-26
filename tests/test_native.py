from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest

import struct

from py2bin.native import (
    NativeCompileError,
    compile_all,
    compile_native,
    resolve_target,
    supported_targets,
)
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

    def test_every_target_establishes_a_static_base(self):
        """No target lacks static storage now: the POSIX encoders reserve the
        block with an anonymous mmap and the Windows ones with VirtualAlloc,
        both giving the zero-filled writable memory C requires."""

        from py2bin.c_native import compile_c_native

        source = (
            "int counter;\n"
            "void bump(void) { counter = counter + 5; }\n"
            "int main(void) { counter = 1; bump(); return counter; }\n"
        )
        for target in supported_targets():
            with self.subTest(target=target):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    entry = root / "g.c"
                    entry.write_text(source, encoding="utf-8")
                    compile_c_native(
                        entry, root / "g.bin", target=target, clean=True
                    )


    def test_the_address_of_an_undefined_function_is_refused(self):
        from py2bin.native.ir import ExitValue, FunctionAddress, Store

        module = Module(
            [Store(0, FunctionAddress("missing")), ExitValue(IntConstant(0))], 1
        )
        with self.assertRaisesRegex(ValueError, "undefined native IR function"):
            encode_darwin_arm64(module, 0x100004000)


class Arm64FarStackSlotTests(unittest.TestCase):
    """Frame references past the 12-bit reach of LDR/STR.

    The scaled immediate stops at 32760 bytes, which used to cap a function at
    4095 slots. Past that the offset's high 12 bits go into X17 and the access
    is made off X17 instead of X29. The encodings below were checked against
    ``clang -target arm64-apple-darwin``.
    """

    def _reference(self, offset: int, encoding: int, rt: int = 0):
        from py2bin.native.arm64 import _frame_reference

        return _frame_reference(offset, encoding, rt)

    def test_an_offset_inside_the_immediate_is_still_one_word(self):
        # Nothing about the near case may change; existing images must not move.
        self.assertEqual(
            self._reference(0x7FF8, 0xF9400000), [0xF97FFFA0]  # ldr x0,[x29,#32760]
        )
        self.assertEqual(
            self._reference(16, 0xFD000000), [0xFD000BA0]  # str d0, [x29, #16]
        )

    def test_the_first_offset_past_the_immediate_uses_x17(self):
        self.assertEqual(
            self._reference(0x8000, 0xF9400000),
            [0x914023B1, 0xF9400220],  # add x17,x29,#8,lsl#12; ldr x0,[x17]
        )

    def test_the_register_and_the_float_forms_address_off_x17_too(self):
        self.assertEqual(
            self._reference(0x8000 + 0xFF8, 0xF9000000, 3),
            [0x914023B1, 0xF907FE23],  # add x17,x29,#8,lsl#12; str x3,[x17,#4088]
        )
        self.assertEqual(
            self._reference(0x8010, 0xFD400000),
            [0x914023B1, 0xFD400A20],  # add x17,x29,#8,lsl#12; ldr d0,[x17,#16]
        )

    def test_a_far_reference_writes_only_x17(self):
        # X0 and X1 carry the size and the old bump pointer across the three
        # slot accesses of a HeapAlloc, and the prologue holds X0-X7 across its
        # whole spill loop, so a scratch register that is not X17 miscompiles
        # both. Rd of the ADD is bits [4:0].
        words = self._reference(0x40000, 0xF9400000, 1)
        self.assertEqual(words[0] & 0x1F, 17)

    def test_a_frame_reference_past_the_shifted_add_is_refused(self):
        with self.assertRaisesRegex(ValueError, "beyond the 16777208-byte reach"):
            self._reference((0xFFF << 12) + 0x1000, 0xF9400000)

    def test_a_function_frame_past_the_old_ceiling_encodes_and_runs(self):
        # Ten parameters so the ninth and tenth arrive in memory above the
        # frame, and a frame far past 4095 slots so both the incoming-argument
        # load and the body's own slot accesses need the X17 path. Only the IR
        # can build a Function this size; the C front end caps itself lower.
        from py2bin.native.ir import (
            Call,
            ExitValue,
            Function,
            IntBinary,
            IntConstant,
            IntLoad,
            Return,
            Store,
        )
        from py2bin.native.formats.macho import write_macho_arm64

        slots = 40000
        total = IntLoad(0)
        for index in range(1, 10):
            total = IntBinary("add", total, IntLoad(index))
        body = [
            Store(slots - 1, total),  # a slot far past the immediate
            Store(10, IntConstant(7)),  # and a near one, written in between
            Return(IntBinary("add", IntLoad(slots - 1), IntLoad(10))),
        ]
        module = Module(
            [
                Store(
                    0,
                    Call(
                        "wide",
                        tuple(IntConstant(value) for value in range(1, 11)),
                    ),
                ),
                ExitValue(IntLoad(0)),
            ],
            4,
            [Function("wide", 10, slots, body)],
        )
        code = encode_darwin_arm64(module, 0x100004000)
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            return
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "wide"
            artifact.write_bytes(write_macho_arm64(code))
            artifact.chmod(0o755)
            result = subprocess.run([str(artifact)], capture_output=True)
        self.assertEqual(result.returncode, sum(range(1, 11)) + 7)


class NativeStackBudgetTests(unittest.TestCase):
    """A frame is real OS stack, so an unbounded one crashes instead of answering.

    Before this limit existed, x86-64 (whose displacement is 32 bits and has no
    encoding ceiling) compiled a 10 MB frame without complaint and the binary
    died with SIGSEGV while CPython printed an answer.
    """

    @staticmethod
    def _source(assignments: int) -> str:
        # ``n`` is unfoldable, so each assignment pins a runtime value.
        lines = ["n = 0", "for i in range(0, 3):", "    n += 1"]
        lines += [f"v{index} = n + {index}" for index in range(assignments)]
        lines.append(f"print(v0, v1, v4095, v{assignments - 1})")
        return "\n".join(lines) + "\n"

    def test_a_frame_past_the_budget_is_refused_for_every_target(self):
        from py2bin.native.ir import MAXIMUM_STACK_SLOTS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "huge.py"
            entry.write_text(self._source(MAXIMUM_STACK_SLOTS + 1000), encoding="utf-8")
            for target in supported_targets():
                with self.assertRaisesRegex(
                    ValueError, "budget one frame may take from the thread stack"
                ):
                    compile_native(entry, root / "huge.bin", target, clean=True)

    def test_a_frame_well_past_the_old_arm64_ceiling_builds_and_runs(self):
        # 20070 slots: five times the 4095 the immediate allowed. The early,
        # low-numbered values are read back last, after every high slot has
        # been written through X17.
        source = self._source(20000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "wide.py"
            entry.write_text(source, encoding="utf-8")
            for target in supported_targets():
                compile_native(entry, root / f"wide-{target}", target, clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            native = subprocess.run(
                [str(root / "wide-darwin-arm64")], capture_output=True
            )
        self.assertEqual(native.stdout, reference.stdout)
        self.assertEqual(native.returncode, reference.returncode)


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


class FloorDivisionTests(unittest.TestCase):
    """Python's // and % floor; the hardware divide truncates.

    -7 // 2 is -4 in Python and -3 in C, and -7 % 2 is 1 rather than -1. The
    two agree only when the remainder is zero or its sign matches the
    divisor's, so every sign combination is checked against CPython.
    """

    def _matches(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "d.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "d.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.returncode, reference.returncode)

    def test_every_sign_combination_floors_like_python(self):
        for a, b in ((17, 5), (-17, 5), (17, -5), (-17, -5)):
            with self.subTest(a=a, b=b):
                self._matches(
                    "a = 0\n"
                    "for i in range(1, 2):\n"
                    f"    a = {a}\n"
                    f"b = {b}\n"
                    "raise SystemExit((a // b + 20) * 10 + (a % b + 10))\n"
                )

    def test_division_by_zero_reports_and_exits_one(self):
        self._matches(
            "a = 0\n"
            "for i in range(1, 2):\n"
            "    a = 5\n"
            "b = 0\n"
            "raise SystemExit(a // b)\n"
        )

    def test_division_is_rejected_in_an_eagerly_evaluated_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "d.py"
            # b must be a runtime value, or the condition folds away and the
            # division is never lowered at all.
            entry.write_text(
                "a = 0\n"
                "b = 0\n"
                "for i in range(1, 2):\n"
                "    a = 5\n"
                "    b = i - 1\n"
                "raise SystemExit(a // b if b != 0 else 7)\n",
                encoding="utf-8",
            )
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "d.bin", "darwin-arm64", clean=True)
            self.assertIn("ZeroDivisionError", str(caught.exception))


class NativeDictionaryTests(unittest.TestCase):
    """Integer-keyed dicts lowered to an open-addressing table.

    The table is a header of [capacity][count][keys][used] followed by capacity
    entries of [state][key][value]. Every case here compares the binary against
    CPython rather than against a hand-computed number, so a wrong expectation
    cannot hide a wrong answer.
    """

    def _run(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "d.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "d.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            ).returncode
            native = subprocess.run([str(artifact)], capture_output=True).returncode
            self.assertEqual(native, reference)

    def test_literal_insert_and_lookup(self):
        self._run("d = {1: 10, 2: 20}\nd[3] = 30\nraise SystemExit(d[1] + d[2] + d[3])\n")

    def test_keys_that_share_a_home_slot(self):
        # 1, 9 and 17 are congruent modulo the initial capacity of 8, so each
        # one has to be found by probing past the others.
        self._run(
            "d = {1: 10, 9: 90, 17: 170}\n"
            "raise SystemExit(d[1] + d[9] // 10 + d[17] // 10)\n"
        )

    def test_assigning_an_existing_key_replaces_it(self):
        self._run("d = {5: 1}\nd[5] = 99\nd[6] = 1\nraise SystemExit(d[5] + d[6])\n")

    def test_negative_keys(self):
        self._run("d = {}\nd[-1] = 7\nd[-9] = 11\nraise SystemExit(d[-1] + d[-9])\n")

    def test_the_table_grows_instead_of_filling_up(self):
        # More entries than any fixed capacity: every one must survive the
        # rehash and still be reachable afterwards.
        self._run(
            "d = {}\ni = 0\n"
            "while i < 200:\n    d[i * 8] = i\n    i += 1\n"
            "total = 0\nj = 0\n"
            "while j < 200:\n    total += d[j * 8]\n    j += 1\n"
            "raise SystemExit(total % 251)\n"
        )

    def test_growth_does_not_double_count_replaced_keys(self):
        self._run(
            "d = {}\ni = 0\nwhile i < 50:\n    d[i] = 1\n    i += 1\n"
            "i = 0\nwhile i < 50:\n    d[i] = 2\n    i += 1\n"
            "raise SystemExit(len(d) + d[7])\n"
        )

    def test_a_missing_key_fails_like_cpython(self):
        self._run("d = {1: 10}\nraise SystemExit(d[2])\n")

    def test_length_counts_distinct_keys(self):
        self._run(
            "d = {}\ni = 0\nwhile i < 37:\n    d[i * 5] = i\n    i += 1\n"
            "d[0] = 99\nraise SystemExit(len(d))\n"
        )

    def test_membership(self):
        self._run(
            "d = {3: 1, 11: 1}\ntotal = 0\n"
            "if 3 in d:\n    total += 1\n"
            "if 4 in d:\n    total += 10\n"
            "if 7 not in d:\n    total += 100\n"
            "raise SystemExit(total + len(d))\n"
        )

    def test_membership_guards_a_lookup(self):
        self._run(
            "d = {}\ni = 0\nwhile i < 30:\n    d[i * 2] = i\n    i += 1\n"
            "hits = 0\nk = 0\n"
            "while k < 60:\n"
            "    if k in d:\n        hits += d[k]\n"
            "    k += 1\n"
            "raise SystemExit(hits % 251)\n"
        )

    def test_string_keys(self):
        self._run(
            'd = {"alpha": 10, "beta": 20}\nd["gamma"] = 30\n'
            'raise SystemExit(d["alpha"] + d["beta"] + d["gamma"] + len(d))\n'
        )

    def test_a_key_built_at_runtime_finds_a_literal_entry(self):
        # Equal bytes, different pointers: the probe has to compare contents,
        # not addresses.
        self._run(
            'd = {"abc": 42}\nk = ""\nk = k + "a"\nk = k + "b"\nk = k + "c"\n'
            "raise SystemExit(d[k])\n"
        )

    def test_a_prefix_is_not_the_same_key(self):
        self._run(
            'd = {"abc": 1}\nn = 0\nif "ab" in d:\n    n += 10\n'
            'if "abcd" in d:\n    n += 100\nif "abc" in d:\n    n += 1\n'
            "raise SystemExit(n)\n"
        )

    def test_keys_differing_in_one_byte(self):
        self._run(
            'd = {"aaaa": 1, "aaab": 2, "aaba": 3, "abaa": 4, "baaa": 5}\n'
            'raise SystemExit(d["aaaa"] + d["aaab"] * 2 + d["aaba"] * 3 '
            '+ d["abaa"] * 4 + d["baaa"] * 5)\n'
        )

    def test_the_empty_string_is_a_usable_key(self):
        self._run('d = {"": 5}\nd["a"] = 6\nk = ""\nraise SystemExit(d[k] + d["a"] + len(d))\n')

    def test_string_keys_survive_several_rehashes(self):
        self._run(
            'd: dict[str, int] = {}\ni = 0\ns = ""\n'
            'while i < 300:\n    s = s + "k"\n    d[s] = i\n    i += 1\n'
            'total = 0\nt = ""\nj = 0\n'
            'while j < 300:\n    t = t + "k"\n    total += d[t]\n    j += 1\n'
            "raise SystemExit(total % 251)\n"
        )

    def test_float_values(self):
        self._run(
            "d = {1: 1.5, 2: 2.25}\nd[3] = 4.0\nt = d[1] + d[2] + d[3]\n"
            "raise SystemExit(int(t * 4))\n"
        )

    def test_float_values_keep_their_bits_through_a_rehash(self):
        self._run(
            "d: dict[int, float] = {}\ni = 0\n"
            "while i < 200:\n    d[i] = i * 0.5\n    i += 1\n"
            "total = 0.0\nj = 0\n"
            "while j < 200:\n    total += d[j]\n    j += 1\n"
            "raise SystemExit(int(total) % 251)\n"
        )

    def test_negative_and_fractional_float_values(self):
        self._run(
            "d = {1: -1.5, 2: 0.0, 3: 2.75}\nd[4] = -0.125\n"
            "raise SystemExit(int((d[1] + d[2] + d[3] + d[4]) * 8) + 100)\n"
        )

    def test_string_keys_with_float_values(self):
        self._run(
            'scores: dict[str, float] = {}\nname = ""\ni = 0\n'
            'while i < 150:\n    name = name + "n"\n    scores[name] = i * 0.25\n'
            "    i += 1\n"
            'total = 0.0\nprobe = ""\nj = 0\n'
            'while j < 150:\n    probe = probe + "n"\n    total += scores[probe]\n'
            "    j += 1\n"
            "raise SystemExit(int(total) % 251)\n"
        )

    def test_an_annotation_types_an_empty_literal(self):
        self._run(
            'd: dict[str, float] = {}\nd["x"] = 0.5\nd["y"] = 1.25\n'
            'raise SystemExit(int((d["x"] + d["y"]) * 16))\n'
        )

    def test_a_key_of_the_wrong_kind_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source, expected in (
                ('d = {1: 10}\nraise SystemExit(d["a"])\n', "int keys"),
                ('d = {"a": 1}\nraise SystemExit(d[1])\n', "str keys"),
                ("d = {1: 10}\nd[2] = 1.5\nraise SystemExit(1)\n", "integer values"),
                ("d = {1.5: 2}\nraise SystemExit(1)\n", "signed 64-bit integers or runtime strings"),
                ('d: dict[str, str] = {}\nraise SystemExit(1)\n', "dict[int|str, int|float]"),
            ):
                entry = root / "k.py"
                entry.write_text(source, encoding="utf-8")
                with self.assertRaises(NativeCompileError) as caught:
                    compile_native(entry, root / "k.bin", "darwin-arm64", clean=True)
                self.assertIn(expected, str(caught.exception))

    def test_a_lookup_is_rejected_where_both_arms_are_evaluated(self):
        # select_integer lowers both arms, so a lookup that can raise KeyError
        # must not be placed in one: the failing arm would run regardless.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "d.py"
            entry.write_text(
                "d = {1: 10}\nn = 0\nfor k in range(1, 3):\n    n += k\n"
                "raise SystemExit(d[1] if n > 0 else d[2])\n",
                encoding="utf-8",
            )
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "d.bin", "darwin-arm64", clean=True)
            self.assertIn("KeyError", str(caught.exception))


class NativeDictionaryIterationTests(unittest.TestCase):
    """Walking a dict, which CPython does in insertion order.

    The table is open-addressed, so its own order is hash order and printing
    while walking it would be a different program. The dict therefore keeps a
    list of its keys in the order they were first stored, and every case here
    compares stdout with CPython so a wrong order cannot pass.
    """

    def _run(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "i.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "i.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.stdout, reference.stdout)
            self.assertEqual(native.returncode, reference.returncode)

    def test_keys_come_out_in_insertion_order(self):
        # 5 and 3 land in slots 5 and 3, so a walk of the table would print 3
        # first. Insertion order is the other one.
        self._run("d = {5: 10, 3: 20}\nfor k in d:\n    print(k, d[k])\n")

    def test_order_survives_every_rehash(self):
        self._run(
            "d = {}\ni = 0\nwhile i < 40:\n    d[(39 - i) * 8] = i\n    i += 1\n"
            "for k in d:\n    print(k)\n"
        )

    def test_string_keys_come_out_in_insertion_order(self):
        # Hashed, these four sit in the table as two, three, four, one.
        self._run(
            'd = {"one": 1, "two": 2, "three": 3, "four": 4}\n'
            "for k in d:\n    print(k, d[k])\n"
        )

    def test_keys_values_and_items_over_float_values(self):
        self._run(
            "d = {5: 1.5, 3: -0.25, 9: 0.0}\n"
            "for k in d.keys():\n    print(k)\n"
            "for v in d.values():\n    print(v)\n"
            "for k, v in d.items():\n    print(k, v)\n"
        )

    def test_string_keys_with_items(self):
        self._run(
            'd = {"x": 1, "y": 2, "z": 3}\n'
            "for k, v in d.items():\n    print(k, v)\n"
        )

    def test_bool_values_print_as_bools(self):
        self._run(
            "d = {5: True, 3: False}\n"
            "for k, v in d.items():\n    print(k, v)\n"
        )

    def test_bool_keys_print_as_bools(self):
        self._run("d = {True: 5, False: 6}\nfor k in d:\n    print(k)\n")

    def test_an_empty_dict_binds_nothing(self):
        self._run(
            'd: dict[str, int] = {}\nfor k in d:\n    print("never")\n'
            'print("done")\n'
        )

    def test_growing_the_dict_raises_after_the_earlier_keys(self):
        self._run(
            "d = {1: 1, 2: 2, 3: 3}\nfor k in d:\n    print(k)\n"
            "    if k == 2:\n        d[9] = 9\n"
        )

    def test_growing_the_dict_raises_even_when_exhausted(self):
        self._run(
            'd = {1: 1}\nfor k in d:\n    d[2] = 2\nprint("unreachable")\n'
        )

    def test_replacing_an_existing_key_does_not_raise(self):
        self._run("d = {1: 1, 2: 2}\nfor k in d:\n    d[1] = 9\n    print(k)\n")

    def test_the_runtime_error_is_catchable(self):
        self._run(
            "d = {1: 1}\ntry:\n    for k in d:\n        d[2] = 2\n"
            'except RuntimeError:\n    print("caught")\n'
        )

    def test_a_store_from_an_inlined_function_still_raises(self):
        # An AST scan of the loop body would miss this one; the count check at
        # the top of each iteration does not care where the store came from.
        self._run(
            "d = {1: 1, 2: 2}\n"
            "def grow() -> int:\n    d[7] = 7\n    return 0\n"
            "for k in d:\n    print(k)\n    n = grow()\n"
            'print("after")\n'
        )

    def test_breaking_before_the_next_key_does_not_raise(self):
        self._run(
            "d = {1: 1, 2: 2}\nfor k in d:\n    d[9] = 9\n    break\n"
            "print(len(d))\n"
        )

    def test_replacing_a_value_does_not_reorder_its_key(self):
        self._run(
            "d = {5: 1, 3: 2}\nd[5] = 99\nd[7] = 3\n"
            "for k in d:\n    print(k, d[k])\n"
        )

    def test_break_continue_and_nesting(self):
        self._run(
            "d = {5: 1, 3: 2, 9: 3}\n"
            "for k in d:\n    if k == 3:\n        continue\n"
            "    for j in d:\n        print(k, j)\n"
            "    if k == 9:\n        break\n"
        )

    def test_iterating_inside_a_function(self):
        self._run(
            "d = {4: 40, 1: 10, 7: 70}\n"
            "def total() -> int:\n    s = 0\n    for k in d:\n"
            "        s += k * d[k]\n    return s\n"
            "print(total())\n"
        )

    def test_many_string_keys_through_several_rehashes(self):
        self._run(
            'd: dict[str, float] = {}\ni = 0\ns = ""\n'
            "while i < 300:\n    s = s + \"a\"\n    d[s] = i * 0.5\n    i += 1\n"
            'n = 0\ntotal = 0.0\nlast = ""\n'
            "for k, v in d.items():\n    n += 1\n    total += v\n    last = k\n"
            "print(n, total, len(last))\n"
        )

    def test_what_dict_iteration_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source, expected in (
                (
                    "d = {1: 1, 2: 2}\nfor k in d:\n    d = {3: 3}\n    print(k)\n",
                    "rebinding it",
                ),
                (
                    "d = {1: 1}\nfor k, v in d:\n    print(k)\n",
                    "binds one name",
                ),
                (
                    "d = {1: 1}\nfor k in d.items():\n    print(k)\n",
                    "binds two names",
                ),
                ("d = {1: 1}\nn = d.keys()\nprint(n)\n", "native integer subset"),
                ("d = {1: 1}\nprint(len(d.keys()))\n", "native len()"),
                (
                    "d = {1: 1}\nxs = [k for k in d]\nprint(xs)\n",
                    "a range or a runtime list",
                ),
                (
                    "d = {True: 1, 1: 2}\nprint(len(d))\n",
                    "this dict's keys holds bools already",
                ),
            ):
                entry = root / "j.py"
                entry.write_text(source, encoding="utf-8")
                with self.assertRaises(NativeCompileError) as caught:
                    compile_native(entry, root / "j.bin", "darwin-arm64", clean=True)
                self.assertIn(expected, str(caught.exception))


class NativeDictionaryDeleteTests(unittest.TestCase):
    """`del d[k]`, which leaves a tombstone rather than an empty slot.

    An emptied slot would end a probe that had walked past it to reach a later
    key, so the state word gets a third value meaning "keep walking". Every
    case here compares stdout and exit status with CPython.
    """

    def _run(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "x.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "x.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            native = subprocess.run(
                [str(artifact)], capture_output=True, timeout=60
            )
            self.assertEqual(native.stdout, reference.stdout)
            self.assertEqual(native.returncode, reference.returncode)

    def test_a_probe_walks_past_a_deleted_entry(self):
        # 1, 9 and 17 all come home to slot 1 at the initial capacity of 8, so
        # 17 is only reachable through the slot 9 was deleted from.
        self._run(
            "d: dict[int, int] = {}\nd[1] = 1\nd[9] = 9\nd[17] = 17\n"
            "del d[9]\n"
            "print(len(d), 17 in d, d[17], 9 in d, 1 in d)\n"
        )

    def test_inserting_and_deleting_forever_terminates(self):
        # The live count never passes one, so a table that grew on the count
        # would never grow, would fill with tombstones, and would leave the
        # probe for the ninth key with nowhere to stop.
        self._run(
            "d: dict[int, int] = {}\n"
            "for i in range(0, 400):\n    d[i] = i\n    del d[i]\n"
            "print(len(d))\n"
        )

    def test_a_missing_key_raises_KeyError(self):
        self._run(
            "d = {1: 10}\n"
            "try:\n    del d[2]\nexcept KeyError:\n    print('caught')\n"
            "print(len(d))\n"
            "del d[2]\n"
        )

    def test_tombstones_survive_a_rehash(self):
        self._run(
            "d: dict[int, int] = {}\ni = 0\n"
            "while i < 40:\n    d[i * 8] = i\n    i += 1\n"
            "i = 0\n"
            "while i < 40:\n    if i % 3 == 0:\n        del d[i * 8]\n    i += 1\n"
            "i = 0\n"
            "while i < 40:\n    print(i * 8, i * 8 in d)\n    i += 1\n"
            "print(len(d))\n"
        )

    def test_a_deleted_string_key_is_gone_and_the_rest_are_reachable(self):
        self._run(
            'd = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, '
            '"six": 6}\n'
            'del d["two"]\ndel d["four"]\ndel d["six"]\n'
            "for k in d:\n    print(k, d[k], k in d)\n"
            'print("two" in d, "four" in d, len(d))\n'
        )

    def test_a_reinserted_key_goes_to_the_end_of_the_order(self):
        self._run(
            "d = {5: 10, 3: 20, 9: 30}\ndel d[3]\nd[3] = 99\n"
            "for k in d:\n    print(k, d[k])\n"
        )

    def test_deleting_while_walking_the_same_dict_is_rejected(self):
        # The size check this used to rely on cannot see the general case: a
        # body that deletes and inserts in one pass leaves the count where it
        # was, and the walk - which counts along the insertion order - then
        # visits the wrong keys with nothing raising. Refused at build time
        # instead, including the shapes CPython allows.
        for body in (
            "d = {1: 10}\nfor k in d:\n    print(k)\n    del d[k]\n",
            "d = {1: 1, 2: 2}\nfor k in d:\n    del d[k]\n    d[k] = k * 10\n",
            "d = {1: 1, 2: 2}\nfor k, v in d.items():\n    del d[k]\n",
        ):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                entry = root / "d.py"
                entry.write_text(body + 'print("never")\n', encoding="utf-8")
                with self.assertRaises(NativeCompileError) as caught:
                    compile_native(entry, root / "d.bin", "darwin-arm64", clean=True)
                self.assertIn("cannot delete from", str(caught.exception))

    def test_deleting_a_different_dict_inside_a_walk_is_allowed(self):
        self._run(
            "a = {1: 1, 2: 2}\nb = {7: 7, 8: 8, 9: 9}\n"
            "for k in a:\n    del b[k + 6]\n"
            "print(len(b))\nfor k in b:\n    print(k)\n"
        )

    def test_bools_keep_their_identity_across_a_delete(self):
        self._run(
            "d = {True: 5, False: 6}\ndel d[False]\n"
            "for k in d:\n    print(k)\n"
            "e = {5: True, 3: False, 9: True}\ndel e[3]\n"
            "for k, v in e.items():\n    print(k, v)\n"
        )

    def test_deleting_a_float_value_from_a_function(self):
        self._run(
            "d = {5: 1.5, 3: -0.25, 9: 0.0}\n"
            "def drop(k):\n    del d[k]\n"
            "drop(3)\ndel d[9]\n"
            "for k, v in d.items():\n    print(k, v)\n"
        )

    def test_a_long_mixed_run_of_inserts_and_deletes(self):
        # A deterministic pseudo-random walk: the point is that the table stays
        # in step with CPython's dict over thousands of operations, not that
        # any one of them is interesting.
        self._run(
            "d: dict[int, int] = {}\nseed = 1\ni = 0\n"
            "while i < 600:\n"
            "    seed = (seed * 1103515245 + 12345) % 2147483648\n"
            "    key = seed % 64\n"
            "    if seed % 3 == 0:\n"
            "        if key in d:\n            del d[key]\n"
            "    else:\n        d[key] = i\n"
            "    i += 1\n"
            "print(len(d))\n"
            "for k in d:\n    print(k, d[k])\n"
        )

    def test_a_key_deleted_by_a_non_constant_expression(self):
        self._run(
            "n = 0\nfor i in range(0, 3):\n    n += 1\n"
            "d = {3: 1, 4: 2}\ndel d[n]\n"
            "print(len(d), 3 in d, 4 in d)\n"
        )

    def test_deleting_a_key_of_the_wrong_kind_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "y.py"
            entry.write_text(
                'd = {"a": 1}\ndel d[1]\nprint(len(d))\n', encoding="utf-8"
            )
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "y.bin", "darwin-arm64", clean=True)
            self.assertIn("this dict has str keys", str(caught.exception))


class NativeExceptionTests(unittest.TestCase):
    """try/except/else/finally and raise, without runtime type objects.

    Functions are inlined, so there is no frame stack to unwind: an active
    handler is a label in the same instruction stream. Which class was raised
    is a small integer, and whether a clause catches it is decided at build
    time from the static builtin hierarchy.
    """

    def _run(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "e.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "e.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.returncode, reference.returncode)
            self.assertEqual(native.stdout, reference.stdout)

    def _reject(self, source: str, expected: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "e.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "e.bin", "darwin-arm64", clean=True)
            self.assertIn(expected, str(caught.exception))

    def test_a_matching_clause_catches(self):
        self._run(
            'try:\n    raise ValueError("boom")\nexcept ValueError:\n'
            "    raise SystemExit(7)\nraise SystemExit(1)\n"
        )

    def test_a_body_that_does_not_raise_skips_the_clause(self):
        # The two assignments are both build-time constants, so the value has
        # to be pinned in a slot before the branch or the clause's write wins.
        self._run("try:\n    x = 5\nexcept ValueError:\n    x = 9\nraise SystemExit(x)\n")

    def test_a_clause_catches_a_subclass(self):
        self._run(
            'try:\n    raise ZeroDivisionError("d")\nexcept ArithmeticError:\n'
            "    raise SystemExit(11)\nraise SystemExit(1)\n"
        )

    def test_a_clause_does_not_catch_a_superclass(self):
        self._run(
            'try:\n    raise SystemExit(3)\nexcept Exception:\n'
            "    raise SystemExit(99)\n"
        )

    def test_an_unmatched_exception_keeps_going_outward(self):
        self._run(
            'try:\n    try:\n        raise KeyError("k")\n    except TypeError:\n'
            "        raise SystemExit(2)\nexcept LookupError:\n"
            "    raise SystemExit(21)\nraise SystemExit(1)\n"
        )

    def test_an_escaping_system_exit_keeps_its_status(self):
        self._run(
            "try:\n    raise SystemExit(23)\nexcept ValueError:\n"
            "    raise SystemExit(1)\n"
        )

    def test_a_bare_clause_catches_anything(self):
        self._run('try:\n    raise RuntimeError("r")\nexcept:\n    raise SystemExit(31)\n')

    def test_a_tuple_of_classes(self):
        self._run(
            'try:\n    raise KeyError("k")\nexcept (TypeError, LookupError):\n'
            "    raise SystemExit(43)\n"
        )

    def test_the_first_matching_clause_wins(self):
        self._run(
            'try:\n    raise ZeroDivisionError("z")\nexcept ArithmeticError:\n'
            "    raise SystemExit(51)\nexcept ZeroDivisionError:\n"
            "    raise SystemExit(52)\n"
        )

    def test_else_runs_only_when_the_body_did_not_raise(self):
        self._run(
            "total = 0\ntry:\n    total += 1\nexcept ValueError:\n    total += 10\n"
            "else:\n    total += 100\nraise SystemExit(total)\n"
        )

    def test_the_else_body_is_not_covered_by_this_try(self):
        self._run(
            'try:\n    x = 1\nexcept ValueError:\n    x = 2\nelse:\n'
            '    raise ValueError("v")\nraise SystemExit(3)\n'
        )

    def test_finally_runs_on_every_path(self):
        for source in (
            "x = 0\ntry:\n    x = 1\nfinally:\n    x = x + 6\nraise SystemExit(x)\n",
            'x = 0\ntry:\n    raise ValueError("v")\nexcept ValueError:\n'
            "    x = 1\nfinally:\n    x = x + 10\nraise SystemExit(x)\n",
            'try:\n    try:\n        raise ValueError("v")\n    finally:\n'
            '        print("cleanup")\nexcept ValueError:\n    raise SystemExit(41)\n',
        ):
            self._run(source)

    def test_finally_runs_when_a_clause_itself_raises(self):
        self._run(
            'x = 0\ntry:\n    try:\n        raise ValueError("v")\n'
            '    except ValueError:\n        raise KeyError("k")\n'
            "    finally:\n        x = 5\nexcept KeyError:\n"
            "    raise SystemExit(x + 30)\n"
        )

    def test_bare_raise_re_raises(self):
        self._run(
            'try:\n    try:\n        raise ValueError("v")\n    except ValueError:\n'
            "        raise\nexcept ValueError:\n    raise SystemExit(37)\n"
        )

    def test_output_order_across_body_clause_and_finally(self):
        self._run(
            'try:\n    print("body")\n    raise ValueError("v")\n'
            'except ValueError:\n    print("handler")\nfinally:\n'
            '    print("finally")\nprint("after")\n'
        )

    def test_a_try_inside_a_loop_runs_once_per_iteration(self):
        self._run(
            "for i in range(0, 3):\n    try:\n        if i == 1:\n"
            '            raise ValueError("v")\n        print("ok")\n'
            '    except ValueError:\n        print("caught")\n'
            '    finally:\n        print("done")\n'
        )

    def test_a_failed_bounds_check_is_catchable(self):
        self._run(
            "xs = [1, 2, 3]\ni = 0\nwhile i < 3:\n    i += 1\n"
            "try:\n    v = xs[i]\nexcept IndexError:\n    v = 55\n"
            "raise SystemExit(v)\n"
        )

    def test_a_missing_dict_key_is_catchable(self):
        self._run(
            "d = {1: 10}\ntry:\n    v = d[2]\nexcept KeyError:\n    v = 44\n"
            "raise SystemExit(v)\n"
        )

    def test_binding_the_exception_to_a_name_is_rejected(self):
        self._reject(
            'try:\n    raise ValueError("v")\nexcept ValueError as e:\n'
            "    raise SystemExit(1)\n",
            "cannot bind the exception to a name",
        )

    def test_leaving_a_finally_by_break_is_rejected(self):
        # The finally body is emitted on each path out; a break has no path to
        # emit it on, so refuse rather than skip the cleanup.
        self._reject(
            "i = 0\nwhile i < 3:\n    try:\n        break\n    finally:\n"
            "        i += 1\nraise SystemExit(i)\n",
            "cannot leave a try that has a finally",
        )

    def test_a_class_outside_the_builtin_hierarchy_is_rejected(self):
        # Matching a clause is a build-time question about class names, so a
        # name the hierarchy does not know cannot be answered.
        self._reject("raise Boom()\n", "builtin exception classes")
        self._reject(
            'try:\n    raise ValueError("v")\nexcept Boom:\n    pass\n',
            "builtin exception classes",
        )


class RuntimeIntegerPrintingTests(unittest.TestCase):
    """print() of a value that is only known at run time.

    Digits come out least-significant first but have to be written in the other
    order, so the length is counted first and the digits filled in backwards.
    Every case compares the binary's stdout against CPython's.
    """

    def _run(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "p.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "p.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.stdout, reference.stdout)
            self.assertEqual(native.returncode, reference.returncode)

    def test_a_positive_value(self):
        self._run("n = 0\nfor i in range(1, 6):\n    n += i\nprint(n)\n")

    def test_zero_prints_one_digit(self):
        # Peeling digits off zero produces none, so it needs its own answer.
        self._run("n = 5\nfor i in range(0, 1):\n    n -= 5\nprint(n)\n")

    def test_a_negative_value(self):
        self._run("n = 0\nfor i in range(1, 6):\n    n -= i\nprint(n)\n")

    def test_the_widest_values(self):
        for source in (
            "n = 1\nfor i in range(0, 19):\n    n = n * 3\nprint(n)\n",
            "n = 9223372036854775806\nfor i in range(0, 1):\n    n += 1\nprint(n)\n",
            # The smallest signed 64-bit value has no positive counterpart, so
            # the usual negate-then-peel loop cannot produce it.
            "n = -9223372036854775807\nfor i in range(0, 1):\n    n -= 1\nprint(n)\n",
        ):
            self._run(source)

    def test_several_arguments_with_separators(self):
        self._run(
            'a = 0\nfor i in range(1, 4):\n    a += i\nb = ""\nb = b + "items"\n'
            'print("total", a, b, -a)\n'
        )

    def test_printing_in_a_loop(self):
        self._run("i = 0\nwhile i < 12:\n    print(i * i - 30)\n    i += 1\n")

    def test_a_dict_length_is_printable(self):
        self._run(
            'counts: dict[str, int] = {}\ncounts["a"] = 1\ncounts["b"] = 2\n'
            'print("names:", len(counts))\n'
        )

    def test_a_runtime_float_prints_like_cpython(self):
        # Not a fixed number of digits: the shortest decimal that reads back as
        # the same double. 0.1 + 0.1 + 0.1 needs all 17, and its last digit is
        # an exact tie that CPython breaks toward the even digit.
        self._run("x = 0.0\nfor i in range(0, 3):\n    x += 0.1\nprint(x)\n")

    def test_float_layout_matches_cpython_at_the_format_boundaries(self):
        # CPython switches to exponential when the point sits more than four
        # places left of the digits or past the sixteenth place.
        self._run(
            "xs = [1e15, 1e16, 1e17, 1e-4, 1e-5, 0.0001, 0.00001, 1e22, 1e23]\n"
            "i = 0\nwhile i < 9:\n    print(xs[i])\n    i += 1\n"
        )

    def test_float_extremes_and_special_values(self):
        self._run(
            "xs = [5e-324, 1e-308, 2.2250738585072014e-308, 1e308,\n"
            "      1.7976931348623157e308, -1.7976931348623157e308,\n"
            "      0.0, -0.0, 1.0, -1.0, 0.5, 1/3, 3.141592653589793]\n"
            "i = 0\nwhile i < 13:\n    print(xs[i])\n    i += 1\n"
        )

    def test_runtime_infinities_and_nan(self):
        self._run(
            "big = 0.0\nfor i in range(0, 1):\n    big += 1e308\n"
            "huge = big * 10.0\n"
            "print(huge, -huge, huge - huge)\n"
        )

    def test_several_floats_in_one_call_do_not_share_a_buffer_wrongly(self):
        # The rendering scratch is reused, so each argument has to be written
        # out before the next one is rendered.
        self._run(
            "xs = [0.1, 2.5, -3.75, 1e100]\ni = 0\n"
            "while i < 4:\n    print(xs[i], xs[i] * 2.0, i)\n    i += 1\n"
        )


class FormattedStringTests(unittest.TestCase):
    """f-strings whose fields are only known at run time."""

    def _run(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "f.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "f.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.stdout, reference.stdout)
            self.assertEqual(native.returncode, reference.returncode)

    def _reject(self, source: str, expected: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "f.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "f.bin", "darwin-arm64", clean=True)
            self.assertIn(expected, str(caught.exception))

    def test_every_renderable_kind_in_one_string(self):
        self._run(
            'a = 0\nx = 0.0\ns = ""\n'
            "for i in range(0, 3):\n    a += 1\n    x += 0.1\n"
            's = s + "w\u00f6rld"\n'
            'print(f"int {a}, float {x}, str {s}, bool {a > 2}, done")\n'
        )

    def test_an_f_string_is_a_string(self):
        self._run(
            "a = 0\nfor i in range(0, 3):\n    a += 1\n"
            'label = f"item-{a}"\nprint(label, len(label))\n'
        )

    def test_adjacent_fields_and_empty_strings(self):
        self._run(
            "a = 0\nfor i in range(0, 3):\n    a += 1\n"
            'print(f"{a}{a}{a}")\nprint(f"")\nprint(f"no fields here")\n'
        )

    def test_the_same_float_field_repeated(self):
        # Float rendering hands back scratch that the next float overwrites, so
        # each field has to be copied out before the next one is rendered.
        self._run(
            "x = 0.0\nfor i in range(0, 3):\n    x += 0.1\n"
            'print(f"{x} and {x} and {x}")\n'
        )

    # A specifier is compared against CPython in both of the two paths a
    # program can take. The constant folder renders an f-string whose fields
    # are all known at build time without ever reaching the runtime renderers,
    # so a field that folds and the same field that does not are separate code
    # and have to be checked separately.
    _RUNTIME_PRELUDE = (
        "a = 0\nx = 0.0\ns = \"\"\n"
        "for i in range(0, 7):\n    a += 1\n    x += 0.5\n    s = s + \"w\u00f6\"\n"
        "m = 0 - a\ny = 0.0 - x\n"
    )

    def _both_paths(self, body: str) -> None:
        self._run(self._RUNTIME_PRELUDE + 'print(f"' + body + '")\n')
        self._run(
            'a = 7\nx = 3.5\ns = "w\u00f6w\u00f6w\u00f6w\u00f6w\u00f6w\u00f6w\u00f6"\n'
            "m = -7\ny = -3.5\n" + 'print(f"' + body + '")\n'
        )

    def test_width_fill_and_alignment(self):
        self._both_paths(
            "[{a:5}][{a:<5}][{a:>5}][{a:^5}][{a:*>5}][{a:^6}][{a:*<7}][{a:*^7}]"
        )
        self._both_paths("[{s:10}][{s:<20}][{s:>20}][{s:^21}][{s:\u00e9^24}][{s:2}]")
        self._both_paths("[{x:12}][{x:<12}][{x:^13}][{y:12}][{y:0=12}][{y:*>12}]")

    def test_a_bool_with_a_specifier_formats_as_an_integer(self):
        # An empty specifier keeps True and False; anything else is the int.
        self._both_paths("[{a > 3}][{a > 3:5}][{a > 3:d}][{a > 3:05}][{a > 3:.2f}]")

    def test_integer_sign_and_zero_padding(self):
        self._both_paths(
            "[{a:05d}][{m:05d}][{a:+d}][{m:+d}][{a: d}][{m: d}][{a:d}][{m:08d}]"
        )
        # '0>6' and '06' differ on a negative: an explicit alignment keeps the
        # sign inside the padding, a bare zero flag puts the padding after it.
        self._both_paths("[{m:0>6d}][{m:06d}][{m:<06d}][{m:=6d}][{a:+06d}][{m:*=7d}]")

    def test_thousands_separator(self):
        self._both_paths("[{a:,}][{a:,d}][{m:,d}][{a:+,d}][{a * 1000000:,d}]")

    def test_fixed_point_matches_cpython(self):
        # 2.675 is really 2.67499999999999982..., so half-to-even on the exact
        # binary value gives 2.67 and any shortcut off the repr gives 2.68.
        self._run(
            "xs = [2.675, 0.5, 1.5, 2.5, 0.125, 0.35, 0.45, 0.55, 0.75, 9.99,\n"
            "      0.006, 0.005, 0.0001, 1e16, 1e17, -2.5, -0.125, 5e-324]\n"
            "for k in range(0, 18):\n"
            "    v = xs[k]\n"
            '    print(f"{v:.0f}|{v:.1f}|{v:.2f}|{v:.3f}|{v:8.3f}|{v:+.2f}|{v:08.2f}")\n'
        )
        self._run(
            "print(f'{2.675:.2f} {0.5:.0f} {1.5:.0f} {2.5:.0f} {0.125:.2f}"
            " {1e16:.2f} {5e-324:.2f} {-0.0:.2f}')\n"
        )
        self._both_paths("[{x:.0f}][{x:.7f}][{y:+.3f}][{y:012.4f}][{a:.2f}][{m:.2f}]")

    def test_signed_zero_and_infinities(self):
        self._run(
            "z = 0.0\nfor i in range(0, 1):\n    z -= 0.0\n"
            "big = 1e308\ninf = big * 10.0\nnan = inf - inf\n"
            'print(f"{z:.2f}|{z:.0f}|{z:06.2f}|{z:+.2f}")\n'
            'print(f"{inf:.2f}|{inf:08.2f}|{-inf:+10.2f}|{nan:.2f}|{nan:+08.2f}")\n'
            'print(f"{inf}|{inf:8}|{-inf:08}|{nan:08}|{nan:>8}")\n'
        )

    def test_the_widest_and_narrowest_magnitudes(self):
        # 1e308 with two decimals is 312 characters; the digit and text
        # buffers are written without a bound check.
        self._run(
            "v = 1e308\nfor i in range(0, 1):\n    v = v * 1.0\n"
            'print(f"{v:.2f}")\nprint(f"{-v:.100f}")\n'
            "u = 5e-324\nfor i in range(0, 1):\n    u = u * 1.0\n"
            'print(f"{u:.100f}|{u:.0f}|{u:.1f}")\n'
        )

    def test_the_integer_extremes(self):
        self._run(
            "k = -9223372036854775807\nfor i in range(0, 1):\n    k -= 1\n"
            'print(f"{k:d}|{k:025d}|{k:.2f}|{k:,d}")\n'
            "j = 9223372036854775807\nfor i in range(0, 1):\n    j -= 0\n"
            'print(f"{j:d}|{j:+,d}|{j:.2f}")\n'
        )

    def test_conversions(self):
        # !r, !s, and !a all give str() on a number, and the field then
        # formats under string rules - which left-align by default.
        self._both_paths("[{a!r}][{a!r:6}][{x!s:>9}][{x!a:*^11}][{s!s:14}]")

    def test_unsupported_specifiers_are_rejected(self):
        prelude = self._RUNTIME_PRELUDE
        for body, expected in (
            ("{a:5e}", "format type 'e' is not supported"),
            ("{a:5g}", "format type 'g' is not supported"),
            ("{a:x}", "format type 'x' is not supported"),
            ("{a:b}", "format type 'b' is not supported"),
            ("{a:o}", "format type 'o' is not supported"),
            ("{a:n}", "format type 'n' is not supported"),
            ("{a:%}", "format type '%' is not supported"),
            ("{a:#x}", "the '#' flag is not supported"),
            ("{a:z.2f}", "the 'z' flag is not supported"),
            ("{a:_d}", "the '_' separator is not supported"),
            ("{x:d}", "format type 'd' is not supported for a float"),
            ("{a:5s}", "format type 's' is not supported for an int"),
            ("{s:5d}", "format type 'd' is not supported for a str"),
            ("{s:.2}", "a precision truncates a string"),
            ("{s:+5}", "a sign is not allowed in a string format specifier"),
            ("{s:=5}", "'=' alignment, and the zero flag that implies it"),
            ("{s:05}", "'=' alignment, and the zero flag that implies it"),
            ("{s:,}", "a thousands separator is not allowed in a string"),
            ("{x:,.2f}", "a thousands separator is only supported for an integer"),
            ("{a:0,d}", "a thousands separator combined with zero or '=' padding"),
            ("{a:.2}", "a precision is only supported with format type 'f'"),
            ("{x:.2}", "a precision is only supported with format type 'f'"),
            ("{a:.200f}", "a precision above 100 is not supported"),
            ("{a:2000}", "a width above 1000 is not supported"),
            ("{a:{a}}", "format specifier has to be literal text"),
            ("{s!r}", "!r and !a on a string add quotes"),
            ("{s!a}", "!r and !a on a string add quotes"),
        ):
            with self.subTest(body=body):
                self._reject(prelude + 'print(f"' + body + '")\n', expected)
                # The same field folded: the constant path must refuse it too,
                # or a program would compile only because its values folded.
                self._reject(
                    'a = 7\nx = 3.5\ns = "q"\nprint(f"' + body + '")\n', expected
                )

    def test_every_rejection_names_what_is_supported(self):
        self._reject(
            'print(f"{1.5:5g}")\n',
            "[[fill]align][sign][0][width][,][.precision][type]",
        )
        self._reject('print(f"{1.5:5g}")\n', "type one of d, f, s or omitted")


class BooleanRenderingTests(unittest.TestCase):
    """A bool prints as True or False, not as 1 or 0.

    The native subset keeps a bool in an integer slot, which is right for
    arithmetic and wrong for printing. Nothing tells them apart at run time, so
    the question is answered from the source.
    """

    def _run(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "b.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "b.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.stdout, reference.stdout)

    def test_comparisons_and_negations(self):
        self._run(
            "a = 0\nfor i in range(0, 3):\n    a += 1\nflag = a > 2\n"
            "print(a > 2, a < 2, not flag, flag)\nprint(a, a * 2)\n"
        )

    def test_a_bool_used_arithmetically_stops_being_one(self):
        # `n = a > 1` then `n += 1` makes n a number again, and 2 is not True.
        self._run(
            "a = 0\nfor i in range(0, 3):\n    a += 1\n"
            "flag = a > 2\ncopied = flag\nn = a > 1\nn += 1\n"
            "print(copied, n)\n"
        )


class WithStatementTests(unittest.TestCase):
    """`with` over a native class, resolved at build time.

    There is no run-time protocol lookup, so `__enter__` and `__exit__` are
    found on the class and inlined. `__exit__` runs on the way out whether the
    body finished or raised, which is the same problem `finally` solves and is
    emitted the same way.
    """

    HEADER = (
        "class G:\n"
        "    def __init__(self, n):\n        self.n = n\n"
        "    def __enter__(self):\n        return self\n"
        "    def __exit__(self, a, b, c):\n"
        '        print("close", self.n)\n\n'
    )

    def _run(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "w.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "w.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.stdout, reference.stdout)
            self.assertEqual(native.returncode, reference.returncode)

    def _reject(self, source: str, expected: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "w.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "w.bin", "darwin-arm64", clean=True)
            self.assertIn(expected, str(caught.exception))

    def test_enter_binds_and_exit_runs(self):
        self._run(
            self.HEADER
            + 'with G(7) as g:\n    print("inside", g.n)\nprint("after")\n'
        )

    def test_a_manager_that_binds_nothing(self):
        self._run(
            "class Q:\n    def __init__(self, n):\n        self.n = n\n"
            '    def __enter__(self):\n        print("enter")\n'
            '    def __exit__(self, a, b, c):\n        print("exit")\n\n'
            'with Q(1):\n    print("body")\n'
        )

    def test_nested_managers_close_in_reverse(self):
        self._run(self.HEADER + 'with G(1) as x, G(2) as y:\n    print(x.n, y.n)\n')

    def test_exit_runs_when_the_body_raises(self):
        self._run(
            self.HEADER
            + "try:\n    with G(3) as g:\n"
            '        print("before")\n        raise ValueError("boom")\n'
            'except ValueError:\n    print("caught")\n'
        )

    def test_a_loop_inside_the_body_may_break(self):
        self._run(
            self.HEADER
            + "with G(0) as g:\n    i = 0\n    while i < 5:\n"
            "        if i == 2:\n            break\n        print(i)\n        i += 1\n"
        )

    def test_breaking_out_of_a_with_is_rejected(self):
        self._reject(
            self.HEADER
            + "i = 0\nwhile i < 3:\n    with G(i) as g:\n        break\n    i += 1\n",
            "cannot leave a try that has a finally",
        )

    def test_an_exit_that_inspects_the_exception_is_rejected(self):
        self._reject(
            "class G:\n    def __init__(self, n):\n        self.n = n\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, kind, value, trace):\n        print(kind)\n\n"
            'with G(1) as g:\n    print("x")\n',
            "no exception object to pass",
        )

    def test_a_non_object_manager_is_rejected(self):
        self._reject(
            'with 5 as g:\n    print("x")\n', "needs a native object"
        )


class LoopElseTests(unittest.TestCase):
    """`for ... else` and `while ... else`: the else runs unless a break skipped it.

    There is no run-time flag. A loop with an else gets a second exit label
    past the else body, and `break` targets that one, so the skip costs a jump
    and nothing else.
    """

    RUNTIME_N = "n = 0\nfor i in range(0, 3):\n    n += 1\n"

    def _run(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "e.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "e.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.stdout, reference.stdout)
            self.assertEqual(native.returncode, reference.returncode)

    def _reject(self, source: str, expected: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "e.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "e.bin", "darwin-arm64", clean=True)
            self.assertIn(expected, str(caught.exception))

    def test_a_range_loop_else_runs_only_without_a_break(self):
        self._run(
            self.RUNTIME_N
            + "for i in range(0, n):\n    if i == 99:\n        break\n"
            + 'else:\n    print("fell through")\n'
            + "for i in range(0, n):\n    if i == 1:\n        break\n"
            + 'else:\n    print("not reached")\n'
            + 'print("end")\n'
        )

    def test_a_while_else(self):
        self._run(
            self.RUNTIME_N
            + "k = n\nwhile k > 0:\n    k -= 1\n    if k == 1:\n        break\n"
            + 'else:\n    print("drained")\n'
            + "m = n\nwhile m > 0:\n    m -= 1\n"
            + 'else:\n    print("emptied", m)\n'
            + 'print("end", k)\n'
        )

    def test_a_break_in_a_nested_loop_does_not_skip_the_outer_else(self):
        self._run(
            self.RUNTIME_N
            + "for i in range(0, n):\n"
            + "    for j in range(0, n):\n        if j == 1:\n            break\n"
            + '    print("body", i)\n'
            + 'else:\n    print("outer else")\n'
        )

    def test_a_break_in_an_inner_else_breaks_the_outer_loop(self):
        # The target stacks are popped before the else body is emitted, so a
        # break written there belongs to the enclosing loop, as in CPython.
        self._run(
            self.RUNTIME_N
            + "for i in range(0, n):\n"
            + "    for j in range(0, i):\n        if j == 5:\n            break\n"
            + '    else:\n        print("inner else", i)\n'
            + "        if i == 1:\n            break\n"
            + '    print("body", i)\n'
            + 'else:\n    print("outer else")\n'
            + 'print("done")\n'
        )

    def test_a_continue_in_an_else_body_continues_the_outer_loop(self):
        self._run(
            self.RUNTIME_N
            + "for i in range(0, n):\n"
            + "    for j in range(0, 1):\n        pass\n"
            + "    else:\n        continue\n"
            + '    print("unreachable", i)\n'
            + 'print("done")\n'
        )

    def test_a_name_assigned_in_an_else_body_is_not_folded_on_the_break_path(self):
        # The label after the else is reached from two paths, so a name the
        # else body assigns cannot stay a build-time constant.
        self._run(
            self.RUNTIME_N
            + "a = 1\nb = 2.5\ns = 'no'\n"
            + "for i in range(0, n):\n    if i == 1:\n        break\n"
            + "else:\n    a = 9\n    b = 9.5\n    s = 'yes'\n"
            + "print(a * 2, b + 0.5, s)\n"
            + 'if a == 9:\n    print("nine")\nelse:\n    print("not nine")\n'
        )

    def test_a_for_else_over_a_list_reversed_and_enumerate(self):
        self._run(
            self.RUNTIME_N
            + "xs = [1, 2]\nxs.append(n)\n"
            + "for v in xs:\n    if v == 99:\n        break\n"
            + 'else:\n    print("no 99")\n'
            + "for v in reversed(xs):\n    if v == 99:\n        break\n"
            + 'else:\n    print("rev no 99")\n'
            + "for k, v in enumerate(xs):\n    if v == 2:\n        break\n"
            + 'else:\n    print("not reached")\n'
            + 'print("end")\n'
        )

    def test_a_for_else_over_a_dict(self):
        self._run(
            self.RUNTIME_N
            + "d = {}\nd[1] = 10\nd[2] = n\n"
            + "for k, v in d.items():\n    if v == 99:\n        break\n"
            + 'else:\n    print("no 99")\n'
            + 'print("end")\n'
        )

    def test_a_name_first_bound_in_an_else_body_is_refused_after_a_break(self):
        # The break path jumped over the else, so the slot holds nothing there.
        self._reject(
            self.RUNTIME_N
            + "for i in range(0, n):\n    if i == 99:\n        break\n"
            + "else:\n    later = 5\nprint(later)\n",
            "may be unbound here",
        )

    def test_a_break_leaving_a_cleanup_scope_is_still_refused(self):
        # The else body adds a label, not a jump-stack entry, so a `with`
        # opened inside the loop still refuses a break out of it.
        self._reject(
            "class G:\n    def __enter__(self):\n        return self\n"
            "    def __exit__(self, a, b, c):\n        print('close')\n\n"
            + self.RUNTIME_N
            + "for i in range(0, n):\n    with G() as g:\n        break\n"
            + 'else:\n    print("else")\n',
            "cannot leave a try that has a finally",
        )


class NativeSetTests(unittest.TestCase):
    """Sets, which share the dict's open-addressing table.

    A set is the same block with the value word never written and the
    insertion-order field left 0. Every case compares stdout and exit status
    with CPython rather than a hand-computed number, so a wrong expectation
    cannot hide a wrong answer.
    """

    RUNTIME_N = "n = 0\nfor i in range(0, 3):\n    n += 1\n"

    def _run(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "s.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "s.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            reference = subprocess.run(
                [sys.executable, str(entry)], capture_output=True
            )
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.stdout, reference.stdout)
            self.assertEqual(native.returncode, reference.returncode)

    def test_a_literal_drops_duplicates(self):
        self._run("s = {1, 1, 2, 9, 17}\nprint(len(s))\n")

    def test_keys_that_share_a_home_slot(self):
        # 1, 9 and 17 are congruent modulo the initial capacity of 8, so each
        # is found only by probing past the others.
        self._run(
            self.RUNTIME_N
            + "s = {1, 9, 17}\nprint(9 in s, 17 in s, 8 in s, n * 3 in s)\n"
        )

    def test_discarding_the_middle_of_a_probe_chain(self):
        # The case a tombstone-free blank would get wrong: removing 9 would
        # leave a hole that stops the probe for 17 before it reaches it.
        self._run(
            "n = 0\nfor i in range(0, 9):\n    n += 1\n"
            "s = {1, 9, 17}\ns.discard(n)\n"
            "print(len(s), 1 in s, 9 in s, 17 in s)\n"
        )

    def test_discarding_something_absent_changes_nothing(self):
        self._run("s = {5}\ns.discard(6)\nprint(len(s))\ns.discard(5)\nprint(len(s), 5 in s)\n")

    def test_adding_after_discarding_everything(self):
        self._run(
            's: set[str] = set()\ns.add("aa")\ns.add("bb")\ns.discard("aa")\n'
            'print(len(s), "aa" in s, "bb" in s)\ns.discard("bb")\n'
            's.add("cc")\nprint(len(s), "cc" in s)\n'
        )

    def test_growth_and_rehashing_of_integer_elements(self):
        self._run(
            "s = set()\ni = 0\nwhile i < 300:\n    s.add(i * 7)\n    i += 1\n"
            "print(len(s), 700 in s, 701 in s)\n"
        )

    def test_growth_and_rehashing_of_string_elements(self):
        self._run(
            's: set[str] = set()\ni = 0\nt = ""\nwhile i < 300:\n'
            '    t = t + "a"\n    s.add(t)\n    i += 1\n'
            'print(len(s), "aaa" in s, "b" in s)\n'
        )

    def test_discarding_half_a_grown_table(self):
        self._run(
            "s = set()\ni = 0\nwhile i < 40:\n    s.add(i)\n    i += 1\n"
            "i = 0\nwhile i < 20:\n    s.discard(i * 2)\n    i += 1\n"
            "print(len(s), 0 in s, 1 in s, 38 in s, 39 in s)\n"
        )

    def test_an_empty_dict_and_an_empty_set_are_different_things(self):
        self._run("d = {}\ns = set()\ns.add(1)\nprint(len(d), len(s))\n")

    def test_negative_and_extreme_elements(self):
        # The home slot is key & mask on a signed value, so a negative element
        # has to land somewhere the probe can find it again.
        self._run(
            self.RUNTIME_N
            + "s = {-1, -3, 9223372036854775807}\n"
            "print(-n in s, -1 in s, 0 in s, 9223372036854775807 in s)\n"
        )

    def test_not_in(self):
        self._run("s = {1, 2}\nprint(2 not in s, 3 not in s)\n")

    def test_union_intersection_and_difference(self):
        self._run(
            "a = {1, 2, 3}\nb = {2, 3, 4}\n"
            "c = a | b\nprint(len(c), 4 in c)\n"
            "c = a & b\nprint(len(c), 1 in c, 2 in c)\n"
            "c = a - b\nprint(len(c), 1 in c, 2 in c)\n"
            "a |= b\nprint(len(a))\n"
        )

    def test_operators_over_string_elements(self):
        self._run(
            'a: set[str] = set()\na.add("x")\na.add("y")\n'
            'b: set[str] = set()\nb.add("y")\nb.add("z")\n'
            'c = a | b\nprint(len(c), "x" in c, "z" in c)\n'
            'c = a & b\nprint(len(c), "y" in c, "x" in c)\n'
            'c = a - b\nprint(len(c), "x" in c, "y" in c)\n'
        )

    def test_augmented_operators_chain(self):
        self._run(
            "a = {1, 2}\nb = {2, 3}\na |= b\na &= b\n"
            "print(len(a), 2 in a, 3 in a, 1 in a)\na -= b\nprint(len(a))\n"
        )

    def test_union_with_itself_is_a_copy_that_does_not_share(self):
        self._run(
            "a = {1, 2}\nb = a | a\nb.add(3)\n"
            "print(len(a), len(b), 3 in a, 3 in b)\n"
        )

    def test_an_intersection_can_be_empty(self):
        self._run("a = {1, 2}\nb = {3, 4}\nc = a & b\nprint(len(c), 1 in c)\n")

    def test_bools_are_a_set_of_their_own(self):
        self._run(
            self.RUNTIME_N
            + "s = {True, False}\n"
            "print(len(s), True in s, False in s, (n > 2) in s)\n"
        )

    def test_remove_raises_when_the_element_is_absent(self):
        self._run("s = {1, 2, 3}\ns.remove(2)\nprint(len(s), 2 in s)\ns.remove(7)\n")

    def test_a_module_level_set_read_inside_an_inlined_function(self):
        self._run(
            "s = {4, 1, 7}\ndef hits() -> int:\n    return len(s) + (7 in s)\n"
            "print(hits())\n"
        )

    def test_adding_inside_an_inlined_function(self):
        self._run(
            "s = {1, 2}\ndef f() -> int:\n    s.add(3)\n    return len(s)\n"
            "print(f())\n"
        )

    def test_sorted_is_the_one_defined_order(self):
        self._run(
            "s = {5, 1, 9, 1, 3}\nfor v in sorted(s):\n    print(v)\n"
            "xs = sorted(s, reverse=True)\nprint(len(xs), xs[0], xs[3])\n"
        )

    def test_sorted_of_an_empty_set(self):
        self._run("s = set()\nxs = sorted(s)\nprint(len(xs))\n")


class NativeSetRefusalTests(unittest.TestCase):
    """What a set refuses, and why.

    Iteration is refused outright: CPython's set order is unspecified, is not
    insertion order, and for strings is randomized per process, so no order
    produced here could match it for every input.
    """

    def _reject(self, source: str, expected: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "s.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "s.bin", "darwin-arm64", clean=True)
            self.assertIn(expected, str(caught.exception))

    def test_iterating_a_set_is_refused(self):
        self._reject(
            "s = {1, 2}\nfor v in s:\n    print(v)\n",
            "a native set cannot be iterated",
        )

    def test_iterating_a_set_literal_is_refused(self):
        self._reject("for v in {1, 2}:\n    print(v)\n", "a native set cannot be iterated")

    def test_every_way_the_order_could_escape_a_loop_is_refused(self):
        # Why the ban is on iteration and not on printing inside it: each of
        # these carries the order out of the loop with no print in the body,
        # so a print-only ban would pass them through as wrong answers.
        for body in (
            "xs = []\nfor v in s:\n    xs.append(v)\nprint(len(xs))\n",
            "for v in s:\n    print(v)\n    break\n",
            't = ""\nfor v in s:\n    t = t + str(v)\nprint(t)\n',
            "def first() -> int:\n    for v in s:\n        return v\n"
            "    return 0\nprint(first())\n",
        ):
            with self.subTest(body=body):
                self._reject("s = {5, 3}\n" + body, "a native set cannot be iterated")

    def test_printing_a_set_is_refused(self):
        self._reject("s = {1, 2}\nprint(s)\n", "cannot render a runtime set:int")

    def test_a_bool_beside_a_number_is_refused(self):
        # CPython makes {True, 1} one element because True == 1; here the two
        # are the same 64 bits, so one of them would have to print wrongly.
        self._reject("s = {True, 1}\nprint(len(s))\n", "holds bools already")
        self._reject("s = set()\ns.add(True)\ns.add(1)\n", "holds bools already")

    def test_rebinding_a_set_name_to_the_other_bool_kind_is_refused(self):
        # A false refusal, and the deliberate one: the answer belongs to the
        # name, and letting a rebinding reset it would let `if c: s = {1}`
        # leave the build-time answer disagreeing with the slot, which
        # sorted() would then print as 1 and 0 instead of True and False.
        self._reject(
            "s = {True}\ns = {1}\nprint(1 in s)\n", "holds bools already"
        )

    def test_a_set_in_an_integer_context_is_refused(self):
        self._reject("s = {1, 2}\nx = s + 1\nprint(x)\n", "set variable 's' needs len()")
        self._reject("s = {1, 2}\nif s:\n    print(1)\n", "set variable 's' needs len()")
        self._reject(
            "s = {1, 2}\nt = {1, 2}\nprint(s == t)\n", "set variable 's' needs len()"
        )

    def test_a_dict_in_an_integer_context_is_refused(self):
        # The same hole, and it was open for dicts too: `d + 1` answered with
        # an arena address and `if d:` was true for an empty table.
        self._reject("d = {1: 1}\nx = d + 1\nprint(x)\n", "dict variable 'd' needs len()")
        self._reject(
            "d: dict[int, int] = {}\nif d:\n    print(1)\n",
            "dict variable 'd' needs len()",
        )

    def test_a_second_name_for_the_same_set_is_refused(self):
        self._reject("s = {1, 2}\nt = s\nprint(len(t))\n", "cannot be another name for")

    def test_unsupported_methods_are_named(self):
        self._reject(
            "s = {1, 2}\ns.pop()\nprint(len(s))\n",
            "native sets support add(), discard() and remove()",
        )

    def test_mixed_element_kinds_are_refused(self):
        self._reject('s = {1, "a"}\nprint(len(s))\n', "so every element must be int")

    def test_set_takes_no_arguments(self):
        self._reject("s = set([1, 2])\nprint(len(s))\n", "native set() takes no arguments")

    def test_combining_sets_of_different_kinds_is_refused(self):
        self._reject(
            'a = {1, 2}\nb: set[str] = set()\nc = a | b\nprint(len(c))\n',
            "hold different kinds",
        )

    def test_an_operator_needs_a_set_on_both_sides(self):
        self._reject(
            "a = {1, 2}\nc = a | 3\nprint(len(c))\n", "a set on both sides"
        )

    def test_membership_over_a_set_literal_is_refused(self):
        self._reject(
            "n = 0\nfor i in range(0, 2):\n    n += 1\nprint(n in {1, 2, 5})\n",
            "`in` over a set literal is not supported",
        )

    def test_sorted_of_a_string_set_is_refused(self):
        self._reject(
            's: set[str] = set()\ns.add("a")\nxs = sorted(s)\nprint(len(xs))\n',
            "native sorted() takes a set of integers",
        )

    def test_sorted_of_a_set_of_bools_is_refused(self):
        self._reject(
            "s = {True, False}\nxs = sorted(s)\nprint(len(xs))\n",
            "sets of integers, and this one holds bools",
        )

    def test_an_annotation_does_not_convert_a_literal(self):
        self._reject(
            "s: set[str] = {1, 2}\nprint(len(s))\n", "the annotation says set:str"
        )

    def test_an_unsupported_augmented_operator_is_named(self):
        self._reject(
            "a = {1, 2}\nb = {3}\na += b\nprint(len(a))\n",
            "native set augmented assignment supports |= &= -=",
        )
