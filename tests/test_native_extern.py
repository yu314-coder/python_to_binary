"""Slice 4: the adapter-ABI extern boundary.

These tests exercise the ONLY honest "library" path: declaring and calling a
genuine external native symbol resolved through real dyld binding. On
darwin-arm64 the emitted Mach-O is run NATIVELY and its exit code is compared to
the SAME source run under CPython (whose ``py2bin.cabi`` shim calls the same
libc symbols via ctypes). No C/C++/CUDA source is ever translated.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from py2bin.native import NativeCompileError, compile_native
from py2bin.native.compiler import _module_uses_extern
from py2bin.native.frontend import lower


_HOST_IS_DARWIN_ARM64 = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)
_REPO_SRC = str(Path(__file__).resolve().parent.parent / "src")


def _write(source: str) -> tuple[Path, Path]:
    directory = Path(tempfile.mkdtemp())
    entry = directory / "program.py"
    entry.write_text(source, encoding="utf-8")
    return entry, directory


def _cpython_exit(entry: Path) -> int:
    """Run the same source under CPython, using the real cabi shim."""

    result = subprocess.run(
        [sys.executable, str(entry)],
        env={"PYTHONPATH": _REPO_SRC, "PATH": "/usr/bin:/bin"},
    )
    return result.returncode


class NativeExternTests(unittest.TestCase):
    def _build_darwin(self, source: str) -> tuple[Path, Path]:
        entry, directory = _write(source)
        artifact = directory / "program.bin"
        compile_native(entry, artifact, "darwin-arm64", clean=True)
        self.assertEqual(
            artifact.read_bytes()[:4],
            b"\xcf\xfa\xed\xfe",
            "extern Mach-O has a broken header",
        )
        return entry, artifact

    def _match_cpython(self, source: str) -> int:
        entry, artifact = self._build_darwin(source)
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("native execution requires darwin-arm64")
        native = subprocess.run([str(artifact)]).returncode
        cpython = _cpython_exit(entry)
        self.assertEqual(
            native,
            cpython,
            f"native exit {native} != CPython exit {cpython}",
        )
        return native

    def test_abs_int_argument_matches_cpython(self):
        exit_code = self._match_cpython(
            "from py2bin.cabi import abs\n"
            "raise SystemExit(abs(-7))\n"
        )
        self.assertEqual(exit_code, 7)

    def test_strlen_pointer_argument_matches_cpython(self):
        exit_code = self._match_cpython(
            "from py2bin.cabi import strlen\n"
            'raise SystemExit(strlen("hello, native world"))\n'
        )
        self.assertEqual(exit_code, 19)

    def test_mixed_extern_and_arithmetic_matches_cpython(self):
        exit_code = self._match_cpython(
            "from py2bin.cabi import abs, strlen\n"
            'raise SystemExit(abs(-3) + strlen("abcd") * 2)\n'
        )
        self.assertEqual(exit_code, 11)

    def test_getpid_returns_positive_process_id(self):
        # getpid is genuinely nondeterministic across processes, so the exact
        # value cannot match a separate CPython run; instead confirm the real
        # dyld-bound call returns a valid (positive) pid.
        self._match_cpython(
            "from py2bin.cabi import getpid\n"
            "raise SystemExit(0 if getpid() > 0 else 1)\n"
        )

    def test_extern_call_through_helper_function(self):
        # An extern call nested inside a native helper function: the integer
        # argument threads through the function inliner into the ExternCall.
        exit_code = self._match_cpython(
            "from py2bin.cabi import abs\n"
            "def magnitude(x) -> int:\n"
            "    return abs(x)\n"
            "raise SystemExit(magnitude(-9))\n"
        )
        self.assertEqual(exit_code, 9)

    def test_emitted_macho_declares_real_dyld_import(self):
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("nm/dyld_info are darwin tools")
        _entry, artifact = self._build_darwin(
            "from py2bin.cabi import strlen\n"
            'raise SystemExit(strlen("abc"))\n'
        )
        symbols = subprocess.run(
            ["nm", "-mu", str(artifact)], capture_output=True, text=True
        ).stdout
        self.assertIn("_strlen", symbols)
        self.assertIn("libSystem", symbols)
        verified = subprocess.run(
            ["codesign", "--verify", "--verbose=2", str(artifact)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(verified.returncode, 0, verified.stderr)


class NativeLibraryHelperTests(unittest.TestCase):
    """Part (a): a pure-Python helper module compiles through the existing
    function-inlining path -- now covering the float value type -- and runs
    natively, matching CPython."""

    def test_helper_module_with_float_runs_natively(self):
        directory = Path(tempfile.mkdtemp())
        package = directory / "helpers"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "mathlib.py").write_text(
            "def scale(value: int) -> int:\n"
            "    total = float(value) * 2.5\n"
            "    return int(total)\n"
            "\n"
            "def sum_to(limit: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(limit + 1):\n"
            "        total += i\n"
            "    return total\n",
            encoding="utf-8",
        )
        entry = directory / "main.py"
        entry.write_text(
            "from helpers.mathlib import scale, sum_to\n"
            "raise SystemExit(scale(4) + sum_to(5))\n",
            encoding="utf-8",
        )
        artifact = directory / "main.bin"
        compile_native(
            entry,
            artifact,
            "darwin-arm64",
            clean=True,
            source_roots=(directory,),
        )
        self.assertEqual(artifact.read_bytes()[:4], b"\xcf\xfa\xed\xfe")
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("native execution requires darwin-arm64")
        native = subprocess.run([str(artifact)]).returncode
        cpython = subprocess.run(
            [sys.executable, str(entry)],
            env={"PYTHONPATH": str(directory), "PATH": "/usr/bin:/bin"},
        ).returncode
        self.assertEqual(native, cpython)
        self.assertEqual(native, 25)  # scale(4)=10 + sum_to(5)=15


class ExternRejectionTests(unittest.TestCase):
    def _compile_error(self, source: str, target: str = "darwin-arm64") -> str:
        entry, directory = _write(source)
        with self.assertRaises(NativeCompileError) as caught:
            compile_native(entry, directory / "out.bin", target, clean=True)
        return str(caught.exception)

    def test_unknown_symbol_is_rejected(self):
        message = self._compile_error(
            "from py2bin.cabi import frobnicate\n"
            "raise SystemExit(frobnicate())\n"
        )
        self.assertIn("not an available adapter-ABI symbol", message)

    def test_non_constant_cstring_is_rejected(self):
        message = self._compile_error(
            "from py2bin.cabi import strlen\n"
            "x = 5\n"
            "raise SystemExit(strlen(x))\n"
        )
        self.assertIn("compile-time string constant", message)

    def test_wrong_argument_count_is_rejected(self):
        message = self._compile_error(
            "from py2bin.cabi import getpid\n"
            "raise SystemExit(getpid(1))\n"
        )
        self.assertIn("expects 0 argument", message)

    def test_import_star_is_rejected(self):
        message = self._compile_error(
            "from py2bin.cabi import *\n"
            "raise SystemExit(0)\n"
        )
        self.assertIn("import *", message)

    def test_real_native_package_is_rejected(self):
        message = self._compile_error("import numpy\nraise SystemExit(0)\n")
        self.assertIn("not in the native subset", message)

    def test_extern_on_non_darwin_target_is_rejected(self):
        for target in ("linux-arm64", "linux-x86_64", "windows-x86_64"):
            message = self._compile_error(
                "from py2bin.cabi import abs\n"
                "raise SystemExit(abs(-1))\n",
                target=target,
            )
            self.assertIn("only supported for target 'darwin-arm64'", message)

    def test_module_uses_extern_detects_nested_calls(self):
        entry, _directory = _write(
            "from py2bin.cabi import abs\n"
            "raise SystemExit(abs(-4) + 1)\n"
        )
        module = lower(entry, entry.read_text(), ())
        self.assertTrue(_module_uses_extern(module))

    def test_module_uses_extern_false_without_externs(self):
        entry, _directory = _write("raise SystemExit(3)\n")
        module = lower(entry, entry.read_text(), ())
        self.assertFalse(_module_uses_extern(module))


if __name__ == "__main__":
    unittest.main()


class CPythonRuntimeExternTests(unittest.TestCase):
    """Binding the CPython runtime itself as an external native library.

    This is the foundation of the Nuitka-shaped tier: generated machine code
    drives an already-compiled interpreter through real dyld binding. No
    CPython source is translated, and no C toolchain is invoked.
    """

    def test_cpython_symbols_route_to_the_cpython_library(self):
        from py2bin.cabi import LIBSYSTEM, symbol_library

        self.assertEqual(symbol_library("getpid"), LIBSYSTEM)
        library = symbol_library("Py_Initialize")
        self.assertNotEqual(library, LIBSYSTEM)
        self.assertTrue(Path(library).is_file(), library)

    def test_embedding_program_matches_cpython_and_links_two_dylibs(self):
        source = (
            "from py2bin.cabi import Py_Initialize, PyRun_SimpleString, Py_Finalize\n"
            "Py_Initialize()\n"
            "PyRun_SimpleString(\"print('embedded')\")\n"
            "Py_Finalize()\n"
            "raise SystemExit(0)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "embed.py"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "embed.bin"
            compile_native(entry, artifact, "darwin-arm64", clean=True)
            self.assertEqual(artifact.read_bytes()[:4], b"\xcf\xfa\xed\xfe")
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            native = subprocess.run([str(artifact)], capture_output=True)
            self.assertEqual(native.returncode, 0, native.stderr)
            self.assertIn(b"embedded", native.stdout)


class ExternResultWidthTests(unittest.TestCase):
    """A 32-bit C result must be widened before it is used as a 64-bit value.

    AAPCS64 leaves bits 32-63 of the return register unspecified when the
    callee returns a 32-bit type. Without an explicit extension CPython's -1
    failure return reads as 4294967295, so `if (rc < 0)` never fires and a
    pending exception is silently swallowed.
    """

    def test_int_returning_symbols_are_sign_extended(self):
        from py2bin.native.arm64 import encode_darwin_extern
        from py2bin.native.ir import ExternCall, Module, Store

        module = Module([Store(0, ExternCall("PyRun_SimpleString", (), "i32"))], 4)
        code, _externs = encode_darwin_extern(module, 0x100004000)
        # sxtw x0, w0
        self.assertIn((0x93407C00).to_bytes(4, "little"), code)

    def test_word_returning_symbols_are_not_extended(self):
        from py2bin.native.arm64 import encode_darwin_extern
        from py2bin.native.ir import ExternCall, Module, Store

        module = Module([Store(0, ExternCall("strlen", (), "i64"))], 4)
        code, _externs = encode_darwin_extern(module, 0x100004000)
        self.assertNotIn((0x93407C00).to_bytes(4, "little"), code)

    def test_failing_capi_call_is_detected_like_cpython(self):
        source = (
            "extern void Py_Initialize(void);\n"
            "extern void Py_Finalize(void);\n"
            "extern int PyRun_SimpleString(const char *source);\n"
            "int main(void)\n"
            "{\n"
            "    Py_Initialize();\n"
            "    long long rc = PyRun_SimpleString(\"raise ValueError('boom')\");\n"
            "    Py_Finalize();\n"
            "    if (rc < 0) { return 7; }\n"
            "    return 9;\n"
            "}\n"
        )
        from py2bin.c_native import compile_c_native

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "rc.c"
            entry.write_text(source, encoding="utf-8")
            artifact = root / "rc.bin"
            compile_c_native(entry, artifact, target="darwin-arm64", clean=True)
            if not (
                platform.system() == "Darwin" and platform.machine() == "arm64"
            ):
                return
            run = subprocess.run([str(artifact)], capture_output=True)
            # 7 means the -1 was seen as negative; 9 means it was swallowed.
            self.assertEqual(run.returncode, 7)
