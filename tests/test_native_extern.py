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

    def test_extern_on_darwin_x86_64_is_accepted(self):
        """The second architecture whose dynamic-link adapter is real.

        It was refused until the x86-64 encoder learned to emit GOT reference
        sites and the Mach-O writer learned to patch them, which is the same
        pair of things the arm64 path has - spelled with one rip-relative
        displacement rather than an ADRP and an offset.
        """

        entry, directory = _write(
            "from py2bin.cabi import abs\n"
            "raise SystemExit(abs(-1))\n"
        )
        output = directory / "out.bin"
        compile_native(entry, output, "darwin-x86_64", clean=True)
        image = output.read_bytes()
        # A 64-bit little-endian Mach-O naming CPU_TYPE_X86_64.
        self.assertEqual(image[:4], b"\xcf\xfa\xed\xfe")
        self.assertEqual(
            int.from_bytes(image[4:8], "little"), 0x01000007
        )

    def test_extern_on_non_darwin_target_is_rejected(self):
        for target in ("linux-arm64", "linux-x86_64", "windows-x86_64"):
            message = self._compile_error(
                "from py2bin.cabi import abs\n"
                "raise SystemExit(abs(-1))\n",
                target=target,
            )
            self.assertIn("darwin-arm64, darwin-x86_64", message)

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
        code, _externs, _statics = encode_darwin_extern(module, 0x100004000)
        # sxtw x0, w0
        self.assertIn((0x93407C00).to_bytes(4, "little"), code)

    def test_word_returning_symbols_are_not_extended(self):
        from py2bin.native.arm64 import encode_darwin_extern
        from py2bin.native.ir import ExternCall, Module, Store

        module = Module([Store(0, ExternCall("strlen", (), "i64"))], 4)
        code, _externs, _statics = encode_darwin_extern(module, 0x100004000)
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


class ExternFloatArgumentTests(unittest.TestCase):
    """AAPCS64 numbers the integer and floating-point argument files apart.

    A compiled binary must put the first double in D0 and the first integer in
    X0 regardless of what order they appear in, so a call taking both -- C
    ``ldexp(double, int)`` is the smallest one -- is the test that fails loudly
    if the two counters are ever shared. Every case here is diffed against the
    SAME source under CPython, whose ``py2bin.cabi`` shim calls the identical
    libSystem function through ctypes.
    """

    def _match_cpython(self, source: str) -> str:
        entry, directory = _write(source)
        artifact = directory / "program.bin"
        compile_native(entry, artifact, "darwin-arm64", clean=True)
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("native execution requires darwin-arm64")
        native = subprocess.run([str(artifact)], capture_output=True, text=True)
        cpython = subprocess.run(
            [sys.executable, str(entry)],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": _REPO_SRC, "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(native.stdout, cpython.stdout)
        self.assertEqual(native.returncode, cpython.returncode)
        return native.stdout

    def test_ldexp_mixes_the_two_register_files(self):
        """The double must reach D0 and the exponent X0, both "first"."""

        self.assertEqual(
            self._match_cpython(
                "from py2bin.cabi import ldexp\n"
                "print(ldexp(3.25, 4))\n"
                "print(ldexp(-1.5, 10))\n"
                "print(ldexp(1.0, -3))\n"
            ),
            "52.0\n-1536.0\n0.125\n",
        )

    def test_two_double_arguments_land_in_d0_and_d1(self):
        self.assertEqual(
            self._match_cpython(
                "from py2bin.cabi import pow, atan2, copysign\n"
                "print(pow(2.0, 10.0))\n"
                "print(atan2(0.0, -1.0))\n"
                "print(copysign(3.0, -0.0))\n"
            ),
            "1024.0\n3.141592653589793\n-3.0\n",
        )

    def test_a_float_result_flows_through_the_float_expression_path(self):
        self.assertEqual(
            self._match_cpython(
                "from py2bin.cabi import hypot, fmod\n"
                "total = hypot(3.0, 4.0) + fmod(17.0, 5.0)\n"
                "print(total)\n"
                "print(hypot(hypot(3.0, 4.0), 12.0))\n"
            ),
            "7.0\n13.0\n",
        )

    def test_integer_arguments_are_widened_to_doubles(self):
        self.assertEqual(
            self._match_cpython(
                "from py2bin.cabi import pow, hypot\n"
                "n = 7\n"
                "print(pow(2, 10))\n"
                "print(hypot(3, 4))\n"
                "print(pow(n, 2))\n"
            ),
            "1024.0\n5.0\n49.0\n",
        )

    def test_an_extern_argument_runs_once_per_inlined_use(self):
        """Splicing the call in at every use would call it more often than
        CPython does, which is wrong even when the callee happens to be pure.
        """

        source = (
            "from py2bin.cabi import pow\n"
            "def twice(value):\n"
            "    return value + value\n"
            "print(twice(pow(3.0, 2.0)))\n"
        )
        self.assertEqual(self._match_cpython(source), "18.0\n")
        entry, directory = _write(source)
        artifact = directory / "once.bin"
        compile_native(entry, artifact, "darwin-arm64", clean=True)
        module = lower(entry, entry.read_text(), ())
        symbols = []

        def walk(value: object) -> None:
            from py2bin.native.ir import ExternCall

            if isinstance(value, ExternCall):
                symbols.append(value.symbol)
            if isinstance(value, (tuple, list)):
                for item in value:
                    walk(item)
                return
            for name in getattr(type(value), "__slots__", ()) or ():
                walk(getattr(value, name))

        for operation in module.operations:
            walk(operation)
        self.assertEqual(symbols.count("pow"), 1)

    def test_a_double_result_is_not_usable_as_an_integer(self):
        entry, _directory = _write(
            "from py2bin.cabi import pow\n"
            "values = [0, 0, 0]\n"
            "values[pow(1.0, 1.0)] = 5\n"
        )
        with self.assertRaisesRegex(NativeCompileError, "returns a C double"):
            lower(entry, entry.read_text(), ())

    def test_a_double_where_an_int_argument_is_declared_is_rejected(self):
        entry, _directory = _write(
            "from py2bin.cabi import ldexp\n"
            "print(ldexp(1.0, 2.5))\n"
        )
        with self.assertRaises(NativeCompileError):
            lower(entry, entry.read_text(), ())


class ObjectiveCMessageShapeTests(unittest.TestCase):
    """The objc_msgSend casts a window and a web view need.

    objc_msgSend is declared variadic and is not one: it reads its arguments
    from the ordinary registers, so every call site is a cast to the callee's
    real prototype. Each test here builds one of those casts, runs it natively,
    and diffs the result against the same source under CPython, whose shim
    declares the identical ctypes prototype.

    Every value printed is an NSInteger or an NSString comparison, never a BOOL
    or a unichar: a callee returning a type narrower than a word leaves the rest
    of the return register undefined, so reading one would make the test's own
    answer unreliable.
    """

    _PRELUDE = (
        "from py2bin.cabi import (\n"
        "    objc_getClass, sel_registerName, objc_msgSend, objc_msgSend2,\n"
        "    objc_msgSend_str, objc_msgSend_id_id, objc_msgSend_long,\n"
        "    objc_msgSend_bool_void, objc_msgSend_rect, objc_msgSend_rect_id,\n"
        "    objc_msgSend_rect_uint_uint_bool,\n"
        ")\n"
        "NSString = objc_getClass('NSString')\n"
        "with_utf8 = sel_registerName('stringWithUTF8String:')\n"
        "description = sel_registerName('description')\n"
        "compare = sel_registerName('compare:')\n"
        "alloc = sel_registerName('alloc')\n"
        "init = sel_registerName('init')\n"
        "value_for_key = sel_registerName('valueForKey:')\n"
        "integer_value = sel_registerName('integerValue')\n"
    )

    def _match_cpython(self, body: str) -> str:
        entry, directory = _write(self._PRELUDE + body)
        artifact = directory / "program.bin"
        compile_native(entry, artifact, "darwin-arm64", clean=True)
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("native execution requires darwin-arm64")
        native = subprocess.run([str(artifact)], capture_output=True, text=True)
        cpython = subprocess.run(
            [sys.executable, str(entry)],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": _REPO_SRC, "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(native.stdout, cpython.stdout, native.stderr)
        self.assertEqual(native.returncode, cpython.returncode, native.stderr)
        return native.stdout

    def test_an_nsrect_argument_is_a_homogeneous_aggregate(self):
        """Four CGFloats in one struct go to D0-D3, in that order.

        -[NSValue description] spells the rectangle back out, so a wrong
        register or a wrong order shows up as different text rather than as a
        plausible number.
        """

        self.assertEqual(
            self._match_cpython(
                "NSValue = objc_getClass('NSValue')\n"
                "boxed = objc_msgSend_rect(\n"
                "    NSValue, sel_registerName('valueWithRect:'),\n"
                "    12.5, -3.25, 640.0, 480.0)\n"
                "text = objc_msgSend(boxed, description)\n"
                "expected = objc_msgSend_str(\n"
                "    NSString, with_utf8, 'NSRect: {{12.5, -3.25}, {640, 480}}')\n"
                "print(objc_msgSend2(text, compare, expected))\n"
                "print(objc_msgSend(text, sel_registerName('length')))\n"
            ),
            "0\n35\n",
        )

    def test_two_object_arguments(self):
        """The shape of loadHTMLString:baseURL:, checked on NSString."""

        self.assertEqual(
            self._match_cpython(
                "hello = objc_msgSend_str(NSString, with_utf8, 'hello world')\n"
                "world = objc_msgSend_str(NSString, with_utf8, 'world')\n"
                "there = objc_msgSend_str(NSString, with_utf8, 'there')\n"
                "replaced = objc_msgSend_id_id(\n"
                "    hello,\n"
                "    sel_registerName("
                "'stringByReplacingOccurrencesOfString:withString:'),\n"
                "    world, there)\n"
                "expected = objc_msgSend_str(NSString, with_utf8, 'hello there')\n"
                "print(objc_msgSend2(replaced, compare, expected))\n"
                "print(objc_msgSend(replaced, sel_registerName('length')))\n"
            ),
            "0\n11\n",
        )

    def test_an_nsinteger_argument(self):
        self.assertEqual(
            self._match_cpython(
                "hello = objc_msgSend_str(NSString, with_utf8, 'hello world')\n"
                "world = objc_msgSend_str(NSString, with_utf8, 'world')\n"
                "tail = objc_msgSend_long(\n"
                "    hello, sel_registerName('substringFromIndex:'), 6)\n"
                "print(objc_msgSend2(tail, compare, world))\n"
                "print(objc_msgSend(tail, sel_registerName('length')))\n"
            ),
            "0\n5\n",
        )

    def test_a_bool_argument_is_normalised_before_the_call(self):
        """256 has a zero low byte, so an un-normalised word is ambiguous.

        A C BOOL is one byte. Handed 256, a callee that tests the whole
        register sees true and one that reads the low byte sees false, and
        nothing in the ABI says which it does. Both paths therefore send
        ``value != 0``, so the flag read back is 1 either way.
        """

        self.assertEqual(
            self._match_cpython(
                "formatter = objc_msgSend(\n"
                "    objc_msgSend(objc_getClass('NSDateFormatter'), alloc), init)\n"
                "lenient = objc_msgSend_str(NSString, with_utf8, 'lenient')\n"
                "setter = sel_registerName('setLenient:')\n"
                "for flag in [0, 1, 256, -1]:\n"
                "    objc_msgSend_bool_void(formatter, setter, flag)\n"
                "    boxed = objc_msgSend2(formatter, value_for_key, lenient)\n"
                "    print(objc_msgSend(boxed, integer_value))\n"
            ),
            "0\n1\n1\n1\n",
        )

    def test_a_window_and_a_web_view_take_their_rectangles(self):
        """The whole point: AppKit and WebKit classes, from a native binary.

        Nothing is ordered on screen -- the claim is only that the arguments
        arrive. styleMask is an NSUInteger, so reading it back proves the word
        after the rectangle landed in X2 rather than being shifted by the four
        doubles; the web view's frame proves the rectangle itself.
        """

        self.assertEqual(
            self._match_cpython(
                "print(objc_msgSend(\n"
                "    objc_getClass('NSApplication'),\n"
                "    sel_registerName('sharedApplication')) != 0)\n"
                "window = objc_msgSend_rect_uint_uint_bool(\n"
                "    objc_msgSend(objc_getClass('NSWindow'), alloc),\n"
                "    sel_registerName("
                "'initWithContentRect:styleMask:backing:defer:'),\n"
                "    0.0, 0.0, 640.0, 480.0, 15, 2, 0)\n"
                "print(window != 0)\n"
                "print(objc_msgSend(window, sel_registerName('styleMask')))\n"
                "configuration = objc_msgSend(objc_msgSend(\n"
                "    objc_getClass('WKWebViewConfiguration'), alloc), init)\n"
                "view = objc_msgSend_rect_id(\n"
                "    objc_msgSend(objc_getClass('WKWebView'), alloc),\n"
                "    sel_registerName('initWithFrame:configuration:'),\n"
                "    0.0, 0.0, 640.0, 480.0, configuration)\n"
                "print(view != 0)\n"
                "frame = objc_msgSend2(\n"
                "    view, value_for_key,\n"
                "    objc_msgSend_str(NSString, with_utf8, 'frame'))\n"
                "expected = objc_msgSend_str(\n"
                "    NSString, with_utf8, 'NSRect: {{0, 0}, {640, 480}}')\n"
                "print(objc_msgSend2(\n"
                "    objc_msgSend(frame, description), compare, expected))\n"
                "objc_msgSend_id_id(\n"
                "    view, sel_registerName('loadHTMLString:baseURL:'),\n"
                "    objc_msgSend_str(NSString, with_utf8, '<html>hi</html>'), 0)\n"
                "print('sent')\n"
            ),
            "True\nTrue\n15\nTrue\n0\nsent\n",
        )

    def test_the_image_loads_appkit_and_webkit(self):
        """A class does not exist until its framework is in the process."""

        entry, directory = _write(
            "from py2bin.cabi import objc_getClass\n"
            "raise SystemExit(objc_getClass('NSWindow') != 0)\n"
        )
        artifact = directory / "program.bin"
        compile_native(entry, artifact, "darwin-arm64", clean=True)
        image = artifact.read_bytes()
        for framework in (b"Foundation", b"AppKit", b"WebKit"):
            with self.subTest(framework=framework):
                self.assertIn(framework + b".framework", image)

    def test_a_void_returning_message_cannot_be_used_as_a_value(self):
        entry, _directory = _write(
            "from py2bin.cabi import objc_msgSend_bool_void\n"
            "print(objc_msgSend_bool_void(1, 2, 1))\n"
        )
        with self.assertRaisesRegex(NativeCompileError, "returns void"):
            lower(entry, entry.read_text(), ())

    def test_a_double_where_a_flag_is_declared_is_rejected(self):
        entry, _directory = _write(
            "from py2bin.cabi import objc_msgSend_bool_void\n"
            "objc_msgSend_bool_void(1, 2, 0.5)\n"
        )
        with self.assertRaises(NativeCompileError):
            lower(entry, entry.read_text(), ())

    def test_an_integer_where_a_rectangle_member_is_declared_is_widened(self):
        """A CGFloat position accepts an int, as every "f64" argument does."""

        self.assertEqual(
            self._match_cpython(
                "NSValue = objc_getClass('NSValue')\n"
                "boxed = objc_msgSend_rect(\n"
                "    NSValue, sel_registerName('valueWithRect:'), 0, 0, 640, 480)\n"
                "expected = objc_msgSend_str(\n"
                "    NSString, with_utf8, 'NSRect: {{0, 0}, {640, 480}}')\n"
                "print(objc_msgSend2(\n"
                "    objc_msgSend(boxed, description), compare, expected))\n"
            ),
            "0\n",
        )


class ObjectiveCCallbackTests(unittest.TestCase):
    """Method implementations: the runtime calling back into compiled code.

    A Python function is normally inlined by this compiler, so it has no
    address. An Objective-C method implementation cannot be inlined -- the
    runtime keeps the pointer and branches to it later, on a stack of its own,
    with the receiver in x0 and the selector in x1 -- so the def is lowered a
    second time into a real function with a real frame. These tests check that
    Foundation genuinely enters that frame, and that everything it cannot do
    there is refused at build time rather than miscompiled.
    """

    _PRELUDE = (
        "from py2bin.cabi import (\n"
        "    objc_getClass, sel_registerName, objc_msgSend, objc_msgSend2,\n"
        "    objc_msgSend_str, objc_msgSend_long, objc_allocateClassPair,\n"
        "    class_addMethod, objc_registerClassPair,\n"
        ")\n"
        "NSObject = objc_getClass('NSObject')\n"
        "new = sel_registerName('new')\n"
    )

    def _match_cpython(self, body: str) -> str:
        entry, directory = _write(self._PRELUDE + body)
        artifact = directory / "program.bin"
        compile_native(entry, artifact, "darwin-arm64", clean=True)
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("native execution requires darwin-arm64")
        native = subprocess.run([str(artifact)], capture_output=True, text=True)
        cpython = subprocess.run(
            [sys.executable, str(entry)],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": _REPO_SRC, "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(native.stdout, cpython.stdout, native.stderr)
        self.assertEqual(native.returncode, cpython.returncode, native.stderr)
        return native.stdout

    def _refuse(self, body: str) -> str:
        entry, _directory = _write(self._PRELUDE + body)
        with self.assertRaises(NativeCompileError) as caught:
            lower(entry, entry.read_text(), ())
        return str(caught.exception)

    def test_foundation_calls_the_implementation_on_its_own_stack(self):
        """NSMutableArray's sort enters compiled code and uses what it returns.

        This is the whole feature in one program: a class built at run time, an
        implementation that is the address of a Python def, dispatch from inside
        Foundation's own sort, three arguments delivered in x0-x2, and a
        returned value the caller acts on.
        """

        self.assertEqual(
            self._match_cpython(
                "def compare(this, cmd, other):\n"
                "    print('compare: ran')\n"
                "    return -1\n"
                "cls = objc_allocateClassPair(NSObject, 'P2BSortTest', 0)\n"
                "print(class_addMethod(\n"
                "    cls, sel_registerName('compare:'), compare, 'q@:@'))\n"
                "objc_registerClassPair(cls)\n"
                "array = objc_msgSend(\n"
                "    objc_getClass('NSMutableArray'), sel_registerName('array'))\n"
                "add = sel_registerName('addObject:')\n"
                "objc_msgSend2(array, add, objc_msgSend(cls, new))\n"
                "objc_msgSend2(array, add, objc_msgSend(cls, new))\n"
                "objc_msgSend2(array, sel_registerName('sortUsingSelector:'),\n"
                "              sel_registerName('compare:'))\n"
                "print(objc_msgSend(array, sel_registerName('count')))\n"
            ),
            "1\ncompare: ran\n2\n",
        )

    def test_the_selector_arrives_in_the_second_register(self):
        """x1 holds a live SEL, which sel_getName spells back out.

        Reading it proves the implementation was entered with the Objective-C
        convention rather than with the method's own arguments in x0-x1.
        """

        self.assertEqual(
            self._match_cpython(
                "def described(this, cmd):\n"
                "    return objc_msgSend_str(\n"
                "        objc_getClass('NSString'),\n"
                "        sel_registerName('stringWithUTF8String:'),\n"
                "        'answered')\n"
                "cls = objc_allocateClassPair(NSObject, 'P2BSelTest', 0)\n"
                "class_addMethod(cls, sel_registerName('description'),\n"
                "                described, '@@:')\n"
                "objc_registerClassPair(cls)\n"
                "text = objc_msgSend(objc_msgSend(cls, new),\n"
                "                    sel_registerName('description'))\n"
                "print(objc_msgSend(text, sel_registerName('length')))\n"
            ),
            "8\n",
        )

    def test_a_bool_result_is_normalised_to_zero_or_one(self):
        """A BOOL is one byte, so a callback returning 7 must still answer YES.

        NSArray's indexOfObjectPassingTest: is not reachable here (it takes a
        block), so the check is -isEqual:, which Foundation calls from inside
        -containsObject: and reads as a one-byte BOOL.
        """

        self.assertEqual(
            self._match_cpython(
                "def equal(this, cmd, other):\n"
                "    return 7\n"
                "cls = objc_allocateClassPair(NSObject, 'P2BBoolTest', 0)\n"
                "class_addMethod(cls, sel_registerName('isEqual:'), equal, 'B@:@')\n"
                "objc_registerClassPair(cls)\n"
                "array = objc_msgSend(\n"
                "    objc_getClass('NSMutableArray'), sel_registerName('array'))\n"
                "objc_msgSend2(array, sel_registerName('addObject:'),\n"
                "              objc_msgSend(cls, new))\n"
                "print(objc_msgSend2(array, sel_registerName('containsObject:'),\n"
                "                    objc_msgSend(cls, new)))\n"
            ),
            "1\n",
        )

    def test_one_def_may_implement_two_selectors(self):
        """The same body registered twice is lowered once and shared."""

        self.assertEqual(
            self._match_cpython(
                "def answer(this, cmd):\n"
                "    return 21\n"
                "cls = objc_allocateClassPair(NSObject, 'P2BTwiceTest', 0)\n"
                "class_addMethod(cls, sel_registerName('first'), answer, 'q@:')\n"
                "class_addMethod(cls, sel_registerName('second'), answer, 'q@:')\n"
                "objc_registerClassPair(cls)\n"
                "one = objc_msgSend(cls, new)\n"
                "print(objc_msgSend(one, sel_registerName('first'))\n"
                "      + objc_msgSend(one, sel_registerName('second')))\n"
            ),
            "42\n",
        )

    def test_a_second_class_of_the_same_name_is_refused_by_the_runtime(self):
        """objc_allocateClassPair answers nil for a name already registered.

        Both runs must agree about that, because every message to the nil it
        leaves behind silently returns zero and the program would otherwise
        look like it worked.
        """

        self.assertEqual(
            self._match_cpython(
                "def answer(this, cmd):\n"
                "    return 1\n"
                "first = objc_allocateClassPair(NSObject, 'P2BDupTest', 0)\n"
                "class_addMethod(first, sel_registerName('x'), answer, 'q@:')\n"
                "objc_registerClassPair(first)\n"
                "second = objc_allocateClassPair(NSObject, 'P2BDupTest', 0)\n"
                "print(second)\n"
            ),
            "0\n",
        )

    def test_a_selector_the_class_already_has_is_refused_in_both_runs(self):
        """class_addMethod answers NO only for a method the class itself has.

        An INHERITED implementation does not block it -- that is how a subclass
        overrides one -- and registering the class pair first makes no
        difference either. Both runs have to agree about all three, because the
        answer is a one-byte BOOL whose upper register bits are undefined.
        """

        self.assertEqual(
            self._match_cpython(
                "def answer(this, cmd):\n"
                "    return 1\n"
                "def other(this, cmd):\n"
                "    return 2\n"
                "cls = objc_allocateClassPair(NSObject, 'P2BLateTest', 0)\n"
                "x = sel_registerName('x')\n"
                "print(class_addMethod(cls, x, answer, 'q@:'))\n"
                "print(class_addMethod(cls, x, other, 'q@:'))\n"
                "objc_registerClassPair(cls)\n"
                "print(class_addMethod(cls, sel_registerName('y'), other, 'q@:'))\n"
                "print(class_addMethod(cls, x, other, 'q@:'))\n"
            ),
            "1\n0\n1\n0\n",
        )

    def test_an_allocating_body_is_refused(self):
        """The arena's bump pointer is in the ENTRY POINT's frame.

        A callback runs on a frame the runtime made, where that slot is
        somebody else's memory, so an allocation there would write through a
        wild address at some depth inside AppKit.
        """

        message = self._refuse(
            "def m(this, cmd, other):\n"
            "    print(other + 1)\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x:'), m, 'v@:@')\n"
        )
        self.assertIn("allocates", message)

    def test_a_body_reading_a_module_variable_is_refused(self):
        """A callback runs later, so a module value baked into it can be stale."""

        message = self._refuse(
            "count = 0\n"
            "def m(this, cmd, other):\n"
            "    if count > 0:\n"
            "        return 1\n"
            "    return 0\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x:'), m, 'q@:@')\n"
            "count = count + 1\n"
        )
        self.assertIn("module-level name", message)

    def test_a_body_writing_a_module_variable_is_refused(self):
        message = self._refuse(
            "count = 0\n"
            "def m(this, cmd, other):\n"
            "    global count\n"
            "    count = count + 1\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x:'), m, 'v@:@')\n"
            "print(count)\n"
        )
        self.assertIn("module-level", message)

    def test_a_module_constant_bound_once_is_allowed(self):
        """A name nothing can rebind is the one module value that is stable."""

        self.assertEqual(
            self._match_cpython(
                "ANSWER = 21\n"
                "def doubled(value):\n"
                "    return value * 2\n"
                "def answer(this, cmd):\n"
                "    return doubled(ANSWER)\n"
                "cls = objc_allocateClassPair(NSObject, 'P2BConstTest', 0)\n"
                "class_addMethod(cls, sel_registerName('x'), answer, 'q@:')\n"
                "objc_registerClassPair(cls)\n"
                "print(objc_msgSend(objc_msgSend(cls, new), sel_registerName('x')))\n"
            ),
            "42\n",
        )

    def test_a_floating_point_argument_encoding_is_refused(self):
        """A 'd' arrives in d0-d7 and a function prologue only reads x0-x7."""

        message = self._refuse(
            "def m(this, cmd, value):\n"
            "    return 0\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x:'), m, 'q@:d')\n"
        )
        self.assertIn("d0-d7", message)

    def test_a_struct_argument_encoding_is_refused(self):
        message = self._refuse(
            "def m(this, cmd, value):\n"
            "    return 0\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x:'), m, 'q@:{CGRect=dddd}')\n"
        )
        self.assertIn("{...}", message)

    def test_a_floating_point_result_encoding_is_refused(self):
        """A double result belongs in d0, which no Return ever writes."""

        message = self._refuse(
            "def m(this, cmd):\n"
            "    return 0\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x'), m, 'd@:')\n"
        )
        self.assertIn("d0", message)

    def test_an_encoding_without_the_receiver_and_selector_is_refused(self):
        message = self._refuse(
            "def m(value):\n"
            "    return 0\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x:'), m, 'q@')\n"
        )
        self.assertIn("'@:'", message)

    def test_an_arity_that_disagrees_with_the_encoding_is_refused(self):
        message = self._refuse(
            "def m(this, cmd):\n"
            "    return 0\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x:'), m, 'q@:@')\n"
        )
        self.assertIn("takes 2", message)

    def test_a_void_encoding_over_a_value_returning_body_is_refused(self):
        message = self._refuse(
            "def m(this, cmd):\n"
            "    return 3\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x'), m, 'v@:')\n"
        )
        self.assertIn("never read it", message)

    def test_a_value_encoding_over_a_void_body_is_refused(self):
        message = self._refuse(
            "def m(this, cmd):\n"
            "    print('hi')\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x'), m, 'q@:')\n"
        )
        self.assertIn("result register", message)

    def test_an_implementation_that_is_not_a_def_is_refused(self):
        """Only a def has an address; a value cannot be one."""

        message = self._refuse(
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x'), 12345, 'v@:')\n"
        )
        self.assertIn("NAME of a function", message)

    def test_an_encoding_that_is_not_a_constant_is_refused(self):
        message = self._refuse(
            "def m(this, cmd):\n"
            "    print('hi')\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x'), m, objc_getClass('NSObject'))\n"
        )
        self.assertIn("compile-time string constant", message)

    def test_a_callback_may_not_register_a_callback(self):
        message = self._refuse(
            "def inner(this, cmd):\n"
            "    print('inner')\n"
            "def m(this, cmd):\n"
            "    k = objc_allocateClassPair(objc_getClass('NSObject'), 'K2', 0)\n"
            "    class_addMethod(k, sel_registerName('y'), inner, 'v@:')\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x'), m, 'v@:')\n"
        )
        self.assertIn("register every class from the module body", message)

    def test_the_result_of_registering_a_class_pair_cannot_be_used(self):
        """objc_registerClassPair returns void, so its register holds dirt."""

        message = self._refuse(
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "print(objc_registerClassPair(cls))\n"
        )
        self.assertIn("returns void", message)

    def test_the_module_carries_the_implementation_as_a_real_function(self):
        """The IR gains a Function, which nothing else in the Python tier does."""

        entry, _directory = _write(
            self._PRELUDE
            + "def m(this, cmd):\n"
            "    return 5\n"
            "cls = objc_allocateClassPair(NSObject, 'K', 0)\n"
            "class_addMethod(cls, sel_registerName('x'), m, 'q@:')\n"
        )
        module = lower(entry, entry.read_text(), ())
        self.assertEqual(len(module.functions), 1)
        # Slots 0 and 1 are where the prologue spills the receiver and the
        # selector, so a body reads them like any other local.
        self.assertEqual(module.functions[0].parameters, 2)
        self.assertGreaterEqual(module.functions[0].stack_slots, 2)


class ObjectiveCEncodingAgreementTests(unittest.TestCase):
    """The compiler and the CPython shim must read an encoding the same way.

    They are separate tables because the front end has to run on hosts with no
    Objective-C runtime at all. A disagreement between them is exactly a case
    where the interpreted run and the compiled run put a value in a different
    register, so it is checked rather than trusted.
    """

    def test_the_two_encoding_tables_agree(self):
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("the cabi shim loads the Objective-C runtime")
        from py2bin import cabi
        from py2bin.native import frontend

        self.assertEqual(
            set(frontend._IMP_RESULT_CODES), set(cabi._IMP_RESULT_TYPES)
        )
        self.assertEqual(
            set(frontend._IMP_ARGUMENT_CODES), set(cabi._IMP_ARGUMENT_TYPES)
        )

    def test_every_accepted_encoding_parses(self):
        from py2bin.native.frontend import parse_method_encoding

        self.assertEqual(parse_method_encoding("v@:@"), ("void", ("@", ":", "@")))
        self.assertEqual(parse_method_encoding("q@:"), ("int", ("@", ":")))
        self.assertEqual(parse_method_encoding("B@:@"), ("bool", ("@", ":", "@")))
        self.assertEqual(parse_method_encoding("@@:@@"), ("ptr", ("@", ":", "@", "@")))
        for bad in ("", "d@:", "v@:d", "v@:B", "v@", "v:@", "v@::"):
            with self.assertRaises(ValueError, msg=bad):
                parse_method_encoding(bad)
