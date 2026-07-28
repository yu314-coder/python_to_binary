"""The CPython C-API tier: vetted symbols, the C dialect, and the ABI.

py2bin generates C that calls the CPython C-API and then compiles that C to
machine code with its OWN assembler. These tests pin down the three pieces that
have to agree for that to be honest:

* the vetted symbol table, its CPython-side shims, and the fact that every
  listed symbol really is exported by the interpreter's shared library;
* the canonical-C dialect: what it accepts, and -- more importantly -- what it
  refuses rather than miscompiling;
* the darwin-arm64 register ABI, verified by running the emitted binary
  NATIVELY and comparing its stdout and exit status against the same program
  interpreted by CPython.
"""

from __future__ import annotations

import ast
import ctypes
import inspect
import platform
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from py2bin import cabi
from py2bin.c_native import (
    CNativeCompileError,
    c_to_python_source,
    compile_c_native,
)
from py2bin.native import NativeCompileError
from py2bin.native import arm64
from py2bin.native.arm64 import encode_darwin_extern
from py2bin.native.frontend import (
    _CABI_MAX_ARGUMENTS,
    _CABI_RESULT_WIDTH,
    _CABI_RESULTS,
    _CABI_SYMBOLS,
    lower,
)
from py2bin.native.ir import (
    ExitValue,
    ExternCall,
    FloatConstant,
    FloatStore,
    IntConstant,
    Module,
    Store,
)


_HOST_IS_DARWIN_ARM64 = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)
_REPO_SRC = str(Path(__file__).resolve().parent.parent / "src")

# Prototypes every C fixture below reuses. Writing them out is the whole point:
# py2bin never reads Python.h, so the generated C states its own ABI.
_PROTOTYPES = """\
extern void Py_Initialize(void);
extern void Py_Finalize(void);
extern PyObject *PyLong_FromLongLong(long long value);
extern long long PyLong_AsLongLong(PyObject *value);
extern PyObject *PyUnicode_FromString(const char *text);
extern PyObject *PyNumber_Add(PyObject *left, PyObject *right);
extern PyObject *PyObject_Str(PyObject *value);
extern PyObject *PyObject_RichCompare(PyObject *left, PyObject *right, int operation);
extern int PyObject_IsTrue(PyObject *value);
extern PyObject *PySys_GetObject(const char *name);
extern int PyFile_WriteObject(PyObject *value, PyObject *stream, int flags);
extern int PyFile_WriteString(const char *text, PyObject *stream);
extern void PySys_WriteStdout(const char *text);
extern PyObject *PyImport_ImportModule(const char *name);
extern PyObject *PyObject_GetAttrString(PyObject *value, const char *name);
extern PyObject *PyObject_CallOneArg(PyObject *callable, PyObject *argument);
extern PyObject *PyList_New(long long length);
extern int PyList_Append(PyObject *list, PyObject *item);
extern long long PyObject_Size(PyObject *value);
extern PyObject *PyErr_Occurred(void);
extern void Py_IncRef(PyObject *value);
extern void Py_DecRef(PyObject *value);
"""


def _extern_calls(module: Module) -> list[str]:
    """Every ``ExternCall`` symbol in a lowered module, one entry per call site."""

    found: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, ExternCall):
            found.append(value.symbol)
        if isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
            return
        for name in getattr(type(value), "__slots__", ()) or ():
            walk(getattr(value, name))

    for operation in module.operations:
        walk(operation)
    return found


class VettedSymbolTableTests(unittest.TestCase):
    def test_every_vetted_symbol_has_a_cpython_shim(self):
        for name in _CABI_SYMBOLS:
            with self.subTest(symbol=name):
                self.assertIn(name, cabi.__all__)
                self.assertTrue(callable(getattr(cabi, name)))
                self.assertIn(_CABI_RESULTS[name], {"int", "ptr", "void", "float"})

    def test_every_shim_takes_exactly_the_declared_arguments(self):
        """The two tables are one ABI claim, so their arities cannot differ.

        A shim with one parameter too few would still accept the compiler's
        call under CPython -- Python would raise, loudly -- but a shim with one
        too many silently passes a default the native call never sends.
        """

        for name, (_symbol, signature) in _CABI_SYMBOLS.items():
            with self.subTest(symbol=name):
                parameters = inspect.signature(getattr(cabi, name)).parameters
                self.assertEqual(len(parameters), len(signature))

    def test_an_nsrect_shim_passes_one_aggregate_not_four_doubles(self):
        """AAPCS64 puts a four-double struct in four consecutive FP registers.

        That is the same placement four loose doubles get here only because the
        rectangle is the first floating-point argument in each of these
        prototypes. Declaring the aggregate is what makes the shim's ABI the
        callee's ABI rather than a coincidence that a later shape would break.
        """

        self.assertEqual(ctypes.sizeof(cabi._NSRect), 32)
        # A nil receiver makes objc_msgSend return nil without looking at the
        # selector, so these calls are safe and still install the prototype.
        calls = (
            lambda: cabi.objc_msgSend_rect(0, 0, 1.0, 2.0, 3.0, 4.0),
            lambda: cabi.objc_msgSend_rect_id(0, 0, 1.0, 2.0, 3.0, 4.0, 0),
            lambda: cabi.objc_msgSend_rect_uint_uint_bool(
                0, 0, 1.0, 2.0, 3.0, 4.0, 0, 0, 0
            ),
        )
        for call in calls:
            self.assertEqual(call(), 0)
            self.assertIn(cabi._NSRect, cabi._objc.objc_msgSend.argtypes)

    def test_signatures_fit_the_register_argument_budget(self):
        """AAPCS64 counts the integer and floating-point files separately."""

        for name, (_symbol, signature) in _CABI_SYMBOLS.items():
            with self.subTest(symbol=name):
                for kind in signature:
                    self.assertIn(
                        kind, {"int", "ptr", "bool", "cstr", "cfmt", "f64", "imp"}
                    )
                doubles = sum(1 for kind in signature if kind == "f64")
                self.assertLessEqual(len(signature) - doubles, _CABI_MAX_ARGUMENTS)
                self.assertLessEqual(doubles, _CABI_MAX_ARGUMENTS)

    def test_a_float_result_declares_the_f64_result_width(self):
        """The width table is what tells the encoder to read D0 and not X0."""

        for name, result in _CABI_RESULTS.items():
            with self.subTest(symbol=name):
                width = _CABI_RESULT_WIDTH.get(name, "i64")
                self.assertEqual(result == "float", width == "f64")

    def test_a_variadic_callee_never_takes_a_double(self):
        """Apple's arm64 ABI stacks variadic doubles, the opposite rule."""

        for name, (_symbol, signature) in _CABI_SYMBOLS.items():
            with self.subTest(symbol=name):
                if "cfmt" in signature:
                    self.assertNotIn("f64", signature)

    def test_a_method_implementation_is_followed_by_its_type_encoding(self):
        """The encoding is the only statement of the callee's register layout.

        It sits in the argument after the implementation, so the front end can
        only decide whether a def is compilable as that method by reading the
        two together. A signature that separated them would leave the
        implementation lowered against a layout nothing had checked.
        """

        for name, (_symbol, signature) in _CABI_SYMBOLS.items():
            with self.subTest(symbol=name):
                for position, kind in enumerate(signature):
                    if kind == "imp":
                        self.assertEqual(signature[position + 1 :][:1], ("cstr",))

    def test_every_cpython_symbol_resolves_in_the_interpreter_library(self):
        """A compiled binary binds these through dyld, so they must be exported."""

        library = ctypes.CDLL(cabi._cpython_library())
        for name in sorted(cabi._CPYTHON_SYMBOLS):
            with self.subTest(symbol=name):
                self.assertTrue(hasattr(library, name), f"{name} is not exported")

    def test_shims_perform_the_real_c_api_calls(self):
        left = cabi.PyLong_FromLongLong(20)
        right = cabi.PyLong_FromLongLong(22)
        total = cabi.PyNumber_Add(left, right)
        try:
            self.assertNotEqual(total, 0)
            self.assertEqual(cabi.PyLong_AsLongLong(total), 42)
            text = cabi.PyObject_Str(total)
            try:
                self.assertNotEqual(text, 0)
                self.assertEqual(cabi.PyObject_Size(text), 2)
            finally:
                cabi.Py_DecRef(text)
            self.assertEqual(cabi.PyErr_Occurred(), 0)
        finally:
            for handle in (left, right, total):
                cabi.Py_DecRef(handle)

    def test_write_stdout_shim_refuses_a_conversion(self):
        with self.assertRaises(ValueError):
            cabi.PySys_WriteStdout("%d\n")


class CanonicalCHandleDialectTests(unittest.TestCase):
    def _reject(self, source: str, expected: str) -> None:
        with self.assertRaises(CNativeCompileError) as caught:
            c_to_python_source(_PROTOTYPES + source, "handles.c")
        self.assertIn(expected, str(caught.exception))

    def test_handle_locals_helpers_and_null_checks_are_accepted(self):
        reconstructed = c_to_python_source(
            _PROTOTYPES
            + "PyObject *doubled(PyObject *value)\n"
            "{\n"
            "    return PyNumber_Add(value, value);\n"
            "}\n"
            "void show(PyObject *value, PyObject *stream)\n"
            "{\n"
            "    PyObject *text;\n"
            "    text = PyObject_Str(value);\n"
            "    PyFile_WriteObject(text, stream, 1);\n"
            '    PyFile_WriteString("\\n", stream);\n'
            "    Py_DecRef(text);\n"
            "}\n"
            "int main(void)\n"
            "{\n"
            "    PyObject *stream;\n"
            "    PyObject *value;\n"
            "    long long index;\n"
            "    Py_Initialize();\n"
            '    stream = PySys_GetObject("stdout");\n'
            "    if (stream == NULL) {\n"
            "        return 1;\n"
            "    }\n"
            "    index = 0;\n"
            "    while (index < 3) {\n"
            "        value = doubled(PyLong_FromLongLong(index));\n"
            "        show(value, stream);\n"
            "        Py_DecRef(value);\n"
            "        index += 1;\n"
            "    }\n"
            "    return 0;\n"
            "}\n",
            "handles.c",
        )
        self.assertIn("def doubled(value: int) -> int:", reconstructed)
        self.assertIn("def show(value: int, stream: int) -> int:", reconstructed)
        self.assertIn("if stream == 0", reconstructed)

    def test_integer_is_not_accepted_where_a_handle_is_dereferenced(self):
        self._reject(
            "int main(void) { return PyLong_AsLongLong(4660); }\n",
            "needs a pointer handle",
        )

    def test_handle_is_not_accepted_where_an_integer_is_expected(self):
        self._reject(
            "int main(void) {\n"
            "    PyObject *o;\n"
            "    o = PyLong_FromLongLong(1);\n"
            "    return PyLong_AsLongLong(PyLong_FromLongLong(o));\n"
            "}\n",
            "needs a 'long long' value",
        )

    def test_void_result_cannot_be_used_as_a_value(self):
        self._reject(
            "int main(void) {\n"
            "    PyObject *o;\n"
            "    o = PyLong_FromLongLong(1);\n"
            "    return Py_DecRef(o);\n"
            "}\n",
            "does not produce a value",
        )

    def test_handles_are_opaque(self):
        self._reject(
            "int main(void) {\n"
            "    PyObject *o;\n"
            "    o = PyLong_FromLongLong(1);\n"
            "    o += 8;\n"
            "    return 0;\n"
            "}\n",
            "pointer arithmetic is not in the canonical C subset",
        )
        self._reject(
            "int main(void) {\n"
            "    PyObject *o;\n"
            "    o = PyLong_FromLongLong(1);\n"
            "    return o + 1;\n"
            "}\n",
            "needs a 'long long' value",
        )
        self._reject(
            "int main(void) {\n"
            "    PyObject *o;\n"
            "    o = PyLong_FromLongLong(1);\n"
            "    if (o < NULL) { return 1; }\n"
            "    return 0;\n"
            "}\n",
            "have no defined order",
        )

    def test_handle_and_integer_locals_do_not_mix(self):
        self._reject(
            "int main(void) { long long n; n = PyLong_FromLongLong(1); return n; }\n",
            "needs a 'long long' value",
        )
        self._reject(
            "int main(void) { PyObject *o; o = 7; return 0; }\n",
            "needs a pointer handle",
        )

    def test_undeclared_identifier_is_rejected(self):
        self._reject(
            "int main(void) { return missing; }\n",
            "is not a declared local or parameter",
        )

    def test_prototype_must_agree_with_the_vetted_abi(self):
        with self.assertRaisesRegex(CNativeCompileError, "declares 1 parameter"):
            c_to_python_source(
                "extern PyObject *PyNumber_Add(PyObject *left);\n"
                "int main(void) { return 0; }\n",
                "mismatch.c",
            )
        with self.assertRaisesRegex(CNativeCompileError, "returns 'void'"):
            c_to_python_source(
                "extern long long Py_DecRef(PyObject *value);\n"
                "int main(void) { return 0; }\n",
                "mismatch.c",
            )

    def test_dereferenceable_pointers_are_still_rejected(self):
        with self.assertRaisesRegex(CNativeCompileError, "pointer types are not"):
            c_to_python_source(
                "long long helper(long long *value) { return 1; }\n"
                "int main(void) { return 0; }\n",
                "deref.c",
            )


class ExternCallSafetyTests(unittest.TestCase):
    """Rejections and fallbacks that keep an extern call from running wrongly."""

    def _lower_c(self, source: str) -> Module:
        tree = c_to_python_source(_PROTOTYPES + source, "safety.c")
        return lower(Path("safety.c"), tree)

    def test_extern_call_is_rejected_in_a_conditional_expression(self):
        # Both arms of '?:' and of a short-circuit are lowered eagerly *when
        # the condition is only known at run time*, so a call in the untaken
        # arm would still run. A condition settled at build time no longer
        # lowers both arms - only the one that runs - which is checked below
        # to be a tightening rather than a hole: the untaken arm's call is not
        # emitted at all.
        for source in (
            "int main(void) {\n"
            "    long long c;\n"
            "    c = PyLong_AsLongLong(PyLong_FromLongLong(1));\n"
            "    long long n;\n"
            "    n = c ? PyLong_AsLongLong(PyLong_FromLongLong(2)) : 0;\n"
            "    return n;\n"
            "}\n",
            "int main(void) {\n"
            "    long long c;\n"
            "    c = PyLong_AsLongLong(PyLong_FromLongLong(1));\n"
            "    long long n;\n"
            "    n = c && PyLong_AsLongLong(PyLong_FromLongLong(2));\n"
            "    return n;\n"
            "}\n",
        ):
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    NativeCompileError, "cannot appear in a conditional expression"
                ):
                    self._lower_c(source)

    def test_a_settled_condition_lowers_only_the_arm_that_runs(self):
        # The tightening the test above refers to. With the condition known at
        # build time the untaken arm is not lowered, so its call cannot run -
        # which is what the eager-arm refusal existed to prevent.
        taken = self._lower_c(
            "int main(void) {\n"
            "    long long n;\n"
            "    n = 1 ? PyLong_AsLongLong(PyLong_FromLongLong(1)) : 0;\n"
            "    return n;\n"
            "}\n"
        )
        self.assertEqual(len(_extern_calls(taken)), 2)
        skipped = self._lower_c(
            "int main(void) {\n"
            "    long long n;\n"
            "    n = 0 ? PyLong_AsLongLong(PyLong_FromLongLong(1)) : 7;\n"
            "    return n;\n"
            "}\n"
        )
        self.assertEqual(_extern_calls(skipped), [])

    def test_variadic_symbol_rejects_a_format_conversion(self):
        with self.assertRaisesRegex(NativeCompileError, "must not contain '%'"):
            self._lower_c('int main(void) { PySys_WriteStdout("%d\\n"); return 0; }\n')

    def test_variadic_symbol_accepts_a_plain_literal(self):
        module = self._lower_c(
            'int main(void) { PySys_WriteStdout("plain\\n"); return 0; }\n'
        )
        self.assertEqual(_extern_calls(module).count("PySys_WriteStdout"), 1)

    def test_inlining_never_repeats_an_extern_call(self):
        """A parameter used twice must not call its argument's callee twice."""

        module = self._lower_c(
            "PyObject *doubled(PyObject *value)\n"
            "{\n"
            "    return PyNumber_Add(value, value);\n"
            "}\n"
            "long long twice(long long value)\n"
            "{\n"
            "    return value + value;\n"
            "}\n"
            "int main(void)\n"
            "{\n"
            "    PyObject *result;\n"
            "    long long n;\n"
            "    result = doubled(PyLong_FromLongLong(21));\n"
            "    n = twice(PyLong_AsLongLong(result));\n"
            "    return n;\n"
            "}\n"
        )
        calls = _extern_calls(module)
        self.assertEqual(calls.count("PyLong_FromLongLong"), 1)
        self.assertEqual(calls.count("PyLong_AsLongLong"), 1)
        self.assertEqual(calls.count("PyNumber_Add"), 1)

    def test_local_holding_an_extern_result_is_not_inlined_twice(self):
        source = (
            "from py2bin.cabi import getpid\n"
            "def total() -> int:\n"
            "    value = getpid()\n"
            "    return value + value\n"
            "raise SystemExit(total() - getpid() - getpid())\n"
        )
        module = lower(Path("locals.py"), source)
        self.assertEqual(_extern_calls(module).count("getpid"), 3)

    def test_extern_calls_are_rejected_for_non_darwin_arm64_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.c"
            entry.write_text(
                _PROTOTYPES + "int main(void) { Py_Initialize(); return 0; }\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                NativeCompileError, "only supported for target 'darwin-arm64'"
            ):
                compile_c_native(
                    entry, root / "program", target="linux-x86_64", clean=True
                )


class Arm64ExternAbiTests(unittest.TestCase):
    def _encode(self, count: int) -> tuple[bytes, list[tuple[int, str]]]:
        call = ExternCall("PyFile_WriteObject", tuple(
            IntConstant(index + 1) for index in range(count)
        ))
        module = Module([Store(0, call), ExitValue(IntConstant(0))], 1)
        code, externs, _statics = encode_darwin_extern(module, 0x100004000)
        return code, externs

    def test_arguments_land_in_x0_through_x7(self):
        code, externs = self._encode(8)
        words = struct.unpack(f"<{len(code) // 4}I", code[: len(code) // 4 * 4])
        # Each argument is pushed in a 16-byte slot, then popped back into its
        # AAPCS64 register from the last one to the first.
        loads = [word for word in words if word & 0xFFFFFFF8 == 0xF94003E0]
        self.assertEqual([word & 7 for word in loads[:8]], [7, 6, 5, 4, 3, 2, 1, 0])
        self.assertEqual([symbol for _offset, symbol in externs], ["PyFile_WriteObject"])

    def test_stack_is_sixteen_byte_aligned_at_the_branch(self):
        code, externs = self._encode(3)
        words = list(struct.unpack(f"<{len(code) // 4}I", code[: len(code) // 4 * 4]))
        branch = externs[0][0] // 4 + 2  # adrp, ldr, blr
        self.assertEqual(words[branch], 0xD63F0200)  # blr x16
        offset = 0
        for word in words[:branch]:
            if word & 0xFFC003FF == 0xD10003FF:  # sub sp, sp, #imm
                offset -= (word >> 10) & 0xFFF
            elif word & 0xFFC003FF == 0x910003FF:  # add sp, sp, #imm
                offset += (word >> 10) & 0xFFF
            self.assertEqual(offset % 16, 0, "sp left unaligned before a call")
        self.assertEqual(offset, -16, "argument spills were not unwound")

    def test_more_arguments_than_registers_is_refused_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "at most 8 integer and 8 float"):
            self._encode(9)

    def test_nine_arguments_fit_when_four_of_them_are_doubles(self):
        """The two register files are counted apart, so this is a legal call.

        NSWindow's initWithContentRect:styleMask:backing:defer: is exactly this
        shape once the NSRect is spelled as its four CGFloat members.
        """

        call = ExternCall(
            "objc_msgSend",
            (
                IntConstant(1),
                IntConstant(2),
                FloatConstant(0.0),
                FloatConstant(0.0),
                FloatConstant(640.0),
                FloatConstant(480.0),
                IntConstant(15),
                IntConstant(2),
                IntConstant(0),
            ),
            "i64",
        )
        code, externs, _statics = arm64.encode_darwin_extern(
            Module([Store(0, call)], 4), 0x100004000
        )
        words = list(struct.unpack(f"<{len(code) // 4}I", code[: len(code) // 4 * 4]))
        branch = externs[0][0] // 4
        reloads = [
            word
            for word in words[:branch]
            if word & 0xFFFFFFE0 in (0xF94003E0, 0xFD4003E0)
        ]
        self.assertEqual(
            reloads,
            [
                0xF94003E4,  # ldr x4, [sp]   defer
                0xF94003E3,  # ldr x3, [sp]   backing
                0xF94003E2,  # ldr x2, [sp]   styleMask
                0xFD4003E3,  # ldr d3, [sp]   height
                0xFD4003E2,  # ldr d2, [sp]   width
                0xFD4003E1,  # ldr d1, [sp]   origin y
                0xFD4003E0,  # ldr d0, [sp]   origin x
                0xF94003E1,  # ldr x1, [sp]   _cmd
                0xF94003E0,  # ldr x0, [sp]   self
            ],
        )

    def test_integer_only_calls_are_unchanged_by_the_float_argument_path(self):
        """The whole CPython C-API surface rides this path; drift is silent."""

        for count in range(9):
            with self.subTest(arguments=count):
                code, _externs = self._encode(count)
                words = list(
                    struct.unpack(f"<{len(code) // 4}I", code[: len(code) // 4 * 4])
                )
                reloads = [
                    word for word in words if word & 0xFFFFFFE0 == 0xF94003E0
                ]
                self.assertEqual(
                    reloads, [0xF94003E0 | index for index in reversed(range(count))]
                )
                self.assertNotIn(0xFD0003E0, words)  # no str d0, [sp]

    def test_a_float_result_is_not_readable_as_an_integer(self):
        call = ExternCall("pow", (FloatConstant(2.0), FloatConstant(3.0)), "f64")
        with self.assertRaisesRegex(ValueError, "returns a double in D0"):
            arm64.encode_darwin_extern(Module([Store(0, call)], 4), 0x100004000)

    def test_an_integer_result_is_not_readable_as_a_float(self):
        call = ExternCall("getpid", (), "i32")
        with self.assertRaisesRegex(ValueError, "returns an integer word in"):
            arm64.encode_darwin_extern(
                Module([FloatStore(0, call)], 4), 0x100004000
            )


@unittest.skipUnless(
    _HOST_IS_DARWIN_ARM64, "native execution requires a darwin-arm64 host"
)
class NativeCapiExecutionTests(unittest.TestCase):
    """Build real C-API machine code, run it, and diff it against CPython."""

    def _build_and_compare(self, source: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.c"
            entry.write_text(_PROTOTYPES + source, encoding="utf-8")
            artifact = root / "program.bin"
            compile_c_native(entry, artifact, target="darwin-arm64", clean=True)
            self.assertEqual(artifact.read_bytes()[:4], b"\xcf\xfa\xed\xfe")

            interpreted_entry = root / "program.py"
            interpreted_entry.write_text(
                c_to_python_source(entry.read_text(encoding="utf-8"), str(entry)),
                encoding="utf-8",
            )
            native = subprocess.run([str(artifact)], capture_output=True, text=True)
            interpreted = subprocess.run(
                [sys.executable, str(interpreted_entry)],
                capture_output=True,
                text=True,
                env={"PYTHONPATH": _REPO_SRC, "PATH": "/usr/bin:/bin"},
            )
            self.assertEqual(
                native.stdout,
                interpreted.stdout,
                "native stdout differs from the interpreted run",
            )
            self.assertEqual(
                native.returncode,
                interpreted.returncode,
                f"native exit {native.returncode} != CPython exit {interpreted.returncode}",
            )
            return native

    def test_a_recursive_helper_drives_cpython_from_its_own_frame(self):
        """Recursion and the adapter ABI in one body.

        Each frame makes GOT-indirect calls, which clobber the argument
        registers and the link register, and then reads its own locals back --
        so a frame that was not saved and restored shows up as a wrong sum.
        1 + 2 + ... + 10 == 55.
        """

        native = self._build_and_compare(
            """
long long sum_via_python(long long n) {
    PyObject *a;
    PyObject *b;
    PyObject *s;
    long long r;
    if (n == 0) { return 0; }
    a = PyLong_FromLongLong(n);
    b = PyLong_FromLongLong(sum_via_python(n - 1));
    s = PyNumber_Add(a, b);
    r = PyLong_AsLongLong(s);
    Py_DecRef(a); Py_DecRef(b); Py_DecRef(s);
    return r;
}
int main(void) {
    long long r;
    Py_Initialize();
    r = sum_via_python(10);
    Py_Finalize();
    return r;
}
"""
        )
        self.assertEqual(native.returncode, 55)

    def test_building_adding_and_printing_python_objects(self):
        native = self._build_and_compare(
            "int main(void)\n"
            "{\n"
            "    PyObject *stream;\n"
            "    PyObject *left;\n"
            "    PyObject *right;\n"
            "    PyObject *total;\n"
            "    PyObject *text;\n"
            "    PyObject *label;\n"
            "    PyObject *equal;\n"
            "    Py_Initialize();\n"
            '    PySys_WriteStdout("compiled C is driving CPython\\n");\n'
            '    stream = PySys_GetObject("stdout");\n'
            "    if (stream == NULL) {\n"
            "        return 3;\n"
            "    }\n"
            "    left = PyLong_FromLongLong(20);\n"
            "    right = PyLong_FromLongLong(22);\n"
            "    total = PyNumber_Add(left, right);\n"
            "    text = PyObject_Str(total);\n"
            '    PyFile_WriteString("sum = ", stream);\n'
            "    PyFile_WriteObject(text, stream, 1);\n"
            '    PyFile_WriteString("\\n", stream);\n'
            '    label = PyUnicode_FromString("hand-written C");\n'
            "    PyFile_WriteObject(label, stream, 1);\n"
            '    PyFile_WriteString("\\n", stream);\n'
            "    equal = PyObject_RichCompare(total, right, 2);\n"
            "    if (PyObject_IsTrue(equal)) {\n"
            '        PySys_WriteStdout("unexpected equality\\n");\n'
            "    }\n"
            "    if (PyErr_Occurred() != NULL) {\n"
            "        return 4;\n"
            "    }\n"
            "    Py_DecRef(text);\n"
            "    Py_DecRef(label);\n"
            "    Py_DecRef(equal);\n"
            "    Py_DecRef(left);\n"
            "    Py_DecRef(right);\n"
            "    text = PyObject_Str(total);\n"
            "    Py_DecRef(text);\n"
            "    Py_Finalize();\n"
            "    return PyLong_AsLongLong(total);\n"
            "}\n"
        )
        self.assertEqual(native.returncode, 42)
        self.assertIn("sum = 42", native.stdout)
        self.assertIn("hand-written C", native.stdout)

    def test_handle_returning_helpers_loops_and_lists(self):
        native = self._build_and_compare(
            "PyObject *doubled(PyObject *value)\n"
            "{\n"
            "    return PyNumber_Add(value, value);\n"
            "}\n"
            "void show(PyObject *value, PyObject *stream)\n"
            "{\n"
            "    PyObject *text;\n"
            "    text = PyObject_Str(value);\n"
            "    PyFile_WriteObject(text, stream, 1);\n"
            '    PyFile_WriteString("\\n", stream);\n'
            "    Py_DecRef(text);\n"
            "}\n"
            "int main(void)\n"
            "{\n"
            "    PyObject *stream;\n"
            "    PyObject *items;\n"
            "    PyObject *item;\n"
            "    PyObject *twice;\n"
            "    long long index;\n"
            "    long long total;\n"
            "    Py_Initialize();\n"
            '    stream = PySys_GetObject("stdout");\n'
            "    items = PyList_New(0);\n"
            "    if (items == NULL) {\n"
            "        return 5;\n"
            "    }\n"
            "    index = 1;\n"
            "    while (index < 5) {\n"
            "        item = PyLong_FromLongLong(index);\n"
            "        twice = doubled(item);\n"
            "        show(twice, stream);\n"
            "        PyList_Append(items, twice);\n"
            "        Py_DecRef(item);\n"
            "        Py_DecRef(twice);\n"
            "        index += 1;\n"
            "    }\n"
            "    show(items, stream);\n"
            "    total = PyObject_Size(items);\n"
            "    Py_DecRef(items);\n"
            "    Py_Finalize();\n"
            "    return total;\n"
            "}\n"
        )
        self.assertEqual(native.returncode, 4)
        self.assertEqual(
            native.stdout.splitlines(), ["2", "4", "6", "8", "[2, 4, 6, 8]"]
        )

    def test_omitting_py_finalize_loses_buffered_output(self):
        """A documented divergence, pinned so it cannot change silently.

        ``sys.stdout`` is buffered inside the interpreter. Running the twin
        ``.py`` under ``python3`` flushes it at interpreter shutdown, but a
        compiled binary that returns from ``main`` without ``Py_Finalize``
        exits first and prints nothing. A gcc-built embedding behaves the same
        way, so this is not a miscompile -- but it is the one case where the
        compiled and interpreted runs legitimately disagree on stdout, and
        README/DETAILED_GUIDE both say so. The assertion below is what makes
        that prose checkable.
        """

        body = (
            "int main(void)\n"
            "{\n"
            "    Py_Initialize();\n"
            '    PySys_WriteStdout("buffered\\n");\n'
            "    return 0;\n"
            "}\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "program.c"
            entry.write_text(_PROTOTYPES + body, encoding="utf-8")
            artifact = root / "program.bin"
            compile_c_native(entry, artifact, target="darwin-arm64", clean=True)

            interpreted_entry = root / "program.py"
            interpreted_entry.write_text(
                c_to_python_source(entry.read_text(encoding="utf-8"), str(entry)),
                encoding="utf-8",
            )
            native = subprocess.run([str(artifact)], capture_output=True, text=True)
            interpreted = subprocess.run(
                [sys.executable, str(interpreted_entry)],
                capture_output=True,
                text=True,
                env={"PYTHONPATH": _REPO_SRC, "PATH": "/usr/bin:/bin"},
            )
            self.assertEqual(native.returncode, interpreted.returncode)
            self.assertEqual(native.stdout, "", "output appeared without Py_Finalize")
            self.assertEqual(interpreted.stdout, "buffered\n")

        # ...and adding Py_Finalize makes the two runs agree again.
        restored = self._build_and_compare(
            "int main(void)\n"
            "{\n"
            "    Py_Initialize();\n"
            '    PySys_WriteStdout("buffered\\n");\n'
            "    Py_Finalize();\n"
            "    return 0;\n"
            "}\n"
        )
        self.assertEqual(restored.stdout, "buffered\n")

    def test_extern_call_in_a_loop_condition_runs_every_iteration(self):
        """A hoisted condition would be a silent miscompile, so pin it."""

        native = self._build_and_compare(
            "int main(void)\n"
            "{\n"
            "    long long n;\n"
            "    Py_Initialize();\n"
            "    n = 0;\n"
            "    while (PyLong_AsLongLong(PyLong_FromLongLong(n)) < 3) {\n"
            '        PySys_WriteStdout("tick\\n");\n'
            "        n += 1;\n"
            "    }\n"
            "    Py_Finalize();\n"
            "    return n;\n"
            "}\n"
        )
        self.assertEqual(native.stdout, "tick\ntick\ntick\n")
        self.assertEqual(native.returncode, 3)

    def test_importing_a_module_and_calling_a_function(self):
        native = self._build_and_compare(
            "int main(void)\n"
            "{\n"
            "    PyObject *stream;\n"
            "    PyObject *module;\n"
            "    PyObject *function;\n"
            "    PyObject *argument;\n"
            "    PyObject *result;\n"
            "    PyObject *text;\n"
            "    Py_Initialize();\n"
            '    stream = PySys_GetObject("stdout");\n'
            '    module = PyImport_ImportModule("math");\n'
            "    if (module == NULL) {\n"
            "        return 6;\n"
            "    }\n"
            '    function = PyObject_GetAttrString(module, "sqrt");\n'
            "    argument = PyLong_FromLongLong(144);\n"
            "    result = PyObject_CallOneArg(function, argument);\n"
            "    text = PyObject_Str(result);\n"
            '    PyFile_WriteString("sqrt(144) = ", stream);\n'
            "    PyFile_WriteObject(text, stream, 1);\n"
            '    PyFile_WriteString("\\n", stream);\n'
            "    Py_IncRef(result);\n"
            "    Py_DecRef(result);\n"
            "    Py_DecRef(text);\n"
            "    Py_DecRef(result);\n"
            "    Py_DecRef(argument);\n"
            "    Py_DecRef(function);\n"
            "    Py_DecRef(module);\n"
            "    Py_Finalize();\n"
            "    return 0;\n"
            "}\n"
        )
        self.assertEqual(native.stdout, "sqrt(144) = 12.0\n")


_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES = _ROOT / "examples"


class Arm64ReachTests(unittest.TestCase):
    """A literal or a function address further away than ADR reaches.

    ADR carries a signed 21-bit *byte* displacement, so it reaches about a
    megabyte. A 113,000-line translation unit puts its string literals well
    past that, and the build stopped with "arm64 literal is outside ADR
    range". The ADRP/ADD pair reaches +/-4 GB, which is the whole address
    space these images occupy.
    """

    def _decode(self, adrp: int, add: int, instruction: int) -> int:
        """Recover the address the pair computes, straight from the encoding."""

        immlo = (adrp >> 29) & 3
        immhi = (adrp >> 5) & 0x7FFFF
        pages = (immhi << 2) | immlo
        if pages >= (1 << 20):
            pages -= 1 << 21
        offset = (add >> 10) & 0xFFF
        return (instruction & ~0xFFF) + (pages << 12) + offset

    def test_the_pair_reaches_what_adr_cannot(self):
        from py2bin.native.arm64 import _adrp_add

        instruction = 0x100004000
        for distance in (0x10, 0x100000, 0x400000, 0x10000000, -0x8000000):
            target = instruction + distance
            adrp, add = _adrp_add(0, instruction, target)
            self.assertEqual(
                self._decode(adrp, add, instruction),
                target,
                f"distance {distance:#x} does not round-trip",
            )

    def test_the_registers_are_encoded(self):
        from py2bin.native.arm64 import _adrp_add

        for register in (0, 1, 9):
            adrp, add = _adrp_add(register, 0x100004000, 0x100104010)
            self.assertEqual(adrp & 0x1F, register)
            self.assertEqual(add & 0x1F, register)
            self.assertEqual((add >> 5) & 0x1F, register)


class DocumentedSurfaceTests(unittest.TestCase):
    """The prose states a number and a list; both have to be the real ones."""

    _DOCUMENTS = ("README.md", "docs/DETAILED_GUIDE.md")

    def test_documented_entry_point_count_matches_the_vetted_table(self):
        expected = len(cabi._CPYTHON_SYMBOLS)
        for name in self._DOCUMENTS:
            with self.subTest(document=name):
                text = (_ROOT / name).read_text(encoding="utf-8")
                counts = re.findall(
                    r"table of (\d+) exported CPython entry points", text
                )
                self.assertTrue(counts, f"{name} no longer states the table size")
                for stated in counts:
                    self.assertEqual(int(stated), expected)

    def test_readme_lists_exactly_the_vetted_cpython_symbols(self):
        text = (_ROOT / "README.md").read_text(encoding="utf-8")
        start = text.index("A fixed table of")
        bullet = text[start : text.index("\n- ", start)]
        # `PyNumber_Add`/`Subtract`/`Multiply`/`TrueDivide` is a shorthand.
        listed = set(re.findall(r"`(Py[A-Za-z_]+)`", bullet))
        listed |= {
            "PyNumber_" + suffix
            for suffix in ("Subtract", "Multiply", "TrueDivide")
            if f"`{suffix}`" in bullet
        }
        listed -= {"PyObject", "Subtract", "Multiply", "TrueDivide"}
        self.assertEqual(listed, set(cabi._CPYTHON_SYMBOLS))


class ShippedCapiExampleTests(unittest.TestCase):
    """The README points at these two files, so they must stay true."""

    def test_the_c_example_and_its_python_twin_are_the_same_program(self):
        """The .py is the .c parsed by py2bin and printed back out."""

        translated = c_to_python_source(
            (_EXAMPLES / "capi_embedding.c").read_text(encoding="utf-8"),
            str(_EXAMPLES / "capi_embedding.c"),
        )
        shipped = ast.parse(
            (_EXAMPLES / "capi_embedding.py").read_text(encoding="utf-8")
        )
        # Only the hand-written module docstring may differ.
        self.assertIsInstance(shipped.body[0], ast.Expr)
        self.assertIsInstance(shipped.body[0].value, ast.Constant)
        del shipped.body[0]
        self.assertEqual(ast.dump(shipped), ast.dump(ast.parse(translated)))

    def test_the_example_is_rejected_for_a_non_darwin_arm64_target(self):
        from py2bin.native.compiler import compile_native

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                NativeCompileError, "only supported for target 'darwin-arm64'"
            ):
                compile_native(
                    _EXAMPLES / "capi_embedding.py",
                    Path(directory) / "out",
                    "linux-x86_64",
                    clean=True,
                )

    @unittest.skipUnless(
        _HOST_IS_DARWIN_ARM64, "native execution requires a darwin-arm64 host"
    )
    def test_both_entry_points_build_run_and_agree_with_cpython(self):
        interpreted = subprocess.run(
            [sys.executable, str(_EXAMPLES / "capi_embedding.py")],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": _REPO_SRC, "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(interpreted.stdout, "1\n4\n9\n16\n25\nisqrt(total) = 7\n")
        self.assertEqual(interpreted.returncode, 7)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # (1) canonical C in, machine code out.
            from_c = root / "from_c.bin"
            compile_c_native(
                _EXAMPLES / "capi_embedding.c",
                from_c,
                target="darwin-arm64",
                clean=True,
            )
            # (2) the Python twin through the ordinary native compiler.
            from py2bin.native.compiler import compile_native

            from_py = root / "from_py.bin"
            compile_native(
                _EXAMPLES / "capi_embedding.py",
                from_py,
                "darwin-arm64",
                clean=True,
            )
            for artifact in (from_c, from_py):
                with self.subTest(artifact=artifact.name):
                    native = subprocess.run(
                        [str(artifact)], capture_output=True, text=True
                    )
                    self.assertEqual(native.stdout, interpreted.stdout)
                    self.assertEqual(native.returncode, interpreted.returncode)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
