from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from py2bin.native import (
    IRCanonicalCError,
    build_aot_application,
    emit_ir_c,
    parse_ir_c,
    roundtrip_ir_c,
    supported_targets,
)
from py2bin.native.ir import (
    ExitValue,
    IntBinary,
    IntCompare,
    IntConstant,
    IntLoad,
    Jump,
    JumpIfFalse,
    Label,
    Module,
    Store,
    Write,
)


class CanonicalWholeProgramCTests(unittest.TestCase):
    def test_every_ir_operation_round_trips_exactly(self):
        module = Module(
            [
                Store(0, IntConstant(3)),
                Label("loop"),
                Store(
                    0,
                    IntBinary("sub", IntLoad(0), IntConstant(1)),
                ),
                Write(b"\x00native\xff\n"),
                JumpIfFalse(
                    IntCompare("gt", IntLoad(0), IntConstant(0)),
                    "done",
                ),
                Jump("loop"),
                Label("done"),
                ExitValue(IntLoad(0)),
            ],
            stack_slots=1,
        )

        source, reconstructed = roundtrip_ir_c(module)

        self.assertEqual(reconstructed, module)
        self.assertIn('#include <stdio.h>', source)
        self.assertIn('fwrite("\\x00\\x6e\\x61', source)
        self.assertIn("goto py2bin_label_loop;", source)
        self.assertNotIn("Python", source)

    def test_parser_rejects_changed_or_undefined_control_flow(self):
        module = Module(
            [Jump("end"), Label("end"), ExitValue(IntConstant(0))],
        )
        source = emit_ir_c(module).replace(
            "goto py2bin_label_end;",
            "goto py2bin_label_missing;",
        )

        with self.assertRaisesRegex(
            IRCanonicalCError,
            "undefined native label 'missing'",
        ):
            parse_ir_c(source, "tampered.c")

    def test_local_library_round_trips_through_c_for_every_target(self):
        with tempfile.TemporaryDirectory() as directory:
            # /var is a symlink to /private/var on macOS and py2bin
            # reports resolved paths, so compare against resolved ones.
            root = Path(directory).resolve()
            library = root / "native_math.py"
            library.write_text(
                "def transform(value: int) -> int:\n"
                "    adjusted = value * 3\n"
                "    if adjusted > 6:\n"
                "        return adjusted - 1\n"
                "    return adjusted + 1\n"
                "def notify(flag: bool) -> None:\n"
                "    if flag:\n"
                "        print('library procedure')\n",
                encoding="utf-8",
            )
            entry = root / "main.py"
            entry.write_text(
                "from native_math import notify, transform\n"
                "total = 0\n"
                "for value in range(1, 5):\n"
                "    total += transform(value)\n"
                "notify(total > 0)\n"
                "raise SystemExit(total)\n",
                encoding="utf-8",
            )

            for target in supported_targets():
                direct = build_aot_application(
                    entry,
                    root / "direct" / target,
                    target=target,
                    source_root=root,
                )
                c_path = root / "canonical-c" / f"{target}.c"
                via_c = build_aot_application(
                    entry,
                    root / "via-c" / target,
                    target=target,
                    source_root=root,
                    via_c=True,
                    c_output=c_path,
                )

                self.assertEqual(
                    via_c.native.artifact.read_bytes(),
                    direct.native.artifact.read_bytes(),
                    target,
                )
                self.assertEqual(
                    via_c.attestation.pipeline,
                    "python-ir-c-ir-machine",
                )
                self.assertTrue(via_c.attestation.ir_roundtrip_verified)
                self.assertEqual(
                    via_c.attestation.canonical_c_sha256,
                    hashlib.sha256((via_c.c_source or "").encode("utf-8")).hexdigest(),
                )
                self.assertEqual(via_c.c_artifact, c_path)
                self.assertNotIn("from native_math", via_c.c_source or "")
                self.assertIn("py2bin_slot_", via_c.c_source or "")

    def test_blocked_library_writes_neither_c_nor_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            # /var is a symlink to /private/var on macOS and py2bin
            # reports resolved paths, so compare against resolved ones.
            root = Path(directory).resolve()
            entry = root / "main.py"
            entry.write_text(
                "import requests\n"
                "print('must fail')\n",
                encoding="utf-8",
            )
            output = root / "app.exe"
            c_output = root / "app.c"

            with self.assertRaisesRegex(ValueError, "strict CPython-free AOT plan"):
                build_aot_application(
                    entry,
                    output,
                    target="windows-x86_64",
                    source_root=root,
                    via_c=True,
                    c_output=c_output,
                )

            self.assertFalse(output.exists())
            self.assertFalse(c_output.exists())


if __name__ == "__main__":
    unittest.main()
