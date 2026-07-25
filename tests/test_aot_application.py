from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from py2bin.cli import main
from py2bin.native import (
    AOTPlanError,
    build_aot_application,
    plan_aot_application,
)


class AOTApplicationTests(unittest.TestCase):
    def test_closed_world_local_program_is_buildable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "maths.py").write_text(
                "def twice(value: int) -> int:\n"
                "    return value * 2\n",
                encoding="utf-8",
            )
            entry = root / "main.py"
            entry.write_text(
                "from maths import twice\n"
                "total = 0\n"
                "for value in range(4):\n"
                "    total += twice(value)\n"
                "raise SystemExit(total)\n",
                encoding="utf-8",
            )

            plan = plan_aot_application(entry, source_root=root)

            self.assertTrue(plan.buildable, plan.blockers)
            self.assertEqual(plan.reachable_python, (entry, root / "maths.py"))
            self.assertEqual(plan.as_dict()["guarantees"]["uses_cpython"], False)

    def test_build_writes_pe_with_raw_hash_and_no_python_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "main.py"
            entry.write_text("print('native')\n", encoding="utf-8")
            output = root / "native.exe"
            proof_path = root / "native.attestation.json"

            result = build_aot_application(
                entry,
                output,
                target="windows-x86_64",
                attestation=proof_path,
            )

            data = output.read_bytes()
            self.assertEqual(data[:2], b"MZ")
            self.assertNotIn(entry.read_bytes(), data)
            self.assertNotIn(b"PY2BIN-ONEFILE-PAYLOAD-", data)
            self.assertNotIn(b"py2bin_bootstrap.py", data)
            self.assertEqual(
                result.attestation.sha256,
                hashlib.sha256(data).hexdigest(),
            )
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
            self.assertTrue(proof["cpython_free"])
            self.assertTrue(proof["python_payload_free"])
            self.assertTrue(proof["extraction_free"])
            self.assertEqual(proof["backend"], "py2bin-native-aot")

    def test_dynamic_code_and_unported_package_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "main.py"
            entry.write_text(
                "import webview\n"
                "answer = eval('40 + 2')\n"
                "print(answer)\n",
                encoding="utf-8",
            )
            output = root / "must-not-exist"

            plan = plan_aot_application(entry, source_root=root)

            self.assertFalse(plan.buildable)
            self.assertTrue(any("eval()" in item for item in plan.blockers))
            self.assertTrue(any("import webview" in item for item in plan.blockers))
            with self.assertRaises(AOTPlanError):
                build_aot_application(entry, output)
            self.assertFalse(output.exists())

    def test_web_assets_are_catalogued_but_not_embedded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "main.py"
            entry.write_text("print('logic')\n", encoding="utf-8")
            web = root / "web"
            web.mkdir()
            html = web / "index.html"
            html.write_text(
                "<html><body>separate-web-asset</body></html>\n",
                encoding="utf-8",
            )
            javascript = web / "app.js"
            javascript.write_text("window.ready = true;\n", encoding="utf-8")
            native_payload = root / "unused_engine.so"
            native_payload.write_bytes(b"native-engine-placeholder")

            plan = plan_aot_application(entry, source_root=root)
            result = build_aot_application(
                entry,
                root / "logic",
                target="darwin-arm64",
            )

            self.assertTrue(plan.buildable, plan.blockers)
            self.assertEqual(plan.web_assets, (javascript, html))
            self.assertEqual(plan.native_payloads, (native_payload,))
            artifact_data = result.native.artifact.read_bytes()
            self.assertNotIn(b"separate-web-asset", artifact_data)
            self.assertNotIn(entry.read_bytes(), artifact_data)

    def test_cli_strict_plan_and_build_never_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocked = root / "blocked.py"
            blocked.write_text("import requests\nprint('no')\n", encoding="utf-8")
            blocked_output = root / "blocked.exe"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                plan_status = main(
                    [
                        "aot-plan",
                        str(blocked),
                        "--strict",
                        "--json",
                    ]
                )
                build_status = main(
                    [
                        "aot-build",
                        str(blocked),
                        "-o",
                        str(blocked_output),
                        "--target",
                        "windows-x86_64",
                    ]
                )

            self.assertEqual(plan_status, 1)
            self.assertEqual(build_status, 2)
            self.assertFalse(blocked_output.exists())
            self.assertFalse(json.loads(stdout.getvalue())["buildable"])
            self.assertIn("strict CPython-free AOT plan", stderr.getvalue())

    def test_attestation_cannot_overwrite_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "main.py"
            entry.write_text("print('safe')\n", encoding="utf-8")
            output = root / "native"

            with self.assertRaisesRegex(
                ValueError,
                "attestation path must be different",
            ):
                build_aot_application(
                    entry,
                    output,
                    target="darwin-arm64",
                    attestation=output,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
