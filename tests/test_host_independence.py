"""Artifacts must not depend on which OS built them.

py2bin cross-builds Windows, Linux, and macOS artifacts from any host, so a
build on Windows and a build on Linux have to emit the same bytes. Text-mode
writes break that silently: ``Path.write_text`` translates ``\\n`` to
``os.linesep``, so the same call produces CRLF on Windows and LF everywhere
else. That corrupts a ``#!/bin/sh`` launcher's shebang, changes wheel metadata,
and shifts every generated file's hash by host.

Enforced statically, in the same spirit as the stdlib-only invariant: reading
the AST catches the mistake at the call site instead of waiting for one of the
handful of paths that happen to be covered by an end-to-end test.
"""

import ast
from pathlib import Path
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "py2bin"


class HostIndependentArtifactTests(unittest.TestCase):
    def test_every_text_write_pins_its_line_ending(self):
        missing: list[str] = []
        for source in sorted(SOURCE_ROOT.rglob("*.py")):
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if not isinstance(function, ast.Attribute) or function.attr != "write_text":
                    continue
                keywords = {kw.arg for kw in node.keywords}
                if "newline" not in keywords:
                    missing.append(f"{source.name}:{node.lineno} write_text without newline=")
        self.assertEqual(
            missing,
            [],
            "write_text must pass newline= so artifacts are identical on every build host",
        )

    def test_generated_shell_launcher_is_written_with_unix_line_endings(self):
        """The shebang is the sharp edge: '#!/bin/sh\\r' is not an interpreter."""
        import tempfile

        from py2bin.freezer import _shell_launcher

        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "launcher"
            _shell_launcher(launcher, Path("runtime/bin/python3"), {})
            raw = launcher.read_bytes()

        self.assertNotIn(b"\r\n", raw)
        self.assertTrue(raw.startswith(b"#!/bin/sh\n"), raw[:32])


if __name__ == "__main__":
    unittest.main()
