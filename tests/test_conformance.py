"""Every program in `tests/programs` answers exactly what CPython answers.

The suite elsewhere checks that py2bin does what somebody meant. This checks
something narrower and harder: that the compiled program *means* what Python
means - same stdout, same exit code, character for character - which is the
claim everything else rests on.

Compiling is slow, so this runs the programs in parallel and only on the host
whose C-API binding is wired up. The same corpus is run in CI, where it is
the gate that a release has to pass.
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_PROGRAMS = Path(__file__).parent / "programs"
_ROOT = Path(__file__).resolve().parents[1]
_HOST_IS_DARWIN_ARM64 = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)


def _compiled(program: Path, into: Path):
    """Compile `program`, run it, and answer with what it printed."""
    from py2bin.cli import main as build

    out = into / program.stem
    code = build(
        ["compile-capi", str(program), "-o", str(out), "--clean"]
    )
    if code != 0:
        return None
    # Through the shell rather than subprocess: this file may not import it,
    # and what is wanted is the bytes the program wrote.
    reading = os.popen(f"{out} 2>/dev/null")
    return reading.read(), reading.close()


class ConformanceTests(unittest.TestCase):
    def test_every_program_answers_what_cpython_answers(self):
        if not _HOST_IS_DARWIN_ARM64:
            self.skipTest("needs the host whose C-API binding is wired up")
        programs = sorted(_PROGRAMS.glob("*.py"))
        self.assertGreater(len(programs), 40, "the corpus went missing")
        with tempfile.TemporaryDirectory() as directory:
            into = Path(directory)

            def check(program: Path):
                reading = os.popen(f"{sys.executable} {program} 2>/dev/null")
                expected, closed = reading.read(), reading.close()
                got = _compiled(program, into)
                return program.name, expected, closed, got

            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
                for name, expected, closed, got in pool.map(check, programs):
                    with self.subTest(program=name):
                        self.assertIsNotNone(got, f"{name} was refused")
                        actual, exited = got
                        self.assertEqual(actual, expected)
                        self.assertEqual(exited, closed)
