"""Compiling must not ask the operating system for anything a sandbox denies.

This is what lets an iPad cross-build a Windows `.exe`. py2bin has no
compiler, assembler or linker behind it - it writes the machine code and the
container itself - so a build is arithmetic and file writing, and an App Store
sandbox permits both. The claim only holds while nothing on the path reaches
for a process facility, and the way to keep it true is to take those
facilities away and build anyway.

`test_stdlib_only` already forbids *importing* `subprocess` under `src/`.
This is the dynamic half: the modules are made unimportable and the process
primitives are deleted from `os` before the compiler is loaded, so a call
added anywhere on the path fails here rather than on someone's tablet.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"

#: Removed before the compiler is imported. `ctypes` earns its place: it pulls
#: in `ctypes.util`, which shells out, and a phone Python may refuse it
#: outright - the FFI is deliberately kept off the compile path for that
#: reason.
DENIED = (
    "subprocess",
    "multiprocessing",
    "_posixsubprocess",
    "pty",
    "fcntl",
    "ctypes",
    "_ctypes",
    "distutils",
    "setuptools",
)

#: The other half: a module already imported cannot be un-imported, so the
#: primitives themselves go too.
REMOVED = ("fork", "execv", "execve", "posix_spawn", "system", "popen", "spawnv")

_PREAMBLE = f"""
import builtins, os, sys
DENIED = {DENIED!r}
_real = builtins.__import__
def guarded(name, *a, **k):
    if name in DENIED or name.split(".")[0] in DENIED:
        raise ImportError(name + " is unavailable in this sandbox")
    return _real(name, *a, **k)
builtins.__import__ = guarded
for gone in {REMOVED!r}:
    if hasattr(os, gone):
        delattr(os, gone)
sys.path.insert(0, {str(SOURCE)!r})
from py2bin.cli import main
sys.exit(main(sys.argv[1:]))
"""

#: One per container format, plus the tier that emits a whole program rather
#: than one driving CPython. The magic is checked rather than the exit status
#: alone, because a build that writes nothing also exits zero if it decides it
#: has nothing to do.
CASES = (
    ("compile-capi", "darwin-arm64", b"\xcf\xfa\xed\xfe"),
    ("compile-capi", "linux-arm64", b"\x7fELF"),
    ("compile-capi", "windows-x86_64", b"MZ"),
    ("compile", "darwin-arm64", b"\xcf\xfa\xed\xfe"),
)


@pytest.mark.parametrize("tier,target,magic", CASES)
def test_a_build_needs_no_process_facilities(tier, target, magic):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "sandbox.py").write_text(_PREAMBLE, encoding="utf-8")
        program = root / "prog.py"
        program.write_text("def scale(v): return v * 3\nprint(scale(2) + 7)\n")
        out = root / "out"
        done = subprocess.run(
            [
                sys.executable,
                str(root / "sandbox.py"),
                tier,
                str(program),
                "--target",
                target,
                "-o",
                str(out),
                "--clean",
            ],
            capture_output=True,
        )
        detail = (done.stdout + done.stderr).decode()[-600:]
        assert done.returncode == 0, detail
        assert out.is_file(), detail
        assert out.read_bytes()[: len(magic)] == magic, detail


def test_the_sandbox_itself_bites():
    """A guard on the guard.

    If the preamble stopped denying anything - a typo in a name, an import
    hook that no longer runs - every test above would pass while proving
    nothing at all.
    """

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "sandbox.py").write_text(_PREAMBLE, encoding="utf-8")
        probe = root / "probe.py"
        probe.write_text(
            "import os\n"
            "try:\n"
            "    import subprocess\n"
            "except ImportError:\n"
            "    print('denied', not hasattr(os, 'fork'))\n"
            "else:\n"
            "    print('LEAKED')\n",
            encoding="utf-8",
        )
        # The preamble ends by handing argv to py2bin, so it is read for its
        # effect and the probe run under it rather than through it.
        source = (root / "sandbox.py").read_text().split("sys.path.insert")[0]
        (root / "probe_all.py").write_text(
            source + probe.read_text(), encoding="utf-8"
        )
        done = subprocess.run(
            [sys.executable, str(root / "probe_all.py")], capture_output=True
        )
        assert done.stdout.strip() == b"denied True", done.stdout + done.stderr
