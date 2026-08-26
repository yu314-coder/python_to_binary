"""`build.py` shares a name with the PEP 517 frontend, and must not eat it.

`python -m build` puts the working directory first on the path, so from a
clone it finds this repository's `build.py` rather than the packaging tool.
What happened then was not an error: the three questions started, took
`--outdir` for the name of a program to compile, and wrote an app bundle and a
disk image into the repository. A command that means "make me a wheel" has to
either make one or say why not.

The file is not renamed because `python3 build.py` is how the project is
documented to be used with no install at all. It detects the collision
instead.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_script_knows_it_was_reached_as_the_module():
    # `__spec__` is what tells the two apart - None for a script, the module
    # name for `-m` - and it is read at import time, so the check has to be
    # made in a child rather than by importing build.py here.
    probe = (
        "import runpy, sys\n"
        "sys.argv = ['build', '--outdir', 'nowhere']\n"
        "module = runpy.run_path(%r)\n"
        "print(module['_shadowed_the_packaging_tool'].__name__)\n"
    ) % str(ROOT / "build.py")
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT
    )
    assert done.returncode == 0, done.stderr[-400:]
    assert "_shadowed_the_packaging_tool" in done.stdout


def test_running_it_as_a_module_does_not_start_the_three_questions():
    """The regression itself: `-m build` must not begin compiling something.

    Run from the repository, which is the whole point - that is what puts this
    directory first on the path. `-I` was tried here and is wrong: isolated
    mode drops the working directory too, so `build.py` is never reached and
    the test passes without exercising anything.

    `--help` rather than a real build, because whether the frontend is
    installed is not this test's business. Both endings are fine - the
    frontend's own usage, or the explanation of how to install it - and the
    third one is not: the questions this file asks.
    """

    done = subprocess.run(
        [sys.executable, "-m", "build", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    combined = (done.stdout + done.stderr).lower()
    handed_over = "usage: python -m build" in combined
    explained = "build.py" in combined and "pip install build" in combined
    assert handed_over or explained, combined[-400:]
    # What the old behaviour looked like, in its own words.
    for tell in ("which machine", "nothing to build", "one file:"):
        assert tell not in combined, f"the questions started: {combined[-300:]}"


def test_a_real_packaging_run_produces_a_wheel_not_an_app():
    """End to end, when the frontend is there to hand over to.

    Skipped rather than failed where it is not installed, because that is a
    fact about the machine. The assertion is on *what comes out*: the old
    behaviour wrote an `.app` and a `.dmg` into the repository and exited
    zero, which no assertion on the exit status alone would have caught.
    """

    if subprocess.run(
        [sys.executable, "-c", "import build"],
        capture_output=True,
        cwd=tempfile.gettempdir(),
    ).returncode:
        import pytest

        pytest.skip("the PEP 517 frontend is not installed here")

    with tempfile.TemporaryDirectory() as directory:
        done = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", directory],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert done.returncode == 0, (done.stdout + done.stderr)[-500:]
        written = sorted(p.name for p in Path(directory).iterdir())
        assert written and all(n.endswith(".whl") for n in written), written
        assert not any(
            p.suffix in (".app", ".dmg") for p in Path(directory).iterdir()
        )


def test_the_script_still_works_the_way_it_is_documented():
    done = subprocess.run(
        [sys.executable, str(ROOT / "build.py")],
        capture_output=True,
        text=True,
        input="\n",
        cwd=ROOT,
    )
    # Run inside py2bin's own directory it has nothing to offer, and says so
    # rather than raising - which is the same path a user takes before they
    # point it at a program.
    assert "py2bin" in done.stdout


def test_it_takes_every_answer_the_documented_way():
    """The entry point the readme gives people takes all of them.

    A build that is only reachable by typing at it is a build nothing can
    check, and `--define` was the one a header can ask for by name: `py2bin
    cc` has always taken it and this had not, so the one thing a `#error`
    tells an author to do could not be done the documented way.
    """

    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_build_py", root / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    where, target, method, includes, fetch, defines = module._read_arguments(
        [
            "app.cpp",
            "--target", "windows-x86_64",
            "--how", "compile",
            "-I", "vendor",
            "--include", "more",
            "--auto-fetch",
            "-D", "WANTED",
            "--define", "OTHER=3",
        ]
    )
    assert where == "app.cpp"
    assert target == "windows-x86_64"
    assert method == "compile"
    assert includes == ("vendor", "more")
    assert fetch is True
    assert defines == ("WANTED", "OTHER=3")


def test_an_option_with_nothing_after_it_is_reported():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_build_py2", root / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    where, *_rest = module._read_arguments(["app.c", "--define"])
    assert where is module._BAD
