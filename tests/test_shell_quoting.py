"""Nothing py2bin writes into a shell command may be read as shell.

Two places build a command out of values someone else supplied: the launcher
scripts, which paste `--python` and the bundle's name into `/bin/sh`, and the
bootstrapper's PowerShell fallback, which pasted a URL into a quoted argument.

Neither was a way in for an attacker who did not already have one - to steer
the URL you must already control the package about to be installed, and
`--python` is supplied by whoever is running the build. They are closed
because a value that reaches a shell uninspected is a bug waiting for a
context where it does matter, not because either was reachable.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError as missing:  # pragma: no cover - no pytest here
    # The suite is meant to run with nothing installed, and the README says
    # so. This module wants pytest's fixtures; where there is no pytest, say
    # that rather than failing to import - `unittest discover` reports a
    # module that will not import as an error, which reads like a broken
    # test rather than a missing tool.
    import unittest as _unittest

    raise _unittest.SkipTest("pytest is not installed") from missing

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "given",
    [
        "/usr/bin/env python3",
        "python3",
        "/opt/python/bin/python3.12",
        "/usr/bin/env python3.13",
    ],
)
def test_ordinary_interpreters_are_unchanged(given):
    # Why this cannot simply be `shlex.quote`: the default is
    # `/usr/bin/env python3`, two words. Quoting the pair whole would ask the
    # shell for one executable whose name contains a space, and every default
    # build would stop working.
    from py2bin.builder import _runtime_command

    assert _runtime_command(given) == given


@pytest.mark.parametrize(
    "given",
    [
        "python3; touch /tmp/py2bin-pwned",
        "python3 && touch /tmp/py2bin-pwned",
        "python3 $(touch /tmp/py2bin-pwned)",
        "python3 `touch /tmp/py2bin-pwned`",
        "python3 | tee /tmp/py2bin-pwned",
    ],
)
def test_a_command_cannot_smuggle_shell_syntax(given):
    from py2bin.builder import _runtime_command

    written = _runtime_command(given)
    # Whatever survives is a word for `exec` to look up, never syntax for the
    # shell to act on. Checked by asking a shell to expand it and seeing that
    # the operators came back as ordinary text.
    shown = subprocess.run(
        ["/bin/sh", "-c", f"printf '%s\\n' {written}"],
        capture_output=True,
        text=True,
    )
    assert shown.returncode == 0, shown.stderr
    assert not Path("/tmp/py2bin-pwned").exists()
    words = shown.stdout.split("\n")
    assert any(
        any(character in word for character in ";&|$`")
        for word in words
        if word
    ), f"the metacharacter was consumed rather than kept literal: {written}"


def test_an_unreadable_command_is_refused_rather_than_pasted():
    from py2bin.builder import _runtime_command

    with pytest.raises(ValueError):
        _runtime_command('python3 "unbalanced')
    with pytest.raises(ValueError):
        _runtime_command("   ")


def _built(fmt: str, work: Path, python: str = "/usr/bin/env python3") -> Path:
    source = work / "src"
    source.mkdir()
    (source / "app.py").write_text("print('ok')\n", encoding="utf-8")
    out = work / "build"
    out.mkdir()
    done = subprocess.run(
        [
            sys.executable, "-m", "py2bin", "build", str(source / "app.py"),
            "--format", fmt, "--output", str(out / "prog"),
            "--python", python, "--clean",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert done.returncode == 0, (done.stdout + done.stderr)[-400:]
    return out


def test_the_directory_launcher_runs_and_quotes_what_it_was_given():
    with tempfile.TemporaryDirectory() as directory:
        out = _built("dir", Path(directory))
        launcher = out / "prog.run"
        text = launcher.read_text()
        assert "/usr/bin/env python3" in text
        ran = subprocess.run([str(launcher)], capture_output=True, text=True)
        assert ran.stdout.strip() == "ok", ran.stderr[-300:]


def test_a_hostile_interpreter_does_not_execute_from_the_launcher():
    marker = Path(tempfile.gettempdir()) / "py2bin-launcher-pwned"
    if marker.exists():
        marker.unlink()
    with tempfile.TemporaryDirectory() as directory:
        out = _built("dir", Path(directory), python=f"python3; touch {marker}")
        ran = subprocess.run(
            [str(out / "prog.run")], capture_output=True, text=True
        )
        # It fails - there is no program called `python3;` - and that is the
        # point: it failed instead of running the second command.
        assert ran.returncode != 0
        assert not marker.exists(), "the launcher ran what was smuggled in"


def test_the_powershell_fallback_interpolates_nothing():
    """The URL reaches PowerShell through the environment, not the command.

    `Invoke-WebRequest -Uri '{url}'` put the URL inside a quoted argument that
    PowerShell parses itself, so a single quote in it ended the argument and
    what followed was PowerShell's to run. Reading the command back is the
    test: there is no placeholder left to fill.
    """

    module = runpy.run_path(str(ROOT / "get-py2bin.py"))
    downloaders = dict(
        (name, parts) for name, parts in module["_DOWNLOADERS"]
    )
    powershell = " ".join(downloaders["powershell"])
    assert "{url}" not in powershell and "{out}" not in powershell
    assert "$env:PY2BIN_FETCH_URL" in powershell
    assert "$env:PY2BIN_FETCH_OUT" in powershell
    # The others take an argument vector, which no shell reads, so they may
    # keep the placeholders.
    assert "{url}" in " ".join(downloaders["curl"])
