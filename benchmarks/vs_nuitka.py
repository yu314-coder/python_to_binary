#!/usr/bin/env python3
"""Whole-process timings against CPython and Nuitka.

    python3 benchmarks/vs_nuitka.py [--skip-nuitka] [case ...]

Different question from `run.py`, and measured differently on purpose. That
one times a hot loop inside the process, because a row taking a millisecond
would otherwise be buried by a twelve-millisecond start-up. This one times the
**whole process**, start-up included, because that is what someone running the
artifact actually waits for - and because start-up is a real difference
between the three: a `compile-capi` binary calls `Py_Initialize` directly,
CPython scans `sys.path` and unmarshals `__main__`, and a Nuitka standalone
bundle bootstraps its own tree first.

Nuitka is built `--standalone` with `--assume-yes-for-downloads` off, so a
build that wants to fetch something fails loudly instead of quietly changing
what is being measured. It is slow - a minute or so per case - which is why
`--skip-nuitka` exists for when only the other two columns are wanted.

The CPython column uses the interpreter the compiled binary actually binds,
not whichever `python3` happens to be first on PATH. On a machine with both a
python.org framework build and a Homebrew one those differ, and they do not
perform alike.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = Path(__file__).resolve().parent / "vs_nuitka"
REPEATS = 5


def _run_env() -> dict[str, str]:
    return {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}


def _py2bin(source: Path, out: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "py2bin", "compile-capi",
         str(source), "-o", str(out), "--clean"],
        capture_output=True, text=True, env=_run_env(),
    )
    if done.returncode:
        raise SystemExit(f"{source.name}: {done.stdout}{done.stderr}")


def _nuitka(source: Path, work: Path) -> Path | None:
    done = subprocess.run(
        [sys.executable, "-m", "nuitka", "--standalone", "--no-progress-bar",
         f"--output-dir={work}", str(source)],
        capture_output=True, text=True,
    )
    # macOS names it `.bin`; other platforms use the bare stem.
    folder = work / f"{source.stem}.dist"
    built = next(
        (c for c in (folder / f"{source.stem}.bin", folder / source.stem)
         if c.is_file()),
        folder / source.stem,
    )
    if done.returncode or not built.is_file():
        print(f"  (nuitka declined {source.stem}: "
              f"{(done.stderr or done.stdout).strip().splitlines()[-1][:70]})")
        return None
    return built


def _median_seconds(command: list[str]) -> float:
    timings = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        done = subprocess.run(command, capture_output=True)
        timings.append(time.perf_counter() - started)
        if done.returncode:
            raise SystemExit(f"{command[0]} failed: {done.stderr[-300:]!r}")
    return statistics.median(timings)


def _interpreter(work: Path) -> str:
    probe = work / "probe.py"
    probe.write_text("import sys\nprint(sys.prefix)\n", encoding="utf-8")
    binary = work / "probe.bin"
    _py2bin(probe, binary)
    prefix = subprocess.run([str(binary)], capture_output=True, text=True,
                            check=True).stdout.strip()
    for name in ("python3", "python"):
        candidate = Path(prefix) / "bin" / name
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(f"no interpreter under {prefix}")


def main(argv: list[str]) -> int:
    skip = "--skip-nuitka" in argv
    wanted = {a for a in argv if not a.startswith("--")}
    cases = sorted(CASES.glob("*.py"))
    if wanted:
        cases = [c for c in cases if c.stem in wanted]

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        reference = _interpreter(work)
        print(f"CPython: {reference}")
        print(f"whole process, median of {REPEATS}, seconds\n")
        print(f"  {'row':<22} {'this':>8} {'CPython':>9} {'Nuitka':>9}")
        for case in cases:
            source = work / case.name
            shutil.copyfile(case, source)
            binary = work / (case.stem + ".bin")
            _py2bin(source, binary)
            ours = _median_seconds([str(binary)])
            theirs = _median_seconds([reference, str(source)])
            built = None if skip else _nuitka(source, work)
            nuitka = _median_seconds([str(built)]) if built else None
            shown = f"{nuitka:9.3f}" if nuitka is not None else f"{'-':>9}"
            print(f"  {case.stem.replace('_',' '):<22} {ours:8.3f} "
                  f"{theirs:9.3f} {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
