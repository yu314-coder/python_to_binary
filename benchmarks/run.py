#!/usr/bin/env python3
"""Measure `compile-capi` against the interpreter it links.

    python3 benchmarks/run.py            # the whole grid
    python3 benchmarks/run.py subscript  # one row

The README carries these numbers, and for a long time the harness that
produced them did not exist anywhere - it lived in a scratch directory and was
deleted, which made the grid a claim nobody could re-check. This is that
harness, kept.

**The interpreter matters and is not chosen here.** A compiled binary binds
whatever CPython the build found, which on this machine is the python.org
framework rather than the Homebrew one, and the two do not perform alike.
Timing against the wrong one is the easiest way to publish a wrong number, so
the compiled program is asked at run time which interpreter it is using and
that is the one the CPython column runs.

**Each case times only its own `bench()`**, inside the process, so neither
column pays for start-up - a process launch is ~12 ms and would drown a row
that takes 1 ms. The loop is in a function on purpose: module level is a
different scope with different rules, and no real program puts its hot loop
there.

Every case is run in a **fresh process, nine times, and the median taken** -
not nine timings inside one process, which would let one column warm up on the
other's cache state, and not the mean, which one slow scheduling hiccup moves.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = Path(__file__).resolve().parent / "cases"
REPEATS = 9

#: Appended to each case so it times itself and prints microseconds. The
#: import is inside the harness rather than the case so a case file stays a
#: plain program that can also just be run.
_DRIVER = """
if __name__ == "__main__":
    import time
    _started = time.perf_counter()
    bench()
    print(int((time.perf_counter() - _started) * 1_000_000))
"""


def _interpreter_the_binary_links(work: Path) -> str:
    """Ask a compiled program which CPython it ended up bound to."""

    probe = work / "probe.py"
    probe.write_text("import sys\nprint(sys.prefix)\n", encoding="utf-8")
    binary = work / "probe.bin"
    _compile(probe, binary)
    prefix = subprocess.run(
        [str(binary)], capture_output=True, text=True, check=True
    ).stdout.strip()
    for candidate in (
        Path(prefix) / "bin" / "python3",
        Path(prefix) / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(f"no interpreter found under {prefix}")


def _compile(source: Path, out: Path) -> None:
    done = subprocess.run(
        [
            sys.executable, "-m", "py2bin", "compile-capi",
            str(source), "-o", str(out), "--clean",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    if done.returncode:
        raise SystemExit(f"{source.name} did not compile:\n{done.stdout}{done.stderr}")


def _median_microseconds(command: list[str]) -> float:
    timings = []
    for _ in range(REPEATS):
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(f"{command[0]} failed:\n{done.stderr}")
        timings.append(float(done.stdout.strip().splitlines()[-1]))
    return statistics.median(timings)


def main(argv: list[str]) -> int:
    wanted = set(argv)
    cases = sorted(CASES.glob("*.py"))
    if wanted:
        cases = [case for case in cases if case.stem in wanted]
        if not cases:
            raise SystemExit(f"no such case; have {[c.stem for c in CASES.glob('*.py')]}")

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        reference = _interpreter_the_binary_links(work)
        print(f"CPython: {reference}")
        print(f"{REPEATS} fresh processes per column, median taken\n")
        print(f"  {'row':<24} {'py2bin':>9} {'CPython':>9} {'ratio':>8}")

        results = []
        for case in cases:
            source = work / case.name
            source.write_text(case.read_text() + _DRIVER, encoding="utf-8")
            binary = work / (case.stem + ".bin")
            _compile(source, binary)
            ours = _median_microseconds([str(binary)])
            theirs = _median_microseconds([reference, str(source)])
            label = case.stem.replace("_", " ")
            results.append((label, ours / 1000, theirs / 1000, theirs / ours))
            print(
                f"  {label:<24} {ours/1000:8.1f}ms {theirs/1000:8.1f}ms "
                f"{theirs/ours:7.2f}x"
            )

        results.sort(key=lambda row: -row[3])
        beating = sum(1 for row in results if row[3] > 1.0)
        holding = sum(1 for row in results if row[3] >= 0.80)
        print(f"\n  {beating} of {len(results)} beat CPython; {holding} at 0.80x or better")
        (Path(__file__).resolve().parent / "last-run.json").write_text(
            json.dumps(
                {"interpreter": reference,
                 "rows": [{"row": r[0], "py2bin_ms": round(r[1], 2),
                           "cpython_ms": round(r[2], 2), "ratio": round(r[3], 2)}
                          for r in results]},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
