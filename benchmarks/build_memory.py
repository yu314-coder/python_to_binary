#!/usr/bin/env python3
"""What a build costs in memory, against Nuitka.

    python3 benchmarks/build_memory.py [--skip-nuitka] [--warm] [case ...]

py2bin writes the machine code, the object file and the container itself, in
Python. Nuitka writes C and hands it to Apple's clang. That difference should
show up in what a build *costs*, not only in what it produces, and this is
where it shows up: a compiler you cannot run because it needs more memory than
the machine has is not a compiler you can use.

**Peak resident set of the whole process tree**, sampled every 25 ms - the
parent and every descendant, summed. Summed rather than maxed on purpose:
Nuitka runs clang, and a build whose peak is one 400 MB compiler is a
different thing from a build whose peak is four of them at once. Sampling can
miss a spike shorter than the interval, so `ru_maxrss` for the largest single
waited-for child is reported beside it as a floor - if the sampled total ever
came in under that, the sampler missed something and the run says so.

**Nuitka's caches are disabled** (`--disable-cache=all`), and that is not a
handicap - it is the only way the column reproduces. Nuitka keeps a ccache and
a module cache under `~/Library/Caches/Nuitka`, so the first run of this
harness measured cold builds and every run after it measured warm ones: at
3,000 functions the same build came out at 1,522 MB once and 1,175 MB the next
time, which would have been published as a difference in the compilers. py2bin
has no build cache of any kind, so a cold Nuitka is the like-for-like
comparison. Nuitka's *re*builds are faster than these numbers; py2bin's are
not, because it has nothing to reuse.

The scaling case matters more than the small ones. A hundred-line program says
little; the interesting question is what happens to a build as the program
grows, which is the shape that decides whether a phone or a CI box with 2 GB
can run it at all.
"""

from __future__ import annotations

import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = Path(__file__).resolve().parent / "vs_nuitka"
INTERVAL = 0.025


def _tree_kilobytes(root_pid: int) -> int:
    """Resident set of `root_pid` and every descendant, in KB."""

    try:
        listing = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True
        ).stdout
    except OSError:
        return 0
    children: dict[int, list[int]] = {}
    resident: dict[int, int] = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, parent, rss = (int(p) for p in parts)
        children.setdefault(parent, []).append(pid)
        resident[pid] = rss
    total = 0
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        total += resident.get(pid, 0)
        pending.extend(children.get(pid, ()))
    return total


class _Sampler(threading.Thread):
    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.pid = pid
        self.peak = 0
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            self.peak = max(self.peak, _tree_kilobytes(self.pid))
            self.stop.wait(INTERVAL)


def _build(command: list[str], env: dict[str, str] | None = None):
    """Run a build; answer (seconds, peak tree MB, largest child MB, ok)."""

    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.perf_counter()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    sampler = _Sampler(process.pid)
    sampler.start()
    output = process.communicate()[0]
    sampler.stop.set()
    sampler.join()
    seconds = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # `ru_maxrss` for RUSAGE_CHILDREN is a running maximum over every child
    # this process has ever waited for, so it only says something about *this*
    # build when it rose. Comparing it any other way reports a previous
    # build's clang as if it were this one's - which it did, at first.
    scale = 1 if sys.platform == "darwin" else 1024
    largest = after * scale / 1e6 if after > before else 0.0
    return (
        seconds,
        sampler.peak / 1000,
        largest,
        process.returncode == 0,
        output.decode(errors="replace")[-400:],
    )


#: Statements in the driver, held constant while the function count grows.
#: Growing both was the first attempt and it measured the wrong thing: py2bin
#: gave one C function a fixed 512 KB frame it could not reuse, so a single
#: `main()` of about 1,800 statements was refused outright - independent of
#: how large the *program* was, while 3,000 separate `def`s compiled without
#: complaint. That limit is gone as of 0.8.7, but the case stays as written:
#: the axis that matters for a real program is how many functions it has, not
#: how long one of them is.
_DRIVER_CALLS = 200


def _scaling_case(work: Path, functions: int) -> Path:
    """A program that is large in the way a real one is: many small functions."""

    lines = []
    for n in range(functions):
        lines.append(f"def step_{n}(v):")
        lines.append(f"    return v * {n % 7 + 2} + {n % 13}")
    lines.append("def main():")
    lines.append("    t = 1")
    for n in range(_DRIVER_CALLS):
        lines.append(f"    t = step_{n % max(functions, 1)}(t) % 1000003")
    lines.append("    print(t)")
    lines.append("main()")
    source = work / f"scaling_{functions}.py"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return source


def main(argv: list[str]) -> int:
    skip = "--skip-nuitka" in argv
    warm = "--warm" in argv
    wanted = {a for a in argv if not a.startswith("--")}

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        sources = []
        for case in sorted(CASES.glob("*.py")):
            if wanted and case.stem not in wanted:
                continue
            target = work / case.name
            shutil.copyfile(case, target)
            sources.append((case.stem, target))
        if not wanted:
            for count in (200, 1000, 3000):
                sources.append(
                    (f"scaling: {count} functions", _scaling_case(work, count))
                )

        print(f"peak resident set of the build's whole process tree, "
              f"sampled every {int(INTERVAL * 1000)} ms")
        print(f"nuitka: {'cache left in place (warm)' if warm else 'caches disabled (cold)'}"
              f"; py2bin has no build cache either way\n")
        print(f"  {'case':<26} {'py2bin':>9} {'time':>7}   "
              f"{'Nuitka':>9} {'time':>7}")
        for label, source in sources:
            out = work / (source.stem + ".out")
            ours = _build(
                [sys.executable, "-m", "py2bin", "compile-capi",
                 str(source), "-o", str(out), "--clean"],
                env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
            )
            if not ours[3]:
                print(f"  {label:<26} py2bin failed: {ours[4].strip()[-90:]}")
                continue
            if skip:
                print(f"  {label:<26} {ours[1]:8.0f}M {ours[0]:6.1f}s"
                      f"   {'-':>9} {'-':>7}")
                continue
            nuitka_command = [
                sys.executable, "-m", "nuitka", "--standalone",
                "--no-progress-bar", f"--output-dir={work}", str(source),
            ]
            if not warm:
                nuitka_command.insert(4, "--disable-cache=all")
            theirs = _build(nuitka_command)
            shown = (f"{theirs[1]:8.0f}M {theirs[0]:6.1f}s"
                     if theirs[3] else f"{'failed':>9} {'-':>7}")
            print(f"  {label:<26} {ours[1]:8.0f}M {ours[0]:6.1f}s   {shown}")
            for who, result in (("py2bin", ours), ("nuitka", theirs)):
                if result[3] and result[1] < result[2]:
                    print(f"      (note: {who} sampler saw {result[1]:.0f}M but "
                          f"a single child reached {result[2]:.0f}M - "
                          f"the peak was shorter than the sampling interval)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
