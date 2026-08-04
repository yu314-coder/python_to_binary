# benchmarks

The numbers in the project README come from here. They used to come from a
scratch directory that was deleted, which made every figure a claim nobody
could re-check - including me.

```sh
python3 benchmarks/run.py              # the 17-row grid against CPython
python3 benchmarks/run.py subscript    # one row
python3 benchmarks/vs_nuitka.py        # whole-process, against CPython and Nuitka
python3 benchmarks/vs_nuitka.py --skip-nuitka
```

## The machine the published numbers were taken on

| | |
|---|---|
| machine | Apple **M4**, 10 cores (4 performance, 6 efficiency) |
| memory | 24 GB |
| OS | macOS 27.0 (build 26A5388g), arm64 |
| CPython | **3.14.3**, python.org framework build (`/Library/Frameworks`) |
| Nuitka | **4.1.3**, `--standalone`, driving Apple's clang |
| py2bin | its own C compiler and code generator - no clang, no linker |

**The interpreter is not a free choice.** A `compile-capi` binary binds
whichever CPython the build found. This machine also has a Homebrew 3.14.3,
and the two do not perform alike - on the same case CPython measured 5.3 ms
under the framework build and 8.5 ms under Homebrew's. Timing the compiled
binary against the wrong one is the easiest way to publish a number that is
not true, so both harnesses compile a probe, ask *it* which interpreter it
ended up using, and time that one. Neither harness lets you pick.

## Two suites, because there are two questions

`run.py` times **only the hot loop, inside the process**. A row that takes a
millisecond would otherwise be buried under ten milliseconds of start-up.
Nine fresh processes per column, median taken - not nine timings inside one
process, which would let one column warm up on the other's cache state.

`vs_nuitka.py` times the **whole process**, because that is what someone
running the artifact waits for, and because start-up is a genuine difference
between the three. Read those rows knowing that: two of the five finish in
under thirty milliseconds, where start-up is much of the total.

## Writing a case

Put the work in a function called `bench()` (or `main()`, for `vs_nuitka/`).
Two ways to accidentally measure something else, both of which caught me:

- **A loop bound read from a global.** `while i < N` with a module-level `N`
  measures a scope limitation, not the loop - names at module scope are not
  narrowed into registers, and it cost 1.41× → 0.73× on the same loop. Use a
  literal.
- **A helper that ends up nested.** Wrapping a whole program in `def main():`
  turns a module-level `def add` into a closure, so the call stops being a
  direct one and the row silently measures something else.

A case should isolate the thing it is named after. `comparisons` and
`and`/`or` are separate rows for that reason: a plain comparison runs at
1.22× and adding one `and` drops it to 0.67×, so a single row holding both
would have reported neither.
