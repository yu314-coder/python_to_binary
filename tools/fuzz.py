"""Generate random Python programs and check py2bin means what Python means.

The corpus in `tests/programs` is a corpus somebody wrote, which means it
covers what somebody thought of. This covers what nobody thought of: shapes
drawn at random from the grammar - nested control flow, classes, generators,
`try`/`except`/`finally`, comprehensions, f-strings, augmented assignment -
compiled, run, and compared against the interpreter character for character.

Two rules make the comparison possible at all, and both are about the
*program* being deterministic even though its shape is not:

  - nothing that prints an address, a time, or a set of strings. String
    hashing is randomised per process, so a set of them does not even iterate
    the same way twice under CPython.
  - every loop is bounded by a literal, and nothing that could be a loop
    counter is ever augmented. A program that does not stop cannot be
    compared with itself, let alone with anything else.

Run it:

    python3 tools/fuzz.py 1500

Seeds are program numbers, so a divergence is reproducible: `python3
tools/fuzz.py --only 253` writes that one program and nothing else.

Every difference this has ever found has been one thing - printing a function
object, where CPython says `<function f at 0x...>` and a compiled program
says `<built-in function f>`, because a compiled function is a `PyCFunction`.
That is the tier, and it is documented; see the README.
"""

import os
import pathlib
import random
import sys
import tempfile

INT = lambda r: str(r.randint(-20, 20))
STR = lambda r: repr(r.choice(["a", "bb", "ccc", "", "x y"]))

def expr(r, names, depth=0):
    choices = ["int", "str", "name", "list", "tuple", "dict"]
    if depth < 2:
        choices += ["bin", "cmp", "bool", "call", "cond", "sub", "comp", "fstr", "unary"]
    kind = r.choice(choices)
    if kind == "int": return INT(r)
    if kind == "str": return STR(r)
    if kind == "name": return r.choice(names) if names else INT(r)
    if kind == "list": return "[" + ", ".join(expr(r, names, depth+1) for _ in range(r.randint(0,3))) + "]"
    if kind == "tuple":
        n = r.randint(0, 3)
        inner = ", ".join(expr(r, names, depth+1) for _ in range(n))
        return "(" + inner + ("," if n == 1 else "") + ")"
    if kind == "dict":
        return "{" + ", ".join(f"{INT(r)}: {expr(r, names, depth+1)}" for _ in range(r.randint(0,2))) + "}"
    if kind == "bin":
        op = r.choice(["+", "-", "*", "//", "%", "&", "|", "^", "<<", ">>"])
        a, b = INT(r), INT(r)
        if op in ("//", "%") and b == "0": b = "3"
        if op in ("<<", ">>"): b = str(r.randint(0, 5))
        return f"({a} {op} {b})"
    if kind == "unary":
        return f"({r.choice(['-', '+', '~', 'not '])}{INT(r)})"
    if kind == "cmp":
        return f"({INT(r)} {r.choice(['<','<=','>','>=','==','!='])} {INT(r)})"
    if kind == "bool":
        return f"({expr(r,names,depth+1)} {r.choice(['and','or'])} {expr(r,names,depth+1)})"
    if kind == "cond":
        return f"({expr(r,names,depth+1)} if {INT(r)} > 0 else {expr(r,names,depth+1)})"
    if kind == "call":
        f = r.choice(["len", "str", "abs", "sorted", "list", "sum", "max", "min", "bool", "repr"])
        if f in ("len", "sorted", "list", "sum"): return f"{f}([{INT(r)}, {INT(r)}])"
        if f in ("max", "min"): return f"{f}({INT(r)}, {INT(r)})"
        if f == "abs": return f"abs({INT(r)})"
        return f"{f}({expr(r, names, depth+1)})"
    if kind == "sub":
        return f"[{INT(r)}, {INT(r)}, {INT(r)}][{r.randint(0,2)}]"
    if kind == "comp":
        return f"[v * {r.randint(1,3)} for v in range({r.randint(0,4)})]"
    if kind == "fstr":
        inner = r.choice([n for n in names if n[0] == "v"] or ["1"])
        return f'f"{{{INT(r)}}}-{{{inner}!r}}|{{{r.randint(0,9)}:>4}}"'
    return INT(r)

def block(r, names, indent, depth, budget):
    lines, made = [], list(names)
    for _ in range(r.randint(1, 3)):
        if budget[0] <= 0: break
        budget[0] -= 1
        kind = r.choice(["assign","print","if","for","while","try","func","aug","del","class","gen"]
                        if depth < 2 else ["assign","print","aug"])
        pad = "    " * indent
        if kind == "assign":
            n = f"v{len(made)}"; made.append(n)
            lines.append(f"{pad}{n} = {expr(r, made[:-1])}")
        elif kind == "aug" and [m for m in made if m[0] == "v"]:
            n = r.choice([m for m in made if m[0] == "v"])
            lines.append(f"{pad}{n} = {n}")
            lines.append(f"{pad}try:\n{pad}    {n} += {INT(r)}\n{pad}except TypeError:\n{pad}    pass")
        elif kind == "print":
            lines.append(f"{pad}print({expr(r, made)})")
        elif kind == "if":
            lines.append(f"{pad}if {expr(r, made)}:")
            lines.extend(block(r, made, indent+1, depth+1, budget))
            if r.random() < 0.5:
                lines.append(f"{pad}else:")
                lines.extend(block(r, made, indent+1, depth+1, budget))
        elif kind == "for":
            n = f"i{depth}"
            lines.append(f"{pad}for {n} in range({r.randint(0,3)}):")
            lines.extend(block(r, made + [n], indent+1, depth+1, budget))
        elif kind == "while":
            n = f"w{depth}"
            lines.append(f"{pad}{n} = 0")
            lines.append(f"{pad}while {n} < {r.randint(0,3)}:")
            lines.append(f"{pad}    {n} += 1")
            lines.extend(block(r, made + [n], indent+1, depth+1, budget))
        elif kind == "try":
            lines.append(f"{pad}try:")
            if r.random() < 0.5:
                lines.append(f"{pad}    raise {r.choice(['ValueError','KeyError','TypeError'])}({STR(r)})")
            else:
                lines.extend(block(r, made, indent+1, depth+1, budget))
            lines.append(f"{pad}except {r.choice(['ValueError','KeyError','TypeError','Exception'])} as e:")
            lines.append(f"{pad}    print('caught', type(e).__name__)")
            if r.random() < 0.4:
                lines.append(f"{pad}finally:")
                lines.append(f"{pad}    print('done')")
        elif kind == "func":
            n = f"f{depth}_{len(made)}"
            lines.append(f"{pad}def {n}(a, b={INT(r)}):")
            lines.append(f"{pad}    return {expr(r, ['a','b'])}")
            lines.append(f"{pad}print({n}({INT(r)}))")
            made.append(n)
        elif kind == "gen":
            n = f"g{depth}_{len(made)}"
            lines.append(f"{pad}def {n}():")
            lines.append(f"{pad}    for q in range({r.randint(0,3)}):")
            lines.append(f"{pad}        yield q * {r.randint(1,3)}")
            lines.append(f"{pad}print(list({n}()))")
        elif kind == "class":
            n = f"C{depth}_{len(made)}"
            lines.append(f"{pad}class {n}:")
            lines.append(f"{pad}    attr = {INT(r)}")
            lines.append(f"{pad}    def go(self, x):")
            lines.append(f"{pad}        return x + self.attr")
            lines.append(f"{pad}print({n}().go({INT(r)}), {n}.attr)")
        else:
            lines.append(f"{pad}pass")
    if not lines:
        lines.append("    " * indent + "pass")
    return lines

def program(seed):
    r = random.Random(seed)
    return "\n".join(block(r, [], 0, 0, [25])) + "\n"


def _run(program: pathlib.Path, work: pathlib.Path) -> str:
    """Compile, run, and say how it compares with the interpreter."""

    name = program.stem
    out = work / name
    if os.system(
        f"PYTHONPATH=src python3 -m py2bin compile-capi {program} "
        f"-o {out} --clean >/dev/null 2>&1"
    ):
        return "refused"
    reading = os.popen(f"cd /tmp && python3 {program} 2>/dev/null")
    wanted = reading.read()
    wanted_code = reading.close()
    reading = os.popen(f"cd /tmp && {out} 2>/dev/null")
    got = reading.read()
    got_code = reading.close()
    return "same" if (got, got_code) == (wanted, wanted_code) else "differs"


def main(argv: list[str]) -> int:
    count = 200
    only = None
    if "--only" in argv:
        only = int(argv[argv.index("--only") + 1])
    elif len(argv) > 1:
        count = int(argv[1])
    work = pathlib.Path(tempfile.mkdtemp(prefix="py2bin-fuzz-"))
    seeds = [only] if only is not None else list(range(count))
    tally = {"same": 0, "differs": 0, "refused": 0}
    for seed in seeds:
        source = work / f"fz{seed:05d}.py"
        source.write_text(program(seed), encoding="utf-8")
        if only is not None:
            print(source)
            return 0
        verdict = _run(source, work)
        tally[verdict] += 1
        if verdict != "same":
            print(f"{verdict}: seed {seed}  ({source})")
    print(f"{tally['same']} same, {tally['differs']} differ, "
          f"{tally['refused']} refused, out of {len(seeds)}")
    return 1 if tally["differs"] or tally["refused"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
