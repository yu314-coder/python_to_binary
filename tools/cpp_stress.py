"""Time the C++ front end on a large program shaped like a real one.

The corpus is small programs, and cannot see a translator that has gone
quadratic: naming the members of members once put every member of every known
object into the receiver set, and a 900-line program took two minutes where
it had taken sixteen seconds. This writes such a program - many classes each
holding strings, vectors and maps, objects nested in objects, a long main
full of loops with breaks and member paths - checks its answer against what
the program computes, and reports how long `translate_unity` took.

    PYTHONPATH=src python3 tools/cpp_stress.py            # ~17 s is the baseline
    PYTHONPATH=src python3 tools/cpp_stress.py --write x.cpp  # keep the program
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

CLASSES = 30
MEMBERS = 6
STATEMENTS = 120


def program() -> str:
    lines = ["#include <stdio.h>", "#include <string>", "#include <vector>", "#include <map>", ""]
    for i in range(CLASSES):
        members = "".join(
            f'    std::string s{j} = "x{j}";\n    std::vector<int> v{j};\n'
            f"    std::map<std::string, int> m{j};\n"
            for j in range(MEMBERS)
        )
        lines.append(
            f"struct L{i} {{\n{members}"
            "    int count() { int n = 0; for (int k = 0; k < 6; ++k) "
            "{ if (k == 3) break; n += (int)s0.size(); } return n; }\n};"
        )
    lines.append("struct N1 { L1 leaf; L0 tail; int depth() { return 1; } };")
    for i in range(2, CLASSES):
        lines.append(f"struct N{i} {{ L{i} leaf; N{i - 1} tail; int depth() {{ return {i}; }} }};")
    body = ["int main(void) {", f"    N{CLASSES - 1} top;", "    int total = 0;"]
    for i in range(STATEMENTS):
        j = i % MEMBERS
        body.append(
            f"    for (int i{i} = 0; i{i} < 3; ++i{i}) {{ if (i{i} == 2) break; "
            "total += top.leaf.count() + top.tail.depth(); }"
        )
        body.append(
            f"    top.leaf.v{j}.push_back({i}); top.leaf.s{j} += \"y\"; "
            f"total += (int)top.leaf.s{j}.size() + top.leaf.v{j}[0];"
        )
    body.append('    printf("%d\\n", total);\n    return 0;\n}')
    return "\n".join(lines + body) + "\n"


def expected() -> int:
    """What main() prints, worked out the way the program works it out."""

    total = 0
    lengths = [2] * MEMBERS  # "x0".."x5" are two characters long
    first: list[int] = [-1] * MEMBERS  # what v{j}[0] holds: the first push
    for i in range(STATEMENTS):
        j = i % MEMBERS
        for _k in range(2):  # the loop runs twice before its break
            # count(): three rounds of s0.size() before its own break, on the
            # s0 as it is now - it grows by one every sixth statement below.
            total += 3 * lengths[0] + (CLASSES - 2)  # + top.tail.depth()
        if first[j] < 0:
            first[j] = i
        lengths[j] += 1
        total += lengths[j] + first[j]
    return total


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", metavar="FILE", help="also keep the program here")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from py2bin.cpp_frontend import translate_unity  # noqa: E402

    text = program()
    if args.write:
        Path(args.write).write_text(text)
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "stress.cpp"
        source.write_text(text)
        started = time.time()
        out = translate_unity((source,), (), "darwin-arm64")
        took = time.time() - started
    print(f"translate_unity: {took:.1f}s for {text.count(chr(10))} lines -> {len(out)} chars of C")
    print(f"the program should print {expected()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
