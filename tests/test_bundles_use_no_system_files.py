"""A build must not read anything off the machine that runs it.

py2bin carries its own C and C++ front ends, its own assembler, and its own
headers, so compiling a program should never reach for ``/usr/include``, a
host library, or a toolchain sitting on the build machine. That is easy to
say and easy to lose: one ``#include`` fallback added for convenience, one
library opened to read a symbol, and a build quietly starts depending on the
machine it happened to run on - and then produces different output, or none,
on a machine set up differently.

So the invariant is checked the only way it can be checked honestly: a build
is run with every file-opening call watched, and every path it touched is
compared against the three places it is allowed to touch - the program being
compiled, py2bin's own source, and the interpreter running the build.
"""

import builtins
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


class _Watched:
    """Every path opened or copied while this is entered."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def _note(self, path: object) -> None:
        try:
            self.paths.append(str(Path(os.fspath(path)).resolve()))  # type: ignore[arg-type]
        except (TypeError, ValueError, OSError):
            pass

    def __enter__(self) -> "_Watched":
        self._open = builtins.open
        self._path_open = Path.open
        self._copies = {
            name: getattr(shutil, name)
            for name in ("copy", "copy2", "copyfile", "copytree")
        }

        def watched_open(file, *arguments, **named):  # type: ignore[no-untyped-def]
            self._note(file)
            return self._open(file, *arguments, **named)

        def watched_path_open(target, *arguments, **named):  # type: ignore[no-untyped-def]
            self._note(target)
            return self._path_open(target, *arguments, **named)

        builtins.open = watched_open
        Path.open = watched_path_open  # type: ignore[method-assign]
        for name, real in self._copies.items():
            def watched_copy(source, *arguments, _real=real, **named):  # type: ignore[no-untyped-def]
                self._note(source)
                return _real(source, *arguments, **named)

            setattr(shutil, name, watched_copy)
        return self

    def __exit__(self, *_: object) -> None:
        builtins.open = self._open
        Path.open = self._path_open  # type: ignore[method-assign]
        for name, real in self._copies.items():
            setattr(shutil, name, real)

    def outside(self, *allowed: Path) -> list[str]:
        roots = [str(item.resolve()) for item in allowed]
        # The interpreter running the build reads its own standard library to
        # import py2bin at all; that is the test harness, not the build.
        roots.append(str(Path(sys.prefix).resolve()))
        roots.append(str(Path(sys.base_prefix).resolve()))
        # Scratch space is a place to put bytes, not a place to read a
        # toolchain out of.
        roots.append(str(Path(tempfile.gettempdir()).resolve()))
        return sorted(
            {
                path
                for path in self.paths
                if not any(path.startswith(root) for root in roots)
            }
        )


_C_PROGRAM = """\
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

struct point { int x; int y; };

int main(void) {
    struct point *p = malloc(sizeof *p);
    p->x = 3;
    p->y = 4;
    char buffer[32];
    snprintf(buffer, sizeof buffer, "%d,%d", p->x, p->y);
    printf("%s len=%zu\\n", buffer, strlen(buffer));
    free(p);
    return 0;
}
"""

_CPP_PROGRAM = """\
#include <cstdio>
#include <vector>
#include <string>

struct Shape {
    virtual ~Shape() {}
    virtual int sides() const = 0;
};

struct Square : Shape {
    int sides() const { return 4; }
};

struct Triangle : Shape {
    int sides() const { return 3; }
};

int main() {
    std::vector<Shape *> all;
    all.push_back(new Square());
    all.push_back(new Triangle());
    int total = 0;
    for (unsigned i = 0; i < all.size(); i++) {
        total += all[i]->sides();
    }
    std::string name = "sides";
    printf("%s %d\\n", name.c_str(), total);
    return 0;
}
"""


class BuildsReadNothingFromTheMachineTests(unittest.TestCase):
    def _build(self, name: str, source_text: str) -> None:
        from py2bin.cli import main

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / name
            source.write_text(source_text, newline="\n")
            output = work / "built"
            with _Watched() as watched:
                code = main(["cc", str(source), "-o", str(output)])
            self.assertEqual(code, 0, f"{name} did not build")
            self.assertTrue(output.exists(), f"{name} produced no artifact")
            self.assertEqual(
                watched.outside(PROJECT, work),
                [],
                f"compiling {name} read files off the build machine",
            )

    def test_a_c_program_is_compiled_without_touching_the_machine(self):
        self._build("program.c", _C_PROGRAM)

    def test_a_cpp_program_is_compiled_without_touching_the_machine(self):
        self._build("program.cpp", _CPP_PROGRAM)

    def test_a_header_is_never_looked_for_in_a_system_directory(self):
        """The search order is the source's directory, then what -I gave, then
        py2bin's own headers - and nothing after that."""
        from py2bin.c_preprocessor import Preprocessor

        engine = Preprocessor((), "linux-x86_64")
        self.assertEqual(engine.include_dirs, [])

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "needs.c"
            source.write_text('#include <nowhere_at_all.h>\nint main(void){return 0;}\n', newline="\n")
            from py2bin.cli import main

            with _Watched() as watched:
                code = main(["cc", str(source), "-o", str(work / "out")])
            self.assertNotEqual(code, 0, "a missing header must not resolve")
            self.assertEqual(
                watched.outside(PROJECT, work),
                [],
                "looking for a missing header searched the build machine",
            )


if __name__ == "__main__":
    unittest.main()
