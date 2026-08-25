"""C compiled alongside a compiled Python program, for the same machine.

An application is often not one language. `--native` compiles the C beside the
program and carries the executable with it, so a mixture of Python, C, headers
and web assets becomes one artifact through the tier that produces real machine
code rather than the one that ships an interpreter beside the source.

Compiled at build time rather than accepted already built, because nothing
about a finished executable says which machine it was for - and a helper built
for the build machine, dropped into a Windows bundle, is exactly the failure
this is meant to prevent.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from py2bin.cli import _build_native_helper
from py2bin.interactive import _NATIVE_DIRECTORIES, c_programs, c_sources_beside


_HEADER = "#ifndef SUM_H\n#define SUM_H\nint total(int n);\n#endif\n"
_LIB = '#include "sum.h"\nint total(int n) { return n * 2; }\n'
_MAIN = '#include <stdio.h>\n#include "sum.h"\nint main(void) { printf("%d\\n", total(21)); return 0; }\n'


def _project(root: Path) -> Path:
    native = root / "native"
    native.mkdir()
    (native / "sum.h").write_text(_HEADER, encoding="utf-8")
    (native / "sum.c").write_text(_LIB, encoding="utf-8")
    (native / "run.c").write_text(_MAIN, encoding="utf-8")
    return native


class BuildingTheHelper(unittest.TestCase):
    def test_a_directory_builds_the_program_it_holds(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            native = _project(root)
            beside = root / "out"
            beside.mkdir()
            built = _build_native_helper(native, beside, "darwin-arm64")
            self.assertEqual(built.name, "run")
            self.assertTrue(built.is_file())

    def test_the_helper_is_built_for_the_target_not_the_host(self) -> None:
        """The reason this compiles rather than copies.

        A plain `printf` is used rather than one with a conversion: py2bin's C
        compiler emits the write syscall for POSIX only, so a runtime
        conversion cannot be built for Windows. That is the C compiler's limit
        and not this option's, but a fixture has to stay inside it.
        """

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            native = root / "native"
            native.mkdir()
            (native / "sum.h").write_text(_HEADER, encoding="utf-8")
            (native / "sum.c").write_text(_LIB, encoding="utf-8")
            (native / "run.c").write_text(
                '#include <stdio.h>\n#include "sum.h"\n'
                'int main(void) { printf("done\\n"); return total(21); }\n',
                encoding="utf-8",
            )
            beside = root / "out"
            beside.mkdir()
            built = _build_native_helper(native, beside, "windows-x86_64")
            self.assertEqual(built.suffix, ".exe")
            self.assertEqual(built.read_bytes()[:2], b"MZ")

    def test_a_directory_with_no_main_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            native = root / "native"
            native.mkdir()
            (native / "sum.c").write_text(_LIB, encoding="utf-8")
            (native / "sum.h").write_text(_HEADER, encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                _build_native_helper(native, root, "darwin-arm64")

    def test_two_programs_in_one_directory_are_refused(self) -> None:
        """Joined into one translation unit they would be two mains."""

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            native = _project(root)
            (native / "other.c").write_text(_MAIN, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "more than one C program"):
                _build_native_helper(native, root, "darwin-arm64")


class FindingIt(unittest.TestCase):
    def test_only_the_file_with_a_main_is_the_program(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            native = _project(Path(scratch))
            self.assertEqual([p.name for p in c_programs(native)], ["run.c"])
            others, includes = c_sources_beside(native / "run.c")
            self.assertEqual([p.name for p in others], ["sum.c"])
            self.assertIn(str(native), includes)

    def test_native_directories_are_not_the_data_ones(self) -> None:
        # `web/` is carried as it is; `native/` is compiled. Confusing the two
        # would either compile a stylesheet or ship C source as an asset.
        from py2bin.interactive import _DATA_DIRECTORIES

        self.assertFalse(set(_NATIVE_DIRECTORIES) & set(_DATA_DIRECTORIES))
        self.assertIn("native", _NATIVE_DIRECTORIES)


if __name__ == "__main__":
    unittest.main()


class BuildScriptAnswers(unittest.TestCase):
    """`build.py` is the entry point the readme gives people.

    It asked three questions and had no way to be told the answers, so
    nothing could check it - and it had a bug the `cc` command did not.
    """

    def test_a_windows_build_gets_the_extension_that_makes_it_run(self) -> None:
        """Windows decides what is executable by the extension.

        `py2bin cc` has always added it. This path had not, so a program
        built the documented way came out unrunnable and nothing noticed.
        """

        import tempfile
        from pathlib import Path

        from py2bin.interactive import _build_c

        with tempfile.TemporaryDirectory() as scratch:
            program = Path(scratch) / "hello.cpp"
            program.write_text(
                '#include <stdio.h>\nint main(void){ printf("hi\\n"); return 0; }\n',
                encoding="utf-8",
            )
            self.assertEqual(_build_c(program, "windows-x86_64"), 0)
            self.assertTrue((program.parent / "dist" / "hello.exe").is_file())
            self.assertEqual(_build_c(program, "linux-x86_64"), 0)
            self.assertTrue((program.parent / "dist" / "hello").is_file())

    def test_the_three_questions_can_be_answered_in_advance(self) -> None:
        """Which is what lets a script, a test or the sweep use this path."""

        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as scratch:
            program = Path(scratch) / "hello.cpp"
            program.write_text(
                '#include <stdio.h>\nint main(void){ printf("hi\\n"); return 0; }\n',
                encoding="utf-8",
            )
            done = subprocess.run(
                [
                    sys.executable, str(root / "build.py"), str(program),
                    "--target", "windows-arm64",
                ],
                capture_output=True, text=True, timeout=600,
            )
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertTrue((program.parent / "dist" / "hello.exe").is_file())

    def test_a_machine_it_does_not_build_for_is_named(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        done = subprocess.run(
            [sys.executable, str(root / "build.py"), "x.cpp", "--target", "vax"],
            capture_output=True, text=True, timeout=300,
        )
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("darwin-arm64", done.stdout)
