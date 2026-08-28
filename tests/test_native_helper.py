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

    def test_a_directory_of_c_is_compiled_rather_than_carried(self) -> None:
        # Assets are carried as they are; C is compiled. Confusing the two
        # would either compile a stylesheet or ship C source as an asset.
        # Told apart by what is inside, not by the name: a program's assets
        # live in `web`, `ui`, `frontend` or whatever the author called them.
        from py2bin.interactive import _worth_carrying

        with tempfile.TemporaryDirectory() as where:
            here = Path(where)
            (here / "native").mkdir()
            (here / "native" / "helper.c").write_text(
                "int main(void) { return 0; }\n"
            )
            (here / "native" / "notes.txt").write_text("read me\n")
            (here / "frontend").mkdir()
            (here / "frontend" / "index.html").write_text("<h1>hi</h1>\n")

            self.assertFalse(_worth_carrying(here / "native"))
            self.assertTrue(_worth_carrying(here / "frontend"))
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
        """A real file, because a missing one sends it down another path.

        Named with a path that exists: given one that does not, `build.py`
        searches for something to build starting from the *working
        directory*, so what this asserted depended on where the suite
        happened to be - which made it pass alone and fail in company.
        """

        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as scratch:
            program = Path(scratch) / "hello.cpp"
            program.write_text("int main(void){ return 0; }\n", encoding="utf-8")
            done = subprocess.run(
                [
                    sys.executable, str(root / "build.py"), str(program),
                    "--target", "vax",
                ],
                capture_output=True, text=True, timeout=300,
            )
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("darwin-arm64", done.stdout)


class IncludeDirectories(unittest.TestCase):
    """A project's own headers, and where py2bin agrees to look for them."""

    def _project(self, scratch: "Path") -> "Path":
        (scratch / "include").mkdir()
        (scratch / "vendor").mkdir()
        (scratch / "include" / "near.h").write_text(
            "#ifndef N\n#define N\nint near_by(void);\n#endif\n", encoding="utf-8"
        )
        (scratch / "vendor" / "far.h").write_text(
            "#ifndef F\n#define F\nint far_away(void);\n#endif\n", encoding="utf-8"
        )
        program = scratch / "app.cpp"
        program.write_text(
            '#include <stdio.h>\n#include "near.h"\n#include "far.h"\n'
            "int near_by(void){ return 1; }\nint far_away(void){ return 2; }\n"
            "int main(){ printf(\"%d\\n\", near_by() + far_away()); return 0; }\n",
            encoding="utf-8",
        )
        return program

    def test_a_folder_beside_the_program_needs_no_saying(self) -> None:
        import tempfile
        from pathlib import Path

        from py2bin.interactive import _build_c

        with tempfile.TemporaryDirectory() as scratch:
            program = self._project(Path(scratch))
            # `far.h` is not beside it, so this one should not build yet.
            self.assertNotEqual(_build_c(program, "linux-x86_64"), 0)

    def test_a_folder_named_on_the_command_line_is_searched_first(self) -> None:
        """A vendored header that lives somewhere else can be reached.

        `py2bin cc` has always taken `--include-dir`; `build.py`, which is
        the entry point the readme gives people, had no way to say it.
        """

        import tempfile
        from pathlib import Path

        from py2bin.interactive import _build_c

        with tempfile.TemporaryDirectory() as scratch:
            program = self._project(Path(scratch))
            self.assertEqual(
                _build_c(program, "linux-x86_64", (str(Path(scratch) / "vendor"),)),
                0,
            )
            self.assertTrue((program.parent / "dist" / "app").is_file())

    def test_the_missing_header_message_says_what_to_do(self) -> None:
        """It is read at the moment someone most needs to read it."""

        import tempfile
        from pathlib import Path

        from py2bin.c_frontend import CCompileError
        from py2bin.c_native import compile_c_native

        with tempfile.TemporaryDirectory() as scratch:
            program = Path(scratch) / "a.cpp"
            program.write_text('#include "nowhere.h"\nint main(){return 0;}\n',
                               encoding="utf-8")
            with self.assertRaises(CCompileError) as caught:
                compile_c_native(
                    program, Path(scratch) / "a.bin",
                    target="linux-x86_64", clean=True,
                )
        message = str(caught.exception)
        self.assertIn("--include", message)
        # And no path is listed twice.
        listed = [
            line.strip() for line in message.splitlines()
            if line.strip().endswith("nowhere.h")
        ]
        self.assertEqual(len(listed), len(set(listed)))


class OneFileMatchesWhatItLaunches(unittest.TestCase):
    """The launcher in front of a program looks like that program.

    A console launcher wrapping a desktop program flashes a black rectangle
    on every start, and nobody passes a flag for a thing they did not know
    was happening - so it is read off the image being packed.
    """

    def _pe(self, subsystem: int) -> bytes:
        # The smallest thing `_is_windowed` has to read: a DOS stub pointing
        # at a PE header whose optional header names a subsystem.
        image = bytearray(b"MZ" + b"\0" * 0x3E)
        image += b"\0" * (0x40 - len(image))
        image[0x3C:0x40] = (0x40).to_bytes(4, "little")
        header = bytearray(b"PE\0\0" + b"\0" * 20 + b"\0" * 96)
        header[24 + 68: 24 + 70] = subsystem.to_bytes(2, "little")
        return bytes(image) + bytes(header)

    def test_a_desktop_program_gets_a_desktop_launcher(self) -> None:
        from py2bin.onefile import _is_windowed

        with tempfile.TemporaryDirectory() as where:
            windowed = Path(where) / "gui.exe"
            windowed.write_bytes(self._pe(2))
            console = Path(where) / "cli.exe"
            console.write_bytes(self._pe(3))
            self.assertTrue(_is_windowed(windowed))
            self.assertFalse(_is_windowed(console))

    def test_something_that_is_not_a_pe_is_not_windowed(self) -> None:
        from py2bin.onefile import _is_windowed

        with tempfile.TemporaryDirectory() as where:
            elf = Path(where) / "prog"
            elf.write_bytes(b"\x7fELF" + b"\0" * 64)
            self.assertFalse(_is_windowed(elf))
            self.assertFalse(_is_windowed(Path(where) / "nothing-here"))
