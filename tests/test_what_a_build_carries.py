"""What a compiled program needs beside it, and how it is found.

Both of these failed silently: the build reported success and produced a
program that could not start, or one that started with nothing to show. A
build that is wrong about what it carries says nothing at all, which is why
they are pinned here rather than left to a sweep that only checks meaning.
"""

import os
import tempfile
import unittest
from pathlib import Path

from py2bin import header_fetch, interactive


class WhatTheSourcesName(unittest.TestCase):
    def test_it_looks_above_a_relative_source_directory(self):
        # `build.py src/main.cpp` is how a project is built, and the path it
        # is handed is relative. The finder walks one level up because
        # `src/main.cpp` naming `web` means `../web` nearly every time - but
        # `Path("src").parent` is `Path(".")`, whose parent is itself, so the
        # walk stopped on its first step. Given an absolute path the same
        # finder always worked, which is why nothing caught it.
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            (project / "src").mkdir(parents=True)
            (project / "web").mkdir()
            (project / "web" / "index.html").write_text("<!doctype html>")
            (project / "src" / "main.cpp").write_text(
                'int main() { open(L"web"); return 0; }'
            )
            was = os.getcwd()
            try:
                os.chdir(project)
                found = interactive._what_the_c_opens(
                    [Path("src/main.cpp")], Path("src")
                )
            finally:
                os.chdir(was)
            self.assertEqual([path.name for path in found], ["web"])

    def test_an_absolute_source_directory_answers_the_same(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            (project / "src").mkdir(parents=True)
            (project / "web").mkdir()
            (project / "web" / "index.html").write_text("<!doctype html>")
            source = project / "src" / "main.cpp"
            source.write_text('int main() { open(L"web"); return 0; }')
            found = interactive._what_the_c_opens([source], project / "src")
            self.assertEqual([path.name for path in found], ["web"])


class WhatALibraryIsCalled(unittest.TestCase):
    """`--library WebView2Loader` is how a CMakeLists asks for one.

    Read as a filename it is not one, and the carrying step skipped it
    without a word: the symbols resolved, the build reported success, and the
    program could not start because the file it loads at run time was never
    put beside it.
    """

    def carried(self, target, spelled):
        asked = []

        def fetch_library(name, into, architecture, cache=None, say=None):
            asked.append(name)
            made = into / name
            made.write_bytes(b"MZ")
            return made

        was = header_fetch.fetch_library
        header_fetch.fetch_library = fetch_library
        try:
            with tempfile.TemporaryDirectory() as directory:
                room = Path(directory)
                program = room / "main.cpp"
                program.write_text("int main() { return 0; }")
                output = room / "dist" / "main.exe"
                output.parent.mkdir()
                output.write_bytes(b"MZ")
                got = interactive._carry_libraries(
                    program, output, target, (spelled,), True
                )
                return asked, [path.name for path in got]
        finally:
            header_fetch.fetch_library = was

    def test_a_bare_name_is_given_the_windows_suffix(self):
        asked, carried = self.carried("windows-x86_64", "WebView2Loader")
        self.assertEqual(asked, ["WebView2Loader.dll"])
        self.assertEqual(carried, ["WebView2Loader.dll"])

    def test_a_name_written_with_its_suffix_is_left_alone(self):
        asked, carried = self.carried("windows-x86_64", "WebView2Loader.dll")
        self.assertEqual(asked, ["WebView2Loader.dll"])
        self.assertEqual(carried, ["WebView2Loader.dll"])

    def test_each_target_spells_a_shared_library_its_own_way(self):
        for target, named in (
            ("linux-x86_64", "libthing.so"),
            ("darwin-arm64", "libthing.dylib"),
        ):
            asked, carried = self.carried(target, "libthing")
            self.assertEqual(asked, [named], target)
            self.assertEqual(carried, [named], target)

    def test_nothing_is_fetched_without_being_asked(self):
        # A file somebody else wrote, going into what the user is about to
        # ship, is not something to do without `--auto-fetch`.
        with tempfile.TemporaryDirectory() as directory:
            room = Path(directory)
            program = room / "main.cpp"
            program.write_text("int main() { return 0; }")
            output = room / "dist" / "main.exe"
            output.parent.mkdir()
            output.write_bytes(b"MZ")
            self.assertEqual(
                interactive._carry_libraries(
                    program, output, "windows-x86_64", ("WebView2Loader",), False
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()
