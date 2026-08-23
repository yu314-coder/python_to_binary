"""Several .c files compiled together, because there is no linker.

py2bin writes the binary itself and never links, so every function a program
calls must have its body in the translation unit being compiled. A project
split across `main.c` and `util.c` was therefore refused however correct it
was. Joining the files is how a single translation unit has always been got
out of several - a unity build - and it needs nothing that was not already
here.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from py2bin.c_frontend import CCompileError
from py2bin.c_native import _where_it_was_written, compile_c_native, unity_source


_MAIN = '#include <stdio.h>\n#include "mathy.h"\nint main(void) {\n    printf("%d\\n", square(7));\n    return 0;\n}\n'
_MATHY = '#include "mathy.h"\nint square(int n) { return n * n; }\n'
_HEADER = "#ifndef MATHY_H\n#define MATHY_H\nint square(int n);\n#endif\n"


class UnitySource(unittest.TestCase):
    def _project(self, root: Path) -> "tuple[Path, Path]":
        (root / "mathy.h").write_text(_HEADER, encoding="utf-8")
        main = root / "main.c"
        main.write_text(_MAIN, encoding="utf-8")
        mathy = root / "mathy.c"
        mathy.write_text(_MATHY, encoding="utf-8")
        return main, mathy

    def test_every_file_is_present_and_findable_afterwards(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            main, mathy = self._project(Path(scratch))
            joined, spans = unity_source((main, mathy))
            self.assertIn("int main(void)", joined)
            self.assertIn("int square(int n) { return n * n; }", joined)
            # The first file starts at line 1 and the second after it.
            self.assertEqual(spans[0], (1, main))
            self.assertEqual(spans[1][1], mathy)
            self.assertEqual(spans[1][0], _MAIN.count("\n") + 1)

    def test_a_line_maps_back_to_the_file_it_was_written_in(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            main, mathy = self._project(Path(scratch))
            _joined, spans = unity_source((main, mathy))
            self.assertEqual(_where_it_was_written(spans, 1), (main, 1))
            self.assertEqual(_where_it_was_written(spans, 3), (main, 3))
            start = spans[1][0]
            self.assertEqual(_where_it_was_written(spans, start), (mathy, 1))
            self.assertEqual(_where_it_was_written(spans, start + 1), (mathy, 2))

    def test_a_file_without_a_trailing_newline_does_not_glue(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            first = root / "a.c"
            first.write_text("int a(void) { return 1; }", encoding="utf-8")
            second = root / "b.c"
            second.write_text("int b(void) { return 2; }\n", encoding="utf-8")
            joined, spans = unity_source((first, second))
            self.assertIn("}\nint b", joined)
            self.assertEqual(_where_it_was_written(spans, 2), (second, 1))


class BuildingAProject(unittest.TestCase):
    def test_two_files_and_a_header_compile_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "mathy.h").write_text(_HEADER, encoding="utf-8")
            main = root / "main.c"
            main.write_text(_MAIN, encoding="utf-8")
            mathy = root / "mathy.c"
            mathy.write_text(_MATHY, encoding="utf-8")
            result = compile_c_native(
                main,
                root / "app",
                extra_sources=(mathy,),
                include_dirs=(str(root),),
            )
            self.assertTrue(result.artifact.is_file())

    def test_the_diagnostic_names_the_file_the_mistake_is_in(self) -> None:
        """The reason the mapping exists at all.

        Without it an error in the second file reports a line number that
        exists in no file the user wrote, which is worse than no location.
        """

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "mathy.h").write_text(_HEADER, encoding="utf-8")
            main = root / "main.c"
            main.write_text(_MAIN, encoding="utf-8")
            mathy = root / "mathy.c"
            mathy.write_text(
                '#include "mathy.h"\nint square(int n) { return nope; }\n',
                encoding="utf-8",
            )
            with self.assertRaises(CCompileError) as caught:
                compile_c_native(
                    main,
                    root / "app",
                    extra_sources=(mathy,),
                    include_dirs=(str(root),),
                )
            self.assertEqual(caught.exception.filename, str(mathy))
            self.assertEqual(caught.exception.line, 2)

    def test_a_missing_source_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            main = root / "main.c"
            main.write_text("int main(void) { return 0; }\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                compile_c_native(
                    main, root / "app", extra_sources=(root / "absent.c",)
                )


if __name__ == "__main__":
    unittest.main()
