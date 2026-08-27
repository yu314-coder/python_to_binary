import ast
import os
from pathlib import Path
import sys
import unittest


class StandardLibraryOnlyTests(unittest.TestCase):
    def test_project_declares_no_build_or_runtime_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("requires = []", pyproject)
        self.assertNotIn("\ndependencies =", pyproject)

    def test_library_imports_only_itself_and_python_standard_library(self):
        root = Path(__file__).resolve().parents[1]
        unexpected: list[str] = []
        for source in sorted((root / "src" / "py2bin").rglob("*.py")):
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name.partition(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.partition(".")[0]]
                else:
                    continue
                for name in names:
                    if name != "py2bin" and name not in sys.stdlib_module_names:
                        unexpected.append(f"{source.relative_to(root)} imports {name}")
        self.assertEqual(unexpected, [])


class NoExternalToolchainTests(unittest.TestCase):
    """py2bin must never shell out -- least of all to a compiler.

    The whole point of the project is that Python -> C -> machine code happens
    inside py2bin. If any module could start a process, a future change could
    quietly delegate code generation to ``cc`` and the guarantee would be gone
    without a single test failing. So the invariant is checked structurally:
    the library may not even *import* a process-spawning module.
    """

    #: Standard-library modules that can start a process.
    _PROCESS_MODULES = frozenset(
        {"subprocess", "multiprocessing", "pty", "distutils", "setuptools"}
    )
    #: Callables that start a process or look for an external program.
    _PROCESS_CALLS = (
        "os.system",
        "os.popen",
        "os.spawnl",
        "os.spawnv",
        "os.spawnvp",
        "os.execv",
        "os.execvp",
        "os.execve",
        "os.posix_spawn",
        "shutil.which",
    )

    @staticmethod
    def _sources() -> list[Path]:
        root = Path(__file__).resolve().parents[1]
        return sorted((root / "src" / "py2bin").rglob("*.py"))

    def test_no_module_imports_a_process_spawning_module(self):
        root = Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for source in self._sources():
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name.partition(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.partition(".")[0]]
                else:
                    continue
                for name in names:
                    if name in self._PROCESS_MODULES:
                        offenders.append(
                            f"{source.relative_to(root)}:{node.lineno} imports {name}"
                        )
        self.assertEqual(offenders, [], "py2bin must not be able to start a process")

    def test_no_module_calls_a_process_spawning_function(self):
        root = Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for source in self._sources():
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                dotted = self._dotted_name(node.func)
                if dotted in self._PROCESS_CALLS:
                    offenders.append(
                        f"{source.relative_to(root)}:{node.lineno} calls {dotted}"
                    )
        self.assertEqual(offenders, [], "py2bin must not invoke an external program")

    @staticmethod
    def _dotted_name(node: ast.expr) -> str:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return ""
        parts.append(node.id)
        return ".".join(reversed(parts))

    def test_no_module_names_an_external_compiler_or_linker_as_a_program(self):
        """Guard against a string that would be handed to a process launcher.

        Prose may of course *mention* gcc; an executable name used as a value
        may not appear at all.
        """

        root = Path(__file__).resolve().parents[1]
        forbidden = {"gcc", "clang", "cl.exe", "link.exe", "xcrun", "ld64", "as"}
        offenders: list[str] = []
        for source in self._sources():
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value.strip() in forbidden:
                        offenders.append(
                            f"{source.relative_to(root)}:{node.lineno}: {node.value!r}"
                        )
        self.assertEqual(offenders, [], "an external toolchain name is used as a value")

    def test_compiling_needs_nothing_but_python(self):
        """Python in, machine code out, with no FFI and no child process.

        The point of the whole project is that a build asks for an interpreter
        and nothing else. `ctypes` is the interesting one to keep out: it is
        stdlib, so it would not fail the import checks above, but it pulls in
        `ctypes.util` and through it `subprocess`, and there are Pythons - the
        one on a phone, for instance - where a subprocess is not something a
        program may have. So the check is not "does it import" but "what did
        importing it drag in", which only a fresh interpreter can answer.
        """

        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            probe = (
                "import sys, pathlib\n"
                "from py2bin.capi_emit import python_to_capi_c\n"
                "from py2bin.c_native import compile_c_native\n"
                f"root = pathlib.Path({directory!r})\n"
                "source = 'def f():\\n    n = 0\\n"
                "    for i in range(5):\\n        n = n + i\\n"
                "    return n\\nprint(f())\\n'\n"
                "(root / 'p.c').write_text("
                "python_to_capi_c(source, str(root / 'p.py')))\n"
                "compile_c_native(root / 'p.c', root / 'p.bin', "
                "target='darwin-arm64', clean=True)\n"
                "print(sorted(name for name in sys.modules "
                "if name.split('.')[0] in {'ctypes', '_ctypes', 'subprocess'}))\n"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(
                Path(__file__).resolve().parents[1] / "src"
            )
            finished = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                env=environment,
            )
        self.assertEqual(finished.returncode, 0, finished.stderr)
        self.assertEqual(finished.stdout.strip(), "[]", finished.stdout)

    def test_compiling_cpp_needs_nothing_but_python_either(self):
        """The same question, asked of the C++ path and the headers it uses.

        C++ is translated to C by py2bin and compiled by py2bin, and the
        standard headers it includes are C source py2bin compiles like any
        other - so nothing here should reach for a toolchain either. Asked
        separately from the check above because it is a different path
        through the compiler, and "no toolchain" is a claim about all of it.
        """

        import subprocess
        import sys
        import tempfile

        program = (
            "#include <iostream>\n"
            "#include <vector>\n"
            "#include <algorithm>\n"
            "#include <string>\n"
            "#include <stdexcept>\n"
            "#include <string.h>\n"
            "template<typename T> T twice(T v) { return v + v; }\n"
            "class Base { public: virtual int tag() { return 1; } "
            "virtual ~Base() { } };\n"
            "class Sub : public Base { public: int tag() { return 2; } };\n"
            "int risky(int n) { if (n < 0) throw std::runtime_error(\"no\"); "
            "return n; }\n"
            "int main() {\n"
            "  std::vector<int> v; v.push_back(3); v.push_back(1);\n"
            "  std::sort(v.begin(), v.end());\n"
            "  Base *b = new Sub;\n"
            "  const wchar_t *w = L\"wide\";\n"
            "  char buf[8]; strcpy(buf, \"hi\");\n"
            "  try { risky(-1); } catch (std::exception &e) "
            "{ std::cout << e.what() << std::endl; }\n"
            "  std::cout << v[0] << b->tag() << twice(2) << buf << (int)w[0]"
            " << std::endl;\n"
            "  delete b; return 0;\n"
            "}\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            probe = (
                "import sys, pathlib\n"
                "from py2bin.c_native import compile_c_native\n"
                f"root = pathlib.Path({directory!r})\n"
                f"(root / 'p.cpp').write_text({program!r})\n"
                "compile_c_native(root / 'p.cpp', root / 'p.bin', "
                "target='darwin-arm64', clean=True)\n"
                "print(sorted(name for name in sys.modules "
                "if name.split('.')[0] in {'ctypes', '_ctypes', 'subprocess'}))\n"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(
                Path(__file__).resolve().parents[1] / "src"
            )
            finished = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                env=environment,
            )
        self.assertEqual(finished.returncode, 0, finished.stderr)
        self.assertEqual(finished.stdout.strip(), "[]", finished.stdout)


if __name__ == "__main__":
    unittest.main()


class AllocatorIsPointerWide(unittest.TestCase):
    """py2bin's own `<stdlib.h>` holds addresses in a pointer-wide type.

    Windows is LLP64: a `long` is four bytes there and a pointer is eight.
    An arena address held in an `unsigned long` loses its top half, and the
    kernel maps that arena wherever it likes - above four gigabytes, on a
    64-bit process, more often than not. Every pointer the allocator handed
    out was then a low address belonging to nobody.
    """

    def test_the_heap_holds_no_address_in_a_long(self) -> None:
        from py2bin.c_preprocessor import _BUILTIN_HEADERS

        text = _BUILTIN_HEADERS["stdlib.h"]
        for line in text.split("\n"):
            if "__py2bin_heap" not in line and "__py2bin_arena" not in line:
                continue
            self.assertNotRegex(
                line,
                r"\bunsigned\s+long\b(?!\s+long)",
                f"an address in a 32-bit type on Windows: {line.strip()}",
            )

    def test_the_allocator_takes_and_answers_size_t(self) -> None:
        from py2bin.c_preprocessor import _BUILTIN_HEADERS

        text = _BUILTIN_HEADERS["stdlib.h"]
        for spelled in (
            "void *malloc(size_t __n)",
            "void *calloc(size_t __count, size_t __size)",
            "void *realloc(void *__block, size_t __size)",
        ):
            self.assertIn(spelled, text)
