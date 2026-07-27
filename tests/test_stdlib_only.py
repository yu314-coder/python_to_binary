import ast
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


if __name__ == "__main__":
    unittest.main()
