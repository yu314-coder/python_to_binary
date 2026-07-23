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


if __name__ == "__main__":
    unittest.main()
