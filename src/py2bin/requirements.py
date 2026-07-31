"""Work out what a program needs from outside itself.

A build should not have to be told what to download. The program says what it
imports; the standard library is known; the files beside it are found already.
What is left is what has to come from an index, and that is what this reports.

The one thing it will not do is guess. An import name and a project name are
different things - `PIL` is published as `pillow`, `cv2` as `opencv-python` -
and a guess that happens to hit an unrelated project on PyPI would download a
stranger's code and put it in someone's application. So names are translated
through a table that has been checked, and anything not in it is reported as
unknown for a person to name, rather than approximated.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

#: Import names whose project is not spelled the same. Every entry here has
#: been checked against the project's own page rather than inferred from the
#: name, because the failure mode of a wrong guess is downloading a stranger's
#: package.
KNOWN_PROJECTS = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "serial": "pyserial",
    "OpenGL": "pyopengl",
    "webview": "pywebview",
    "winpty": "pywinpty",
    "Cryptodome": "pycryptodomex",
    "Crypto": "pycryptodome",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "google": "protobuf",
    "attr": "attrs",
    "zmq": "pyzmq",
    "usb": "pyusb",
    "gi": "pygobject",
    "cairo": "pycairo",
    "wx": "wxpython",
    "fitz": "pymupdf",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "magic": "python-magic",
    "jwt": "pyjwt",
    "lxml": "lxml",
    "psutil": "psutil",
    "numpy": "numpy",
    "requests": "requests",
    "imageio": "imageio",
    "pygame": "pygame",
    "manim": "manim",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "moviepy": "moviepy",
    "pydub": "pydub",
    "rich": "rich",
    "click": "click",
    "tqdm": "tqdm",
    "pyright": "pyright",
}


class Discovered:
    """What a program turned out to need."""

    __slots__ = ("projects", "unknown", "local", "standard")

    def __init__(self, projects, unknown, local, standard):
        #: PyPI project names, ready to fetch.
        self.projects = projects
        #: Imported names with no checked project name. Not guessed at.
        self.unknown = unknown
        #: Modules found beside the program, which need no download.
        self.local = local
        #: Standard library names, which the interpreter already carries.
        self.standard = standard

    def __repr__(self) -> str:  # pragma: no cover - for debugging only
        return (
            f"Discovered(projects={self.projects!r}, unknown={self.unknown!r}, "
            f"local={self.local!r})"
        )


def _imported_names(tree: ast.AST) -> set[str]:
    """Every top-level module name this file imports."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # A relative import names something beside the file, not a project.
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def _reachable(entry: Path) -> tuple[set[str], set[str]]:
    """Walk the program and the files it imports from its own directory."""
    root = entry.parent
    seen: set[Path] = set()
    local: set[str] = set()
    imported: set[str] = set()
    pending = [entry]
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for name in _imported_names(tree):
            beside = root / f"{name}.py"
            package = root / name / "__init__.py"
            if beside.is_file():
                local.add(name)
                pending.append(beside)
            elif package.is_file():
                local.add(name)
                pending.append(package)
            else:
                imported.add(name)
    return imported, local


def discover(entry: Path) -> Discovered:
    """What ``entry`` needs that is neither its own nor the interpreter's."""
    entry = entry.expanduser().resolve()
    imported, local = _reachable(entry)
    standard = {name for name in imported if name in sys.stdlib_module_names}
    outside = imported - standard - local
    projects = sorted(
        {KNOWN_PROJECTS[name] for name in outside if name in KNOWN_PROJECTS}
    )
    unknown = sorted(name for name in outside if name not in KNOWN_PROJECTS)
    return Discovered(projects, unknown, sorted(local), sorted(standard))
