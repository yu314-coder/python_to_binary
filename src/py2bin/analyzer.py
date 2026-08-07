from __future__ import annotations
import tokenize

import ast
import importlib.metadata as metadata
import importlib.util
import re
import sys
from collections import deque
from pathlib import Path

def _read_source(path):
    """A source file's text, decoded the way Python decodes one.

    A file may open with a byte-order mark and may say what it is in a
    `# -*- coding: ... -*-` line; Python honours both, so reading everything
    as UTF-8 refused a Latin-1 file with a codec error naming a byte offset.
    """

    with tokenize.open(path) as stream:
        return stream.read()


from .hooks import hooks_for
from .model import ImportAnalysis


_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _imports_in(path: Path) -> tuple[set[str], list[tuple[int, str | None, tuple[str, ...]]]]:
    try:
        tree = ast.parse(_read_source(path), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set(), []
    found: set[str] = set()
    relative: list[tuple[int, str | None, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                found.add(node.module)
            elif node.level:
                relative.append(
                    (node.level, node.module, tuple(alias.name for alias in node.names))
                )
    return found, relative


def _local_candidate(root: Path, module: str) -> Path | None:
    parts = module.split(".")
    module_path = root.joinpath(*parts).with_suffix(".py")
    package_path = root.joinpath(*parts, "__init__.py")
    if module_path.is_file():
        return module_path
    if package_path.is_file():
        return package_path
    return None


def _distribution_closure(initial: set[str]) -> set[str]:
    def canonical(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    available = {canonical(dist.metadata["Name"]): dist for dist in metadata.distributions()}
    result: set[str] = set()
    pending = deque(initial)
    while pending:
        requested = pending.popleft()
        key = canonical(requested)
        if key in result:
            continue
        dist = available.get(key)
        if dist is None:
            continue
        distribution_name = dist.metadata["Name"]
        result.add(canonical(distribution_name))
        for requirement in dist.requires or ():
            # Optional extras are not runtime dependencies merely because their
            # distributions happen to be installed in the build environment.
            if "extra ==" in requirement or "extra==" in requirement:
                continue
            match = _REQUIREMENT_NAME.match(requirement)
            if match:
                pending.append(match.group(1))
    return {available[name].metadata["Name"] for name in result if name in available}


def analyze(
    entry: Path,
    source_root: Path,
    includes: tuple[str, ...] = (),
    excludes: tuple[str, ...] = (),
    dependency_mode: str = "closure",
) -> ImportAnalysis:
    analysis = ImportAnalysis(local_files={entry})
    pending = deque([entry])
    visited: set[Path] = set()
    excluded_roots = {name.partition(".")[0] for name in excludes}

    while pending:
        path = pending.popleft()
        if path in visited:
            continue
        visited.add(path)
        modules, relative_imports = _imports_in(path)
        for module in modules:
            root_name = module.partition(".")[0]
            if root_name in excluded_roots:
                continue
            analysis.modules.add(module)
            candidate = _local_candidate(source_root, module)
            if candidate:
                if candidate not in analysis.local_files:
                    analysis.local_files.add(candidate)
                    pending.append(candidate)
        for level, module, names in relative_imports:
            base = path.parent
            for _ in range(level - 1):
                base = base.parent
            relative_names = [module] if module else list(names)
            for relative_name in relative_names:
                if not relative_name:
                    continue
                relative_path = base.joinpath(*relative_name.split("."))
                candidates = (relative_path.with_suffix(".py"), relative_path / "__init__.py")
                for candidate in candidates:
                    if candidate.is_file() and candidate not in analysis.local_files:
                        analysis.local_files.add(candidate)
                        pending.append(candidate)
                        break

    analysis.modules.update(includes)
    package_map = metadata.packages_distributions()
    stdlib = getattr(sys, "stdlib_module_names", set())
    distributions: set[str] = set()
    for module in sorted(analysis.modules):
        root_name = module.partition(".")[0]
        if root_name in excluded_roots or root_name in stdlib or _local_candidate(source_root, module):
            continue
        matches = package_map.get(root_name, [])
        if matches:
            distributions.update(matches)
        elif importlib.util.find_spec(root_name) is None:
            analysis.unresolved.add(root_name)

    for hook in hooks_for(analysis.modules):
        distributions.update(hook.distributions)
        analysis.hook_notes.append(hook.note)
    if dependency_mode == "none":
        distributions.clear()
    elif dependency_mode == "closure":
        distributions = _distribution_closure(distributions)
    analysis.distributions = distributions
    return analysis
