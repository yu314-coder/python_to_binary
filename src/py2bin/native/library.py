"""Static whole-library validation for the direct native backend.

The audit never imports or executes inspected code. It asks the real frontend
to lower each top-level function through a synthetic call, so its answer
matches compilation rather than relying on an optimistic syntax checklist.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .frontend import Frontend, NativeCompileError, lower
from .kernels import StaticI64Tensor


_NATIVE_SUFFIXES = frozenset({".a", ".dll", ".dylib", ".lib", ".pyd", ".so"})
_WEB_SUFFIXES = frozenset({".css", ".htm", ".html", ".js", ".mjs", ".cjs", ".wasm"})
_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        "__pycache__",
        "build",
        "dist",
    }
)


@dataclass(frozen=True, slots=True)
class NativeFunctionAudit:
    module: str
    name: str
    path: Path
    native: bool
    reason: str


@dataclass(frozen=True, slots=True)
class NativeLibraryAudit:
    root: Path
    python_files: int
    native_payloads: tuple[Path, ...]
    web_assets: tuple[Path, ...]
    functions: tuple[NativeFunctionAudit, ...]
    module_blockers: tuple[str, ...]

    @property
    def native_functions(self) -> int:
        return sum(item.native for item in self.functions)

    @property
    def blockers(self) -> tuple[str, ...]:
        return (
            *self.module_blockers,
            *(item.reason for item in self.functions if not item.native),
            *(
                f"{path}: prebuilt native payload is machine code, but its "
                "CPython/C ABI adapter has not been proven replaceable"
                for path in self.native_payloads
            ),
        )

    @property
    def fully_native(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "fully_native": self.fully_native,
            "python_files": self.python_files,
            "native_payloads": [str(path) for path in self.native_payloads],
            "web_assets": [str(path) for path in self.web_assets],
            "functions": [
                {
                    "module": item.module,
                    "name": item.name,
                    "path": str(item.path),
                    "native": item.native,
                    "reason": item.reason,
                }
                for item in self.functions
            ],
            "module_blockers": list(self.module_blockers),
            "native_payload_blockers": [
                blocker
                for blocker in self.blockers
                if "CPython/C ABI adapter" in blocker
            ],
            "native_functions": self.native_functions,
            "blocked_functions": len(self.functions) - self.native_functions,
        }


def _included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return not any(part in _IGNORED_PARTS or part.startswith(".") for part in relative.parts)


def _module_name(path: Path, root: Path) -> tuple[str, Path]:
    relative = path.relative_to(root)
    if relative.name == "__init__.py":
        parts = relative.parent.parts
        if parts:
            return ".".join(parts), root
        return root.name, root.parent
    return ".".join((*relative.parts[:-1], relative.stem)), root


def _module_level_blockers(
    path: Path,
    tree: ast.Module,
    source_roots: tuple[Path, ...],
    experimental_kernels: bool,
) -> tuple[str, ...]:
    blockers: list[str] = []
    validator = Frontend(
        path,
        source_roots,
        experimental_kernels=experimental_kernels,
    )
    for statement in tree.body:
        try:
            if isinstance(statement, ast.FunctionDef):
                continue
            if isinstance(statement, (ast.AsyncFunctionDef, ast.ClassDef)):
                name = getattr(statement, "name", type(statement).__name__)
                blockers.append(
                    f"{path}:{getattr(statement, 'lineno', 1)}: "
                    f"{type(statement).__name__} {name!r} is not in the native subset"
                )
                continue
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                if validator.is_kernel_expression(statement.value):
                    value = validator.kernel_value(statement.value)
                    if not isinstance(value, StaticI64Tensor):
                        raise NativeCompileError(
                            path,
                            statement,
                            "module-level numerical kernel must produce a static "
                            "tensor, not runtime scalar work",
                        )
                    validator.values[statement.targets[0].id] = value
                else:
                    validator.values[statement.targets[0].id] = validator.constant(
                        statement.value
                    )
                continue
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
            ):
                if validator.is_kernel_expression(statement.value):
                    value = validator.kernel_value(statement.value)
                    if not isinstance(value, StaticI64Tensor):
                        raise NativeCompileError(
                            path,
                            statement,
                            "module-level numerical kernel must produce a static "
                            "tensor, not runtime scalar work",
                        )
                    validator.values[statement.target.id] = value
                else:
                    validator.values[statement.target.id] = validator.constant(
                        statement.value
                    )
                continue
            if isinstance(statement, ast.ImportFrom):
                if statement.level == 0 and statement.module == "__future__":
                    continue
                validator.import_from(statement)
                continue
            if isinstance(statement, ast.Import):
                validator.import_statement(statement)
                continue
            if isinstance(statement, ast.Pass):
                continue
            blockers.append(
                f"{path}:{getattr(statement, 'lineno', 1)}: "
                f"module-level {type(statement).__name__} requires runtime semantics"
            )
        except NativeCompileError as error:
            blockers.append(str(error))
    return tuple(blockers)


def audit_native_library(
    root: Path,
    *,
    source_roots: tuple[Path, ...] = (),
    experimental_kernels: bool = False,
) -> NativeLibraryAudit:
    """Validate every top-level function below ``root`` using the real frontend."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"native library root does not exist: {root}")
    extra_roots = tuple(path.expanduser().resolve() for path in source_roots)
    files = tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file() and _included(path, root)),
            key=lambda path: path.as_posix(),
        )
    )
    python_paths = tuple(path for path in files if path.suffix == ".py")
    native_payloads = tuple(path for path in files if path.suffix.lower() in _NATIVE_SUFFIXES)
    web_assets = tuple(path for path in files if path.suffix.lower() in _WEB_SUFFIXES)
    functions: list[NativeFunctionAudit] = []
    module_blockers: list[str] = []

    for path in python_paths:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            module_blockers.append(f"{path}: cannot parse native source: {error}")
            continue
        module, import_root = _module_name(path, root)
        roots = tuple(dict.fromkeys((import_root, root, *extra_roots)))
        function_nodes = tuple(
            node for node in tree.body if isinstance(node, ast.FunctionDef)
        )
        module_blockers.extend(
            _module_level_blockers(
                path,
                tree,
                roots,
                experimental_kernels,
            )
        )
        for function in function_nodes:
            positional = (*function.args.posonlyargs, *function.args.args)
            arguments = ", ".join("0" for _ in positional)
            probe = (
                f"from {module} import {function.name} as __py2bin_probe\n"
                f"raise SystemExit(__py2bin_probe({arguments}))\n"
            )
            try:
                lower(
                    root / "__py2bin_library_audit__.py",
                    probe,
                    roots,
                    experimental_kernels=experimental_kernels,
                )
            except (NativeCompileError, ValueError) as error:
                functions.append(
                    NativeFunctionAudit(
                        module,
                        function.name,
                        path,
                        False,
                        f"{module}.{function.name}: {error}",
                    )
                )
            else:
                functions.append(
                    NativeFunctionAudit(
                        module,
                        function.name,
                        path,
                        True,
                        "lowered to native IR without CPython",
                    )
                )

    return NativeLibraryAudit(
        root,
        len(python_paths),
        native_payloads,
        web_assets,
        tuple(functions),
        tuple(module_blockers),
    )


def require_native_library(
    root: Path,
    *,
    source_roots: tuple[Path, ...] = (),
    experimental_kernels: bool = False,
) -> NativeLibraryAudit:
    """Return a successful audit or raise with the first exact blocker."""

    audit = audit_native_library(
        root,
        source_roots=source_roots,
        experimental_kernels=experimental_kernels,
    )
    if audit.blockers:
        raise NativeCompileError(
            root / "__py2bin_library_audit__.py",
            ast.Constant(value=None),
            f"strict native library audit found {len(audit.blockers)} blocker(s); "
            f"first: {audit.blockers[0]}",
        )
    return audit
