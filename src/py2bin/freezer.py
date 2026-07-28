from __future__ import annotations

import json
import hashlib
import os
import plistlib
import re
import shutil
import stat
import sys
import sysconfig
import tempfile
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath

from .analyzer import analyze
from .builder import _copy_distribution, _copy_project
from .icons import install_macos_icon, macos_info_plist
from .native.compiler import host_target
from .native.launcher import macos_shell_launcher
from .onefile import create_onefile
from .runtime_packs import (
    RuntimePackInfo,
    install_runtime_pack,
    inspect_runtime_pack,
    write_runtime_manifest,
)
from .windows_icon import install_windows_icon, install_windows_identity

# CPython version fetched for a cross-target bundle when none is requested.
# It is pinned rather than "latest" so an unattended build is reproducible.
_DEFAULT_FETCH_PYTHON = "3.12.9"


@dataclass(frozen=True, slots=True)
class FreezeResult:
    bundle: Path
    launcher: Path
    files: int
    bytes: int
    distributions: tuple[str, ...]
    target: str
    python: str
    onefile: bool


@dataclass(frozen=True, slots=True)
class RuntimePackResult:
    pack: Path
    files: int
    bytes: int
    target: str
    python: str


@dataclass(frozen=True, slots=True)
class WheelInfo:
    path: Path
    name: str
    top_levels: tuple[str, ...]
    requirements: tuple[str, ...]
    python_tag: str
    abi_tag: str
    platform_tag: str


_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_WEB_ASSET_KINDS = {
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".wasm": "webassembly",
}


#: Parts of a CPython installation that exist to *build* things, to test
#: itself, or to document itself. A compiled application runs none of them, and
#: together they are the difference between a bundle that is four times the
#: size of what other compilers produce and one that is comparable.
_UNUSED_AT_RUNTIME = (
    "site-packages",     # the application's own, added separately
    "config-*",          # libpython.a and friends: for building extensions
    "ensurepip",         # installing pip into a venv
    "pydoc_data",        # topic text for the help() browser
    "idlelib",           # the bundled editor
    "turtledemo",
    "test",
    "tests",
    "lib2to3",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.a",               # static libraries
    "*.exe",
)


def bundle_site_packages(bundle: Path, sources: tuple[Path, ...]) -> None:
    """Copy the application's packages into the bundle, without their baggage.

    A site-packages directory carries the test suites, the build metadata and
    pip itself. None of it runs; together, for one small application, it was
    29 MB of 66.
    """

    import shutil

    destination = bundle / "Contents" / "Resources" / "site-packages"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        for item in source.iterdir():
            if any(
                item.match(pattern)
                for pattern in (*_UNUSED_AT_RUNTIME, "pip", "*Test", "*.dist-info")
            ):
                continue
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(
                    item,
                    target,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(*_UNUSED_AT_RUNTIME),
                    symlinks=True,
                )
            else:
                shutil.copy2(item, target)


#: Packages the import machinery reaches by *name* rather than by an import
#: statement, so no amount of reading the source finds them. `encodings` is the
#: clearest case: a codec is looked up by its name at run time, and a bundle
#: without it cannot even open a text file.
_REACHED_BY_NAME = (
    "encodings",
    "codecs",
    "importlib",
    "site",
    "sitecustomize",
    "abc",
    "io",
    "os",
    "posixpath",
    "genericpath",
    "stat",
    "_collections_abc",
    "warnings",
    "types",
    "enum",
    "re",
    "sre_compile",
    "sre_parse",
    "sre_constants",
    "functools",
    "operator",
    "collections",
    "keyword",
    "heapq",
    "itertools",
    "reprlib",
    "copyreg",
    "traceback",
    "linecache",
    "tokenize",
    "token",
    "zipimport",
    "runpy",
    "threading",
    "weakref",
    "contextlib",
)


#: Extensions the interpreter itself reaches during start-up or through
#: machinery no import statement mentions. Keeping these is cheap; finding out
#: the hard way that one was needed is not.
_INTERPRETER_NEEDS = frozenset(
    {
        "codecs", "io", "abc", "collections", "functools", "locale",
        "opcode", "operator", "signal", "sre", "stat", "struct", "thread",
        "weakref", "posixsubprocess", "socket", "select", "math", "errno",
        "time", "datetime", "random", "hashlib", "ssl", "zlib", "bz2",
        "lzma", "asyncio", "queue", "heapq", "bisect", "json", "pickle",
        "csv", "decimal", "uuid", "contextvars", "typing", "zoneinfo",
        "statistics", "multiprocessing", "scproxy", "unicodedata",
        "elementtree", "pyexpat", "binascii", "fcntl", "termios", "grp",
        "pwd", "sha2", "sha3", "blake2", "md5", "sha1", "hmac", "posixshmem",
    }
)


def _module_imports(source: str) -> set[str]:
    """Every module name this source imports, dotted names included."""

    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


def prune_unreachable(bundle: Path, entry: Path) -> int:
    """Drop bundled modules the program cannot import.

    Only a fraction of a Python installation is ever reached: of nearly twelve
    thousand standard-library files, one application touched under two hundred.
    Shipping the rest is the difference between a bundle larger than what other
    compilers produce and one smaller.

    The walk is *static*, so it cannot see an import built from a string. What
    the import machinery reaches by name is kept unconditionally - see
    `_REACHED_BY_NAME` - and a whole package is kept as soon as anything in it
    is reached, because a package's modules routinely import their siblings in
    ways that only run-time knows about.
    """

    import shutil

    roots = [
        bundle / "Contents" / "Resources" / "site-packages",
        *(bundle / "Contents" / "lib").glob("python*"),
    ]

    def locate(name: str) -> Path | None:
        for root in roots:
            for candidate in (
                root / (name.replace(".", "/") + ".py"),
                root / name.replace(".", "/") / "__init__.py",
            ):
                if candidate.is_file():
                    return candidate
        return None

    seen: set[str] = set()
    queue = set(_REACHED_BY_NAME)
    for source in (entry, *(p for p in entry.parent.glob("*.py"))):
        queue |= _module_imports(source.read_text(encoding="utf-8", errors="replace"))
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        # A dotted name needs each package above it.
        parts = name.split(".")
        queue |= {".".join(parts[:i]) for i in range(1, len(parts))} - seen
        found = locate(name)
        if found is None:
            continue
        queue |= _module_imports(
            found.read_text(encoding="utf-8", errors="replace")
        ) - seen

    # A package is kept whole, so everything *its* modules need must be kept
    # too - and that is not visible from the package's __init__. `encodings` is
    # the case that proves it: the codec registry imports `encodings.idna` by
    # name, idna imports `stringprep`, and a bundle without stringprep fails
    # with "unknown encoding: idna" from inside socket.getfqdn.
    settled = False
    while not settled:
        settled = True
        for package in {name.split(".")[0] for name in seen}:
            found = locate(package)
            if found is None or found.name != "__init__.py":
                continue
            for module in found.parent.rglob("*.py"):
                for wanted in _module_imports(
                    module.read_text(encoding="utf-8", errors="replace")
                ):
                    if wanted not in seen:
                        seen.add(wanted)
                        parts = wanted.split(".")
                        seen |= {".".join(parts[:i]) for i in range(1, len(parts))}
                        settled = False

    keep_packages = {name.split(".")[0] for name in seen}
    freed = 0
    for root in roots:
        if not root.is_dir():
            continue
        for item in sorted(root.iterdir()):
            if item.name in {"lib-dynload", "__pycache__"} or item.name.startswith("_"):
                continue
            stem = item.name[:-3] if item.name.endswith(".py") else item.name
            if stem in keep_packages or not (item.is_dir() or item.name.endswith(".py")):
                continue
            freed += sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) \
                if item.is_dir() else item.stat().st_size
            shutil.rmtree(item, ignore_errors=True) if item.is_dir() else item.unlink()
        dynload = root / "lib-dynload"
        if dynload.is_dir():
            for module in dynload.glob("*.so"):
                name = module.name.split(".")[0]
                # A private extension is not automatically kept: most of
                # lib-dynload is private, and keeping all of it kept curses,
                # tkinter, the database bindings and everything else the
                # program never mentions. What keeps one is being named -
                # `socket.py` says `import _socket`, and the walk read that.
                if name in seen or name.lstrip("_") in seen:
                    continue
                if name.lstrip("_") in _INTERPRETER_NEEDS:
                    continue
                freed += module.stat().st_size
                module.unlink()
    return freed


def compile_bundle_sources(bundle: Path) -> int:
    """Replace the bundled `.py` files with bytecode, in place.

    Python imports a lone `.pyc` sitting where the `.py` would be, so nothing
    needs to know this happened. It is smaller, and it means the first run does
    not compile the standard library into a bundle it usually cannot write to.

    The application's own modules are not here at all - they are machine code
    inside the executable. This is only what the program *imports*.
    """

    import compileall
    import py_compile

    saved = 0
    for directory in (bundle / "Contents" / "lib", bundle / "Contents" / "Resources"):
        if not directory.is_dir():
            continue
        for source in list(directory.rglob("*.py")):
            target = source.with_suffix(".pyc")
            try:
                py_compile.compile(
                    str(source),
                    cfile=str(target),
                    doraise=True,
                    optimize=2,
                    invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                )
            except (py_compile.PyCompileError, SyntaxError, ValueError, OSError):
                # A file this interpreter cannot compile - a Python 2 relic in
                # a package's test data, say - is left as source rather than
                # dropped, because something may still import it.
                continue
            saved += source.stat().st_size - target.stat().st_size
            source.unlink()
        for cache in list(directory.rglob("__pycache__")):
            shutil.rmtree(cache, ignore_errors=True)
    return saved


def embed_cpython_in_app(bundle: Path) -> int:
    """Put this CPython inside a compiled ``.app`` so the bundle can travel.

    A compiled artifact names its interpreter in an ``LC_LOAD_DYLIB``, and dyld
    resolves that before a line of the program runs - so an absolute path to
    the build machine's CPython is a refusal to launch anywhere else, with no
    message from the program at all. Compiled with
    ``@executable_path/../Frameworks/Python``, the bundle needs that file to be
    there, and the interpreter needs its standard library beside it.

    The standard library goes to ``Contents/lib/python3.X`` rather than inside
    the framework, because CPython finds its prefix by walking up from the
    executable looking for ``lib/pythonX.Y/os.py`` - and from
    ``Contents/MacOS`` the first place it looks up is ``Contents``.
    """

    import shutil
    import sysconfig

    from .cabi_tables import _cpython_library

    dylib = Path(_cpython_library())
    if not dylib.is_file():
        raise FileNotFoundError(f"no CPython shared library at {dylib}")
    frameworks = bundle / "Contents" / "Frameworks"
    frameworks.mkdir(parents=True, exist_ok=True)
    version_directory = dylib.parent
    # The framework *layout*, not the bare library: the signature on that
    # library seals its neighbouring Resources/Info.plist, so a dylib lifted
    # out on its own is reported as modified and dyld refuses it. The bytes are
    # identical either way - what is missing is the file the seal names.
    carried_version = (
        frameworks / "Python.framework" / "Versions" / version_directory.name
    )
    if carried_version.exists():
        shutil.rmtree(carried_version)
    carried_version.mkdir(parents=True)
    shutil.copy2(dylib, carried_version / dylib.name)
    # Only the file the signature seals. The framework's Resources directory
    # is another 76 MB - a second copy of the standard library and the Tcl/Tk
    # frameworks - and nothing loads it from here.
    plist = version_directory / "Resources" / "Info.plist"
    if plist.is_file():
        (carried_version / "Resources").mkdir(parents=True, exist_ok=True)
        shutil.copy2(plist, carried_version / "Resources" / "Info.plist")

    version = sysconfig.get_config_var("py_version_short")
    standard_library = Path(sysconfig.get_path("stdlib"))
    destination = bundle / "Contents" / "lib" / f"python{version}"
    if destination.exists():
        shutil.rmtree(destination)
    # site-packages is the application's business, not the interpreter's: it
    # is what --site names, and copying it here would bury a second copy.
    shutil.copytree(
        standard_library,
        destination,
        ignore=shutil.ignore_patterns(*_UNUSED_AT_RUNTIME),
    )
    # The shared libraries python.org's own extension modules link against -
    # OpenSSL for `ssl` and `hashlib`, and the compression libraries - are
    # named by *absolute* path inside those .so files. Carrying the
    # interpreter is not enough: the first `import ssl` on a machine without
    # that framework fails, and every one of those imports is in the standard
    # library rather than anything the application chose.
    carried_libraries = bundle / "Contents" / "lib"
    source_libraries = version_directory / "lib"
    prefix = f"{version_directory.parent.parent}/Versions/{version_directory.name}/lib/"
    if source_libraries.is_dir():
        for library in source_libraries.glob("*.dylib"):
            shutil.copy2(library, carried_libraries / library.name)
    wanted: set[str] = set()
    for module in bundle.rglob("*.so"):
        wanted |= _point_at_carried_libraries(module, prefix, carried_libraries)
    _drop_unreferenced_libraries(carried_libraries, prefix, wanted)
    # Thinned last, so the rewriting above worked on the file as shipped.
    for binary in (
        *bundle.rglob("*.so"),
        *bundle.rglob("*.dylib"),
        carried_version / dylib.name,
    ):
        if binary.is_file():
            _thin_to_arm64(binary)
    return sum(
        item.stat().st_size
        for item in (
            *carried_version.rglob("*"),
            *destination.rglob("*"),
            *carried_libraries.glob("*.dylib"),
        )
        if item.is_file()
    )


def drop_unused_libraries(bundle: Path) -> int:
    """Discard carried libraries nothing left in the bundle refers to.

    Worth running *after* the modules have been pruned: the closure is computed
    from the extensions present, and pruning removes extensions. `_curses.so`
    kept ncurses and panel in a bundle for an application with no terminal
    interface, because the library question was settled before the module
    question was.
    """

    libraries = bundle / "Contents" / "lib"
    if not libraries.is_dir():
        return 0
    before = sum(f.stat().st_size for f in libraries.glob("*.dylib"))
    wanted: set[str] = set()
    for module in bundle.rglob("*.so"):
        wanted |= _referenced_libraries(module, libraries)
    _drop_unreferenced_libraries(libraries, None, wanted)
    return before - sum(f.stat().st_size for f in libraries.glob("*.dylib"))


def _referenced_libraries(binary: Path, libraries: Path) -> set[str]:
    """Which carried libraries this file names, however it names them."""

    data = binary.read_bytes()
    return {
        library.name
        for library in libraries.glob("*.dylib")
        if library.name.encode() in data
    }


def _drop_unreferenced_libraries(
    libraries: Path, prefix: str | None, wanted: set[str]
) -> None:
    """Follow the closure, then delete what is left over."""

    pending = list(wanted)
    while pending:
        library = libraries / pending.pop()
        if not library.is_file():
            continue
        found = _referenced_libraries(library, libraries) - {library.name}
        pending.extend(found - wanted)
        wanted |= found
    for library in libraries.glob("*.dylib"):
        if library.name not in wanted:
            library.unlink()





#: The architecture a darwin-arm64 bundle runs. CPU_TYPE_ARM64 is
#: CPU_TYPE_ARM (12) with the 64-bit ABI bit set.
_CPU_TYPE_ARM64 = 0x0100000C


def _thin_to_arm64(binary: Path) -> bool:
    """Keep only the arm64 slice of a universal file. True if it changed.

    Every slice of a universal binary carries its own signature, so lifting one
    out yields a thin file that is still signed - which is why this is safe on
    the interpreter's own library, where an invalid signature is refused.

    Half of what a Mach-O universal binary weighs is an architecture this
    bundle will never execute.
    """

    import struct

    data = binary.read_bytes()
    if len(data) < 8 or data[:4] not in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
        return False
    wide = data[:4] == b"\xca\xfe\xba\xbf"
    count = struct.unpack_from(">I", data, 4)[0]
    entry = 32 if wide else 20
    for index in range(count):
        at = 8 + index * entry
        if wide:
            cpu, _sub, offset, size = struct.unpack_from(">IIQQ", data, at)
        else:
            cpu, _sub, offset, size = struct.unpack_from(">IIII", data, at)
        if cpu == _CPU_TYPE_ARM64:
            binary.write_bytes(data[offset : offset + size])
            return True
    return False


def _point_at_carried_libraries(
    module: Path, prefix: str, libraries: Path
) -> set[str]:
    """Rewrite absolute library references in one extension module.

    The path is patched *in place*, which is what makes this possible at all:
    the replacement is shorter than what it replaces, so the load command keeps
    its size and every offset in the file stays where it was. A longer one
    would mean rebuilding the header.

    The signature this invalidates is not enforced for these: a patched module
    loads. That is not true of the interpreter's own library, whose signature
    seals a neighbouring file - which is why *that* one is copied whole rather
    than edited.
    """

    data = bytearray(module.read_bytes())
    if prefix.encode() not in data:
        return set()
    depth = len(module.parent.relative_to(libraries.parent).parts)
    reach = "/".join([".."] * depth) or "."
    referenced: set[str] = set()
    for library in libraries.glob("*.dylib"):
        old = f"{prefix}{library.name}".encode()
        new = f"@loader_path/{reach}/lib/{library.name}".encode()
        if len(new) > len(old) or old not in data:
            continue
        while (at := data.find(old)) >= 0:
            data[at : at + len(old)] = new + b"\0" * (len(old) - len(new))
        referenced.add(library.name)
    if referenced:
        module.write_bytes(bytes(data))
    return referenced


def _required_suffix(path: Path, suffix: str) -> Path:
    """Append a required artifact suffix without discarding dotted names."""

    return path if path.suffix.lower() == suffix else Path(f"{path}{suffix}")


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _web_assets(
    bundle_root: Path,
    content_roots: tuple[Path, ...],
) -> list[dict[str, object]]:
    assets: list[dict[str, object]] = []
    files = (
        candidate
        for content_root in content_roots
        for candidate in content_root.rglob("*")
        if candidate.is_file()
    )
    for path in sorted(
        files,
        key=lambda candidate: candidate.relative_to(bundle_root).as_posix(),
    ):
        kind = _WEB_ASSET_KINDS.get(path.suffix.lower())
        if kind is None:
            continue
        assets.append(
            {
                "path": path.relative_to(bundle_root).as_posix(),
                "kind": kind,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return assets


def _windows_app_user_model_id(name: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", name)
    product = "".join(part[:1].upper() + part[1:] for part in parts) or "App"
    return f"PythonToBinary.{product}"[:128]


def inspect_wheel(wheel: Path) -> WheelInfo:
    wheel = wheel.expanduser().resolve()
    if not wheel.is_file() or not zipfile.is_zipfile(wheel):
        raise ValueError(f"wheel is not a valid ZIP archive: {wheel}")
    filename_parts = wheel.name[:-4].rsplit("-", 3) if wheel.name.endswith(".whl") else []
    if len(filename_parts) != 4:
        raise ValueError(f"wheel filename does not contain Python/ABI/platform tags: {wheel.name}")
    python_tag, abi_tag, platform_tag = filename_parts[-3:]
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name
            for name in archive.namelist()
            if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError(f"wheel must contain exactly one .dist-info/METADATA: {wheel}")
        metadata = BytesParser(policy=email_policy).parsebytes(
            archive.read(metadata_names[0])
        )
        name = metadata.get("Name")
        if not name:
            raise ValueError(f"wheel metadata has no Name field: {wheel}")
        top_level_name = metadata_names[0].rsplit("/", 1)[0] + "/top_level.txt"
        if top_level_name in archive.namelist():
            top_levels = {
                item.strip().partition(".")[0]
                for item in archive.read(top_level_name).decode("utf-8").splitlines()
                if item.strip()
            }
        else:
            top_levels: set[str] = set()
            for member in archive.namelist():
                relative = _safe_wheel_member(member)
                if relative is None or not relative.parts:
                    continue
                first = relative.parts[0]
                if first.endswith((".dist-info", ".data")):
                    continue
                root = Path(first).stem if first.endswith(".py") else first
                if root.isidentifier():
                    top_levels.add(root)
        requirements = tuple(metadata.get_all("Requires-Dist", []))
    return WheelInfo(
        wheel,
        str(name),
        tuple(sorted(top_levels)),
        requirements,
        python_tag,
        abi_tag,
        platform_tag,
    )


def _wheel_matches_target(wheel: WheelInfo, target: str, python: str) -> None:
    major, minor = (int(part) for part in python.split(".")[:2])
    python_tags = wheel.python_tag.split(".")
    compatible_python = False
    for tag in python_tags:
        match = re.fullmatch(r"cp(\d)(\d+)", tag)
        if tag in {"py3", f"py{major}", f"py{major}{minor}"}:
            compatible_python = True
        elif match:
            tagged = (int(match.group(1)), int(match.group(2)))
            compatible_python = tagged == (major, minor) or (
                wheel.abi_tag == "abi3" and tagged <= (major, minor)
            )
        if compatible_python:
            break
    if not compatible_python:
        raise ValueError(
            f"wheel {wheel.path.name} does not match runtime Python {major}.{minor}"
        )

    platform_tags = wheel.platform_tag.split(".")
    if "any" in platform_tags:
        return
    if target == "windows-x86_64":
        compatible_platform = any(tag == "win_amd64" for tag in platform_tags)
    elif target == "windows-arm64":
        compatible_platform = any(tag == "win_arm64" for tag in platform_tags)
    elif target == "darwin-x86_64":
        compatible_platform = any(
            tag.startswith("macosx_") and tag.endswith(("_x86_64", "_universal2"))
            for tag in platform_tags
        )
    elif target == "darwin-arm64":
        compatible_platform = any(
            tag.startswith("macosx_") and tag.endswith(("_arm64", "_universal2"))
            for tag in platform_tags
        )
    elif target == "linux-x86_64":
        compatible_platform = any(
            "linux" in tag and tag.endswith("_x86_64") for tag in platform_tags
        )
    else:
        compatible_platform = any(
            "linux" in tag and tag.endswith(("_aarch64", "_arm64"))
            for tag in platform_tags
        )
    if not compatible_platform:
        raise ValueError(
            f"wheel {wheel.path.name} does not match target {target}"
        )


def _safe_wheel_member(name: str) -> Path | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    parts = list(path.parts)
    if parts[0].endswith(".data"):
        if len(parts) < 3 or parts[1] not in {"purelib", "platlib"}:
            return None
        parts = parts[2:]
    if not parts:
        return None
    return Path(*parts)


def extract_wheel(
    wheel: Path,
    destination: Path,
    *,
    compact: bool = False,
) -> int:
    """Install a wheel as data, without pip or executing package code."""
    count = 0
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            relative = _safe_wheel_member(info.filename)
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if (
                relative is None
                or info.is_dir()
                or stat.S_ISLNK(unix_mode)
                or (
                    compact
                    and any(
                        part.lower()
                        in {
                            "__pycache__",
                            ".pytest_cache",
                            "pyobjctest",
                            "test",
                            "tests",
                        }
                        for part in relative.parts
                    )
                )
            ):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            if unix_mode:
                target.chmod(unix_mode & 0o777)
            count += 1
    return count


def _copy_stdlib(source: Path, destination: Path, compact: bool = False) -> None:
    patterns = [
        "site-packages",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "test",
        "tests",
        "idlelib",
        "ensurepip",
    ]
    if compact:
        patterns.extend(
            [
                "config-*",
                "lib2to3",
                "turtledemo",
                "tkinter",
                "unittest",
                "pydoc_data",
            ]
        )
    ignored = shutil.ignore_patterns(*patterns)
    shutil.copytree(source, destination, ignore=ignored)


def _freeze_current_runtime(
    destination: Path, compact: bool = False
) -> tuple[Path, dict[str, str]]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    stdlib = Path(sysconfig.get_path("stdlib"))
    if sys.platform == "darwin" and sysconfig.get_config_var("PYTHONFRAMEWORK"):
        framework_name = str(sysconfig.get_config_var("PYTHONFRAMEWORK"))
        framework = destination / f"{framework_name}.framework"
        version_root = framework / "Versions" / version
        version_root.mkdir(parents=True)
        framework_binary = Path(sys.base_prefix) / framework_name
        shutil.copy2(framework_binary, version_root / framework_name)
        executable = version_root / "bin" / "python3"
        executable.parent.mkdir()
        # Use the normal framework command-line executable. Modern python.org
        # Python.app executables are signed together with their original app
        # Info.plist and are killed by macOS if copied out of that bundle.
        # The bin/python executable has an independent signature and relocates
        # through DYLD_FRAMEWORK_PATH without a post-build codesign step.
        executable_source = Path(sys.executable).resolve()
        framework_executable = (
            Path(sys.base_prefix)
            / "Resources"
            / "Python.app"
            / "Contents"
            / "MacOS"
            / "Python"
        )
        if not executable_source.is_file():
            executable_source = framework_executable
        shutil.copy2(executable_source, executable)
        _copy_stdlib(stdlib, version_root / "lib" / f"python{version}", compact)
        return executable, {
            "PYTHONHOME": str(version_root.relative_to(destination.parent)),
            "DYLD_FRAMEWORK_PATH": str(destination.relative_to(destination.parent)),
        }
    if os.name == "nt":
        executable = destination / "python.exe"
        shutil.copy2(Path(sys.executable), executable)
        for candidate in Path(sys.base_prefix).glob("python*.dll"):
            shutil.copy2(candidate, destination / candidate.name)
        _copy_stdlib(stdlib, destination / "Lib", compact)
        return executable, {"PYTHONHOME": str(destination.relative_to(destination.parent))}

    executable = destination / "bin" / "python3"
    executable.parent.mkdir(parents=True)
    shutil.copy2(Path(sys.executable).resolve(), executable)
    library_directory = Path(str(sysconfig.get_config_var("LIBDIR") or ""))
    library_name = str(sysconfig.get_config_var("LDLIBRARY") or "")
    if library_name and (library_directory / library_name).is_file():
        (destination / "lib").mkdir()
        shutil.copy2(library_directory / library_name, destination / "lib" / library_name)
    _copy_stdlib(stdlib, destination / "lib" / f"python{version}", compact)
    return executable, {
        "PYTHONHOME": str(destination.relative_to(destination.parent)),
        "LD_LIBRARY_PATH": str((destination / "lib").relative_to(destination.parent)),
    }


def create_runtime_pack(
    output: Path,
    *,
    compact: bool = False,
    clean: bool = False,
) -> RuntimePackResult:
    """Snapshot the current target-compatible CPython runtime for later reuse."""

    output = output.expanduser().resolve()
    if output.exists() and not clean:
        raise FileExistsError(f"output already exists: {output} (use --clean)")
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="py2bin-runtime-pack-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        runtime_root = stage if os.name == "nt" else stage / "runtime"
        if runtime_root != stage:
            runtime_root.mkdir()
        executable, environment = _freeze_current_runtime(
            runtime_root, compact=compact
        )
        target = host_target()
        python = (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )
        write_runtime_manifest(
            stage,
            target=target,
            python=python,
            executable=executable.relative_to(stage),
            environment=environment,
        )
        stage.replace(output)
    files = [path for path in output.rglob("*") if path.is_file()]
    return RuntimePackResult(
        output,
        len(files),
        sum(path.stat().st_size for path in files),
        target,
        python,
    )


def _shell_launcher(path: Path, runtime: Path, environment: dict[str, str]) -> None:
    lines = [
        "#!/bin/sh",
        "set -eu",
        'case "$0" in /*) SELF="$0" ;; *) SELF="$PWD/$0" ;; esac',
        'ROOT=${SELF%/*}',
        'ROOT=$(CDPATH= cd -- "$ROOT" && pwd)',
    ]
    for key, relative in environment.items():
        lines.append(f'export {key}="$ROOT/{relative}"')
    lines.extend(
        [
            'export PYTHONNOUSERSITE=1',
            'export PYTHONDONTWRITEBYTECODE=1',
            'export PYTHONPATH="$ROOT/app:$ROOT/site-packages"',
            f'exec "$ROOT/{runtime.as_posix()}" -s "$ROOT/py2bin_bootstrap.py" "$@"',
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _frozen_macos_app(
    payload: Path,
    app: Path,
    name: str,
    payload_launcher: Path,
    icon: Path | None,
    runtime: Path,
    environment: dict[str, str],
    target: str,
) -> Path:
    contents = app / "Contents"
    resources = contents / "Resources"
    macos = contents / "MacOS"
    resources.mkdir(parents=True)
    macos.mkdir()
    icon_filename = install_macos_icon(icon, resources)
    bundle = resources / "bundle"
    payload.replace(bundle)
    info_plist = macos_info_plist(name, name, icon_filename)
    resource_files = {
        path.relative_to(contents).as_posix(): path
        for path in resources.rglob("*")
        if path.is_file()
    }
    code_resources = plistlib.dumps(
        {
            "files": {
                relative: hashlib.sha1(path.read_bytes()).digest()
                for relative, path in resource_files.items()
            },
            "files2": {
                relative: {"hash2": hashlib.sha256(path.read_bytes()).digest()}
                for relative, path in resource_files.items()
            },
            "rules": {
                "^Resources/": True,
                "^Resources/.*\\.lproj/": {"optional": True, "weight": 1000},
                "^Resources/.*\\.lproj/locversion.plist$": {
                    "omit": True,
                    "weight": 1100,
                },
                "^Resources/Base\\.lproj/": {"weight": 1010},
                "^version.plist$": True,
            },
            "rules2": {
                ".*\\.dSYM($|/)": {"weight": 11},
                "^(.*/)?\\.DS_Store$": {"omit": True, "weight": 2000},
                "^(Frameworks|SharedFrameworks|PlugIns|Plug-ins|XPCServices|Helpers|MacOS)/": {
                    "nested": True,
                    "weight": 10,
                },
                "^.*": True,
                "^Info\\.plist$": {"omit": True, "weight": 20},
                "^PkgInfo$": {"omit": True, "weight": 20},
                "^Resources/": {"weight": 20},
            },
        },
        fmt=plistlib.FMT_XML,
        sort_keys=True,
    )
    signature_directory = contents / "_CodeSignature"
    signature_directory.mkdir()
    (signature_directory / "CodeResources").write_bytes(code_resources)
    payload_launcher.relative_to(payload)  # validate that the launcher belongs to the payload
    launcher = macos / name
    exports = " ".join(
        f'export {key}="$ROOT/{relative}";'
        for key, relative in environment.items()
    )
    command = (
        'set -eu; SELF="$0"; CONTENTS=${SELF%/*/*}; '
        'ROOT="$CONTENTS/Resources/bundle"; '
        'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; '
        f"{exports} "
        'export PYTHONPATH="$ROOT/app:$ROOT/site-packages"; '
        f'exec "$ROOT/{runtime.as_posix()}" -B -s "$ROOT/py2bin_bootstrap.py" "$@"'
    )
    launcher.write_bytes(
        macos_shell_launcher(
            command,
            machine=target.rpartition("-")[2],
            info_plist=info_plist,
            code_resources=code_resources,
        )
    )
    launcher.chmod(0o755)
    (contents / "Info.plist").write_bytes(info_plist)
    return launcher


def _frozen_macos_onefile_app(
    payload: Path,
    app: Path,
    name: str,
    payload_launcher: Path,
    icon: Path | None,
    target: str,
) -> Path:
    """Create the mandatory .app shell around one embedded payload file."""

    contents = app / "Contents"
    resources = contents / "Resources"
    macos = contents / "MacOS"
    resources.mkdir(parents=True)
    macos.mkdir()
    icon_filename = install_macos_icon(icon, resources)
    info_plist = macos_info_plist(name, name, icon_filename)
    resource_files = {
        path.relative_to(contents).as_posix(): path
        for path in resources.rglob("*")
        if path.is_file()
    }
    code_resources = plistlib.dumps(
        {
            "files": {
                relative: hashlib.sha1(path.read_bytes()).digest()
                for relative, path in resource_files.items()
            },
            "files2": {
                relative: {"hash2": hashlib.sha256(path.read_bytes()).digest()}
                for relative, path in resource_files.items()
            },
            "rules": {
                "^Resources/": True,
                "^Resources/.*\\.lproj/": {"optional": True, "weight": 1000},
                "^Resources/.*\\.lproj/locversion.plist$": {
                    "omit": True,
                    "weight": 1100,
                },
                "^Resources/Base\\.lproj/": {"weight": 1010},
                "^version.plist$": True,
            },
            "rules2": {
                ".*\\.dSYM($|/)": {"weight": 11},
                "^(.*/)?\\.DS_Store$": {"omit": True, "weight": 2000},
                "^(Frameworks|SharedFrameworks|PlugIns|Plug-ins|XPCServices|Helpers|MacOS)/": {
                    "nested": True,
                    "weight": 10,
                },
                "^.*": True,
                "^Info\\.plist$": {"omit": True, "weight": 20},
                "^PkgInfo$": {"omit": True, "weight": 20},
                "^Resources/": {"weight": 20},
            },
        },
        fmt=plistlib.FMT_XML,
        sort_keys=True,
    )
    launcher = macos / name
    create_onefile(
        payload,
        launcher,
        target=target,
        launcher=payload_launcher,
        info_plist=info_plist,
        code_resources=code_resources,
    )
    (contents / "Info.plist").write_bytes(info_plist)
    signature_directory = contents / "_CodeSignature"
    signature_directory.mkdir()
    (signature_directory / "CodeResources").write_bytes(code_resources)
    return launcher


def default_fetch_cache() -> Path:
    """Where verified downloads are cached between builds."""

    override = os.environ.get("PY2BIN_CACHE_DIR")
    root = Path(override) if override else Path.home() / ".cache" / "py2bin"
    return root / "fetch"


def _auto_fetch_runtime(
    target: str,
    fetch_cache: Path | None,
    fetch_lock: Path | None,
    fetch_python: str | None,
) -> Path:
    """Download and verify a target CPython runtime pack."""

    from .runtime_fetch import FetchError, FetchLock, fetch_windows_runtime

    cache = (fetch_cache or default_fetch_cache()).expanduser().resolve()
    version = fetch_python or _DEFAULT_FETCH_PYTHON
    destination = cache / "runtimes" / f"cpython-{version}-{target}"
    lock = FetchLock.load(fetch_lock)
    lock.path = fetch_lock
    if destination.is_dir() and (destination / "py2bin-runtime.json").is_file():
        return destination
    try:
        fetch_windows_runtime(
            version,
            target,
            destination,
            cache=cache / "blobs",
            lock=lock,
            clean=True,
        )
    except FetchError as error:
        raise ValueError(
            f"could not fetch a {target} CPython runtime: {error}"
        ) from error
    return destination


def _auto_fetch_wheels(
    missing: set[str],
    target: str,
    python_version: str,
    fetch_cache: Path | None,
    fetch_lock: Path | None,
) -> tuple[Path, ...]:
    """Download verified target wheels for every missing distribution."""

    from .runtime_fetch import FetchError, FetchLock, fetch_wheel

    cache = (fetch_cache or default_fetch_cache()).expanduser().resolve()
    output = cache / "wheels" / target
    lock = FetchLock.load(fetch_lock)
    lock.path = fetch_lock
    fetched: list[Path] = []
    failures: list[str] = []
    short_version = ".".join(python_version.split(".")[:2])
    for project in sorted(missing):
        try:
            result = fetch_wheel(
                project,
                target,
                short_version,
                output,
                cache=cache / "blobs",
                lock=lock,
            )
        except FetchError as error:
            failures.append(f"{project}: {error}")
            continue
        fetched.append(result.path)
    lock.save()
    if failures:
        raise ValueError(
            "could not fetch every target wheel:\n  " + "\n  ".join(failures)
        )
    return tuple(fetched)


def _missing_pack_wheels(analysis, wheels: tuple[WheelInfo, ...]) -> set[str]:
    """Distributions a runtime-pack build still needs a target wheel for.

    This mirrors :func:`_validate_pack_wheel_closure`, which reports the same
    set as an error. Unresolved imports are included by their import name so a
    project that is only reachable as a bare import can still be fetched.
    """

    by_name = {_canonical_distribution(wheel.name) for wheel in wheels}
    by_top_level = {
        top_level: wheel for wheel in wheels for top_level in wheel.top_levels
    }
    required = {_canonical_distribution(name) for name in analysis.distributions}
    for module in analysis.modules:
        wheel = by_top_level.get(module.partition(".")[0])
        if wheel is not None:
            required.add(_canonical_distribution(wheel.name))
    missing = {name for name in required if name not in by_name}
    return missing


def _unmapped_imports(analysis, wheels: tuple[WheelInfo, ...]) -> set[str]:
    """Bare imports with no wheel and no distribution behind them."""

    by_top_level = {
        top_level for wheel in wheels for top_level in wheel.top_levels
    }
    return {
        unresolved
        for unresolved in analysis.unresolved
        if unresolved not in by_top_level
    }


def _missing_wheel_requirements(wheels: tuple[WheelInfo, ...]) -> set[str]:
    """Unsatisfied unconditional requirements of the supplied wheels."""

    by_name = {_canonical_distribution(wheel.name) for wheel in wheels}
    missing: set[str] = set()
    for wheel in wheels:
        for requirement in wheel.requirements:
            # Requirements carrying an environment marker are conditional and
            # are left to the caller, matching the validator below.
            if ";" in requirement:
                continue
            match = _REQUIREMENT_NAME.match(requirement)
            if not match:
                continue
            dependency = _canonical_distribution(match.group(1))
            if dependency not in by_name:
                missing.add(dependency)
    return missing


def _validate_pack_wheel_closure(
    analysis,
    wheels: tuple[WheelInfo, ...],
    dependency_mode: str,
) -> None:
    by_name = {_canonical_distribution(wheel.name): wheel for wheel in wheels}
    by_top_level = {
        top_level: wheel
        for wheel in wheels
        for top_level in wheel.top_levels
    }
    required = {
        _canonical_distribution(name) for name in analysis.distributions
    }
    for module in analysis.modules:
        wheel = by_top_level.get(module.partition(".")[0])
        if wheel is not None:
            required.add(_canonical_distribution(wheel.name))
    for unresolved in tuple(analysis.unresolved):
        wheel = by_top_level.get(unresolved)
        if wheel is not None:
            analysis.unresolved.remove(unresolved)
            required.add(_canonical_distribution(wheel.name))
    if analysis.unresolved:
        raise ValueError(
            "runtime-pack build has unresolved imports without target wheels: "
            + ", ".join(sorted(analysis.unresolved))
        )
    missing = required - by_name.keys()
    if missing:
        raise ValueError(
            "runtime-pack build requires target wheels for: "
            + ", ".join(sorted(missing))
        )
    if dependency_mode == "closure":
        pending = list(required)
        visited: set[str] = set()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            wheel = by_name[name]
            for requirement in wheel.requirements:
                # Requirements with environment markers cannot be evaluated
                # safely without executing a third-party marker engine. The
                # target pack author supplies those conditionally.
                if ";" in requirement:
                    continue
                match = _REQUIREMENT_NAME.match(requirement)
                if not match:
                    continue
                dependency = _canonical_distribution(match.group(1))
                if dependency not in by_name:
                    raise ValueError(
                        f"wheel {wheel.path.name} requires target wheel "
                        f"{match.group(1)!r}; supply the complete wheel closure"
                    )
                pending.append(dependency)


def freeze(
    entry: Path,
    output: Path,
    source_root: Path | None = None,
    includes: tuple[str, ...] = (),
    excludes: tuple[str, ...] = (),
    wheels: tuple[Path, ...] = (),
    dependency_mode: str = "closure",
    clean: bool = False,
    *,
    app: bool = False,
    name: str | None = None,
    icon: Path | None = None,
    compact: bool = False,
    runtime_pack: Path | None = None,
    target: str | None = None,
    onefile: bool = True,
    auto_fetch: bool = False,
    fetch_cache: Path | None = None,
    fetch_lock: Path | None = None,
    fetch_python: str | None = None,
    fetch_map: dict[str, str] | None = None,
) -> FreezeResult:
    """Create a no-installed-Python bundle for a compatible target runtime."""
    fetch_map = dict(fetch_map or {})
    entry = entry.expanduser().resolve()
    output = output.expanduser().resolve()
    source_root = (source_root or entry.parent).expanduser().resolve()
    name = name or (
        output.stem
        if output.suffix.lower() in {".app", ".bin", ".exe"}
        else output.name
    )
    if (
        auto_fetch
        and runtime_pack is None
        and target is not None
        and target != host_target()
    ):
        # The build machine has no runtime for this target, but the target's
        # CPython is published, so retrieve and verify it instead of failing.
        runtime_pack = _auto_fetch_runtime(
            target, fetch_cache, fetch_lock, fetch_python
        )
    runtime_pack_info = (
        inspect_runtime_pack(runtime_pack) if runtime_pack is not None else None
    )
    bundle_target = (
        runtime_pack_info.target if runtime_pack_info is not None else host_target()
    )
    if target is not None and target != bundle_target:
        requirement = (
            f"runtime pack for {target}"
            if runtime_pack_info is not None
            else f"--runtime-pack for {target}"
        )
        raise ValueError(
            f"requested target {target} does not match available runtime "
            f"{bundle_target}; supply {requirement}"
        )
    runtime_python_version = (
        runtime_pack_info.python
        if runtime_pack_info is not None
        else f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    macos_app = app and bundle_target.startswith("darwin-")
    windows_app = app and bundle_target.startswith("windows-")
    if app:
        if not (macos_app or windows_app):
            raise ValueError(
                "--app currently requires a macOS or Windows runtime"
            )
        if macos_app:
            output = _required_suffix(output, ".app")
        elif windows_app and onefile:
            output = _required_suffix(output, ".exe")
    elif onefile:
        required_suffix = ".exe" if bundle_target.startswith("windows-") else ".bin"
        output = _required_suffix(output, required_suffix)
    if icon is not None and not (
        app or bundle_target.startswith("windows-")
    ):
        raise ValueError("--icon requires a macOS app or Windows executable")
    if icon is not None:
        icon = icon.expanduser().resolve()
        if not icon.is_file():
            raise FileNotFoundError(f"icon does not exist: {icon}")
    if not entry.is_file():
        raise FileNotFoundError(f"entry script does not exist: {entry}")
    if not entry.is_relative_to(source_root):
        raise ValueError("entry must be inside source_root")
    missing_wheels = [str(path) for path in wheels if not path.expanduser().is_file()]
    if missing_wheels:
        raise FileNotFoundError("wheel does not exist: " + ", ".join(missing_wheels))
    wheel_infos = tuple(inspect_wheel(wheel) for wheel in wheels)
    for wheel_info in wheel_infos:
        _wheel_matches_target(wheel_info, bundle_target, runtime_python_version)
    if output.exists() and not clean:
        raise FileExistsError(f"output already exists: {output} (use --clean)")
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()

    analysis_mode = (
        "imported"
        if runtime_pack_info is not None and dependency_mode == "closure"
        else dependency_mode
    )
    analysis = analyze(entry, source_root, includes, excludes, analysis_mode)
    if runtime_pack_info is not None:
        if auto_fetch:
            missing = _missing_pack_wheels(analysis, wheel_infos)
            # An import name is not a PyPI project name, and the two namespaces
            # are not even close for common packages ("webview" is pywebview,
            # "PIL" is pillow). Fetching a project guessed from an import name
            # could install an unrelated or hostile package, so require an
            # explicit mapping instead of guessing.
            unmapped = _unmapped_imports(analysis, wheel_infos)
            for module in sorted(unmapped):
                project = fetch_map.get(module)
                if project is None:
                    raise ValueError(
                        f"cannot auto-fetch a wheel for the bare import {module!r}: "
                        "an import name is not a PyPI project name. Supply "
                        f"--fetch-map {module}=PROJECT, or provide the wheel with "
                        "--wheel/--wheel-dir."
                    )
                missing.add(_canonical_distribution(project))
            # Fetch, then follow each new wheel's own requirements, until the
            # closure is satisfied. The round limit stops a dependency cycle or
            # a mis-resolving index from looping forever.
            for _round in range(16):
                if not missing:
                    break
                extra = _auto_fetch_wheels(
                    missing,
                    bundle_target,
                    runtime_python_version,
                    fetch_cache,
                    fetch_lock,
                )
                wheel_infos = (
                    *wheel_infos,
                    *(inspect_wheel(path) for path in extra),
                )
                wheels = (*wheels, *extra)
                missing = _missing_pack_wheels(analysis, wheel_infos)
                if dependency_mode == "closure":
                    missing |= _missing_wheel_requirements(wheel_infos)
        _validate_pack_wheel_closure(analysis, wheel_infos, dependency_mode)
        analysis.distributions = {wheel.name for wheel in wheel_infos}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="py2bin-freeze-", dir=output.parent) as temporary:
        stage = Path(temporary) / "payload"
        stage.mkdir()
        excluded = [Path(temporary)]
        if runtime_pack is not None:
            excluded.append(runtime_pack.expanduser().resolve())
        excluded.extend(wheel.path for wheel in wheel_infos)
        _copy_project(source_root, stage / "app", tuple(excluded))
        packages = stage / "site-packages"
        packages.mkdir()
        if runtime_pack_info is None:
            for distribution in sorted(analysis.distributions, key=str.lower):
                _copy_distribution(distribution, packages, compact=compact)
        for wheel in wheels:
            extract_wheel(
                wheel.expanduser().resolve(),
                packages,
                compact=compact,
            )

        if runtime_pack is not None:
            installed_pack = install_runtime_pack(
                runtime_pack,
                stage,
                compact=compact,
            )
            runtime_executable = stage / installed_pack.executable
            runtime_environment = installed_pack.environment
        else:
            runtime_root = stage if os.name == "nt" else stage / "runtime"
            if runtime_root != stage:
                runtime_root.mkdir()
            runtime_executable, runtime_environment = _freeze_current_runtime(
                runtime_root, compact=compact
            )
        runtime_relative = runtime_executable.relative_to(stage)
        entry_relative = entry.relative_to(source_root).as_posix()
        windows_app_id = (
            _windows_app_user_model_id(name) if windows_app else None
        )
        manifest = {
            "schema": 1,
            "entry": entry_relative,
            "python": runtime_python_version,
            "target": bundle_target,
            "distributions": sorted(analysis.distributions, key=str.lower),
            "wheels": [path.name for path in wheels],
            "compact": compact,
            "web_assets": _web_assets(
                stage,
                (stage / "app", packages),
            ),
            "windows_app_user_model_id": windows_app_id,
        }
        (stage / "py2bin-freeze.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        bootstrap_entry = repr(entry_relative)
        bootstrap_app_id = repr(windows_app_id)
        (stage / "py2bin_bootstrap.py").write_text(
            "import os, runpy, sys\n"
            f"_ENTRY = {bootstrap_entry}\n"
            f"_WINDOWS_APP_ID = {bootstrap_app_id}\n"
            "def main(from_site=False):\n"
            "    root = os.path.dirname(os.path.abspath(__file__))\n"
            "    if _WINDOWS_APP_ID and sys.platform == 'win32':\n"
            "        try:\n"
            "            import ctypes\n"
            "            setter = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID\n"
            "            setter.argtypes = [ctypes.c_wchar_p]\n"
            "            setter.restype = ctypes.c_long\n"
            "            setter(_WINDOWS_APP_ID)\n"
            "        except BaseException:\n"
            "            pass\n"
            "    app_root = os.path.join(root, 'app')\n"
            "    sys.path[:0] = [app_root, os.path.join(root, 'site-packages')]\n"
            "    entry = os.path.join(app_root, *_ENTRY.split('/'))\n"
            "    sys.argv[0] = entry\n"
            "    os.environ['PY2BIN_BUNDLE_ROOT'] = root\n"
            "    if not from_site:\n"
            "        runpy.run_path(entry, run_name='__main__')\n"
            "        return\n"
            "    status = 0\n"
            "    try:\n"
            "        runpy.run_path(entry, run_name='__main__')\n"
            "    except SystemExit as error:\n"
            "        status = error.code if isinstance(error.code, int) else 1\n"
            "    except BaseException:\n"
            "        import traceback\n"
            "        report = traceback.format_exc()\n"
            "        try:\n"
            "            with open(os.path.join(root, 'py2bin-error.log'), 'w', encoding='utf-8') as stream:\n"
            "                stream.write(report)\n"
            "        except BaseException:\n"
            "            pass\n"
            "        try:\n"
            "            if sys.stderr is not None:\n"
            "                sys.stderr.write(report)\n"
            "        except BaseException:\n"
            "            pass\n"
            "        status = 1\n"
            "    for stream in (sys.stdout, sys.stderr):\n"
            "        try:\n"
            "            if stream is not None:\n"
            "                stream.flush()\n"
            "        except BaseException:\n"
            "            pass\n"
            "    os._exit(status)\n"
            "if __name__ == '__main__': main()\n",
            encoding="utf-8",
            newline="\n",
        )
        if bundle_target.startswith("windows-"):
            launcher = stage / f"{name}.exe"
            runtime_path_files = tuple(
                runtime_executable.parent.glob("python*._pth")
            )
            launcher_source = runtime_executable
            if windows_app:
                windowed_runtime = runtime_executable.with_name("pythonw.exe")
                if not windowed_runtime.is_file():
                    raise ValueError(
                        "Windows --app requires pythonw.exe in the runtime pack"
                    )
                launcher_source = windowed_runtime
            launcher_source.replace(launcher)
            major, minor = runtime_python_version.split(".")[:2]
            isolated_path = (
                f"python{major}{minor}.zip\n"
                "Lib\nsite-packages\napp\n.\nimport site\n"
            )
            for path_file in (
                *runtime_path_files,
                launcher.with_suffix("._pth"),
            ):
                path_file.write_text(isolated_path, encoding="utf-8", newline="\n")
            (stage / "sitecustomize.py").write_text(
                "from py2bin_bootstrap import main\nmain(from_site=True)\n",
                encoding="utf-8",
                newline="\n",
            )
            if windows_app:
                install_windows_identity(
                    launcher,
                    name,
                    version="1.0.0.0",
                    icon=icon,
                )
            elif icon is not None:
                install_windows_icon(launcher, icon)
        else:
            launcher = stage / f"{name}.bin"
            _shell_launcher(launcher, runtime_relative, runtime_environment)
        if macos_app:
            app_stage = Path(temporary) / output.name
            if onefile:
                launcher = _frozen_macos_onefile_app(
                    stage,
                    app_stage,
                    name,
                    launcher,
                    icon,
                    bundle_target,
                )
            else:
                launcher = _frozen_macos_app(
                    stage,
                    app_stage,
                    name,
                    launcher,
                    icon,
                    runtime_relative,
                    runtime_environment,
                    bundle_target,
                )
            app_stage.replace(output)
        elif onefile:
            file_stage = Path(temporary) / output.name
            create_onefile(
                stage,
                file_stage,
                target=bundle_target,
                launcher=launcher,
                icon=icon,
                windows_windowed=windows_app,
            )
            launcher = file_stage
            file_stage.replace(output)
        else:
            stage.replace(output)

    if macos_app:
        launcher = output / "Contents" / "MacOS" / name
    else:
        launcher_suffix = ".exe" if bundle_target.startswith("windows-") else ".bin"
        launcher = (
            output
            if onefile
            else output / f"{name}{launcher_suffix}"
        )
    files = (
        [output]
        if output.is_file()
        else [path for path in output.rglob("*") if path.is_file()]
    )
    return FreezeResult(
        output,
        launcher,
        len(files),
        sum(path.stat().st_size for path in files),
        tuple(sorted(analysis.distributions, key=str.lower)),
        bundle_target,
        runtime_python_version,
        onefile,
    )
