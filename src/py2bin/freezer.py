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


def _required_suffix(path: Path, suffix: str) -> Path:
    """Append a required artifact suffix without discarding dotted names."""

    return path if path.suffix.lower() == suffix else Path(f"{path}{suffix}")


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


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
) -> FreezeResult:
    """Create a no-installed-Python bundle for a compatible target runtime."""
    entry = entry.expanduser().resolve()
    output = output.expanduser().resolve()
    source_root = (source_root or entry.parent).expanduser().resolve()
    name = name or (
        output.stem
        if output.suffix.lower() in {".app", ".bin", ".exe"}
        else output.name
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
    macos_app = app and bundle_target == "darwin-arm64"
    windows_app = app and bundle_target.startswith("windows-")
    if app:
        if not (macos_app or windows_app):
            raise ValueError(
                "--app currently requires a darwin-arm64 or Windows runtime"
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
