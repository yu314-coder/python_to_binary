from __future__ import annotations

import json
import hashlib
import os
import plistlib
import shutil
import stat
import sys
import sysconfig
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .analyzer import analyze
from .builder import _copy_distribution, _copy_project
from .icons import install_macos_icon, macos_info_plist
from .native.launcher import macos_shell_launcher


@dataclass(frozen=True, slots=True)
class FreezeResult:
    bundle: Path
    launcher: Path
    files: int
    bytes: int
    distributions: tuple[str, ...]


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


def extract_wheel(wheel: Path, destination: Path) -> int:
    """Install a wheel as data, without pip or executing package code."""
    count = 0
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            relative = _safe_wheel_member(info.filename)
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if relative is None or info.is_dir() or stat.S_ISLNK(unix_mode):
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
        # Framework installations commonly expose a small launcher in bin/
        # which tries to posix_spawn Resources/Python.app. Copy the real
        # framework executable instead so the frozen runtime does not retain
        # that unbundled path.
        framework_executable = (
            Path(sys.base_prefix)
            / "Resources"
            / "Python.app"
            / "Contents"
            / "MacOS"
            / "Python"
        )
        executable_source = (
            framework_executable
            if framework_executable.is_file()
            else Path(sys.executable).resolve()
        )
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _frozen_macos_app(
    payload: Path,
    app: Path,
    name: str,
    payload_launcher: Path,
    icon: Path | None,
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
    runtime_python = (
        "runtime/Python.framework/Versions/"
        f"{sys.version_info.major}.{sys.version_info.minor}/bin/python3"
    )
    launcher = macos / name
    command = (
        'set -eu; SELF="$0"; CONTENTS=${SELF%/*/*}; '
        'ROOT="$CONTENTS/Resources/bundle"; '
        'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; '
        'export PYTHONHOME="$ROOT/runtime/Python.framework/Versions/'
        f'{sys.version_info.major}.{sys.version_info.minor}"; '
        'export DYLD_FRAMEWORK_PATH="$ROOT/runtime"; '
        'export PYTHONPATH="$ROOT/app:$ROOT/site-packages"; '
        f'exec "$ROOT/{runtime_python}" -B -s "$ROOT/py2bin_bootstrap.py" "$@"'
    )
    launcher.write_bytes(
        macos_shell_launcher(
            command,
            info_plist=info_plist,
            code_resources=code_resources,
        )
    )
    launcher.chmod(0o755)
    (contents / "Info.plist").write_bytes(info_plist)
    return launcher


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
) -> FreezeResult:
    """Create a no-installed-Python bundle for the current OS and CPU."""
    entry = entry.expanduser().resolve()
    output = output.expanduser().resolve()
    source_root = (source_root or entry.parent).expanduser().resolve()
    name = name or output.stem
    if app:
        if sys.platform != "darwin":
            raise ValueError("--app currently requires a macOS build host")
        if output.suffix != ".app":
            output = output.with_suffix(".app")
    elif icon is not None:
        raise ValueError("--icon currently requires --app")
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
    if output.exists() and not clean:
        raise FileExistsError(f"output already exists: {output} (use --clean)")
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()

    analysis = analyze(entry, source_root, includes, excludes, dependency_mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="py2bin-freeze-", dir=output.parent) as temporary:
        stage = Path(temporary) / "payload"
        stage.mkdir()
        _copy_project(source_root, stage / "app")
        packages = stage / "site-packages"
        packages.mkdir()
        for distribution in sorted(analysis.distributions, key=str.lower):
            _copy_distribution(distribution, packages, compact=compact)
        for wheel in wheels:
            extract_wheel(wheel.expanduser().resolve(), packages)

        runtime_root = stage if os.name == "nt" else stage / "runtime"
        if runtime_root != stage:
            runtime_root.mkdir()
        runtime_executable, runtime_environment = _freeze_current_runtime(
            runtime_root, compact=compact
        )
        runtime_relative = runtime_executable.relative_to(stage)
        entry_relative = entry.relative_to(source_root).as_posix()
        manifest = {
            "schema": 1,
            "entry": entry_relative,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "distributions": sorted(analysis.distributions, key=str.lower),
            "wheels": [path.name for path in wheels],
            "compact": compact,
        }
        (stage / "py2bin-freeze.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (stage / "py2bin_bootstrap.py").write_text(
            "import json, os, runpy, sys, traceback\n"
            "from pathlib import Path\n"
            "def main(from_site=False):\n"
            "    root = Path(__file__).resolve().parent\n"
            "    manifest = json.loads((root / 'py2bin-freeze.json').read_text())\n"
            "    sys.path[:0] = [str(root / 'app'), str(root / 'site-packages')]\n"
            "    entry = root / 'app' / manifest['entry']\n"
            "    sys.argv[0] = str(entry)\n"
            "    os.environ['PY2BIN_BUNDLE_ROOT'] = str(root)\n"
            "    if not from_site:\n"
            "        runpy.run_path(str(entry), run_name='__main__')\n"
            "        return\n"
            "    status = 0\n"
            "    try:\n"
            "        runpy.run_path(str(entry), run_name='__main__')\n"
            "    except SystemExit as error:\n"
            "        status = error.code if isinstance(error.code, int) else 1\n"
            "    except BaseException:\n"
            "        traceback.print_exc()\n"
            "        status = 1\n"
            "    sys.stdout.flush(); sys.stderr.flush(); os._exit(status)\n"
            "if __name__ == '__main__': main()\n",
            encoding="utf-8",
        )
        if os.name == "nt":
            launcher = stage / f"{name}.exe"
            runtime_executable.replace(launcher)
            launcher.with_suffix("._pth").write_text(
                "Lib\nsite-packages\napp\n.\nimport site\n", encoding="utf-8"
            )
            (stage / "sitecustomize.py").write_text(
                "from py2bin_bootstrap import main\nmain(from_site=True)\n",
                encoding="utf-8",
            )
        else:
            launcher = stage / f"{name}.bin"
            _shell_launcher(launcher, runtime_relative, runtime_environment)
        if app:
            app_stage = Path(temporary) / output.name
            launcher = _frozen_macos_app(stage, app_stage, name, launcher, icon)
            app_stage.replace(output)
        else:
            stage.replace(output)

    if app:
        launcher = output / "Contents" / "MacOS" / name
    else:
        launcher_suffix = ".exe" if os.name == "nt" else ".bin"
        launcher = output / f"{name}{launcher_suffix}"
    files = [path for path in output.rglob("*") if path.is_file()]
    return FreezeResult(
        output,
        launcher,
        len(files),
        sum(path.stat().st_size for path in files),
        tuple(sorted(analysis.distributions, key=str.lower)),
    )
