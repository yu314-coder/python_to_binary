from __future__ import annotations

import platform
import plistlib
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from .arm64 import encode_darwin as encode_darwin_arm64
from .arm64 import encode_linux as encode_linux_arm64
from .formats.elf import write_elf_arm64, write_elf_x86_64
from .formats.macho import write_macho_arm64, write_macho_x86_64
from .formats.pe import write_pe_arm64, write_pe_x86_64
from .frontend import lower
from .optimizer import optimize
from .x86_64 import encode


TARGETS = (
    "linux-x86_64",
    "linux-arm64",
    "darwin-x86_64",
    "darwin-arm64",
    "windows-x86_64",
    "windows-arm64",
)

OS_ALIASES = {
    "darwin": "darwin",
    "mac": "darwin",
    "macos": "darwin",
    "osx": "darwin",
    "linux": "linux",
    "windows": "windows",
    "win": "windows",
}
ARCH_ALIASES = {
    "x86_64": "x86_64",
    "x64": "x86_64",
    "amd64": "x86_64",
    "arm64": "arm64",
    "aarch64": "arm64",
}


@dataclass(frozen=True, slots=True)
class NativeResult:
    artifact: Path
    target: str
    bytes: int
    operations: int


def supported_targets() -> tuple[str, ...]:
    return TARGETS


def resolve_target(os_name: str, architecture: str) -> str:
    normalized_os = OS_ALIASES.get(os_name.lower())
    normalized_arch = ARCH_ALIASES.get(architecture.lower())
    if normalized_os is None:
        raise ValueError(
            f"unknown OS {os_name!r}; choose linux, windows, or macos"
        )
    if normalized_arch is None:
        raise ValueError(
            f"unknown architecture {architecture!r}; choose x86_64 or arm64"
        )
    target = f"{normalized_os}-{normalized_arch}"
    if target not in TARGETS:
        raise ValueError(
            f"target {target!r} is not implemented; supported: {', '.join(TARGETS)}"
        )
    return target


def host_target() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    try:
        return resolve_target(system, machine)
    except ValueError as error:
        raise ValueError(
            f"no native backend for host {system}-{machine}; choose --target explicitly"
        ) from error


def _app_metadata(executable_name: str) -> tuple[bytes, bytes]:
    info = plistlib.dumps(
        {
            "CFBundleExecutable": executable_name,
            "CFBundleName": executable_name,
            "CFBundleIdentifier": "local.py2bin.native",
            "CFBundlePackageType": "APPL",
        },
        sort_keys=False,
    )
    resources = plistlib.dumps(
        {
            "files": {},
            "files2": {},
            "rules": {"^Resources/": True, "^version.plist$": True},
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
        sort_keys=False,
    )
    return info, resources


def compile_native(
    entry: Path,
    output: Path,
    target: str | None = None,
    clean: bool = False,
    app: bool = False,
    source_roots: tuple[Path, ...] = (),
) -> NativeResult:
    entry = entry.expanduser().resolve()
    output = output.expanduser().resolve()
    target = target or host_target()
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; supported: {', '.join(TARGETS)}")
    if not entry.is_file():
        raise FileNotFoundError(f"entry script does not exist: {entry}")
    if app:
        if target != "darwin-arm64":
            raise ValueError("--app currently requires target darwin-arm64")
        if output.suffix != ".app":
            output = output.with_suffix(".app")
    if output.exists() and not clean:
        raise FileExistsError(f"output already exists: {output} (use --clean)")
    source = entry.read_text(encoding="utf-8")
    module, _optimization = optimize(lower(entry, source, source_roots))
    info_plist, code_resources = _app_metadata(entry.stem) if app else (None, None)
    if target == "windows-x86_64":
        image = write_pe_x86_64(module)
    elif target == "windows-arm64":
        image = write_pe_arm64(module)
    elif target == "linux-arm64":
        code = encode_linux_arm64(module, 0x401000)
        image = write_elf_arm64(code)
    elif target == "darwin-arm64":
        code = encode_darwin_arm64(module, 0x100004000)
        image = write_macho_arm64(code, info_plist, code_resources)
    else:
        code_address = 0x401000 if target == "linux-x86_64" else 0x100001000
        platform_name = target.partition("-")[0]
        code = encode(module, platform_name, code_address)
        image = write_elf_x86_64(code) if platform_name == "linux" else write_macho_x86_64(code)
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    if app:
        executable = output / "Contents" / "MacOS" / entry.stem
        executable.parent.mkdir(parents=True)
        executable.write_bytes(image)
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        assert info_plist is not None and code_resources is not None
        (output / "Contents" / "Info.plist").write_bytes(info_plist)
        signature_directory = output / "Contents" / "_CodeSignature"
        signature_directory.mkdir()
        (signature_directory / "CodeResources").write_bytes(code_resources)
        size = len(image) + len(info_plist) + len(code_resources)
    else:
        output.write_bytes(image)
        output.chmod(output.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        size = len(image)
    return NativeResult(output, target, size, len(module.operations))


def compile_all(
    entry: Path,
    output_directory: Path,
    clean: bool = False,
    mac_app: bool = False,
    os_name: str | None = None,
    architecture: str | None = None,
) -> tuple[NativeResult, ...]:
    """Cross-compile every backend using only this Python process.

    No target SDK, assembler, linker, compiler, emulator, or target Python is
    consulted. Existing unrelated files in output_directory are never removed.
    """
    entry = entry.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    results: list[NativeResult] = []
    if (os_name is None) != (architecture is None):
        raise ValueError("os_name and architecture must be supplied together")
    selected = (
        (resolve_target(os_name, architecture),)
        if os_name is not None and architecture is not None
        else TARGETS
    )
    for target in selected:
        extension = ".exe" if target.startswith("windows-") else ".bin"
        output = output_directory / f"{entry.stem}-{target}{extension}"
        results.append(compile_native(entry, output, target, clean=clean))
    if mac_app:
        if selected != TARGETS and "darwin-arm64" not in selected:
            raise ValueError("--mac-app requires the darwin-arm64 selection")
        output = output_directory / f"{entry.stem}-darwin-arm64.app"
        results.append(
            compile_native(entry, output, "darwin-arm64", clean=clean, app=True)
        )
    return tuple(results)
