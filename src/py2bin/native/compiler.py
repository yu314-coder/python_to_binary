from __future__ import annotations

import ast
import platform
import plistlib
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from .arm64 import encode_darwin as encode_darwin_arm64
from .arm64 import encode_darwin_extern as encode_darwin_extern_arm64
from .arm64 import encode_linux as encode_linux_arm64
from .formats.elf import write_elf_arm64, write_elf_x86_64
from .formats.macho import (
    write_macho_arm64,
    write_macho_arm64_dynamic,
    write_macho_x86_64,
    write_macho_x86_64_dynamic,
)
from .formats.pe import write_pe_arm64, write_pe_x86_64
from .frontend import NativeCompileError, lower
from .ir import CStringConstant, ExternCall, HeapInit, Module
from .optimizer import optimize
from .x86_64 import encode
from .x86_64 import encode_darwin_extern as encode_darwin_extern_x86_64


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


def _app_metadata(
    executable_name: str,
    display_name: str | None = None,
    icon_filename: str | None = None,
) -> tuple[bytes, bytes]:
    from ..icons import macos_info_plist

    # The same plist the freeze path writes, so a compiled .app and a frozen
    # one are described to macOS in the same terms - including the icon, which
    # a bundle without one simply does not show.
    info = macos_info_plist(
        display_name or executable_name, executable_name, icon_filename
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
    experimental_kernels: bool = False,
) -> NativeResult:
    entry = entry.expanduser().resolve()
    if not entry.is_file():
        raise FileNotFoundError(f"entry script does not exist: {entry}")
    return compile_native_source(
        entry,
        entry.read_text(encoding="utf-8"),
        output,
        target=target,
        clean=clean,
        app=app,
        source_roots=source_roots,
        experimental_kernels=experimental_kernels,
    )


def compile_native_source(
    entry: Path,
    source: str,
    output: Path,
    target: str | None = None,
    clean: bool = False,
    app: bool = False,
    source_roots: tuple[Path, ...] = (),
    experimental_kernels: bool = False,
) -> NativeResult:
    """Compile in-memory Python source with py2bin's handwritten backends.

    ``entry`` supplies diagnostics and the executable name. It does not need to
    exist, which lets other handwritten frontends feed verified source into the
    same IR and PE/ELF/Mach-O writers without a temporary file.
    """
    entry = entry.expanduser().resolve()
    output = output.expanduser().resolve()
    target = target or host_target()
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; supported: {', '.join(TARGETS)}")
    if app:
        if target != "darwin-arm64":
            raise ValueError("--app currently requires target darwin-arm64")
        if output.suffix != ".app":
            output = output.with_suffix(".app")
    if output.exists() and not clean:
        raise FileExistsError(f"output already exists: {output} (use --clean)")
    module, _optimization = optimize(
        lower(
            entry,
            source,
            source_roots,
            experimental_kernels=experimental_kernels,
        )
    )
    return _emit_native_module(entry, module, output, target, app)


def compile_native_module(
    entry: Path,
    module: Module,
    output: Path,
    target: str | None = None,
    clean: bool = False,
    app: bool = False,
    app_name: str | None = None,
    icon: Path | None = None,
    python_dylib: str | None = None,
) -> NativeResult:
    """Write a verified py2bin IR module with the handwritten binary backends."""

    entry = entry.expanduser().resolve()
    output = output.expanduser().resolve()
    target = target or host_target()
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; supported: {', '.join(TARGETS)}")
    if app:
        if target != "darwin-arm64":
            raise ValueError("--app currently requires target darwin-arm64")
        if output.suffix != ".app":
            output = output.with_suffix(".app")
    if output.exists() and not clean:
        raise FileExistsError(f"output already exists: {output} (use --clean)")
    return _emit_native_module(
        entry, module, output, target, app, app_name, icon, python_dylib
    )


def _contains_extern(value: object) -> bool:
    """Recursively detect any ``ExternCall``/``CStringConstant`` in the IR."""

    if isinstance(value, (ExternCall, CStringConstant)):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_extern(item) for item in value)
    slots = getattr(type(value), "__slots__", None)
    if slots:
        return any(_contains_extern(getattr(value, name)) for name in slots)
    return False


def _module_uses_extern(module: Module) -> bool:
    if any(_contains_extern(operation) for operation in module.operations):
        return True
    return any(
        _contains_extern(operation)
        for function in module.functions
        for operation in function.operations
    )


#: Targets whose encoder implements the internal call ABI (a real frame with a
#: saved link register, arguments in the platform argument registers, and a
#: direct branch-and-link). Everything else must reject a module that contains
#: a ``Function`` rather than emit a binary that cannot make the call.
#: Targets whose dynamic-link adapter is implemented rather than designed.
#: Both are darwin: the Mach-O writer lays out `__got` and the bind opcodes,
#: and each architecture's encoder emits the reference sites it patches.
_EXTERN_CAPABLE_TARGETS = frozenset({"darwin-arm64", "darwin-x86_64"})

CALL_CAPABLE_TARGETS = frozenset(
    {
        "darwin-arm64",
        "linux-arm64",
        "darwin-x86_64",
        "linux-x86_64",
        "windows-arm64",
        "windows-x86_64",
        "windows-arm64",
    }
)

#: Targets whose encoder establishes the module's static storage block (see
#: ``Module.static_bytes``). It is a kernel-supplied anonymous mapping whose
#: base lives in a reserved callee-saved register for the whole run, and only
#: the ARM64 syscall encoders set that register up.
STATIC_CAPABLE_TARGETS = frozenset(
    {
        "darwin-arm64",
        "linux-arm64",
        "darwin-x86_64",
        "linux-x86_64",
        "windows-x86_64",
        "windows-arm64",
    }
)


def _emit_native_module(
    entry: Path,
    module: Module,
    output: Path,
    target: str,
    app: bool,
    app_name: str | None = None,
    icon: Path | None = None,
    python_dylib: str | None = None,
) -> NativeResult:
    if module.functions and target not in CALL_CAPABLE_TARGETS:
        # The call ABI (frame with a saved link register, arguments in the
        # platform argument registers, direct branch-and-link) is implemented
        # by the ARM64 encoder only. Emitting anything for the other targets
        # would produce a binary that cannot perform the call at all.
        raise NativeCompileError(
            entry,
            ast.parse("pass").body[0],
            f"function calls (and therefore recursion) are only supported for "
            f"targets {', '.join(sorted(CALL_CAPABLE_TARGETS))} so far, not "
            f"{target!r}",
        )
    if module.static_bytes and target not in STATIC_CAPABLE_TARGETS:
        raise NativeCompileError(
            entry,
            ast.parse("pass").body[0],
            f"static storage (C file-scope variables) is only supported for "
            f"targets {', '.join(sorted(STATIC_CAPABLE_TARGETS))} so far, not "
            f"{target!r}",
        )
    if target not in _EXTERN_CAPABLE_TARGETS and _module_uses_extern(module):
        # Extern (adapter-ABI) symbol calls are only wired through real dyld
        # binding on darwin-arm64 so far. Rather than emit a binary that would
        # not link its imports, reject with an exact message. The remaining
        # per-platform link design is documented in docs/NATIVE_EXTERN_ABI.md.
        raise NativeCompileError(
            entry,
            ast.parse("pass").body[0],
            f"external native symbol calls (py2bin.cabi) are only supported for "
            f"targets {', '.join(sorted(_EXTERN_CAPABLE_TARGETS))} so far, not "
            f"{target!r}; the dynamic-link adapter for this platform is "
            "design-only",
        )
    executable_name = app_name or entry.stem
    # Written before the plist that names it, because the plist is embedded in
    # the arm64 image's own signature and cannot be revised afterwards.
    icon_filename = None
    if app and icon is not None:
        from ..icons import icon_to_icns

        icon_to_icns(icon)  # refuse an unreadable icon before anything is built
        icon_filename = "AppIcon.icns"
    info_plist, code_resources = (
        _app_metadata(executable_name, app_name, icon_filename)
        if app
        else (None, None)
    )
    if target == "windows-x86_64":
        image = write_pe_x86_64(module)
    elif target == "windows-arm64":
        image = write_pe_arm64(module)
    elif target == "linux-arm64":
        code = encode_linux_arm64(module, 0x401000)
        image = write_elf_arm64(code)
    elif target == "darwin-arm64":
        code, externs, statics = encode_darwin_extern_arm64(module, 0x100004000)
        if externs:
            # Each extern names the library that provides it. libSystem is
            # always loaded; any other library (for example the CPython
            # runtime) is added as a further LC_LOAD_DYLIB with its own
            # two-level namespace ordinal.
            # The tables, not `cabi` itself: knowing which library a symbol
            # lives in is a lookup, and going through the module that calls
            # them would pull `ctypes` - and with it `subprocess` - into a
            # build that needs neither.
            from ..cabi_tables import _cpython_library, symbol_library

            interpreter = _cpython_library()
            symbol_libraries = {}
            for _offset, symbol in externs:
                library = symbol_library(symbol)
                if library is None:
                    continue
                if python_dylib is not None and library == interpreter:
                    # The build machine's absolute path is what makes an
                    # artifact refuse to start anywhere else: dyld resolves
                    # LC_LOAD_DYLIB before a line of the program runs, so a
                    # Mac without that exact interpreter at that exact path
                    # gets no error from the program, only a refusal to
                    # launch. A path relative to the executable travels.
                    library = python_dylib
                symbol_libraries[symbol] = library
            from ..cabi_tables import OBJC_FRAMEWORKS, _OBJC_SYMBOLS

            # A framework has to be loaded for its classes to exist, even when
            # no symbol is taken from it directly: the runtime looks classes up
            # by name and finds nothing until the framework registers them.
            frameworks = (
                OBJC_FRAMEWORKS
                if any(symbol in _OBJC_SYMBOLS for symbol in symbol_libraries)
                else ()
            )
            libraries = ("/usr/lib/libSystem.B.dylib", *dict.fromkeys(
                library
                for library in (*symbol_libraries.values(), *frameworks)
                if library != "/usr/lib/libSystem.B.dylib"
            ))
            image = write_macho_arm64_dynamic(
                code,
                externs,
                statics,
                module.static_bytes,
                info_plist,
                code_resources,
                libraries=libraries,
                symbol_libraries=symbol_libraries,
            )
        else:
            # No externs, so nothing can call back in and the mapped static
            # block through X28 is sound. That image has no __DATA to put
            # statics in, so it is encoded the original way.
            code, _externs, _statics = encode_darwin_extern_arm64(
                module, 0x100004000, image_statics=False
            )
            image = write_macho_arm64(code, info_plist, code_resources)
    elif target == "darwin-x86_64" and _module_uses_extern(module):
        code, externs, statics = encode_darwin_extern_x86_64(module, 0x100001000)
        from ..cabi_tables import (
            OBJC_FRAMEWORKS,
            _OBJC_SYMBOLS,
            _cpython_library,
            symbol_library,
        )

        interpreter = _cpython_library()
        symbol_libraries = {}
        for _offset, symbol in externs:
            library = symbol_library(symbol)
            if library is None:
                continue
            if python_dylib is not None and library == interpreter:
                library = python_dylib
            symbol_libraries[symbol] = library
        frameworks = (
            OBJC_FRAMEWORKS
            if any(symbol in _OBJC_SYMBOLS for symbol in symbol_libraries)
            else ()
        )
        libraries = ("/usr/lib/libSystem.B.dylib", *dict.fromkeys(
            library
            for library in (*symbol_libraries.values(), *frameworks)
            if library != "/usr/lib/libSystem.B.dylib"
        ))
        image = write_macho_x86_64_dynamic(
            code,
            externs,
            statics,
            module.static_bytes,
            info_plist,
            code_resources,
            libraries=libraries,
            symbol_libraries=symbol_libraries,
        )
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
        executable = output / "Contents" / "MacOS" / executable_name
        executable.parent.mkdir(parents=True)
        executable.write_bytes(image)
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        assert info_plist is not None and code_resources is not None
        (output / "Contents" / "Info.plist").write_bytes(info_plist)
        if icon is not None:
            from ..icons import install_macos_icon

            install_macos_icon(icon, output / "Contents" / "Resources")
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
    source_roots: tuple[Path, ...] = (),
    experimental_kernels: bool = False,
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
        results.append(
            compile_native(
                entry,
                output,
                target,
                clean=clean,
                source_roots=source_roots,
                experimental_kernels=experimental_kernels,
            )
        )
    if mac_app:
        if selected != TARGETS and "darwin-arm64" not in selected:
            raise ValueError("--mac-app requires the darwin-arm64 selection")
        output = output_directory / f"{entry.stem}-darwin-arm64.app"
        results.append(
            compile_native(
                entry,
                output,
                "darwin-arm64",
                clean=clean,
                app=True,
                source_roots=source_roots,
                experimental_kernels=experimental_kernels,
            )
        )
    return tuple(results)
