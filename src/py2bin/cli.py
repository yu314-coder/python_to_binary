from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from . import __version__
from .analyzer import analyze
from .assembler import assemble
from .builder import build
from .capabilities import assess_entry, common_libraries, format_catalog
from .c_native import (
    CNativeCompileError,
    compile_c_native,
    compile_python_via_c,
)
from .csource import CSourceError, compile_c_file, plan_c
from .freezer import create_runtime_pack, freeze
from .model import ArtifactKind, BuildConfig
from .native import (
    AOTPlanError,
    NativeCompileError,
    audit_native_library,
    build_aot_application,
    compile_all,
    compile_native,
    host_target,
    plan_aot_application,
    require_native_library,
    resolve_target,
    supported_targets,
)
from .source_compile import compile_locked_sources
from .source_fetch import fetch_sources_for_entry
from .wheel_builder import build_payload_wheel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py2bin",
        description="Analyze and bundle Python applications using only the standard library.",
    )
    parser.add_argument("--version", action="version", version=f"py2bin {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    def shared(command: argparse.ArgumentParser) -> None:
        command.add_argument("entry", type=Path, help="application entry .py file")
        command.add_argument("--source-root", type=Path, help="project root copied into the bundle")
        command.add_argument("--include", action="append", default=[], metavar="MODULE")
        command.add_argument("--exclude", action="append", default=[], metavar="MODULE")
        command.add_argument(
            "--dependency-mode",
            choices=("none", "imported", "closure"),
            default="closure",
            help="none, directly imported distributions, or recursive installed dependencies",
        )

    def target_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--target", choices=supported_targets())
        command.add_argument("--os", dest="target_os", help="linux, windows, or macos")
        command.add_argument("--arch", help="x86_64/x64 or arm64/aarch64")

    def wheel_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--wheel", action="append", default=[], type=Path)
        command.add_argument(
            "--wheel-dir",
            action="append",
            default=[],
            type=Path,
            help="directory containing a complete set of target-compatible .whl files",
        )

    def onefile_options(command: argparse.ArgumentParser) -> None:
        layout = command.add_mutually_exclusive_group()
        layout.add_argument(
            "--onefile",
            dest="onefile",
            action="store_true",
            default=True,
            help="embed the runtime bundle in one self-extracting executable (default)",
        )
        layout.add_argument(
            "--onedir",
            dest="onefile",
            action="store_false",
            help="keep the runtime and application as an unpacked directory",
        )

    def kernel_options(command: argparse.ArgumentParser) -> None:
        # The former ``--experimental-kernels`` flag reimplemented a static
        # integer NumPy/Torch subset from scratch. That produced binaries whose
        # observable result differed from CPython (a numpy/torch reduction is an
        # np.int64 / 0-d tensor, not a plain int), so it was removed to keep the
        # compiler's absolute-honesty guarantee. The attribute is retained as a
        # constant ``False`` so every subcommand keeps a stable namespace.
        command.set_defaults(experimental_kernels=False)

    analyze_parser = commands.add_parser("analyze", help="show imports and package requirements")
    shared(analyze_parser)
    build_parser = commands.add_parser("build", help="build a runnable artifact")
    shared(build_parser)
    build_parser.add_argument("--output", "-o", required=True, type=Path)
    build_parser.add_argument("--format", choices=[kind.value for kind in ArtifactKind], default="bin")
    build_parser.add_argument("--name")
    build_parser.add_argument("--icon", type=Path, help="app icon (.ico, .png, or .icns)")
    build_parser.add_argument("--python", default="/usr/bin/env python3", help="runtime command in launcher/shebang")
    build_parser.add_argument("--clean", action="store_true")
    native_parser = commands.add_parser(
        "compile", help="write a native executable directly (no assembler, linker, or Python at runtime)"
    )
    native_parser.add_argument("entry", type=Path)
    native_parser.add_argument("--output", "-o", required=True, type=Path)
    target_options(native_parser)
    native_parser.add_argument("--app", action="store_true", help="create a native macOS .app bundle")
    native_parser.add_argument(
        "--source-root",
        type=Path,
        help="application root used to distinguish local and locked imports",
    )
    native_parser.add_argument(
        "--strict-library-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "require every top-level function below this pure-Python library "
            "root to lower through the native backend"
        ),
    )
    native_parser.add_argument(
        "--source-lock",
        type=Path,
        help="pinned JSON source lock; imported sources are fetched automatically",
    )
    native_parser.add_argument(
        "--source-cache",
        type=Path,
        help="required cache/output root when --source-lock is used",
    )
    native_parser.add_argument("--clean", action="store_true")
    kernel_options(native_parser)
    all_parser = commands.add_parser(
        "compile-all",
        help="cross-compile every native target with no external toolchain",
    )
    all_parser.add_argument("entry", type=Path)
    all_parser.add_argument("--output-dir", "-o", required=True, type=Path)
    all_parser.add_argument("--mac-app", action="store_true")
    all_parser.add_argument(
        "--source-root",
        type=Path,
        help="project root containing pure-Python modules eligible for native inlining",
    )
    all_parser.add_argument(
        "--strict-library-root",
        action="append",
        default=[],
        type=Path,
        help="reject the build if any function in this library root needs CPython",
    )
    all_parser.add_argument("--os", dest="target_os", help="linux, windows, or macos")
    all_parser.add_argument("--arch", help="x86_64/x64 or arm64/aarch64")
    all_parser.add_argument("--clean", action="store_true")
    kernel_options(all_parser)
    c_parser = commands.add_parser(
        "emit-c", help="translate the supported Python subset to portable C source"
    )
    c_parser.add_argument("entry", type=Path)
    c_parser.add_argument("--output", "-o", required=True, type=Path)
    c_parser.add_argument("--container", action="store_true", help="write a checksummed .py2cbin")
    c_parser.add_argument("--clean", action="store_true")
    capi_parser = commands.add_parser(
        "compile-capi",
        help=(
            "translate Python into C that drives the CPython C API, then "
            "compile that C to machine code with py2bin's own compiler"
        ),
    )
    capi_parser.add_argument("entry", type=Path)
    capi_parser.add_argument("--output", "-o", required=True, type=Path)
    capi_parser.add_argument(
        "--emit-c", type=Path, help="also write the generated C to this path"
    )
    capi_parser.add_argument("--target", choices=supported_targets())
    capi_parser.add_argument("--os", dest="target_os")
    capi_parser.add_argument("--arch")
    capi_parser.add_argument(
        "--app", action="store_true", help="wrap the binary in a macOS .app"
    )
    capi_parser.add_argument(
        "--runtime",
        metavar="DIR",
        help="where to find an interpreter to carry: an embeddable CPython for a Windows executable, or a Python.framework for a macOS bundle "
        "(for Windows, omit it and pass --auto-fetch to download one)",
    )
    capi_parser.add_argument(
        "--include",
        metavar="PATH",
        action="append",
        default=[],
        help="carry this file or directory beside the program - templates, "
        "web assets, anything it opens rather than imports; repeatable",
    )
    capi_parser.add_argument(
        "--auto-fetch",
        action="store_true",
        help="download whatever is missing over verified HTTPS instead of "
        "asking for a path: the target's interpreter, and any --fetch-package",
    )
    capi_parser.add_argument(
        "--fetch-package",
        metavar="NAME",
        action="append",
        default=[],
        help="download this PyPI project's wheel for the target and carry it "
        "in the bundle; repeatable (implies --auto-fetch)",
    )
    capi_parser.add_argument(
        "--fetch-cache",
        type=Path,
        help="directory for verified downloads (default: ~/.cache/py2bin/fetch)",
    )
    capi_parser.add_argument(
        "--dmg",
        action="store_true",
        help="also write a mountable .dmg beside the .app (macOS targets)",
    )
    capi_parser.add_argument("--name", help="application display name")
    capi_parser.add_argument(
        "--embed-python",
        action="store_true",
        help=(
            "carry the interpreter inside the .app and load it by a path "
            "relative to the executable, so the bundle starts on a Mac that "
            "does not have this exact CPython installed"
        ),
    )
    capi_parser.add_argument(
        "--crash-log",
        action="store_true",
        help=(
            "write the traceback to crash.txt as well as printing it. A "
            "windowed application has no console to print to, so without this "
            "an uncaught exception makes it vanish with nothing to read; the "
            "file goes beside the executable, or in the user's home when that "
            "is not writable"
        ),
    )
    capi_parser.add_argument(
        "--exclude",
        action="append",
        metavar="MODULE",
        help=(
            "drop this module from the bundle even though the walk kept its "
            "package, and with it anything only it referred to. Repeatable. "
            "For an optional codec, name both halves - PIL.AvifImagePlugin "
            "and PIL._avif - since the extension is what the library hangs "
            "off. What the program can then no longer do is the caller's to "
            "judge"
        ),
    )
    capi_parser.add_argument(
        "--zip-stdlib",
        nargs="?",
        const="compress",
        choices=["store", "compress"],
        help=(
            "pack the carried standard library into the pythonXY.zip the "
            "interpreter already looks for. 'compress' (the default) took a "
            "real application's library from 8.4 MB to 3.5 MB with no "
            "measurable effect on startup; 'store' is for a filesystem that "
            "compresses already"
        ),
    )
    capi_parser.add_argument(
        "--prune-unused",
        action="store_true",
        help=(
            "drop bundled modules the program cannot import. The walk is "
            "static, so a module imported from a name built at run time would "
            "go with them"
        ),
    )
    capi_parser.add_argument(
        "--bundle-site",
        action="append",
        default=[],
        metavar="DIR",
        type=Path,
        help=(
            "copy DIR into the .app as Contents/Resources/site-packages, "
            "leaving out what does not run: tests, build metadata, pip"
        ),
    )
    capi_parser.add_argument(
        "--site",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "put DIR on sys.path before the program runs; repeatable. The "
            "interpreter a compiled binary links is the build machine's, and "
            "its search path does not know where this program's dependencies "
            "were installed"
        ),
    )
    capi_parser.add_argument("--icon", type=Path, help="app icon (.icns, .ico, .png)")
    capi_parser.add_argument("--clean", action="store_true")
    via_c_parser = commands.add_parser(
        "compile-via-c",
        help=(
            "generate canonical C, parse it with py2bin, and write native "
            "machine code without an external C compiler"
        ),
    )
    via_c_parser.add_argument("entry", type=Path)
    via_c_parser.add_argument("--output", "-o", required=True, type=Path)
    via_c_parser.add_argument(
        "--c-output",
        type=Path,
        help="also retain the exact canonical C source parsed by py2bin",
    )
    target_options(via_c_parser)
    via_c_parser.add_argument("--clean", action="store_true")
    cc_parser = commands.add_parser(
        "cc",
        help="compile a C file to a native executable (the easy front door)",
        description=(
            "Compile C to machine code. With no --output the executable takes "
            "the source file's name, and with no --target it is built for this "
            "machine, so `py2bin cc hello.c` writes ./hello. No assembler, "
            "linker, or C toolchain is used."
        ),
    )
    cc_parser.add_argument("entry", type=Path, help=".c source file")
    cc_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="executable to write (default: the source name without .c)",
    )
    cc_parser.add_argument(
        "--include-dir", "-I", action="append", default=[], metavar="DIR",
        help="a directory the preprocessor searches for #include",
    )
    cc_parser.add_argument(
        "--define", "-D", action="append", default=[], metavar="NAME[=VALUE]",
        help="define a macro before the file is preprocessed",
    )
    target_options(cc_parser)
    cc_parser.add_argument(
        "--keep",
        action="store_true",
        help="fail instead of overwriting an existing output file",
    )
    compile_c_parser = commands.add_parser(
        "compile-c",
        help="compile C to machine code with py2bin's own C compiler",
    )
    compile_c_parser.add_argument("entry", type=Path, help=".c source file")
    compile_c_parser.add_argument("--output", "-o", required=True, type=Path)
    compile_c_parser.add_argument(
        "--include-dir",
        "-I",
        action="append",
        default=[],
        metavar="DIR",
        help="a directory the preprocessor searches for #include",
    )
    compile_c_parser.add_argument(
        "--define",
        "-D",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help="define a macro before the file is preprocessed",
    )
    target_options(compile_c_parser)
    compile_c_parser.add_argument("--clean", action="store_true")
    c_plan_parser = commands.add_parser("plan-c", help="choose C or CPython bundle without running code")
    c_plan_parser.add_argument("entry", type=Path)
    freeze_parser = commands.add_parser(
        "freeze",
        aliases=["bundle"],
        help="bundle CPython and complete libraries so target Python is not required",
    )
    shared(freeze_parser)
    freeze_parser.add_argument("--output", "-o", required=True, type=Path)
    wheel_options(freeze_parser)
    target_options(freeze_parser)
    freeze_parser.add_argument(
        "--runtime-pack",
        type=Path,
        help="target CPython runtime pack directory or ZIP; target wheels are then required",
    )
    freeze_parser.add_argument(
        "--auto-fetch",
        action="store_true",
        help="download the missing target CPython runtime and target wheels "
        "over verified HTTPS (SHA-256 checked, no pip)",
    )
    freeze_parser.add_argument(
        "--fetch-cache",
        type=Path,
        help="directory for verified downloads (default: ~/.cache/py2bin/fetch)",
    )
    freeze_parser.add_argument(
        "--fetch-lock",
        type=Path,
        help="record/verify the SHA-256 of every fetched file for reproducible builds",
    )
    freeze_parser.add_argument(
        "--fetch-python",
        help="CPython version to fetch for the target, such as 3.12.9",
    )
    freeze_parser.add_argument(
        "--fetch-map",
        action="append",
        default=[],
        metavar="IMPORT=PROJECT",
        help="map an import name to its PyPI project (for example "
        "webview=pywebview); py2bin never guesses this, because import names "
        "and project names are different namespaces",
    )
    freeze_parser.add_argument(
        "--app",
        action="store_true",
        help="create a windowed Windows executable or frozen macOS .app",
    )
    freeze_parser.add_argument("--name", help="application display and executable name")
    freeze_parser.add_argument("--icon", type=Path, help="app icon (.ico, .png, or .icns)")
    freeze_parser.add_argument(
        "--compact",
        "--optimize-size",
        action="store_true",
        help="omit tests, caches, and build/debug support from packages and runtime packs",
    )
    onefile_options(freeze_parser)
    freeze_parser.add_argument("--clean", action="store_true")
    assemble_parser = commands.add_parser(
        "assemble",
        help="automatically choose native compilation or a compatible frozen runtime",
    )
    shared(assemble_parser)
    assemble_parser.add_argument("--output", "-o", required=True, type=Path)
    assemble_parser.add_argument(
        "--mode",
        choices=("auto", "native", "compatible"),
        default="auto",
        help=(
            "auto falls back to CPython, native rejects any unsupported source, "
            "compatible always bundles CPython"
        ),
    )
    wheel_options(assemble_parser)
    target_options(assemble_parser)
    assemble_parser.add_argument("--runtime-pack", type=Path)
    assemble_parser.add_argument("--app", action="store_true")
    assemble_parser.add_argument("--name")
    assemble_parser.add_argument("--icon", type=Path)
    assemble_parser.add_argument(
        "--compact",
        "--optimize-size",
        action="store_true",
        help="omit tests, caches, and build/debug support from compatible bundles",
    )
    onefile_options(assemble_parser)
    assemble_parser.add_argument("--clean", action="store_true")
    kernel_options(assemble_parser)
    runtime_parser = commands.add_parser(
        "runtime-pack",
        aliases=["pack-runtime"],
        help="snapshot the current CPython runtime for compatible cross-target assembly",
    )
    runtime_parser.add_argument("--output", "-o", required=True, type=Path)
    runtime_parser.add_argument(
        "--compact",
        "--optimize-size",
        action="store_true",
    )
    runtime_parser.add_argument("--clean", action="store_true")
    capability_parser = commands.add_parser(
        "capabilities",
        help="report what is native machine code and what still requires CPython",
    )
    capability_parser.add_argument(
        "entry",
        nargs="?",
        type=Path,
        help="optional application entry .py file to inspect without executing it",
    )
    capability_parser.add_argument("--json", action="store_true")
    capability_parser.add_argument(
        "--strict",
        action="store_true",
        help="return status 1 when the entry is outside the native subset",
    )
    kernel_options(capability_parser)
    aot_plan_parser = commands.add_parser(
        "aot-plan",
        help=(
            "prove whether a complete application can be built without "
            "CPython, source/bytecode payloads, extraction, or fallback"
        ),
    )
    aot_plan_parser.add_argument("entry", type=Path)
    aot_plan_parser.add_argument("--source-root", type=Path)
    aot_plan_parser.add_argument("--json", action="store_true")
    aot_plan_parser.add_argument(
        "--strict",
        action="store_true",
        help="return status 1 when any operation or import lacks a native route",
    )
    kernel_options(aot_plan_parser)
    aot_build_parser = commands.add_parser(
        "aot-build",
        help=(
            "write a CPython-free native artifact or fail; never freeze, "
            "extract, or select another backend"
        ),
    )
    aot_build_parser.add_argument("entry", type=Path)
    aot_build_parser.add_argument("--output", "-o", required=True, type=Path)
    aot_build_parser.add_argument("--source-root", type=Path)
    target_options(aot_build_parser)
    aot_build_parser.add_argument(
        "--app",
        action="store_true",
        help="create a direct-native macOS .app for a supported entry",
    )
    aot_build_parser.add_argument(
        "--attestation",
        type=Path,
        help="write a JSON proof record after artifact verification",
    )
    aot_build_parser.add_argument(
        "--via-c",
        action="store_true",
        help=(
            "lower the complete supported app/library graph to native IR, "
            "emit canonical C, reparse it with py2bin, then write machine code"
        ),
    )
    aot_build_parser.add_argument(
        "--c-output",
        type=Path,
        help="retain the exact whole-program canonical C used by --via-c",
    )
    aot_build_parser.add_argument("--clean", action="store_true")
    kernel_options(aot_build_parser)
    library_audit_parser = commands.add_parser(
        "audit-library",
        help="validate every function in a pure-Python source tree against native AOT",
    )
    library_audit_parser.add_argument("root", type=Path)
    library_audit_parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        type=Path,
        help="additional local import root used while validating functions",
    )
    library_audit_parser.add_argument("--json", action="store_true")
    library_audit_parser.add_argument(
        "--strict",
        action="store_true",
        help="return status 1 when any Python function or module needs runtime semantics",
    )
    kernel_options(library_audit_parser)
    wheel_parser = commands.add_parser(
        "wheel",
        help="create a wheel from Python files or already-built native extensions",
    )
    wheel_parser.add_argument("source", type=Path, help="tree stored at the wheel root")
    wheel_parser.add_argument("--output-dir", "-o", required=True, type=Path)
    wheel_parser.add_argument("--name", required=True)
    wheel_parser.add_argument("--version", required=True)
    wheel_parser.add_argument("--python-tag", default="py3")
    wheel_parser.add_argument("--abi-tag", default="none")
    wheel_parser.add_argument("--platform-tag", default="any")
    wheel_parser.add_argument("--requires", action="append", default=[])
    wheel_parser.add_argument("--clean", action="store_true")
    fetch_parser = commands.add_parser(
        "fetch-sources",
        help="fetch statically imported sources from a pinned SHA-256 source lock",
    )
    fetch_parser.add_argument("entry", type=Path)
    fetch_parser.add_argument("--source-root", type=Path)
    fetch_parser.add_argument("--source-lock", required=True, type=Path)
    fetch_parser.add_argument("--source-cache", required=True, type=Path)
    fetch_parser.add_argument("--json", action="store_true")
    source_compile_parser = commands.add_parser(
        "compile-source",
        help="fetch pinned sources and attempt only real handwritten native compilation",
    )
    source_compile_parser.add_argument("entry", type=Path)
    source_compile_parser.add_argument("--source-root", type=Path)
    source_compile_parser.add_argument("--source-lock", required=True, type=Path)
    source_compile_parser.add_argument("--source-cache", required=True, type=Path)
    source_compile_parser.add_argument("--output", "-o", required=True, type=Path)
    target_options(source_compile_parser)
    source_compile_parser.add_argument("--app", action="store_true")
    source_compile_parser.add_argument("--clean", action="store_true")
    kernel_options(source_compile_parser)
    commands.add_parser("targets", help="list native code-generation targets")
    return parser


def _analysis_dict(result) -> dict[str, object]:
    return {
        "modules": sorted(result.modules),
        "distributions": sorted(result.distributions, key=str.lower),
        "unresolved": sorted(result.unresolved),
        "local_files": sorted(str(path) for path in result.local_files),
        "notes": result.hook_notes,
    }


def _target_from_args(args) -> str | None:
    if args.target and (args.target_os or args.arch):
        raise ValueError("--target cannot be combined with --os or --arch")
    if (args.target_os is None) != (args.arch is None):
        raise ValueError("--os and --arch must be supplied together")
    return (
        resolve_target(args.target_os, args.arch)
        if args.target_os is not None and args.arch is not None
        else args.target
    )


def _fetch_map_from_args(args) -> dict[str, str]:
    """Parse repeated --fetch-map IMPORT=PROJECT options."""

    mapping: dict[str, str] = {}
    for item in getattr(args, "fetch_map", None) or ():
        module, separator, project = item.partition("=")
        if not separator or not module.strip() or not project.strip():
            raise ValueError(
                f"--fetch-map expects IMPORT=PROJECT, received {item!r}"
            )
        mapping[module.strip()] = project.strip()
    return mapping


def _wheels_from_args(args) -> tuple[Path, ...]:
    wheels = [path.expanduser().resolve() for path in args.wheel]
    for directory in args.wheel_dir:
        directory = directory.expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"wheel directory does not exist: {directory}")
        wheels.extend(sorted(directory.glob("*.whl")))
    unique: dict[Path, None] = {}
    for wheel in wheels:
        unique[wheel] = None
    return tuple(unique)


def _site_paths(values: list[str]) -> tuple[str, ...]:
    """What `--site` hands the emitter.

    A relative one stays relative: it is resolved against the *running* binary,
    which is what lets a bundle carry its own packages. Resolving it here
    against the build directory produced an absolute path to somewhere that
    never existed, and the bundle failed to import what was sitting inside it.
    """

    return tuple(
        str(Path(value).expanduser().resolve())
        if value.startswith("~") or Path(value).is_absolute()
        else value
        for value in values
    )


def _embedded_python_path() -> str:
    """Where the carried interpreter sits, relative to the executable."""

    import sysconfig

    from .cabi_tables import _cpython_library

    dylib = Path(_cpython_library())
    if dylib.is_file() and dylib.parent.parent.name == "Versions":
        # A real framework on this machine: use its own layout, which is what
        # everything else about this interpreter already agrees with.
        return (
            f"@executable_path/../Frameworks/Python.framework/Versions/"
            f"{dylib.parent.name}/{dylib.name}"
        )
    # No framework here, or one whose path says nothing useful - a portable
    # build, or an interpreter reporting where it was built rather than where
    # it is. The carried one is laid out canonically instead, and the step
    # that carries it puts it exactly here.
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        f"@executable_path/../Frameworks/Python.framework/Versions/"
        f"{version}/Python"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "targets":
        print("\n".join(supported_targets()))
        return 0
    if args.command == "capabilities":
        try:
            if args.entry is None:
                if args.json:
                    print(
                        json.dumps(
                            [dataclasses.asdict(item) for item in common_libraries()],
                            indent=2,
                        )
                    )
                else:
                    print(format_catalog())
                return 0
            report = assess_entry(
                args.entry,
                experimental_kernels=args.experimental_kernels,
            )
        except (FileNotFoundError, SyntaxError, UnicodeDecodeError, OSError) as error:
            print(f"py2bin: error: {error}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
        else:
            print(f"entry: {report.entry}")
            print(f"native compile: {'yes' if report.native_compile else 'no'}")
            print(f"reason: {report.native_reason}")
            if report.libraries:
                print()
                print(format_catalog(report.libraries))
                for item in report.libraries:
                    print(f"- {item.module}: {item.requirement}")
        return 1 if args.strict and not report.native_compile else 0
    if args.command == "aot-plan":
        try:
            plan = plan_aot_application(
                args.entry,
                source_root=args.source_root,
                experimental_kernels=args.experimental_kernels,
            )
        except (
            FileNotFoundError,
            NotADirectoryError,
            SyntaxError,
            UnicodeDecodeError,
            ValueError,
            OSError,
        ) as error:
            print(f"py2bin: error: {error}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(plan.as_dict(), indent=2))
        else:
            print(f"entry: {plan.entry}")
            print(f"strict native AOT: {'yes' if plan.buildable else 'no'}")
            print(f"reason: {plan.compiler_reason}")
            print(f"reachable Python inputs: {len(plan.reachable_python)}")
            print(f"web assets kept outside machine code: {len(plan.web_assets)}")
            print(f"prebuilt native payloads requiring adapters: {len(plan.native_payloads)}")
            for blocker in plan.blockers:
                print(f"- blocker: {blocker}")
        return 1 if args.strict and not plan.buildable else 0
    if args.command == "aot-build":
        try:
            result = build_aot_application(
                args.entry,
                args.output,
                target=_target_from_args(args),
                clean=args.clean,
                app=args.app,
                source_root=args.source_root,
                experimental_kernels=args.experimental_kernels,
                attestation=args.attestation,
                via_c=args.via_c,
                c_output=args.c_output,
            )
        except (
            AOTPlanError,
            FileNotFoundError,
            FileExistsError,
            NativeCompileError,
            SyntaxError,
            ValueError,
            OSError,
        ) as error:
            print(f"py2bin: error: {error}", file=sys.stderr)
            return 2
        proof = result.attestation
        print(
            f"built strict CPython-free {proof.target} machine-code artifact "
            f"{proof.artifact} ({proof.bytes} bytes, {proof.operations} IR operations)"
        )
        print(f"sha256: {proof.sha256}")
        print(f"pipeline: {proof.pipeline}")
        if result.c_artifact is not None:
            print(f"retained reparsed whole-program C: {result.c_artifact}")
        if result.attestation_path is not None:
            print(f"attestation: {result.attestation_path}")
        return 0
    if args.command == "audit-library":
        try:
            report = audit_native_library(
                args.root,
                source_roots=tuple(args.source_root),
                experimental_kernels=args.experimental_kernels,
            )
        except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as error:
            print(f"py2bin: error: {error}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
        else:
            print(f"library: {report.root}")
            print(f"fully native: {'yes' if report.fully_native else 'no'}")
            print(
                f"Python files: {report.python_files}; functions: "
                f"{report.native_functions}/{len(report.functions)} native"
            )
            print(
                f"prebuilt native payloads: {len(report.native_payloads)}; "
                f"web assets: {len(report.web_assets)}"
            )
            for item in report.functions:
                state = "native" if item.native else "blocked"
                print(f"- {item.module}.{item.name}: {state} — {item.reason}")
            for blocker in report.module_blockers:
                print(f"- module blocker: {blocker}")
            for path in report.native_payloads:
                print(
                    f"- ABI blocker: {path} is already machine code, but its "
                    "CPython/C ABI adapter has not been proven replaceable"
                )
        return 1 if args.strict and not report.fully_native else 0
    if args.command == "wheel":
        try:
            result = build_payload_wheel(
                args.source,
                args.output_dir,
                name=args.name,
                version=args.version,
                python_tag=args.python_tag,
                abi_tag=args.abi_tag,
                platform_tag=args.platform_tag,
                requirements=tuple(args.requires),
                clean=args.clean,
            )
        except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
            print(f"py2bin: error: {error}", file=sys.stderr)
            return 2
        print(
            f"built wheel {result.wheel} ({result.files} files, "
            f"{result.bytes} bytes, tag {result.tag})"
        )
        if result.native_files:
            print(
                "prebuilt native files preserved: "
                + ", ".join(result.native_files)
            )
        if result.cython_sources:
            print(
                "note: Cython source files were packaged but not compiled: "
                + ", ".join(result.cython_sources)
            )
        return 0
    if args.command == "fetch-sources":
        entry = args.entry.expanduser().resolve()
        source_root = (args.source_root or entry.parent).expanduser().resolve()
        try:
            result = fetch_sources_for_entry(
                entry,
                source_root,
                args.source_lock,
                args.source_cache,
            )
        except (FileNotFoundError, SyntaxError, ValueError, OSError) as error:
            print(f"py2bin: error: {error}", file=sys.stderr)
            return 2
        payload = {
            "imports": list(result.imports),
            "roots": [str(root) for root in result.roots],
            "fetched": [
                {
                    "module": item.module,
                    "root": str(item.root),
                    "revision": item.revision,
                    "sha256": item.sha256,
                    "origin": item.origin,
                }
                for item in result.fetched
            ],
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for item in result.fetched:
                print(
                    f"{item.module}: {item.root} "
                    f"(revision {item.revision}, sha256 {item.sha256})"
                )
            if not result.fetched:
                print("no external static imports required locked sources")
        return 0
    if args.command == "compile-source":
        try:
            result = compile_locked_sources(
                args.entry,
                args.output,
                source_lock=args.source_lock,
                source_cache=args.source_cache,
                source_root=args.source_root,
                target=_target_from_args(args),
                clean=args.clean,
                app=args.app,
                experimental_kernels=args.experimental_kernels,
            )
        except (
            FileNotFoundError,
            FileExistsError,
            NativeCompileError,
            SyntaxError,
            ValueError,
            OSError,
        ) as error:
            print(f"py2bin: error: {error}", file=sys.stderr)
            return 2
        for item in result.fetched:
            print(
                f"fetched {item.module} revision {item.revision} "
                f"with sha256 {item.sha256}"
            )
        print(
            f"compiled {result.native.artifact} for {result.native.target} "
            f"({result.native.bytes} bytes, {result.native.operations} IR operations)"
        )
        return 0
    if args.command in {"runtime-pack", "pack-runtime"}:
        try:
            result = create_runtime_pack(
                args.output, compact=args.compact, clean=args.clean
            )
        except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
            print(f"py2bin: error: {error}", file=sys.stderr)
            return 2
        print(
            f"packed {result.target} CPython {result.python} runtime at "
            f"{result.pack} ({result.files} files, {result.bytes} bytes)"
        )
        return 0
    entry = args.entry.expanduser().resolve()
    try:
        if args.command == "compile-via-c":
            bridge = compile_python_via_c(
                entry,
                args.output,
                target=_target_from_args(args),
                clean=args.clean,
                c_output=args.c_output,
            )
            result = bridge.native
            print(
                f"compiled Python -> canonical C -> {result.target} machine code "
                f"at {result.artifact} ({result.bytes} bytes, "
                f"{result.operations} IR operations)"
            )
            if bridge.c_artifact is not None:
                print(f"retained parsed C source at {bridge.c_artifact}")
            return 0
        if args.command == "cc":
            # Defaults chosen so the common case needs no flags at all:
            # `py2bin cc hello.c` builds ./hello for this machine.
            output = args.output
            if output is None:
                output = entry.with_suffix("")
                if output == entry:
                    output = entry.with_name(entry.name + ".bin")
            target = _target_from_args(args)
            result = compile_c_native(
                entry,
                output,
                target=target,
                clean=not args.keep,
                include_dirs=tuple(args.include_dir),
                defines=tuple(args.define),
            )
            print(
                f"{result.artifact} ({result.bytes} bytes, {result.target})",
            )
            return 0
        if args.command == "compile-c":
            result = compile_c_native(
                entry,
                args.output,
                target=_target_from_args(args),
                clean=args.clean,
                include_dirs=tuple(args.include_dir),
                defines=tuple(args.define),
            )
            print(
                f"compiled C to {result.target} machine code at "
                f"{result.artifact} ({result.bytes} bytes, "
                f"{result.operations} IR operations)"
            )
            return 0
        if args.command == "compile-all":
            if (args.target_os is None) != (args.arch is None):
                raise ValueError("--os and --arch must be supplied together")
            source_root = (args.source_root or entry.parent).expanduser().resolve()
            for library_root in args.strict_library_root:
                require_native_library(
                    library_root,
                    source_roots=(source_root,),
                    experimental_kernels=args.experimental_kernels,
                )
            results = compile_all(
                entry,
                args.output_dir,
                args.clean,
                args.mac_app,
                args.target_os,
                args.arch,
                (
                    source_root,
                ),
                args.experimental_kernels,
            )
            for result in results:
                print(f"{result.target}: {result.artifact} ({result.bytes} bytes)")
            return 0
        if args.command == "compile-capi":
            from .capi_emit import python_program_to_capi_c

            target = _target_from_args(args)
            generated, linked = python_program_to_capi_c(
                entry,
                crash_log=args.crash_log,
                extra_paths=_site_paths(args.site),
            )
            if linked:
                print(
                    f"linking {len(linked)} module(s) of the program itself: "
                    + ", ".join(linked),
                    file=sys.stderr,
                )
            source = args.emit_c or args.output.with_suffix(".capi.c")
            source.parent.mkdir(parents=True, exist_ok=True)
            # newline pinned: the generated C has to be byte-identical
            # whatever host wrote it.
            source.write_text(generated, encoding="utf-8", newline="\n")
            output = args.output
            if args.app and output.suffix != ".app":
                output = output.with_suffix(".app")
            if args.embed_python and not args.app:
                # `main` does not hold the parser, and reaching for one that
                # is not there turned a refusal a user should be able to act
                # on into a NameError and a traceback.
                print(
                    "py2bin: error: --embed-python needs --app: "
                    "it fills the bundle",
                    file=sys.stderr,
                )
                return 2
            artifact = compile_c_native(
                source,
                output,
                target=target,
                clean=args.clean,
                app=args.app,
                app_name=args.name,
                icon=args.icon,
                python_dylib=(
                    _embedded_python_path() if args.embed_python else None
                ),
            )
            # _target_from_args returns None when the caller did not name a
            # target and the host's own is meant. Asking that None whether it
            # starts with "windows-" is how every --app build without an
            # explicit --target came to abort before it was ever sealed.
            chosen = target or host_target()
            fetched_runtime = None
            fetched_sites: list[Path] = []
            if args.auto_fetch or args.fetch_package:
                from .freezer import default_fetch_cache
                from .runtime_fetch import (
                    FetchError,
                    extract_zip,
                    fetch_macos_runtime,
                    fetch_wheel,
                    fetch_windows_runtime,
                )

                cache = args.fetch_cache or default_fetch_cache()
                # The wheel tags want major.minor; the runtime archive is
                # published per patch release, so it needs all three.
                version = f"{sys.version_info.major}.{sys.version_info.minor}"
                full_version = f"{version}.{sys.version_info.micro}"
                room = output.parent / ".py2bin-fetched"
                # Downloaded rather than asked for. Everything here is checked
                # against a hash the index published before it is used, and
                # kept in the cache so a second build does not go out again.
                if chosen.startswith("windows-") and not args.runtime:
                    pack = room / "runtime"
                    fetch_windows_runtime(
                        full_version, chosen, pack, cache=cache, clean=True
                    )
                    # The pack is unpacked into a directory of its own
                    # inside this one, so the interpreter is looked for
                    # rather than assumed to be at the top.
                    found = sorted(pack.rglob("python*.dll"))
                    fetched_runtime = found[0].parent if found else pack
                    print(
                        f"fetched an embeddable CPython {full_version} "
                        f"for {chosen}",
                        file=sys.stderr,
                    )
                if (
                    chosen.startswith("darwin-")
                    and args.app
                    and args.embed_python
                    and not args.runtime
                ):
                    # Only when this machine cannot supply one. A Mac has a
                    # framework already and its own matches everything else
                    # about it; anywhere else there is none, which is what
                    # stopped a macOS bundle being built off a Mac at all.
                    from .cabi_tables import _cpython_library

                    if not Path(_cpython_library()).is_file():
                        fetched_runtime = fetch_macos_runtime(
                            version, chosen, room / "macos-runtime", cache=cache
                        )
                        print(
                            f"fetched a portable CPython for {chosen} - this "
                            f"machine has none to carry",
                            file=sys.stderr,
                        )

                wanted = list(args.fetch_package)
                if args.auto_fetch:
                    from .requirements import discover

                    needs = discover(entry)
                    for project in needs.projects:
                        if project not in wanted:
                            wanted.append(project)
                    if needs.projects:
                        print(
                            "the program needs " + ", ".join(needs.projects),
                            file=sys.stderr,
                        )
                    if needs.unknown:
                        # Named rather than guessed at: an import name is not
                        # a project name, and a guess that lands on an
                        # unrelated project would put a stranger's code in
                        # someone's application.
                        print(
                            "py2bin: cannot tell which project publishes "
                            + ", ".join(needs.unknown)
                            + "; name it with --fetch-package",
                            file=sys.stderr,
                        )
                from .requirements import required_by

                missing = []
                seen: set[str] = set()
                queue = list(wanted)
                while queue:
                    project = queue.pop(0)
                    key = project.lower().replace("_", "-")
                    if key in seen:
                        continue
                    seen.add(key)
                    into = room / "site"
                    try:
                        got = fetch_wheel(
                            project, chosen, version, room / "wheels", cache=cache
                        )
                    except FetchError as reason:
                        if "only a source distribution" in str(reason):
                            # Nothing to build is not the same as unbuildable.
                            try:
                                from .runtime_fetch import fetch_pure_sdist

                                names = fetch_pure_sdist(
                                    project, into, cache=cache
                                )
                                print(
                                    f"took {', '.join(names)} from the source "
                                    f"distribution for {project} - it has "
                                    f"nothing to compile",
                                    file=sys.stderr,
                                )
                                if into not in fetched_sites:
                                    fetched_sites.append(into)
                                continue
                            except FetchError as second:
                                reason = second
                        # One package with no wheel for this target is not a
                        # reason to throw away the build. A project publishes
                        # wheels for the interpreters that existed when it was
                        # released, so a new Python leaves some behind for a
                        # while. The program is compiled either way, and only
                        # fails if it actually reaches for the missing one.
                        missing.append((project, str(reason)))
                        continue
                    # A wheel is a zip, and what goes beside a program is
                    # what is inside it - carrying the archive itself would
                    # put a file on the path that nothing can import.
                    extract_zip(got.path, into)
                    print(
                        f"fetched and unpacked {got.path.name}", file=sys.stderr
                    )
                    if into not in fetched_sites:
                        fetched_sites.append(into)
                    # What that wheel stands on. A program imports pywebview
                    # and never mentions proxy_tools; pywebview mentions it,
                    # in the metadata that just arrived.
                    for dependency in required_by(into):
                        if dependency.lower().replace("_", "-") not in seen:
                            queue.append(dependency)
                for project, reason in missing:
                    print(f"py2bin: no wheel for {project}: {reason}", file=sys.stderr)
                if missing:
                    print(
                        "py2bin: the build goes on without "
                        + ", ".join(name for name, _ in missing)
                        + "; supply a wheel with --fetch-package once one exists, "
                        "or --bundle-site a directory holding it",
                        file=sys.stderr,
                    )

            runtime = args.runtime or fetched_runtime
            sites = [Path(entry) for entry in (args.bundle_site or [])]
            sites.extend(fetched_sites)
            if chosen.startswith("windows-") and (sites or runtime):
                from .windows_bundle import carry_packages, carry_runtime

                # There is no bundle here: the executable, the interpreter and
                # the packages share one directory, and what is importable is
                # decided by the interpreter's own path file. Placing packages
                # and naming them on that path is one step, so a caller cannot
                # end up with a directory full of modules the program cannot
                # see - which fails as ModuleNotFoundError for something
                # plainly on disk, and says nothing at all from a windowed
                # executable.
                room = output.parent
                if runtime:
                    carried = carry_runtime(room, Path(runtime))
                    print(
                        f"carried the interpreter beside {output.name} "
                        f"({carried} bytes)",
                        file=sys.stderr,
                    )
                if sites:
                    packed = carry_packages(room, tuple(sites))
                    print(
                        f"carried packages into Lib\\site-packages "
                        f"({packed} bytes), and named it on the interpreter's "
                        f"path",
                        file=sys.stderr,
                    )
            elif sites:
                # A macOS bundle keeps them in Contents/Resources/site-packages,
                # which --site names relative to the executable. Fetched ones
                # belong here too: they were downloaded and then left behind,
                # so a bundle carrying its own interpreter reached for pywebview
                # and found nothing - ModuleNotFoundError from an application
                # whose packages had been fetched minutes earlier.
                from .freezer import bundle_site_packages

                bundle_site_packages(output, tuple(sites))
                print(
                    f"carried {len(sites)} package source(s) into the bundle",
                    file=sys.stderr,
                )
            carried = 0
            embedded = args.embed_python
            if embedded:
                from .freezer import embed_cpython_in_app

                # `chosen`, not `target`: the latter is None when the caller
                # did not name one and the host's is meant, and the
                # architecture tables have no entry for None.
                #
                # No falling back here, unlike a package with no wheel. The
                # executable was linked against
                # @executable_path/../Frameworks/Python.framework before this
                # ran, so a bundle without one is a bundle dyld refuses to
                # start - which is worse than saying so now.
                framework = (
                    Path(args.runtime) if args.runtime else fetched_runtime
                )
                carried = embed_cpython_in_app(output, chosen, framework)
            if embedded:
                freed = 0
                if args.prune_unused:
                    from .freezer import (
                        drop_debug_symbols,
                        drop_unused_libraries,
                        prune_unreachable,
                    )

                    freed = prune_unreachable(
                        output, entry, tuple(args.exclude or ())
                    )
                    # Debug companions are nobody's dependency, so the order
                    # against the other two does not matter.
                    freed += drop_debug_symbols(output)
                    # After the modules, not before: the library closure is
                    # computed from the extensions present, and pruning
                    # removes extensions.
                    freed += drop_unused_libraries(output)
                    print(
                        f"dropped {freed // 1048576} MB the program cannot import",
                        file=sys.stderr,
                    )
                from .freezer import compile_bundle_sources

                compile_bundle_sources(output)
                if args.zip_stdlib:
                    from .freezer import zip_bytecode

                    # Strictly after the sources are compiled: this packs the
                    # bytecode, and until that call there is none to pack. Put
                    # before it, the step archived an empty tree and said it
                    # had saved nothing, which was true and useless.
                    saved = zip_bytecode(
                        output, compress=args.zip_stdlib == "compress"
                    )
                    print(
                        "packed the standard library into one archive"
                        + (
                            f", {saved // 1024} KB smaller"
                            if saved > 0
                            else " (stored, so the same size)"
                        ),
                        file=sys.stderr,
                    )
                print(
                    f"carried the interpreter into the bundle "
                    f"({carried} bytes)",
                    file=sys.stderr,
                )
            if chosen.startswith("windows-") and args.icon:
                from .windows_icon import install_windows_identity

                # A Windows executable carries its icon as a resource inside
                # itself; without one the shell shows a default, which says
                # nothing about the program and looks like nobody finished it.
                install_windows_identity(
                    output,
                    args.name or output.stem,
                    version="1.0.0",
                    icon=Path(args.icon),
                )
                print(f"put {Path(args.icon).name} inside {output.name}", file=sys.stderr)

            if args.include:
                import shutil as _shutil

                # Before the seal, and beside the program rather than in the
                # library: a directory of templates or web assets is opened,
                # not imported, and the program looks for it next to itself.
                beside = (
                    output / "Contents" / "MacOS"
                    if args.app and chosen.startswith("darwin")
                    else output.parent
                )
                beside.mkdir(parents=True, exist_ok=True)
                for named in args.include:
                    item = Path(named).expanduser()
                    if not item.exists():
                        print(
                            f"py2bin: nothing to include at {item}",
                            file=sys.stderr,
                        )
                        continue
                    target = beside / item.name
                    if target.exists():
                        _shutil.rmtree(target) if target.is_dir() else target.unlink()
                    if item.is_dir():
                        _shutil.copytree(item, target)
                    else:
                        _shutil.copy2(item, target)
                    print(f"carried {item.name} beside the program", file=sys.stderr)

            if args.app and chosen.startswith("darwin"):
                from .macos_seal import seal

                # Last, once the interpreter, the packages and the program's
                # own files are all in place. A seal written any earlier
                # describes a bundle that does not exist yet.
                sealed = seal(output)
                print(f"sealed the bundle over {sealed} files", file=sys.stderr)
            staging = output.parent / ".py2bin-fetched"
            if staging.is_dir():
                import shutil as _shutil

                # What was downloaded to build with, not what was built. It is
                # the interpreter and every wheel over again, and leaving it
                # beside the result made a 30 MB bundle look like 500 MB.
                _shutil.rmtree(staging, ignore_errors=True)
            if args.dmg:
                import shutil as _shutil
                import tempfile as _tempfile

                from .dmg import write_compressed_image

                # The image holds the bundle, not the bundle's insides, so
                # it mounts showing one thing to drag out.
                image = output.with_suffix(".dmg")
                with _tempfile.TemporaryDirectory(
                    prefix="py2bin-dmg-", dir=output.parent
                ) as staging:
                    room = Path(staging) / output.name
                    _shutil.copytree(output, room, symlinks=True)
                    # Compressed: the image is what gets handed over, and
                    # a bundle is mostly native code, which deflates well.
                    # What it holds is unchanged - macOS inflates as the
                    # volume is read.
                    size = write_compressed_image(
                        Path(staging), image, args.name or output.stem
                    )
                print(
                    f"wrote {image.name} ({size // 1024 // 1024} MB)",
                    file=sys.stderr,
                )
            print(
                f"compiled {entry} through the CPython C API to "
                f"{artifact.artifact} ({artifact.bytes} bytes, C at {source})"
            )
            return 0
        if args.command == "emit-c":
            result = compile_c_file(entry, args.output, container=args.container, clean=args.clean)
            print(f"wrote {result.kind} {result.artifact} ({result.bytes} bytes)")
            return 0
        if args.command == "plan-c":
            result = plan_c(entry.read_text(encoding="utf-8"), str(entry))
            print(json.dumps(dataclasses.asdict(result), indent=2))
            return 0 if result.backend == "c-source" else 1
        if args.command == "compile":
            target = _target_from_args(args)
            if (args.source_lock is None) != (args.source_cache is None):
                raise ValueError("--source-lock and --source-cache must be supplied together")
            source_root = (
                args.source_root or entry.parent
            ).expanduser().resolve()
            for library_root in args.strict_library_root:
                require_native_library(
                    library_root,
                    source_roots=(source_root,),
                    experimental_kernels=args.experimental_kernels,
                )
            if args.source_lock is not None:
                source_result = compile_locked_sources(
                    entry,
                    args.output,
                    source_lock=args.source_lock,
                    source_cache=args.source_cache,
                    source_root=args.source_root,
                    target=target,
                    clean=args.clean,
                    app=args.app,
                    experimental_kernels=args.experimental_kernels,
                )
                for item in source_result.fetched:
                    print(
                        f"fetched {item.module} revision {item.revision} "
                        f"with sha256 {item.sha256}"
                    )
                result = source_result.native
            else:
                result = compile_native(
                    entry,
                    args.output,
                    target,
                    args.clean,
                    args.app,
                    source_roots=(source_root,),
                    experimental_kernels=args.experimental_kernels,
                )
            print(
                f"compiled {result.artifact} for {result.target} "
                f"({result.bytes} bytes, {result.operations} IR operations)"
            )
            return 0
        source_root = (args.source_root or entry.parent).expanduser().resolve()
        if args.command == "assemble":
            result = assemble(
                entry,
                args.output,
                mode=args.mode,
                target=_target_from_args(args),
                source_root=source_root,
                includes=tuple(args.include),
                excludes=tuple(args.exclude),
                wheels=_wheels_from_args(args),
                dependency_mode=args.dependency_mode,
                runtime_pack=args.runtime_pack,
                app=args.app,
                name=args.name,
                icon=args.icon,
                compact=args.compact,
                clean=args.clean,
                onefile=args.onefile,
                experimental_kernels=args.experimental_kernels,
            )
            print(
                f"assembled {result.artifact} for {result.target} with "
                f"{result.backend} backend ({result.bytes} bytes); run {result.launcher}"
            )
            print(f"reason: {result.reason}")
            return 0
        if args.command in {"freeze", "bundle"}:
            result = freeze(
                entry,
                args.output,
                source_root,
                tuple(args.include),
                tuple(args.exclude),
                _wheels_from_args(args),
                args.dependency_mode,
                args.clean,
                app=args.app,
                name=args.name,
                icon=args.icon,
                compact=args.compact,
                runtime_pack=args.runtime_pack,
                target=_target_from_args(args),
                onefile=args.onefile,
                auto_fetch=args.auto_fetch,
                fetch_cache=args.fetch_cache,
                fetch_lock=args.fetch_lock,
                fetch_python=args.fetch_python,
                fetch_map=_fetch_map_from_args(args),
            )
            print(
                f"froze {result.bundle} for {result.target} with CPython "
                f"{result.python} ({result.files} files, {result.bytes} bytes); "
                f"run {result.launcher}"
            )
            return 0
        if args.command == "analyze":
            result = analyze(
                entry,
                source_root,
                tuple(args.include),
                tuple(args.exclude),
                args.dependency_mode,
            )
            print(json.dumps(_analysis_dict(result), indent=2))
            return 1 if result.unresolved else 0
        result = build(
            BuildConfig(
                entry=entry,
                output=args.output,
                kind=ArtifactKind(args.format),
                name=args.name,
                source_root=source_root,
                includes=tuple(args.include),
                excludes=tuple(args.exclude),
                dependency_mode=args.dependency_mode,
                python=args.python,
                icon=args.icon,
                clean=args.clean,
            )
        )
    except (
        FileNotFoundError,
        FileExistsError,
        NativeCompileError,
        CNativeCompileError,
        CSourceError,
        ValueError,
        OSError,
    ) as error:
        print(f"py2bin: error: {error}", file=sys.stderr)
        return 2
    print(f"built {result.artifact} ({result.files} files, {result.bytes} bytes)")
    if result.analysis.unresolved:
        print("unresolved imports: " + ", ".join(sorted(result.analysis.unresolved)), file=sys.stderr)
    for note in result.analysis.hook_notes:
        print("note: " + note, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
