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
    compile_c_parser = commands.add_parser(
        "compile-c",
        help="compile C to machine code with py2bin's own C compiler",
    )
    compile_c_parser.add_argument("entry", type=Path, help=".c source file")
    compile_c_parser.add_argument("--output", "-o", required=True, type=Path)
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
        if args.command == "compile-c":
            result = compile_c_native(
                entry,
                args.output,
                target=_target_from_args(args),
                clean=args.clean,
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
