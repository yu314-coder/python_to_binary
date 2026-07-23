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
from .csource import CSourceError, compile_c_file, plan_c
from .freezer import create_runtime_pack, freeze
from .model import ArtifactKind, BuildConfig
from .native import (
    NativeCompileError,
    compile_all,
    compile_native,
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
    all_parser = commands.add_parser(
        "compile-all",
        help="cross-compile every native target with no external toolchain",
    )
    all_parser.add_argument("entry", type=Path)
    all_parser.add_argument("--output-dir", "-o", required=True, type=Path)
    all_parser.add_argument("--mac-app", action="store_true")
    all_parser.add_argument("--os", dest="target_os", help="linux, windows, or macos")
    all_parser.add_argument("--arch", help="x86_64/x64 or arm64/aarch64")
    all_parser.add_argument("--clean", action="store_true")
    c_parser = commands.add_parser(
        "emit-c", help="translate the supported Python subset to portable C source"
    )
    c_parser.add_argument("entry", type=Path)
    c_parser.add_argument("--output", "-o", required=True, type=Path)
    c_parser.add_argument("--container", action="store_true", help="write a checksummed .py2cbin")
    c_parser.add_argument("--clean", action="store_true")
    c_plan_parser = commands.add_parser("plan-c", help="choose C or CPython bundle without running code")
    c_plan_parser.add_argument("entry", type=Path)
    freeze_parser = commands.add_parser(
        "freeze",
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
        "--app",
        action="store_true",
        help="create a windowed Windows executable or frozen macOS .app",
    )
    freeze_parser.add_argument("--name", help="application display and executable name")
    freeze_parser.add_argument("--icon", type=Path, help="app icon (.ico, .png, or .icns)")
    freeze_parser.add_argument(
        "--compact",
        action="store_true",
        help="omit package tests, build support, and bytecode caches",
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
    )
    wheel_options(assemble_parser)
    target_options(assemble_parser)
    assemble_parser.add_argument("--runtime-pack", type=Path)
    assemble_parser.add_argument("--app", action="store_true")
    assemble_parser.add_argument("--name")
    assemble_parser.add_argument("--icon", type=Path)
    assemble_parser.add_argument("--compact", action="store_true")
    onefile_options(assemble_parser)
    assemble_parser.add_argument("--clean", action="store_true")
    runtime_parser = commands.add_parser(
        "runtime-pack",
        aliases=["pack-runtime"],
        help="snapshot the current CPython runtime for compatible cross-target assembly",
    )
    runtime_parser.add_argument("--output", "-o", required=True, type=Path)
    runtime_parser.add_argument("--compact", action="store_true")
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
            report = assess_entry(args.entry)
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
        if args.command == "compile-all":
            if (args.target_os is None) != (args.arch is None):
                raise ValueError("--os and --arch must be supplied together")
            results = compile_all(
                entry,
                args.output_dir,
                args.clean,
                args.mac_app,
                args.target_os,
                args.arch,
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
                )
                for item in source_result.fetched:
                    print(
                        f"fetched {item.module} revision {item.revision} "
                        f"with sha256 {item.sha256}"
                    )
                result = source_result.native
            else:
                result = compile_native(entry, args.output, target, args.clean, args.app)
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
            )
            print(
                f"assembled {result.artifact} for {result.target} with "
                f"{result.backend} backend ({result.bytes} bytes); run {result.launcher}"
            )
            print(f"reason: {result.reason}")
            return 0
        if args.command == "freeze":
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
    except (FileNotFoundError, FileExistsError, NativeCompileError, CSourceError, ValueError, OSError) as error:
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
