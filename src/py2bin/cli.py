from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .analyzer import analyze
from .builder import build
from .csource import CSourceError, compile_c_file, plan_c
from .freezer import freeze
from .model import ArtifactKind, BuildConfig
from .native import (
    NativeCompileError,
    compile_all,
    compile_native,
    resolve_target,
    supported_targets,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py2bin",
        description="Analyze and bundle Python applications using only the standard library.",
    )
    parser.add_argument("--version", action="version", version="py2bin 0.1.0")
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
    native_parser.add_argument("--target", choices=supported_targets())
    native_parser.add_argument("--os", dest="target_os", help="linux, windows, or macos")
    native_parser.add_argument("--arch", help="x86_64/x64 or arm64/aarch64")
    native_parser.add_argument("--app", action="store_true", help="create a native macOS .app bundle")
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
    freeze_parser.add_argument("--wheel", action="append", default=[], type=Path)
    freeze_parser.add_argument("--app", action="store_true", help="create a frozen macOS .app")
    freeze_parser.add_argument("--name", help="application display and executable name")
    freeze_parser.add_argument("--icon", type=Path, help="app icon (.ico, .png, or .icns)")
    freeze_parser.add_argument(
        "--compact",
        action="store_true",
        help="omit package tests, build support, and bytecode caches",
    )
    freeze_parser.add_argument("--clean", action="store_true")
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "targets":
        print("\n".join(supported_targets()))
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
            if args.target and (args.target_os or args.arch):
                raise ValueError("--target cannot be combined with --os or --arch")
            if (args.target_os is None) != (args.arch is None):
                raise ValueError("--os and --arch must be supplied together")
            target = (
                resolve_target(args.target_os, args.arch)
                if args.target_os is not None and args.arch is not None
                else args.target
            )
            result = compile_native(entry, args.output, target, args.clean, args.app)
            print(
                f"compiled {result.artifact} for {result.target} "
                f"({result.bytes} bytes, {result.operations} IR operations)"
            )
            return 0
        source_root = (args.source_root or entry.parent).expanduser().resolve()
        if args.command == "freeze":
            result = freeze(
                entry,
                args.output,
                source_root,
                tuple(args.include),
                tuple(args.exclude),
                tuple(args.wheel),
                args.dependency_mode,
                args.clean,
                app=args.app,
                name=args.name,
                icon=args.icon,
                compact=args.compact,
            )
            print(
                f"froze {result.bundle} ({result.files} files, {result.bytes} bytes); "
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
