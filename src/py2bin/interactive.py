"""The three questions, asked wherever py2bin was installed from.

`build.py` in a clone runs this, and so does `py2bin build` for anyone who
installed with pip. Same flow either way: which file is the program, which
machine it is for, and which of the two ways to build it - everything else
found or downloaded rather than typed.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: py2bin's own directory, so it is never offered as somebody's program. When
#: installed rather than cloned this is site-packages, which nobody builds
#: from either.
_OWN_ROOT = Path(__file__).resolve().parent.parent.parent

_OURS = {"build.py", "get-py2bin.py", "setup.py", "conftest.py", "noxfile.py"}

#: Icon files a program is likely to ship, best first. Windows wants .ico and
#: macOS wants .icns; either is converted if the other is what is there.
_ICONS = ("icon.ico", "icon.icns", "icon.png", "app.ico", "app.icns")

#: Directories a program opens rather than imports - templates, web assets,
#: icons. They are carried beside it, because that is where it looks.
_DATA_DIRECTORIES = ("web", "assets", "static", "templates", "resources", "data")

#: Directories inside the clone that hold py2bin, not anyone's program. Run
#: from the clone root, these are all there is, and offering them would be
#: offering to compile the compiler.
_NOT_PROGRAMS = {"src", "tests", "docs", "dist", "build", "__pycache__"}

#: What can be built, and what each one produces. Kept in the order someone
#: is most likely to want, with this machine's own first at run time.
TARGETS = (
    ("darwin-arm64", "macOS, Apple silicon"),
    ("darwin-x86_64", "macOS, Intel"),
    ("windows-x86_64", "Windows, 64-bit Intel/AMD"),
    ("windows-arm64", "Windows on ARM"),
    ("linux-x86_64", "Linux, 64-bit Intel/AMD"),
    ("linux-arm64", "Linux, 64-bit ARM"),
)

#: The two ways py2bin can turn a program into something you hand over, and
#: there are only two. One ships the program next to a real interpreter; the
#: other translates it to C and then to machine code. Everything else - a
#: folder or one file, a .dmg or a .exe - follows from the target and is
#: decided here rather than asked about.
#:
#: There used to be three choices, and all three ran the same compiler. They
#: differed in *packaging*: a `.app` folder beside the `.dmg` that holds it,
#: an `.exe` folder beside the one-file `.exe` that unpacks to it. Picking
#: the one file gives the folder too, so nothing was gained by asking.
METHODS = (
    (
        "freeze",
        "SHIP PYTHON WITH IT - the program travels beside a real interpreter, "
        "the way PyInstaller does it.\n      Quickest to build, and runs any "
        "Python there is.",
    ),
    (
        "compile",
        "COMPILE IT - Python translated to C, and that C to machine code by "
        "py2bin's own compiler.\n      Slower to build; no source and no "
        "bytecode in what comes out.",
    ),
)

#: What the compiled result needs from the machine that runs it. A macOS
#: bundle carries a Python.framework and a Windows build carries the DLL it
#: was linked against, so both stand alone. Nothing is published to carry for
#: Linux, so a compiled Linux program uses the Python that is already there.
_COMPILED_CARRIES_PYTHON = {"darwin": True, "windows": True, "linux": False}


def can_freeze(target: str) -> bool:
    """Whether this machine can freeze for that one.

    Freezing needs a whole CPython built for the target. One is published for
    Windows and can be downloaded; for anything else it has to come from a
    machine like the target, which means the host itself. Asked before the
    menu is shown, because offering a choice that cannot be carried out is
    worse than not offering it.
    """
    return target.startswith("windows-") or target == host_target()


def methods_for(target: str):
    return METHODS if can_freeze(target) else METHODS[1:]


def say(message: str = "") -> None:
    print(message, flush=True)


def ask(question: str, options, default: int) -> int:
    """Offer a numbered list and answer with the chosen index."""
    say()
    say(question)
    say()
    for index, (_value, description) in enumerate(options, start=1):
        mark = "  <-" if index == default else ""
        say(f"  {index:>2}. {description}{mark}")
    while True:
        try:
            answer = input(f"\nNumber [{default}]: ").strip() or str(default)
        except EOFError:
            # Nothing to read from - a pipe, or a runtime with no console.
            say(f"\n  nothing to read from; taking {default}")
            answer = str(default)
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        say("  that is not one of the numbers above")


def host_target() -> str:
    import platform

    system = {"Darwin": "darwin", "Windows": "windows", "Linux": "linux"}.get(
        platform.system(), "linux"
    )
    machine = platform.machine().lower()
    architecture = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    return f"{system}-{architecture}"


def programs(here: Path) -> list[Path]:
    """The Python files that could be someone's program."""
    found = []
    for path in sorted(here.glob("*.py")):
        if path.name in _OURS or path.name.startswith("_"):
            continue
        found.append(path)
    return found


def nearby(here: Path) -> list[Path]:
    """Directories next to this one that hold something to build.

    Run from inside the clone there is nothing to build, and the useful thing
    is not an error - it is the list of places that do hold a program.
    """
    places = []
    for parent in (here, here.parent):
        if not parent.is_dir():
            continue
        try:
            entries = sorted(parent.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.name in _NOT_PROGRAMS or entry.resolve() == _OWN_ROOT:
                continue
            if programs(entry) and entry not in places:
                places.append(entry)
    return places


def where_to_build(here: Path) -> Path | None:
    """The directory whose program is to be built, asking if need be."""
    if programs(here):
        return here
    say(f"\n  Nothing to build in {here}")
    if here.resolve() == _OWN_ROOT:
        say("  (this is py2bin's own directory - your program goes elsewhere)")
    places = nearby(here)
    if places:
        chosen = ask(
            "Where is your program?",
            [(path, f"{path.name}/  ({len(programs(path))} file(s))") for path in places]
            + [(None, "somewhere else - let me type the path")],
            1,
        )
        if chosen < len(places):
            return places[chosen]
    while True:
        try:
            typed = input("\n  Path to the folder holding your program: ").strip()
        except EOFError:
            return None
        if not typed:
            return None
        candidate = Path(typed).expanduser()
        if candidate.is_file() and candidate.suffix == ".py":
            return candidate.parent
        if programs(candidate):
            return candidate
        say(f"  no Python files to build in {candidate}")


def main(where: str | None = None) -> int:
    # Everything is said on stdout. Some editors show only that, and an
    # explanation nobody sees is the same as no explanation: this exact
    # script reported "nothing to build" to a console that dropped stderr,
    # and looked from the outside like a crash during imports.
    import py2bin

    say(f"py2bin {py2bin.__version__}")

    start = Path(where).expanduser() if where else Path.cwd()
    if start.is_file() and start.suffix == ".py":
        here, forced = start.parent, start
    else:
        here, forced = where_to_build(start), None
    if here is None:
        say("\n  Nowhere to build from. Pass the folder as an argument:")
        say(f"    python3 {Path(__file__).name} /path/to/your/program")
        return 1
    candidates = programs(here)
    if forced is not None:
        candidates = [forced]
    say(f"  building from {here}")

    if len(candidates) == 1:
        program = candidates[0]
        say(f"\n  building {program.name}, the only program here")
    else:
        obvious = [
            path
            for path in candidates
            if path.name in ("main.py", "app.py", "__main__.py")
        ]
        default = candidates.index(obvious[0]) + 1 if obvious else 1
        chosen = ask(
            "Which file is the program? Any others it imports are found on "
            "their own.",
            [(path, path.name) for path in candidates],
            default,
        )
        program = candidates[chosen]

    ordered = sorted(TARGETS, key=lambda entry: entry[0] != host_target())
    target = ordered[
        ask(
            "Which machine is it for?",
            [(name, f"{label}  ({name})") for name, label in ordered],
            1,
        )
    ][0]

    system = target.split("-")[0]
    offered = methods_for(target)
    if len(offered) == 1:
        say(
            f"\n  Compiling it: freezing needs a CPython built for "
            f"{target}, and\n  one can only be downloaded for Windows or "
            f"taken from a machine\n  like the target. Build on {system} "
            f"itself for the other way."
        )
        method = offered[0][0]
    else:
        method = offered[ask("How should it be built?", offered, 1)][0]

    if method == "compile" and not _COMPILED_CARRIES_PYTHON[system]:
        # Said before the build rather than after it: everything up to that
        # point takes a while and downloads a good deal, and finding out at
        # the end that the target still needs Python is finding out too late.
        say(
            "\n  Note: a compiled Linux program uses the Python already on the\n"
            "  machine that runs it. Only Windows and macOS have an interpreter\n"
            "  py2bin can carry along."
        )

    from py2bin.requirements import discover

    needs = discover(program)
    if needs.local:
        say(f"\n  it imports {', '.join(needs.local)} from beside it")
    if needs.projects:
        say(f"  it needs {', '.join(needs.projects)}, which will be downloaded")
    if needs.unknown:
        say(
            f"  it imports {', '.join(needs.unknown)}, which this cannot name a\n"
            f"  project for - the build will go on without them"
        )

    # Anything the program opens rather than imports. Found rather than asked
    # for: a directory of web assets beside a program is what it is, and a
    # bundle without it is a bundle that starts and then cannot draw anything.
    carried = [
        here / name
        for name in _DATA_DIRECTORIES
        if (here / name).is_dir()
    ]
    if carried:
        say(f"  carrying {', '.join(path.name + '/' for path in carried)}")

    icon = next((here / name for name in _ICONS if (here / name).is_file()), None)
    if icon is not None:
        say(f"  using {icon.name} as the icon")

    windows = target.startswith("windows-")

    if method == "freeze":
        # One file, always: freezing has nothing to gain from a folder, and a
        # single file is the thing somebody can actually send. The program's
        # own directory travels with it, so the data directories found above
        # are already inside and need not be named.
        output = here / "dist" / f"{program.stem}{'.exe' if windows else ''}"
        arguments = [
            "freeze",
            str(program),
            "--target",
            target,
            "--auto-fetch",
            "--clean",
            "--onefile",
            "--compact",
            "--name",
            program.stem,
        ]
        if icon is not None:
            arguments += ["--icon", str(icon)]
        arguments += ["-o", str(output)]

        say(f"\nFreezing {output.name} for {target}.")
        say("It carries an interpreter, so this takes a while.")
        say()

        from py2bin.cli import main as build

        code = build(arguments)
        if code == 0:
            # freeze names the file itself when the target has no suffix of
            # its own, so report what is on disk rather than what was asked
            # for.
            final = output if output.exists() else output.with_suffix(".bin")
            say(f"\n  done: {final}")
        return code

    # Compiled. The shape follows from the target rather than being asked
    # about: a Mac gets the disk image, everything else gets the one file.
    shape = "dmg" if system == "darwin" else "onefile"
    output = here / "dist" / (
        f"{program.stem}.app" if shape == "dmg"
        else f"{program.stem}{'.exe' if windows else ''}"
    )
    arguments = [
        "compile-capi",
        str(program),
        "--target",
        target,
        "--crash-log",
        "--clean",
        "--auto-fetch",
        # Without these a build that carries an interpreter carries the whole
        # standard library twice over - once as source and once as bytecode -
        # along with every module the program cannot reach. It is the
        # difference between 220 MB and about a third of that.
        "--prune-unused",
        "--zip-stdlib",
    ]
    if shape == "dmg":
        # --site is baked into the program when it is compiled, which happens
        # before anything is downloaded, so where the packages will end up has
        # to be said now. It stays relative: it is resolved against the running
        # executable, which is what lets the bundle carry its own.
        arguments += [
            "--app",
            "--embed-python",
            "--site",
            "../Resources/site-packages",
            "--name",
            program.stem,
            "--dmg",
        ]
    elif not windows:
        # A target with no bundle folds the program and everything carried
        # beside it into the executable itself. Windows keeps its own path
        # below, which wraps the built folder rather than the program.
        arguments.append("--onefile")
    if icon is not None:
        arguments += ["--icon", str(icon)]
    for path in carried:
        arguments += ["--include", str(path)]
    arguments += ["-o", str(output)]

    delivered = {
        "dmg": f"{program.stem}.dmg",
        "onefile": f"{program.stem}-onefile.exe" if windows else output.name,
    }[shape]
    say(f"\nBuilding {delivered} for {target}.")
    if _COMPILED_CARRIES_PYTHON[system]:
        say("It carries an interpreter, so this takes a while.")
    say()

    from py2bin.cli import main as build

    code = build(arguments)
    if code == 0 and shape == "onefile" and windows:
        # One file rather than a folder: the same payload, wrapped in an
        # executable that unpacks itself where it runs.
        say("\n  packing it into a single file ...")
        from py2bin.onefile import create_onefile

        # Beside dist/, not inside it: the packer reads dist/, and an
        # archive written where it is being read contains itself.
        single = here / f"{program.stem}-onefile.exe"
        create_onefile(
            payload_root=output.parent,
            output=single,
            target=target,
            launcher=output,
            icon=icon,
            windows_windowed=True,
        )
        say(f"  done: {single}")
        return 0
    if code == 0:
        # Name what came out, not what it was built from: for a disk image the
        # .app is a step along the way, and the file to hand over is the .dmg.
        final = output.with_suffix(".dmg") if shape == "dmg" else output
        say(f"\n  done: {final}")
        if shape == "dmg":
            say("  (the .app beside it is what the image holds)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
