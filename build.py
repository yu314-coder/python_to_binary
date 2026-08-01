#!/usr/bin/env python3
"""Build a program with the py2bin sitting next to this file.

    python3 build.py

Nothing is installed and nothing is downloaded to get started: this runs the
`src/py2bin` in the clone it lives in. Clone the repository, drop your program
in beside it or run this from the directory your program is in, and answer
three questions - which file, which machine, what shape.

Written for editors on tablets and other places where pip is awkward and a
path is worse. `get-py2bin.py` is the other half of this: it fetches py2bin
when you have no clone. This one assumes you do.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "src"

#: Files that belong to py2bin rather than to anyone's program, so they are
#: not offered as something to build when this runs inside its own clone.
_OURS = {"build.py", "get-py2bin.py", "setup.py", "conftest.py", "noxfile.py"}

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
    ("linux-arm64", "Linux, 64-bit ARM"),
)

#: The shapes each target can take, one file first because that is what a
#: person hands to someone else. A .app is a directory however it is built, so
#: on macOS the single file is the disk image holding it.
SHAPES = {
    "darwin": (
        ("dmg", "ONE FILE: a .dmg holding the app, ready to hand over"),
        ("app", "a .app folder, carrying its own interpreter"),
        ("bin", "a plain executable, needing Python on the machine"),
    ),
    "windows": (
        ("onefile", "ONE FILE: an .exe that unpacks itself when it runs"),
        ("exe", "a folder holding the .exe and its interpreter"),
        ("bin", "a plain .exe, needing Python on the machine"),
    ),
    "linux": (
        ("bin", "ONE FILE: an executable linking the machine's libpython"),
    ),
}


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
            if entry.name in _NOT_PROGRAMS or entry.resolve() == HERE:
                continue
            if programs(entry) and entry not in places:
                places.append(entry)
    return places


def where_to_build(here: Path) -> Path | None:
    """The directory whose program is to be built, asking if need be."""
    if programs(here):
        return here
    say(f"\n  Nothing to build in {here}")
    if here.resolve() == HERE:
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


def main() -> int:
    # Everything is said on stdout. Some editors show only that, and an
    # explanation nobody sees is the same as no explanation: this exact
    # script reported "nothing to build" to a console that dropped stderr,
    # and looked from the outside like a crash during imports.
    if not (SOURCE / "py2bin" / "__init__.py").is_file():
        say(f"No py2bin beside this script - expected {SOURCE / 'py2bin'}.")
        say("Run this from a clone of the repository, or use get-py2bin.py.")
        return 1
    sys.path.insert(0, str(SOURCE))
    import py2bin

    say(f"py2bin {py2bin.__version__}, from {SOURCE}")

    start = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.cwd()
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
    shapes = SHAPES[system]
    shape = shapes[ask("What shape should it be?", shapes, 1)][0]

    # A macOS bundle links against a macOS Python.framework, and only a Mac
    # has one. Said before the build rather than after it, because everything
    # up to that point takes a while and downloads a good deal.
    if system == "darwin" and shape in ("app", "dmg"):
        import platform as _platform

        if _platform.system() != "Darwin":
            say(
                "\n  Note: a macOS bundle carries a macOS Python.framework, and\n"
                "  this is not a Mac. The build will ask for one. A Windows\n"
                "  target downloads its own interpreter and needs nothing from\n"
                "  this machine, if that suits."
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

    suffix = {
        "app": ".app", "dmg": ".app", "exe": ".exe", "onefile": ".exe", "bin": ""
    }[shape]
    output = here / "dist" / f"{program.stem}{suffix}"
    arguments = [
        "compile-capi",
        str(program),
        "--target",
        target,
        "--crash-log",
        "--clean",
        "--auto-fetch",
    ]
    if shape in ("app", "dmg", "exe"):
        # Without these a bundle carries the whole standard library twice over
        # - once as source and once as bytecode - along with every module the
        # program cannot reach. It is the difference between 220 MB and about
        # a third of that.
        arguments += ["--prune-unused", "--zip-stdlib"]
    if shape in ("app", "dmg"):
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
        ]
    if shape == "dmg":
        arguments.append("--dmg")
    for path in carried:
        arguments += ["--include", str(path)]
    arguments += ["-o", str(output)]

    delivered = {
        "dmg": f"{program.stem}.dmg",
        "onefile": f"{program.stem}-onefile.exe",
    }.get(shape, output.name)
    say(f"\nBuilding {delivered} for {target}.")
    if shape in ("app", "dmg", "exe"):
        say("It carries an interpreter, so this takes a while.")
    say()

    from py2bin.cli import main as build

    code = build(arguments)
    if code == 0 and shape == "onefile":
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
