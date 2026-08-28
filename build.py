#!/usr/bin/env python3
"""Build a program with the py2bin sitting next to this file.

    python3 build.py [path/to/your/program]

Nothing is installed and nothing is downloaded to get started: this runs the
`src/py2bin` in the clone it lives in. Answer three questions - which file,
which machine, and which of the two ways to build it: ship Python with it the
way PyInstaller does, or compile it to machine code. Everything else is found
or downloaded rather than typed.

The questions themselves live in `py2bin.interactive`, so `py2bin make` asks
exactly the same ones for anyone who installed with pip. This file is the way
in when there is no install, only a clone.

`get-py2bin.py` is the other half: it fetches py2bin when there is no clone
either.

One thing this file has to handle that it did not ask for: it is called
`build.py`, and `build` is also the name of the standard PEP 517 frontend that
packages a project. `python -m build` from this directory puts the working
directory first on the path and finds *this* file, so a command meant to write
a wheel instead started asking which machine to compile for - and, given an
argument like `--outdir`, went looking for a program of that name. That is not
a mistake anyone would think to make twice, but they would make it once, and
the failure did not say what had happened. :func:`_shadowed_the_packaging_tool`
notices and hands over.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "src"


def _shadowed_the_packaging_tool() -> bool:
    """True when `python -m build` reached this file instead of the frontend.

    `__spec__` is what tells them apart: running a file as a script leaves it
    None, and `-m` sets it to the module name that was asked for. Nobody types
    `python -m build` wanting the three questions, so the intent is not in
    doubt.
    """

    return __spec__ is not None and __spec__.name in ("build", "build.__main__")


def _hand_over_to_the_packaging_tool() -> int:
    """Run the real `build`, found by looking past this directory."""

    import runpy

    # This directory is what shadowed it, so take it out and let the ordinary
    # import machinery find the installed package. Both spellings go: `-m`
    # inserts the working directory as an absolute path, and a plain empty
    # string means the same thing.
    sys.path = [
        entry
        for entry in sys.path
        if entry not in ("", ".") and Path(entry).resolve() != HERE
    ]
    sys.modules.pop("build", None)
    try:
        runpy.run_module("build", run_name="__main__", alter_sys=True)
    except ImportError:
        print(
            "`python -m build` found this repository's build.py rather than "
            "the packaging frontend of the same name, and no installed "
            "`build` was there to hand over to.",
            file=sys.stderr,
        )
        print(
            "Install it with `pip install build`, or use `uv build`, which "
            "does not go through a module name this file can shadow.",
            file=sys.stderr,
        )
        print(
            "To reach this file on purpose - the three questions that compile "
            "a program - run `python3 build.py`.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    if not (SOURCE / "py2bin" / "__init__.py").is_file():
        # Said on stdout: some editors show only that, and an explanation
        # nobody sees is the same as no explanation.
        print(f"No py2bin beside this script - expected {SOURCE / 'py2bin'}.")
        print("Run this from a clone of the repository, or use get-py2bin.py.")
        return 1
    sys.path.insert(0, str(SOURCE))
    from py2bin.interactive import main as ask

    answers = _read_arguments(sys.argv[1:])
    if answers is None:
        return 2
    return ask(**answers)


#: Every question this can answer in advance, and what it answers with when
#: nothing is said. The names are `py2bin.interactive.main`'s own parameters,
#: so an answer added there is added here by writing it down once.
_ANSWERS = {
    "where": None,
    "target": None,
    "method": None,
    "include_dirs": (),
    "auto_fetch": False,
    "defines": (),
    "libraries": (),
    "watch": True,
    "onefile": True,
}

#: The options that take a value, and which answer each fills in. Repeatable
#: ones collect; the rest replace.
_TAKES_A_VALUE = {
    "--target": ("target", False),
    "--how": ("method", False),
    "--include": ("include_dirs", True),
    "-I": ("include_dirs", True),
    "--define": ("defines", True),
    "-D": ("defines", True),
    "--library": ("libraries", True),
    "-l": ("libraries", True),
}

#: The options that are a yes or a no on their own.
_SWITCHES = {
    "--auto-fetch": ("auto_fetch", True),
    "--watch": ("watch", True),
    "--no-watch": ("watch", False),
    "--onefile": ("onefile", True),
    "--no-onefile": ("onefile", False),
}

#: What each is for, in the order `--help` should list them.
_EXPLAINED = (
    ("--target NAME", "which machine, without being asked"),
    ("--how NAME", "compile-capi, freeze, or compile"),
    ("--include DIR", "where to look for headers (repeatable)"),
    ("--auto-fetch", "download a header or library this cannot find here"),
    ("--define NAME", "define a macro before the file is read"),
    ("--library NAME", "a DLL a called function lives in (repeatable)"),
    ("--no-watch", "do not run the program to see what it opens"),
    ("--no-onefile", "leave what is carried beside the program"),
)


def _read_arguments(given: "list[str]") -> "dict | None":
    """The path, and any of the questions answered in advance. None if unread.

        python3 build.py app.py
        python3 build.py app.cpp --target windows-arm64
        python3 build.py app.cpp --auto-fetch
        python3 build.py app.c --define NDEBUG -D VERSION=3
        python3 build.py app.cpp --library WebView2Loader.dll
        python3 build.py app.py --no-watch --no-onefile

    Answering them here is what lets a script use this same entry point
    rather than a different one - a build that is only reachable by typing at
    it is a build nothing can check.

    Read into a dict rather than a tuple in a fixed order. The tuple grew a
    field every time this learned another answer, and every caller and every
    early return had to grow with it in the same order; one of them did not,
    and the mistake was silent until something ran.
    """

    answers = dict(_ANSWERS)
    collected: "dict[str, list[str]]" = {}
    index = 0
    while index < len(given):
        piece = given[index]
        if piece in _TAKES_A_VALUE:
            if index + 1 >= len(given):
                print(f"{piece} needs a value after it.")
                return None
            key, repeats = _TAKES_A_VALUE[piece]
            if repeats:
                collected.setdefault(key, []).append(given[index + 1])
            else:
                answers[key] = given[index + 1]
            index += 2
            continue
        if piece in _SWITCHES:
            key, value = _SWITCHES[piece]
            answers[key] = value
            index += 1
            continue
        if piece in ("-h", "--help"):
            print(__doc__.strip().split("\n\n")[0])
            print()
            for spelled, why in _EXPLAINED:
                print(f"  {spelled:16s} {why}")
            return None
        if piece.startswith("-"):
            print(f"{piece} is not an option this understands.")
            return None
        answers["where"] = piece
        index += 1
    for key, values in collected.items():
        answers[key] = tuple(values)
    return answers


if __name__ == "__main__":
    if _shadowed_the_packaging_tool():
        raise SystemExit(_hand_over_to_the_packaging_tool())
    raise SystemExit(main())
