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

    read = _read_arguments(sys.argv[1:])
    if read[0] is _BAD:
        return 2
    where, target, method, includes, fetch, defines, libraries, watch = read
    return ask(
        where, target, method, includes, fetch, defines, libraries, watch
    )


#: What `_read_arguments` returns when it could not read them.
_BAD = object()


def _read_arguments(
    given: "list[str]",
) -> "tuple[str | None, str | None, str | None, tuple[str, ...], bool, tuple[str, ...], tuple[str, ...]]":
    """The path, and any of the three questions answered in advance.

        python3 build.py app.py
        python3 build.py app.cpp --target windows-arm64
        python3 build.py app.cpp --include vendor/include
        python3 build.py app.py --target linux-x86_64 --how freeze
        python3 build.py app.cpp --auto-fetch
        python3 build.py app.c --define NDEBUG -D VERSION=3
        python3 build.py app.cpp --library WebView2Loader.dll
        python3 build.py app.py --watch

    Answering them on the command line is what lets a script use this same
    entry point rather than a different one - a build that is only reachable
    by typing at it is a build nothing can check.

    `--include` may be given more than once. Directories called `include`,
    `inc`, `headers` or `src` beside the program are searched anyway; this is
    for headers that live somewhere else.

    `--auto-fetch` says that a header py2bin cannot find here may be looked
    up in a package index and downloaded. Without it nothing reaches the
    network, which is what keeps a build the same on a machine with none.

    `--define NAME` or `--define NAME=VALUE` (`-D` for short, repeatable) is
    what a header means when it says you must define something: `py2bin cc`
    has always taken these and this entry point had not, so the one thing a
    header asked for could not be given to it the documented way.

    `--watch` runs the program once, here, and adds the files it opened to
    the ones reading it found. Reading finds every branch without taking any;
    running takes one branch and finds what only that branch knows - a path
    read out of a config file, say. Neither is the whole answer, so this adds
    rather than replaces. Off unless asked for: running a program may write,
    send, or want a password, and a build should not do that unasked.

    `--library NAME.dll` (repeatable) names a shared library holding a
    function the program calls and never defines. With `--auto-fetch` this is
    usually not needed: the component that declared the function shipped the
    library too, and py2bin reads what that library exports to find out which
    one it is. It is here for a header supplied by hand, where there is no
    package to look in.
    """

    where = target = method = None
    fetch = False
    includes: "list[str]" = []
    defines: "list[str]" = []
    libraries: "list[str]" = []
    watch = False
    index = 0
    while index < len(given):
        piece = given[index]
        if piece in (
            "--target", "--how", "--include", "-I", "--define", "-D",
            "--library", "-l",
        ):
            if index + 1 >= len(given):
                print(f"{piece} needs a value after it.")
                return _BAD, None, None, (), False, (), (), False
            if piece == "--target":
                target = given[index + 1]
            elif piece == "--how":
                method = given[index + 1]
            elif piece in ("--define", "-D"):
                defines.append(given[index + 1])
            elif piece in ("--library", "-l"):
                libraries.append(given[index + 1])
            else:
                includes.append(given[index + 1])
            index += 2
            continue
        if piece == "--auto-fetch":
            fetch = True
            index += 1
            continue
        if piece == "--watch":
            watch = True
            index += 1
            continue
        if piece in ("-h", "--help"):
            print(__doc__.strip().split("\n\n")[0])
            print("\n  --target NAME    which machine, without being asked")
            print("  --how NAME       compile-capi, freeze, or compile")
            print("  --include DIR    where to look for headers (repeatable)")
            print("  --auto-fetch     download a header this cannot find here")
            print("  --define NAME    define a macro before the file is read")
            print("  --library NAME   a DLL a called function lives in")
            print("  --watch          run it once and note the files it opens")
            return _BAD, None, None, (), False, (), (), False
        if piece.startswith("-"):
            print(f"{piece} is not an option this understands.")
            return _BAD, None, None, (), False, (), (), False
        where = piece
        index += 1
    return (
        where, target, method, tuple(includes), fetch, tuple(defines),
        tuple(libraries), watch,
    )


if __name__ == "__main__":
    if _shadowed_the_packaging_tool():
        raise SystemExit(_hand_over_to_the_packaging_tool())
    raise SystemExit(main())
