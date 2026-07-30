"""Lay a Windows program out beside its interpreter, so it can find it.

There is no bundle on Windows: the executable, the interpreter and the
packages all sit in one directory. What makes that harder than it sounds is
that the embeddable CPython decides for itself what is importable.

It ships a `pythonXY._pth` naming exactly two places - the zip it came with,
and the directory beside it - and once that file exists `sys.path` is those
two entries and nothing else. Site handling is off. So packages copied into
`Lib\\site-packages` are simply invisible, and the program reports
ModuleNotFoundError for a directory that is plainly there. Nothing warns, and
a windowed executable has no console to say it in.

So placing packages and naming them on the path are one operation here, not
two steps a caller has to remember to do in order.
"""

from __future__ import annotations

import shutil
from pathlib import Path

#: Where packages go, relative to the executable.
SITE_PACKAGES = Path("Lib") / "site-packages"

#: The same place, spelled the way the target writes it. Building a Windows
#: program on a Mac makes Path render separators the host's way, and a path
#: file is read by Windows rather than by whatever produced it.
SITE_PACKAGES_ENTRY = "Lib\\site-packages"


class BundleError(Exception):
    """A Windows layout could not be assembled."""


def path_files(root: Path) -> list[Path]:
    """The interpreter's path files, if it brought any."""
    return sorted(root.glob("python*._pth"))


def name_site_packages(root: Path) -> int:
    """Add the site-packages directory to every path file that omits it.

    Returns how many files were changed. Doing nothing is the right answer
    when the interpreter ships no path file: then the default rules apply and
    site-packages is found without help.
    """
    entry = SITE_PACKAGES_ENTRY
    changed = 0
    for path in path_files(root):
        try:
            lines = path.read_text().splitlines()
        except OSError as error:
            raise BundleError(f"cannot read {path}: {error}") from error
        if any(line.strip().lower() == entry.lower() for line in lines):
            continue
        # After the directory entry, so a package beside the executable still
        # wins over one installed into site-packages - the order a normal
        # interpreter would use.
        for index, line in enumerate(lines):
            if line.strip() == ".":
                lines.insert(index + 1, entry)
                break
        else:
            lines.append(entry)
        # newline= pinned: a path file written on a Mac and one written on
        # Windows must come out byte for byte the same.
        path.write_text("\n".join(lines) + "\n", newline="\n")
        changed += 1
    return changed


def carry_runtime(root: Path, runtime: Path) -> int:
    """Copy an interpreter in beside the executable. Returns bytes copied."""
    runtime = runtime.expanduser().resolve()
    if not runtime.is_dir():
        raise BundleError(f"not an interpreter directory: {runtime}")
    if not any(runtime.glob("python*.dll")):
        raise BundleError(
            f"no pythonXY.dll in {runtime}: this wants the directory an "
            f"embeddable CPython was unpacked into"
        )
    copied = 0
    for source in sorted(runtime.rglob("*")):
        if not source.is_file():
            continue
        destination = root / source.relative_to(runtime)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += destination.stat().st_size
    return copied


def carry_packages(root: Path, sources: tuple[Path, ...]) -> int:
    """Copy site-packages trees in, and make sure the path names them."""
    target = root / SITE_PACKAGES
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in sources:
        source = source.expanduser().resolve()
        if not source.is_dir():
            raise BundleError(f"not a directory: {source}")
        for item in sorted(source.rglob("*")):
            if not item.is_file():
                continue
            destination = target / item.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            copied += destination.stat().st_size
    name_site_packages(root)
    return copied
