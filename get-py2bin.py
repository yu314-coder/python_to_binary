#!/usr/bin/env python3
"""Install py2bin and build a program, with nothing but Python.

    python3 get-py2bin.py

No pip, no git, no compiler, no paths to type. This downloads py2bin from
PyPI, puts it somewhere importable, looks at the Python files beside it, asks
which one is the program, and builds it.

Written to run where less is available than usual - a phone runtime, a locked
-down box, a fresh machine. It uses only the standard library, and only
urllib to reach the network: shelling out to curl or wget would need a
subprocess, and a subprocess is exactly what some of those runtimes will not
give you. urllib is there wherever Python is.
"""

import hashlib
import json
import os
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

INDEX = "https://pypi.org/pypi/python-to-binary/json"
HOME = Path(os.environ.get("PY2BIN_HOME", Path.home() / ".py2bin"))


def say(message: str) -> None:
    print(message, flush=True)


#: Downloaders to try after urllib, in order. Each is a template taking the
#: destination and the URL. They exist because some Python builds are shipped
#: without a working ssl module, or with the network kept away from the
#: interpreter while the shell beside it can still reach out - a code editor
#: on a tablet, typically. This script may shell out; the library may not,
#: and does not.
_DOWNLOADERS = (
    ("curl", ["curl", "-fsSL", "--retry", "2", "-o", "{out}", "{url}"]),
    ("wget", ["wget", "-q", "-O", "{out}", "{url}"]),
    ("fetch", ["fetch", "-q", "-o", "{out}", "{url}"]),
    ("powershell", [
        "powershell", "-NoProfile", "-Command",
        "Invoke-WebRequest -UseBasicParsing -Uri '{url}' -OutFile '{out}'",
    ]),
)


def _by_command(url: str, label: str) -> bytes | None:
    """Ask whatever downloader this machine has, if Python itself cannot."""
    import shutil
    import subprocess  # not importable under src/ - see the module docstring

    for name, template in _DOWNLOADERS:
        if shutil.which(name) is None:
            continue
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch) / "payload"
            command = [
                part.format(out=str(out), url=url) for part in template
            ]
            say(f"  (python could not reach the network; trying {name})")
            try:
                finished = subprocess.run(
                    command, capture_output=True, timeout=300
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if finished.returncode == 0 and out.is_file() and out.stat().st_size:
                return out.read_bytes()
            say(f"  {name} could not fetch {label}")
    return None


def fetch(url: str, label: str) -> bytes:
    if not url.startswith("https://"):
        raise SystemExit(f"refusing to fetch {label} over anything but HTTPS")
    say(f"  downloading {label} ...")
    try:
        with urllib.request.urlopen(url, timeout=120) as stream:
            return stream.read()
    except Exception as reason:  # any failure at all is worth falling back on
        payload = _by_command(url, label)
        if payload is not None:
            return payload
        raise SystemExit(
            f"could not download {label}: {reason}\n"
            f"  Python's own networking failed and none of "
            f"{', '.join(name for name, _ in _DOWNLOADERS)} could be used.\n"
            f"  Download it by hand and this will find it:\n    {url}"
        ) from reason


def newest_release() -> tuple[str, str, str]:
    """The newest published version, and where its source archive is."""
    data = json.loads(fetch(INDEX, "the package index").decode())
    version = data["info"]["version"]
    for entry in data["urls"]:
        if entry["packagetype"] == "sdist":
            return version, entry["url"], entry["digests"]["sha256"]
    for entry in data["urls"]:
        if entry["packagetype"] == "bdist_wheel":
            return version, entry["url"], entry["digests"]["sha256"]
    raise SystemExit("the index lists no archive for python-to-binary")


def already_here() -> bool:
    """Whether this interpreter can already import py2bin.

    Someone with a working pip may well have installed it that way. Running
    this script is still worth it then: it lends the library a downloader that
    falls back to curl, which is the whole point on a machine whose Python can
    usually reach the network but not always.
    """
    try:
        import py2bin  # noqa: F401
    except ImportError:
        return False
    return True


def install() -> Path:
    """Put py2bin where this interpreter can import it, and answer where."""
    version, url, digest = newest_release()
    root = HOME / version
    if (root / "py2bin" / "__init__.py").is_file():
        say(f"  py2bin {version} is already in {root}")
        return root
    payload = fetch(url, f"py2bin {version}")
    got = hashlib.sha256(payload).hexdigest()
    if got != digest:
        raise SystemExit(
            f"the download does not match the hash the index published\n"
            f"  expected {digest}\n  got      {got}"
        )
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / Path(url).name
        archive.write_bytes(payload)
        unpacked = Path(scratch) / "unpacked"
        unpacked.mkdir()
        if archive.suffix == ".whl" or archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(unpacked)
        else:
            with tarfile.open(archive) as bundle:
                _safe_extract(bundle, unpacked)
        source = next(unpacked.rglob("py2bin/__init__.py"), None)
        if source is None:
            raise SystemExit("the archive holds no py2bin package")
        _copy_tree(source.parent, root / "py2bin")
    say(f"  installed py2bin {version} into {root}")
    return root


def _safe_extract(bundle: tarfile.TarFile, destination: Path) -> None:
    """Unpack, refusing any member that would land outside the directory."""
    for member in bundle.getmembers():
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(destination.resolve())):
            raise SystemExit(f"the archive tries to write outside: {member.name}")
    bundle.extractall(destination)


def _copy_tree(source: Path, destination: Path) -> None:
    for item in sorted(source.rglob("*")):
        target = destination / item.relative_to(source)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())


def choose_program(here: Path) -> Path:
    """Ask which file is the program, when it is not obvious."""
    candidates = sorted(
        path
        for path in here.glob("*.py")
        if path.name != Path(__file__).name and not path.name.startswith("_")
    )
    if not candidates:
        raise SystemExit(f"no Python files to build in {here}")
    if len(candidates) == 1:
        say(f"  building {candidates[0].name}, the only program here")
        return candidates[0]
    obvious = [p for p in candidates if p.name in ("main.py", "app.py", "__main__.py")]
    say("\nWhich file is the program? The others are found on their own if it")
    say("imports them.\n")
    for index, path in enumerate(candidates, start=1):
        mark = "  <- looks like it" if path in obvious else ""
        say(f"  {index:>2}. {path.name}{mark}")
    default = candidates.index(obvious[0]) + 1 if obvious else 1
    while True:
        try:
            answer = input(f"\nNumber [{default}]: ").strip() or str(default)
        except EOFError:
            # No one to ask - a pipe, a build script, a runtime with no
            # console. The default is the one that looked like the program.
            say(f"\n  nothing to read from; taking {default}")
            answer = str(default)
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            return candidates[int(answer) - 1]
        say("  that is not one of the numbers above")


def main() -> int:
    say("py2bin\n")
    if already_here():
        import py2bin

        say(f"  using the py2bin already installed here ({py2bin.__version__})")
    else:
        sys.path.insert(0, str(install()))
    here = Path.cwd()
    program = choose_program(here)

    try:
        from py2bin.requirements import discover  # noqa: E402
    except ImportError:
        # An older py2bin than the one that learned to work this out. The
        # build still runs; it just cannot say in advance what it will fetch.
        discover = None

    needs = discover(program) if discover else None
    if needs is None:
        say("  (this py2bin cannot list what the program needs in advance)")
    elif needs.local:
        say(f"  it imports {', '.join(needs.local)} from beside it - carried in")
    if needs and needs.projects:
        say(f"  it needs {', '.join(needs.projects)} - downloaded during the build")
    if needs and needs.unknown:
        say(
            f"  it imports {', '.join(needs.unknown)}, which this cannot name a\n"
            f"  project for; pass --fetch-package NAME to say which"
        )

    # Hand the library this script's downloader, so the interpreter and the
    # wheels it fetches during the build come down the same way the package
    # itself did. The library cannot shell out; this can, and lends it the
    # result.
    try:
        from py2bin import runtime_fetch  # noqa: E402

        runtime_fetch.DOWNLOADER = fetch
    except (ImportError, AttributeError):
        pass

    say("\nBuilding. This carries an interpreter, so it takes a moment.\n")
    from py2bin.cli import main as build  # noqa: E402

    output = here / "dist" / program.stem
    arguments = ["compile-capi", str(program), "--crash-log", "--clean"]
    if discover is not None:
        # Both arrived in the same release, so the one answers for the other:
        # an older py2bin that cannot work out what is needed cannot fetch it
        # either, and would refuse the flag.
        arguments.append("--auto-fetch")
    arguments += ["-o", str(output)]
    return build(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
