"""What an archive may not talk py2bin into doing.

py2bin never *runs* what it downloads - no `setup.py`, no pip, no install
hooks. Unpacking is therefore the only moment a hostile wheel or runtime pack
acts at all, and these are the ways it used to be able to.

The escape in the first test was reproduced against the old code on Python
3.11 and wrote a file outside the destination. It is invisible on 3.14, where
`extractall` defaults to the `data` filter - which is why the check has to be
py2bin's own: `requires-python` is 3.10.
"""

from __future__ import annotations

import io
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

try:
    import pytest
except ModuleNotFoundError as missing:  # pragma: no cover - no pytest here
    # The suite is meant to run with nothing installed, and the README says
    # so. This module wants pytest's fixtures; where there is no pytest, say
    # that rather than failing to import - `unittest discover` reports a
    # module that will not import as an error, which reads like a broken
    # test rather than a missing tool.
    import unittest as _unittest

    raise _unittest.SkipTest("pytest is not installed") from missing

from py2bin.archives import (
    ArchiveError,
    extract_tar,
    extract_zip,
    safe_relative,
)


def _tar(members, path: Path) -> Path:
    with tarfile.open(path, "w") as archive:
        for info, payload in members:
            archive.addfile(info, io.BytesIO(payload) if payload is not None else None)
    return path


def _symlink(name: str, target: str):
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return (info, None)


def _regular(name: str, payload: bytes, mode: int = 0o644):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = mode
    return (info, payload)


def test_a_symlink_that_escapes_is_refused_before_anything_is_written():
    # `esc -> ..` then `esc/SECRET`. The old check resolved each name and
    # compared strings: at the time it ran, `esc` did not exist to resolve
    # through, so both members looked innocent and `extractall` then followed
    # the link it had just made.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "SECRET.txt").write_text("original\n")
        archive = _tar(
            [_symlink("esc", ".."), _regular("esc/SECRET.txt", b"PWNED\n")],
            root / "evil.tar",
        )
        with pytest.raises(ArchiveError):
            extract_tar(archive, root / "out")
        assert (root / "SECRET.txt").read_text() == "original\n"


def test_an_absolute_symlink_is_refused():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = _tar([_symlink("bad", "/etc/passwd")], root / "abs.tar")
        with pytest.raises(ArchiveError):
            extract_tar(archive, root / "out")


def test_a_hard_link_out_of_the_archive_is_refused():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        info = tarfile.TarInfo("hard")
        info.type = tarfile.LNKTYPE
        info.linkname = "../outside"
        archive = _tar([(info, None)], root / "hard.tar")
        with pytest.raises(ArchiveError):
            extract_tar(archive, root / "out")


def test_a_device_node_is_refused():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        info = tarfile.TarInfo("dev/null")
        info.type = tarfile.CHRTYPE
        archive = _tar([(info, None)], root / "dev.tar")
        with pytest.raises(ArchiveError):
            extract_tar(archive, root / "out")


def test_links_inside_the_archive_still_work():
    """The feature the refusal must not cost.

    A CPython framework *is* a structure of symbolic links - `Versions/
    Current`, `bin/python3` - so a runtime pack that lost them would be one
    that does not run. Rejecting every link would have been the easy fix and
    the wrong one.
    """

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        out = root / "out"
        archive = _tar(
            [
                _regular("Versions/3.14/bin/python3.14", b"#!/bin/sh\n", 0o755),
                _symlink("Versions/Current", "3.14"),
                _symlink("bin/python3", "../Versions/3.14/bin/python3.14"),
            ],
            root / "framework.tar",
        )
        assert extract_tar(archive, out) == 3
        assert (out / "Versions/Current").is_symlink()
        # Reached through both links, which is what the bundle does at run time.
        assert (out / "bin/python3").resolve().is_file()
        assert (out / "Versions/Current/bin/python3.14").exists()


def test_the_executable_bit_survives():
    # A wheel shipping a helper program - Qt's QtWebEngineProcess, a console
    # script - is useless without it, and this was fixed once already.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        out = root / "out"
        archive = _tar(
            [_regular("bin/helper", b"#!/bin/sh\n", 0o755)], root / "x.tar"
        )
        extract_tar(archive, out)
        assert (out / "bin/helper").stat().st_mode & 0o111


def test_an_archive_that_expands_too_far_is_refused():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # The declared sizes are what is added up, and the refusal happens
        # before a byte is written - the point of a limit is not to watch a
        # disk fill. A real payload against a small ceiling says the same
        # thing as a fake enormous one without a malformed archive.
        archive = _tar(
            [
                _regular("a", b"x" * 4096),
                _regular("b", b"y" * 4096),
            ],
            root / "bomb.tar",
        )
        with pytest.raises(ArchiveError):
            extract_tar(archive, root / "out", max_expanded=1024)
        # And the same archive is fine when it fits.
        assert extract_tar(archive, root / "fits", max_expanded=1 << 20) == 2


def test_a_zip_that_expands_too_far_is_refused():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "bomb.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("big", b"\0" * (4 * 1024 * 1024))
        with pytest.raises(ArchiveError):
            extract_zip(archive, root / "out", max_expanded=1024)


@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "/absolute.txt",
        "..\\..\\windows.txt",
        "dir\\file.txt",
        "",
        "a/../b",
    ],
)
def test_names_that_may_not_land_anywhere(name):
    assert safe_relative(name) is None


@pytest.mark.parametrize("name", ["pkg/mod.py", "a/b/c.txt", "./x.py"])
def test_ordinary_names_still_land(name):
    assert safe_relative(name) is not None


def test_a_zip_member_with_a_backslash_is_refused():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "evil.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("..\\..\\outside.txt", b"x")
        with pytest.raises(ArchiveError):
            extract_zip(archive, root / "out")


def test_a_wheel_member_with_a_backslash_is_refused():
    """The Windows-only hole, which POSIX hosts cannot feel.

    `..\\..\\x` is one atom to `PurePosixPath`, so it holds no `..` part and
    is not absolute - the check passed - and then `Path(*parts)` on Windows
    splits it again into a traversal. `runtime_fetch` refused the character
    and `freezer` did not, which is what two copies of a rule drift into.
    """

    from py2bin.freezer import _safe_wheel_member

    assert _safe_wheel_member("..\\..\\outside.txt") is None
    assert _safe_wheel_member("pkg\\mod.py") is None
    # The shapes a real wheel uses are untouched, including the `.data`
    # redirection that puts purelib content at the top.
    assert _safe_wheel_member("pkg/mod.py") == Path("pkg/mod.py")
    assert _safe_wheel_member("d-1.0.data/purelib/pkg/x.py") == Path("pkg/x.py")


def test_the_bootstrapper_carries_the_same_rule():
    """`get-py2bin.py` cannot import any of this - it is what installs it.

    So it holds a copy, and a copy is exactly what drifted last time. This
    checks the two still answer alike rather than that the copy exists.
    """

    import runpy

    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "get-py2bin.py"))
    their_check = module["_safe_relative"]
    for name in ("../x", "/abs", "..\\..\\x", "a\\b", "", "a/../b"):
        assert their_check(name) is None, name
    for name in ("pkg/mod.py", "a/b.txt"):
        assert their_check(name) is not None, name


def test_no_fetch_path_still_calls_extractall():
    """The check above is worth nothing if a caller bypasses it.

    `archives.py` could be perfect and unused - it was three separate,
    subtly-wrong copies that caused this, so what needs pinning is that
    nothing downloading from the network unpacks its own way again. Bootstrap
    is exempt and named: it unpacks py2bin's *own* onefile payload, which
    py2bin wrote and signed, out of the binary the user is already running.
    """

    root = Path(__file__).resolve().parents[1]
    exempt = {"bootstrap.py", "archives.py"}
    offenders = []
    for source in sorted((root / "src" / "py2bin").rglob("*.py")):
        if source.name in exempt:
            continue
        if "extractall(" in source.read_text(encoding="utf-8"):
            offenders.append(source.name)
    assert not offenders, f"these unpack without the shared checks: {offenders}"

    # And the bootstrapper, which cannot import the module at all.
    bootstrapper = (root / "get-py2bin.py").read_text(encoding="utf-8")
    assert "extractall(" not in bootstrapper
