"""One hardened way to unpack an archive, for everything that unpacks one.

py2bin's safety story is that it never *runs* what it downloads - no
`setup.py`, no pip, no install hooks. Unpacking is therefore the only moment a
hostile wheel or runtime pack gets to act at all, which makes this module the
place that story is kept or lost.

**What was wrong before.** Three callers each wrote their own check, of the
form "resolve where the member would land and make sure the string starts with
the destination". Both halves fail:

*`resolve()` runs before extraction*, so a symlink the archive is about to
create is not there to be resolved through. An archive holding `esc -> ..`
followed by `esc/passwd` passes the check - neither member looks like it
escapes - and then `extractall` follows the link it just made. This was not
theoretical: it was reproduced against the old code on Python 3.11, writing a
file outside the destination.

*`str.startswith` is not a path comparison.* `/tmp/outsider` starts with
`/tmp/out`.

Python 3.14 happens to stop the first one, because `extractall` there defaults
to the `data` filter. py2bin supports 3.10 upwards, so on most of the range it
did not.

**What this does instead.** Every member is validated *before* anything is
written, and then written one at a time - never `extractall`. Containment is
decided lexically, on the archive's own names, because that is the only
question that can be answered before the tree exists. A link is checked by
normalising it against the directory it sits in and requiring the result to
stay inside; nothing on disk is consulted, so nothing on disk can lie.

**Links are allowed, and have to be.** A macOS framework is a structure of
symbolic links - `Versions/Current`, `bin/python3` - so a runtime pack that
lost them would be a runtime pack that does not work. `allow_links=False` is
for archives that have no business holding one, which is what source
distributions are held to.
"""

from __future__ import annotations

import os
import posixpath
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

#: Enough for a large scientific wheel, far short of a disk.
DEFAULT_MAX_EXPANDED = 4 * 1024 * 1024 * 1024
DEFAULT_MAX_MEMBERS = 200_000


class ArchiveError(ValueError):
    """An archive asked for something an archive may not have."""


def safe_relative(name: str) -> PurePosixPath | None:
    """Where a member may land inside the destination, or None if nowhere.

    Backslashes are refused rather than translated, and that is not
    fussiness. `..\\..\\x` is one atom to `PurePosixPath`, so it holds no
    `..` part and looks harmless - and then `Path(*parts)` on Windows splits
    it again and it is a traversal. Refusing the character is the only check
    that means the same thing on both kinds of host.
    """

    if not name or name.startswith("/") or "\\" in name:
        return None
    relative = PurePosixPath(name)
    if relative.is_absolute():
        return None
    parts = relative.parts
    if not parts or any(part in {"..", ".", ""} for part in parts):
        return None
    return PurePosixPath(*parts)


def _link_stays_inside(member_name: str, linkname: str, *, relative_to_root: bool) -> bool:
    """Whether following this link keeps you under the destination.

    Decided by normalising text, never by touching the filesystem: the tree
    the link would point into does not exist yet, and once it does it can
    contain links of its own that would answer the question wrongly.

    A symbolic link's target is read relative to the directory the link sits
    in; a hard link's names a member of the archive, so it is read from the
    archive root. Getting those two the same way round matters - a hard link
    checked as though it were relative to its own directory would refuse
    ordinary archives.
    """

    if not linkname or "\\" in linkname or posixpath.isabs(linkname):
        return False
    base = "" if relative_to_root else posixpath.dirname(member_name)
    combined = posixpath.normpath(posixpath.join(base, linkname))
    return combined != ".." and not combined.startswith("../")


def _permissions(mode: int) -> int | None:
    """The permission bits worth keeping from an archive's recorded mode.

    Owner-write is added back: a wheel that ships a read-only file left a
    tree that could not afterwards be pruned or packed. Nothing is made more
    permissive than the archive asked for otherwise, and the executable bit -
    which a package's own helper programs need - is preserved.
    """

    permissions = mode & 0o777
    return (permissions | 0o200) if permissions else None


def extract_tar(
    archive_path: Path,
    destination: Path,
    *,
    allow_links: bool = True,
    max_expanded: int = DEFAULT_MAX_EXPANDED,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> int:
    """Unpack a tar archive, member by member, refusing anything that escapes."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(archive_path, "r:*") as bundle:
        members = bundle.getmembers()
        if len(members) > max_members:
            raise ArchiveError(f"archive contains too many members: {archive_path}")

        # Everything is judged before anything is written. A hostile archive
        # that fails halfway through would otherwise leave what it managed to
        # place, which is most of what it wanted.
        planned: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        total = 0
        for member in members:
            relative = safe_relative(member.name)
            if relative is None:
                raise ArchiveError(
                    f"archive member escapes its root: {member.name!r}"
                )
            if member.issym() or member.islnk():
                if not allow_links:
                    raise ArchiveError(
                        f"archive contains a link: {member.name!r}"
                    )
                if not _link_stays_inside(
                    member.name, member.linkname, relative_to_root=member.islnk()
                ):
                    raise ArchiveError(
                        f"archive member links outside its root: "
                        f"{member.name!r} -> {member.linkname!r}"
                    )
            elif not (member.isfile() or member.isdir()):
                raise ArchiveError(
                    f"archive member is a device or special file: {member.name!r}"
                )
            if member.isfile():
                total += member.size
                if total > max_expanded:
                    raise ArchiveError(
                        f"archive expands past the size limit: {archive_path}"
                    )
            planned.append((member, relative))

        # Directories first, so a file never has to guess whether its parent
        # was a directory the archive described or one invented for it.
        for member, relative in planned:
            if member.isdir():
                (destination / relative).mkdir(parents=True, exist_ok=True)

        for member, relative in planned:
            target = destination / relative
            if member.isdir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.exists():
                # An archive naming the same path twice would otherwise write
                # through whatever the first one left, link included.
                target.unlink()
            if member.issym():
                os.symlink(member.linkname, target)
                written += 1
                continue
            if member.islnk():
                source_path = destination / PurePosixPath(member.linkname)
                if not source_path.exists():
                    raise ArchiveError(
                        f"archive hard-links a member it does not contain: "
                        f"{member.name!r}"
                    )
                os.link(source_path, target)
                written += 1
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise ArchiveError(f"cannot read archive member: {member.name!r}")
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
            permissions = _permissions(member.mode)
            if permissions:
                target.chmod(permissions)
            written += 1
    return written


def extract_zip(
    archive_path: Path,
    destination: Path,
    *,
    allow_links: bool = False,
    max_expanded: int = DEFAULT_MAX_EXPANDED,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> int:
    """Unpack a ZIP, refusing traversal paths, links and special files.

    Links default to refused here where they default to allowed for tar,
    because the archives that need them are tar - a wheel is a ZIP and the
    format's own tooling does not make links on extraction either.
    """

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > max_members:
            raise ArchiveError(f"archive contains too many members: {archive_path}")

        planned: list[tuple[zipfile.ZipInfo, PurePosixPath, int]] = []
        total = 0
        for member in members:
            if member.is_dir():
                continue
            relative = safe_relative(member.filename)
            if relative is None:
                raise ArchiveError(
                    f"archive member escapes its root: {member.filename!r}"
                )
            unix_mode = (member.external_attr >> 16) & 0xFFFF
            if unix_mode and not stat.S_ISREG(unix_mode) and unix_mode & 0xF000:
                if not (allow_links and stat.S_ISLNK(unix_mode)):
                    raise ArchiveError(
                        f"archive member is not a regular file: "
                        f"{member.filename!r}"
                    )
            total += member.file_size
            if total > max_expanded:
                raise ArchiveError(
                    f"archive expands past the size limit: {archive_path}"
                )
            planned.append((member, relative, unix_mode))

        for member, relative, unix_mode in planned:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
            permissions = _permissions(unix_mode)
            if permissions:
                target.chmod(permissions)
            written += 1
    return written
