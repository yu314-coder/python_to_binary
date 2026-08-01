"""Write a mountable disk image, without asking the system for help.

macOS is the only target that wants a .dmg, and the obvious way to make one is
to run hdiutil. This library cannot: nothing under src/ may reach for a
subprocess, which is the rule that keeps "the only requirement is Python"
true. So the filesystem is written here, byte by byte.

The filesystem is ISO 9660 with Joliet. Two reasons it fits rather than HFS+,
which is what hdiutil would produce: it is simple enough to write correctly -
no catalog B-tree, no allocation bitmap, no extents - and macOS mounts files
from it executable, which is what an .app needs to launch. Read-only is not a
limitation for something whose purpose is to be dragged to /Applications.

Plain ISO 9660 allows eight characters, a dot and three more, which no real
bundle survives, so every name is carried twice: mangled into that shape for
the primary descriptor, and in full UCS-2 in the Joliet one. macOS reads the
Joliet tree and shows the real names.
"""

from __future__ import annotations

import plistlib
import struct
import tempfile
from pathlib import Path

SECTOR = 2048

#: Sectors reserved before the first volume descriptor. Nothing reads them.
_SYSTEM_SECTORS = 16

_ALLOWED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class ImageError(Exception):
    """A disk image could not be written."""


def _both16(value: int) -> bytes:
    """A 16-bit number as ISO 9660 stores it: little-endian, then big."""
    return struct.pack("<H", value) + struct.pack(">H", value)


def _both32(value: int) -> bytes:
    """A 32-bit number as ISO 9660 stores it: little-endian, then big."""
    return struct.pack("<I", value) + struct.pack(">I", value)


def _sectors(size: int) -> int:
    return (size + SECTOR - 1) // SECTOR


def _mangle(name: str, is_directory: bool, taken: set[str]) -> str:
    """Force a name into the shape plain ISO 9660 permits, uniquely.

    The result is only ever read by software that ignores the Joliet tree, so
    it needs to be legal and distinct rather than recognisable.
    """
    upper = "".join(character if character.upper() in _ALLOWED else "_"
                    for character in name.upper())
    if is_directory:
        stem, extension = upper[:8], ""
    else:
        base, _, suffix = upper.rpartition(".")
        if not base:
            base, suffix = upper, ""
        stem, extension = base[:8] or "_", suffix[:3]
    candidate = f"{stem}.{extension}" if extension else stem
    if candidate in taken:
        for index in range(1, 10000):
            tail = f"~{index}"
            short = stem[: max(1, 8 - len(tail))] + tail
            candidate = f"{short}.{extension}" if extension else short
            if candidate not in taken:
                break
        else:
            raise ImageError(f"cannot find a unique short name for {name}")
    taken.add(candidate)
    return candidate


class _Node:
    """One file or directory in the tree being written."""

    __slots__ = ("path", "name", "short", "is_directory", "children",
                 "number", "lba", "size", "primary_size", "joliet_size")

    def __init__(self, path: Path, name: str, is_directory: bool):
        self.path = path
        self.name = name
        self.short = ""
        self.is_directory = is_directory
        self.children: list[_Node] = []
        self.number = 0
        self.lba = 0
        self.size = 0 if is_directory else path.stat().st_size
        self.primary_size = 0
        self.joliet_size = 0


def _build(root: Path) -> _Node:
    """Read the tree off disk, refusing what the format cannot represent."""
    node = _Node(root, "", True)

    def walk(directory: Path, parent: _Node) -> None:
        taken: set[str] = set()
        entries = sorted(directory.iterdir(), key=lambda item: item.name.upper())
        for entry in entries:
            if entry.is_symlink():
                # A symlink needs Rock Ridge, which this writer does not emit.
                # Bundles produced here contain none, so this is a guard
                # against a surprise rather than a routine case.
                raise ImageError(
                    f"cannot put a symlink in this image: {entry}"
                )
            child = _Node(entry, entry.name, entry.is_dir())
            child.short = _mangle(entry.name, child.is_directory, taken)
            parent.children.append(child)
            if child.is_directory:
                walk(entry, child)

    walk(root, node)
    return node


def _directories(node: _Node) -> list[_Node]:
    """Every directory, parents before children, as the path table wants."""
    found = [node]
    index = 0
    while index < len(found):
        current = found[index]
        found.extend(child for child in current.children if child.is_directory)
        index += 1
    return found


def _record(identifier: bytes, lba: int, size: int, is_directory: bool) -> bytes:
    """One directory record. Its length must be even, so it may be padded."""
    length = 33 + len(identifier)
    padding = length % 2
    record = bytearray()
    record += bytes([length + padding, 0])
    record += _both32(lba)
    record += _both32(size)
    record += bytes([126, 7, 30, 8, 0, 0, 0])  # a fixed, valid timestamp
    record += bytes([2 if is_directory else 0, 0, 0])
    record += _both16(1)
    record += bytes([len(identifier)])
    record += identifier
    if padding:
        record += b"\x00"
    return bytes(record)


def _identifier(node: _Node, joliet: bool) -> bytes:
    if joliet:
        name = node.name if node.is_directory else f"{node.name};1"
        return name.encode("utf-16-be")
    return node.short.encode() if node.is_directory else f"{node.short};1".encode()


def _directory_bytes(node: _Node, parent: _Node, joliet: bool) -> bytes:
    """The records for one directory, packed so none straddles a sector.

    A record split across a sector boundary is unreadable, so a record that
    would not fit is pushed to the next sector and the gap is left empty.
    """
    blocks = [
        _record(b"\x00", node.lba, node.primary_size if not joliet
                else node.joliet_size, True),
        _record(b"\x01", parent.lba, parent.primary_size if not joliet
                else parent.joliet_size, True),
    ]
    children = sorted(
        node.children,
        key=lambda child: _identifier(child, joliet),
    )
    for child in children:
        size = (child.joliet_size if joliet else child.primary_size) \
            if child.is_directory else child.size
        blocks.append(
            _record(_identifier(child, joliet), child.lba, size,
                    child.is_directory)
        )
    out = bytearray()
    for block in blocks:
        if len(out) % SECTOR + len(block) > SECTOR:
            out += b"\x00" * (SECTOR - len(out) % SECTOR)
        out += block
    return bytes(out)


def _path_table(directories: list[_Node], joliet: bool,
                big_endian: bool) -> bytes:
    """The path table, which lists every directory and names its parent."""
    parents = {}
    for directory in directories:
        for child in directory.children:
            if child.is_directory:
                parents[id(child)] = directory
    out = bytearray()
    for directory in directories:
        if directory.number == 1:
            identifier = b"\x00"
        else:
            identifier = _identifier(directory, joliet)
        padding = len(identifier) % 2
        out += bytes([len(identifier), 0])
        out += struct.pack(">I" if big_endian else "<I", directory.lba)
        parent = parents.get(id(directory), directory)
        out += struct.pack(">H" if big_endian else "<H", parent.number)
        out += identifier
        if padding:
            out += b"\x00"
    return bytes(out)


def _descriptor(kind: int, volume: str, total: int, path_table_size: int,
                path_left: int, path_right: int, root_record: bytes,
                joliet: bool) -> bytes:
    """A primary or supplementary volume descriptor."""
    block = bytearray(b"\x00" * SECTOR)
    block[0] = kind
    block[1:6] = b"CD001"
    block[6] = 1

    def text(value: str, width: int) -> bytes:
        if joliet:
            return value.encode("utf-16-be")[:width].ljust(width, b"\x00")
        return value.upper().encode("ascii", "replace")[:width].ljust(width)

    block[8:40] = text("PY2BIN", 32)
    block[40:72] = text(volume, 32)
    block[80:88] = _both32(total)
    if joliet:
        # The escape sequence that says these names are UCS-2.
        block[88:120] = b"%/E".ljust(32, b"\x00")
    block[120:124] = _both16(1)
    block[124:128] = _both16(1)
    block[128:132] = _both16(SECTOR)
    block[132:140] = _both32(path_table_size)
    struct.pack_into("<I", block, 140, path_left)
    struct.pack_into(">I", block, 148, path_right)
    block[156:156 + len(root_record)] = root_record
    for start, width in ((318, 128), (446, 128), (574, 128)):
        block[start:start + width] = text("PY2BIN", width)
    block[813:830] = b"2026073008000000\x00"
    block[830:847] = b"2026073008000000\x00"
    block[847:864] = b"0000000000000000\x00"
    block[864:881] = b"0000000000000000\x00"
    block[881] = 1
    return bytes(block)


def write_image(source: Path, output: Path, volume_name: str = "py2bin") -> int:
    """Write a mountable ISO 9660 + Joliet image of a directory.

    Returns the size of the image in bytes.
    """
    source = source.resolve()
    if not source.is_dir():
        raise ImageError(f"not a directory: {source}")
    tree = _build(source)
    directories = _directories(tree)
    for number, directory in enumerate(directories, start=1):
        directory.number = number

    # Two passes over the layout. The first only needs each directory's size,
    # which depends on the names in it and not on where anything lands; the
    # second hands out the addresses now that the sizes are known.
    for joliet in (False, True):
        for directory in directories:
            body = _directory_bytes(directory, directory, joliet)
            if joliet:
                directory.joliet_size = _sectors(len(body)) * SECTOR
            else:
                directory.primary_size = _sectors(len(body)) * SECTOR

    primary_table = _path_table(directories, False, False)
    joliet_table = _path_table(directories, True, False)
    primary_sectors = _sectors(len(primary_table))
    joliet_sectors = _sectors(len(joliet_table))

    lba = _SYSTEM_SECTORS + 3  # primary, supplementary, terminator
    primary_left, lba = lba, lba + primary_sectors
    primary_right, lba = lba, lba + primary_sectors
    joliet_left, lba = lba, lba + joliet_sectors
    joliet_right, lba = lba, lba + joliet_sectors

    for directory in directories:
        directory.lba = lba
        lba += directory.primary_size // SECTOR
    joliet_lba = {}
    for directory in directories:
        joliet_lba[id(directory)] = lba
        lba += directory.joliet_size // SECTOR

    files = [node for directory in directories for node in directory.children
             if not node.is_directory]
    for node in files:
        node.lba = lba
        lba += max(1, _sectors(node.size))
    total = lba

    image = bytearray(b"\x00" * (SECTOR * _SYSTEM_SECTORS))
    root_primary = _record(b"\x00", tree.lba, tree.primary_size, True)
    image += _descriptor(1, volume_name, total, len(primary_table),
                         primary_left, primary_right, root_primary, False)
    # The Joliet tree has its own root, at its own address.
    root_joliet = _record(b"\x00", joliet_lba[id(tree)], tree.joliet_size, True)
    image += _descriptor(2, volume_name, total, len(joliet_table),
                         joliet_left, joliet_right, root_joliet, True)
    terminator = bytearray(b"\x00" * SECTOR)
    terminator[0] = 255
    terminator[1:6] = b"CD001"
    terminator[6] = 1
    image += bytes(terminator)

    def pad(data: bytes) -> bytes:
        return data.ljust(_sectors(len(data)) * SECTOR, b"\x00")

    image += pad(primary_table)
    image += pad(_path_table(directories, False, True))
    image += pad(joliet_table)
    image += pad(_path_table(directories, True, True))

    parents = {}
    for directory in directories:
        for child in directory.children:
            if child.is_directory:
                parents[id(child)] = directory

    for directory in directories:
        parent = parents.get(id(directory), directory)
        body = _directory_bytes(directory, parent, False)
        image += body.ljust(directory.primary_size, b"\x00")
    # The Joliet records point at the same file data but their own directories.
    saved = {id(node): node.lba for node in directories}
    for directory in directories:
        directory.lba = joliet_lba[id(directory)]
    for directory in directories:
        parent = parents.get(id(directory), directory)
        body = _directory_bytes(directory, parent, True)
        image += body.ljust(directory.joliet_size, b"\x00")
    for directory in directories:
        directory.lba = saved[id(directory)]

    for node in files:
        payload = node.path.read_bytes()
        image += payload.ljust(max(1, _sectors(len(payload))) * SECTOR, b"\x00")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(image))
    return len(image)


# --- compressed images ----------------------------------------------------
#
# The image above is the filesystem laid out plainly, which macOS mounts and
# which is exactly as large as what it holds. Apple's compressed form (UDZO)
# wraps that same image: cut into chunks, each deflated, with a table saying
# where each one went and a trailer saying where the table is. Reads are
# inflated as the volume is used, so what is handed over is smaller while what
# it holds is unchanged.
#
# zlib is the standard library, so this needs nothing new.

import base64 as _base64
import zlib as _zlib

_UDIF_SECTOR = 512
#: One megabyte at a time. Larger chunks compress slightly better and cost
#: more to read a little of; this is what Apple's own tools use.
_CHUNK_SECTORS = 2048

_ZLIB_CHUNK = 0x80000005
_LAST_CHUNK = 0xFFFFFFFF


def _blkx(image: bytes) -> "tuple[bytes, bytes]":
    """Deflate the image chunk by chunk, and describe where each one went."""
    sectors = (len(image) + _UDIF_SECTOR - 1) // _UDIF_SECTOR
    compressed = bytearray()
    entries = bytearray()
    count = 0
    for start in range(0, sectors, _CHUNK_SECTORS):
        length = min(_CHUNK_SECTORS, sectors - start)
        piece = image[start * _UDIF_SECTOR:(start + length) * _UDIF_SECTOR]
        piece = piece.ljust(length * _UDIF_SECTOR, b"\0")
        packed = _zlib.compress(piece, 9)
        entries += struct.pack(
            ">IIQQQQ", _ZLIB_CHUNK, 0, start, length, len(compressed), len(packed)
        )
        compressed += packed
        count += 1
    entries += struct.pack(
        ">IIQQQQ", _LAST_CHUNK, 0, sectors, 0, len(compressed), 0
    )
    count += 1

    table = bytearray()
    table += b"mish" + struct.pack(">I", 1)
    table += struct.pack(">QQQ", 0, sectors, 0)
    table += struct.pack(">II", 0x00000208, 0)
    table += b"\0" * 24
    table += struct.pack(">II", 0, 0) + b"\0" * 128   # no checksum
    table += struct.pack(">I", count)
    table += entries
    return bytes(compressed), bytes(table)


def _koly(data_length: int, plist_offset: int, plist_length: int, sectors: int) -> bytes:
    """The trailer macOS reads first: where the table is and how big all this is."""
    trailer = bytearray()
    trailer += b"koly" + struct.pack(">III", 4, 512, 1)
    trailer += struct.pack(">QQQQQ", 0, 0, data_length, 0, 0)
    trailer += struct.pack(">II", 1, 1)
    trailer += b"\0" * 16                                   # segment id
    trailer += struct.pack(">II", 0, 0) + b"\0" * 128       # data fork checksum
    trailer += struct.pack(">QQ", plist_offset, plist_length)
    trailer += b"\0" * 120                                  # reserved
    trailer += struct.pack(">II", 0, 0) + b"\0" * 128       # master checksum
    trailer += struct.pack(">I", 2)                         # image variant
    trailer += struct.pack(">Q", sectors)
    trailer += b"\0" * 12
    return bytes(trailer).ljust(512, b"\0")


def compress_image(raw: bytes) -> bytes:
    """Wrap a plain disk image as a compressed one macOS can mount."""
    compressed, table = _blkx(raw)
    document = {
        "resource-fork": {
            "blkx": [
                {
                    "Attributes": "0x0050",
                    "Data": table,
                    "ID": "0",
                    "Name": "whole disk",
                }
            ]
        }
    }
    plist = plistlib.dumps(document, fmt=plistlib.FMT_XML)
    sectors = (len(raw) + _UDIF_SECTOR - 1) // _UDIF_SECTOR
    return (
        compressed
        + plist
        + _koly(len(compressed), len(compressed), len(plist), sectors)
    )


def write_compressed_image(
    source: Path, output: Path, volume_name: str = "py2bin"
) -> int:
    """Write a compressed disk image of a directory. Returns its size."""
    with tempfile.TemporaryDirectory() as scratch:
        plain = Path(scratch) / "plain.img"
        write_image(source, plain, volume_name)
        packed = compress_image(plain.read_bytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(packed)
    return len(packed)
