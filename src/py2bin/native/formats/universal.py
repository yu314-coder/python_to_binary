"""Wrap per-architecture Mach-O images in one universal ("fat") file.

A universal binary is not a merged program. It is the two programs, whole and
unaltered, laid end to end behind a small table saying which is which and where
each one starts. The kernel reads that table at exec and maps the slice its
processor can run.

Which is what makes this safe to do after the fact. Each slice keeps its own
ad-hoc signature, because a signature covers the bytes of the image it was
written over and knows nothing about the wrapper those bytes were later placed
in - the same fact that lets `freezer._thin_to_arm64` lift one slice out of
Apple's universal framework and end up with a file that is still validly
signed. Wrapping is that operation run backwards.

Alignment is the one thing that has to be right. Each slice must begin on a
page boundary for the architecture that will map it, and Apple Silicon's page
is 16 KB rather than 4 KB, so an arm64 slice is placed on 2**14. The alignment
is recorded per slice as a power of two, which is what the loader reads.
"""

from __future__ import annotations

import struct

from .macho import _DARWIN_ARCHITECTURES, _align

#: FAT_MAGIC, big-endian on disk whatever the host is. The 64-bit variant
#: (FAT_MAGIC_64, 0xCAFEBABF) exists for slices past 4 GB and is not written
#: here; a program that large is refused rather than silently truncated.
_FAT_MAGIC = 0xCAFEBABE
_FAT_ARCH_SIZE = 20
_HEADER_SIZE = 8

#: The page each slice starts on, as a power of two. 16 KB for both, including
#: x86-64, whose own page is 4 KB.
#:
#: This is not a rounding-up for tidiness. A code-signed x86-64 slice placed on
#: a 4 KB boundary is *killed at exec* on Apple Silicon - SIGKILL, before the
#: program prints anything - because the machine mapping it has 16 KB pages and
#: the slice does not begin on one. The signature verifies, `codesign` calls the
#: file valid, and the same bytes lifted back out to a file of their own run
#: perfectly; only in place, at the wrong offset, does it die. An unsigned slice
#: survives it, which is what makes the symptom so misleading.
#:
#: Apple's own universal2 builds agree: every slice of the python.org framework,
#: x86-64 included, is recorded at 2**14.
_SLICE_ALIGNMENT = {"x86_64": 14, "arm64": 14}

#: A slice offset is a 32-bit field, so the whole file has to stay inside 4 GB.
_LIMIT = 1 << 32


def write_universal(slices: "dict[str, bytes]") -> bytes:
    """One universal image from `{"arm64": image, "x86_64": image}`.

    Order follows `_SLICE_ALIGNMENT`, so the same input always gives the same
    bytes - a build that is reproducible per architecture stays reproducible
    once the two are joined.
    """

    if not slices:
        raise ValueError("a universal image needs at least one architecture")
    unknown = set(slices) - set(_DARWIN_ARCHITECTURES)
    if unknown:
        known = ", ".join(sorted(_DARWIN_ARCHITECTURES))
        raise ValueError(
            f"not a Darwin architecture: {', '.join(sorted(unknown))} "
            f"(this writer knows {known})"
        )
    ordered = [name for name in _SLICE_ALIGNMENT if name in slices]

    # Place every slice first, so the table can be written knowing where each
    # one landed. The table's own size depends only on how many there are.
    cursor = _HEADER_SIZE + _FAT_ARCH_SIZE * len(ordered)
    placed: list[tuple[str, int, bytes]] = []
    for name in ordered:
        image = slices[name]
        if not image:
            raise ValueError(f"the {name} slice is empty")
        alignment = 1 << _SLICE_ALIGNMENT[name]
        start = _align(cursor, alignment)
        placed.append((name, start, image))
        cursor = start + len(image)
    if cursor >= _LIMIT:
        raise ValueError(
            f"a universal image of {cursor} bytes passes the {_LIMIT}-byte "
            "ceiling a 32-bit fat header can address; the slices would have to "
            "be shipped separately, or written as FAT_MAGIC_64"
        )

    header = bytearray(struct.pack(">II", _FAT_MAGIC, len(placed)))
    for name, start, image in placed:
        architecture = _DARWIN_ARCHITECTURES[name]
        header.extend(
            struct.pack(
                ">iiIII",
                architecture["cputype"],
                architecture["cpusubtype"],
                start,
                len(image),
                _SLICE_ALIGNMENT[name],
            )
        )

    out = bytearray(header)
    for _name, start, image in placed:
        out.extend(b"\0" * (start - len(out)))
        out.extend(image)
    return bytes(out)


def read_universal(image: bytes) -> "dict[str, bytes]":
    """The inverse: every slice of a universal image, by architecture name.

    Returns an empty mapping for a thin image, which is how callers tell the
    two apart without parsing the header themselves.
    """

    if len(image) < _HEADER_SIZE:
        return {}
    magic, count = struct.unpack_from(">II", image, 0)
    wide = magic == 0xCAFEBABF
    if magic != _FAT_MAGIC and not wide:
        return {}
    by_cpu = {
        (architecture["cputype"], architecture["cpusubtype"]): name
        for name, architecture in _DARWIN_ARCHITECTURES.items()
    }
    found: dict[str, bytes] = {}
    entry = 32 if wide else _FAT_ARCH_SIZE
    for index in range(count):
        at = _HEADER_SIZE + index * entry
        if at + entry > len(image):
            break
        if wide:
            cputype, cpusubtype, start, size = struct.unpack_from(">iiQQ", image, at)
        else:
            cputype, cpusubtype, start, size = struct.unpack_from(">iiII", image, at)
        name = by_cpu.get((cputype, cpusubtype))
        if name is None:
            # An architecture this writer does not emit is still carried, so a
            # caller that only wants to know "is this fat" is not misled by it.
            name = f"cputype-{cputype:#x}"
        found[name] = image[start: start + size]
    return found
