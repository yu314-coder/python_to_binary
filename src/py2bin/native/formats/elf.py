from __future__ import annotations

import struct


def write_elf(code: bytes, machine: int) -> bytes:
    """Return a static 64-bit little-endian ELF with one executable segment."""
    page = 0x1000
    base = 0x400000
    entry = base + page
    image_size = page + len(code)
    identification = b"\x7fELF" + bytes((2, 1, 1, 0)) + bytes(8)
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        identification,
        2,  # ET_EXEC
        machine,
        1,
        entry,
        64,
        0,
        0,
        64,
        56,
        1,
        0,
        0,
        0,
    )
    program_header = struct.pack(
        "<IIQQQQQQ",
        1,  # PT_LOAD
        5,  # PF_R | PF_X
        0,
        base,
        base,
        image_size,
        image_size,
        page,
    )
    return header + program_header + bytes(page - len(header) - len(program_header)) + code


def write_elf_x86_64(code: bytes) -> bytes:
    return write_elf(code, 62)


def write_elf_arm64(code: bytes) -> bytes:
    return write_elf(code, 183)
