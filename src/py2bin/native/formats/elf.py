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


# --- dynamically linked images -------------------------------------------
#
# The static writer above needs nothing from the system. An image that drives
# CPython needs the interpreter's exports, which means asking the loader for
# them: an interpreter path for the kernel to start, a symbol table for the
# loader to search, a GOT for it to fill, and relocations saying which slot
# holds which symbol. That is the ELF counterpart of the Mach-O __got and its
# bind opcodes, and the shape of the calls in the code is identical - the
# encoders already emit an indirect call through a slot.

import struct as _struct

_PAGE = 0x1000
_BASE = 0x400000

#: Where each architecture's loader lives. The kernel reads this path out of
#: PT_INTERP and runs it; getting it wrong is an exec failure, not a link one.
_INTERPRETERS = {
    "x86_64": b"/lib64/ld-linux-x86-64.so.2\0",
    "arm64": b"/lib/ld-linux-aarch64.so.1\0",
}

#: R_*_GLOB_DAT: "put this symbol's address in this slot". The relocation a
#: GOT entry wants, on both architectures.
_GLOB_DAT = {"x86_64": 6, "arm64": 1025}

_MACHINE = {"x86_64": 0x3E, "arm64": 0xB7}


def _elf_hash_table(count: int) -> bytes:
    """A DT_HASH table the loader will accept.

    Nothing looks this image's symbols up - it exports none - but glibc wants
    a hash table present, so this is the smallest well-formed one: a single
    bucket, and a chain with an entry per symbol.
    """
    return _struct.pack("<II", 1, count) + _struct.pack("<I", 0) + b"\0" * 4 * count


def write_elf_dynamic(
    architecture: str,
    encode: "callable",
    static_bytes: int,
    needed: "tuple[str, ...]",
) -> bytes:
    """An ELF64 executable that imports its symbols from shared libraries.

    ``encode`` is called with the address `.text` will load at, and answers the
    code together with its call and static sites. It is called twice, because
    where `.text` lands depends on how large the symbol tables are, and those
    depend on which symbols the code calls - while the code's own length does
    not depend on the address, every reference being a fixed-size instruction.
    Encoding once at a guessed address and laying the file out around it would
    leave every PC-relative reference in the code pointing somewhere else.
    """

    if architecture not in _MACHINE:
        raise ValueError(f"no ELF writer for {architecture}")
    code, externs, statics = encode(_BASE + _PAGE)
    symbols = sorted({symbol for _offset, symbol in externs})
    slot_of = {symbol: index for index, symbol in enumerate(symbols)}

    interp = _INTERPRETERS[architecture]
    # dynstr holds every name the loader reads: the libraries and the symbols.
    dynstr = bytearray(b"\0")
    offsets: "dict[str, int]" = {}
    for name in (*needed, *symbols):
        offsets[name] = len(dynstr)
        dynstr += name.encode() + b"\0"

    # dynsym: index 0 is the reserved null entry, then one undefined global
    # per symbol, which is what makes the loader go looking for it.
    dynsym = bytearray(24)
    for symbol in symbols:
        dynsym += _struct.pack(
            "<IBBHQQ", offsets[symbol], (1 << 4) | 2, 0, 0, 0, 0  # GLOBAL, FUNC, UNDEF
        )
    hashes = _elf_hash_table(len(symbols) + 1)

    header_size = 64
    phnum = 4  # PHDR is not emitted; INTERP, two LOADs, DYNAMIC
    phoff = header_size
    cursor = phoff + phnum * 56
    interp_off = cursor
    cursor += len(interp)
    cursor = (cursor + 7) & ~7
    hash_off, cursor = cursor, cursor + len(hashes)
    cursor = (cursor + 7) & ~7
    dynsym_off, cursor = cursor, cursor + len(dynsym)
    dynstr_off, cursor = cursor, cursor + len(dynstr)
    cursor = (cursor + 7) & ~7
    rela_off = cursor
    rela_size = len(symbols) * 24
    cursor += rela_size
    cursor = (cursor + 15) & ~15
    text_off = cursor
    text_addr = _BASE + text_off
    # Now that the address is known, encode against it. The layout above is
    # settled: it depends on the symbols, and those do not change with where
    # the code sits.
    code, externs, statics = encode(text_addr)
    if sorted({symbol for _offset, symbol in externs}) != symbols:
        raise AssertionError("the symbol set changed when the code moved")

    # The writable half starts on its own page: a segment cannot be both
    # executable and writable, and the loader maps whole pages.
    data_off = (text_off + len(code) + _PAGE - 1) & ~(_PAGE - 1)
    data_addr = _BASE + data_off
    got_size = len(symbols) * 8
    statics_addr = data_addr + ((got_size + 15) & ~15)
    dynamic_off = data_off + ((got_size + 15) & ~15) + static_bytes
    dynamic_off = (dynamic_off + 7) & ~7
    dynamic_addr = _BASE + dynamic_off

    relocations = bytearray()
    for symbol in symbols:
        relocations += _struct.pack(
            "<QQq",
            data_addr + slot_of[symbol] * 8,
            (_struct.unpack("<Q", _struct.pack("<Q", (slot_of[symbol] + 1)))[0] << 32)
            | _GLOB_DAT[architecture],
            0,
        )

    entries = [(1, offsets[name]) for name in needed]  # DT_NEEDED
    entries += [
        (4, _BASE + hash_off),        # DT_HASH
        (5, _BASE + dynstr_off),      # DT_STRTAB
        (10, len(dynstr)),            # DT_STRSZ
        (6, _BASE + dynsym_off),      # DT_SYMTAB
        (11, 24),                     # DT_SYMENT
        (7, _BASE + rela_off),        # DT_RELA
        (8, rela_size),               # DT_RELASZ
        (9, 24),                      # DT_RELAENT
        (0, 0),                       # DT_NULL
    ]
    dynamic = b"".join(_struct.pack("<Qq", tag, value) for tag, value in entries)

    patched = bytearray(code)
    _patch_indirect(architecture, patched, text_addr, externs, data_addr, slot_of)
    _patch_statics(architecture, patched, text_addr, statics, statics_addr)

    image = bytearray()
    image += _struct.pack(
        "<4sBBBBB7sHHIQQQIHHHHHH",
        b"\x7fELF", 2, 1, 1, 0, 0, b"\0" * 7,
        2,                              # ET_EXEC
        _MACHINE[architecture],
        1, text_addr, phoff, 0, 0,
        header_size, 56, phnum, 64, 0, 0,
    )
    file_end = dynamic_off + len(dynamic)
    load_one_size = text_off + len(code)
    write_size = file_end - data_off
    memory_size = write_size
    for kind, offset, address, filesz, memsz, flags, align in (
        (3, interp_off, _BASE + interp_off, len(interp), len(interp), 4, 1),
        (1, 0, _BASE, load_one_size, load_one_size, 5, _PAGE),
        (1, data_off, data_addr, write_size, memory_size, 6, _PAGE),
        (2, dynamic_off, dynamic_addr, len(dynamic), len(dynamic), 6, 8),
    ):
        image += _struct.pack(
            "<IIQQQQQQ", kind, flags, offset, address, address, filesz, memsz, align
        )
    def place(target: int, payload: bytes) -> None:
        image.extend(b"\0" * (target - len(image)))
        image.extend(payload)

    place(interp_off, interp)
    place(hash_off, hashes)
    place(dynsym_off, bytes(dynsym))
    place(dynstr_off, bytes(dynstr))
    place(rela_off, bytes(relocations))
    place(text_off, bytes(patched))
    place(data_off, b"\0" * (((got_size + 15) & ~15) + static_bytes))
    place(dynamic_off, dynamic)
    return bytes(image)


def _patch_indirect(architecture, code, text_addr, externs, got_addr, slot_of) -> None:
    """Point each call site at its GOT slot, the way the Mach-O writer does."""
    for byte_offset, symbol in externs:
        target = got_addr + slot_of[symbol] * 8
        here = text_addr + byte_offset
        if architecture == "x86_64":
            delta = target - (here + 6)
            if not -(1 << 31) <= delta < (1 << 31):
                raise ValueError("x86-64 GOT reference is out of rip-relative range")
            _struct.pack_into("<i", code, byte_offset + 2, delta)
            continue
        pages = ((target & ~0xFFF) - (here & ~0xFFF)) >> 12
        if not -(1 << 20) <= pages < (1 << 20):
            raise ValueError("arm64 GOT reference is outside ADRP range")
        encoded = pages & 0x1FFFFF
        adrp = 0x90000010 | ((encoded & 3) << 29) | (((encoded >> 2) & 0x7FFFF) << 5)
        offset = target & 0xFFF
        if offset & 7:
            raise ValueError("arm64 GOT slot is not 8-byte aligned")
        _struct.pack_into("<II", code, byte_offset, adrp, 0xF9400210 | ((offset // 8) << 10))


def _patch_statics(architecture, code, text_addr, statics, statics_addr) -> None:
    for byte_offset, static_offset in statics:
        target = statics_addr + static_offset
        here = text_addr + byte_offset
        if architecture == "x86_64":
            delta = target - (here + 4)
            if not -(1 << 31) <= delta < (1 << 31):
                raise ValueError("x86-64 static reference is out of rip-relative range")
            _struct.pack_into("<i", code, byte_offset, delta)
            continue
        pages = ((target & ~0xFFF) - (here & ~0xFFF)) >> 12
        if not -(1 << 20) <= pages < (1 << 20):
            raise ValueError("arm64 static reference is outside ADRP range")
        encoded = pages & 0x1FFFFF
        adrp = 0x90000000 | ((encoded & 3) << 29) | (((encoded >> 2) & 0x7FFFF) << 5)
        _struct.pack_into(
            "<II", code, byte_offset, adrp, 0x91000000 | ((target & 0xFFF) << 10)
        )
