from __future__ import annotations

import struct

from ..ir import Module
from ..arm64 import encode_windows as encode_windows_arm64
from ..x86_64 import encode_windows


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _imports(section_rva: int, image_base: int) -> tuple[bytes, dict[str, int], int, int]:
    symbols = ("GetStdHandle", "WriteFile", "ExitProcess")
    descriptor_size = 40
    lookup_offset = _align(descriptor_size, 8)
    lookup_size = (len(symbols) + 1) * 8
    iat_offset = lookup_offset + lookup_size
    dll_offset = iat_offset + lookup_size
    data = bytearray(dll_offset)
    data.extend(b"KERNEL32.dll\0")
    if len(data) & 1:
        data.append(0)
    name_offsets: dict[str, int] = {}
    for symbol in symbols:
        name_offsets[symbol] = len(data)
        data.extend(b"\0\0" + symbol.encode("ascii") + b"\0")
        if len(data) & 1:
            data.append(0)
    for index, symbol in enumerate(symbols):
        name_rva = section_rva + name_offsets[symbol]
        struct.pack_into("<Q", data, lookup_offset + index * 8, name_rva)
        struct.pack_into("<Q", data, iat_offset + index * 8, name_rva)
    struct.pack_into(
        "<IIIII",
        data,
        0,
        section_rva + lookup_offset,
        0,
        0,
        section_rva + dll_offset,
        section_rva + iat_offset,
    )
    addresses = {
        symbol: image_base + section_rva + iat_offset + index * 8
        for index, symbol in enumerate(symbols)
    }
    return bytes(data), addresses, iat_offset, lookup_size


def _write_pe(module: Module, machine: int, arm64: bool) -> bytes:
    image_base = 0x140000000
    section_alignment = 0x1000
    file_alignment = 0x200
    text_rva, rdata_rva = 0x1000, 0x2000
    rdata, imports, iat_offset, iat_size = _imports(rdata_rva, image_base)
    encoder = encode_windows_arm64 if arm64 else encode_windows
    code = encoder(module, image_base + text_rva, imports)
    text_raw_size = _align(len(code), file_alignment)
    rdata_raw_size = _align(len(rdata), file_alignment)
    headers_size = file_alignment
    text_file_offset = headers_size
    rdata_file_offset = text_file_offset + text_raw_size
    image_size = _align(rdata_rva + len(rdata), section_alignment)

    dos = bytearray(0x80)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x80)
    coff = struct.pack("<HHIIIHH", machine, 2, 0, 0, 0, 240, 0x0022)
    optional = bytearray(240)
    struct.pack_into("<HBB", optional, 0, 0x20B, 1, 0)
    struct.pack_into("<III", optional, 4, text_raw_size, rdata_raw_size, 0)
    struct.pack_into("<II", optional, 16, text_rva, text_rva)
    struct.pack_into("<Q", optional, 24, image_base)
    struct.pack_into("<II", optional, 32, section_alignment, file_alignment)
    struct.pack_into("<HHHHHH", optional, 40, 6, 0, 0, 1, 6, 0)
    struct.pack_into("<III", optional, 52, 0, image_size, headers_size)
    struct.pack_into("<IHH", optional, 64, 0, 3, 0x8160)
    struct.pack_into("<QQQQ", optional, 72, 0x100000, 0x1000, 0x100000, 0x1000)
    struct.pack_into("<II", optional, 104, 0, 16)
    struct.pack_into("<II", optional, 120, rdata_rva, 40)
    struct.pack_into("<II", optional, 208, rdata_rva + iat_offset, iat_size)

    def section(name: bytes, virtual_size: int, rva: int, raw_size: int, raw_offset: int, flags: int) -> bytes:
        return struct.pack(
            "<8sIIIIIIHHI",
            name.ljust(8, b"\0"), virtual_size, rva, raw_size, raw_offset, 0, 0, 0, 0, flags,
        )

    sections = section(b".text", len(code), text_rva, text_raw_size, text_file_offset, 0x60000020)
    sections += section(b".rdata", len(rdata), rdata_rva, rdata_raw_size, rdata_file_offset, 0x40000040)
    headers = bytes(dos) + b"PE\0\0" + coff + bytes(optional) + sections
    headers += bytes(headers_size - len(headers))
    return headers + code.ljust(text_raw_size, b"\0") + rdata.ljust(rdata_raw_size, b"\0")


def write_pe_x86_64(module: Module) -> bytes:
    """Write a Windows x86-64 PE32+ console executable."""
    return _write_pe(module, 0x8664, arm64=False)


def write_pe_arm64(module: Module) -> bytes:
    """Write a Windows ARM64 PE32+ console executable."""
    return _write_pe(module, 0xAA64, arm64=True)
