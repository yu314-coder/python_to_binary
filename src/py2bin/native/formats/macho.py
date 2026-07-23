from __future__ import annotations

import struct
import hashlib


def _name(value: str) -> bytes:
    return value.encode("ascii").ljust(16, b"\0")


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _adhoc_signature(
    image: bytes,
    code_limit: int,
    exec_base: int,
    exec_size: int,
    info_plist: bytes | None = None,
    code_resources: bytes | None = None,
    page_size: int = 4096,
) -> bytes:
    """Create an Apple embedded ad-hoc SHA-256 code signature SuperBlob."""
    identifier = b"local.py2bin.native\0"
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError("code-signature page size must be a power of two")
    page_exponent = page_size.bit_length() - 1
    slots = (code_limit + page_size - 1) // page_size
    directory_header_size = 88
    identifier_offset = directory_header_size
    requirements = struct.pack(">III", 0xFADE0C01, 12, 0)
    special_data = (
        {1: info_plist, 2: requirements, 3: code_resources}
        if info_plist is not None and code_resources is not None
        else {}
    )
    special_slots = max(special_data, default=0)
    special_offset = _align(identifier_offset + len(identifier), 4)
    hash_offset = special_offset + special_slots * 32
    directory_size = hash_offset + slots * 32
    directory = bytearray(directory_size)
    struct.pack_into(
        ">IIIIIIIII4BI",
        directory,
        0,
        0xFADE0C02,
        directory_size,
        0x20400,
        0x20002,  # CS_ADHOC | CS_LINKER_SIGNED
        hash_offset,
        identifier_offset,
        special_slots,
        slots,
        code_limit,
        32,
        2,
        0,
        page_exponent,
        0,
    )
    struct.pack_into(">II", directory, 44, 0, 0)
    struct.pack_into(">IQ", directory, 52, 0, 0)
    struct.pack_into(">QQQ", directory, 64, exec_base, exec_size, 1)
    directory[identifier_offset:identifier_offset + len(identifier)] = identifier
    for slot, data in special_data.items():
        if data is not None:
            start = hash_offset - slot * 32
            directory[start:start + 32] = hashlib.sha256(data).digest()
    for index in range(slots):
        block = image[index * page_size:min((index + 1) * page_size, code_limit)]
        digest = hashlib.sha256(block).digest()
        start = hash_offset + index * 32
        directory[start:start + 32] = digest
    blob_count = 2 if special_data else 1
    index_size = 12 + blob_count * 8
    directory_offset = index_size
    requirements_offset = directory_offset + len(directory)
    superblob_size = requirements_offset + (len(requirements) if special_data else 0)
    result = bytearray(struct.pack(">III", 0xFADE0CC0, superblob_size, blob_count))
    result.extend(struct.pack(">II", 0, directory_offset))
    if special_data:
        result.extend(struct.pack(">II", 2, requirements_offset))
    result.extend(directory)
    if special_data:
        result.extend(requirements)
    return bytes(result)


def write_macho_x86_64(code: bytes) -> bytes:
    """Return a static x86-64 Mach-O executable using LC_UNIXTHREAD."""
    page = 0x1000
    base = 0x100000000
    entry = base + page

    pagezero = struct.pack(
        "<II16sQQQQiiII",
        0x19, 72, _name("__PAGEZERO"), 0, base, 0, 0, 0, 0, 0, 0,
    )
    text_command_size = 72 + 80
    text_segment = struct.pack(
        "<II16sQQQQiiII",
        0x19,
        text_command_size,
        _name("__TEXT"),
        base,
        page + len(code),
        0,
        page + len(code),
        7,
        5,
        1,
        0,
    )
    text_section = struct.pack(
        "<16s16sQQIIIIIIII",
        _name("__text"),
        _name("__TEXT"),
        entry,
        len(code),
        page,
        4,
        0,
        0,
        0x80000400,
        0,
        0,
        0,
    )
    registers = [0] * 21
    registers[16] = entry  # rip
    registers[17] = 0x202  # rflags
    thread = struct.pack("<IIII", 0x5, 184, 4, 42) + struct.pack("<21Q", *registers)
    commands = pagezero + text_segment + text_section + thread
    header = struct.pack(
        "<IIIIIIII",
        0xFEEDFACF,
        0x01000007,  # CPU_TYPE_X86_64
        3,  # CPU_SUBTYPE_X86_64_ALL
        2,  # MH_EXECUTE
        3,
        len(commands),
        1,  # MH_NOUNDEFS
        0,
    )
    if len(header) + len(commands) > page:
        raise ValueError("Mach-O load commands exceed header page")
    return header + commands + bytes(page - len(header) - len(commands)) + code


def write_macho_arm64(
    code: bytes,
    info_plist: bytes | None = None,
    code_resources: bytes | None = None,
) -> bytes:
    """Return a static arm64 Mach-O executable using LC_UNIXTHREAD."""
    page = 0x4000
    base = 0x100000000
    entry = base + page
    signature_offset = _align(page + len(code), page)
    pagezero = struct.pack(
        "<II16sQQQQiiII",
        0x19, 72, _name("__PAGEZERO"), 0, base, 0, 0, 0, 0, 0, 0,
    )
    text_command_size = 72 + 80
    text_segment = struct.pack(
        "<II16sQQQQiiII",
        0x19, text_command_size, _name("__TEXT"), base, page + len(code),
        0, page + len(code), 7, 5, 1, 0,
    )
    text_section = struct.pack(
        "<16s16sQQIIIIIIII",
        _name("__text"), _name("__TEXT"), entry, len(code), page, 2,
        0, 0, 0x80000400, 0, 0, 0,
    )
    dylinker = struct.pack("<III", 0xE, 32, 12) + b"/usr/lib/dyld\0".ljust(20, b"\0")
    build_version = struct.pack("<IIIIII", 0x32, 24, 1, 13 << 16, 0, 0)
    uuid_command = struct.pack("<II", 0x1B, 24) + hashlib.sha256(code).digest()[:16]
    main_command = struct.pack("<IIQQ", 0x80000028, 24, page, 0)
    dylib_name = b"/usr/lib/libSystem.B.dylib\0"
    load_dylib = (
        struct.pack("<IIIIII", 0xC, 56, 24, 2, 0x054C0000, 0x00010000)
        + dylib_name.ljust(32, b"\0")
    )
    # Signature length depends on the number of pages before it, but not their
    # contents. Build a placeholder once to size the __LINKEDIT segment.
    placeholder_image = bytes(signature_offset)
    placeholder_signature = _adhoc_signature(
        placeholder_image,
        signature_offset,
        page,
        len(code),
        info_plist,
        code_resources,
        page_size=page,
    )
    linkedit = struct.pack(
        "<II16sQQQQiiII",
        0x19,
        72,
        _name("__LINKEDIT"),
        base + signature_offset,
        _align(len(placeholder_signature), page),
        signature_offset,
        len(placeholder_signature),
        7,
        1,
        0,
        0,
    )
    signature_command = struct.pack(
        "<IIII", 0x1D, 16, signature_offset, len(placeholder_signature)
    )
    commands = (
        pagezero
        + text_segment
        + text_section
        + linkedit
        + dylinker
        + uuid_command
        + build_version
        + main_command
        + load_dylib
        + signature_command
    )
    header = struct.pack(
        "<IIIIIIII",
        0xFEEDFACF,
        0x0100000C,
        0,
        2,
        9,
        len(commands),
        0x00200085,
        0,
    )
    if len(header) + len(commands) > page:
        raise ValueError("Mach-O load commands exceed header page")
    image = header + commands + bytes(page - len(header) - len(commands)) + code
    image += bytes(signature_offset - len(image))
    signature = _adhoc_signature(
        image,
        signature_offset,
        page,
        len(code),
        info_plist,
        code_resources,
        page_size=page,
    )
    if len(signature) != len(placeholder_signature):
        raise AssertionError("Mach-O signature sizing changed during finalization")
    return image + signature
