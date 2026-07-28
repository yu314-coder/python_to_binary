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


def _uleb(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def write_macho_arm64_dynamic(
    code: bytes,
    externs: list[tuple[int, str]],
    statics: list[tuple[int, int]] = (),
    static_bytes: int = 0,
    info_plist: bytes | None = None,
    code_resources: bytes | None = None,
    libraries: tuple[str, ...] = ("/usr/lib/libSystem.B.dylib",),
    symbol_libraries: dict[str, str] | None = None,
) -> bytes:
    """Return an arm64 Mach-O that binds external symbols through dyld.

    ``code`` is the ``.text`` produced by ``encode_darwin_extern`` and mapped at
    ``base + page``. ``externs`` lists ``(byte_offset_in_code, symbol)`` GOT call
    sites, where each site is ``adrp x16 / ldr x16,[x16,#off] / blr x16``. Every
    unique symbol gets one 8-byte ``__got`` slot in a writable ``__DATA``
    segment; classic ``LC_DYLD_INFO_ONLY`` bind opcodes tell dyld to store the
    resolved libSystem address there before the entry point runs. This is real
    dynamic linking, not a translation of the callee's source.
    """
    page = 0x4000
    base = 0x100000000
    code_fileoff = page
    code_vmaddr = base + code_fileoff

    # Order symbols by first reference so GOT slot indices are deterministic.
    symbols: list[str] = []
    for _, symbol in externs:
        if symbol not in symbols:
            symbols.append(symbol)
    slot_of = {symbol: index for index, symbol in enumerate(symbols)}

    # Two-level namespace: every undefined symbol names the library that
    # provides it. Ordinals are 1-based in the order of LC_LOAD_DYLIB.
    symbol_libraries = symbol_libraries or {}
    ordinal_of_library = {name: index + 1 for index, name in enumerate(libraries)}
    default_ordinal = 1

    def ordinal(symbol: str) -> int:
        library = symbol_libraries.get(symbol)
        if library is None:
            return default_ordinal
        try:
            return ordinal_of_library[library]
        except KeyError as error:
            raise ValueError(
                f"symbol {symbol!r} names library {library!r}, which is not loaded"
            ) from error

    code = bytearray(code)
    text_end = code_fileoff + len(code)
    text_filesize = _align(text_end, page)

    data_fileoff = text_filesize
    data_vmaddr = base + data_fileoff
    got_size = len(symbols) * 8
    # Static storage follows the GOT in the same writable segment. Putting it
    # in the image rather than in a runtime mapping is what lets a static be
    # addressed PC-relatively, which is what a function called back from a
    # foreign library needs: it has no register it can trust to hold a base.
    # The bytes are zeros, which is the initial value C gives a static.
    static_base = data_vmaddr + _align(got_size, 16)
    data_used = _align(got_size, 16) + static_bytes if static_bytes else got_size
    data_filesize = _align(data_used, page) if data_used else page

    # Patch each GOT call site now that the GOT address is fixed.
    for byte_offset, symbol in externs:
        instruction_addr = code_vmaddr + byte_offset
        target = data_vmaddr + slot_of[symbol] * 8
        page_delta = (target & ~0xFFF) - (instruction_addr & ~0xFFF)
        pages = page_delta >> 12
        if not -(1 << 20) <= pages < (1 << 20):
            raise ValueError("arm64 __got reference is outside ADRP range")
        encoded = pages & 0x1FFFFF
        adrp = 0x90000010 | ((encoded & 3) << 29) | (((encoded >> 2) & 0x7FFFF) << 5)
        offset = target & 0xFFF
        if offset & 7:
            raise ValueError("arm64 __got slot is not 8-byte aligned")
        ldr = 0xF9400210 | ((offset // 8) << 10)
        struct.pack_into("<II", code, byte_offset, adrp, ldr)

    # Patch each static address now that __DATA is fixed: adrp x0, <page>
    # then add x0, x0, #<offset in page>.
    for byte_offset, static_offset in statics:
        instruction_addr = code_vmaddr + byte_offset
        target = static_base + static_offset
        page_delta = (target & ~0xFFF) - (instruction_addr & ~0xFFF)
        pages = page_delta >> 12
        if not -(1 << 20) <= pages < (1 << 20):
            raise ValueError("arm64 static reference is outside ADRP range")
        encoded = pages & 0x1FFFFF
        adrp = 0x90000000 | ((encoded & 3) << 29) | (((encoded >> 2) & 0x7FFFF) << 5)
        add = 0x91000000 | ((target & 0xFFF) << 10)
        struct.pack_into("<II", code, byte_offset, adrp, add)

    linkedit_fileoff = data_fileoff + data_filesize

    # __DATA is segment index 2 (PAGEZERO=0, TEXT=1, DATA=2, LINKEDIT=3).
    bind = bytearray()
    for index, symbol in enumerate(symbols):
        bind.append(0x10 | (ordinal(symbol) & 0x0F))  # SET_DYLIB_ORDINAL_IMM
        bind.append(0x40)  # SET_SYMBOL_TRAILING_FLAGS_IMM, flags 0
        bind += b"_" + symbol.encode("ascii") + b"\0"
        bind.append(0x50 | 1)  # SET_TYPE_IMM, BIND_TYPE_POINTER
        bind.append(0x70 | 2)  # SET_SEGMENT_AND_OFFSET_ULEB, __DATA
        bind += _uleb(index * 8)
        bind.append(0x90)  # DO_BIND
    bind.append(0x00)  # DONE
    while len(bind) % 8:
        bind.append(0x00)
    bind_off = linkedit_fileoff
    bind_size = len(bind)

    string_table = bytearray(b"\0")
    symbol_table = bytearray()
    for symbol in symbols:
        name_offset = len(string_table)
        string_table += b"_" + symbol.encode("ascii") + b"\0"
        # nlist_64: undefined external; n_desc carries the library ordinal in
        # its high byte for the two-level namespace.
        symbol_table += struct.pack(
            "<IBBHQ", name_offset, 0x01, 0, (ordinal(symbol) & 0xFF) << 8, 0
        )
    while len(string_table) % 8:
        string_table.append(0)
    symbol_table_off = bind_off + bind_size
    indirect_off = symbol_table_off + len(symbol_table)
    indirect = b"".join(struct.pack("<I", index) for index in range(len(symbols)))
    string_table_off = indirect_off + len(indirect)
    linkedit_used = bind_size + len(symbol_table) + len(indirect) + len(string_table)
    signature_offset = _align(linkedit_fileoff + linkedit_used, 16)

    def sign(image: bytes) -> bytes:
        return _adhoc_signature(
            image,
            signature_offset,
            code_fileoff,
            len(code),
            info_plist,
            code_resources,
            page_size=page,
        )

    placeholder = sign(bytes(signature_offset))
    linkedit_filesize = (signature_offset - linkedit_fileoff) + len(placeholder)
    linkedit_vmsize = _align(linkedit_filesize, page)

    def segment(name, vmaddr, vmsize, fileoff, filesize, maxp, initp, nsects, flags):
        return struct.pack(
            "<II16sQQQQiiII",
            0x19,
            72 + nsects * 80,
            _name(name),
            vmaddr,
            vmsize,
            fileoff,
            filesize,
            maxp,
            initp,
            nsects,
            flags,
        )

    def section(sname, segn, addr, size, offset, align_pow2, flags, reserved1):
        return struct.pack(
            "<16s16sQQIIIIIIII",
            _name(sname),
            _name(segn),
            addr,
            size,
            offset,
            align_pow2,
            0,
            0,
            flags,
            reserved1,
            0,
            0,
        )

    pagezero = segment("__PAGEZERO", 0, base, 0, 0, 0, 0, 0, 0)
    text_segment = segment(
        "__TEXT", base, text_filesize, 0, text_filesize, 5, 5, 1, 0
    ) + section(
        "__text", "__TEXT", code_vmaddr, len(code), code_fileoff, 2, 0x80000400, 0
    )
    data_segment = segment(
        "__DATA", data_vmaddr, data_filesize, data_fileoff, data_filesize, 3, 3, 1, 0
    ) + section(
        # S_NON_LAZY_SYMBOL_POINTERS, reserved1 = indirect symbol table index.
        "__got", "__DATA", data_vmaddr, got_size, data_fileoff, 3, 0x00000006, 0
    )
    linkedit_segment = segment(
        "__LINKEDIT",
        base + linkedit_fileoff,
        linkedit_vmsize,
        linkedit_fileoff,
        linkedit_filesize,
        1,
        1,
        0,
        0,
    )
    dyld_info = struct.pack(
        "<12I", 0x80000022, 48, 0, 0, bind_off, bind_size, 0, 0, 0, 0, 0, 0
    )
    symtab = struct.pack(
        "<IIIIII", 0x2, 24, symbol_table_off, len(symbols), string_table_off, len(string_table)
    )
    dysymtab = struct.pack(
        "<20I",
        0xB,
        80,
        0,
        0,  # local
        0,
        0,  # external defined
        0,
        len(symbols),  # undefined
        0,
        0,  # toc
        0,
        0,  # module table
        0,
        0,  # external reference symbols
        indirect_off,
        len(symbols),  # indirect symbols
        0,
        0,
        0,
        0,
    )
    dylinker = struct.pack("<III", 0xE, 32, 12) + b"/usr/lib/dyld\0".ljust(20, b"\0")
    build_version = struct.pack("<IIIIII", 0x32, 24, 1, 13 << 16, 0, 0)
    uuid_command = struct.pack("<II", 0x1B, 24) + hashlib.sha256(code).digest()[:16]
    main_command = struct.pack("<IIQQ", 0x80000028, 24, code_fileoff, 0)

    def _load_dylib(path: str) -> bytes:
        # LC_LOAD_DYLIB: fixed 24-byte header then the NUL-terminated path,
        # padded so the whole command is 8-byte aligned.
        raw = path.encode("utf-8") + b"\0"
        size = _align(24 + len(raw), 8)
        return (
            struct.pack("<IIIIII", 0xC, size, 24, 2, 0x054C0000, 0x00010000)
            + raw.ljust(size - 24, b"\0")
        )

    load_dylib = b"".join(_load_dylib(path) for path in libraries)
    signature_command = struct.pack(
        "<IIII", 0x1D, 16, signature_offset, len(placeholder)
    )
    commands = (
        pagezero
        + text_segment
        + data_segment
        + linkedit_segment
        + dyld_info
        + symtab
        + dysymtab
        + dylinker
        + uuid_command
        + build_version
        + main_command
        + load_dylib
        + signature_command
    )
    # MH_DYLDLINK | MH_TWOLEVEL | MH_PIE. NOT MH_NOUNDEFS: the image imports.
    # Twelve fixed commands plus one LC_LOAD_DYLIB per loaded library.
    command_count = 12 + len(libraries)
    header = struct.pack(
        "<IIIIIIII",
        0xFEEDFACF,
        0x0100000C,
        0,
        2,
        command_count,
        len(commands),
        0x00200084,
        0,
    )
    if len(header) + len(commands) > page:
        raise ValueError("Mach-O load commands exceed header page")

    image = bytearray(header + commands)
    image += bytes(code_fileoff - len(image))
    image += code
    image += bytes(data_fileoff - len(image))
    image += bytes(data_used)
    image += bytes(linkedit_fileoff - len(image))
    image += bind
    image += symbol_table
    image += indirect
    image += string_table
    image += bytes(signature_offset - len(image))
    signature = sign(bytes(image))
    if len(signature) != len(placeholder):
        raise AssertionError("Mach-O signature sizing changed during finalization")
    return bytes(image + signature)


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
    # Tell dyld explicitly that this image has no rebases, binds, or exports.
    # Older dyld releases otherwise fall back to legacy relocation metadata
    # that this intentionally minimal image does not contain.
    dyld_info = struct.pack("<12I", 0x80000022, 48, *([0] * 10))
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
        + dyld_info
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
        10,
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
