from __future__ import annotations

import struct

from ..ir import Module
from ..arm64 import _adr, _mov, _sub_sp
from ..arm64 import encode_windows as encode_windows_arm64
from ..x86_64 import encode_windows


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _imports_for(
    section_rva: int,
    image_base: int,
    symbols: tuple[str, ...],
) -> tuple[bytes, dict[str, int], int, int]:
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


def _imports_from_libraries(
    section_rva: int,
    image_base: int,
    libraries: "list[tuple[str, tuple[str, ...]]]",
) -> tuple[bytes, dict[str, int], int, int]:
    """An import directory naming several DLLs rather than one.

    A program driving CPython imports from at least two: the kernel for the
    process services, and `pythonXY.dll` for the interpreter itself. The single
    descriptor the rest of this writer emits cannot say that - a descriptor
    names one DLL - so the table is built here as one descriptor per library,
    terminated by a zero descriptor, with every library's thunks in one
    contiguous IAT so the data directory can describe it with one range.

    Windows resolves a thunk by name at load time and writes the address into
    the IAT slot. That is all the binding there is: no relocations, no bind
    opcodes, and nothing for the program to do at startup.
    """

    descriptor_size = 20
    directory_size = descriptor_size * (len(libraries) + 1)
    offset = _align(directory_size, 8)
    # Lookup and address thunks, per library, each NULL-terminated.
    layout = []
    for name, symbols in libraries:
        lookup_offset = offset
        offset += (len(symbols) + 1) * 8
        layout.append([name, symbols, lookup_offset, 0])
    iat_start = offset
    for entry in layout:
        entry[3] = offset
        offset += (len(entry[1]) + 1) * 8
    iat_size = offset - iat_start
    data = bytearray(offset)
    name_offsets: dict[str, int] = {}
    library_name_offsets: dict[str, int] = {}
    for name, symbols, _lookup, _iat in layout:
        library_name_offsets[name] = len(data)
        data.extend(name.encode("ascii") + b"\0")
        if len(data) & 1:
            data.append(0)
        for symbol in symbols:
            name_offsets[(name, symbol)] = len(data)
            # IMAGE_IMPORT_BY_NAME: a 2-byte hint the loader may ignore, then
            # the name.
            data.extend(b"\0\0" + symbol.encode("ascii") + b"\0")
            if len(data) & 1:
                data.append(0)
    addresses: dict[str, int] = {}
    for index, (name, symbols, lookup_offset, iat_offset) in enumerate(layout):
        for position, symbol in enumerate(symbols):
            name_rva = section_rva + name_offsets[(name, symbol)]
            struct.pack_into("<Q", data, lookup_offset + position * 8, name_rva)
            struct.pack_into("<Q", data, iat_offset + position * 8, name_rva)
            addresses[symbol] = image_base + section_rva + iat_offset + position * 8
        struct.pack_into(
            "<IIIII",
            data,
            index * descriptor_size,
            section_rva + lookup_offset,
            0,
            0,
            section_rva + library_name_offsets[name],
            section_rva + iat_offset,
        )
    return bytes(data), addresses, iat_start, iat_size


def _imports(section_rva: int, image_base: int) -> tuple[bytes, dict[str, int], int, int]:
    return _imports_for(
        section_rva,
        image_base,
        # VirtualAlloc supplies the writable block that holds file-scope
        # variables, the way an anonymous mmap does on POSIX.
        (
            "GetStdHandle",
            "WriteFile",
            "ExitProcess",
            "VirtualAlloc",
            # File access. CreateFileA takes the same narrow path bytes a
            # native string already holds, so nothing has to be widened to
            # UTF-16 on the way in.
            "CreateFileA",
            "ReadFile",
            "CloseHandle",
        ),
    )


# A thread stack past this is a runaway rather than a program, and committing
# it up front would be a real cost to every process that runs the image.
_MAXIMUM_STACK_RESERVE = 16 * 1024 * 1024


def _pe_image(
    code: bytes,
    rdata: bytes,
    *,
    machine: int,
    iat_offset: int,
    iat_size: int,
    writable_rdata: bool = False,
    subsystem: int = 3,
    stack_bytes: int = 0,
) -> bytes:
    # Windows grows a thread stack one page at a time, by faulting on the
    # guard page that sits just below the committed region. Both prologues here
    # move the stack pointer down by the whole frame and then write at the new
    # low end, so a frame larger than a page steps clean over the guard page and
    # the write lands in reserved-but-uncommitted memory - an access violation
    # rather than a stack that grows. Committing enough up front for the largest
    # frame in the image means every such write lands inside committed memory
    # and no guard page is ever involved. The alternative, a probe loop in each
    # prologue touching one byte per page, is what MSVC emits; it is also
    # machine code for two architectures that cannot be run or tested here,
    # where this is a header field with exactly this purpose.
    page = 0x1000
    guard_slack = 4 * page  # room for the guard pages Windows keeps below it
    stack_commit = max(page, _align(stack_bytes + guard_slack, page))
    stack_reserve = max(0x100000, _align(stack_commit + guard_slack, page))
    if stack_reserve > _MAXIMUM_STACK_RESERVE:
        raise ValueError(
            f"a frame of {stack_bytes} bytes would need a "
            f"{stack_reserve}-byte thread stack, past the "
            f"{_MAXIMUM_STACK_RESERVE}-byte ceiling this writer commits; split "
            "the code into smaller functions"
        )
    image_base = 0x140000000
    section_alignment = 0x1000
    file_alignment = 0x200
    text_rva = 0x1000
    # After the code, not at a fixed address. Every image this writer produced
    # before was smaller than one page, so 0x2000 was after the code by
    # accident; a program driving CPython is fifty times that, and the two
    # sections overlapped in virtual address space. Windows maps sections by
    # these fields, so the loader would have mapped data over code.
    rdata_rva = _align(text_rva + len(code), section_alignment)
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
    struct.pack_into("<IHH", optional, 64, 0, subsystem, 0x8160)
    struct.pack_into(
        "<QQQQ",
        optional,
        72,
        stack_reserve,
        stack_commit,
        0x100000,
        0x1000,
    )
    struct.pack_into("<II", optional, 104, 0, 16)
    struct.pack_into("<II", optional, 120, rdata_rva, 40)
    struct.pack_into("<II", optional, 208, rdata_rva + iat_offset, iat_size)

    def section(
        name: bytes,
        virtual_size: int,
        rva: int,
        raw_size: int,
        raw_offset: int,
        flags: int,
    ) -> bytes:
        return struct.pack(
            "<8sIIIIIIHHI",
            name.ljust(8, b"\0"),
            virtual_size,
            rva,
            raw_size,
            raw_offset,
            0,
            0,
            0,
            0,
            flags,
        )

    sections = section(
        b".text", len(code), text_rva, text_raw_size, text_file_offset, 0x60000020
    )
    rdata_flags = 0xC0000040 if writable_rdata else 0x40000040
    sections += section(
        b".data" if writable_rdata else b".rdata",
        len(rdata),
        rdata_rva,
        rdata_raw_size,
        rdata_file_offset,
        rdata_flags,
    )
    headers = bytes(dos) + b"PE\0\0" + coff + bytes(optional) + sections
    headers += bytes(headers_size - len(headers))
    return (
        headers
        + code.ljust(text_raw_size, b"\0")
        + rdata.ljust(rdata_raw_size, b"\0")
    )


def _write_pe(module: Module, machine: int, arm64: bool) -> bytes:
    image_base = 0x140000000
    text_rva = 0x1000
    encoder = encode_windows_arm64 if arm64 else encode_windows
    # The same measure-then-place loop the dynamic writers below use, and for
    # the same reason. `_pe_image` puts the data section after the code, but
    # the RVAs *inside* the import table - the DLL name, the lookup table, the
    # address table - are baked in here, so they have to be computed against
    # the address the section actually gets. Fixing this address at 0x2000 was
    # right only while the code fitted in one page: past that the loader read
    # the middle of .text as a DLL name and refused the image, which is a
    # failure to start rather than anything the program could report.
    def build(rdata_rva: int) -> tuple[bytes, bytes, int, int]:
        blob, imports, iat_offset, iat_size = _imports(rdata_rva, image_base)
        return encoder(module, image_base + text_rva, imports), blob, iat_offset, iat_size

    code, _blob, _io, _is = build(_align(text_rva + 0x1000, 0x1000))
    settled = _align(text_rva + len(code), 0x1000)
    code, rdata, iat_offset, iat_size = build(settled)
    if _align(text_rva + len(code), 0x1000) != settled:
        raise AssertionError("PE code length changed when the data section moved")
    # An upper bound on how much stack the program can be using at once. The
    # largest single frame is not enough: the C front end emits real calls, so
    # frames nest, and a deep chain lands below the committed region exactly the
    # way one oversized frame does. Summing every body bounds any non-recursive
    # nesting - conservative, since most of those bodies never share the stack,
    # but the cost is committed pages and the alternative is a fault.
    #
    # Recursion deeper than this still walks off the end. That is stack
    # exhaustion rather than this bug, and it behaved the same way before.
    slots = module.stack_slots + sum(
        function.stack_slots for function in module.functions
    )
    stack_bytes = slots * 8 + 0x200 * (1 + len(module.functions))
    return _pe_image(
        code,
        rdata,
        machine=machine,
        iat_offset=iat_offset,
        iat_size=iat_size,
        stack_bytes=stack_bytes,
    )


def _descriptors(
    ordered: "dict[str, list[str]]",
) -> "list[tuple[str, tuple[str, ...]]]":
    """One import descriptor per DLL, with the kernel's own names folded in.

    A program that calls a Windows API from C imports from the same KERNEL32
    the encoder does. Two descriptors naming it is legal - the loader resolves
    each thunk array in turn - but one is what every other linker writes, and
    a single list is one fewer thing for a loader to disagree about.
    """

    merged: "dict[str, set[str]]" = {_PE_KERNEL_LIBRARY: set(_PE_KERNEL_IMPORTS)}
    for name, symbols in ordered.items():
        merged.setdefault(name, set()).update(symbols)
    return [
        (_PE_KERNEL_LIBRARY, tuple(sorted(merged.pop(_PE_KERNEL_LIBRARY)))),
        *((name, tuple(sorted(symbols))) for name, symbols in sorted(merged.items())),
    ]


def write_pe_x86_64_dynamic(
    module: Module,
    symbol_libraries: "dict[str, str]",
    static_bytes: int = 0,
) -> bytes:
    """A PE that imports from several DLLs, one of them the interpreter's.

    The rest of this writer emits one import descriptor, which names one DLL -
    enough for a program that only asks the kernel for services. A program
    driving CPython needs at least two, and the interpreter's exports are what
    the second names.

    Static storage goes in the image, in the writable data section, rather than
    in the VirtualAlloc block the non-CPython path parks in r15. That register
    is callee-saved, so while a CPython frame is live it holds CPython's value,
    and a compiled function called back from the interpreter would read a
    static through it and get whatever was there.
    """

    image_base = 0x140000000
    text_rva = 0x1000
    ordered: "dict[str, list[str]]" = {}
    for symbol, library in symbol_libraries.items():
        ordered.setdefault(library, []).append(symbol)
    libraries = _descriptors(ordered)
    # Where the data section lands depends on how long the code is, and how
    # long the code is does not depend on where the data lands: every reference
    # is a fixed-size displacement. So encode once to measure, place the
    # section after it, and encode again against the real addresses.
    def build(rdata_rva: int) -> tuple[bytes, bytes, int, int, int]:
        blob, imports, iat_offset, iat_size = _imports_from_libraries(
            rdata_rva, image_base, libraries
        )
        # The statics follow the import table in the same section, which is why
        # it is written writable here and read-only everywhere else.
        statics_rva = rdata_rva + _align(len(blob), 16)
        blob = blob.ljust(_align(len(blob), 16), b"\0") + bytes(static_bytes)
        encoded = encode_windows(
            module,
            image_base + text_rva,
            imports,
            statics_address=image_base + statics_rva if static_bytes else None,
        )
        return encoded, blob, iat_offset, iat_size, statics_rva

    code, _blob, _io, _is, _sr = build(_align(text_rva + 0x1000, 0x1000))
    settled = _align(text_rva + len(code), 0x1000)
    code, rdata, iat_offset, iat_size, _statics_rva = build(settled)
    if _align(text_rva + len(code), 0x1000) != settled:
        raise AssertionError("PE code length changed when the data section moved")
    slots = module.stack_slots + sum(
        function.stack_slots for function in module.functions
    )
    return _pe_image(
        code,
        rdata,
        machine=0x8664,
        iat_offset=iat_offset,
        iat_size=iat_size,
        writable_rdata=True,
        stack_bytes=slots * 8 + 0x200 * (1 + len(module.functions)),
    )


def write_pe_arm64_dynamic(
    module: Module,
    symbol_libraries: "dict[str, str]",
    static_bytes: int = 0,
) -> bytes:
    """The ARM64 counterpart of the writer above: a PE importing several DLLs.

    Statics go in the image rather than in a block reached through X28. The
    register form is fine for a program that only calls the kernel, but an
    image binding CPython can hand it a function to call, and a callback
    entered from inside CPython's frames finds CPython's value in that
    callee-saved register. Every static reference is therefore left as two
    words - adrp then add - and filled in here, once the data section is
    placed and its address is finally known.
    """

    image_base = 0x140000000
    text_rva = 0x1000
    ordered: "dict[str, list[str]]" = {}
    for symbol, library in symbol_libraries.items():
        ordered.setdefault(library, []).append(symbol)
    libraries = _descriptors(ordered)

    def build(rdata_rva: int):
        blob, imports, iat_offset, iat_size = _imports_from_libraries(
            rdata_rva, image_base, libraries
        )
        statics_rva = rdata_rva + _align(len(blob), 16)
        blob = blob.ljust(_align(len(blob), 16), b"\0") + bytes(static_bytes)
        encoded, sites = encode_windows_arm64(
            module, image_base + text_rva, imports, image_statics=True
        )
        return encoded, blob, iat_offset, iat_size, statics_rva, sites

    # Two passes, for the same reason the x86-64 writer needs them: every
    # reference is a fixed-size instruction, so the code's length does not
    # depend on where the data lands, but the data's place depends on the
    # code's length.
    code, _blob, _io, _is, _sr, _sites = build(_align(text_rva + 0x1000, 0x1000))
    settled = _align(text_rva + len(code), 0x1000)
    code, rdata, iat_offset, iat_size, statics_rva, sites = build(settled)
    if _align(text_rva + len(code), 0x1000) != settled:
        raise AssertionError("PE code length changed when the data section moved")

    # adrp reaches a 4 KB page at +/-4 GB, and add supplies the offset within
    # it. Both are relative to the instruction, so the pair is position
    # independent and reads the same object from whoever's frame calls in.
    patched = bytearray(code)
    static_base = image_base + statics_rva
    for byte_offset, static_offset in sites:
        instruction_addr = image_base + text_rva + byte_offset
        target = static_base + static_offset
        pages = ((target & ~0xFFF) - (instruction_addr & ~0xFFF)) >> 12
        if not -(1 << 20) <= pages < (1 << 20):
            raise ValueError("arm64 static reference is outside ADRP range")
        encoded = pages & 0x1FFFFF
        adrp = 0x90000000 | ((encoded & 3) << 29) | (((encoded >> 2) & 0x7FFFF) << 5)
        add = 0x91000000 | ((target & 0xFFF) << 10)
        struct.pack_into("<II", patched, byte_offset, adrp, add)

    slots = module.stack_slots + sum(
        function.stack_slots for function in module.functions
    )
    return _pe_image(
        bytes(patched),
        rdata,
        machine=0xAA64,
        iat_offset=iat_offset,
        iat_size=iat_size,
        writable_rdata=True,
        stack_bytes=slots * 8 + 0x200 * (1 + len(module.functions)),
    )


#: What every image asks the kernel for, whatever else it imports.
_PE_KERNEL_LIBRARY = "KERNEL32.dll"
_PE_KERNEL_IMPORTS = (
    "GetStdHandle",
    "WriteFile",
    "ExitProcess",
    "VirtualAlloc",
    "CreateFileA",
    "ReadFile",
    "CloseHandle",
)


_LAUNCHER_IMPORTS = (
    "lstrcpyA",
    "GetModuleFileNameW",
    "GetCommandLineW",
    "SetEnvironmentVariableW",
    "CreateProcessA",
    "WaitForSingleObject",
    "GetExitCodeProcess",
    "CloseHandle",
    "ExitProcess",
)


def _launcher_rdata(
    command_prefix: bytes,
) -> tuple[bytes, dict[str, int], int, int, int, int, int, int, int]:
    image_base = 0x140000000
    rdata_rva = 0x2000
    raw, imports, iat_offset, iat_size = _imports_for(
        rdata_rva, image_base, _LAUNCHER_IMPORTS
    )
    data = bytearray(raw)
    while len(data) % 16:
        data.append(0)
    prefix_offset = len(data)
    data.extend(command_prefix + b"\0")
    while len(data) % 16:
        data.append(0)
    buffer_offset = len(data)
    # CreateProcess may modify its command-line buffer in place.
    data.extend(b"\0" * (len(command_prefix) + 1))
    while len(data) % 16:
        data.append(0)
    self_environment_offset = len(data)
    data.extend("PY2BIN_ONEFILE_SELF".encode("utf-16-le") + b"\0\0")
    command_environment_offset = len(data)
    data.extend("PY2BIN_ONEFILE_COMMAND".encode("utf-16-le") + b"\0\0")
    while len(data) % 16:
        data.append(0)
    module_path_offset = len(data)
    # GetModuleFileNameW accepts the extended Windows path limit in WCHARs.
    data.extend(b"\0" * (32768 * 2))
    prefix_address = image_base + rdata_rva + prefix_offset
    buffer_address = image_base + rdata_rva + buffer_offset
    self_environment_address = image_base + rdata_rva + self_environment_offset
    command_environment_address = (
        image_base + rdata_rva + command_environment_offset
    )
    module_path_address = image_base + rdata_rva + module_path_offset
    return (
        bytes(data),
        imports,
        iat_offset,
        iat_size,
        prefix_address,
        buffer_address,
        self_environment_address,
        command_environment_address,
        module_path_address,
    )


def _x86_64_launcher_code(
    code_address: int,
    imports: dict[str, int],
    prefix_address: int,
    buffer_address: int,
    self_environment_address: int,
    command_environment_address: int,
    module_path_address: int,
    creation_flags: int,
) -> bytes:
    code = bytearray()
    calls: list[tuple[int, str]] = []
    addresses: list[tuple[int, int]] = []
    branches: list[tuple[int, str]] = []
    labels: dict[str, int] = {}

    def call(symbol: str) -> None:
        position = len(code)
        code.extend(b"\xff\x15\0\0\0\0")
        calls.append((position + 2, symbol))

    def lea(register: bytes, address: int) -> None:
        code.extend(register)
        position = len(code)
        code.extend(b"\0\0\0\0")
        addresses.append((position, address))

    code.extend(b"\x48\x81\xec\xe8\0\0\0")  # sub rsp, 0xe8
    code.extend(b"\x31\xc0\x48\x8d\x7c\x24\x20\xb9\x19\0\0\0\xf3\x48\xab")
    code.extend(b"\x31\xc9")
    lea(b"\x48\x8d\x15", module_path_address)
    code.extend(b"\x41\xb8\0\x80\0\0")
    call("GetModuleFileNameW")
    code.extend(b"\x85\xc0\x0f\x84\0\0\0\0")
    branches.append((len(code) - 4, "failure"))
    lea(b"\x48\x8d\x0d", self_environment_address)
    lea(b"\x48\x8d\x15", module_path_address)
    call("SetEnvironmentVariableW")
    code.extend(b"\x85\xc0\x0f\x84\0\0\0\0")
    branches.append((len(code) - 4, "failure"))
    call("GetCommandLineW")
    code.extend(b"\x48\x85\xc0\x0f\x84\0\0\0\0")
    branches.append((len(code) - 4, "failure"))
    code.extend(b"\x48\x89\xc2")
    lea(b"\x48\x8d\x0d", command_environment_address)
    call("SetEnvironmentVariableW")
    code.extend(b"\x85\xc0\x0f\x84\0\0\0\0")
    branches.append((len(code) - 4, "failure"))
    lea(b"\x48\x8d\x0d", buffer_address)
    lea(b"\x48\x8d\x15", prefix_address)
    call("lstrcpyA")
    code.extend(b"\xc7\x44\x24\x60\x68\0\0\0")  # STARTUPINFOA.cb = 104
    code.extend(b"\x31\xc9")  # application name = NULL
    lea(b"\x48\x8d\x15", buffer_address)
    code.extend(b"\x45\x31\xc0\x45\x31\xc9")
    # bInheritHandles = TRUE. With FALSE the child gets none of this process's
    # standard handles, so a redirected stdout - `frozen.exe > out.txt` - is not
    # passed down and everything the program prints is written to a handle that
    # is not there. The child then fails on its first print and the traceback
    # goes the same nowhere, which reads from outside as a silent exit 1.
    code.extend(b"\x48\xc7\x44\x24\x20\x01\0\0\0")
    # CREATE_NO_WINDOW only for a windowed build, where the point is to stop a
    # console flashing up. For a console build it does the opposite of what is
    # wanted: it denies the child the console it is supposed to be writing to.
    code.extend(b"\x48\xc7\x44\x24\x28" + struct.pack("<I", creation_flags))
    code.extend(b"\x48\xc7\x44\x24\x30\0\0\0\0")
    code.extend(b"\x48\xc7\x44\x24\x38\0\0\0\0")
    code.extend(b"\x48\x8d\x44\x24\x60\x48\x89\x44\x24\x40")
    code.extend(b"\x48\x8d\x84\x24\xc8\0\0\0\x48\x89\x44\x24\x48")
    call("CreateProcessA")
    code.extend(b"\x85\xc0\x0f\x84\0\0\0\0")
    branches.append((len(code) - 4, "failure"))
    code.extend(b"\x48\x8b\x8c\x24\xd0\0\0\0")
    call("CloseHandle")
    code.extend(b"\x48\x8b\x8c\x24\xc8\0\0\0\xba\xff\xff\xff\xff")
    call("WaitForSingleObject")
    code.extend(b"\x48\x8b\x8c\x24\xc8\0\0\0")
    code.extend(b"\x48\x8d\x94\x24\xe0\0\0\0")
    call("GetExitCodeProcess")
    code.extend(b"\x48\x8b\x8c\x24\xc8\0\0\0")
    call("CloseHandle")
    code.extend(b"\x8b\x8c\x24\xe0\0\0\0")
    call("ExitProcess")
    labels["failure"] = len(code)
    code.extend(b"\xb9\x6f\0\0\0")
    call("ExitProcess")

    for position, symbol in calls:
        next_address = code_address + position + 4
        struct.pack_into("<i", code, position, imports[symbol] - next_address)
    for position, address in addresses:
        next_address = code_address + position + 4
        struct.pack_into("<i", code, position, address - next_address)
    for position, label in branches:
        struct.pack_into("<i", code, position, labels[label] - (position + 4))
    return bytes(code)


def _arm64_launcher_code(
    code_address: int,
    imports: dict[str, int],
    prefix_address: int,
    buffer_address: int,
    self_environment_address: int,
    command_environment_address: int,
    module_path_address: int,
    creation_flags: int,
) -> bytes:
    words: list[int] = [_sub_sp(160)]
    calls: list[tuple[int, str]] = []
    addresses: list[tuple[int, int, int]] = []

    def call(symbol: str) -> None:
        index = len(words)
        words.extend((0, 0, 0xD63F0200))
        calls.append((index, symbol))

    def address(register: int, target: int) -> None:
        index = len(words)
        words.append(0)
        addresses.append((index, register, target))

    for offset in range(0, 160, 8):
        words.append(0xF90003FF | ((offset // 8) << 10))  # str xzr,[sp,#offset]
    words.append(0xAA1F03E0)  # x0 = NULL
    address(1, module_path_address)
    words.extend(_mov(2, 32768))
    call("GetModuleFileNameW")
    failure_branches = [len(words)]
    words.append(0)
    address(0, self_environment_address)
    address(1, module_path_address)
    call("SetEnvironmentVariableW")
    failure_branches.append(len(words))
    words.append(0)
    call("GetCommandLineW")
    words.append(0xAA0003E1)  # x1 = returned command line
    address(0, command_environment_address)
    call("SetEnvironmentVariableW")
    failure_branches.append(len(words))
    words.append(0)
    address(0, buffer_address)
    address(1, prefix_address)
    call("lstrcpyA")
    words.extend(_mov(9, 104))
    words.append(0xB90013E9)  # str w9,[sp,#16]
    words.append(0xAA1F03E0)  # x0 = NULL
    address(1, buffer_address)
    words.extend((0xAA1F03E2, 0xAA1F03E3))
    # x4 is bInheritHandles and x5 the creation flags; see the x86-64 stub above
    # for why the child has to inherit this process's handles.
    words.extend(_mov(4, 1))
    words.extend(_mov(5, creation_flags))
    words.extend((0xAA1F03E6, 0xAA1F03E7))
    words.extend(
        (
            0x910043E8,  # add x8,sp,#16
            0xF90003E8,  # str x8,[sp]
            0x9101E3E9,  # add x9,sp,#120
            0xF90007E9,  # str x9,[sp,#8]
        )
    )
    call("CreateProcessA")
    failure_branches.append(len(words))
    words.append(0)
    words.append(0xF94043E0)  # thread handle at process-info + 8
    call("CloseHandle")
    words.append(0xF9403FE0)  # process handle at sp + 120
    words.append(0x92800001)  # mov x1, #-1
    call("WaitForSingleObject")
    words.append(0xF9403FE0)
    words.append(0x910243E1)  # add x1,sp,#144
    call("GetExitCodeProcess")
    words.append(0xF9403FE0)
    call("CloseHandle")
    words.append(0xB94093E0)  # ldr w0,[sp,#144]
    words.append(0x910283FF)  # add sp,sp,#160
    call("ExitProcess")
    failure_index = len(words)
    words.extend(_mov(0, 111))
    words.append(0x910283FF)
    call("ExitProcess")
    for failure_branch in failure_branches:
        words[failure_branch] = (
            0x34000000
            | (((failure_index - failure_branch) & 0x7FFFF) << 5)
        )

    image = bytearray(struct.pack(f"<{len(words)}I", *words))
    for index, register, target in addresses:
        instruction_address = code_address + index * 4
        struct.pack_into(
            "<I",
            image,
            index * 4,
            _adr(register, target - instruction_address),
        )
    for index, symbol in calls:
        instruction_address = code_address + index * 4
        target = imports[symbol]
        pages = ((target & ~0xFFF) - (instruction_address & ~0xFFF)) >> 12
        encoded = pages & 0x1FFFFF
        adrp = (
            0x90000010
            | ((encoded & 3) << 29)
            | (((encoded >> 2) & 0x7FFFF) << 5)
        )
        page_offset = target & 0xFFF
        ldr = 0xF9400210 | ((page_offset // 8) << 10)
        struct.pack_into("<II", image, index * 4, adrp, ldr)
    return bytes(image)


def write_pe_shell_launcher(
    command_prefix: bytes,
    machine: str,
    *,
    windowed: bool = False,
) -> bytes:
    """Write a PE launcher that runs and waits for a fixed PowerShell command.

    The command is copied to writable image data because CreateProcess may
    modify its command-line buffer.
    """

    (
        rdata,
        imports,
        iat_offset,
        iat_size,
        prefix_address,
        buffer_address,
        self_environment_address,
        command_environment_address,
        module_path_address,
    ) = _launcher_rdata(command_prefix)
    code_address = 0x140001000
    # CREATE_NO_WINDOW, and only when windowed: see the stubs for why a console
    # build must not have it.
    creation_flags = 0x08000000 if windowed else 0
    if machine == "x86_64":
        code = _x86_64_launcher_code(
            code_address,
            imports,
            prefix_address,
            buffer_address,
            self_environment_address,
            command_environment_address,
            module_path_address,
            creation_flags,
        )
        machine_id = 0x8664
    elif machine == "arm64":
        code = _arm64_launcher_code(
            code_address,
            imports,
            prefix_address,
            buffer_address,
            self_environment_address,
            command_environment_address,
            module_path_address,
            creation_flags,
        )
        machine_id = 0xAA64
    else:
        raise ValueError(f"unsupported Windows launcher machine: {machine}")
    # `_launcher_rdata` placed the data at 0x2000, one page after the code, and
    # the addresses it handed back are already baked into `code`. That holds
    # only while the stub fits in a page. It does, with room to spare - but the
    # same assumption silently produced an unloadable image in the writer above
    # once the code outgrew it, so say so here rather than emit one again.
    if len(code) > 0x1000:
        raise AssertionError(
            f"the Windows launcher stub grew to {len(code)} bytes, past the one "
            "page its data section is placed after; give `_launcher_rdata` the "
            "same measure-then-place loop the other writers use"
        )
    return _pe_image(
        code,
        rdata,
        machine=machine_id,
        iat_offset=iat_offset,
        iat_size=iat_size,
        writable_rdata=True,
        subsystem=2 if windowed else 3,
    )


def write_pe_x86_64(module: Module) -> bytes:
    """Write a Windows x86-64 PE32+ console executable."""
    return _write_pe(module, 0x8664, arm64=False)


def write_pe_arm64(module: Module) -> bytes:
    """Write a Windows ARM64 PE32+ console executable."""
    return _write_pe(module, 0xAA64, arm64=True)
