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


def _imports(section_rva: int, image_base: int) -> tuple[bytes, dict[str, int], int, int]:
    return _imports_for(
        section_rva,
        image_base,
        ("GetStdHandle", "WriteFile", "ExitProcess"),
    )


def _pe_image(
    code: bytes,
    rdata: bytes,
    *,
    machine: int,
    iat_offset: int,
    iat_size: int,
    writable_rdata: bool = False,
    subsystem: int = 3,
) -> bytes:
    image_base = 0x140000000
    section_alignment = 0x1000
    file_alignment = 0x200
    text_rva, rdata_rva = 0x1000, 0x2000
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
    struct.pack_into("<QQQQ", optional, 72, 0x100000, 0x1000, 0x100000, 0x1000)
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
    text_rva, rdata_rva = 0x1000, 0x2000
    rdata, imports, iat_offset, iat_size = _imports(rdata_rva, image_base)
    encoder = encode_windows_arm64 if arm64 else encode_windows
    code = encoder(module, image_base + text_rva, imports)
    return _pe_image(
        code,
        rdata,
        machine=machine,
        iat_offset=iat_offset,
        iat_size=iat_size,
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
    code.extend(b"\x48\xc7\x44\x24\x20\0\0\0\0")
    code.extend(b"\x48\xc7\x44\x24\x28\0\0\0\x08")  # CREATE_NO_WINDOW
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
    words.extend((0xAA1F03E2, 0xAA1F03E3, 0xAA1F03E4))
    words.extend(_mov(5, 0x08000000))
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
    if machine == "x86_64":
        code = _x86_64_launcher_code(
            code_address,
            imports,
            prefix_address,
            buffer_address,
            self_environment_address,
            command_environment_address,
            module_path_address,
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
        )
        machine_id = 0xAA64
    else:
        raise ValueError(f"unsupported Windows launcher machine: {machine}")
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
