from __future__ import annotations

import struct

from .ir import Exit, Module, Write


def _mov_imm32(register_opcode: bytes, value: int) -> bytes:
    return b"\x48\xc7" + register_opcode + struct.pack("<I", value & 0xFFFFFFFF)


def encode(module: Module, platform: str, code_address: int) -> bytes:
    """Encode native syscalls directly; no text assembly is produced."""
    if platform == "linux":
        write_number, exit_number = 1, 60
    elif platform == "darwin":
        write_number, exit_number = 0x02000004, 0x02000001
    else:
        raise ValueError(f"unsupported x86-64 syscall platform: {platform}")

    chunks: list[bytearray] = []
    pending_strings: list[tuple[int, bytes]] = []
    offset = 0
    for operation in module.operations:
        if isinstance(operation, Write):
            # mov rax, write; mov rdi, 1; lea rsi, [rip+disp32];
            # mov rdx, len; syscall
            chunk = bytearray()
            chunk += _mov_imm32(b"\xc0", write_number)
            chunk += _mov_imm32(b"\xc7", 1)
            lea_displacement_position = offset + len(chunk) + 3
            chunk += b"\x48\x8d\x35\x00\x00\x00\x00"
            chunk += _mov_imm32(b"\xc2", len(operation.data))
            chunk += b"\x0f\x05"
            chunks.append(chunk)
            pending_strings.append((lea_displacement_position, operation.data))
            offset += len(chunk)
        elif isinstance(operation, Exit):
            chunk = bytearray()
            chunk += _mov_imm32(b"\xc0", exit_number)
            chunk += _mov_imm32(b"\xc7", operation.status)
            chunk += b"\x0f\x05"
            chunks.append(chunk)
            offset += len(chunk)

    code = bytearray().join(chunks)
    for displacement_position, data in pending_strings:
        data_offset = len(code)
        # Displacement is relative to the instruction following lea.
        displacement = data_offset - (displacement_position + 4)
        struct.pack_into("<i", code, displacement_position, displacement)
        code += data
    return bytes(code)


def encode_windows(module: Module, code_address: int, imports: dict[str, int]) -> bytes:
    """Encode calls through a Windows x64 import-address table."""
    code = bytearray(b"\x48\x83\xec\x38")
    address_patches: list[tuple[int, int]] = []
    string_patches: list[tuple[int, bytes]] = []

    def indirect_call(symbol: str) -> None:
        instruction = len(code)
        code.extend(b"\xff\x15\x00\x00\x00\x00")
        address_patches.append((instruction + 2, imports[symbol]))

    for operation in module.operations:
        if isinstance(operation, Write):
            code.extend(b"\xb9\xf5\xff\xff\xff")
            indirect_call("GetStdHandle")
            code.extend(b"\x48\x89\xc1")
            displacement_position = len(code) + 3
            code.extend(b"\x48\x8d\x15\x00\x00\x00\x00")
            string_patches.append((displacement_position, operation.data))
            code.extend(b"\x41\xb8" + struct.pack("<I", len(operation.data)))
            code.extend(b"\x4c\x8d\x4c\x24\x28")
            code.extend(b"\x48\xc7\x44\x24\x20\x00\x00\x00\x00")
            indirect_call("WriteFile")
        elif isinstance(operation, Exit):
            code.extend(b"\xb9" + struct.pack("<I", operation.status))
            indirect_call("ExitProcess")

    for position, target_address in address_patches:
        next_address = code_address + position + 4
        struct.pack_into("<i", code, position, target_address - next_address)
    for position, data in string_patches:
        data_address = code_address + len(code)
        next_address = code_address + position + 4
        struct.pack_into("<i", code, position, data_address - next_address)
        code.extend(data)
    return bytes(code)

