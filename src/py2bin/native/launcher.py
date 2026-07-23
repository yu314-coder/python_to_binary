from __future__ import annotations

import platform
import struct

from .arm64 import _adr, _mov
from .formats.macho import write_macho_arm64, write_macho_x86_64


def _arm64_shell_launcher(
    command: str,
    info_plist: bytes | None = None,
    code_resources: bytes | None = None,
) -> bytes:
    # LC_MAIN enters with argc/argv/envp in x0/x1/x2. Build:
    #   ["/bin/sh", "-c", command, original argv[0...argc], NULL]
    # so the shell receives the app's original command-line arguments.
    words = [
        0xAA0003F3,  # mov x19, x0 (argc)
        0xAA0103F4,  # mov x20, x1 (argv)
        0xAA0203F5,  # mov x21, x2 (envp)
        0xF101027F,  # cmp x19, #64
        0,  # b.hi failure
        0xD10883FF,  # sub sp, sp, #544
        0,  # adr x0, shell
        0,  # adr x3, "-c"
        0,  # adr x4, command
        0xF90003E0,  # str x0, [sp]
        0xF90007E3,  # str x3, [sp, #8]
        0xF9000BE4,  # str x4, [sp, #16]
        0x910063E9,  # add x9, sp, #24
        0xAA1F03E7,  # mov x7, xzr
        0xF8677A88,  # loop: ldr x8, [x20, x7, lsl #3]
        0xF8277928,  # str x8, [x9, x7, lsl #3]
        0,  # cbz x8, exec
        0x910004E7,  # add x7, x7, #1
        0,  # b loop
        0x910003E1,  # mov x1, sp
        0xAA1503E2,  # mov x2, x21
        *_mov(16, 59),  # SYS_execve
        0xD4001001,  # svc #0x80
        *_mov(0, 64),  # failure: EX_USAGE or execve failure
        *_mov(16, 1),  # SYS_exit
        0xD4001001,
    ]
    failure_index = len(words) - len(_mov(0, 64)) - len(_mov(16, 1)) - 1
    loop_index = 14
    cbz_index = 16
    back_branch_index = 18
    exec_index = 19
    words[4] = 0x54000008 | ((failure_index - 4) << 5)  # b.hi
    words[cbz_index] = 0xB4000008 | ((exec_index - cbz_index) << 5)
    words[back_branch_index] = 0x14000000 | (
        (loop_index - back_branch_index) & 0x03FFFFFF
    )
    image = bytearray(struct.pack(f"<{len(words)}I", *words))
    strings = (b"/bin/sh\0", b"-c\0", command.encode("utf-8") + b"\0")
    for instruction_index, register, data in zip((6, 7, 8), (0, 3, 4), strings):
        instruction_offset = instruction_index * 4
        data_offset = len(image)
        struct.pack_into(
            "<I",
            image,
            instruction_offset,
            _adr(register, data_offset - instruction_offset),
        )
        image.extend(data)
    return write_macho_arm64(bytes(image), info_plist, code_resources)


def _x86_64_shell_launcher(command: str) -> bytes:
    code = bytearray()
    code += b"\x4c\x8b\x06"  # mov r8, [rsi] (original argv[0])
    references: list[tuple[int, bytes]] = []

    def lea(prefix: bytes, data: bytes) -> None:
        code.extend(prefix)
        displacement = len(code)
        code.extend(b"\0\0\0\0")
        references.append((displacement, data))

    lea(b"\x48\x8d\x3d", b"/bin/sh\0")  # rdi
    lea(b"\x48\x8d\x0d", b"-c\0")  # rcx
    lea(b"\x4c\x8d\x0d", command.encode("utf-8") + b"\0")  # r9
    code += b"\x48\x83\xec\x30"  # sub rsp, 48
    code += b"\x48\x89\x3c\x24"  # argv[0] = shell
    code += b"\x48\x89\x4c\x24\x08"  # argv[1] = -c
    code += b"\x4c\x89\x4c\x24\x10"  # argv[2] = command
    code += b"\x4c\x89\x44\x24\x18"  # argv[3] = original argv[0]
    code += b"\x48\xc7\x44\x24\x20\0\0\0\0"  # argv[4] = NULL
    code += b"\x48\x89\xe6"  # rsi = argv; rdx still points to envp
    code += b"\xb8\x3b\0\0\x02\x0f\x05"  # SYS_execve
    code += b"\x89\xc7\xb8\x01\0\0\x02\x0f\x05"  # SYS_exit(errno)
    for displacement, data in references:
        data_offset = len(code)
        struct.pack_into("<i", code, displacement, data_offset - (displacement + 4))
        code.extend(data)
    return write_macho_x86_64(bytes(code))


def macos_shell_launcher(
    command: str,
    machine: str | None = None,
    info_plist: bytes | None = None,
    code_resources: bytes | None = None,
) -> bytes:
    """Return a directly executable Mach-O that invokes a fixed shell command."""
    machine = machine or platform.machine()
    if machine == "arm64":
        return _arm64_shell_launcher(command, info_plist, code_resources)
    if machine in {"x86_64", "AMD64"}:
        if info_plist is not None or code_resources is not None:
            raise ValueError("signed x86-64 app launchers are not implemented yet")
        return _x86_64_shell_launcher(command)
    raise ValueError(f"native macOS launcher is not implemented for {machine}")
