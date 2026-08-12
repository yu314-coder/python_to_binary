from __future__ import annotations

import platform
import struct

from .arm64 import _adr, _mov
from .formats.elf import write_elf_arm64, write_elf_x86_64
from .formats.macho import write_macho_arm64, write_macho_x86_64


def _arm64_shell_launcher(
    command: str,
    info_plist: bytes | None = None,
    code_resources: bytes | None = None,
    extra_data: bytes = b"",
    *,
    platform_name: str = "darwin",
) -> bytes:
    # Build ["/bin/sh", "-c", command, original argv[0...argc], NULL].
    # sh uses original argv[0] as $0 and therefore forwards argv[1:] as
    # "$@". LC_MAIN supplies x0/x1/x2 on macOS; Linux supplies the same
    # information on the initial process stack.
    if platform_name == "darwin":
        words = [
            0xAA0003F3,  # mov x19, x0 (argc)
            0xAA0103F4,  # mov x20, x1 (argv)
            0xAA0203F5,  # mov x21, x2 (envp)
        ]
        exec_number, exit_number, svc = 59, 1, 0xD4001001
    elif platform_name == "linux":
        words = [
            0x910003F6,  # mov x22, sp
            0xF94002D3,  # ldr x19, [x22] (argc)
            0x910022D4,  # add x20, x22, #8 (argv)
            0x8B130ED5,  # add x21, x22, x19, lsl #3
            0x910042B5,  # add x21, x21, #16 (envp)
        ]
        exec_number, exit_number, svc = 221, 93, 0xD4000001
    else:
        raise ValueError(f"unsupported ARM64 shell platform: {platform_name}")

    words.append(0xF101027F)  # cmp x19, #64
    failure_branch = len(words)
    words.append(0)  # b.hi failure
    words.append(0xD10883FF)  # sub sp, sp, #544
    shell_adr = len(words)
    words.append(0)
    option_adr = len(words)
    words.append(0)
    command_adr = len(words)
    words.append(0)
    words.extend(
        (
            0xF90003E0,  # str x0, [sp]
            0xF90007E3,  # str x3, [sp, #8]
            0xF9000BE4,  # str x4, [sp, #16]
            0x910063E9,  # add x9, sp, #24
            0xAA1F03E7,  # mov x7, xzr
        )
    )
    loop = len(words)
    words.extend(
        (
            0xF8677A88,  # ldr x8, [x20, x7, lsl #3]
            0xF8277928,  # str x8, [x9, x7, lsl #3]
        )
    )
    null_branch = len(words)
    words.append(0)  # cbz x8, exec
    words.append(0x910004E7)  # add x7, x7, #1
    loop_branch = len(words)
    words.append(0)
    execute = len(words)
    words.extend((0x910003E1, 0xAA1503E2))  # x1=sp, x2=envp
    words.extend(_mov(16 if platform_name == "darwin" else 8, exec_number))
    words.append(svc)
    failure = len(words)
    words.extend(_mov(0, 64))
    words.extend(_mov(16 if platform_name == "darwin" else 8, exit_number))
    words.append(svc)

    words[failure_branch] = 0x54000008 | ((failure - failure_branch) << 5)
    words[null_branch] = 0xB4000008 | ((execute - null_branch) << 5)
    words[loop_branch] = 0x14000000 | (
        (loop - loop_branch) & 0x03FFFFFF
    )
    image = bytearray(struct.pack(f"<{len(words)}I", *words))
    strings = (b"/bin/sh\0", b"-c\0", command.encode("utf-8") + b"\0")
    for instruction_index, register, data in zip(
        (shell_adr, option_adr, command_adr),
        (0, 3, 4),
        strings,
    ):
        instruction_offset = instruction_index * 4
        data_offset = len(image)
        struct.pack_into(
            "<I",
            image,
            instruction_offset,
            _adr(register, data_offset - instruction_offset),
        )
        image.extend(data)
    image.extend(extra_data)
    if platform_name == "linux":
        return write_elf_arm64(bytes(image))
    return write_macho_arm64(bytes(image), info_plist, code_resources)


def _x86_64_shell_launcher(
    command: str,
    extra_data: bytes = b"",
    *,
    platform_name: str = "darwin",
    info_plist: bytes | None = None,
    code_resources: bytes | None = None,
) -> bytes:
    code = bytearray()
    if platform_name == "linux":
        exec_number, exit_number = 59, 60
    elif platform_name == "darwin":
        exec_number, exit_number = 0x0200003B, 0x02000001
    else:
        raise ValueError(f"unsupported x86-64 shell platform: {platform_name}")

    # Both platforms hand the entry point the initial process stack rather than
    # registers: Linux starts at _start, and the x86-64 Mach-O writer uses
    # LC_UNIXTHREAD, which starts at the raw entry point the same way. (Only the
    # arm64 Mach-O uses LC_MAIN, where argc/argv/envp really do arrive in
    # x0/x1/x2.) Reading rdi/rsi/rdx here would read uninitialised registers.
    code += b"\x49\x89\xe5"  # mov r13, rsp
    code += b"\x4d\x8b\x65\x00"  # mov r12, [r13] (argc)
    code += b"\x4d\x8d\x75\x08"  # lea r14, [r13+8] (argv)
    code += b"\x4f\x8d\x7c\xe5\x10"  # lea r15, [r13+r12*8+16] (envp)

    code += b"\x49\x83\xfc\x40"  # cmp r12, 64
    high_branch = len(code)
    code += b"\x0f\x87\0\0\0\0"  # ja failure
    references: list[tuple[int, bytes]] = []

    def lea(prefix: bytes, data: bytes) -> None:
        code.extend(prefix)
        displacement = len(code)
        code.extend(b"\0\0\0\0")
        references.append((displacement, data))

    lea(b"\x48\x8d\x3d", b"/bin/sh\0")
    lea(b"\x48\x8d\x0d", b"-c\0")
    lea(b"\x4c\x8d\x0d", command.encode("utf-8") + b"\0")
    code += b"\x48\x81\xec\x30\x02\0\0"  # sub rsp, 560
    code += b"\x48\x89\x3c\x24"
    code += b"\x48\x89\x4c\x24\x08"
    code += b"\x4c\x89\x4c\x24\x10"
    code += b"\x45\x31\xd2"  # r10 = 0
    loop = len(code)
    code += b"\x4b\x8b\x04\xd6"  # rax = original argv[r10]
    code += b"\x4a\x89\x44\xd4\x18"  # new argv[r10+3] = rax
    code += b"\x49\xff\xc2"
    code += b"\x4d\x39\xe2"
    loop_branch = len(code)
    code += b"\x0f\x86\0\0\0\0"
    struct.pack_into("<i", code, loop_branch + 2, loop - (loop_branch + 6))
    code += b"\x48\x89\xe6"  # rsi = new argv
    code += b"\x4c\x89\xfa"  # rdx = original envp
    code += b"\xb8" + struct.pack("<I", exec_number) + b"\x0f\x05"
    failure = len(code)
    code += b"\xbf\x40\0\0\0"
    code += b"\xb8" + struct.pack("<I", exit_number) + b"\x0f\x05"
    struct.pack_into("<i", code, high_branch + 2, failure - (high_branch + 6))
    for displacement, data in references:
        data_offset = len(code)
        struct.pack_into(
            "<i",
            code,
            displacement,
            data_offset - (displacement + 4),
        )
        code.extend(data)
    code.extend(extra_data)
    if platform_name == "linux":
        return write_elf_x86_64(bytes(code))
    return write_macho_x86_64(bytes(code), info_plist, code_resources)


def macos_shell_launcher(
    command: str,
    machine: str | None = None,
    info_plist: bytes | None = None,
    code_resources: bytes | None = None,
    extra_data: bytes = b"",
) -> bytes:
    """Return a directly executable Mach-O that invokes a fixed shell command."""

    machine = machine or platform.machine()
    if machine == "arm64":
        return _arm64_shell_launcher(
            command,
            info_plist,
            code_resources,
            extra_data,
        )
    if machine in {"x86_64", "AMD64"}:
        # Signed too, sealing the same two documents the arm64 launcher does.
        # Intel macOS loads unsigned executables perfectly well, and this was
        # emitted unsigned for that reason - but a universal binary is only as
        # signed as its least signed slice, so an unsigned x86-64 half made the
        # whole bundle report as unsigned however carefully the arm64 half had
        # been sealed.
        return _x86_64_shell_launcher(
            command,
            extra_data,
            info_plist=info_plist,
            code_resources=code_resources,
        )
    raise ValueError(f"native macOS launcher is not implemented for {machine}")


def linux_shell_launcher(
    command: str,
    machine: str,
    extra_data: bytes = b"",
) -> bytes:
    """Return a static ELF launcher that invokes a fixed shell command."""

    if machine == "arm64":
        return _arm64_shell_launcher(
            command,
            extra_data=extra_data,
            platform_name="linux",
        )
    if machine == "x86_64":
        return _x86_64_shell_launcher(
            command,
            extra_data=extra_data,
            platform_name="linux",
        )
    raise ValueError(f"native Linux launcher is not implemented for {machine}")
