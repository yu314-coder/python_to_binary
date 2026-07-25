from __future__ import annotations

import struct

from .ir import (
    BitsFloat,
    Exit,
    ExitValue,
    FloatBinary,
    FloatBits,
    FloatCompare,
    FloatConstant,
    FloatExpression,
    FloatLoad,
    FloatStore,
    FloatToInt,
    FloatUnary,
    HeapAlloc,
    HeapInit,
    HeapLoad,
    HeapStore,
    IntBinary,
    IntCompare,
    IntConstant,
    IntExpression,
    IntLoad,
    IntToFloat,
    IntUnary,
    Jump,
    JumpIfFalse,
    Label,
    Module,
    SlotAddress,
    Store,
    Write,
    WriteRuntime,
)


# Loads widen into RAX exactly as the C integer conversions require; stores
# touch only the low bytes of the destination object.
_X86_LOADS = {
    (8, False): b"\x48\x8b\x00",  # mov   rax, [rax]
    (8, True): b"\x48\x8b\x00",
    (4, False): b"\x8b\x00",  # mov   eax, [rax]   (zero-extends)
    (4, True): b"\x48\x63\x00",  # movsxd rax, dword [rax]
    (2, False): b"\x0f\xb7\x00",  # movzx eax, word [rax]
    (2, True): b"\x48\x0f\xbf\x00",  # movsx rax, word [rax]
    (1, False): b"\x0f\xb6\x00",  # movzx eax, byte [rax]
    (1, True): b"\x48\x0f\xbe\x00",  # movsx rax, byte [rax]
}
_X86_STORES = {
    8: b"\x48\x89\x01",  # mov [rcx], rax
    4: b"\x89\x01",  # mov [rcx], eax
    2: b"\x66\x89\x01",  # mov [rcx], ax
    1: b"\x88\x01",  # mov [rcx], al
}


def _mov_imm32(register_opcode: bytes, value: int) -> bytes:
    return b"\x48\xc7" + register_opcode + struct.pack("<I", value & 0xFFFFFFFF)


def _sub_stack(amount: int) -> bytes:
    if amount <= 0:
        return b""
    if amount <= 0x7F:
        return b"\x48\x83\xec" + bytes((amount,))
    return b"\x48\x81\xec" + struct.pack("<I", amount)


def _frame_bytes(stack_slots: int, base: int = 0) -> int:
    variable_bytes = (stack_slots * 8 + 15) & ~15
    return base + variable_bytes


def _expression(code: bytearray, expression: IntExpression, slot_base: int) -> None:
    """Place one signed 64-bit native integer expression in RAX."""

    if isinstance(expression, IntConstant):
        code.extend(b"\x48\xb8" + struct.pack("<Q", expression.value & 0xFFFFFFFFFFFFFFFF))
        return
    if isinstance(expression, IntLoad):
        displacement = slot_base + expression.slot * 8
        code.extend(b"\x48\x8b\x85" + struct.pack("<i", displacement))
        return
    if isinstance(expression, SlotAddress):
        displacement = slot_base + expression.slot * 8
        code.extend(b"\x48\x8d\x85" + struct.pack("<i", displacement))  # lea rax,[rbp+d]
        return
    if isinstance(expression, IntUnary):
        _expression(code, expression.operand, slot_base)
        if expression.operator == "pos":
            return
        if expression.operator == "neg":
            code.extend(b"\x48\xf7\xd8")
            return
        if expression.operator == "invert":
            code.extend(b"\x48\xf7\xd0")
            return
        if expression.operator == "not":
            code.extend(b"\x48\x85\xc0\x0f\x94\xc0\x48\x0f\xb6\xc0")
            return
        raise ValueError(f"unknown x86-64 unary operation {expression.operator!r}")
    if isinstance(expression, (IntBinary, IntCompare)):
        _expression(code, expression.left, slot_base)
        code.extend(b"\x50")  # push rax
        _expression(code, expression.right, slot_base)
        code.extend(b"\x48\x89\xc1\x58")  # mov rcx, rax; pop rax
        if isinstance(expression, IntBinary):
            instructions = {
                "add": b"\x48\x01\xc8",
                "sub": b"\x48\x29\xc8",
                "mul": b"\x48\x0f\xaf\xc1",
                "and": b"\x48\x21\xc8",
                "or": b"\x48\x09\xc8",
                "xor": b"\x48\x31\xc8",
                "lshift": b"\x48\xd3\xe0",  # shl rax, cl
                "rshift": b"\x48\xd3\xf8",  # sar rax, cl (arithmetic)
                "urshift": b"\x48\xd3\xe8",  # shr rax, cl (logical)
                # cqo/xor sets up the 128-bit dividend; the quotient lands in
                # rax and the remainder in rdx, so % is one extra move.
                "sdiv": b"\x48\x99\x48\xf7\xf9",  # cqo; idiv rcx
                "udiv": b"\x48\x31\xd2\x48\xf7\xf1",  # xor rdx,rdx; div rcx
                "smod": b"\x48\x99\x48\xf7\xf9\x48\x89\xd0",  # ...; mov rax, rdx
                "umod": b"\x48\x31\xd2\x48\xf7\xf1\x48\x89\xd0",
            }
            instruction = instructions.get(expression.operator)
            if instruction is None:
                raise ValueError(
                    f"unknown x86-64 binary operation {expression.operator!r}"
                )
            code.extend(instruction)
            return
        conditions = {
            "eq": 0x94,  # sete
            "ne": 0x95,  # setne
            "lt": 0x9C,  # setl
            "le": 0x9E,  # setle
            "gt": 0x9F,  # setg
            "ge": 0x9D,  # setge
            "ult": 0x92,  # setb
            "ule": 0x96,  # setbe
            "ugt": 0x97,  # seta
            "uge": 0x93,  # setae
        }
        condition = conditions.get(expression.operator)
        if condition is None:
            raise ValueError(
                f"unknown x86-64 comparison {expression.operator!r}"
            )
        code.extend(b"\x48\x39\xc8\x0f" + bytes((condition,)) + b"\xc0\x48\x0f\xb6\xc0")
        return
    if isinstance(expression, FloatToInt):
        _float_expression(code, expression.value, slot_base)
        if expression.signed:
            code.extend(b"\xf2\x48\x0f\x2c\xc0")  # cvttsd2si rax, xmm0 (toward zero)
            return
        # There is no unsigned convert on SSE2. Values below 2**63 go through
        # the signed instruction unchanged; the rest have 2**63 subtracted
        # first (exactly, since 2**63 is a power of two) and the sign bit put
        # back afterwards. Skipping this makes every C conversion of a double
        # above 2**63 to an unsigned 64-bit type saturate to 0x8000000000000000.
        code.extend(b"\x48\xb8" + struct.pack("<Q", 0x43E0000000000000))  # mov rax, 2**63
        code.extend(b"\x66\x48\x0f\x6e\xc8")  # movq xmm1, rax
        code.extend(b"\x66\x0f\x2e\xc1")  # ucomisd xmm0, xmm1
        code.extend(b"\x73\x07")  # jae +7 (past the small-value path)
        code.extend(b"\xf2\x48\x0f\x2c\xc0")  # cvttsd2si rax, xmm0
        code.extend(b"\xeb\x16")  # jmp +22 (past the large-value path)
        code.extend(b"\xf2\x0f\x5c\xc1")  # subsd xmm0, xmm1
        code.extend(b"\xf2\x48\x0f\x2c\xc0")  # cvttsd2si rax, xmm0
        code.extend(b"\x48\xb9" + struct.pack("<Q", 0x8000000000000000))  # mov rcx, sign
        code.extend(b"\x48\x31\xc8")  # xor rax, rcx
        return
    if isinstance(expression, FloatBits):
        _float_expression(code, expression.value, slot_base)
        if expression.size == 8:
            code.extend(b"\x66\x48\x0f\x7e\xc0")  # movq rax, xmm0
        elif expression.size == 4:
            code.extend(b"\xf2\x0f\x5a\xc0")  # cvtsd2ss xmm0, xmm0
            code.extend(b"\x66\x0f\x7e\xc0")  # movd eax, xmm0 (zero-extends)
        else:
            raise ValueError(f"unsupported x86-64 float bit width {expression.size}")
        return
    if isinstance(expression, FloatCompare):
        _float_expression(code, expression.left, slot_base)
        code.extend(b"\x48\x83\xec\x10\xf2\x0f\x11\x04\x24")  # sub rsp,16; movsd [rsp],xmm0
        _float_expression(code, expression.right, slot_base)
        code.extend(b"\xf2\x0f\x10\xc8")  # movsd xmm1, xmm0  (right operand)
        code.extend(b"\xf2\x0f\x10\x04\x24\x48\x83\xc4\x10")  # movsd xmm0,[rsp]; add rsp,16
        # UCOMISD reports "unordered" (either operand NaN) as ZF=PF=CF=1, which
        # is indistinguishable from equality on ZF and CF alone. C requires
        # every relational operator to be false for an unordered pair, so the
        # orderings are taken with the operands swapped -- SETA/SETAE are false
        # when CF is set -- and equality also tests PF.
        operator = expression.operator
        if operator in {"lt", "le"}:
            code.extend(b"\x66\x0f\x2e\xc8")  # ucomisd xmm1, xmm0
        else:
            code.extend(b"\x66\x0f\x2e\xc1")  # ucomisd xmm0, xmm1
        if operator == "eq":
            code.extend(b"\x0f\x94\xc0")  # sete  al   (ZF=1)
            code.extend(b"\x0f\x9b\xc1")  # setnp cl   (ordered)
            code.extend(b"\x20\xc8")  # and al, cl
        elif operator == "ne":
            code.extend(b"\x0f\x95\xc0")  # setne al  (ZF=0)
            code.extend(b"\x0f\x9a\xc1")  # setp  cl  (unordered)
            code.extend(b"\x08\xc8")  # or al, cl
        elif operator in {"lt", "gt"}:
            code.extend(b"\x0f\x97\xc0")  # seta  al  (CF=0 and ZF=0)
        elif operator in {"le", "ge"}:
            code.extend(b"\x0f\x93\xc0")  # setae al  (CF=0)
        else:
            raise ValueError(
                f"unknown x86-64 float comparison {expression.operator!r}"
            )
        code.extend(b"\x48\x0f\xb6\xc0")  # movzx rax, al
        return
    if isinstance(expression, HeapLoad):
        _expression(code, expression.address, slot_base)  # rax = address
        instruction = _X86_LOADS.get((expression.size, bool(expression.signed)))
        if instruction is None:
            raise ValueError(f"unsupported x86-64 heap load size {expression.size}")
        code.extend(instruction)
        return
    raise TypeError(f"unknown x86-64 integer expression {type(expression).__name__}")


def _float_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _float_expression(
    code: bytearray, expression: FloatExpression, slot_base: int
) -> None:
    """Place one IEEE-754 double expression in XMM0."""

    if isinstance(expression, FloatConstant):
        code.extend(b"\x48\xb8" + struct.pack("<Q", _float_bits(expression.value)))
        code.extend(b"\x66\x48\x0f\x6e\xc0")  # movq xmm0, rax
        return
    if isinstance(expression, FloatLoad):
        displacement = slot_base + expression.slot * 8
        code.extend(b"\xf2\x0f\x10\x85" + struct.pack("<i", displacement))  # movsd xmm0,[rbp+disp]
        return
    if isinstance(expression, IntToFloat):
        _expression(code, expression.value, slot_base)
        if expression.signed:
            code.extend(b"\xf2\x48\x0f\x2a\xc0")  # cvtsi2sd xmm0, rax
            return
        # SSE2 has no unsigned convert either. A non-negative value goes
        # straight through; otherwise the value is halved (keeping the low bit
        # as a sticky bit so round-to-nearest still sees it) and the result
        # doubled, which is exact.
        code.extend(b"\x48\x85\xc0")  # test rax, rax
        code.extend(b"\x78\x07")  # js +7 (to the negative path)
        code.extend(b"\xf2\x48\x0f\x2a\xc0")  # cvtsi2sd xmm0, rax
        code.extend(b"\xeb\x16")  # jmp +22 (past the negative path)
        code.extend(b"\x48\x89\xc1")  # mov rcx, rax
        code.extend(b"\x48\xd1\xe9")  # shr rcx, 1
        code.extend(b"\x48\x83\xe0\x01")  # and rax, 1
        code.extend(b"\x48\x09\xc1")  # or rcx, rax
        code.extend(b"\xf2\x48\x0f\x2a\xc1")  # cvtsi2sd xmm0, rcx
        code.extend(b"\xf2\x0f\x58\xc0")  # addsd xmm0, xmm0
        return
    if isinstance(expression, BitsFloat):
        _expression(code, expression.value, slot_base)
        if expression.size == 8:
            code.extend(b"\x66\x48\x0f\x6e\xc0")  # movq xmm0, rax
        elif expression.size == 4:
            code.extend(b"\x66\x0f\x6e\xc0")  # movd xmm0, eax
            code.extend(b"\xf3\x0f\x5a\xc0")  # cvtss2sd xmm0, xmm0 (exact)
        else:
            raise ValueError(f"unsupported x86-64 float bit width {expression.size}")
        return
    if isinstance(expression, FloatUnary):
        _float_expression(code, expression.operand, slot_base)
        if expression.operator == "pos":
            return
        if expression.operator == "neg":
            code.extend(b"\x66\x48\x0f\x7e\xc0")  # movq rax, xmm0
            code.extend(b"\x48\xb9" + struct.pack("<Q", 0x8000000000000000))  # mov rcx, sign bit
            code.extend(b"\x48\x31\xc8")  # xor rax, rcx
            code.extend(b"\x66\x48\x0f\x6e\xc0")  # movq xmm0, rax
            return
        raise ValueError(f"unknown x86-64 float unary operation {expression.operator!r}")
    if isinstance(expression, FloatBinary):
        _float_expression(code, expression.left, slot_base)
        code.extend(b"\x48\x83\xec\x10\xf2\x0f\x11\x04\x24")  # sub rsp,16; movsd [rsp],xmm0
        _float_expression(code, expression.right, slot_base)
        code.extend(b"\xf2\x0f\x10\xc8")  # movsd xmm1, xmm0  (right operand)
        code.extend(b"\xf2\x0f\x10\x04\x24\x48\x83\xc4\x10")  # movsd xmm0,[rsp]; add rsp,16
        instructions = {
            "add": b"\xf2\x0f\x58\xc1",  # addsd xmm0, xmm1
            "sub": b"\xf2\x0f\x5c\xc1",  # subsd xmm0, xmm1
            "mul": b"\xf2\x0f\x59\xc1",  # mulsd xmm0, xmm1
            "div": b"\xf2\x0f\x5e\xc1",  # divsd xmm0, xmm1
        }
        instruction = instructions.get(expression.operator)
        if instruction is None:
            raise ValueError(
                f"unknown x86-64 float binary operation {expression.operator!r}"
            )
        code.extend(instruction)
        return
    raise TypeError(f"unknown x86-64 float expression {type(expression).__name__}")


def _patch_branches(
    code: bytearray,
    labels: dict[str, int],
    patches: list[tuple[int, str]],
) -> None:
    for position, target in patches:
        if target not in labels:
            raise ValueError(f"undefined native IR label {target!r}")
        displacement = labels[target] - (position + 4)
        struct.pack_into("<i", code, position, displacement)


def encode(module: Module, platform: str, code_address: int) -> bytes:
    """Encode native syscalls directly; no text assembly is produced."""
    if module.functions:
        raise ValueError(
            "internal function calls (and therefore recursion) are not "
            "implemented for x86-64 yet; the ARM64 backend implements them"
        )
    if module.static_bytes:
        raise ValueError(
            "static storage (C file-scope variables) is not implemented for "
            "x86-64 yet; the ARM64 backend implements it"
        )
    if platform == "linux":
        write_number, exit_number = 1, 60
        mmap_number, mmap_flags = 9, 0x22  # MAP_PRIVATE | MAP_ANONYMOUS
    elif platform == "darwin":
        write_number, exit_number = 0x02000004, 0x02000001
        mmap_number, mmap_flags = 0x020000C5, 0x1002  # MAP_ANON | MAP_PRIVATE
    else:
        raise ValueError(f"unsupported x86-64 syscall platform: {platform}")

    code = bytearray()
    code.extend(_sub_stack(_frame_bytes(module.stack_slots)))
    code.extend(b"\x48\x89\xe5")  # mov rbp, rsp; stable variable base
    pending_strings: list[tuple[int, bytes]] = []
    labels: dict[str, int] = {}
    branches: list[tuple[int, str]] = []
    for operation in module.operations:
        if isinstance(operation, HeapInit):
            # mmap(addr=0, len, prot=3, flags, fd=-1, off=0); base -> rax
            code.extend(b"\x31\xff")  # xor edi, edi (addr = 0)
            code.extend(b"\x48\xc7\xc6" + struct.pack("<I", operation.size))  # mov rsi, len
            code.extend(b"\xba\x03\x00\x00\x00")  # mov edx, 3 (PROT_READ|WRITE)
            code.extend(b"\x49\xc7\xc2" + struct.pack("<I", mmap_flags))  # mov r10, flags
            code.extend(b"\x49\xc7\xc0\xff\xff\xff\xff")  # mov r8, -1 (fd)
            code.extend(b"\x45\x31\xc9")  # xor r9d, r9d (off = 0)
            code.extend(b"\x48\xc7\xc0" + struct.pack("<I", mmap_number))  # mov rax, mmap
            code.extend(b"\x0f\x05")  # syscall
            code.extend(b"\x48\x89\x85" + struct.pack("<i", operation.slot * 8))  # mov [rbp+off], rax
        elif isinstance(operation, HeapAlloc):
            _expression(code, operation.size, 0)  # rax = size (8-aligned)
            code.extend(b"\x48\x8b\x8d" + struct.pack("<i", operation.bump_slot * 8))  # mov rcx, [rbp+bump]
            code.extend(b"\x48\x89\x8d" + struct.pack("<i", operation.dest_slot * 8))  # mov [rbp+dest], rcx
            code.extend(b"\x48\x01\xc8")  # add rax, rcx (new bump)
            code.extend(b"\x48\x89\x85" + struct.pack("<i", operation.bump_slot * 8))  # mov [rbp+bump], rax
        elif isinstance(operation, HeapStore):
            _expression(code, operation.address, 0)  # rax = address
            code.extend(b"\x50")  # push rax
            _expression(code, operation.value, 0)  # rax = value
            code.extend(b"\x59")  # pop rcx (rcx = address)
            instruction = _X86_STORES.get(operation.size)
            if instruction is None:
                raise ValueError(f"unsupported x86-64 heap store size {operation.size}")
            code.extend(instruction)
        elif isinstance(operation, WriteRuntime):
            _expression(code, operation.length, 0)  # rax = length
            code.extend(b"\x50")  # push rax
            _expression(code, operation.address, 0)  # rax = address
            code.extend(b"\x48\x89\xc6")  # mov rsi, rax (buf)
            code.extend(b"\x5a")  # pop rdx (count = length)
            code.extend(b"\x48\xc7\xc7\x01\x00\x00\x00")  # mov rdi, 1 (stdout)
            code.extend(b"\x48\xc7\xc0" + struct.pack("<I", write_number))  # mov rax, write
            code.extend(b"\x0f\x05")  # syscall
        elif isinstance(operation, Write):
            # mov rax, write; mov rdi, 1; lea rsi, [rip+disp32];
            # mov rdx, len; syscall
            code.extend(_mov_imm32(b"\xc0", write_number))
            code.extend(_mov_imm32(b"\xc7", operation.fd))
            lea_displacement_position = len(code) + 3
            code.extend(b"\x48\x8d\x35\x00\x00\x00\x00")
            code.extend(_mov_imm32(b"\xc2", len(operation.data)))
            code.extend(b"\x0f\x05")
            pending_strings.append((lea_displacement_position, operation.data))
        elif isinstance(operation, Store):
            _expression(code, operation.value, 0)
            code.extend(
                b"\x48\x89\x85" + struct.pack("<i", operation.slot * 8)
            )
        elif isinstance(operation, FloatStore):
            _float_expression(code, operation.value, 0)
            code.extend(
                b"\xf2\x0f\x11\x85" + struct.pack("<i", operation.slot * 8)
            )  # movsd [rbp+disp], xmm0
        elif isinstance(operation, Label):
            labels[operation.name] = len(code)
        elif isinstance(operation, Jump):
            code.extend(b"\xe9\x00\x00\x00\x00")
            branches.append((len(code) - 4, operation.target))
        elif isinstance(operation, JumpIfFalse):
            _expression(code, operation.condition, 0)
            code.extend(b"\x48\x85\xc0\x0f\x84\x00\x00\x00\x00")
            branches.append((len(code) - 4, operation.target))
        elif isinstance(operation, Exit):
            code.extend(_mov_imm32(b"\xc0", exit_number))
            code.extend(_mov_imm32(b"\xc7", operation.status))
            code.extend(b"\x0f\x05")
        elif isinstance(operation, ExitValue):
            _expression(code, operation.value, 0)
            code.extend(b"\x48\x89\xc7")  # mov rdi, rax
            code.extend(_mov_imm32(b"\xc0", exit_number))
            code.extend(b"\x0f\x05")

    _patch_branches(code, labels, branches)
    for displacement_position, data in pending_strings:
        data_offset = len(code)
        # Displacement is relative to the instruction following lea.
        displacement = data_offset - (displacement_position + 4)
        struct.pack_into("<i", code, displacement_position, displacement)
        code += data
    return bytes(code)


def encode_windows(module: Module, code_address: int, imports: dict[str, int]) -> bytes:
    """Encode calls through a Windows x64 import-address table."""
    if module.functions:
        raise ValueError(
            "internal function calls (and therefore recursion) are not "
            "implemented for x86-64 yet; the ARM64 backend implements them"
        )
    if module.static_bytes:
        raise ValueError(
            "static storage (C file-scope variables) is not implemented for "
            "x86-64 yet; the ARM64 backend implements it"
        )
    variable_base = 0x38
    code = bytearray(_sub_stack(_frame_bytes(module.stack_slots, variable_base)))
    code.extend(b"\x48\x89\xe5")  # mov rbp, rsp
    address_patches: list[tuple[int, int]] = []
    string_patches: list[tuple[int, bytes]] = []
    labels: dict[str, int] = {}
    branches: list[tuple[int, str]] = []

    def indirect_call(symbol: str) -> None:
        instruction = len(code)
        code.extend(b"\xff\x15\x00\x00\x00\x00")
        address_patches.append((instruction + 2, imports[symbol]))

    for operation in module.operations:
        if isinstance(operation, (HeapInit, HeapAlloc, WriteRuntime)):
            raise ValueError(
                "runtime heap lists/strings are not supported for windows-x86_64 yet"
            )
        if isinstance(operation, HeapStore):
            # Plain memory, not the arena: a C store through a pointer needs no
            # mmap, so it works on Windows exactly as it does on POSIX.
            _expression(code, operation.address, variable_base)
            code.extend(b"\x50")  # push rax
            _expression(code, operation.value, variable_base)
            code.extend(b"\x59")  # pop rcx (rcx = address)
            instruction = _X86_STORES.get(operation.size)
            if instruction is None:
                raise ValueError(f"unsupported x86-64 heap store size {operation.size}")
            code.extend(instruction)
            continue
        if isinstance(operation, Write):
            # STD_OUTPUT_HANDLE (-11) for fd 1, STD_ERROR_HANDLE (-12) for fd 2.
            handle = -11 if operation.fd == 1 else -12
            code.extend(b"\xb9" + struct.pack("<i", handle))
            indirect_call("GetStdHandle")
            code.extend(b"\x48\x89\xc1")
            displacement_position = len(code) + 3
            code.extend(b"\x48\x8d\x15\x00\x00\x00\x00")
            string_patches.append((displacement_position, operation.data))
            code.extend(b"\x41\xb8" + struct.pack("<I", len(operation.data)))
            code.extend(b"\x4c\x8d\x4c\x24\x28")
            code.extend(b"\x48\xc7\x44\x24\x20\x00\x00\x00\x00")
            indirect_call("WriteFile")
        elif isinstance(operation, Store):
            _expression(code, operation.value, variable_base)
            code.extend(
                b"\x48\x89\x85"
                + struct.pack("<i", variable_base + operation.slot * 8)
            )
        elif isinstance(operation, FloatStore):
            _float_expression(code, operation.value, variable_base)
            code.extend(
                b"\xf2\x0f\x11\x85"
                + struct.pack("<i", variable_base + operation.slot * 8)
            )  # movsd [rbp+disp], xmm0
        elif isinstance(operation, Label):
            labels[operation.name] = len(code)
        elif isinstance(operation, Jump):
            code.extend(b"\xe9\x00\x00\x00\x00")
            branches.append((len(code) - 4, operation.target))
        elif isinstance(operation, JumpIfFalse):
            _expression(code, operation.condition, variable_base)
            code.extend(b"\x48\x85\xc0\x0f\x84\x00\x00\x00\x00")
            branches.append((len(code) - 4, operation.target))
        elif isinstance(operation, Exit):
            code.extend(b"\xb9" + struct.pack("<I", operation.status))
            indirect_call("ExitProcess")
        elif isinstance(operation, ExitValue):
            _expression(code, operation.value, variable_base)
            code.extend(b"\x89\xc1")  # mov ecx, eax
            indirect_call("ExitProcess")

    _patch_branches(code, labels, branches)
    for position, target_address in address_patches:
        next_address = code_address + position + 4
        struct.pack_into("<i", code, position, target_address - next_address)
    for position, data in string_patches:
        data_address = code_address + len(code)
        next_address = code_address + position + 4
        struct.pack_into("<i", code, position, data_address - next_address)
        code.extend(data)
    return bytes(code)
