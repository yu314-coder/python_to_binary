from __future__ import annotations

import dataclasses
import struct

from .ir import (
    BitsFloat,
    Call,
    Function,
    FunctionAddress,
    GlobalAddress,
    IndirectCall,
    Return,
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


@dataclasses.dataclass
class _X86Refs:
    """Fixups collected while encoding: call sites and function addresses.

    Both are PC-relative and can only be resolved once every body's offset in
    .text is known, exactly like the existing branch patcher.
    """

    calls: list = dataclasses.field(default_factory=list)
    addresses: list = dataclasses.field(default_factory=list)
    #: Integer argument registers: six under System V, four under Microsoft x64.
    registers: int = 6
    #: Bytes the caller reserves below the stack arguments. Microsoft x64
    #: requires 32 so a callee can spill rcx-r9; System V has none.
    shadow: int = 0


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


def _expression(
    code: bytearray,
    expression: IntExpression,
    slot_base: int,
    refs: "_X86Refs | None" = None,
) -> None:
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
        _expression(code, expression.operand, slot_base, refs)
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
        _expression(code, expression.left, slot_base, refs)
        # A 16-byte spill, not "push rax": System V requires rsp % 16 == 0
        # at every call, and a call in the right operand would otherwise
        # run with rsp misaligned by 8.
        code.extend(b"\x48\x83\xec\x10\x48\x89\x04\x24")  # sub rsp,16; mov [rsp],rax
        _expression(code, expression.right, slot_base, refs)
        code.extend(b"\x48\x89\xc1")  # mov rcx, rax
        code.extend(b"\x48\x8b\x04\x24\x48\x83\xc4\x10")  # mov rax,[rsp]; add rsp,16
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
        _float_expression(code, expression.value, slot_base, refs)
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
        _float_expression(code, expression.value, slot_base, refs)
        if expression.size == 8:
            code.extend(b"\x66\x48\x0f\x7e\xc0")  # movq rax, xmm0
        elif expression.size == 4:
            code.extend(b"\xf2\x0f\x5a\xc0")  # cvtsd2ss xmm0, xmm0
            code.extend(b"\x66\x0f\x7e\xc0")  # movd eax, xmm0 (zero-extends)
        else:
            raise ValueError(f"unsupported x86-64 float bit width {expression.size}")
        return
    if isinstance(expression, FloatCompare):
        _float_expression(code, expression.left, slot_base, refs)
        code.extend(b"\x48\x83\xec\x10\xf2\x0f\x11\x04\x24")  # sub rsp,16; movsd [rsp],xmm0
        _float_expression(code, expression.right, slot_base, refs)
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
        _expression(code, expression.address, slot_base, refs)  # rax = address
        instruction = _X86_LOADS.get((expression.size, bool(expression.signed)))
        if instruction is None:
            raise ValueError(f"unsupported x86-64 heap load size {expression.size}")
        code.extend(instruction)
        return
    if isinstance(expression, Call):
        if refs is None:
            raise ValueError("x86-64 calls need the function-aware encoder")
        _direct_call(code, expression, slot_base, refs)
        return
    if isinstance(expression, IndirectCall):
        if refs is None:
            raise ValueError("x86-64 calls need the function-aware encoder")
        _indirect_call_x86(code, expression, slot_base, refs)
        return
    if isinstance(expression, FunctionAddress):
        if refs is None:
            raise ValueError("x86-64 calls need the function-aware encoder")
        # lea rax, [rip+rel32]: PC-relative, so the image may still be slid.
        code.extend(b"\x48\x8d\x05\x00\x00\x00\x00")
        refs.addresses.append((len(code) - 4, expression.name))
        return
    if isinstance(expression, GlobalAddress):
        # The static block's base lives in r15 for the whole run.
        code.extend(b"\x4c\x89\xf8")  # mov rax, r15
        if expression.offset:
            code.extend(b"\x48\x05" + struct.pack("<i", expression.offset))
        return
    raise TypeError(f"unknown x86-64 integer expression {type(expression).__name__}")


def _float_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _float_expression(
    code: bytearray,
    expression: FloatExpression,
    slot_base: int,
    refs: "_X86Refs | None" = None,
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
        _expression(code, expression.value, slot_base, refs)
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
        _expression(code, expression.value, slot_base, refs)
        if expression.size == 8:
            code.extend(b"\x66\x48\x0f\x6e\xc0")  # movq xmm0, rax
        elif expression.size == 4:
            code.extend(b"\x66\x0f\x6e\xc0")  # movd xmm0, eax
            code.extend(b"\xf3\x0f\x5a\xc0")  # cvtss2sd xmm0, xmm0 (exact)
        else:
            raise ValueError(f"unsupported x86-64 float bit width {expression.size}")
        return
    if isinstance(expression, FloatUnary):
        _float_expression(code, expression.operand, slot_base, refs)
        if expression.operator == "pos":
            return
        if expression.operator == "sqrt":
            code.extend(b"\xf2\x0f\x51\xc0")  # sqrtsd xmm0, xmm0
            return
        if expression.operator == "abs":
            # Clear the sign bit through the integer register file, which needs
            # no constant pool.
            code.extend(b"\x66\x48\x0f\x7e\xc0")  # movq rax, xmm0
            code.extend(b"\x48\x0f\xba\xf0\x3f")  # btr rax, 63
            code.extend(b"\x66\x48\x0f\x6e\xc0")  # movq xmm0, rax
            return
        # roundsd has no ties-away-from-zero mode, so C round() is not a
        # single instruction here and is rejected by the front end.
        rounding = {"floor": 1, "ceil": 2, "trunc": 3}
        if expression.operator in rounding:
            # roundsd xmm0, xmm0, imm8 (SSE4.1, universal on x86-64 since 2008)
            code.extend(b"\x66\x0f\x3a\x0b\xc0" + bytes((rounding[expression.operator],)))
            return
        if expression.operator == "neg":
            code.extend(b"\x66\x48\x0f\x7e\xc0")  # movq rax, xmm0
            code.extend(b"\x48\xb9" + struct.pack("<Q", 0x8000000000000000))  # mov rcx, sign bit
            code.extend(b"\x48\x31\xc8")  # xor rax, rcx
            code.extend(b"\x66\x48\x0f\x6e\xc0")  # movq xmm0, rax
            return
        raise ValueError(f"unknown x86-64 float unary operation {expression.operator!r}")
    if isinstance(expression, FloatBinary):
        _float_expression(code, expression.left, slot_base, refs)
        code.extend(b"\x48\x83\xec\x10\xf2\x0f\x11\x04\x24")  # sub rsp,16; movsd [rsp],xmm0
        _float_expression(code, expression.right, slot_base, refs)
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



# --- System V AMD64 calls ----------------------------------------------------
#
# Integer and pointer arguments go in rdi, rsi, rdx, rcx, r8, r9 and the result
# comes back in rax. rsp must be a multiple of 16 immediately before every
# call, which is why every spill in this backend moves rsp by 16.

X86_ARGUMENT_REGISTERS = 6

# mov <reg>, [rsp] for each argument register, in order.
_LOAD_ARGUMENT = (
    b"\x48\x8b\x3c\x24",  # rdi
    b"\x48\x8b\x34\x24",  # rsi
    b"\x48\x8b\x14\x24",  # rdx
    b"\x48\x8b\x0c\x24",  # rcx
    b"\x4c\x8b\x04\x24",  # r8
    b"\x4c\x8b\x0c\x24",  # r9
)
_SPILL = b"\x48\x83\xec\x10\x48\x89\x04\x24"  # sub rsp,16; mov [rsp],rax
# mov <reg>, [rsp + disp32] for each argument register, in order.
_LOAD_ARGUMENT_AT = (
    b"\x48\x8b\xbc\x24",  # rdi
    b"\x48\x8b\xb4\x24",  # rsi
    b"\x48\x8b\x94\x24",  # rdx
    b"\x48\x8b\x8c\x24",  # rcx
    b"\x4c\x8b\x84\x24",  # r8
    b"\x4c\x8b\x8c\x24",  # r9
)
_DROP = b"\x48\x83\xc4\x10"  # add rsp, 16


def _push_arguments(
    code: bytearray,
    arguments,
    slot_base: int,
    refs: "_X86Refs",
) -> int:
    """Evaluate arguments left to right and deliver them in the ABI registers.

    Each result is spilled as it is computed, because evaluating a later
    argument would otherwise clobber rax, and a later argument may itself
    contain a call. The spills are unwound in reverse so each value lands in
    its own register.
    """

    count = len(arguments)
    registers = refs.registers
    shadow = refs.shadow
    for argument in arguments:
        _expression(code, argument, slot_base, refs)
        code.extend(_SPILL)
    # Cell for argument i sits at [rsp + (count - 1 - i) * 16].
    overflow = max(0, count - registers)
    area = ((overflow * 8 + shadow) + 15) & ~15
    if area:
        code.extend(b"\x48\x81\xec" + struct.pack("<I", area))  # sub rsp, area
        for index in range(registers, count):
            source = area + (count - 1 - index) * 16
            destination = shadow + (index - registers) * 8
            # mov r11, [rsp+src]; mov [rsp+dst], r11   (r11 is caller-saved and
            # is not an argument register in either convention)
            code.extend(b"\x4c\x8b\x9c\x24" + struct.pack("<I", source))
            code.extend(b"\x4c\x89\x9c\x24" + struct.pack("<I", destination))
    for index in range(min(count, registers)):
        offset = area + (count - 1 - index) * 16
        code.extend(_LOAD_ARGUMENT_AT[index] + struct.pack("<I", offset))
    return area + count * 16


def _direct_call(
    code: bytearray, expression, slot_base: int, refs: "_X86Refs"
) -> None:
    allocated = _push_arguments(code, expression.arguments, slot_base, refs)
    code.extend(b"\xe8\x00\x00\x00\x00")  # call rel32, patched after layout
    refs.calls.append((len(code) - 4, expression.name))
    if allocated:
        code.extend(b"\x48\x81\xc4" + struct.pack("<I", allocated))


def _indirect_call_x86(
    code: bytearray, expression, slot_base: int, refs: "_X86Refs"
) -> None:
    # The target is evaluated first and pinned, so a target expression with a
    # side effect happens exactly once and cannot be clobbered by an argument.
    _expression(code, expression.target, slot_base, refs)
    code.extend(_SPILL)
    allocated = _push_arguments(code, expression.arguments, slot_base, refs)
    # The target's own cell sits just above everything the arguments allocated.
    code.extend(b"\x4c\x8b\x9c\x24" + struct.pack("<I", allocated))  # mov r11,[rsp+n]
    if allocated:
        code.extend(b"\x48\x81\xc4" + struct.pack("<I", allocated))
    code.extend(b"\x48\x83\xc4\x10")  # add rsp, 16  (release the target cell)
    code.extend(b"\x41\xff\xd3")  # call r11



# mov <arg register>, rax is not needed; parameters arrive in the ABI
# registers and are stored straight into the frame's first slots.
_STORE_PARAMETER = (
    b"\x48\x89\xb8",  # mov [rax+disp32], rdi  -- rewritten below per slot
)



@dataclasses.dataclass
class _Syscalls86:
    write_number: int
    exit_number: int
    mmap_number: int
    mmap_flags: int


def _emit_x86_operations(
    code: bytearray,
    operations,
    slot_base: int,
    refs: "_X86Refs",
    pending_strings: list,
    system: "_Syscalls86",
) -> None:
    """Encode one body's operations. Labels are body-local; strings are shared."""

    labels: dict[str, int] = {}
    branches: list[tuple[int, str]] = []
    for operation in operations:
        if isinstance(operation, HeapInit):
            code.extend(b"\x31\xff")  # xor edi, edi (addr = 0)
            code.extend(b"\x48\xc7\xc6" + struct.pack("<I", operation.size))
            code.extend(b"\xba\x03\x00\x00\x00")  # mov edx, 3
            code.extend(b"\x49\xc7\xc2" + struct.pack("<I", system.mmap_flags))
            code.extend(b"\x49\xc7\xc0\xff\xff\xff\xff")  # mov r8, -1
            code.extend(b"\x45\x31\xc9")  # xor r9d, r9d
            code.extend(b"\x48\xc7\xc0" + struct.pack("<I", system.mmap_number))
            code.extend(b"\x0f\x05")  # syscall
            code.extend(b"\x48\x89\x85" + struct.pack("<i", slot_base + operation.slot * 8))
        elif isinstance(operation, HeapAlloc):
            _expression(code, operation.size, slot_base, refs)
            code.extend(b"\x48\x8b\x8d" + struct.pack("<i", slot_base + operation.bump_slot * 8))
            code.extend(b"\x48\x89\x8d" + struct.pack("<i", slot_base + operation.dest_slot * 8))
            code.extend(b"\x48\x01\xc8")
            code.extend(b"\x48\x89\x85" + struct.pack("<i", slot_base + operation.bump_slot * 8))
        elif isinstance(operation, HeapStore):
            _expression(code, operation.address, slot_base, refs)
            code.extend(_SPILL)  # 16 bytes: a call may appear in the value
            _expression(code, operation.value, slot_base, refs)
            code.extend(b"\x48\x8b\x0c\x24")  # mov rcx, [rsp] (address)
            code.extend(_DROP)
            instruction = _X86_STORES.get(operation.size)
            if instruction is None:
                raise ValueError(f"unsupported x86-64 heap store size {operation.size}")
            code.extend(instruction)
        elif isinstance(operation, WriteRuntime):
            _expression(code, operation.length, slot_base, refs)
            code.extend(_SPILL)
            _expression(code, operation.address, slot_base, refs)
            code.extend(b"\x48\x89\xc6")  # mov rsi, rax (buf)
            code.extend(b"\x48\x8b\x14\x24")  # mov rdx, [rsp] (count)
            code.extend(_DROP)
            code.extend(b"\x48\xc7\xc7\x01\x00\x00\x00")  # mov rdi, 1
            code.extend(b"\x48\xc7\xc0" + struct.pack("<I", system.write_number))
            code.extend(b"\x0f\x05")
        elif isinstance(operation, Write):
            code.extend(_mov_imm32(b"\xc0", system.write_number))
            code.extend(_mov_imm32(b"\xc7", operation.fd))
            position = len(code) + 3
            code.extend(b"\x48\x8d\x35\x00\x00\x00\x00")
            code.extend(_mov_imm32(b"\xc2", len(operation.data)))
            code.extend(b"\x0f\x05")
            pending_strings.append((position, operation.data))
        elif isinstance(operation, Store):
            _expression(code, operation.value, slot_base, refs)
            code.extend(b"\x48\x89\x85" + struct.pack("<i", slot_base + operation.slot * 8))
        elif isinstance(operation, FloatStore):
            _float_expression(code, operation.value, slot_base, refs)
            code.extend(b"\xf2\x0f\x11\x85" + struct.pack("<i", slot_base + operation.slot * 8))
        elif isinstance(operation, Label):
            labels[operation.name] = len(code)
        elif isinstance(operation, Jump):
            code.extend(b"\xe9\x00\x00\x00\x00")
            branches.append((len(code) - 4, operation.target))
        elif isinstance(operation, JumpIfFalse):
            _expression(code, operation.condition, slot_base, refs)
            code.extend(b"\x48\x85\xc0\x0f\x84\x00\x00\x00\x00")
            branches.append((len(code) - 4, operation.target))
        elif isinstance(operation, Return):
            if operation.value is not None:
                _expression(code, operation.value, slot_base, refs)
            code.extend(b"\x48\x89\xec\x5d\xc3")  # mov rsp,rbp; pop rbp; ret
        elif isinstance(operation, Exit):
            code.extend(_mov_imm32(b"\xc0", system.exit_number))
            code.extend(_mov_imm32(b"\xc7", operation.status))
            code.extend(b"\x0f\x05")
        elif isinstance(operation, ExitValue):
            _expression(code, operation.value, slot_base, refs)
            code.extend(b"\x48\x89\xc7")  # mov rdi, rax
            code.extend(_mov_imm32(b"\xc0", system.exit_number))
            code.extend(b"\x0f\x05")
    _patch_branches(code, labels, branches)


def _emit_function(
    code: bytearray,
    function: Function,
    refs: "_X86Refs",
    pending_strings: list,
    system: "_Syscalls86",
) -> None:
    """Encode one callable body with a System V AMD64 frame.

    The frame is `push rbp; mov rbp, rsp; sub rsp, N`, so the saved frame
    pointer and return address sit above rbp and the function's slots below it.
    Slots are addressed at a negative displacement from rbp, which is what
    ``slot_base`` carries. N is a multiple of 16, so rsp stays 16-aligned and
    every nested call meets the ABI's alignment rule.
    """

    if function.parameters > function.stack_slots:
        raise ValueError(
            f"x86-64 function {function.name!r} has fewer stack slots than parameters"
        )
    frame = (function.stack_slots * 8 + 15) & ~15
    code.extend(b"\x55")  # push rbp
    code.extend(b"\x48\x89\xe5")  # mov rbp, rsp
    if frame:
        code.extend(b"\x48\x81\xec" + struct.pack("<I", frame))
    slot_base = -frame
    # Spill the incoming argument registers into slots 0..parameters-1 so the
    # body reads a parameter exactly like any other local.
    shadow = 0  # System V has no shadow space; Microsoft x64 reserves 32 bytes
    stores = (
        b"\x48\x89\xbd",  # mov [rbp+disp32], rdi
        b"\x48\x89\xb5",  # rsi
        b"\x48\x89\x95",  # rdx
        b"\x48\x89\x8d",  # rcx
        b"\x4c\x89\x85",  # r8
        b"\x4c\x89\x8d",  # r9
    )
    for index in range(min(function.parameters, len(stores))):
        code.extend(stores[index] + struct.pack("<i", slot_base + index * 8))
    # Arguments past the register count arrived in memory. [rbp] holds the
    # saved frame pointer and [rbp+8] the return address, so the caller's
    # outgoing area starts at [rbp+16].
    for index in range(len(stores), function.parameters):
        incoming = 16 + shadow + (index - len(stores)) * 8
        code.extend(b"\x48\x8b\x85" + struct.pack("<i", incoming))  # mov rax,[rbp+in]
        code.extend(b"\x48\x89\x85" + struct.pack("<i", slot_base + index * 8))
    _emit_x86_operations(
        code, function.operations, slot_base, refs, pending_strings, system
    )
    # Fall off the end: return an unspecified value, as the IR allows.
    code.extend(b"\x48\x89\xec")  # mov rsp, rbp
    code.extend(b"\x5d")  # pop rbp
    code.extend(b"\xc3")  # ret


def encode(module: Module, platform: str, code_address: int) -> bytes:
    """Encode native syscalls directly; no text assembly is produced."""
    if platform == "linux":
        write_number, exit_number = 1, 60
        mmap_number, mmap_flags = 9, 0x22  # MAP_PRIVATE | MAP_ANONYMOUS
    elif platform == "darwin":
        write_number, exit_number = 0x02000004, 0x02000001
        mmap_number, mmap_flags = 0x020000C5, 0x1002  # MAP_ANON | MAP_PRIVATE
    else:
        raise ValueError(f"unsupported x86-64 syscall platform: {platform}")
    system = _Syscalls86(write_number, exit_number, mmap_number, mmap_flags)

    code = bytearray()
    code.extend(_sub_stack(_frame_bytes(module.stack_slots)))
    code.extend(b"\x48\x89\xe5")  # mov rbp, rsp; stable variable base
    if module.static_bytes:
        # One anonymous read/write mapping for every file-scope object, with
        # its base parked in r15 for the whole run. r15 is callee-saved and
        # this backend never writes it elsewhere, so a global keeps its address
        # across calls.
        code.extend(b"\x31\xff")  # xor edi, edi
        code.extend(b"\x48\xc7\xc6" + struct.pack("<I", module.static_bytes))
        code.extend(b"\xba\x03\x00\x00\x00")  # mov edx, PROT_READ|PROT_WRITE
        code.extend(b"\x49\xc7\xc2" + struct.pack("<I", mmap_flags))
        code.extend(b"\x49\xc7\xc0\xff\xff\xff\xff")  # mov r8, -1
        code.extend(b"\x45\x31\xc9")  # xor r9d, r9d
        code.extend(b"\x48\xc7\xc0" + struct.pack("<I", mmap_number))
        code.extend(b"\x0f\x05")  # syscall
        code.extend(b"\x49\x89\xc7")  # mov r15, rax

    refs = _X86Refs()
    pending_strings: list[tuple[int, bytes]] = []
    _emit_x86_operations(
        code, module.operations, 0, refs, pending_strings, system
    )

    offsets: dict[str, int] = {}
    for function in module.functions:
        if function.name in offsets:
            raise ValueError(f"duplicate native IR function {function.name!r}")
        offsets[function.name] = len(code)
        _emit_function(code, function, refs, pending_strings, system)

    # call rel32 and lea rip-relative are both relative to the END of the
    # instruction, which is the four displacement bytes plus nothing more.
    for position, name in refs.calls:
        if name not in offsets:
            raise ValueError(f"call to undefined native IR function {name!r}")
        struct.pack_into("<i", code, position, offsets[name] - (position + 4))
    for position, name in refs.addresses:
        if name not in offsets:
            raise ValueError(
                f"address taken of undefined native IR function {name!r}"
            )
        struct.pack_into("<i", code, position, offsets[name] - (position + 4))

    for displacement_position, data in pending_strings:
        data_offset = len(code)
        displacement = data_offset - (displacement_position + 4)
        struct.pack_into("<i", code, displacement_position, displacement)
        code += data
    return bytes(code)


def encode_windows(module: Module, code_address: int, imports: dict[str, int]) -> bytes:
    """Encode calls through a Windows x64 import-address table."""
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

    refs = _X86Refs(registers=4, shadow=32)

    if module.static_bytes:
        # VirtualAlloc(NULL, size, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE)
        # The result is a zero-filled writable block; its base stays in
        # r15, which is callee-saved and untouched elsewhere here.
        code.extend(b"\x48\x31\xc9")  # xor rcx, rcx (lpAddress = NULL)
        code.extend(b"\x48\xc7\xc2" + struct.pack("<I", module.static_bytes))
        code.extend(b"\x41\xb8\x00\x30\x00\x00")  # mov r8d, 0x3000
        code.extend(b"\x41\xb9\x04\x00\x00\x00")  # mov r9d, PAGE_READWRITE
        code.extend(b"\x48\x83\xec\x20")  # shadow space
        indirect_call("VirtualAlloc")
        code.extend(b"\x48\x83\xc4\x20")
        code.extend(b"\x49\x89\xc7")  # mov r15, rax

    def emit(operations, slot_base: int, _epilogue=b"") -> None:
        """Encode one body. Labels are body-local; imports are shared."""

        nonlocal labels, branches
        labels = {}
        branches = []
        for operation in operations:
            if isinstance(operation, Return):
                if operation.value is not None:
                    _expression(code, operation.value, slot_base, refs)
                code.extend(_epilogue)
                continue
            if isinstance(operation, (HeapInit, HeapAlloc, WriteRuntime)):
                raise ValueError(
                    "runtime heap lists/strings are not supported for windows-x86_64 yet"
                )
            if isinstance(operation, HeapStore):
                # Plain memory, not the arena: a C store through a pointer needs no
                # mmap, so it works on Windows exactly as it does on POSIX.
                _expression(code, operation.address, slot_base, refs)
                code.extend(b"\x50")  # push rax
                _expression(code, operation.value, slot_base, refs)
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
                _expression(code, operation.value, slot_base, refs)
                code.extend(
                    b"\x48\x89\x85"
                    + struct.pack("<i", slot_base + operation.slot * 8)
                )
            elif isinstance(operation, FloatStore):
                _float_expression(code, operation.value, slot_base, refs)
                code.extend(
                    b"\xf2\x0f\x11\x85"
                    + struct.pack("<i", slot_base + operation.slot * 8)
                )  # movsd [rbp+disp], xmm0
            elif isinstance(operation, Label):
                labels[operation.name] = len(code)
            elif isinstance(operation, Jump):
                code.extend(b"\xe9\x00\x00\x00\x00")
                branches.append((len(code) - 4, operation.target))
            elif isinstance(operation, JumpIfFalse):
                _expression(code, operation.condition, slot_base, refs)
                code.extend(b"\x48\x85\xc0\x0f\x84\x00\x00\x00\x00")
                branches.append((len(code) - 4, operation.target))
            elif isinstance(operation, Exit):
                code.extend(b"\xb9" + struct.pack("<I", operation.status))
                indirect_call("ExitProcess")
            elif isinstance(operation, ExitValue):
                _expression(code, operation.value, slot_base, refs)
                code.extend(b"\x89\xc1")  # mov ecx, eax
                indirect_call("ExitProcess")

        _patch_branches(code, labels, branches)

    emit(module.operations, variable_base)

    # Microsoft x64: arguments in rcx, rdx, r8, r9; the caller also owns 32
    # bytes of shadow space, which each body reserves as part of its frame.
    offsets: dict[str, int] = {}
    for function in module.functions:
        if function.name in offsets:
            raise ValueError(f"duplicate native IR function {function.name!r}")
        offsets[function.name] = len(code)
        frame = (function.stack_slots * 8 + 15) & ~15
        code.extend(b"\x55\x48\x89\xe5")  # push rbp; mov rbp, rsp
        if frame:
            code.extend(b"\x48\x81\xec" + struct.pack("<I", frame))
        stores = (
            b"\x48\x89\x8d",  # mov [rbp+disp32], rcx
            b"\x48\x89\x95",  # rdx
            b"\x4c\x89\x85",  # r8
            b"\x4c\x89\x8d",  # r9
        )
        for index in range(min(function.parameters, len(stores))):
            code.extend(stores[index] + struct.pack("<i", -frame + index * 8))
        # Past the fourth, arguments arrive in memory above the caller's 32
        # bytes of shadow space: [rbp] is the saved frame pointer, [rbp+8] the
        # return address, [rbp+16..47] the shadow area.
        for index in range(len(stores), function.parameters):
            incoming = 16 + 32 + (index - len(stores)) * 8
            code.extend(b"\x48\x8b\x85" + struct.pack("<i", incoming))
            code.extend(b"\x48\x89\x85" + struct.pack("<i", -frame + index * 8))
        epilogue = b"\x48\x89\xec\x5d\xc3"  # mov rsp,rbp; pop rbp; ret
        emit(function.operations, -frame, epilogue)
        code.extend(epilogue)
    for position, name in refs.calls:
        if name not in offsets:
            raise ValueError(f"call to undefined native IR function {name!r}")
        struct.pack_into("<i", code, position, offsets[name] - (position + 4))
    for position, name in refs.addresses:
        if name not in offsets:
            raise ValueError(f"address taken of undefined function {name!r}")
        struct.pack_into("<i", code, position, offsets[name] - (position + 4))
    for position, target_address in address_patches:
        next_address = code_address + position + 4
        struct.pack_into("<i", code, position, target_address - next_address)
    for position, data in string_patches:
        data_address = code_address + len(code)
        next_address = code_address + position + 4
        struct.pack_into("<i", code, position, data_address - next_address)
        code.extend(data)
    return bytes(code)
