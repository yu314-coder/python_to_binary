from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .ir import (
    CStringConstant,
    Exit,
    ExitValue,
    ExternCall,
    FloatBinary,
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
    Store,
    Write,
    WriteRuntime,
)


def _mov(register: int, value: int) -> list[int]:
    value &= 0xFFFFFFFFFFFFFFFF
    instructions = [0xD2800000 | ((value & 0xFFFF) << 5) | register]
    for halfword in range(1, 4):
        part = (value >> (halfword * 16)) & 0xFFFF
        if part:
            instructions.append(0xF2800000 | (halfword << 21) | (part << 5) | register)
    return instructions


def _adr(register: int, distance: int) -> int:
    if not -(1 << 20) <= distance < (1 << 20):
        raise ValueError("arm64 literal is outside ADR range")
    encoded = distance & 0x1FFFFF
    return 0x10000000 | ((encoded & 3) << 29) | (((encoded >> 2) & 0x7FFFF) << 5) | register


def _frame_bytes(stack_slots: int, base: int = 0) -> int:
    return (base + stack_slots * 8 + 15) & ~15


def _sub_sp(amount: int) -> int:
    if not 0 <= amount <= 0xFFF:
        raise ValueError("ARM64 native stack frame exceeds immediate range")
    return 0xD10003FF | (amount << 10)


def _slot_instruction(load: bool, slot: int, slot_base: int, rt: int = 0) -> int:
    offset = slot_base + slot * 8
    if offset < 0 or offset > 0x7FF8 or offset & 7:
        raise ValueError("ARM64 native variable is outside stack-slot range")
    # ldr/str x<rt>, [x29, #offset]. Rn=x29 (0x1D) is encoded in bits [9:5].
    base = 0xF9400000 if load else 0xF9000000
    return base | ((offset // 8) << 10) | (29 << 5) | rt


def _float_slot_instruction(load: bool, slot: int, slot_base: int) -> int:
    """Load or store a double between stack slot ``slot`` and D0."""
    offset = slot_base + slot * 8
    if offset < 0 or offset > 0x7FF8 or offset & 7:
        raise ValueError("ARM64 native variable is outside stack-slot range")
    base = 0xFD4003A0 if load else 0xFD0003A0  # ldr/str d0, [x29, #offset]
    return base | ((offset // 8) << 10)


def _condition_code(operator: str) -> int:
    conditions = {
        "eq": 0x0,
        "ne": 0x1,
        "ge": 0xA,
        "lt": 0xB,
        "gt": 0xC,
        "le": 0xD,
    }
    try:
        return conditions[operator]
    except KeyError as error:
        raise ValueError(f"unknown ARM64 comparison {operator!r}") from error


@dataclass
class _Refs:
    """Fixups collected while encoding the darwin dynamic (extern) image.

    ``strings`` records ``(word_index, data, register)`` for ADR loads of
    embedded constant byte blobs; ``externs`` records ``(word_index, symbol)``
    for each ``adrp x16``/``ldr x16``/``blr x16`` GOT call site so the Mach-O
    writer can point it at the bound symbol pointer.
    """

    strings: list[tuple[int, bytes, int]] = field(default_factory=list)
    externs: list[tuple[int, str]] = field(default_factory=list)


def _external_call(
    words: list[int], expression: "ExternCall", slot_base: int, refs: "_Refs | None"
) -> None:
    if refs is None:
        raise TypeError("ARM64 external calls require the darwin dynamic encoder")
    if len(expression.arguments) > 8:
        raise ValueError("ARM64 external calls support at most 8 integer arguments")
    for argument in expression.arguments:
        _expression(words, argument, slot_base, refs)  # arg -> x0
        words.extend((_sub_sp(16), 0xF90003E0))  # str x0, [sp]
    for index in reversed(range(len(expression.arguments))):
        words.append(0xF94003E0 | index)  # ldr x{index}, [sp]
        words.append(0x910043FF)  # add sp, sp, #16
    call_index = len(words)
    # adrp x16, <got page>; ldr x16, [x16, #got_off]; blr x16. The two leading
    # words are placeholders patched by the Mach-O writer once the GOT address
    # is known; the result is returned by the callee in x0.
    words.extend((0x90000010, 0xF9400210, 0xD63F0200))
    refs.externs.append((call_index, expression.symbol))
    # AAPCS64 leaves bits 32-63 of x0 UNSPECIFIED when the callee returns a
    # 32-bit type, so a C ``int`` result must be widened before it is treated
    # as a signed 64-bit value. Skipping this makes CPython's -1 failure return
    # read as 4294967295, so `if (rc < 0)` never fires and a pending exception
    # is silently swallowed.
    if expression.result == "i32":
        words.append(0x93407C00)  # sxtw x0, w0
    elif expression.result == "u32":
        words.append(0x2A0003E0)  # mov w0, w0 (zero-extends into x0)


def _expression(
    words: list[int],
    expression: IntExpression,
    slot_base: int,
    refs: "_Refs | None" = None,
) -> None:
    """Place one signed 64-bit native integer expression in X0."""

    if isinstance(expression, IntConstant):
        words.extend(_mov(0, expression.value))
        return
    if isinstance(expression, IntLoad):
        words.append(_slot_instruction(True, expression.slot, slot_base))
        return
    if isinstance(expression, CStringConstant):
        if refs is None:
            raise TypeError(
                "ARM64 C-string constants require the darwin dynamic encoder"
            )
        index = len(words)
        words.append(0)  # adr x0, <cstring> (patched with the data address)
        refs.strings.append((index, expression.data, 0))
        return
    if isinstance(expression, ExternCall):
        _external_call(words, expression, slot_base, refs)
        return
    if isinstance(expression, IntUnary):
        _expression(words, expression.operand, slot_base, refs)
        if expression.operator == "pos":
            return
        if expression.operator == "neg":
            words.append(0xCB0003E0)  # neg x0, x0
            return
        if expression.operator == "invert":
            words.append(0xAA2003E0)  # mvn x0, x0
            return
        if expression.operator == "not":
            words.extend((0xF100001F, 0x9A9F17E0))  # cmp x0,#0; cset x0,eq
            return
        raise ValueError(f"unknown ARM64 unary operation {expression.operator!r}")
    if isinstance(expression, (IntBinary, IntCompare)):
        _expression(words, expression.left, slot_base, refs)
        words.extend((_sub_sp(16), 0xF90003E0))  # push x0 in a 16-byte slot
        _expression(words, expression.right, slot_base, refs)
        words.extend((0xAA0003E1, 0xF94003E0, 0x910043FF))  # x1=x0; pop x0
        if isinstance(expression, IntBinary):
            instructions = {
                "add": 0x8B010000,
                "sub": 0xCB010000,
                "mul": 0x9B017C00,
                "and": 0x8A010000,
                "or": 0xAA010000,
                "xor": 0xCA010000,
                "lshift": 0x9AC12000,
                "rshift": 0x9AC12800,
            }
            try:
                words.append(instructions[expression.operator])
            except KeyError as error:
                raise ValueError(
                    f"unknown ARM64 binary operation {expression.operator!r}"
                ) from error
            return
        condition = _condition_code(expression.operator)
        # CSET is CSINC with both inputs XZR and the inverted condition.
        words.extend((0xEB01001F, 0x9A9F07E0 | ((condition ^ 1) << 12)))
        return
    if isinstance(expression, FloatToInt):
        _float_expression(words, expression.value, slot_base, refs)
        words.append(0x9E780000)  # fcvtzs x0, d0 (truncate toward zero)
        return
    if isinstance(expression, FloatCompare):
        _float_expression(words, expression.left, slot_base, refs)
        words.extend((_sub_sp(16), 0xFD0003E0))  # str d0, [sp]
        _float_expression(words, expression.right, slot_base, refs)
        words.extend((0x1E604001, 0xFD4003E0, 0x910043FF))  # fmov d1,d0; ldr d0,[sp]; add sp,#16
        words.append(0x1E612000)  # fcmp d0, d1
        condition = _condition_code(expression.operator)
        words.append(0x9A9F07E0 | ((condition ^ 1) << 12))  # cset x0, cond
        return
    if isinstance(expression, HeapLoad):
        _expression(words, expression.address, slot_base, refs)  # x0 = address
        if expression.size == 8:
            words.append(0xF9400000)  # ldr x0, [x0]
        elif expression.size == 1:
            words.append(0x39400000)  # ldrb w0, [x0]
        else:
            raise ValueError(f"unsupported ARM64 heap load size {expression.size}")
        return
    raise TypeError(f"unknown ARM64 integer expression {type(expression).__name__}")


def _float_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _float_expression(
    words: list[int],
    expression: FloatExpression,
    slot_base: int,
    refs: "_Refs | None" = None,
) -> None:
    """Place one IEEE-754 double expression in D0."""

    if isinstance(expression, FloatConstant):
        words.extend(_mov(0, _float_bits(expression.value)))
        words.append(0x9E670000)  # fmov d0, x0
        return
    if isinstance(expression, FloatLoad):
        words.append(_float_slot_instruction(True, expression.slot, slot_base))
        return
    if isinstance(expression, IntToFloat):
        _expression(words, expression.value, slot_base, refs)
        words.append(0x9E620000)  # scvtf d0, x0
        return
    if isinstance(expression, FloatUnary):
        _float_expression(words, expression.operand, slot_base, refs)
        if expression.operator == "pos":
            return
        if expression.operator == "neg":
            words.append(0x1E614000)  # fneg d0, d0
            return
        raise ValueError(f"unknown ARM64 float unary operation {expression.operator!r}")
    if isinstance(expression, FloatBinary):
        _float_expression(words, expression.left, slot_base, refs)
        words.extend((_sub_sp(16), 0xFD0003E0))  # str d0, [sp]
        _float_expression(words, expression.right, slot_base, refs)
        words.extend((0x1E604001, 0xFD4003E0, 0x910043FF))  # fmov d1,d0; ldr d0,[sp]; add sp,#16
        instructions = {
            "add": 0x1E612800,  # fadd d0, d0, d1
            "sub": 0x1E613800,  # fsub d0, d0, d1
            "mul": 0x1E610800,  # fmul d0, d0, d1
            "div": 0x1E611800,  # fdiv d0, d0, d1
        }
        try:
            words.append(instructions[expression.operator])
        except KeyError as error:
            raise ValueError(
                f"unknown ARM64 float binary operation {expression.operator!r}"
            ) from error
        return
    raise TypeError(f"unknown ARM64 float expression {type(expression).__name__}")


def _patch_branches(
    words: list[int],
    labels: dict[str, int],
    branches: list[tuple[int, str, bool]],
) -> None:
    for instruction_index, target, conditional in branches:
        if target not in labels:
            raise ValueError(f"undefined native IR label {target!r}")
        distance = labels[target] - instruction_index
        if conditional:
            if not -(1 << 18) <= distance < (1 << 18):
                raise ValueError("ARM64 conditional branch is outside range")
            words[instruction_index] = 0xB4000000 | ((distance & 0x7FFFF) << 5)
        else:
            if not -(1 << 25) <= distance < (1 << 25):
                raise ValueError("ARM64 branch is outside range")
            words[instruction_index] = 0x14000000 | (distance & 0x3FFFFFF)


def _darwin_encode(module: Module, code_address: int) -> tuple[bytes, list[tuple[int, str]]]:
    return _encode(
        module,
        code_address,
        write_number=4,
        exit_number=1,
        mmap_number=197,
        mmap_flags=0x1002,  # MAP_ANON | MAP_PRIVATE
        svc=0xD4001001,
    )


def encode_linux(module: Module, code_address: int) -> bytes:
    code, externs = _encode(
        module,
        code_address,
        write_number=64,
        exit_number=93,
        mmap_number=222,
        mmap_flags=0x22,  # MAP_PRIVATE | MAP_ANONYMOUS
        svc=0xD4000001,
    )
    if externs:
        raise ValueError("external symbol calls are not supported for linux-arm64")
    return code


def encode_darwin(module: Module, code_address: int) -> bytes:
    code, externs = _darwin_encode(module, code_address)
    if externs:
        raise ValueError(
            "external symbol calls require encode_darwin_extern and the dynamic "
            "Mach-O writer"
        )
    return code


def encode_darwin_extern(
    module: Module, code_address: int
) -> tuple[bytes, list[tuple[int, str]]]:
    """Encode a darwin module that may call external (dyld-bound) symbols.

    Returns the ``.text`` bytes and the ordered extern GOT call sites so the
    dynamic Mach-O writer can lay out the ``__got`` slots and bind opcodes.
    """
    return _darwin_encode(module, code_address)


def encode_windows(module: Module, code_address: int, imports: dict[str, int]) -> bytes:
    """Encode calls through a Windows ARM64 import-address table."""
    slot_base = 16
    words: list[int] = [
        _sub_sp(_frame_bytes(module.stack_slots, slot_base)),
        0x910003FD,  # mov x29, sp; stable variable base
    ]
    function_references: list[tuple[int, str]] = []
    string_references: list[tuple[int, bytes]] = []
    labels: dict[str, int] = {}
    branches: list[tuple[int, str, bool]] = []

    def call(symbol: str) -> None:
        index = len(words)
        words.extend((0, 0, 0xD63F0200))  # adrp x16; ldr x16, [x16,#off]; blr x16
        function_references.append((index, symbol))

    for operation in module.operations:
        if isinstance(operation, (HeapInit, HeapAlloc, HeapStore, WriteRuntime)):
            raise ValueError(
                "runtime heap lists/strings are not supported for windows-arm64 yet"
            )
        if isinstance(operation, Write):
            # STD_OUTPUT_HANDLE (-11) for fd 1, STD_ERROR_HANDLE (-12) for fd 2.
            words.extend(_mov(0, -11 if operation.fd == 1 else -12))
            call("GetStdHandle")
            string_index = len(words)
            words.append(0)  # adr x1, data
            string_references.append((string_index, operation.data))
            words.extend(_mov(2, len(operation.data)))
            words.append(0x910003E3)  # mov x3, sp
            words.extend(_mov(4, 0))
            call("WriteFile")
        elif isinstance(operation, Store):
            _expression(words, operation.value, slot_base)
            words.append(_slot_instruction(False, operation.slot, slot_base))
        elif isinstance(operation, FloatStore):
            _float_expression(words, operation.value, slot_base)
            words.append(_float_slot_instruction(False, operation.slot, slot_base))
        elif isinstance(operation, Label):
            labels[operation.name] = len(words)
        elif isinstance(operation, Jump):
            branches.append((len(words), operation.target, False))
            words.append(0)
        elif isinstance(operation, JumpIfFalse):
            _expression(words, operation.condition, slot_base)
            branches.append((len(words), operation.target, True))
            words.append(0)
        elif isinstance(operation, Exit):
            words.extend(_mov(0, operation.status))
            call("ExitProcess")
        elif isinstance(operation, ExitValue):
            _expression(words, operation.value, slot_base)
            call("ExitProcess")

    _patch_branches(words, labels, branches)
    image = bytearray(struct.pack(f"<{len(words)}I", *words))
    for instruction_index, symbol in function_references:
        instruction_address = code_address + instruction_index * 4
        target = imports[symbol]
        page_delta = (target & ~0xFFF) - (instruction_address & ~0xFFF)
        pages = page_delta >> 12
        if not -(1 << 20) <= pages < (1 << 20):
            raise ValueError("Windows ARM64 import is outside ADRP range")
        encoded = pages & 0x1FFFFF
        adrp = 0x90000010 | ((encoded & 3) << 29) | (((encoded >> 2) & 0x7FFFF) << 5)
        page_offset = target & 0xFFF
        if page_offset & 7:
            raise ValueError("Windows ARM64 IAT entry is not 8-byte aligned")
        ldr = 0xF9400210 | ((page_offset // 8) << 10)
        struct.pack_into("<II", image, instruction_index * 4, adrp, ldr)
    for instruction_index, data in string_references:
        instruction_address = code_address + instruction_index * 4
        data_address = code_address + len(image)
        struct.pack_into(
            "<I",
            image,
            instruction_index * 4,
            _adr(1, data_address - instruction_address),
        )
        image.extend(data)
    return bytes(image)


def _encode(
    module: Module,
    code_address: int,
    write_number: int,
    exit_number: int,
    mmap_number: int,
    mmap_flags: int,
    svc: int,
) -> tuple[bytes, list[tuple[int, str]]]:
    """Encode a module to ARM64 machine code.

    Returns the ``.text`` bytes plus the extern GOT call sites as
    ``(byte_offset_in_text, symbol)`` pairs (empty unless the module contains
    ``ExternCall`` operations, which only the darwin dynamic path allows).
    """
    syscall_register = 8 if svc == 0xD4000001 else 16
    words: list[int] = []
    frame = _frame_bytes(module.stack_slots)
    if frame:
        words.append(_sub_sp(frame))
    words.append(0x910003FD)  # mov x29, sp
    refs = _Refs()
    string_references = refs.strings
    labels: dict[str, int] = {}
    branches: list[tuple[int, str, bool]] = []
    for operation in module.operations:
        if isinstance(operation, HeapInit):
            # x0=addr=0, x1=len, x2=prot RW=3, x3=flags, x4=fd=-1, x5=off=0
            words.extend(_mov(0, 0))
            words.extend(_mov(1, operation.size))
            words.extend(_mov(2, 3))
            words.extend(_mov(3, mmap_flags))
            words.extend(_mov(4, (-1) & 0xFFFFFFFFFFFFFFFF))
            words.extend(_mov(5, 0))
            words.extend(_mov(syscall_register, mmap_number))
            words.append(svc)
            words.append(_slot_instruction(False, operation.slot, 0))  # str x0,[x29,#slot]
            continue
        if isinstance(operation, HeapAlloc):
            _expression(words, operation.size, 0, refs)  # x0 = size (already 8-aligned)
            words.append(_slot_instruction(True, operation.bump_slot, 0, rt=1))  # x1=bump
            words.append(_slot_instruction(False, operation.dest_slot, 0, rt=1))  # dest=bump
            words.append(0x8B000020)  # add x0, x1, x0  (new bump)
            words.append(_slot_instruction(False, operation.bump_slot, 0))  # bump=x0
            continue
        if isinstance(operation, HeapStore):
            _expression(words, operation.address, 0, refs)  # x0 = address
            words.extend((_sub_sp(16), 0xF90003E0))  # str x0, [sp]
            _expression(words, operation.value, 0, refs)  # x0 = value
            words.extend((0xF94003E1, 0x910043FF))  # ldr x1,[sp]; add sp,#16 (x1=address)
            if operation.size == 8:
                words.append(0xF9000020)  # str x0, [x1]
            elif operation.size == 1:
                words.append(0x39000020)  # strb w0, [x1]
            else:
                raise ValueError(f"unsupported ARM64 heap store size {operation.size}")
            continue
        if isinstance(operation, WriteRuntime):
            _expression(words, operation.length, 0, refs)  # x0 = length
            words.extend((_sub_sp(16), 0xF90003E0))  # str x0, [sp]
            _expression(words, operation.address, 0, refs)  # x0 = address
            words.append(0xAA0003E1)  # mov x1, x0  (buf)
            words.extend((0xF94003E2, 0x910043FF))  # ldr x2,[sp]; add sp,#16 (x2=length)
            words.extend(_mov(0, 1))  # fd = stdout
            words.extend(_mov(syscall_register, write_number))
            words.append(svc)
            continue
        if isinstance(operation, Write):
            words.extend(_mov(0, operation.fd))
            adr_index = len(words)
            words.append(0)
            string_references.append((adr_index, operation.data, 1))
            words.extend(_mov(2, len(operation.data)))
            syscall_register = 8 if svc == 0xD4000001 else 16
            words.extend(_mov(syscall_register, write_number))
            words.append(svc)
        elif isinstance(operation, Store):
            _expression(words, operation.value, 0, refs)
            words.append(_slot_instruction(False, operation.slot, 0))
        elif isinstance(operation, FloatStore):
            _float_expression(words, operation.value, 0, refs)
            words.append(_float_slot_instruction(False, operation.slot, 0))
        elif isinstance(operation, Label):
            labels[operation.name] = len(words)
        elif isinstance(operation, Jump):
            branches.append((len(words), operation.target, False))
            words.append(0)
        elif isinstance(operation, JumpIfFalse):
            _expression(words, operation.condition, 0, refs)
            branches.append((len(words), operation.target, True))
            words.append(0)
        elif isinstance(operation, Exit):
            words.extend(_mov(0, operation.status))
            syscall_register = 8 if svc == 0xD4000001 else 16
            words.extend(_mov(syscall_register, exit_number))
            words.append(svc)
        elif isinstance(operation, ExitValue):
            _expression(words, operation.value, 0, refs)
            syscall_register = 8 if svc == 0xD4000001 else 16
            words.extend(_mov(syscall_register, exit_number))
            words.append(svc)
    _patch_branches(words, labels, branches)
    externs = [(index * 4, symbol) for index, symbol in refs.externs]
    image = bytearray(struct.pack(f"<{len(words)}I", *words))
    for instruction_index, data, register in string_references:
        instruction_address = code_address + instruction_index * 4
        data_address = code_address + len(image)
        struct.pack_into(
            "<I",
            image,
            instruction_index * 4,
            _adr(register, data_address - instruction_address),
        )
        image.extend(data)
    return bytes(image), externs
