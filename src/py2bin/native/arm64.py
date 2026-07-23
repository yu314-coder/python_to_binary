from __future__ import annotations

import struct

from .ir import (
    Exit,
    ExitValue,
    IntBinary,
    IntCompare,
    IntConstant,
    IntExpression,
    IntLoad,
    IntUnary,
    Jump,
    JumpIfFalse,
    Label,
    Module,
    Store,
    Write,
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


def _slot_instruction(load: bool, slot: int, slot_base: int) -> int:
    offset = slot_base + slot * 8
    if offset < 0 or offset > 0x7FF8 or offset & 7:
        raise ValueError("ARM64 native variable is outside stack-slot range")
    base = 0xF94003A0 if load else 0xF90003A0
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


def _expression(words: list[int], expression: IntExpression, slot_base: int) -> None:
    """Place one signed 64-bit native integer expression in X0."""

    if isinstance(expression, IntConstant):
        words.extend(_mov(0, expression.value))
        return
    if isinstance(expression, IntLoad):
        words.append(_slot_instruction(True, expression.slot, slot_base))
        return
    if isinstance(expression, IntUnary):
        _expression(words, expression.operand, slot_base)
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
        _expression(words, expression.left, slot_base)
        words.extend((_sub_sp(16), 0xF90003E0))  # push x0 in a 16-byte slot
        _expression(words, expression.right, slot_base)
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
    raise TypeError(f"unknown ARM64 integer expression {type(expression).__name__}")


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


def encode_linux(module: Module, code_address: int) -> bytes:
    return _encode(module, code_address, write_number=64, exit_number=93, svc=0xD4000001)


def encode_darwin(module: Module, code_address: int) -> bytes:
    return _encode(module, code_address, write_number=4, exit_number=1, svc=0xD4001001)


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
        if isinstance(operation, Write):
            words.extend(_mov(0, -11))  # STD_OUTPUT_HANDLE
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
    svc: int,
) -> bytes:
    words: list[int] = []
    frame = _frame_bytes(module.stack_slots)
    if frame:
        words.append(_sub_sp(frame))
    words.append(0x910003FD)  # mov x29, sp
    string_references: list[tuple[int, bytes]] = []
    labels: dict[str, int] = {}
    branches: list[tuple[int, str, bool]] = []
    for operation in module.operations:
        if isinstance(operation, Write):
            words.extend(_mov(0, 1))
            adr_index = len(words)
            words.append(0)
            string_references.append((adr_index, operation.data))
            words.extend(_mov(2, len(operation.data)))
            syscall_register = 8 if svc == 0xD4000001 else 16
            words.extend(_mov(syscall_register, write_number))
            words.append(svc)
        elif isinstance(operation, Store):
            _expression(words, operation.value, 0)
            words.append(_slot_instruction(False, operation.slot, 0))
        elif isinstance(operation, Label):
            labels[operation.name] = len(words)
        elif isinstance(operation, Jump):
            branches.append((len(words), operation.target, False))
            words.append(0)
        elif isinstance(operation, JumpIfFalse):
            _expression(words, operation.condition, 0)
            branches.append((len(words), operation.target, True))
            words.append(0)
        elif isinstance(operation, Exit):
            words.extend(_mov(0, operation.status))
            syscall_register = 8 if svc == 0xD4000001 else 16
            words.extend(_mov(syscall_register, exit_number))
            words.append(svc)
        elif isinstance(operation, ExitValue):
            _expression(words, operation.value, 0)
            syscall_register = 8 if svc == 0xD4000001 else 16
            words.extend(_mov(syscall_register, exit_number))
            words.append(svc)
    _patch_branches(words, labels, branches)
    image = bytearray(struct.pack(f"<{len(words)}I", *words))
    for instruction_index, data in string_references:
        instruction_address = code_address + instruction_index * 4
        data_address = code_address + len(image)
        struct.pack_into("<I", image, instruction_index * 4, _adr(1, data_address - instruction_address))
        image.extend(data)
    return bytes(image)
