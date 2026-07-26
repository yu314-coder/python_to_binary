from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .ir import (
    BitsFloat,
    Call,
    CStringConstant,
    Exit,
    ExitValue,
    ExternCall,
    FloatBinary,
    FloatBits,
    FloatCompare,
    FloatConstant,
    FloatExpression,
    FloatLoad,
    FloatStore,
    FloatToInt,
    FloatUnary,
    Function,
    FunctionAddress,
    GlobalAddress,
    HeapAlloc,
    HeapInit,
    HeapLoad,
    HeapStore,
    IntBinary,
    IntCompare,
    IntConstant,
    IndirectCall,
    IntExpression,
    IntLoad,
    IntToFloat,
    IntUnary,
    Jump,
    JumpIfFalse,
    Label,
    Module,
    Return,
    SlotAddress,
    Store,
    Write,
    WriteRuntime,
)


# Load/store encodings keyed by (bytes, signed). The load column widens the
# value into X0 exactly as C's integer conversions require; the store column
# writes only the low bytes and leaves the rest of the object alone.
_ARM64_LOADS = {
    (8, False): 0xF9400000,  # ldr   x0, [x0]
    (8, True): 0xF9400000,
    (4, False): 0xB9400000,  # ldr   w0, [x0]   (zero-extends into x0)
    (4, True): 0xB9800000,  # ldrsw x0, [x0]
    (2, False): 0x79400000,  # ldrh  w0, [x0]
    (2, True): 0x79800000,  # ldrsh x0, [x0]
    (1, False): 0x39400000,  # ldrb  w0, [x0]
    (1, True): 0x39800000,  # ldrsb x0, [x0]
}
_ARM64_STORES = {
    8: 0xF9000020,  # str  x0, [x1]
    4: 0xB9000020,  # str  w0, [x1]
    2: 0x79000020,  # strh w0, [x1]
    1: 0x39000020,  # strb w0, [x1]
}


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


def _frame_sub(amount: int) -> list[int]:
    """Lower ``sub sp, sp, #amount`` for a frame of any supported size.

    SUB (immediate) carries a 12-bit immediate with an optional 12-bit left
    shift, so a frame larger than 4095 bytes needs the shifted form as well.
    C local arrays make that a routine size, not an exotic one.
    """

    if amount == 0:
        return []
    if amount < 0 or amount > 0xFFF + (0xFFF << 12):
        raise ValueError("ARM64 native stack frame exceeds immediate range")
    words: list[int] = []
    high = amount >> 12
    if high:
        words.append(0xD14003FF | (high << 10))  # sub sp, sp, #high, lsl #12
    low = amount & 0xFFF
    if low:
        words.append(_sub_sp(low))
    return words


def _frame_add(amount: int) -> list[int]:
    """Lower ``add sp, sp, #amount``, the mirror image of :func:`_frame_sub`."""

    if amount == 0:
        return []
    if amount < 0 or amount > 0xFFF + (0xFFF << 12):
        raise ValueError("ARM64 native stack frame exceeds immediate range")
    words: list[int] = []
    high = amount >> 12
    if high:
        words.append(0x914003FF | (high << 10))  # add sp, sp, #high, lsl #12
    low = amount & 0xFFF
    if low:
        words.append(0x910003FF | (low << 10))  # add sp, sp, #low
    return words


#: The register that holds the base of the module's static storage block for
#: the whole run. X28 is callee-saved in AAPCS64 and this backend never writes
#: it anywhere else, so a value established once in the entry prologue is still
#: there inside every function body and after every external (libSystem) call.
#: That is what lets a C file-scope variable be one object shared by the entry
#: point and every function, none of which share a stack frame.
_STATIC_BASE = 28


def _static_address(offset: int) -> list[int]:
    """Place the address of static byte ``offset`` in X0."""

    if offset < 0:
        raise ValueError("a static-storage offset cannot be negative")
    if offset <= 0xFFF:
        # add x0, x28, #offset
        return [0x91000000 | (offset << 10) | (_STATIC_BASE << 5)]
    # mov x0, #offset; add x0, x28, x0
    return [*_mov(0, offset), 0x8B000000 | (_STATIC_BASE << 5)]


def _slot_instruction(load: bool, slot: int, slot_base: int, rt: int = 0) -> int:
    offset = slot_base + slot * 8
    if offset < 0 or offset > 0x7FF8 or offset & 7:
        raise ValueError("ARM64 native variable is outside stack-slot range")
    # ldr/str x<rt>, [x29, #offset]. Rn=x29 (0x1D) is encoded in bits [9:5].
    base = 0xF9400000 if load else 0xF9000000
    return base | ((offset // 8) << 10) | (29 << 5) | rt


def _slot_address(slot: int, slot_base: int) -> list[int]:
    """Place the address of stack slot ``slot`` in X0."""

    offset = slot_base + slot * 8
    if offset < 0:
        raise ValueError("ARM64 native variable is outside stack-slot range")
    if offset <= 0xFFF:
        return [0x91000000 | (offset << 10) | (29 << 5)]  # add x0, x29, #offset
    return [*_mov(0, offset), 0x8B0003A0]  # mov x0, #offset; add x0, x29, x0


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
        # Unsigned orderings. C compares two unsigned 64-bit values with these,
        # and using the signed forms instead is exactly how 0xFFFFFFFFFFFFFFFF
        # would compare as less than 1.
        "uge": 0x2,  # hs / cs
        "ult": 0x3,  # lo / cc
        "ugt": 0x8,  # hi
        "ule": 0x9,  # ls
    }
    try:
        return conditions[operator]
    except KeyError as error:
        raise ValueError(f"unknown ARM64 comparison {operator!r}") from error


def _float_condition_code(operator: str) -> int:
    """Condition codes for FCMP, where the integer ones are not interchangeable.

    An IEEE comparison has four outcomes, not three: FCMP sets NZCV to 0011
    when either operand is NaN, and C requires every relational operator to be
    false in that case. The signed integer codes LT (N!=V) and LE (Z=1 or
    N!=V) are both TRUE for that flag pattern, so ``x < y`` with a NaN operand
    would answer yes. MI (N=1) and LS (C=0 or Z=1) are the ordered forms and
    answer no, which is what C requires.
    """

    conditions = {
        "eq": 0x0,  # eq  -- Z=1, and unordered clears Z
        "ne": 0x1,  # ne
        "lt": 0x4,  # mi  -- ordered less than
        "le": 0x9,  # ls  -- ordered less than or equal
        "gt": 0xC,  # gt
        "ge": 0xA,  # ge
    }
    try:
        return conditions[operator]
    except KeyError as error:
        raise ValueError(f"unknown ARM64 float comparison {operator!r}") from error


@dataclass
class _Refs:
    """Fixups collected while encoding the darwin dynamic (extern) image.

    ``strings`` records ``(word_index, data, register)`` for ADR loads of
    embedded constant byte blobs; ``externs`` records ``(word_index, symbol)``
    for each ``adrp x16``/``ldr x16``/``blr x16`` GOT call site so the Mach-O
    writer can point it at the bound symbol pointer. ``calls`` records
    ``(word_index, function name)`` for each internal ``bl``, patched with the
    real displacement once every body's offset in ``.text`` is known; and
    ``addresses`` records ``(word_index, function name)`` for each ``adr x0``
    that materializes a function's entry address, patched the same way.
    """

    strings: list[tuple[int, bytes, int]] = field(default_factory=list)
    externs: list[tuple[int, str]] = field(default_factory=list)
    calls: list[tuple[int, str]] = field(default_factory=list)
    addresses: list[tuple[int, str]] = field(default_factory=list)


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


#: The maximum number of arguments an internal call passes. AAPCS64 gives eight
#: integer parameter registers; anything past that has to be passed on the
#: stack, which this backend does not implement and therefore rejects.
ARM64_ARGUMENT_REGISTERS = 8

#: Label name reserved for a function's epilogue. Every ``Return`` branches
#: here rather than duplicating the teardown. The character is not producible
#: by any frontend label, so it cannot collide with an IR label.
_EPILOGUE = "\0epilogue"


def _internal_call(
    words: list[int], expression: "Call", slot_base: int, refs: "_Refs | None"
) -> None:
    """Emit an AAPCS64 call to another function in this image.

    Each argument is evaluated into X0 and immediately spilled to a 16-byte
    stack cell, because evaluating the next one would otherwise clobber it (the
    expression encoder owns X0-X2 outright). The cells are then popped into
    X0-X7 in reverse, which both keeps SP 16-byte aligned throughout and leaves
    SP exactly where it started before the ``bl``.
    """

    if refs is None:
        raise ValueError(
            "ARM64 internal calls are not supported by this encoder; the "
            "call-capable encoders are encode_darwin/encode_darwin_extern and "
            "encode_linux"
        )
    if len(expression.arguments) > ARM64_ARGUMENT_REGISTERS:
        raise ValueError(
            f"ARM64 internal calls pass at most {ARM64_ARGUMENT_REGISTERS} "
            "arguments in registers"
        )
    for argument in expression.arguments:
        _expression(words, argument, slot_base, refs)  # arg -> x0
        words.extend((_sub_sp(16), 0xF90003E0))  # str x0, [sp]
    for index in reversed(range(len(expression.arguments))):
        words.append(0xF94003E0 | index)  # ldr x{index}, [sp]
        words.append(0x910043FF)  # add sp, sp, #16
    refs.calls.append((len(words), expression.name))
    words.append(0)  # bl <function> (patched once every body offset is known)


def _indirect_call(
    words: list[int], expression: "IndirectCall", slot_base: int, refs: "_Refs | None"
) -> None:
    """Emit an AAPCS64 call through a computed address.

    The only difference from :func:`_internal_call` is where the target comes
    from. It is evaluated FIRST and spilled below the arguments, so a target
    expression is evaluated exactly once and before them, and so nothing an
    argument computes can clobber it. It is then popped into X16 (the
    intra-procedure scratch register, which no argument occupies) immediately
    before the ``blr``.
    """

    if refs is None:
        raise ValueError(
            "ARM64 indirect calls are not supported by this encoder; the "
            "call-capable encoders are encode_darwin/encode_darwin_extern and "
            "encode_linux"
        )
    if len(expression.arguments) > ARM64_ARGUMENT_REGISTERS:
        raise ValueError(
            f"ARM64 indirect calls pass at most {ARM64_ARGUMENT_REGISTERS} "
            "arguments in registers"
        )
    _expression(words, expression.target, slot_base, refs)  # target -> x0
    words.extend((_sub_sp(16), 0xF90003E0))  # str x0, [sp]
    for argument in expression.arguments:
        _expression(words, argument, slot_base, refs)  # arg -> x0
        words.extend((_sub_sp(16), 0xF90003E0))  # str x0, [sp]
    for index in reversed(range(len(expression.arguments))):
        words.append(0xF94003E0 | index)  # ldr x{index}, [sp]
        words.append(0x910043FF)  # add sp, sp, #16
    words.append(0xF94003F0)  # ldr x16, [sp]
    words.append(0x910043FF)  # add sp, sp, #16
    words.append(0xD63F0200)  # blr x16


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
    if isinstance(expression, SlotAddress):
        words.extend(_slot_address(expression.slot, slot_base))
        return
    if isinstance(expression, GlobalAddress):
        if refs is None:
            # Only the encoders that run _emit_static_block establish X28, and
            # addressing static storage through a register nothing set would
            # store through whatever the loader happened to leave there.
            raise ValueError(
                "ARM64 static storage is not supported by this encoder; the "
                "encoders that establish the static base are encode_darwin/"
                "encode_darwin_extern and encode_linux"
            )
        words.extend(_static_address(expression.offset))
        return
    if isinstance(expression, FunctionAddress):
        if refs is None:
            raise ValueError(
                "ARM64 function addresses are not supported by this encoder; the "
                "call-capable encoders are encode_darwin/encode_darwin_extern "
                "and encode_linux"
            )
        refs.addresses.append((len(words), expression.name))
        words.append(0)  # adr x0, <function> (patched with the body's offset)
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
    if isinstance(expression, Call):
        _internal_call(words, expression, slot_base, refs)
        return
    if isinstance(expression, IndirectCall):
        _indirect_call(words, expression, slot_base, refs)
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
                "lshift": 0x9AC12000,  # lslv x0, x0, x1
                "rshift": 0x9AC12800,  # asrv x0, x0, x1 (arithmetic)
                "urshift": 0x9AC12400,  # lsrv x0, x0, x1 (logical)
                "sdiv": 0x9AC10C00,  # sdiv x0, x0, x1
                "udiv": 0x9AC10800,  # udiv x0, x0, x1
            }
            remainders = {
                # x2 = x0 / x1, then x0 = x0 - x2 * x1. C's % truncates toward
                # zero and keeps the dividend's sign, which is exactly what
                # sdiv/msub compute.
                "smod": (0x9AC10C02, 0x9B018040),  # sdiv x2,x0,x1; msub x0,x2,x1,x0
                "umod": (0x9AC10802, 0x9B018040),  # udiv x2,x0,x1; msub x0,x2,x1,x0
            }
            if expression.operator in remainders:
                words.extend(remainders[expression.operator])
                return
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
        if expression.signed:
            words.append(0x9E780000)  # fcvtzs x0, d0 (truncate toward zero)
        else:
            words.append(0x9E790000)  # fcvtzu x0, d0
        return
    if isinstance(expression, FloatBits):
        _float_expression(words, expression.value, slot_base, refs)
        if expression.size == 8:
            words.append(0x9E660000)  # fmov x0, d0
        elif expression.size == 4:
            words.append(0x1E624000)  # fcvt s0, d0  (round to binary32)
            words.append(0x1E260000)  # fmov w0, s0  (zero-extends into x0)
        else:
            raise ValueError(f"unsupported ARM64 float bit width {expression.size}")
        return
    if isinstance(expression, FloatCompare):
        _float_expression(words, expression.left, slot_base, refs)
        words.extend((_sub_sp(16), 0xFD0003E0))  # str d0, [sp]
        _float_expression(words, expression.right, slot_base, refs)
        words.extend((0x1E604001, 0xFD4003E0, 0x910043FF))  # fmov d1,d0; ldr d0,[sp]; add sp,#16
        words.append(0x1E612000)  # fcmp d0, d1
        condition = _float_condition_code(expression.operator)
        words.append(0x9A9F07E0 | ((condition ^ 1) << 12))  # cset x0, cond
        return
    if isinstance(expression, HeapLoad):
        _expression(words, expression.address, slot_base, refs)  # x0 = address
        instruction = _ARM64_LOADS.get((expression.size, bool(expression.signed)))
        if instruction is None:
            raise ValueError(f"unsupported ARM64 heap load size {expression.size}")
        words.append(instruction)
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
        if expression.signed:
            words.append(0x9E620000)  # scvtf d0, x0
        else:
            words.append(0x9E630000)  # ucvtf d0, x0
        return
    if isinstance(expression, BitsFloat):
        _expression(words, expression.value, slot_base, refs)
        if expression.size == 8:
            words.append(0x9E670000)  # fmov d0, x0
        elif expression.size == 4:
            words.append(0x1E270000)  # fmov s0, w0
            words.append(0x1E22C000)  # fcvt d0, s0  (exact widening)
        else:
            raise ValueError(f"unsupported ARM64 float bit width {expression.size}")
        return
    if isinstance(expression, FloatUnary):
        _float_expression(words, expression.operand, slot_base, refs)
        if expression.operator == "pos":
            return
        if expression.operator == "neg":
            words.append(0x1E614000)  # fneg d0, d0
            return
        # The C math functions that ARM64 implements as one instruction. No
        # library and no linker is involved: these ARE the operations.
        unary = {
            "sqrt": 0x1E61C000,  # fsqrt  d0, d0
            "abs": 0x1E60C000,  # fabs   d0, d0
            "floor": 0x1E654000,  # frintm d0, d0  (toward -inf)
            "ceil": 0x1E64C000,  # frintp d0, d0  (toward +inf)
            "trunc": 0x1E65C000,  # frintz d0, d0  (toward zero)
            # C round() breaks ties AWAY from zero, which is FRINTA, not
            # FRINTN (ties to even): round(2.5) is 3.0, not 2.0.
            "round": 0x1E664000,  # frinta d0, d0
        }
        if expression.operator in unary:
            words.append(unary[expression.operator])
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
    if module.functions:
        raise ValueError(
            "internal function calls are not supported for windows-arm64 yet"
        )
    if module.static_bytes:
        raise ValueError(
            "static storage (C file-scope variables) is not supported for "
            "windows-arm64 yet; it needs VirtualAlloc in the PE import table"
        )
    slot_base = 16
    words: list[int] = [
        *_frame_sub(_frame_bytes(module.stack_slots, slot_base)),
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
        if isinstance(operation, Return):
            raise ValueError(
                "internal function calls are not supported for windows-arm64 yet"
            )
        if isinstance(operation, (HeapInit, HeapAlloc, WriteRuntime)):
            raise ValueError(
                "runtime heap lists/strings are not supported for windows-arm64 yet"
            )
        if isinstance(operation, HeapStore):
            # Plain memory, not the arena: a C store through a pointer needs no
            # mmap, so it works on Windows exactly as it does on POSIX.
            _expression(words, operation.address, slot_base)
            words.extend((_sub_sp(16), 0xF90003E0))  # str x0, [sp]
            _expression(words, operation.value, slot_base)
            words.extend((0xF94003E1, 0x910043FF))  # ldr x1,[sp]; add sp,#16
            instruction = _ARM64_STORES.get(operation.size)
            if instruction is None:
                raise ValueError(f"unsupported ARM64 heap store size {operation.size}")
            words.append(instruction)
            continue
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


@dataclass(frozen=True, slots=True)
class _Syscalls:
    """The three kernel entry points the syscall encoders need."""

    write_number: int
    exit_number: int
    mmap_number: int
    mmap_flags: int
    svc: int

    @property
    def register(self) -> int:
        """The register the kernel reads the syscall number from."""

        return 8 if self.svc == 0xD4000001 else 16


def _emit_operations(
    words: list[int],
    operations: list,
    slot_base: int,
    refs: _Refs,
    system: _Syscalls,
    *,
    in_function: bool,
) -> None:
    """Encode one straight-line operation list into ``words``.

    ``slot_base`` is the byte offset of stack slot 0 from X29, which differs
    between the entry point (0) and a called function (16, because the saved
    frame pointer and link register sit at the bottom of its frame).
    """

    syscall_register = system.register
    labels: dict[str, int] = {}
    branches: list[tuple[int, str, bool]] = []
    for operation in operations:
        if isinstance(operation, HeapInit):
            # x0=addr=0, x1=len, x2=prot RW=3, x3=flags, x4=fd=-1, x5=off=0
            words.extend(_mov(0, 0))
            words.extend(_mov(1, operation.size))
            words.extend(_mov(2, 3))
            words.extend(_mov(3, system.mmap_flags))
            words.extend(_mov(4, (-1) & 0xFFFFFFFFFFFFFFFF))
            words.extend(_mov(5, 0))
            words.extend(_mov(syscall_register, system.mmap_number))
            words.append(system.svc)
            words.append(_slot_instruction(False, operation.slot, slot_base))
            continue
        if isinstance(operation, HeapAlloc):
            _expression(words, operation.size, slot_base, refs)  # x0 = size
            words.append(_slot_instruction(True, operation.bump_slot, slot_base, rt=1))
            words.append(_slot_instruction(False, operation.dest_slot, slot_base, rt=1))
            words.append(0x8B000020)  # add x0, x1, x0  (new bump)
            words.append(_slot_instruction(False, operation.bump_slot, slot_base))
            continue
        if isinstance(operation, HeapStore):
            _expression(words, operation.address, slot_base, refs)  # x0 = address
            words.extend((_sub_sp(16), 0xF90003E0))  # str x0, [sp]
            _expression(words, operation.value, slot_base, refs)  # x0 = value
            words.extend((0xF94003E1, 0x910043FF))  # ldr x1,[sp]; add sp,#16
            instruction = _ARM64_STORES.get(operation.size)
            if instruction is None:
                raise ValueError(f"unsupported ARM64 heap store size {operation.size}")
            words.append(instruction)
            continue
        if isinstance(operation, WriteRuntime):
            _expression(words, operation.length, slot_base, refs)  # x0 = length
            words.extend((_sub_sp(16), 0xF90003E0))  # str x0, [sp]
            _expression(words, operation.address, slot_base, refs)  # x0 = address
            words.append(0xAA0003E1)  # mov x1, x0  (buf)
            words.extend((0xF94003E2, 0x910043FF))  # ldr x2,[sp]; add sp,#16
            words.extend(_mov(0, 1))  # fd = stdout
            words.extend(_mov(syscall_register, system.write_number))
            words.append(system.svc)
            continue
        if isinstance(operation, Write):
            words.extend(_mov(0, operation.fd))
            adr_index = len(words)
            words.append(0)
            refs.strings.append((adr_index, operation.data, 1))
            words.extend(_mov(2, len(operation.data)))
            words.extend(_mov(syscall_register, system.write_number))
            words.append(system.svc)
        elif isinstance(operation, Store):
            _expression(words, operation.value, slot_base, refs)
            words.append(_slot_instruction(False, operation.slot, slot_base))
        elif isinstance(operation, FloatStore):
            _float_expression(words, operation.value, slot_base, refs)
            words.append(_float_slot_instruction(False, operation.slot, slot_base))
        elif isinstance(operation, Label):
            labels[operation.name] = len(words)
        elif isinstance(operation, Jump):
            branches.append((len(words), operation.target, False))
            words.append(0)
        elif isinstance(operation, JumpIfFalse):
            _expression(words, operation.condition, slot_base, refs)
            branches.append((len(words), operation.target, True))
            words.append(0)
        elif isinstance(operation, Exit):
            words.extend(_mov(0, operation.status))
            words.extend(_mov(syscall_register, system.exit_number))
            words.append(system.svc)
        elif isinstance(operation, ExitValue):
            _expression(words, operation.value, slot_base, refs)
            words.extend(_mov(syscall_register, system.exit_number))
            words.append(system.svc)
        elif isinstance(operation, Return):
            if not in_function:
                raise ValueError(
                    "a native IR Return is only legal inside a Function body"
                )
            if operation.value is not None:
                _expression(words, operation.value, slot_base, refs)  # x0 = result
            branches.append((len(words), _EPILOGUE, False))
            words.append(0)
        else:
            raise TypeError(f"unknown ARM64 operation {type(operation).__name__}")
    if in_function:
        labels[_EPILOGUE] = len(words)
    _patch_branches(words, labels, branches)


def _emit_function(
    words: list[int], function: Function, refs: _Refs, system: _Syscalls
) -> None:
    """Encode one callable body with a real AAPCS64 frame.

    The frame is a single fixed allocation holding the saved ``x29``/``x30``
    pair at its base and the function's stack slots above them, so slot 0 lives
    at ``[x29, #16]``. Building the frame with one ``sub sp`` and saving the
    pair at offset 0 (rather than the more familiar pre-indexed ``stp``) keeps
    the layout identical for a two-slot frame and a 32 KiB one: the pre-indexed
    form's 7-bit immediate cannot reach past 504 bytes, and a C function with a
    local array routinely needs more than that.
    """

    if function.parameters > ARM64_ARGUMENT_REGISTERS:
        raise ValueError(
            f"ARM64 function {function.name!r} declares {function.parameters} "
            f"parameters; at most {ARM64_ARGUMENT_REGISTERS} are passed in registers"
        )
    if function.parameters > function.stack_slots:
        raise ValueError(
            f"ARM64 function {function.name!r} has fewer stack slots than parameters"
        )
    frame = _frame_bytes(function.stack_slots, 16)
    words.extend(_frame_sub(frame))
    words.append(0xA9007BFD)  # stp x29, x30, [sp]   (save the CALLER's frame)
    words.append(0x910003FD)  # mov x29, sp
    for index in range(function.parameters):
        # Spill the incoming argument registers into the slots the body reads.
        words.append(_slot_instruction(False, index, 16, rt=index))
    _emit_operations(
        words, function.operations, 16, refs, system, in_function=True
    )
    # Epilogue. SP is restored from X29 rather than adjusted, so the frame is
    # sound even though the expression encoder pushes and pops SP freely.
    words.append(0x910003BF)  # mov sp, x29
    words.append(0xA9407BFD)  # ldp x29, x30, [sp]
    words.extend(_frame_add(frame))
    words.append(0xD65F03C0)  # ret


def _emit_static_block(words: list[int], size: int, system: _Syscalls) -> None:
    """Establish the module's static storage block in X28, or exit.

    The block is a single anonymous read-write mapping, so it is zero-filled by
    the kernel -- which is exactly the initial value C gives an object with
    static storage duration and no initializer -- and it outlives every stack
    frame. Its address goes in X28 and stays there for the whole run.

    The mapping is checked rather than assumed. A failed ``mmap`` returns a
    small errno on Darwin and a negative ``-errno`` on Linux, and both readings
    are rejected here: writing a global through a base of 12 would be a wild
    store, and this compiler exits instead of running on wrongly.
    """

    if size <= 0:
        return
    words.extend(_mov(0, 0))  # addr: let the kernel choose
    words.extend(_mov(1, size))  # length
    words.extend(_mov(2, 3))  # PROT_READ | PROT_WRITE
    words.extend(_mov(3, system.mmap_flags))
    words.extend(_mov(4, (-1) & 0xFFFFFFFFFFFFFFFF))  # fd
    words.extend(_mov(5, 0))  # offset
    words.extend(_mov(system.register, system.mmap_number))
    words.append(system.svc)
    failure = [
        *_mov(0, 71),  # EX_OSERR: the mapping this image needs was refused
        *_mov(system.register, system.exit_number),
        system.svc,
    ]
    # cmp x0, #0 / b.le failure -- catches Linux's negative -errno.
    words.append(0xF100001F)
    words.append(0x5400000D | ((3 & 0x7FFFF) << 5))  # b.le +3 words
    # cmp x0, #1, lsl #12 / b.hs ok -- catches Darwin's small positive errno.
    words.append(0xF140041F)
    words.append(0x54000002 | (((len(failure) + 1) & 0x7FFFF) << 5))  # b.hs ok
    words.extend(failure)
    words.append(0xAA0003E0 | _STATIC_BASE)  # mov x28, x0


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
    system = _Syscalls(write_number, exit_number, mmap_number, mmap_flags, svc)
    words: list[int] = list(_frame_sub(_frame_bytes(module.stack_slots)))
    words.append(0x910003FD)  # mov x29, sp
    _emit_static_block(words, module.static_bytes, system)
    refs = _Refs()
    _emit_operations(words, module.operations, 0, refs, system, in_function=False)
    offsets: dict[str, int] = {}
    for function in module.functions:
        if function.name in offsets:
            raise ValueError(f"duplicate native IR function {function.name!r}")
        offsets[function.name] = len(words)
        _emit_function(words, function, refs, system)
    for instruction_index, name in refs.calls:
        if name not in offsets:
            raise ValueError(f"call to undefined native IR function {name!r}")
        distance = offsets[name] - instruction_index
        if not -(1 << 25) <= distance < (1 << 25):
            raise ValueError("ARM64 call is outside branch range")
        words[instruction_index] = 0x94000000 | (distance & 0x3FFFFFF)
    for instruction_index, name in refs.addresses:
        if name not in offsets:
            raise ValueError(
                f"address taken of undefined native IR function {name!r}"
            )
        # ADR is PC-relative in BYTES, so the image may still be slid.
        words[instruction_index] = _adr(
            0, (offsets[name] - instruction_index) * 4
        )
    externs = [(index * 4, symbol) for index, symbol in refs.externs]
    image = bytearray(struct.pack(f"<{len(words)}I", *words))
    for instruction_index, data, register in refs.strings:
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
