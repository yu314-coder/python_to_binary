from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IntConstant:
    value: int


@dataclass(frozen=True, slots=True)
class IntLoad:
    slot: int


@dataclass(frozen=True, slots=True)
class IntUnary:
    operator: str
    operand: "IntExpression"


@dataclass(frozen=True, slots=True)
class IntBinary:
    operator: str
    left: "IntExpression"
    right: "IntExpression"


@dataclass(frozen=True, slots=True)
class IntCompare:
    operator: str
    left: "IntExpression"
    right: "IntExpression"


IntExpression = IntConstant | IntLoad | IntUnary | IntBinary | IntCompare


@dataclass(frozen=True, slots=True)
class Write:
    data: bytes


@dataclass(frozen=True, slots=True)
class Store:
    slot: int
    value: IntExpression


@dataclass(frozen=True, slots=True)
class Label:
    name: str


@dataclass(frozen=True, slots=True)
class Jump:
    target: str


@dataclass(frozen=True, slots=True)
class JumpIfFalse:
    condition: IntExpression
    target: str


@dataclass(frozen=True, slots=True)
class Exit:
    status: int


@dataclass(frozen=True, slots=True)
class ExitValue:
    value: IntExpression


Operation = Write | Store | Label | Jump | JumpIfFalse | Exit | ExitValue


@dataclass(slots=True)
class Module:
    operations: list[Operation]
    stack_slots: int = 0
