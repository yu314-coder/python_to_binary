from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Write:
    data: bytes


@dataclass(frozen=True, slots=True)
class Exit:
    status: int


Operation = Write | Exit


@dataclass(slots=True)
class Module:
    operations: list[Operation]

