from typing import TypeVar, Generic, Protocol
from enum import Enum
from dataclasses import dataclass
T = TypeVar("T")
class Box(Generic[T]):
    def __init__(self, v): self.v = v
class Named(Protocol):
    pass
class Colour(Enum):
    RED = 1
@dataclass
class Row:
    k: str = "x"
class Plain: pass
class Sub(Plain): pass
print(Box(2).v, Colour.RED.name, Row(), Sub.__mro__[1].__name__)
print(Box.__orig_bases__, hasattr(Sub, "__orig_bases__"))
