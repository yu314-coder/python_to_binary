# `f.__annotations__` needs the function to hold a dictionary, and a compiled
# function has no `__dict__`. It is carried in the same holder `wraps` and
# `abstractmethod` write on - and only where the program asks, because
# annotating a parameter is far commoner than asking what the annotation was.
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch, wraps
from typing import Iterator, List, Optional, get_type_hints


def plain(a: int, b: str = "x") -> bool:
    return True


def every_shape(a, *rest: int, k: float = 1.0, **more: str) -> None:
    pass


def only_returns() -> int:
    return 1


print(plain.__annotations__)
print(every_shape.__annotations__)
print(only_returns.__annotations__)
print(get_type_hints(plain))


def written_down(a: int, b: "Optional[str]" = None) -> "List[int]":
    return [a]


print(get_type_hints(written_down))


class Holder:
    def method(self, v: int) -> str:
        return str(v)

    @staticmethod
    def stayed(v: int) -> str:
        return str(v)

    @classmethod
    def classed(cls, v: int) -> str:
        return str(v)


print(Holder.method.__annotations__, Holder().method(1))
print(get_type_hints(Holder.stayed), get_type_hints(Holder.classed))


def counted(f):
    @wraps(f)
    def inner(*args):
        return f(*args)

    return inner


@counted
def decorated(x: int) -> str:
    return str(x)


print(decorated(1), decorated.__name__, decorated.__annotations__)


class Interface(ABC):
    @abstractmethod
    def go(self, v: int) -> str:
        ...


class Doer(Interface):
    def go(self, v: int) -> str:
        return str(v)


print(Doer().go(3), get_type_hints(Interface.go), sorted(Interface.__abstractmethods__))


@singledispatch
def dispatched(v):
    return "any"


@dispatched.register
def _(v: int):
    return "int"


@dispatched.register
def _(v: str):
    return "str"


print(dispatched(1.0), dispatched(1), dispatched("s"))


def generating(n: int) -> Iterator[int]:
    for value in range(n):
        yield value


async def awaiting(n: int) -> int:
    await asyncio.sleep(0)
    return n * 2


print(list(generating(3)), generating.__annotations__)
print(asyncio.run(awaiting(3)), awaiting.__annotations__)


@dataclass
class Point:
    x: int
    y: str = "a"

    def shift(self, by: int) -> "Point":
        return Point(self.x + by, self.y)


print(Point(1), Point(1).shift(2), Point.shift.__annotations__)

# Calls are untouched by any of this: a function of a fixed shape is called
# directly in C and never goes through the name.
total = 0
for step in range(1000):
    total = plain(step) and total + step
print(total)
