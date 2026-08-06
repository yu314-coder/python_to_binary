# `abc.abstractmethod(f)` does one thing: it sets `f.__isabstractmethod__`.
# A compiled function has no `__dict__` to set it on, so the whole program
# used to stop at its first `class Shape(ABC)`.
import abc
from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
    def area(self):
        """How much of it there is."""

    @abstractmethod
    def name(self):
        """What it is called."""
        return "shape"

    def described(self):
        return f"{self.name()} of {self.area()}"


class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side * self.side

    def name(self):
        return "square"


print(Square(3).described())
for attempt in (Shape, type("Partial", (Shape,), {"area": lambda self: 0})):
    try:
        attempt()
    except TypeError as error:
        print(type(error).__name__, sorted(attempt.__abstractmethods__))


class WithProperty(ABC):
    @property
    @abstractmethod
    def value(self):
        ...


class Concrete(WithProperty):
    @property
    def value(self):
        return 7


print(Concrete().value, sorted(WithProperty.__abstractmethods__))


class WithClassMethod(ABC):
    @classmethod
    @abstractmethod
    def make(cls):
        ...


class Made(WithClassMethod):
    @classmethod
    def make(cls):
        return "made"


print(Made.make())


class Dotted(abc.ABC):
    @abc.abstractmethod
    def go(self):
        ...


class Went(Dotted):
    def go(self):
        return "went"


print(Went().go())


class Calls(ABC):
    @abstractmethod
    def base(self):
        return "base"


class Uses(Calls):
    def base(self):
        return super().base() + "+more"


print(Uses().base())
# An abstract method still answers for itself: its name, and its docstring.
print(Shape.area.__isabstractmethod__, Shape.area.__doc__)
print(Shape.name.__doc__, Square(2).name(), Square(2).area())
