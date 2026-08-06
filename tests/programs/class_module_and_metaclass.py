from enum import Enum
from dataclasses import dataclass
class M(type):
    def __new__(m, n, b, d):
        d["tag"] = 9
        return super().__new__(m, n, b, d)
class Named(metaclass=M):
    pass
class Child(Named):
    pass
class Colour(Enum):
    RED = 1
@dataclass
class Row:
    k: str = "x"
class Plain:
    pass
class Sub(Plain):
    pass
print(Named.tag, type(Named).__name__, type(Child).__name__)
print(Colour.RED.name, Row(), Plain, Sub, type(Plain).__name__)
print(Plain.__module__, Sub.__qualname__, Colour.__module__)
try:
    class Bad(42): pass
except TypeError as e:
    print("TypeError:", e)
