from enum import Enum
from dataclasses import dataclass
class Config:
    name = "top"
    class Colour(Enum):
        RED = 1
        BLUE = 2
    @dataclass
    class Point:
        x: int = 0
    class Deep:
        class Deeper:
            v = 9
    def method(self):
        return Config.Colour.RED.name
def make():
    class Local:
        w = 5
    return Local
print(Config.name, Config.Colour.RED.name, Config.Point(3), Config.Deep.Deeper.v)
print(Config().method(), make().w, make().__qualname__)
print(Config.Colour.__qualname__, Config.Deep.Deeper.__qualname__, Config.Point)
class Plain:
    pass
print(Plain, Plain.__qualname__)
