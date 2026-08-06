from enum import Enum, auto, IntEnum
class C(Enum):
    A=auto()
    B=auto()
class I(IntEnum):
    X=1
print(list(C), C.A.value, I.X+1)
