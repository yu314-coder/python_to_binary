import pickle, copy
from dataclasses import dataclass
from enum import Enum
class P:
    def __init__(self, v): self.v = v
    def __eq__(self, o): return self.v == o.v
@dataclass
class Row:
    k: str = "x"
class Colour(Enum):
    RED = 1
class Outer:
    class Inner:
        def __init__(self): self.n = 7
def helper(a): return a * 2
print(pickle.loads(pickle.dumps(P(3))).v)
print(pickle.loads(pickle.dumps(Row("q"))))
print(pickle.loads(pickle.dumps(Colour.RED)) is Colour.RED)
print(pickle.loads(pickle.dumps(Outer.Inner())).n)
print(pickle.loads(pickle.dumps(helper))(4))
print(copy.deepcopy(P(5)).v, copy.copy(Row("z")))
print(pickle.loads(pickle.dumps([P(1), Row("a"), Colour.RED]))[0].v)
