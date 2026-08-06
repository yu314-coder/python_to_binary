import os
import os.path as osp
from os import sep, linesep as nl
from dataclasses import dataclass, field
from collections import namedtuple as nt

@dataclass
class R:
    k: str = "x"

Pair = nt("Pair", "a b")
print(R(), Pair(1, 2), osp.basename("/a/b"), isinstance(sep, str), isinstance(nl, str))
print(sorted(n for n in globals() if not n.startswith("_")))
gone = 1
del gone
print("gone" in globals(), "gone" in sorted(globals()))
try:
    print(gone)
except NameError as e:
    print("NameError:", e)
del nl
print("nl" in globals(), sorted(n for n in globals() if not n.startswith("_")))
