from dataclasses import dataclass, field
@dataclass(frozen=True, order=True)
class P:
    x: int
    y: int = 0
print(P(1) < P(2), P(1,2))
