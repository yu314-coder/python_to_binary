d = {"k": [1]}
alias = d["k"]
d["k"] += [2]
print(d["k"], alias, d["k"] is alias)
class Box:
    def __init__(self): self.v = [1]
b = Box()
held = b.v
b.v += [2]
print(b.v, held, b.v is held)
n = 5
n += 2
n *= 3
n //= 2
print(n)
s = "a"
s += "b"
print(s)
