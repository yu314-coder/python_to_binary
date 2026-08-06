# `global` in a class body binds the module's name, not an attribute.
seen = 1


class Plain:
    global seen
    seen = 2


print(seen, hasattr(Plain, "seen"))

registry = {}


class Registering:
    global registry
    registry["here"] = True
    kept = 5


print(registry, Registering.kept, hasattr(Registering, "registry"))

tally = 0


class Counting:
    global tally
    for step in range(3):
        tally += 1


print(tally, Counting.step, hasattr(Counting, "tally"))

flag = None


class Branching:
    global flag
    if True:
        flag = "set"
        local = "kept"


print(flag, Branching.local, hasattr(Branching, "flag"))
