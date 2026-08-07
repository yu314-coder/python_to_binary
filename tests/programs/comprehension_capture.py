# A closure written inside a comprehension shares the comprehension's
# variable, so every one of them sees the last value it took. Captures here
# are taken by value, which would answer differently, so this was refused -
# in every form but the generator expression, where it raised `NameError` at
# the call instead. The target gets a cell now, as the same shape written as
# a `for` statement already did.
print([f() for f in [lambda: i for i in range(3)]])

# The cell is named after the comprehension, not the variable, so a name of
# the same spelling outside is never involved - which is what 3.12 onwards
# arrives at by saving and restoring it around an inlined comprehension.
i = "outer"
print([f() for f in [lambda: i for i in range(2)]], repr(i))


def in_a_function():
    j = "outer"
    made = [lambda: j for j in range(3)]
    return [f() for f in made], j


print(in_a_function())

# Every comprehension form, including the one that used to raise.
print([f() for f in list((lambda: q) for q in range(3))])
print(sorted(f() for f in {(lambda: s) for s in range(3)}))
print({k: v() for k, v in {n: (lambda: n) for n in range(2)}.items()})

# Two comprehensions do not share a cell.
first = [lambda: i for i in range(2)]
second = [lambda: i for i in range(3)]
print([f() for f in first], [f() for f in second])

# One inside a loop gets a cell per turn, because each turn's comprehension
# is its own.
gathered = []
for _ in range(2):
    gathered.append([lambda: i for i in range(2)])
print([[f() for f in made] for made in gathered])

# Several clauses, and a condition.
print([f() for f in [lambda: (a, b) for a in range(2) for b in range(2)]])
print([f() for f in [lambda: i for i in range(4) if i % 2 == 0]])

# A default says the by-value thing deliberately: the parameter is the
# lambda's own name, and only the default is the comprehension's.
print([f() for f in [lambda i=i: i for i in range(3)]])

# Capturing something from outside the comprehension as well.
outer = 10
print([f() for f in [lambda: i + outer for i in range(3)]])

# The ordinary forms are untouched.
print([i * 2 for i in range(4)], sorted({i for i in range(3)}), sum(i for i in range(5)))
