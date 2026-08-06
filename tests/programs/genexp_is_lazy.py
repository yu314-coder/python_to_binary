import itertools
calls = []
def note(v):
    calls.append(v)
    return v
g = (note(i) for i in range(3))
print("before", calls)
print(next(g), calls)
endless = (x * 2 for x in itertools.count())
print("made", next(endless), next(endless))
def outer():
    def inner(n):
        yield from range(n)
    for i in inner(2):
        yield i * 10
print(list(outer()))
if True:
    def guarded():
        yield 1
        yield 2
print(list(guarded()))
