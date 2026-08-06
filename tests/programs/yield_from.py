def a():
    yield 1
    yield 2
def b():
    yield 0
    r = yield from a()
    yield 3
print(list(b()))
