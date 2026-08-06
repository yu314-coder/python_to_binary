def a():
    yield 1
    return 'v'
def b():
    r = yield from a()
    yield r
print(list(b()))
