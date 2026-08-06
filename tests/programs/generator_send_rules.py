def g():
    x = yield 1
    yield x
it = g()
try:
    it.send(5)
except TypeError as e:
    print("just-started:", e)
print("send(None):", g().send(None))
it2 = g(); print(next(it2), it2.send(7))
it3 = g(); next(it3); next(it3)
try:
    it3.send(1)
except StopIteration:
    print("after end: StopIteration")
