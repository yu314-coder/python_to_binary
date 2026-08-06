def g():
    try:
        yield 1
    except ValueError:
        yield "caught"
it = g(); next(it); print(it.throw(ValueError()))
def h():
    try:
        yield 1
    finally:
        print("cleanup")
i2 = h(); next(i2); i2.close()
def k():
    x = yield 1
    yield x * 2
i3 = k(); print(next(i3), i3.send(5))
def m():
    yield 1
i4 = m(); next(i4)
try:
    i4.throw(KeyError("boom"))
except KeyError as e:
    print("escaped:", e)
