# While a generator is delegating it is the sub-iterator that is suspended,
# so closing or throwing at the outer one has to reach the inner one.
def inner():
    try:
        yield 1
        yield 2
    finally:
        print("inner cleaned")


def outer():
    yield from inner()
    yield "never"


delegating = outer()
print(next(delegating))
delegating.close()


def catching():
    try:
        yield 1
    except ValueError as error:
        print("inner caught", error)
        yield 99


def over():
    answer = yield from catching()
    yield ("returned", answer)


throwing = over()
print(next(throwing))
print(throwing.throw(ValueError("v")))


def over_a_list():
    yield from [1, 2, 3]


listed = over_a_list()
print(next(listed))
listed.close()
print("a list has neither close nor throw, and that is allowed")


def deep_inner():
    try:
        yield 1
    finally:
        print("deep cleaned")


def middle():
    yield from deep_inner()


def top():
    yield from middle()


stacked = top()
next(stacked)
stacked.close()
