# The two ends of a generator's life, where there is no `yield` suspended to
# resume: before it has run at all, and after it has finished.
def counted():
    yield 1
    yield 2


it = counted()
print(next(it), next(it))
for _ in range(3):
    try:
        next(it)
    except StopIteration:
        print("stopped")

fresh = counted()
print("close:", fresh.close(), fresh.close())

thrown = counted()
try:
    thrown.throw(KeyError("k"))
except KeyError:
    print("throw into a fresh generator comes straight back")
print("and it is finished:", list(thrown))


def raises():
    raise ValueError("boom")
    yield


broken = raises()
try:
    next(broken)
except ValueError as error:
    print("raised", error)
try:
    next(broken)
except StopIteration:
    print("a generator that raised does not raise again")


def cleans():
    try:
        yield 1
    finally:
        print("cleaned")


running = cleans()
next(running)
running.close()


def refuses():
    try:
        yield 1
    except GeneratorExit:
        yield 2


stubborn = refuses()
next(stubborn)
try:
    stubborn.close()
except RuntimeError as error:
    print(error)
