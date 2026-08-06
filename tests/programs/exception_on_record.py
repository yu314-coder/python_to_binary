# Which exception is being handled belongs to the call, not to the thread:
# CPython keeps it per frame and gives the caller's back when a frame ends.
import sys


def finally_raises():
    try:
        raise ValueError("a")
    finally:
        raise KeyError("b")


try:
    finally_raises()
except KeyError as error:
    print(type(error).__name__, type(error.__context__).__name__)
print("after a finally raised:", sys.exc_info()[0])


def handler_raises():
    try:
        raise ValueError("a")
    except ValueError:
        raise KeyError("b")


try:
    handler_raises()
except KeyError:
    pass
print("after a handler raised:", sys.exc_info()[0])


def inner():
    try:
        raise ValueError("v")
    except ValueError:
        print("inner handler:", sys.exc_info()[0].__name__)
    print("inner after:", sys.exc_info()[0].__name__)


def outer():
    try:
        raise TypeError("t")
    except TypeError:
        print("outer handler:", sys.exc_info()[0].__name__)
        inner()
        print("back in outer:", sys.exc_info()[0].__name__)


outer()
print("at the module:", sys.exc_info()[0])


def suspending():
    try:
        yield 1
    except RuntimeError:
        # The clause runs in a block of its own here, reached after the
        # `except` that caught it has ended - so it has to put the exception
        # on record itself.
        print("in a generator:", sys.exc_info()[0].__name__)
        try:
            raise KeyError("k")
        except KeyError as error:
            print("chained to:", type(error.__context__).__name__)


machine = suspending()
next(machine)
try:
    machine.throw(RuntimeError("r"))
except StopIteration:
    pass
print("after the generator:", sys.exc_info()[0])


class Boom:
    def __enter__(self):
        return self

    def __exit__(self, *ignored):
        raise KeyError("from __exit__")


def with_raises():
    with Boom():
        raise ValueError("body")


try:
    with_raises()
except KeyError as error:
    print("with:", type(error.__context__).__name__)
print("after the with:", sys.exc_info()[0])
