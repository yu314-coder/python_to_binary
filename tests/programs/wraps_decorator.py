# `functools.wraps` writes six attributes onto the wrapper it is given. A
# compiled function has no `__dict__` to write them on, so the idiom nearly
# every decorator in Python is written with used to raise at import time.
import functools
from functools import wraps


def double(f):
    @functools.wraps(f)
    def wrapper(*args):
        return f(*args) * 2

    return wrapper


@double
def greet(n):
    """says hello"""
    return n


def keep(f):
    @wraps(f)
    def inner(*args, **named):
        return f(*args, **named)

    return inner


@keep
def plain():
    return "p"


def prefixed(text):
    def outer(f):
        @wraps(f)
        def inner(*args):
            return text + str(f(*args))

        return inner

    return outer


@prefixed("v=")
def valued():
    return 9


class Holder:
    @double
    def method(self, n):
        return n + 1


print(greet(3), greet.__name__, greet.__doc__)
print(greet.__wrapped__(3), plain(), plain.__name__)
print(valued(), valued.__name__, Holder().method(1))
# Read off the instance, which is where a bound one is asked for its name.
print(Holder().method.__name__, Holder.method.__name__)
print(greet.__doc__, Holder().method.__doc__)
