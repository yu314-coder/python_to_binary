# A compiled function's doc slot carries its signature, so `inspect` can read
# one. What the function itself says goes after it, which is where CPython
# reads `__doc__` from - left out, `help()` said nothing about anything.
import inspect


def one_line():
    "says one thing"
    return 1


def paragraphs(a, b=2):
    """Says several.

    And then some more.
    """
    return a + b


class Described:
    """What this class is for."""

    def method(self):
        """What this method is for."""
        return 1


class Undescribed:
    pass


print(repr(one_line.__doc__))
print(repr(paragraphs.__doc__))
print(repr(Described.__doc__), repr(Undescribed.__doc__))
print(repr(Described.method.__doc__))
print(str(inspect.signature(paragraphs)))
print(inspect.getdoc(one_line))
