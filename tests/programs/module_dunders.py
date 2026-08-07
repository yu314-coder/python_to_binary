# Every module has these besides the names its own statements bind. Without
# them a bare mention fell through to the builtins module - where most of them
# exist and are *its* - so `__spec__` answered with the spec of `builtins`,
# whose `.name` is "builtins", and `__package__` with its empty string.
print(__name__, __package__, __spec__, __debug__)
print(type(__builtins__).__name__, isinstance(__doc__, (str, type(None))))
print(bool(__file__))

# Defined rather than looked for among the builtins. What it holds cannot
# match: a compiled module was not loaded by anything, so there is no loader
# to name and it is None, where CPython names the one that read the source.
try:
    __loader__
    print("loader is defined")
except NameError:
    print("loader is missing")


def read_from_a_function():
    # The same names, read where the module's globals are C slots rather than
    # a dictionary - which is where the fall-through used to happen.
    return (__name__, __package__, __spec__, type(__builtins__).__name__)


print(read_from_a_function())


class ReadFromAClassBody:
    seen = (__name__, __package__)


print(ReadFromAClassBody.seen)
