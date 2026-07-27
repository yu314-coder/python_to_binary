"""User-defined classes compiled to real machine code.

Instances are heap blocks with a statically known layout: attribute *i* lives at
``pointer + i * 8``. Because the class of every instance is known at build time,
there is no object header, type pointer, or vtable, and each method call
resolves directly to one body that is inlined into the caller. No CPython
runtime, interpreter, or external compiler is involved.

    PYTHONPATH=src python3 -m py2bin compile examples/native_classes.py \
        --target darwin-arm64 --output dist/native-classes --clean
    ./dist/native-classes; echo $?      # -> 36

Objects share the same bump arena as runtime lists and strings, so this example
compiles for the POSIX targets. The Windows targets reject it, because their
allocator is not wired up yet -- an explicit gap rather than a broken binary.
"""


class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def grow(self, amount):
        self.width = self.width + amount
        self.height = self.height + amount


class Tally:
    def __init__(self):
        self.total = 0

    def add(self, value):
        self.total = self.total + value


small = Rectangle(2, 3)          # area 6
large = Rectangle(4, 5)          # area 20

small.grow(1)                    # now 3 x 4 -> area 12

running = Tally()
running.add(small.area())        # 12
running.add(large.area())        # 20

# Attributes can also be read and written directly.
large.height = large.height + 3  # 4 x 8 -> area 32

sides = [small.width, large.width]
raise SystemExit(running.total + sides[0] + sides[1] + large.height - 11)
