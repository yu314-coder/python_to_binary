# `super()` with nothing in the brackets stands for two values CPython
# supplies through a cell it makes for the method. A `def` written inside a
# method is not itself a method and gets neither.
class Base:
    def go(self):
        return "base"


class Derived(Base):
    def go(self):
        def inner():
            return super().go()

        return inner()

    def fine(self):
        return super().go() + "!"

    def in_a_lambda(self):
        return (lambda: super().go())()

    def spelled_out(self):
        return super(Derived, self).go() + "?"


print(Derived().fine())
print(Derived().spelled_out())
for attempt in (Derived().go, Derived().in_a_lambda):
    try:
        attempt()
    except RuntimeError as error:
        print(type(error).__name__, error)
