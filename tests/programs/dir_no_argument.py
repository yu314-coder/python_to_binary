# `dir()` with nothing passed is `sorted(locals())`, and in a module
# `locals()` is `globals()` - the same two answers the compiler already
# builds, with the names taken out and put in order. It was refused for
# wanting a frame it does not want.
def in_a_function(a, b=2):
    c = a + b
    print(dir())
    d = "bound after the first call"
    print(dir())
    return c, d


print(in_a_function(1)[0])


def with_every_parameter_kind(a, *rest, k=1, **more):
    return dir()


print(with_every_parameter_kind(1, 2, k=3, z=4))


def only_on_one_branch(flag):
    if flag:
        only_here = 1
    return dir()


print(only_on_one_branch(False), only_on_one_branch(True))


def after_a_del():
    a = 1
    b = 2
    del b
    return dir()


print(after_a_del())


def names_never_read():
    used = 1
    never_read = "x"
    print(dir())
    also_never = "y"
    print(dir())
    return used


names_never_read()


class InAClassBody:
    x = 1
    y = 2
    # A class body's `locals()` is the namespace the class is being made
    # from, not the module's.
    seen = [n for n in dir() if not n.startswith("_")]


print(InAClassBody.seen)


class AMethod:
    def m(self, v):
        w = v + 1
        return dir()


print(AMethod().m(1))

alpha = 1
beta = 2
at_module_level = dir()
print([n for n in at_module_level if not n.startswith("_")])
print(at_module_level == sorted(at_module_level))
for dunder in ("__name__", "__doc__", "__file__", "__package__", "__builtins__"):
    print(dunder, dunder in at_module_level)
