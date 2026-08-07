# An owned reference taken for a clause has to be let go however the clause
# ends. These all end some way other than falling off the end, which is the
# way that used to be the only one that released anything.
import gc
import resource


def settled(work, turns=40000):
    """Whether the second identical run adds memory the first gave back."""
    work(500)
    gc.collect()
    resource.getrusage(resource.RUSAGE_SELF)
    work(turns)
    gc.collect()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    work(turns)
    gc.collect()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (after - before) < 1024 * 1024


def handler_raises(turns):
    def inner():
        try:
            raise ValueError("a")
        except ValueError:
            raise KeyError("b")

    for _ in range(turns):
        try:
            inner()
        except KeyError:
            pass


def finally_raises(turns):
    def inner():
        try:
            raise ValueError("a")
        finally:
            raise KeyError("b")

    for _ in range(turns):
        try:
            inner()
        except KeyError:
            pass


def handler_returns(turns):
    def inner():
        try:
            raise ValueError("a")
        except ValueError:
            return 1

    for _ in range(turns):
        inner()


def generator_handler_suspends(turns):
    def machine():
        try:
            yield 1
        except ValueError:
            yield 2

    for _ in range(turns):
        it = machine()
        next(it)
        it.throw(ValueError("x"))


for name, work in (
    ("handler raises", handler_raises),
    ("finally raises", finally_raises),
    ("handler returns", handler_returns),
    ("generator handler suspends", generator_handler_suspends),
):
    print(f"{name}: settled {settled(work)}")
