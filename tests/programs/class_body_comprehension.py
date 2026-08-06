class C:
    n = 3
    counted = [i for i in range(n)]
    xs = [1, 2, 3]
    picked = [v for v in xs if v > 1]
    nested = [(a, b) for a in xs for b in (10, 20) if a != 2]
    mapped = {k: k for k in xs}
    fn = staticmethod(lambda: 7)
print(C.counted, C.picked, C.nested, C.mapped, C.fn())
def make():
    try:
        class D:
            limit = 2
            bad = [limit for _ in range(2)]
    except NameError as e:
        return str(e)
print(make())
