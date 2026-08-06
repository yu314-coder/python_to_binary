def d(n):
    def outer(f):
        def w(*a): return f(*a)*n
        return w
    return outer
@d(3)
def f(x): return x
print(f(2))
