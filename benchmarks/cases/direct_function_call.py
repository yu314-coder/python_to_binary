def add(a, b):
    return a + b
def bench():
    t = 0
    for i in range(300000):
        t = add(t, i)
    return t
