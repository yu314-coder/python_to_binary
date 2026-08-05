def bench():
    p = (1, 2)
    t = 0
    for i in range(300000):
        a, b = p
        t += a + b
    return t
