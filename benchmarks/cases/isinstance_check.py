def bench():
    x = 5
    t = 0
    for i in range(300000):
        if isinstance(x, int):
            t += 1
    return t
