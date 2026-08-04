def bench():
    t = 0
    for i in range(300000):
        t = t + i * 2 - 1
    return t
