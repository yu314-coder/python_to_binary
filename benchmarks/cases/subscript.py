def bench():
    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    t = 0
    for i in range(300000):
        t = t + xs[i % 8]
    return t
