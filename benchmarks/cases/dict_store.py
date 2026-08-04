def bench():
    d = {}
    for i in range(300000):
        d[i % 64] = i
    return len(d)
