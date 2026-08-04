def bench():
    t = 0
    for i in range(300000):
        if i > 5 and i < 299995:
            t = t + 1
    return t
