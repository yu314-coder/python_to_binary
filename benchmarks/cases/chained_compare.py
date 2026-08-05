def bench():
    t = 0
    for i in range(300000):
        if 0 < i < 299999:
            t += 1
    return t
