def bench():
    t = 0
    for i in range(300000):
        try:
            t += 1
        except ValueError:
            pass
    return t
