def bench():
    t = 0
    for i in range(300000):
        try:
            raise ValueError(i)
        except ValueError:
            t = t + 1
    return t
