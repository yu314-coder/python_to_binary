def bench():
    t = 0.0
    step = 0.5
    for i in range(300000):
        t = t + step * 2.0 - 0.25
    return t
