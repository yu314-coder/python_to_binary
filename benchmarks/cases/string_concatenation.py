def bench():
    t = 0
    for i in range(300000):
        t = t + len("value" + "-" + "tail")
    return t
