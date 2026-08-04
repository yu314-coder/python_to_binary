def bench():
    t = 0
    for i in range(300000):
        t = t + len(f"value {i} tail")
    return t
