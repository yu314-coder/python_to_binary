def bench():
    xs = list(range(1000))
    t = 0
    for _ in range(300):
        for x in xs:
            t += x
    return t
