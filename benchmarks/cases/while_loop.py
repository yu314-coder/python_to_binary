def bench():
    i = 0
    t = 0
    while i < 300000:
        t = t + i
        i = i + 1
    return t
