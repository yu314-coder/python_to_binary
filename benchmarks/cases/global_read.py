LIMIT = 7
def bench():
    t = 0
    for i in range(300000):
        t += LIMIT
    return t
