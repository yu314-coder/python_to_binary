def helper(v, step):
    return v + step
def bench():
    t = 0
    for i in range(300000):
        t = helper(t, step=1)
    return t
