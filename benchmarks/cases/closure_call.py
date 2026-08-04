def make(step):
    def bump(v):
        return v + step
    return bump
def bench():
    bump = make(2)
    t = 0
    for i in range(300000):
        t = bump(t)
    return t
