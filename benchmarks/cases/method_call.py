class Counter:
    def __init__(self):
        self.n = 0
    def step(self, by):
        return by + 1
def bench():
    c = Counter()
    t = 0
    for i in range(300000):
        t = c.step(t)
    return t
