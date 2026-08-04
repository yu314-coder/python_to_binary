class Holder:
    def __init__(self):
        self.value = 1
def bench():
    h = Holder()
    t = 0
    for i in range(300000):
        t = t + h.value
    return t
