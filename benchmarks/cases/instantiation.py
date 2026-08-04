class Point:
    def __init__(self, x):
        self.x = x
def bench():
    t = 0
    for i in range(300000):
        t = t + Point(i).x
    return t
