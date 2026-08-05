class C:
    def __init__(s): s.v = 0
def bench():
    c = C()
    for i in range(300000):
        c.v = i
    return c.v
