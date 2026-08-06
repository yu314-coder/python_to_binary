def d(c):
    c.tag=7
    return c
@d
class A: pass
print(A.tag)
