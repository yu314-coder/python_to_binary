class A:
    def __init_subclass__(c,**kw):
        c.tag='set'
class B(A): pass
print(B.tag)
