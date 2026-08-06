class A:
    def __new__(cls, *args):
        made = super().__new__(cls)
        made.tag = "new"
        return made
    def __init_subclass__(cls, **kw):
        cls.sub = "set"
    def __class_getitem__(cls, item):
        return ("item", item)
class B(A):
    pass
a = A()
print(a.tag, B.sub, A[int])
