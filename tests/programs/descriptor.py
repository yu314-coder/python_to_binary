class D:
    def __get__(s,o,t): return 42
class A:
    d=D()
print(A().d)
