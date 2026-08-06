class A: pass
class B(A): pass
class C(A): pass
class D(B,C): pass
print([c.__name__ for c in D.__mro__])
