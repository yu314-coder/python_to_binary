import weakref
class A: pass
a=A(); r=weakref.ref(a)
print(r() is a)
