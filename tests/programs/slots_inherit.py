class A:
    __slots__=('x',)
class B(A):
    __slots__=('y',)
b=B(); b.x=1; b.y=2
print(b.x,b.y)
