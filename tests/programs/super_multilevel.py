class A:
    def f(s): return 'A'
class B(A):
    def f(s): return 'B'+super().f()
class C(B):
    def f(s): return 'C'+super().f()
print(C().f())
