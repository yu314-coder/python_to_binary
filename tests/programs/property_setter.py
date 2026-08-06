class A:
    def __init__(s): s._v=1
    @property
    def v(s): return s._v
    @v.setter
    def v(s,x): s._v=x*2
a=A(); a.v=5; print(a.v)
