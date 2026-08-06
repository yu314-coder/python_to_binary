class A:
    def __init__(s): s.v=0
    def __iadd__(s,o): s.v+=o; return s
a=A(); a+=3; print(a.v)
