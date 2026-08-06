class A:
    def __init__(s,v): s.v=v
    def __eq__(s,o): return s.v==o.v
    def __hash__(s): return hash(s.v)
print(A(1)==A(1), len({A(1),A(1)}))
