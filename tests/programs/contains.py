class A:
    def __contains__(s,v): return v==2
print(2 in A(), 3 in A())
