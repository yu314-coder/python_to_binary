class A:
    def __radd__(s,o): return 'r%d'%o
print(1+A())
