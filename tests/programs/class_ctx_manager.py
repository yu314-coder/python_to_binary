class A:
    def __enter__(s): return 1
    def __exit__(s,*a): print('exit'); return False
with A() as v: print(v)
