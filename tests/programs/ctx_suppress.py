class A:
    def __enter__(s): return 1
    def __exit__(s,*a): return True
with A(): raise ValueError('x')
print('suppressed')
