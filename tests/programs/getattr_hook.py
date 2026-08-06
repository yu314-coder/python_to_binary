class A:
    def __getattr__(s,n): return 'made:'+n
print(A().zzz)
