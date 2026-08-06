def f(a,b,c=3,*rest,**kw): return (a,b,c,rest,sorted(kw))
print(f(*[1,2],**{'d':4}))
