def f(v):
    match v:
        case [a,*rest] if a>0: return (a,rest)
        case _: return None
print(f([1,2,3]), f([-1,2]))
