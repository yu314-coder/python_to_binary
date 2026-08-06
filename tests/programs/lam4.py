fs = [lambda i=i: i for i in range(3)]
print([f() for f in fs])
pairs = {k: (lambda k=k: k * 2) for k in (1, 2)}
print({k: f() for k, f in pairs.items()})
