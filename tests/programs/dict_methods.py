d={'a':1}
print(d.setdefault('b',2), d.pop('a'), list(d.items()), d.get('z','dflt'))
