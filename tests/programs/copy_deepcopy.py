import copy
d={'a':[1]}
c=copy.deepcopy(d); c['a'].append(2)
print(d,c)
