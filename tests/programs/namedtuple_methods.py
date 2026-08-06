from collections import namedtuple
P=namedtuple('P','x y')
p=P(1,2)
print(p._asdict(), p._replace(x=9), p[0], tuple(p))
