import sys
OUTER = 99
class C:
    if sys.platform:
        v = 1
    else:
        v = 2
    vs = []
    for i in (1, 2):
        vs.append(i)
    try:
        import json
        have = True
    except ImportError:
        have = False
    OUTER = "inner"
    while len(vs) < 3:
        vs.append(9)
print(C.v, C.vs, C.have, C.i, C.OUTER, OUTER)
print(hasattr(C, "json"), C.__dict__["v"])
