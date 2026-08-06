class P:
    __match_args__=('x','y')
    def __init__(s,x,y): s.x=x; s.y=y
def f(v):
    match v:
        case P(0,y): return 'axis %d'%y
        case P(x,y): return 'pt %d %d'%(x,y)
print(f(P(0,5)), f(P(1,2)))
