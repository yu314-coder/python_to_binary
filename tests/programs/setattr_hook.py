class A:
    def __setattr__(s,n,v): object.__setattr__(s,n,v*2)
a=A(); a.x=3; print(a.x)
