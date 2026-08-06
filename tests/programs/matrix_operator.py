class M:
    def __matmul__(s,o): return 'mm'
    def __imatmul__(s,o): return 'imm'
m=M()
print(m@m)
m@=m
print(m)
