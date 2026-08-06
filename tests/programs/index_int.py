class A:
    def __index__(s): return 2
print([1,2,3][A()])
