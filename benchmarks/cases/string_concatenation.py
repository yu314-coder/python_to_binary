def bench():
    # Locals, not literals. `"a" + "b"` is folded at compile time, so a case
    # built from literals measured `len` of a constant and an integer add -
    # the generated C held no concatenation at all, and the row was named
    # after something it never ran.
    head = "value"
    tail = "tail"
    t = 0
    for i in range(300000):
        t = t + len(head + "-" + tail)
    return t
