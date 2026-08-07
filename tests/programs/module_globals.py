# `globals()` answers with the running module's own dictionary, and a module's
# globals are its own. One slot held the entry's, so this read `__main__`'s
# from anywhere - and where the entry needed none, there was no slot at all.
NAME = "entry"


def mine():
    return sorted(k for k in globals() if not k.startswith("_") and k.isupper())


print(mine(), globals().get("NAME"), globals().get("ELSEWHERE", "absent"))
exec("MADE = 1")
print(globals().get("MADE"))
