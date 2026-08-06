try:
    d = {[1]: 2}
except TypeError as e:
    print("dict:", e)
try:
    s = {[1]}
except TypeError as e:
    print("set:", e)
