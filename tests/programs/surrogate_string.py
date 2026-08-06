s = "a\udcffb"
print(s.encode("utf-8", "replace"), len(s), ord(s[1]))
