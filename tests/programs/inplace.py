xs = [1]
alias = xs
xs += [2]
print(xs, alias, xs is alias)
s = {1}
t = s
s |= {2}
print(sorted(s), sorted(t), s is t)
