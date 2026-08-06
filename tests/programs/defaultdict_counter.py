from collections import defaultdict, Counter
d=defaultdict(list); d['a'].append(1)
print(dict(d), Counter('aab').most_common(1))
