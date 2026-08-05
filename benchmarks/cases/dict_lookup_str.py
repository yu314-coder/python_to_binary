def bench():
    d = {'alpha': 1, 'beta': 2, 'gamma': 3}
    t = 0
    for i in range(300000):
        t += d['beta']
    return t
