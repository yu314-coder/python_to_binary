def add(a, b):
    return a + b


def main():
    t = 0
    for i in range(1000000):
        t = add(t, i)
    print(t)


main()
