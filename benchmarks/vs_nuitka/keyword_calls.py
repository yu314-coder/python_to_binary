def add(a, step):
    return a + step


def main():
    t = 0
    for i in range(1000000):
        t = add(t, step=i)
    print(t)


main()
