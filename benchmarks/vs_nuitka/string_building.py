def main():
    parts = []
    for i in range(200000):
        parts.append("value-" + str(i))
    print(len("".join(parts)))


main()
