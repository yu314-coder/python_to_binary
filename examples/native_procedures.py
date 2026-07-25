def announce(flag: bool, repeat: int = 2) -> None:
    for index in range(repeat):
        if flag:
            print("native procedure")
        else:
            print("alternate path")
    if not flag:
        return
    print("procedure complete")


announce(True)
raise SystemExit(7)
