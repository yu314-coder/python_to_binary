BIAS = 3


def twice(value: int) -> int:
    doubled = value * 2
    return doubled


def affine(value: int) -> int:
    adjusted = twice(value) + BIAS
    if adjusted < 9:
        return adjusted + 1
    return adjusted - 1


def sum_affine(start: int, stop: int) -> int:
    total = 0
    for value in range(start, stop):
        total += affine(value)
    return total
