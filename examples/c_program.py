def triangular(limit: int) -> int:
    total = 0
    for value in range(limit + 1):
        total += value
    return total


answer = triangular(10)
print(f"triangular(10) = {answer}")
