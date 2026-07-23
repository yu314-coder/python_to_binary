"""Runtime integer/control-flow example for the direct machine-code backend."""

total = 0
for value in range(1, 11):
    total += value

counter = 0
while counter < 3:
    total += counter
    counter += 1

if total != 58:
    raise SystemExit(1)

raise SystemExit(total - 3)
