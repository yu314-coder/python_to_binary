import asyncio
async def below(n, limit):
    await asyncio.sleep(0)
    return n < limit
async def m():
    seen = []
    n = 0
    while await below(n, 3):
        seen.append(n)
        n = n + 1
    else:
        seen.append("else")
    n = 0
    while await below(n, 5):
        if n == 2:
            break
        n = n + 1
    else:
        seen.append("not reached")
    return seen, n
print(asyncio.run(m()))
