import asyncio
async def one(v):
    await asyncio.sleep(0)
    return v * 2
async def gen(n):
    for i in range(n):
        yield i
async def main():
    a = [await one(i) for i in range(3)]
    b = {await one(i) for i in range(2)}
    c = {i: await one(i) for i in range(2)}
    d = [await one(x) async for x in gen(2)]
    e = [await one(i) for i in range(4) if i % 2 == 0]
    return a, sorted(b), c, d, e
print(asyncio.run(main()))
