import asyncio
calls = []
async def g(v=1):
    calls.append(v)
    await asyncio.sleep(0)
    return v
async def m():
    a = False and await g("a")
    b = True and await g("b")
    c = True or await g("c")
    d = False or await g("d")
    e = await g("e") if True else await g("no")
    f = await g("f1") if False else await g("f2")
    h = 1 and 2 and await g("h")
    return a, b, c, d, e, f, h, calls
print(asyncio.run(m()))
