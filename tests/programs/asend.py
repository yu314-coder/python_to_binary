import asyncio
async def g():
    x = yield 1
    yield x*2
async def m():
    it=g(); a=await it.asend(None); b=await it.asend(5); return a,b
print(asyncio.run(m()))
