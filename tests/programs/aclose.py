import asyncio
async def g():
    try:
        yield 1
    finally: print('closed')
async def m():
    it=g()
    await it.__anext__()
    await it.aclose()
asyncio.run(m())
