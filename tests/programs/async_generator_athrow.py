# `athrow` raises at the suspension point of an async generator, and is
# awaited rather than called - so it is an object with an `__await__`, like
# `asend` and `aclose`.
import asyncio


async def catching():
    try:
        yield 1
    except ValueError as error:
        print("caught", error)
        yield 2


async def plain():
    yield 1
    yield 2


async def cleaning():
    try:
        yield 1
    finally:
        print("cleaned")


async def awaiting():
    try:
        yield 1
    except KeyError:
        await asyncio.sleep(0)
        yield "after an await"


async def main():
    it = catching()
    print(await it.__anext__())
    print(await it.athrow(ValueError("v")))

    straight = plain()
    print(await straight.__anext__())
    try:
        await straight.athrow(RuntimeError("r"))
    except RuntimeError as error:
        print("propagated", error)

    held = cleaning()
    print(await held.__anext__())
    try:
        await held.athrow(KeyError("k"))
    except KeyError:
        print("and the cleanup ran first")

    waited = awaiting()
    print(await waited.__anext__())
    print(await waited.athrow(KeyError("k")))


asyncio.run(main())
