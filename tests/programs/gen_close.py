def g():
    try:
        yield 1
    finally: print('closed')
it=g(); next(it); it.close()
