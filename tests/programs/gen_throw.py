def g():
    try:
        yield 1
    except ValueError:
        yield 'caught'
it=g(); next(it); print(it.throw(ValueError()))
