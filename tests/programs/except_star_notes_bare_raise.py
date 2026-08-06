def show(fn):
    try:
        fn()
        print('(none)')
    except BaseException as e:
        print(spell(e))
def spell(e):
    if isinstance(e, BaseExceptionGroup):
        return type(e).__name__ + '(' + repr(e.args[0]) + ', [' + ', '.join(spell(x) for x in e.exceptions) + '])'
    return type(e).__name__ + '(' + repr(e.args[0]) + ')' if e.args else type(e).__name__
def f():
    g = ExceptionGroup('g',[ValueError('a'),TypeError('b')])
    g.add_note('n')
    try:
        raise g
    except* ValueError as e:
        raise
show(f)
