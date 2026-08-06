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
import sys
def f():
    try:
        raise ExceptionGroup('g',[ValueError('a')])
    except* ValueError as e:
        print('same', sys.exc_info()[1] is e, e.__context__ is None)
show(f)
