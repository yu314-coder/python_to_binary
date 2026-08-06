def f():
    out=[]
    try:
        try: raise ValueError('a')
        finally: out.append('inner')
    except ValueError: out.append('caught')
    finally: out.append('outer')
    return out
print(f())
