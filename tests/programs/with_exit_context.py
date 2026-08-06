class C:
    def __enter__(self): return self
    def __exit__(self, *a): raise KeyError("from exit")
try:
    with C():
        raise ValueError("body")
except KeyError as e:
    print(type(e).__name__, type(e.__context__).__name__)
class Quiet:
    def __enter__(self): return self
    def __exit__(self, *a): return False
try:
    with Quiet():
        raise ValueError("v")
except ValueError as e:
    print("plain", e.__context__ is None)
import contextlib
with contextlib.suppress(ValueError):
    raise ValueError("gone")
print("suppressed")
