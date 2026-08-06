# A `finally` that runs while an exception is on its way out runs with that
# exception on record, so anything it raises is chained to it.
import sys


def replaces():
    try:
        raise ValueError("first")
    finally:
        raise KeyError("second")


try:
    replaces()
except KeyError as error:
    print(type(error).__name__, type(error.__context__).__name__, error.__context__)


def reports():
    try:
        raise ValueError("v")
    finally:
        kind = sys.exc_info()[0]
        print("handling:", kind.__name__ if kind else None)


try:
    reports()
except ValueError:
    print("still on its way out")


def quiet():
    try:
        pass
    finally:
        print("nothing being handled:", sys.exc_info()[0])


quiet()


def nested():
    try:
        raise ValueError("outer")
    finally:
        try:
            raise TypeError("inner")
        except TypeError as error:
            print("nested context:", type(error.__context__).__name__)


try:
    nested()
except ValueError:
    print("outer survives")


def restores():
    try:
        try:
            raise ValueError("gone")
        finally:
            pass
    except ValueError:
        pass
    print("put back:", sys.exc_info()[0])


restores()
