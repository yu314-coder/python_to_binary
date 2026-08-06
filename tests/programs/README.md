# The conformance corpus

Each of these is a whole program that prints something. The check is not that
py2bin does what somebody meant - the test suite is for that - it is that the
compiled program's stdout and exit code are what CPython answers for the same
source, character for character.

That is the claim the project rests on, and it is the only check that takes
py2bin's word for nothing. `.github/workflows/checks.yml` runs it on every
push; `tests/test_conformance.py` runs it with the suite.

They are here because they each caught something. A default argument
evaluated per call, `x += y` rebuilding a list instead of extending it, a
closure reading the module's `f` rather than the one it was handed - none of
those were found by reading code, and none of them would have been noticed
without a program that prints what it got.

Add one whenever a shape turns out to be wrong, and leave it here afterwards.
