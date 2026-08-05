"""A local nothing ever reads does not need a slot of its own.

The frame a compiled function gets is a single fixed allocation with a
ceiling on it, so the number of locals is not merely a matter of taste: a
generated program with sixty-seven thousand of them - each assigned once and
never read - could not be compiled at all, and was the one program in an
889-program corpus that py2bin refused. It needed two slots and asked for
67,001.

The value is still computed. `v = f()` runs `f` whether or not anything reads
`v`; only the storage is shared. What has to be got right is the refusals -
every way a binding can be observed without the name being loaded - and those
are most of what is tested here.
"""

from __future__ import annotations

import ast

from py2bin.capi_emit import write_only_locals


def _unread(source: str, parameters: set[str] | None = None) -> set[str]:
    body = ast.parse(source).body
    return write_only_locals(body, parameters or set())


def test_a_name_never_read_is_found():
    assert _unread("a = 1\nb = 2\nprint(a)\n") == {"b"}


def test_a_name_that_is_read_is_kept():
    assert _unread("a = 1\nprint(a)\n") == set()


def test_a_parameter_is_never_one():
    # It arrives holding something the caller owns, whoever reads it.
    assert _unread("n = 1\n", {"n"}) == set()


def test_reading_it_later_counts_however_far_away():
    assert _unread("a = 1\nif x:\n    while y:\n        print(a)\n") == set()


def test_a_closure_reading_it_counts():
    """The read is inside a nested `def`, and it is still a read."""
    source = "a = 1\ndef inner():\n    return a\nreturn inner\n"
    assert "a" not in _unread(source)


def test_an_augmented_assignment_reads_first():
    assert _unread("t = 0\nt += 1\n") == set()


def test_anything_that_can_read_every_local_refuses_all_of_them():
    for reader in ("locals", "vars", "eval", "exec", "globals"):
        source = f"a = 1\nb = 2\nreturn {reader}()\n"
        assert _unread(source) == set(), reader


def test_a_global_or_nonlocal_name_is_not_a_local():
    assert _unread("global a\na = 1\n") == set()
    assert _unread("nonlocal a\na = 1\n") == set()


def test_a_deleted_name_has_to_exist_to_delete():
    assert _unread("a = 1\ndel a\n") == set()


def test_a_walrus_is_written_for_its_value():
    assert _unread("if (a := 1):\n    pass\n") == set()


def test_an_except_name_is_unbound_when_the_handler_ends():
    source = "try:\n    pass\nexcept ValueError as e:\n    pass\n"
    assert _unread(source) == set()


def test_a_loop_target_nothing_reads_is_one():
    assert _unread("for i in range(3):\n    pass\n") == {"i"}


def test_a_loop_target_something_reads_is_not():
    assert _unread("for i in range(3):\n    print(i)\n") == set()


def test_unpacking_into_names_nothing_reads():
    assert _unread("a, b = 1, 2\nprint(a)\n") == {"b"}


def test_an_attribute_store_reads_the_name_it_stores_through():
    # `a.x = 1` loads a and then sets on it, so a is read.
    assert _unread("a = C()\na.x = 1\n") == set()


def test_a_subscript_store_reads_the_name_too():
    assert _unread("a = []\na[0] = 1\n") == set()
