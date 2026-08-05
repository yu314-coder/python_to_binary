"""`f(a, step=1)` is the call `f(a, 1)` is, so it is written as that one.

Naming an argument says something at the call site about the callee's
parameters, and when both are in the same module both are known here. Settling
it at compile time is not about the name lookup saved: a keyword stopped the
call being inlined and stopped it being a direct C call, so the argument
binding ran on every call *and* - because the value came back through a
PyObject - the loop around it kept everything boxed. The benchmark row went
from 27.8 ms to 3.4 ms against the interpreter's 7.8, which is 0.28x to 2.28x.

What matters for correctness is the other half: every call this cannot settle
*exactly* has to be left alone, so the interpreter binds it at run time and
raises what it has always raised. The tests below are mostly those.
"""

from __future__ import annotations

import ast

from py2bin.capi_inline import place_keywords


def _placed(source: str) -> str:
    return ast.unparse(place_keywords(ast.parse(source)))


def _call(source: str) -> str:
    """The last line, which is the call under test."""
    return _placed(source).splitlines()[-1]


TWO = "def two(a, b):\n    return a + b\n"
DEFAULTS = "def defaults(a, b=2, c=3):\n    return (a, b, c)\n"


def test_a_named_argument_becomes_a_positional_one():
    assert _call(TWO + "two(1, b=2)") == "two(1, 2)"


def test_every_argument_named_is_still_the_same_call():
    assert _call(TWO + "two(a=1, b=2)") == "two(1, 2)"


def test_names_out_of_order_are_put_in_order():
    # Both arguments are literals, so moving them past each other cannot be
    # noticed.
    assert _call(TWO + "two(b=2, a=1)") == "two(1, 2)"


def test_names_out_of_order_are_left_alone_when_moving_would_show():
    """Python evaluates arguments in the order they are written.

    `two(b=g(), a=h())` calls g before h. Written positionally it would call h
    first, which is a different program - so this one stays as it is and the
    interpreter binds it.
    """
    source = TWO + "two(b=g(), a=h())"
    assert _call(source) == "two(b=g(), a=h())"


def test_a_call_it_cannot_reorder_still_keeps_the_ones_it_can():
    both = _placed(TWO + DEFAULTS + "two(b=g(), a=h())\ndefaults(1, b=5)")
    assert "two(b=g(), a=h())" in both
    assert "defaults(1, 5)" in both


def test_a_gap_is_left_for_the_interpreter():
    # `defaults(1, c=9)` leaves b to its default. There is nothing to write in
    # b's place that means "the default", so the call is left as written.
    assert _call(DEFAULTS + "defaults(1, c=9)") == "defaults(1, c=9)"


def test_a_run_that_reaches_the_end_is_placed():
    assert _call(DEFAULTS + "defaults(1, b=5)") == "defaults(1, 5)"
    assert _call(DEFAULTS + "defaults(a=1, b=5, c=6)") == "defaults(1, 5, 6)"


def test_a_mapping_is_not_known_until_it_runs():
    assert _call(TWO + "two(**pairs)") == "two(**pairs)"


def test_a_name_that_is_not_a_parameter_is_left_to_raise():
    assert _call(TWO + "two(1, z=2)") == "two(1, z=2)"


def test_a_parameter_given_twice_is_left_to_raise():
    assert _call(TWO + "two(1, a=2)") == "two(1, a=2)"


def test_more_arguments_than_parameters_is_left_to_raise():
    assert _call(TWO + "two(1, 2, 3)") == "two(1, 2, 3)"


def test_a_positional_only_parameter_cannot_be_reached_by_name():
    source = "def only(a, /, b):\n    return a + b\nonly(a=1, b=2)"
    assert _call(source) == "only(a=1, b=2)"
    # Naming the one that is not positional-only is fine.
    assert _call("def only(a, /, b):\n    return a\nonly(1, b=2)") == "only(1, 2)"


def test_a_starred_argument_is_left_alone():
    assert _call(TWO + "two(*rest, b=1)") == "two(*rest, b=1)"


def test_a_signature_with_somewhere_else_for_a_name_to_land():
    """`**kwargs`, `*args` and keyword-only parameters all take names.

    A name that is not in the listed parameters may still be legal for these,
    and where it goes is the callee's business, so none of them is placed.
    """
    for signature in ("a, **rest", "a, *rest", "a, *, b"):
        source = f"def f({signature}):\n    return a\nf(1, b=2)"
        assert _call(source) == "f(1, b=2)"


def test_a_decorated_function_is_a_different_function():
    # The call reaches whatever the decorator answered with, whose parameters
    # are its own.
    source = "@wrap\ndef f(a, b):\n    return a\nf(1, b=2)"
    assert _call(source) == "f(1, b=2)"


def test_a_name_bound_more_than_once_may_not_be_this_function():
    source = TWO + "if flag:\n    two = other\ntwo(1, b=2)"
    assert _placed(source).splitlines()[-1] == "two(1, b=2)"


def test_a_method_is_not_a_module_function():
    source = "def two(a, b):\n    return a\nobj.two(1, b=2)"
    assert _call(source) == "obj.two(1, b=2)"


def test_a_call_inside_a_function_is_placed_too():
    """Where it matters: the call in a loop, not the one at module scope."""
    source = TWO + "def bench():\n    return two(1, b=2)\n"
    assert "two(1, 2)" in _placed(source)


def test_a_call_with_no_keywords_is_untouched():
    assert _call(TWO + "two(1, 2)") == "two(1, 2)"


def test_something_that_is_not_a_module_is_handed_back():
    tree = ast.parse("f(a=1)", mode="eval")
    assert place_keywords(tree) is tree
