"""A branch whose condition is already known is not a branch.

Lowering folds a constant comparison to a constant - `8 == 0` becomes 0 - but
the *branch* on it used to survive, so the machine code loaded a constant,
compared it and jumped. It comes from generated code rather than from anything
a person writes: `%` and `/` by a literal each guard against a zero and a
minus-one divisor that a literal already answers.

The effect is modest and is stated here so nobody has to re-measure it to find
out: across the seventeen benchmark cases it removes 24 operations from 8,624.
It is kept because it is strictly less work with nothing traded for it, not
because it is large.
"""

from __future__ import annotations

from py2bin.native.ir import (
    Function,
    IntConstant,
    Jump,
    JumpIfFalse,
    Label,
    Module,
    Write,
)
from py2bin.native.optimizer import optimize


def _module(operations, functions=()):
    return Module(list(operations), 4, list(functions), b"")


def test_a_false_condition_becomes_the_jump():
    # `if (0) { ... }` always jumps past the body, so the test is the jump.
    out, report = optimize(
        _module([
            JumpIfFalse(IntConstant(0), "skip"),
            Write(b"never", 1),
            Label("skip"),
            Write(b"after", 1),
        ])
    )
    kinds = [type(operation).__name__ for operation in out.operations]
    assert "JumpIfFalse" not in kinds
    assert kinds[0] == "Jump"
    assert report.settled_branches >= 1
    # The unreachable body went with it rather than being left to take space.
    written = b"".join(o.data for o in out.operations if isinstance(o, Write))
    assert b"never" not in written and b"after" in written


def test_a_true_condition_drops_the_test_and_keeps_the_body():
    out, _ = optimize(
        _module([
            JumpIfFalse(IntConstant(1), "skip"),
            Write(b"body", 1),
            Label("skip"),
        ])
    )
    kinds = [type(operation).__name__ for operation in out.operations]
    assert "JumpIfFalse" not in kinds and "Jump" not in kinds
    written = b"".join(o.data for o in out.operations if isinstance(o, Write))
    assert b"body" in written


def test_a_real_condition_is_left_alone():
    """The pass may only take branches whose answer is already written down."""

    class Runtime:
        pass

    condition = Runtime()
    out, report = optimize(
        _module([JumpIfFalse(condition, "skip"), Write(b"x", 1), Label("skip")])
    )
    assert any(isinstance(o, JumpIfFalse) for o in out.operations)
    assert report.settled_branches == 0


def test_a_label_ends_the_unreachable_stretch():
    # Only what follows the jump *before the next label* is unreachable; a
    # label is where something else may arrive.
    out, _ = optimize(
        _module([
            JumpIfFalse(IntConstant(0), "skip"),
            Write(b"dead", 1),
            Label("elsewhere"),
            Write(b"reachable", 1),
            Label("skip"),
        ])
    )
    written = b"".join(o.data for o in out.operations if isinstance(o, Write))
    assert b"dead" not in written
    assert b"reachable" in written


def test_function_bodies_are_optimised_too():
    """Where a compiled program actually spends its time.

    Function bodies used to be carried through untouched. The entry point of a
    `compile-capi` build is a few dozen operations; every loop the program runs
    is inside one of these.
    """

    body = Function(
        name="f",
        parameters=0,
        stack_slots=2,
        operations=[
            JumpIfFalse(IntConstant(0), "skip"),
            Write(b"never", 1),
            Label("skip"),
        ],
    )
    out, report = optimize(_module([Write(b"main", 1)], [body]))
    kept = out.functions[0].operations
    assert not any(isinstance(o, JumpIfFalse) for o in kept)
    assert report.settled_branches >= 1
