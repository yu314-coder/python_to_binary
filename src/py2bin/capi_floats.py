"""Which of a function's locals can be held as a machine double.

The integer story in :mod:`capi_ints` has an exact counterpart in floats, and
until this existed the float half was the worse of the two: an integer loop was
already faster than CPython while the same loop written in floats ran at about
four tenths of its speed. The reason is the same one, and so is the fix. Every
`+` between two floats went to `PyNumber_Add`, which allocated a `PyFloat` for
the result, and the loop spent its time in the allocator rather than the FPU.

What differs from the integer case is where the fast path has to give up.

*Overflow is not a failure.* An integer that leaves the machine word has to
fall back, because Python's integers do not stop at 64 bits. A double that
overflows produces an infinity, which is exactly what CPython produces, so
there is nothing to check and nothing to fall back to.

*Division is.* `x / 0.0` raises `ZeroDivisionError` in Python and yields an
infinity in C, so a division tests its divisor and hands the zero case to the
slow arm, which raises. That is one compare against a saving of an allocation
per division, and division was already the expensive one.

*A float is never an int.* The two representations are kept apart: a name is
held as a double, or as a machine integer, or as an object, and never changes
which. Mixing them would mean tracking at run time which of the two a slot
holds, and the payoff for `x` being an int on one path and a float on another
is not worth a branch on every read.

Narrowing is seeded from float literals and spreads through arithmetic, never
from inspecting an object. `PyFloat_AsDouble` would happily convert an `int`,
or anything with `__float__`, and a value that arrived as an `int` and came
back out as a `float` would be a wrong program - `1` and `1.0` are different
objects, print differently and hash the same but are not interchangeable in
`repr`, in JSON, or in a dictionary meant to hold one of them.
"""

from __future__ import annotations

import ast

from .capi_ints import _Uses

#: Operators whose result on two machine doubles is what Python would have
#: produced. Floor division is missing: Python's rounds towards minus infinity
#: and returns a float, where C has no operator for it at all. The modulo is
#: missing for the same reason - `fmod` truncates towards zero and Python does
#: not.
ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div)

#: Comparisons that are a single machine instruction on two doubles. Ordered
#: comparisons involving a NaN answer false in both languages, so there is
#: nothing to special-case.
COMPARISONS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


def is_machine_float(node: ast.expr) -> bool:
    """True for a literal the fast path can spell as a C constant.

    `bool` is excluded for the reason it is excluded from the integer test:
    it is a subclass of `int`, not of `float`, but a careless `isinstance`
    would still be wrong somewhere, and being explicit costs nothing.
    """

    return isinstance(node, ast.Constant) and type(node.value) is float


def _integer_literal(node: ast.expr) -> bool:
    """True for an `int` literal, which mixes into float arithmetic cleanly.

    `x * 2` where `x` is a float is a float, and the `2` needs no conversion:
    C promotes it. Only literals qualify - a *name* holding an int would make
    the result an int when the other side turned out not to be a float.
    """

    return isinstance(node, ast.Constant) and type(node.value) is int


class _FloatUses(_Uses):
    """The integer analysis's bookkeeping, with float arithmetic counted too.

    `/` is not in the integer operator set - true division of two ints is a
    float, so it is not an integer operation - which left a name used only in
    divisions looking as though it were never calculated with at all.
    """

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ARITHMETIC):
            for side in (node.left, node.right):
                if isinstance(side, ast.Name):
                    self.calculated.add(side.id)
        super().visit_BinOp(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) == 1 and isinstance(node.ops[0], COMPARISONS):
            for side in (node.left, node.comparators[0]):
                if isinstance(side, ast.Name):
                    self.calculated.add(side.id)
        super().visit_Compare(node)


class _Sources(ast.NodeVisitor):
    """Every expression each name is bound to, so narrowing can be decided."""

    def __init__(self) -> None:
        self.bindings: dict[str, list[ast.expr]] = {}

    def _record(self, target: ast.expr, value: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.bindings.setdefault(target.id, []).append(value)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._record(node.target, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # `x += e` binds x to `x + e`, and the synthesised node is what decides
        # whether the result stays a float. Built rather than reused so that
        # the operator is the binary one, not the augmented spelling.
        if isinstance(node.target, ast.Name) and isinstance(node.op, ARITHMETIC):
            combined = ast.BinOp(
                left=ast.Name(id=node.target.id, ctx=ast.Load()),
                op=node.op,
                right=node.value,
            )
            self._record(node.target, combined)
        self.generic_visit(node)


def _narrow(node: ast.expr, doubles: set[str]) -> bool:
    """True when this expression is certainly a Python float."""

    if is_machine_float(node):
        return True
    if isinstance(node, ast.Name):
        return node.id in doubles
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub)
    ):
        return _narrow(node.operand, doubles)
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ARITHMETIC):
        return False
    left, right = node.left, node.right
    if isinstance(node.op, ast.Div):
        # True division answers a float whenever either side is one, and both
        # sides being ints answers a float too - but an int/int fast path would
        # have to be an integer one, so it is left to the slow arm.
        return (_narrow(left, doubles) and _floatish(right, doubles)) or (
            _narrow(right, doubles) and _floatish(left, doubles)
        )
    # For the rest, one float makes the result a float, provided the other side
    # is something C will promote rather than an object of unknown type.
    return (_narrow(left, doubles) and _floatish(right, doubles)) or (
        _narrow(right, doubles) and _floatish(left, doubles)
    )


def _floatish(node: ast.expr, doubles: set[str]) -> bool:
    """True when C can compute with this operand directly, float or int."""

    return _integer_literal(node) or _narrow(node, doubles)


def unboxable_locals(body: list[ast.stmt], parameters: set[str]) -> set[str]:
    """The names in `body` worth holding as a machine double.

    Eligibility - bound only in ways the emitter can narrow, reachable by
    nothing else, and actually used in arithmetic - is the integer analysis's
    question and is answered by reusing it. What is decided here is the part
    that differs: which of those eligible names certainly hold a float.

    The answer is a *greatest* fixed point: every eligible name is assumed to
    be a float, and any name with a binding that is then not a float is struck
    out, until a pass strikes out nothing.

    Growing from nothing instead would answer the accumulator wrongly, and the
    accumulator is the whole point. `t = 0.0` followed by `t = t + 1.5` binds
    `t` to an expression mentioning `t`, so `t` is a float only if `t` is a
    float, and a set that only ever admits a name on independent evidence
    never admits it at all.

    Assuming and shrinking is sound because the property is closed under
    execution. A name survives only if every expression it is bound to is a
    float given that the survivors are floats - so if the first value assigned
    is a float, induction over the assignments says every later one is too.
    A name whose value comes from somewhere this cannot see has no binding
    recorded and is struck out on the first pass, which is what removes a
    `for` target: it is handed an object by an iterator, and the analysis has
    nothing to say about what kind.
    """

    uses = _FloatUses()
    sources = _Sources()
    for statement in body:
        uses.visit(statement)
        sources.visit(statement)
    doubles = (
        (uses.calculated & uses.narrowly_bound)
        - uses.otherwise_bound
        - uses.escaped
        - parameters
    )

    while True:
        struck = {
            name
            for name in doubles
            if not sources.bindings.get(name)
            or not all(
                _narrow(value, doubles) for value in sources.bindings[name]
            )
        }
        if not struck:
            return doubles
        doubles -= struck
