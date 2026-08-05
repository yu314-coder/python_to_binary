"""Which locals are certainly an exact `list` or an exact `dict`.

`xs.append(v)` costs a type lookup, a bound method and two dispatch layers to
reach a function that adds one pointer to an array. An earlier attempt to skip
that behind a *run-time* exact-type guard was measured seventeen per cent
slower - the guard cost more than the machinery it bypassed. This is the
static version, which costs nothing at run time: a name bound only ever by
list displays is exactly a `list`, because a display makes one and nothing
else ever lands in the name. `PyList_Append` is then simply correct - no
guard, no lookup, no bound method.

The reasoning is the one the float analysis uses, with an easier invariant.
A float has to be tracked through arithmetic; a container's type only changes
when the *name is rebound*, so the whole question is what the bindings are.
Mutation is irrelevant - `xs.append`, `xs += [1]` through `list.__iadd__`,
passing `xs` to a function that fills it - none of that changes what type the
name holds.

What disqualifies a name:

*Any binding that is not the right display.* `xs = []` then `xs = get()`
means `get` decides, and this cannot see `get`. A comprehension counts as a
display: a list comprehension makes an exact `list`, however it is compiled.

*Being bound some other way.* An import, a `for` target, an except name, a
`del`, `global`, `nonlocal` - the same exclusions the integer analysis makes,
for the same reasons, and reused from it.

*Being read by a nested scope.* A closure only *reading* the list could not
change its type - but the exclusion is inherited from the shared eligibility
walk, and the cost of keeping it is nil against the risk of relaxing it.

Subclasses are what the exactness is for. `xs = MyList()` is a call, not a
display, so it is excluded and keeps its overridden `append`; a display can
never make a subclass.
"""

from __future__ import annotations

import ast

from .capi_ints import _Uses


def _list_display(node: ast.expr) -> bool:
    """True for an expression that certainly makes an exact `list`."""

    return isinstance(node, (ast.List, ast.ListComp))


def _dict_display(node: ast.expr) -> bool:
    """True for an expression that certainly makes an exact `dict`."""

    return isinstance(node, (ast.Dict, ast.DictComp))


class _Bindings(ast.NodeVisitor):
    """Every expression each name is bound to, by plain assignment.

    Augmented assignment is *not* recorded as a binding here, and that is
    load-bearing: `xs += ys` on a list is `list.__iadd__`, which mutates and
    answers the same object, so the name keeps its exact type. Recording it
    would need the analysis to reason about the right-hand side, and refusing
    it outright would exclude a common idiom for nothing. The name is instead
    disqualified only if `_Uses` saw it bound some other way - and an
    augmented assignment to a name the integer analysis can narrow is exactly
    such a way, which keeps `n += 1` from making `n` look like a container.
    """

    def __init__(self) -> None:
        self.bound: dict[str, list[ast.expr]] = {}
        self.augmented: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.bound.setdefault(target.id, []).append(node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and isinstance(node.target, ast.Name):
            self.bound.setdefault(node.target.id, []).append(node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.augmented.add(node.target.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # A nested scope's assignments are its own; without stopping here a
        # closure's `xs = compute()` would look like this scope's binding.
        return

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef


def _certainly(
    body: list[ast.stmt], parameters: set[str], display
) -> set[str]:
    uses = _Uses()
    bindings = _Bindings()
    for statement in body:
        uses.visit(statement)
        bindings.visit(statement)
    eligible = (
        set(bindings.bound)
        - uses.otherwise_bound
        - uses.escaped
        - parameters
    )
    return {
        name
        for name in eligible
        if all(display(value) for value in bindings.bound[name])
        # An augmented assignment to a container mutates in place and keeps
        # the type; to anything else it rebinds to an unknown. The distinction
        # needs the *current* type, which is what is being decided - so a name
        # that is also augmented is only kept when every plain binding is a
        # display, and the augment is then `__iadd__` on that display's type.
        # For a dict there is no `+=`, so an augmented dict name is refused.
        and (name not in bindings.augmented or display is _list_display)
    }


def _str_display(node: ast.expr) -> bool:
    """True for an expression that certainly makes an exact `str`.

    A literal is one, and an f-string is one however it is spelled: both are
    built by the interpreter itself and neither can answer a subclass. `str(x)`
    is deliberately *not* here - it answers whatever `__str__` returned, which
    a subclass may be.
    """

    if isinstance(node, ast.JoinedStr):
        return True
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def exact_strs(body: list[ast.stmt], parameters: set[str]) -> set[str]:
    """The names in `body` that always hold an exact `str`.

    What this buys is `+`. Concatenation has to go through `PyNumber_Add`,
    because a `str` subclass may override `__add__` and Python would call it -
    and that dispatch is most of what the operation costs. Where both sides
    are certainly exact, there is no `__add__` to find and `PyUnicode_Concat`
    is simply what `+` means.
    """

    return _certainly(body, parameters, _str_display)


def exact_lists(body: list[ast.stmt], parameters: set[str]) -> set[str]:
    """The names in `body` that always hold an exact `list`."""

    return _certainly(body, parameters, _list_display)


def exact_dicts(body: list[ast.stmt], parameters: set[str]) -> set[str]:
    """The names in `body` that always hold an exact `dict`."""

    return _certainly(body, parameters, _dict_display)
