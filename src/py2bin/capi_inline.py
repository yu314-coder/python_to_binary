"""Which small functions can be written out at the point they are called.

`add(t, i)` where `add` is `def add(a, b): return a + b` costs a call, a
reference count on each argument and a return, to compute one addition. Worse,
it hides the addition from the analysis that decides what can live in a
register: the result comes back from a call, so the accumulator it lands in has
to be an object, and a loop that could have been machine arithmetic is not.

Writing the body out at the call site removes both costs at once - the call,
and the opacity. The measured effect is not the few nanoseconds of the call; it
is that the surrounding loop becomes narrowable.

This refuses far more than it accepts, because an inliner that is subtly wrong
is worse than no inliner. A function qualifies only when all of this holds.

*It is one expression.* The body is exactly `return <expression>`. Nothing with
statements in it, so there is no control flow to reproduce and nothing to
rename.

*Its body names nothing but its own parameters.* This is the rule that does the
most work. A body mentioning a global would be substituted into a scope where
that name may mean something else - `def f(a): return a + K` inlined into a
function with its own local `K` would read the local, silently. Refusing every
non-parameter name makes the substitution independent of where it lands.

*Its arguments are names or literals.* Then substituting them is free of
consequence: an argument used twice is evaluated twice and it does not matter,
an argument the body never mentions is not evaluated and it does not matter,
and the order they appear in the body may differ from the order they were
written in and that does not matter either. Every one of those is a real
difference for an argument that can have an effect, which is why anything else
is left as a call.

*The signature is plain.* No defaults, no `*args`, no `**kwargs`, no
keyword-only parameters, no decorators - and the call passes exactly the
declared number of positional arguments, with no keywords and no spreading.

Recursion needs no separate check: a recursive function names itself, and its
own name is not one of its parameters.
"""

from __future__ import annotations

import ast
import copy

#: How large an inlinable body may be, counted in AST nodes. A one-expression
#: function can still be a big expression, and writing a big one out at twenty
#: call sites trades a little speed for a lot of C.
_BIGGEST = 40


class Inlinable:
    """A function that can be written out where it is called."""

    __slots__ = ("parameters", "expression", "repeated")

    def __init__(self, parameters, expression, repeated):
        #: Parameter names, in order.
        self.parameters = parameters
        #: The expression its `return` answers with.
        self.expression = expression
        #: True when some parameter is used more than once, which is only
        #: allowed because the arguments are restricted to names and literals.
        self.repeated = repeated


def _names(node: ast.expr) -> set[str]:
    return {
        inner.id for inner in ast.walk(node) if isinstance(inner, ast.Name)
    }


def _plain_signature(node: ast.FunctionDef) -> bool:
    arguments = node.args
    return not (
        node.decorator_list
        or arguments.vararg
        or arguments.kwarg
        or arguments.kwonlyargs
        or arguments.defaults
        or arguments.kw_defaults
    )


def _simple_body(node: ast.FunctionDef) -> ast.expr | None:
    """The expression a one-line `return` answers with, if that is the body.

    A docstring in front is allowed and skipped: it is not code, and refusing
    over one would exclude most functions anybody documented.
    """

    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    return body[0].value


def describe(
    node: ast.FunctionDef, bound: dict[str, int] | None = None
) -> Inlinable | None:
    """Answer how to inline this function, or None if it cannot be.

    `bound` counts how many times each name is bound anywhere in the module.
    Without it every non-parameter name is refused, which is the safe answer
    when nothing is known about them. With it, a name that the whole module
    binds *at most once* is allowed: there is then only one thing it can mean,
    so it means the same at the call site as it did in the body. That is what
    lets `return v * SCALE` past a rule written to keep `return a + K` out -
    the danger was never the module constant, it was a second `K` somewhere
    else.
    """

    if not isinstance(node, ast.FunctionDef) or not _plain_signature(node):
        return None
    expression = _simple_body(node)
    if expression is None:
        return None
    if sum(1 for _ in ast.walk(expression)) > _BIGGEST:
        return None
    parameters = [
        argument.arg
        for argument in (*node.args.posonlyargs, *node.args.args)
    ]
    if len(set(parameters)) != len(parameters):
        return None
    # Every other name has to mean the same thing wherever the body lands.
    # A name nothing binds is a builtin and means one thing; a name the module
    # binds once means that one thing; a name bound twice could be either, and
    # substituting it would silently read whichever the call site had.
    outside = _names(expression) - set(parameters)
    if outside and bound is None:
        return None
    for name in outside:
        if bound.get(name, 0) > 1:
            return None
        if name == node.name:
            return None  # recursive: writing it out would never terminate
    for inner in ast.walk(expression):
        if isinstance(
            inner, (ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr)
        ):
            # A walrus binds a name in the *caller's* scope once substituted,
            # which is a name the caller never wrote.
            return None
        if isinstance(
            inner,
            (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        ):
            # These introduce a scope of their own, and a scope built at the
            # call site is not the scope the function was written in.
            return None
    counts = [
        sum(
            1
            for inner in ast.walk(expression)
            if isinstance(inner, ast.Name) and inner.id == name
        )
        for name in parameters
    ]
    return Inlinable(parameters, expression, any(count > 1 for count in counts))


def _in_order(expression: ast.expr, parameters: list[str]) -> bool:
    """True when the body evaluates its parameters exactly as a call would.

    `return a + b` is the shape almost every tiny helper has, and for it the
    substitution is provably order-preserving: Python evaluates the left
    operand, then the right, then applies the operator - which is the order the
    call evaluated its arguments in. So an argument that can have an effect is
    still safe to substitute, and that is what lets `add(t, sq(i))` collapse
    once `sq(i)` has become `i * i`.

    The demands are exact. Every leaf is a parameter or a literal; the
    parameters appear left to right in the order they were declared, each
    once; and the only operators are the arithmetic ones, which evaluate both
    operands before doing anything. Anything else - a subscript, an attribute,
    a call, a conditional, a boolean operator that may not evaluate its right
    side at all - is refused, and the caller falls back to demanding arguments
    that cannot have an effect in the first place.
    """

    seen: list[str] = []

    def walk(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            seen.append(node.id)
            return isinstance(node.ctx, ast.Load)
        if isinstance(node, ast.UnaryOp):
            return walk(node.operand)
        if isinstance(node, ast.BinOp):
            return walk(node.left) and walk(node.right)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            return walk(node.left) and walk(node.comparators[0])
        return False

    return walk(expression) and seen == parameters


def transparent(node: ast.expr) -> bool:
    """True for an argument that can be substituted without consequence.

    A name or a literal: evaluating it twice, or not at all, or in a different
    order from how it was written, are all the same as evaluating it once where
    it stands. Nothing else qualifies, however harmless it looks - a call, a
    subscript and an attribute can all run code.
    """

    return isinstance(node, (ast.Name, ast.Constant))


class _Substitute(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, ast.expr]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self.mapping.get(node.id)
        if replacement is None or not isinstance(node.ctx, ast.Load):
            return node
        # Copied, because the same argument may land in more than one place and
        # the emitter attaches position to the nodes it walks.
        return ast.copy_location(copy.deepcopy(replacement), node)


def expand(inlinable: Inlinable, arguments: list[ast.expr]) -> ast.expr | None:
    """The body with the arguments put in, or None if this call cannot use it."""

    if len(arguments) != len(inlinable.parameters):
        return None
    if not all(transparent(argument) for argument in arguments):
        # An argument that can have an effect may still be substituted when the
        # body reads its parameters in the order the call evaluated them, once
        # each - then nothing moves.
        if not _in_order(inlinable.expression, inlinable.parameters):
            return None
    mapping = dict(zip(inlinable.parameters, arguments))
    expanded = _Substitute(mapping).visit(copy.deepcopy(inlinable.expression))
    ast.fix_missing_locations(expanded)
    return expanded


def _bindings(tree: ast.AST) -> dict[str, int]:
    """How many times each name is bound anywhere in this tree.

    A candidate bound more than once - by its own `def` and by anything else -
    is refused outright. That is what makes the substitution safe without
    tracking scopes: a nested function with its own local `add`, a parameter
    called `add`, an `import add`, or a later `add = something` all mean the
    name at a call site may not be the function this pass is looking at.
    """

    counts: dict[str, int] = {}

    def note(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            note(node.name)
            for argument in (
                *node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
            ) if not isinstance(node, ast.ClassDef) else ():
                note(argument.arg)
            if not isinstance(node, ast.ClassDef):
                if node.args.vararg:
                    note(node.args.vararg.arg)
                if node.args.kwarg:
                    note(node.args.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            for argument in (
                *node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
            ):
                note(argument.arg)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            note(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                note((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            note(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                note(name)
    return counts


class _Expander(ast.NodeTransformer):
    def __init__(self, candidates: dict[str, Inlinable]):
        self.candidates = candidates

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.keywords:
            return node
        if any(isinstance(item, ast.Starred) for item in node.args):
            return node
        described = self.candidates.get(node.func.id)
        if described is None:
            return node
        expanded = expand(described, node.args)
        return node if expanded is None else expanded


def expand_module(tree: ast.AST) -> ast.AST:
    """Write every qualifying module-level function out where it is called.

    Done to the tree, before anything looks at it, because the point is not the
    call saved. The register analysis reads assignments to decide what can live
    in one, and `t = add(t, i)` tells it nothing - the value comes back from a
    call. `t = t + i` tells it everything. Inlining after that analysis would
    save the call and leave the loop boxed, which is most of the cost.

    The `def` itself is left in place: the name may still be passed as a value,
    and a function that is inlined at every call site is still a function.
    """

    if not isinstance(tree, ast.Module):
        return tree
    bound = _bindings(tree)
    candidates: dict[str, Inlinable] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if bound.get(node.name, 0) != 1:
            continue  # something else binds this name; the call may not be this
        described = describe(node, bound)
        if described is not None:
            candidates[node.name] = described
    if not candidates:
        return tree
    # A candidate whose body calls another candidate is itself worth writing
    # out, but only once the call inside it has been. Expanding the bodies
    # against each other first is what makes `bump` - which calls `weigh` -
    # inlinable at all. It runs until nothing more changes, and cannot run
    # away: a body only ever grows by the bodies of *other* candidates, and a
    # candidate that names itself was refused above.
    for _ in range(len(candidates)):
        settled = True
        for name, described in list(candidates.items()):
            grown = _Expander(
                {k: v for k, v in candidates.items() if k != name}
            ).visit(copy.deepcopy(described.expression))
            if ast.dump(grown) == ast.dump(described.expression):
                continue
            ast.fix_missing_locations(grown)
            if sum(1 for _ in ast.walk(grown)) > _BIGGEST:
                continue
            candidates[name] = Inlinable(
                described.parameters, grown, described.repeated
            )
            settled = False
        if settled:
            break
    expanded = _Expander(candidates).visit(tree)
    ast.fix_missing_locations(expanded)
    return expanded
