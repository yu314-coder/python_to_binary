"""Give `nonlocal` names something a closure can rebind.

A closure here captures by value: what it reads is copied into the tuple it
carries. That is enough for reading an enclosing name and wrong for rebinding
one, which is what `nonlocal` asks for - both scopes have to see the same
object change.

CPython gives them a cell. This gives them a one-element list, and rewrites
every mention of the name into a subscript of it:

    def outer():                     def outer():
        total = 0                        _cell_total = [0]
        def add(n):          ->          def add(n):
            nonlocal total                   _cell_total[0] = _cell_total[0] + n
            total += n                   add(2)
        add(2)                           return _cell_total[0]
        return total

The list is one object, captured by value like anything else, and both scopes
reach through it to the same place. Done before anything is emitted, so the
emitter needs to know nothing about cells: it is already able to read and
write a subscript.
"""

from __future__ import annotations

import ast

PREFIX = "_py2bin_cell_"


class CellError(Exception):
    """A `nonlocal` this rewriting cannot express."""

    def __init__(self, node: ast.AST, message: str):
        super().__init__(message)
        self.node = node
        self.message = message


def _functions(node: ast.AST):
    """Every function directly inside this one, not looking through them."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield child
        elif not isinstance(child, ast.ClassDef):
            yield from _functions(child)


def _in_this_scope(function: ast.AST):
    """Every node in this function's own scope, stopping at nested ones.

    ast.walk cannot be stopped part way - it yields every descendant, and
    skipping a function node still visits its body. A `nonlocal` inside a
    nested function is that function's statement about this one, so counting
    it here says this function's own body declared it, and the rewriting then
    decides there is nothing to do.
    """
    body = function.body if isinstance(function.body, list) else [function.body]
    pending = list(body)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        pending.extend(ast.iter_child_nodes(node))


def _own_declarations(function: ast.AST) -> set[str]:
    """Names this function's own body says belong to an outer scope.

    Only its own body: a `nonlocal` inside a function nested in this one is
    that function's statement about this one, not this one's about its parent.
    """
    declared: set[str] = set()
    for node in _in_this_scope(function):
        if isinstance(node, (ast.Nonlocal, ast.Global)):
            declared.update(node.names)
    return declared


def _bound_here(function: ast.AST) -> set[str]:
    """Names this function binds itself: parameters, assignments, imports.

    A name the body declares `nonlocal` or `global` is not one of them, even
    though it is assigned to - that declaration is what says the assignment
    lands somewhere else.
    """
    bound: set[str] = set()
    arguments = getattr(function, "args", None)
    if arguments is not None:
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            bound.add(argument.arg)
        if arguments.vararg:
            bound.add(arguments.vararg.arg)
        if arguments.kwarg:
            bound.add(arguments.kwarg.arg)

    for node in _in_this_scope(function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
    return bound - _own_declarations(function)


def _declared_nonlocal(function: ast.AST) -> set[str]:
    """The names declared `nonlocal` anywhere below, before another function."""
    found: set[str] = set()
    body = function.body if isinstance(function.body, list) else [function.body]
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Nonlocal):
                found.update(node.names)
    return found


class _ToCell(ast.NodeTransformer):
    """Rewrite one name into a subscript of its cell, wherever it is used."""

    def __init__(self, name: str):
        self.name = name
        self.cell = PREFIX + name

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id != self.name:
            return node
        return ast.copy_location(
            ast.Subscript(
                value=ast.Name(id=self.cell, ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=node.ctx,
            ),
            node,
        )

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST | None:
        remaining = [name for name in node.names if name != self.name]
        if not remaining:
            # Nothing left to declare; the cell says what this said.
            return ast.copy_location(ast.Pass(), node)
        node.names = remaining
        return node

    def visit_Global(self, node: ast.Global) -> ast.AST:
        if self.name in node.names:
            raise CellError(
                node, f"`{self.name}` is declared both global and nonlocal"
            )
        return node


def _nested(function: ast.AST):
    """Every function defined inside this one, at any depth."""
    for node in _in_this_scope(function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield node


def _read_by(closure: ast.AST) -> set[str]:
    """Every name this closure reads, including from further inside it."""
    return {
        node.id
        for node in ast.walk(closure)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _still_moving(function: ast.AST) -> set[str]:
    """Locals a closure captures whose value has not settled yet.

    Python closes over the variable, so a closure made before the name is
    assigned again sees the later value, and one made inside a loop sees the
    loop's last. Capturing by value gives the value at the moment the closure
    was made, which is a different answer. A cell restores the variable, so
    the two agree again - and these are exactly the cases that were refused
    rather than quietly disagreeing.
    """
    local = _bound_here(function)
    # A cell is already the answer to this question. Promoting one again gives
    # a cell holding a cell, and the inner one starts as the `None` the outer
    # was made with - so the first write said `NoneType does not support item
    # assignment`, from a name the program never wrote.
    local = {name for name in local if not name.startswith(PREFIX)}
    moving: set[str] = set()

    bindings = [
        (node, getattr(node, "lineno", 0))
        for node in _in_this_scope(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    ]
    loops = [
        node
        for node in _in_this_scope(function)
        if isinstance(node, (ast.For, ast.While))
    ]

    for closure in _nested(function):
        line = getattr(closure, "lineno", 0)
        captured = _read_by(closure) & local
        if not captured:
            continue
        for node, bound_at in bindings:
            if bound_at > line and node.id in captured:
                moving.add(node.id)
        for loop in loops:
            start = getattr(loop, "lineno", 0)
            end = getattr(loop, "end_lineno", start) or start
            if not start <= line <= end:
                continue
            inside = list(loop.body)
            if isinstance(loop, ast.For):
                inside.append(loop.target)
            for statement in inside:
                for node in ast.walk(statement):
                    if (
                        isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Store)
                        and node.id in captured
                    ):
                        moving.add(node.id)
    return moving


def _rewrite(function: ast.AST) -> None:
    """Give this function a cell for each name a closure below rebinds."""
    wanted = sorted(
        {
            name
            for name in _declared_nonlocal(function)
            if name in _bound_here(function)
        }
        | _still_moving(function)
    )
    if not wanted:
        return

    arguments = getattr(function, "args", None)
    parameters = set()
    if arguments is not None:
        parameters = {
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        }

    for name in wanted:
        transformer = _ToCell(name)
        function.body = [transformer.visit(node) for node in function.body]
        # A parameter already holds a value, so the cell starts from it; any
        # other name starts empty, exactly as an unassigned local does.
        first = (
            ast.Name(id=name, ctx=ast.Load())
            if name in parameters
            else ast.Constant(value=None)
        )
        function.body.insert(
            0,
            ast.Assign(
                targets=[ast.Name(id=PREFIX + name, ctx=ast.Store())],
                value=ast.List(elts=[first], ctx=ast.Load()),
            ),
        )
    ast.fix_missing_locations(function)


def expand(tree: ast.AST) -> ast.AST:
    """Rewrite every `nonlocal` in this tree into a cell its closures share.

    Innermost first, so a `nonlocal` inside a function that is itself the
    target of one is rewritten before the outer pass looks at it.
    """
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for function in reversed(functions):
        _rewrite(function)
    return tree


#: The comprehension forms that bind a target of their own.
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


class _Named(ast.NodeTransformer):
    """One name into a subscript of a cell, under a spelling of our choosing.

    `_ToCell` names the cell after the variable, which is right for a
    `nonlocal`: there is one such name in the scope and the cell replaces it.
    A comprehension's target is not that - a variable of the same spelling may
    sit outside it and must be left alone - so the cell gets a name of its own
    and nothing outside the comprehension is touched.
    """

    def __init__(self, name: str, cell: str) -> None:
        self.name = name
        self.cell = cell

    def _binds_it(self, node) -> bool:
        """Whether this closure has a name of its own with that spelling."""

        arguments = getattr(node, "args", None)
        if arguments is not None and any(
            argument.arg == self.name
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *([arguments.vararg] if arguments.vararg else []),
                *([arguments.kwarg] if arguments.kwarg else []),
            )
        ):
            return True
        body = node.body if isinstance(node.body, list) else [node.body]
        for statement in body:
            for inner in ast.walk(statement):
                if (
                    isinstance(inner, ast.Name)
                    and isinstance(inner.ctx, ast.Store)
                    and inner.id == self.name
                ):
                    return True
        return False

    def _closure(self, node):
        # A closure with a name of its own of that spelling is not talking
        # about the comprehension's - `lambda i=i: i` says the by-value thing
        # deliberately, and its body means the parameter. The *defaults* are
        # evaluated out here, though, and do mean ours.
        if not self._binds_it(node):
            self.generic_visit(node)
            return node
        arguments = getattr(node, "args", None)
        if arguments is not None:
            arguments.defaults = [
                self.visit(default) for default in arguments.defaults
            ]
            arguments.kw_defaults = [
                None if default is None else self.visit(default)
                for default in arguments.kw_defaults
            ]
        return node

    visit_Lambda = _closure
    visit_FunctionDef = _closure
    visit_AsyncFunctionDef = _closure

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id != self.name:
            return node
        return ast.copy_location(
            ast.Subscript(
                value=ast.Name(id=self.cell, ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=node.ctx,
            ),
            node,
        )


def _targets_of(node: ast.expr) -> set[str]:
    """Every name a comprehension's `for` clauses bind."""

    return {
        inner.id
        for clause in node.generators
        for inner in ast.walk(clause.target)
        if isinstance(inner, ast.Name)
    }


def _closure_reads(node: ast.expr, wanted: set[str]) -> set[str]:
    """Which of `wanted` a closure written inside this comprehension reads."""

    found: set[str] = set()
    for inner in ast.walk(node):
        if not isinstance(
            inner, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        found |= {
            name.id
            for name in ast.walk(inner)
            if isinstance(name, ast.Name)
            and isinstance(name.ctx, ast.Load)
            and name.id in wanted
        }
    return found


class _CompCells(ast.NodeTransformer):
    """Give a comprehension's target a cell when a closure inside captures it.

    Python closes over the variable, so every closure made in a comprehension
    sees the last value it took: `[lambda: i for i in range(3)]` is
    `[2, 2, 2]`. Capturing by value would answer `[0, 1, 2]`, so this used to
    be refused. A cell settles it, exactly as it settles the same shape
    written as a `for` statement.

    The cell is named after the comprehension rather than the variable, and
    every mention *inside* the comprehension is rewritten to it. So a
    variable of the same spelling outside is not touched at all - which is
    what 3.12 onwards does by saving and restoring it around an inlined
    comprehension, arrived at here by never involving it.

    The first iterable is the exception: it is evaluated in the scope around
    the comprehension, before the target exists, so a mention of the name
    there is the *outer* one and is left alone.
    """

    def __init__(self, counter: list[int]) -> None:
        self.made: list[ast.stmt] = []
        # Shared across the whole module: two comprehensions in two different
        # statements each got cell number one, so the second overwrote the
        # first and its closures answered with the other one's last value.
        self.counter = counter

    def generic_visit(self, node):
        # Not into a statement list. Those are walked by the caller, which
        # puts each cell in front of the statement that needs it - and a
        # second pass over the same comprehension would give the cell a cell.
        for field, old in ast.iter_fields(node):
            if isinstance(old, list):
                if old and isinstance(old[0], ast.stmt):
                    continue
                rebuilt = []
                for item in old:
                    if isinstance(item, ast.AST):
                        item = self.visit(item)
                        if item is None:
                            continue
                    # A `None` that was never a node stays: `{**more}` is a
                    # dict whose *key* is None, and dropping it slid every
                    # key along one and paired them with the wrong values.
                    rebuilt.append(item)
                setattr(node, field, rebuilt)
            elif isinstance(old, ast.AST):
                setattr(node, field, self.visit(old))
        return node

    def visit_Lambda(self, node):
        # A comprehension inside a lambda body belongs to the lambda's own
        # scope; the statement this pass is rewriting cannot hold its cell.
        return node

    def visit_FunctionDef(self, node):
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def _rewrite(self, node):
        # Innermost first: a comprehension inside another is settled before
        # the outer one looks at it.
        self.generic_visit(node)
        caught = sorted(_closure_reads(node, _targets_of(node)))
        if not caught:
            return node
        self.counter[0] += 1
        outermost = node.generators[0].iter
        for name in caught:
            cell = f"{PREFIX}comp{self.counter[0]}_{name}"
            for clause in node.generators:
                clause.target = _Named(name, cell).visit(clause.target)
                if clause.iter is not outermost:
                    clause.iter = _Named(name, cell).visit(clause.iter)
                clause.ifs = [
                    _Named(name, cell).visit(test) for test in clause.ifs
                ]
            if isinstance(node, ast.DictComp):
                node.key = _Named(name, cell).visit(node.key)
                node.value = _Named(name, cell).visit(node.value)
            else:
                node.elt = _Named(name, cell).visit(node.elt)
            self.made.append(
                ast.Assign(
                    targets=[ast.Name(id=cell, ctx=ast.Store())],
                    value=ast.List(
                        elts=[ast.Constant(value=None)], ctx=ast.Load()
                    ),
                )
            )
        return node

    visit_ListComp = _rewrite
    visit_SetComp = _rewrite
    visit_DictComp = _rewrite
    visit_GeneratorExp = _rewrite


def expand_comprehension_cells(
    body: list[ast.stmt], counter: list[int] | None = None
) -> list[ast.stmt]:
    """Settle every comprehension in these statements that a closure captures.

    The cell has to exist before the comprehension runs, so it is made by a
    statement written in front of the one holding it - which puts it inside
    whatever loop that statement sits in, so each turn gets a cell of its own,
    as each turn's comprehension is its own.
    """

    counter = [0] if counter is None else counter
    rebuilt: list[ast.stmt] = []
    for statement in body:
        for field in ("body", "orelse", "finalbody"):
            held = getattr(statement, field, None)
            if isinstance(held, list) and held and isinstance(held[0], ast.stmt):
                setattr(
                    statement, field, expand_comprehension_cells(held, counter)
                )
        for handler in getattr(statement, "handlers", ()):
            handler.body = expand_comprehension_cells(handler.body, counter)
        if isinstance(
            statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            rebuilt.append(statement)
            continue
        maker = _CompCells(counter)
        statement = maker.visit(statement)
        for made in maker.made:
            ast.copy_location(made, statement)
            ast.fix_missing_locations(made)
            rebuilt.append(made)
        rebuilt.append(statement)
    return rebuilt
