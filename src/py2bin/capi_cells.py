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


def _rewrite(function: ast.AST) -> None:
    """Give this function a cell for each name a closure below rebinds."""
    wanted = sorted(
        name
        for name in _declared_nonlocal(function)
        if name in _bound_here(function)
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
