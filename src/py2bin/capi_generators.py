"""Rewrite a generator function into a class whose `__next__` is a state machine.

A generator has to stop in the middle of itself and carry on later from the
same place, which a compiled C function cannot do: it has one entry and its
locals live on a stack frame that ends when it returns. CPython manages it by
keeping a frame object alive across the suspension. Nothing in the vetted C-API
table hands out a frame, so this tier cannot borrow that trick.

What it can do is the trick every language without coroutines uses: turn the
function inside out. The body is cut into blocks at each `yield`, the blocks are
numbered, and the whole thing becomes a loop that dispatches on which block to
run next. Locals stop being locals and become attributes, because they have to
outlive a `return`. So::

    def counter(limit):
        n = 0
        while n < limit:
            yield n
            n = n + 1

becomes, before anything is compiled::

    class _py2bin_gen_counter:
        def __init__(self, limit):
            self._py2bin_state = 0
            self.limit = limit
            self.n = None
        def __iter__(self):
            return self
        def __next__(self):
            while True:
                if self._py2bin_state == 0:
                    self.n = 0
                    self._py2bin_state = 1
                    continue
                ...

    def counter(limit):
        return _py2bin_gen_counter(limit)

and the class is compiled by the machinery that already compiles classes. There
is no new C, no new entry point, and no interpreter reached for at run time -
the generator is machine code like everything else, and the state it resumes
from is a number in an attribute.

What is accepted is stated rather than discovered: `yield` as a statement, in
straight-line code, `if`/`else`, `while` and `for`. A `yield` whose value is
used (`x = yield`) needs `send`, and a `yield` inside `try` or `with` needs the
handler to survive the suspension; both are refused by name rather than
mistranslated.
"""

from __future__ import annotations

import ast

#: The attribute holding which block to run next. Prefixed because it shares a
#: namespace with the function's own locals, which become attributes too.
STATE = "_py2bin_state"
#: Where a `for` loop keeps the iterator it is walking, one per loop.
ITERATOR = "_py2bin_iter"
#: What `send` last put in, which a resumed `x = yield` reads.
SENT = "_py2bin_sent"


class GeneratorRewriteError(Exception):
    """A generator shape this transformation does not express."""

    def __init__(self, node: ast.AST, message: str) -> None:
        super().__init__(message)
        self.node = node
        self.message = message


def is_generator(node: ast.FunctionDef) -> bool:
    """True when this function's own body yields.

    A `yield` inside a nested function belongs to that function, so the walk
    stops at anything that introduces a scope of its own.
    """

    return any(_yields(statement) for statement in node.body)


#: PEP 380's expansion of `yield from`, less the `throw` and `close`
#: passthrough - which would need `throw` and `close` on the generator this
#: compiles to, and it has neither. What is here is the half that matters for
#: ordinary use and for `await`: values sent in reach the sub-iterator, and the
#: sub-iterator's return value is the value of the expression.
_DELEGATION = """
_py2bin_i{n} = iter(_py2bin_src{n})
_py2bin_r{n} = None
_py2bin_go{n} = 1
try:
    _py2bin_y{n} = next(_py2bin_i{n})
except StopIteration as _py2bin_e{n}:
    _py2bin_r{n} = _py2bin_e{n}.value
    _py2bin_go{n} = 0
while _py2bin_go{n}:
    _py2bin_s{n} = yield _py2bin_y{n}
    try:
        if _py2bin_s{n} is None:
            _py2bin_y{n} = next(_py2bin_i{n})
        else:
            _py2bin_y{n} = _py2bin_i{n}.send(_py2bin_s{n})
    except StopIteration as _py2bin_e{n}:
        _py2bin_r{n} = _py2bin_e{n}.value
        _py2bin_go{n} = 0
"""


class _Hoister(ast.NodeTransformer):
    """Lift `await` and `yield from` out of the expressions they sit in.

    `total = total + await f(x)` has to become two statements, because the
    expansion of an `await` is a loop and a loop is not an expression. The
    lifted value goes into a temporary that the statement then reads.

    Where lifting would change *when* something runs, it is refused instead:
    the second operand of `and`/`or` and the arms of `a if c else b` are
    conditional, and a comprehension's body runs once per item. Hoisting any
    of those would run the await unconditionally, or once instead of many
    times, and quietly give a different program.
    """

    def __init__(self) -> None:
        self.count = 0
        self.lifted: list[ast.stmt] = []

    def _lift(self, node: ast.expr) -> ast.expr:
        self.count += 1
        name = f"_py2bin_await{self.count}"
        self.lifted.append(
            ast.copy_location(
                ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())], value=node
                ),
                node,
            )
        )
        return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)

    def visit_Await(self, node: ast.Await) -> ast.expr:
        node.value = self.visit(node.value)
        return self._lift(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> ast.expr:
        node.value = self.visit(node.value)
        return self._lift(node)

    def _refuse_inside(self, node: ast.AST, where: str) -> None:
        for inner in ast.walk(node):
            if isinstance(inner, (ast.Await, ast.YieldFrom)):
                raise GeneratorRewriteError(
                    inner, f"an `await` inside {where}"
                )

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        for value in node.values[1:]:
            self._refuse_inside(value, "`and` or `or`, which may not run it")
        node.values[0] = self.visit(node.values[0])
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        for arm in (node.body, node.orelse):
            self._refuse_inside(arm, "an `a if c else b` arm")
        node.test = self.visit(node.test)
        return node

    def visit_ListComp(self, node):
        self._refuse_inside(node, "a comprehension")
        return node

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_FunctionDef(self, node):
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


#: The bodies a compound statement holds, which are hoisted in their own right.
_NESTED = ("body", "orelse", "finalbody")


def _hoist(body: list[ast.stmt]) -> list[ast.stmt]:
    """Lift awaits out of each statement, and out of the bodies it holds.

    A `while` test is left alone and refused if it awaits: the test runs again
    on every turn of the loop, and a value hoisted in front of the loop would
    be computed once. That is a different program, so it is not written.
    """

    out: list[ast.stmt] = []
    for statement in body:
        if isinstance(statement, ast.While):
            hoister = _Hoister()
            hoister._refuse_inside(statement.test, "a `while` condition")
        for field in _NESTED:
            held = getattr(statement, field, None)
            if isinstance(held, list) and held and isinstance(held[0], ast.stmt):
                setattr(statement, field, _hoist(held))
        if isinstance(statement, ast.Try):
            for handler in statement.handlers:
                handler.body = _hoist(handler.body)
        hoister = _Hoister()
        if isinstance(
            statement,
            (ast.Expr, ast.Assign, ast.AugAssign, ast.Return, ast.If, ast.For),
        ):
            # The statement's own expression - an `if` test, a `for` iterable,
            # the value of an assignment. The bodies were done above.
            rebuilt = _hoist_head(statement, hoister)
        else:
            rebuilt = statement
        out.extend(hoister.lifted)
        out.append(rebuilt)
    return out


def _hoist_head(statement: ast.stmt, hoister: "_Hoister") -> ast.stmt:
    """Hoist out of a statement's own expression, leaving its bodies alone."""

    if isinstance(statement, ast.If):
        statement.test = hoister.visit(statement.test)
    elif isinstance(statement, ast.For):
        statement.iter = hoister.visit(statement.iter)
    elif isinstance(statement, ast.Return):
        if statement.value is not None:
            statement.value = hoister.visit(statement.value)
    elif isinstance(statement, ast.AugAssign):
        statement.value = hoister.visit(statement.value)
    else:
        statement.value = hoister.visit(statement.value)
    return statement


class _DelegationRewriter(ast.NodeTransformer):
    """Write `yield from` out as the loop the language defines it to be.

    PEP 380 gives `yield from` a formal expansion in terms of `yield`, `next`
    and `send`, and that expansion is ordinary Python - so it is written out
    here and the state machine never has to know delegation exists. What is
    left out is the `throw` and `close` passthrough, which would need a
    `throw` and a `close` on the thing this compiles to.

    `await x` is the same expansion over `x.__await__()`, which is what the
    language says an `await` of an object with `__await__` means.
    """

    def __init__(self) -> None:
        self.count = 0

    def _expand(self, source: ast.expr, target: ast.expr | None) -> list[ast.stmt]:
        self.count += 1
        name = self.count
        body = ast.parse(_DELEGATION.format(n=name)).body
        # The first statement is `_i = iter(_src)`; give it the real source.
        body[0].value.args[0] = source
        if target is not None:
            body.append(
                ast.Assign(
                    targets=[target],
                    value=ast.Name(id=f"_py2bin_r{name}", ctx=ast.Load()),
                )
            )
        return body

    def visit_Expr(self, node: ast.Expr) -> ast.stmt | list[ast.stmt]:
        inner = node.value
        if isinstance(inner, ast.YieldFrom):
            return self._expand(inner.value, None)
        if isinstance(inner, ast.Await):
            return self._expand(_awaited(inner.value), None)
        self.generic_visit(node)
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.stmt | list[ast.stmt]:
        if isinstance(node.value, (ast.YieldFrom, ast.Await)) and len(
            node.targets
        ) == 1:
            source = (
                node.value.value
                if isinstance(node.value, ast.YieldFrom)
                else _awaited(node.value.value)
            )
            return self._expand(source, node.targets[0])
        self.generic_visit(node)
        return node

    def visit_YieldFrom(self, node: ast.YieldFrom) -> ast.AST:
        raise GeneratorRewriteError(
            node, "a `yield from` in an expression this does not expand"
        )

    def visit_Await(self, node: ast.Await) -> ast.AST:
        raise GeneratorRewriteError(
            node, "an `await` in an expression this does not expand"
        )

    def visit_FunctionDef(self, node):
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


def _awaited(value: ast.expr) -> ast.expr:
    """`x` becomes `x.__await__()`, which is what awaiting it means."""

    return ast.Call(
        func=ast.Attribute(value=value, attr="__await__", ctx=ast.Load()),
        args=[],
        keywords=[],
    )


def _yields(node: ast.AST) -> bool:
    if isinstance(node, (ast.Yield, ast.YieldFrom)):
        return True
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
        ):
            continue
        if _yields(child):
            return True
    return False


class _ReturnRewriter(ast.NodeTransformer):
    """Turn `return` into the end of the iteration.

    A bare `return` in a generator means "stop", and left as written it becomes
    `return None` out of `__next__` - which the iterator protocol reads as
    *yielding* None, forever. It has to raise StopIteration instead, and that
    is true wherever it appears, including inside a statement that does not
    yield and is otherwise copied across unchanged.
    """

    def __init__(self, machine: "_Machine") -> None:
        self.machine = machine

    def visit_Return(self, node: ast.Return) -> list[ast.stmt]:
        return [
            ast.copy_location(part, node)
            for part in self.machine._stop(node.value)
        ]

    def visit_FunctionDef(self, node):  # a nested function's return is its own
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


class _ExitRewriter(ast.NodeTransformer):
    """Turn `break` and `continue` into jumps to the right block.

    Once a loop has been cut into blocks, the loop no longer exists as a loop -
    what runs is the dispatch loop, and a `break` left as written would leave
    *that*, ending the generator instead of the loop. So each becomes a jump:
    `continue` to the block that tests the condition, `break` to the block
    after it.

    A loop nested inside this one is left alone, because its `break` belongs to
    it. That is why this does not simply walk the whole subtree.
    """

    def __init__(self, machine: "_Machine", test: int, after: int) -> None:
        self.machine = machine
        self.test = test
        self.after = after

    def visit_Break(self, node: ast.Break) -> list[ast.stmt]:
        return [ast.copy_location(part, node) for part in self.machine._goto(self.after)]

    def visit_Continue(self, node: ast.Continue) -> list[ast.stmt]:
        return [ast.copy_location(part, node) for part in self.machine._goto(self.test)]

    def visit_For(self, node):
        return node

    visit_While = visit_For
    visit_FunctionDef = visit_For
    visit_Lambda = visit_For
    visit_ClassDef = visit_For


class _Machine:
    """Cuts one function body into numbered blocks."""

    def __init__(self, function: ast.FunctionDef) -> None:
        self.function = function
        #: block number -> the statements that block runs.
        self.blocks: list[list[ast.stmt]] = []
        #: For each block, the `except` clauses that are live while it runs,
        #: innermost last. An exception can only be raised while a block is
        #: executing, so a handler that is re-established on every entry to
        #: every block of the region is a handler that survives the
        #: suspension - which is the whole difficulty with `try` and `yield`.
        self.guards: list[list[tuple[list[ast.ExceptHandler], list[int]]]] = []
        self.region: list[tuple[list[ast.ExceptHandler], list[int]]] = []
        #: Every name the body binds, which becomes an attribute.
        self.names: set[str] = set()
        self.loops = 0

    # --- helpers that build the small AST pieces ------------------------

    def _self(self, attribute: str, store: bool = False) -> ast.Attribute:
        return ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr=attribute,
            ctx=ast.Store() if store else ast.Load(),
        )

    def _goto(self, block: int) -> list[ast.stmt]:
        """Set the next block and go round the dispatch loop again."""

        return [
            ast.Assign(
                targets=[self._self(STATE, store=True)],
                value=ast.Constant(value=block),
            ),
            ast.Continue(),
        ]

    def _stop(self, value: ast.expr | None = None) -> list[ast.stmt]:
        """End the iteration, carrying a return value if there was one.

        `return v` in a generator is `raise StopIteration(v)`; the value is
        what `yield from` answers with, which is the only way to see it.
        """

        return [
            ast.Assign(
                targets=[self._self(STATE, store=True)],
                value=ast.Constant(value=-1),
            ),
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="StopIteration", ctx=ast.Load()),
                    args=[] if value is None else [value],
                    keywords=[],
                ),
                cause=None,
            ),
        ]

    def _new_block(self) -> int:
        self.blocks.append([])
        self.guards.append(list(self.region))
        return len(self.blocks) - 1

    # --- the cut itself ---------------------------------------------------

    def build(self) -> None:
        # Every `return` first, wherever it sits, so that the cut below never
        # meets one and no copied statement carries one through.
        # One pass at a time, each over a flat list: a rewriter that answers
        # with several statements cannot be fed to the next one directly.
        def pass_over(body: list[ast.stmt], rewriter) -> list[ast.stmt]:
            out: list[ast.stmt] = []
            for statement in body:
                replaced = rewriter.visit(statement)
                out.extend(replaced if isinstance(replaced, list) else [replaced])
            return out

        body = _hoist(self.function.body)
        body = pass_over(body, _DelegationRewriter())
        body = pass_over(body, _ReturnRewriter(self))
        first = self._new_block()
        end = self._emit(body, first)
        self.blocks[end].extend(self._stop())

    def _emit(self, body: list[ast.stmt], block: int) -> int:
        """Emit `body` starting in `block`; answer the block it ends in."""

        for statement in body:
            block = self._statement(statement, block)
        return block

    def _statement(self, statement: ast.stmt, block: int) -> int:
        if isinstance(statement, (ast.AsyncFor, ast.AsyncWith)):
            raise GeneratorRewriteError(
                statement,
                f"an `{type(statement).__name__[5:].lower()}` needs the "
                "asynchronous iteration protocol",
            )
        if not _yields(statement):
            # Nothing suspends inside it, so it stays exactly as written -
            # only its names are rewritten to attributes, which happens later.
            self.blocks[block].append(statement)
            self._collect(statement)
            return block
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Yield):
            resume = self._new_block()
            value = statement.value.value or ast.Constant(value=None)
            self.blocks[block].extend(
                [
                    ast.Assign(
                        targets=[self._self(STATE, store=True)],
                        value=ast.Constant(value=resume),
                    ),
                    ast.Return(value=value),
                ]
            )
            return resume
        if isinstance(statement, ast.Assign) and isinstance(
            statement.value, ast.Yield
        ):
            if len(statement.targets) != 1 or not isinstance(
                statement.targets[0], ast.Name
            ):
                raise GeneratorRewriteError(
                    statement, "a `yield` assigned to anything but one name"
                )
            target = statement.targets[0].id
            self.names.add(target)
            resume = self._new_block()
            value = statement.value.value or ast.Constant(value=None)
            self.blocks[block].extend(
                [
                    ast.Assign(
                        targets=[self._self(STATE, store=True)],
                        value=ast.Constant(value=resume),
                    ),
                    ast.Return(value=value),
                ]
            )
            # What `send` put there, which `__next__` leaves as None.
            self.blocks[resume].append(
                ast.Assign(
                    targets=[self._self(target, store=True)],
                    value=self._self(SENT),
                )
            )
            return resume
        if isinstance(statement, ast.If):
            after = self._new_block()
            then = self._new_block()
            otherwise = self._new_block() if statement.orelse else after
            self.blocks[block].append(
                ast.If(
                    test=statement.test,
                    body=self._goto(then),
                    orelse=self._goto(otherwise),
                )
            )
            self.blocks[self._emit(statement.body, then)].extend(self._goto(after))
            if statement.orelse:
                self.blocks[self._emit(statement.orelse, otherwise)].extend(
                    self._goto(after)
                )
            return after
        if isinstance(statement, ast.While):
            if statement.orelse:
                raise GeneratorRewriteError(
                    statement, "a `while ... else` containing `yield`"
                )
            test = self._new_block()
            body = self._new_block()
            after = self._new_block()
            self.blocks[block].extend(self._goto(test))
            self.blocks[test].append(
                ast.If(
                    test=statement.test,
                    body=self._goto(body),
                    orelse=self._goto(after),
                )
            )
            self.blocks[
                self._emit(self._exits(statement.body, test, after), body)
            ].extend(self._goto(test))
            return after
        if isinstance(statement, ast.For):
            if statement.orelse:
                raise GeneratorRewriteError(
                    statement, "a `for ... else` containing `yield`"
                )
            if not isinstance(statement.target, ast.Name):
                raise GeneratorRewriteError(
                    statement, "a `for` over a tuple target containing `yield`"
                )
            # Desugared to a while over an explicit iterator, because the
            # iterator has to outlive the suspension and a `for` keeps it
            # somewhere this rewriting cannot reach.
            self.loops += 1
            holder = f"{ITERATOR}{self.loops}"
            self.names.add(holder)
            self.names.add(statement.target.id)
            probe = f"{holder}_item"
            self.names.add(probe)
            test = self._new_block()
            body = self._new_block()
            after = self._new_block()
            self.blocks[block].append(
                ast.Assign(
                    targets=[self._self(holder, store=True)],
                    value=ast.Call(
                        func=ast.Name(id="iter", ctx=ast.Load()),
                        args=[statement.iter],
                        keywords=[],
                    ),
                )
            )
            self.blocks[block].extend(self._goto(test))
            # `next(it, _py2bin_absent)` rather than catching StopIteration,
            # so that no handler has to survive the suspension.
            self.blocks[test].append(
                ast.Assign(
                    targets=[self._self(probe, store=True)],
                    value=ast.Call(
                        func=ast.Name(id="next", ctx=ast.Load()),
                        args=[
                            self._self(holder),
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_py2bin_absent",
                                ctx=ast.Load(),
                            ),
                        ],
                        keywords=[],
                    ),
                )
            )
            self.blocks[test].append(
                ast.If(
                    test=ast.Compare(
                        left=self._self(probe),
                        ops=[ast.Is()],
                        comparators=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_py2bin_absent",
                                ctx=ast.Load(),
                            )
                        ],
                    ),
                    body=self._goto(after),
                    orelse=self._goto(body),
                )
            )
            self.blocks[body].append(
                ast.Assign(
                    targets=[self._self(statement.target.id, store=True)],
                    value=self._self(probe),
                )
            )
            self.blocks[
                self._emit(self._exits(statement.body, test, after), body)
            ].extend(self._goto(test))
            return after
        if isinstance(statement, ast.Try):
            if statement.finalbody:
                raise GeneratorRewriteError(
                    statement,
                    "a `try ... finally` containing `yield`, which would have "
                    "to run its cleanup when the generator is closed",
                )
            if statement.orelse:
                raise GeneratorRewriteError(
                    statement, "a `try ... else` containing `yield`"
                )
            after = self._new_block()
            # One entry block per clause, outside the region: a handler does
            # not guard itself.
            entries = [self._new_block() for _ in statement.handlers]
            body = self._new_block()
            self.blocks[block].extend(self._goto(body))
            self.region.append((statement.handlers, entries))
            self.guards[body] = list(self.region)
            self.blocks[self._emit(statement.body, body)].extend(self._goto(after))
            self.region.pop()
            for handler, entry in zip(statement.handlers, entries):
                if handler.name:
                    self.names.add(handler.name)
                self.blocks[self._emit(handler.body, entry)].extend(
                    self._goto(after)
                )
            return after
        if isinstance(statement, ast.Return):
            self.blocks[block].extend(self._stop(statement.value))
            return self._new_block()
        raise GeneratorRewriteError(
            statement,
            f"a `{type(statement).__name__.lower()}` containing `yield`",
        )

    def _exits(self, body: list[ast.stmt], test: int, after: int) -> list[ast.stmt]:
        """Rewrite this loop body's own `break`/`continue` into jumps."""

        rewriter = _ExitRewriter(self, test, after)
        rebuilt: list[ast.stmt] = []
        for statement in body:
            replaced = rewriter.visit(statement)
            if isinstance(replaced, list):
                rebuilt.extend(replaced)
            else:
                rebuilt.append(replaced)
        return rebuilt

    def _collect(self, statement: ast.stmt) -> None:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self.names.add(node.id)
            elif isinstance(node, (ast.Yield, ast.YieldFrom)):
                raise GeneratorRewriteError(
                    node, "a `yield` whose value is used needs `send`"
                )


class _ToAttributes(ast.NodeTransformer):
    """Turn the function's own names into attributes of the instance."""

    def __init__(self, names: set[str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id not in self.names:
            return node
        return ast.copy_location(
            ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr=node.id,
                ctx=node.ctx,
            ),
            node,
        )

    def visit_FunctionDef(self, node):  # a nested scope keeps its own names
        return node

    visit_Lambda = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


def rewrite(
    node: ast.FunctionDef, index: int, awaitable: bool = False
) -> tuple[ast.ClassDef, ast.FunctionDef]:
    """The class that runs the generator, and the function that makes one."""

    if node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
        raise GeneratorRewriteError(
            node, "a generator taking *args or **kwargs"
        )
    parameters = [argument.arg for argument in node.args.args]
    machine = _Machine(node)
    machine.names.update(parameters)
    machine.build()
    rename = _ToAttributes(machine.names)
    dispatch: list[ast.stmt] = []
    for number in reversed(range(len(machine.blocks))):
        body = [rename.visit(statement) for statement in machine.blocks[number]] or [
            ast.Pass()
        ]
        # Innermost first, so the nearest handler is the one that catches.
        for handlers, entries in reversed(machine.guards[number]):
            caught = []
            for handler, entry in zip(handlers, entries):
                jump: list[ast.stmt] = []
                if handler.name:
                    # The caught object outlives this call, so it is kept
                    # where every other local of the generator is kept.
                    jump.append(
                        ast.Assign(
                            targets=[
                                ast.Attribute(
                                    value=ast.Name(id="self", ctx=ast.Load()),
                                    attr=handler.name,
                                    ctx=ast.Store(),
                                )
                            ],
                            value=ast.Name(id="_py2bin_caught", ctx=ast.Load()),
                        )
                    )
                jump.extend(machine._goto(entry))
                caught.append(
                    ast.ExceptHandler(
                        type=handler.type,
                        name="_py2bin_caught" if handler.name else None,
                        body=jump,
                    )
                )
            body = [ast.Try(body=body, handlers=caught, orelse=[], finalbody=[])]
        dispatch = [
            ast.If(
                test=ast.Compare(
                    left=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr=STATE,
                        ctx=ast.Load(),
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=number)],
                ),
                body=body,
                orelse=dispatch,
            )
        ]
    if not dispatch:
        dispatch = [ast.Pass()]
    setup: list[ast.stmt] = [
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=STATE,
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=0),
        ),
        # A sentinel `next` can answer with, so exhaustion is a comparison
        # rather than an exception crossing a suspension point.
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="_py2bin_absent",
                    ctx=ast.Store(),
                )
            ],
            value=ast.Call(func=ast.Name(id="object", ctx=ast.Load()), args=[], keywords=[]),
        ),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=SENT,
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=None),
        ),
    ]
    for name in sorted(machine.names):
        setup.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr=name,
                        ctx=ast.Store(),
                    )
                ],
                value=(
                    ast.Name(id=name, ctx=ast.Load())
                    if name in parameters
                    else ast.Constant(value=None)
                ),
            )
        )
    class_name = f"_py2bin_gen{index}_{node.name}"
    made = ast.ClassDef(
        name=class_name,
        bases=[],
        keywords=[],
        decorator_list=[],
        type_params=[],
        body=[
            ast.FunctionDef(
                name="__init__",
                args=_arguments(["self", *parameters]),
                body=setup,
                decorator_list=[],
                type_params=[],
                returns=None,
            ),
            ast.FunctionDef(
                name="__iter__",
                args=_arguments(["self"]),
                body=[ast.Return(value=ast.Name(id="self", ctx=ast.Load()))],
                decorator_list=[],
                type_params=[],
                returns=None,
            ),
            # What `await` asks for. The language defines awaiting an object
            # with `__await__` as delegating to the iterator it answers with,
            # and this state machine is one - so a coroutine is a generator
            # wearing a second name, which is what it is in CPython too.
            *(
                [
                    ast.FunctionDef(
                        name="__await__",
                        args=_arguments(["self"]),
                        body=[
                            ast.Return(value=ast.Name(id="self", ctx=ast.Load()))
                        ],
                        decorator_list=[],
                        type_params=[],
                        returns=None,
                    )
                ]
                if awaitable
                else []
            ),
            # The dispatch itself, shared: `next(g)` is `g.send(None)` in the
            # protocol, and it is the same here - the only difference is what
            # a resumed `x = yield` finds waiting for it.
            ast.FunctionDef(
                name="_py2bin_run",
                args=_arguments(["self"]),
                body=[ast.While(test=ast.Constant(value=True), body=dispatch, orelse=[])],
                decorator_list=[],
                type_params=[],
                returns=None,
            ),
            ast.FunctionDef(
                name="__next__",
                args=_arguments(["self"]),
                body=[
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr=SENT,
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Constant(value=None),
                    ),
                    ast.Return(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_py2bin_run",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        )
                    ),
                ],
                decorator_list=[],
                type_params=[],
                returns=None,
            ),
            ast.FunctionDef(
                name="send",
                args=_arguments(["self", "value"]),
                body=[
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr=SENT,
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Name(id="value", ctx=ast.Load()),
                    ),
                    ast.Return(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_py2bin_run",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        )
                    ),
                ],
                decorator_list=[],
                type_params=[],
                returns=None,
            ),
        ],
    )
    maker = ast.FunctionDef(
        name=node.name,
        args=node.args,
        body=[
            ast.Return(
                value=ast.Call(
                    func=ast.Name(id=class_name, ctx=ast.Load()),
                    args=[ast.Name(id=name, ctx=ast.Load()) for name in parameters],
                    keywords=[],
                )
            )
        ],
        decorator_list=node.decorator_list,
        type_params=[],
        returns=None,
    )
    for tree in (made, maker):
        ast.fix_missing_locations(ast.copy_location(tree, node))
    return made, maker


def _arguments(names: list[str]) -> ast.arguments:
    return ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=name) for name in names],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )


def expand(tree: ast.Module) -> ast.Module:
    """Replace every generator function in `tree` with its class and maker.

    Applied to a whole module before anything is compiled, so the rest of the
    emitter never sees a `yield` - it sees a class, which it already knows how
    to compile, and a function that returns an instance of it.
    """

    counter = 0

    def walk(body: list[ast.stmt]) -> list[ast.stmt]:
        nonlocal counter
        rebuilt: list[ast.stmt] = []
        for statement in body:
            if isinstance(statement, ast.AsyncFunctionDef):
                counter += 1
                # Every `async def` becomes the machine, whether or not it
                # awaits: what makes it a coroutine is being awaitable, not
                # being suspended.
                plain = ast.FunctionDef(
                    name=statement.name,
                    args=statement.args,
                    body=statement.body,
                    decorator_list=statement.decorator_list,
                    type_params=[],
                    returns=None,
                )
                ast.copy_location(plain, statement)
                made, maker = rewrite(plain, counter, awaitable=True)
                rebuilt.append(made)
                rebuilt.append(maker)
                continue
            if isinstance(statement, ast.FunctionDef) and is_generator(statement):
                counter += 1
                made, maker = rewrite(statement, counter)
                rebuilt.append(made)
                rebuilt.append(maker)
                continue
            if isinstance(statement, (ast.ClassDef, ast.FunctionDef)):
                statement.body = walk(statement.body)
            rebuilt.append(statement)
        return rebuilt

    tree.body = walk(tree.body)
    return tree
