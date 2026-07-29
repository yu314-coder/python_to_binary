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


class _DelegationRewriter(ast.NodeTransformer):
    """`yield from xs` is a loop, so it is written as one before the cut.

    Delegation forwards iteration, which is what almost every use of it wants.
    It does not forward `send` into the sub-generator, and it does not answer
    with the sub-generator's return value, so `x = yield from g` is refused
    rather than quietly answering None.
    """

    def __init__(self) -> None:
        self.count = 0

    def visit_Expr(self, node: ast.Expr) -> ast.stmt:
        if not isinstance(node.value, ast.YieldFrom):
            self.generic_visit(node)
            return node
        self.count += 1
        item = f"_py2bin_from{self.count}"
        loop = ast.For(
            target=ast.Name(id=item, ctx=ast.Store()),
            iter=node.value.value,
            body=[
                ast.Expr(
                    value=ast.Yield(value=ast.Name(id=item, ctx=ast.Load()))
                )
            ],
            orelse=[],
            type_comment=None,
        )
        return ast.copy_location(loop, node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> ast.AST:
        raise GeneratorRewriteError(
            node, "a `yield from` whose value is used"
        )

    def visit_FunctionDef(self, node):
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


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
        if node.value is not None:
            raise GeneratorRewriteError(
                node, "a `return` with a value inside a generator"
            )
        return [ast.copy_location(part, node) for part in self.machine._stop()]

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

    def _stop(self) -> list[ast.stmt]:
        return [
            ast.Assign(
                targets=[self._self(STATE, store=True)],
                value=ast.Constant(value=-1),
            ),
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="StopIteration", ctx=ast.Load()),
                    args=[],
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
        delegation = _DelegationRewriter()
        rewriter = _ReturnRewriter(self)
        body: list[ast.stmt] = []
        for statement in self.function.body:
            replaced = rewriter.visit(delegation.visit(statement))
            body.extend(replaced if isinstance(replaced, list) else [replaced])
        first = self._new_block()
        end = self._emit(body, first)
        self.blocks[end].extend(self._stop())

    def _emit(self, body: list[ast.stmt], block: int) -> int:
        """Emit `body` starting in `block`; answer the block it ends in."""

        for statement in body:
            block = self._statement(statement, block)
        return block

    def _statement(self, statement: ast.stmt, block: int) -> int:
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
            if statement.value is not None:
                raise GeneratorRewriteError(
                    statement, "a `return` with a value inside a generator"
                )
            self.blocks[block].extend(self._stop())
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


def rewrite(node: ast.FunctionDef, index: int) -> tuple[ast.ClassDef, ast.FunctionDef]:
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
