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
import copy

#: The attribute holding which block to run next. Prefixed because it shares a
#: namespace with the function's own locals, which become attributes too.
STATE = "_py2bin_state"
#: Where a `for` loop keeps the iterator it is walking, one per loop.
ITERATOR = "_py2bin_iter"
#: What `send` last put in, which a resumed `x = yield` reads.
SENT = "_py2bin_sent"
#: What `throw` left for the machine to raise where it was suspended.
THROWN = "_py2bin_thrown"


def _suspends(body: list[ast.stmt]) -> bool:
    """Whether these statements stop in the middle of themselves."""
    return any(
        isinstance(inner, (ast.Yield, ast.YieldFrom, ast.Await))
        for node in body
        for inner in ast.walk(node)
    )


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


#: PEP 380's expansion of `yield from`, whole: values sent in reach the
#: sub-iterator, its return value is the value of the expression, and what is
#: thrown or closed at the delegating generator is passed on to it.
#:
#: The passthrough is the part that reads oddly, and it is the part that
#: matters for cleanup. While a generator is delegating it is not itself
#: suspended at a `yield` of its own - the sub-iterator is - so closing it has
#: to close the sub-iterator, or that one's `finally` never runs. Python says
#: the same by putting `close` and `throw` in the expansion; both are asked
#: for by name rather than assumed, because a plain iterator has neither and
#: delegating to a list is allowed.
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
    _py2bin_s{n} = None
    _py2bin_t{n} = None
    try:
        _py2bin_s{n} = yield _py2bin_y{n}
    except GeneratorExit as _py2bin_g{n}:
        _py2bin_c{n} = getattr(_py2bin_i{n}, "close", None)
        if _py2bin_c{n} is not None:
            _py2bin_c{n}()
        raise _py2bin_g{n}
    except BaseException as _py2bin_b{n}:
        _py2bin_t{n} = _py2bin_b{n}
    if _py2bin_t{n} is None:
        try:
            if _py2bin_s{n} is None:
                _py2bin_y{n} = next(_py2bin_i{n})
            else:
                _py2bin_y{n} = _py2bin_i{n}.send(_py2bin_s{n})
        except StopIteration as _py2bin_e{n}:
            _py2bin_r{n} = _py2bin_e{n}.value
            _py2bin_go{n} = 0
    else:
        _py2bin_h{n} = getattr(_py2bin_i{n}, "throw", None)
        if _py2bin_h{n} is None:
            raise _py2bin_t{n}
        try:
            _py2bin_y{n} = _py2bin_h{n}(_py2bin_t{n})
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

    PEP 380 gives `yield from` a formal expansion in terms of `yield`, `next`,
    `send`, `throw` and `close`, and that expansion is ordinary Python - so it
    is written out here and the state machine never has to know delegation
    exists.

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


class _CleanupBeforeExit(ast.NodeTransformer):
    """Run a `finally` before a `break` or `continue` that leaves past it.

    Those jump straight to the loop's own blocks, going round the block that
    holds the cleanup - so it would simply not run. A copy of it is put
    immediately before the jump instead, which is what the jump would have
    reached had it gone the ordinary way.

    A loop written inside the protected region is left alone: a `break` there
    belongs to that loop and never leaves the region at all. So is a nested
    function, whose loops are its own.
    """

    def __init__(self, cleanup: "list[ast.stmt]"):
        self.cleanup = cleanup

    def visit_For(self, node):
        return node

    visit_While = visit_For

    def visit_FunctionDef(self, node):
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def _leave(self, node):
        return [*(copy.deepcopy(item) for item in self.cleanup), node]

    def visit_Break(self, node):
        return self._leave(node)

    def visit_Continue(self, node):
        return self._leave(node)


def _awaited(value: ast.expr) -> ast.expr:
    """`x` becomes `x.__await__()`, which is what awaiting it means."""

    return ast.Call(
        func=ast.Attribute(value=value, attr="__await__", ctx=ast.Load()),
        args=[],
        keywords=[],
    )


def _yields(node: ast.AST) -> bool:
    """Whether this suspends the function it is written in.

    A nested `def` does not, however many yields are inside it - they belong
    to that function. The walk below already skips a function it meets as a
    *child*; being handed one directly is the same question and was answered
    the other way, so `def outer(): def inner(): yield` made the machine
    treat the inner `def` as a statement containing a yield and refuse it.
    """

    if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    ):
        return False
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
            # A `yield` whose value nobody wanted still suspends, so `throw`
            # has to be able to reach it - the block a resume lands in is
            # wrapped in whatever `try` surrounded the `yield`, which is what
            # puts the exception where the program's own `except` can see it.
            self.blocks[resume].extend(_raise_thrown())
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
            # `throw` leaves the exception here and resumes; the language
            # says it is raised *at the suspension point*, which is exactly
            # the top of the block a `yield` comes back to.
            self.blocks[resume].extend(_raise_thrown())
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
            if not isinstance(statement.target, (ast.Name, ast.Subscript)):
                raise GeneratorRewriteError(
                    statement, "a `for` over a tuple target containing `yield`"
                )
            # Desugared to a while over an explicit iterator, because the
            # iterator has to outlive the suspension and a `for` keeps it
            # somewhere this rewriting cannot reach.
            self.loops += 1
            holder = f"{ITERATOR}{self.loops}"
            self.names.add(holder)
            # A target that is not a plain name is a place rather than a
            # binding - `for cell[0] in ...`, which is what a comprehension
            # whose target a closure captures becomes. It needs no slot on
            # the machine; the assignment goes through it as written.
            if isinstance(statement.target, ast.Name):
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
                    targets=[
                        self._self(statement.target.id, store=True)
                        if isinstance(statement.target, ast.Name)
                        else statement.target
                    ],
                    value=self._self(probe),
                )
            )
            self.blocks[
                self._emit(self._exits(statement.body, test, after), body)
            ].extend(self._goto(test))
            return after
        if isinstance(statement, ast.Try):
            if statement.orelse:
                raise GeneratorRewriteError(
                    statement, "a `try ... else` containing `yield`"
                )
            cleanup_body = statement.finalbody
            if cleanup_body:
                # A `break` or `continue` leaving the region jumps to the
                # loop's own blocks and would go round the cleanup. A copy of
                # it runs first instead, which is what the jump would have
                # reached had it left the ordinary way.
                through = _CleanupBeforeExit(cleanup_body)
                statement.body = [
                    inner
                    for item in statement.body
                    for inner in (
                        lambda got: got if isinstance(got, list) else [got]
                    )(through.visit(item))
                ]
                ast.fix_missing_locations(statement)
            after = self._new_block()
            # One entry block per clause, outside the region: a handler does
            # not guard itself.
            entries = [self._new_block() for _ in statement.handlers]
            body = self._new_block()
            # The cleanup is a block of its own, reached from every way out -
            # finishing, and raising. A block may suspend, which is what lets
            # the cleanup itself hold a `yield`; inlining it into the handler
            # could not. What was raised is held in a name until the cleanup
            # has run, and put back afterwards.
            leave = after
            pending = None
            if cleanup_body:
                leave = self._new_block()
                pending = f"_py2bin_pending{leave}"
                self.names.add(pending)
            self.blocks[block].extend(self._goto(body))
            if pending is not None:
                self.blocks[block].append(
                    ast.Assign(
                        targets=[ast.Name(id=pending, ctx=ast.Store())],
                        value=ast.Constant(value=None),
                    )
                )
                self.blocks[block][-1], self.blocks[block][-2] = (
                    self.blocks[block][-2],
                    self.blocks[block][-1],
                )
            self.region.append((statement.handlers, entries, cleanup_body, leave, pending))
            self.guards[body] = list(self.region)
            end = self._emit(statement.body, body)
            self.region.pop()
            if cleanup_body:
                # Emitted, not appended: the cleanup may hold a `yield`, and
                # only going through here cuts it into blocks at that point.
                finished = self._emit(
                    [copy.deepcopy(inner) for inner in cleanup_body], leave
                )
                self.blocks[finished].append(
                    ast.If(
                        test=ast.Compare(
                            left=ast.Name(id=pending, ctx=ast.Load()),
                            ops=[ast.IsNot()],
                            comparators=[ast.Constant(value=None)],
                        ),
                        body=[
                            ast.Raise(
                                exc=ast.Name(id=pending, ctx=ast.Load()),
                                cause=None,
                            )
                        ],
                        orelse=[],
                    )
                )
                self.blocks[finished].extend(self._goto(after))
            self.blocks[end].extend(self._goto(leave))
            for handler, entry in zip(statement.handlers, entries):
                if handler.name:
                    self.names.add(handler.name)
                # A clause here runs in a block of its own, reached after the
                # C `except` that caught the exception has already ended - so
                # nothing had it on record while the clause ran, and
                # `sys.exc_info()` inside one answered None where Python
                # answers with the exception, and a `raise` there was chained
                # to nothing. The record is put on at the top of the block and
                # taken off where the block hands control on.
                self.names.add(_kept(entry))
                self.names.add(_before(entry))
                self.blocks[entry][:0] = ast.parse(
                    f"self.{_before(entry)} = _py2bin_get_handled()\n"
                    f"_py2bin_set_handled(self.{_kept(entry)})\n"
                ).body
                ending = self._emit(handler.body, entry)
                self.blocks[ending].extend(
                    ast.parse(
                        f"_py2bin_set_handled(self.{_before(entry)})\n"
                    ).body
                )
                self.blocks[ending].extend(self._goto(leave))
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


class _WithRewriter(ast.NodeTransformer):
    """Turn a `with` that suspends into the try/finally it already means.

    The language defines `with` in terms of a try, and the machine can now cut
    a try into blocks - so the shortest way to let a `with` hold a `yield` is
    to write out what it stands for and let the existing path take it.

    The expansion follows the one in the reference: the manager and its
    `__exit__` are looked up once, on the type, before the body runs, so
    rebinding the name inside the body cannot change which object is left. A
    flag records whether a handler already dealt with an exception, because
    `__exit__` is called once either way and with different arguments.
    """

    def __init__(self) -> None:
        self.count = 0

    def visit_With(self, node: ast.With) -> ast.AST:
        self.generic_visit(node)
        if not _suspends(node.body):
            return node  # the ordinary compiler does better with these
        body = node.body
        for item in reversed(node.items):
            body = self._expand(item, body)
        return body if len(body) != 1 else body[0]

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
        self.generic_visit(node)
        body = node.body
        for item in reversed(node.items):
            body = self._expand(item, body, awaited=True)
        return body if len(body) != 1 else body[0]

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
        """`async for` written out as a `while` over the iterator.

        Every piece it expands to - a call, an `await` inside a `try`, a flag
        the handler sets - is a shape the machine already cuts into blocks. The
        handler sets a flag rather than breaking, because the rewriter that
        turns `break` into a jump does not reach inside `except` clauses, so a
        `break` there would outlive the loop it belonged to.
        """
        self.generic_visit(node)
        if node.orelse:
            raise GeneratorRewriteError(
                node, "an `async for ... else` containing `yield`"
            )
        self.count += 1
        iterator = f"_py2bin_aiter{self.count}"
        finished = f"_py2bin_adone{self.count}"

        def name(text: str, store: bool = False) -> ast.Name:
            return ast.Name(id=text, ctx=ast.Store() if store else ast.Load())

        step = ast.Try(
            body=[
                ast.Assign(
                    targets=[node.target],
                    value=ast.Await(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=name(iterator), attr="__anext__",
                                ctx=ast.Load(),
                            ),
                            args=[], keywords=[],
                        )
                    ),
                )
            ],
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id="StopAsyncIteration", ctx=ast.Load()),
                    name=None,
                    body=[
                        ast.Assign(
                            targets=[name(finished, True)],
                            value=ast.Constant(value=True),
                        )
                    ],
                )
            ],
            orelse=[],
            finalbody=[],
        )
        return [
            ast.Assign(
                targets=[name(iterator, True)],
                value=ast.Call(
                    func=ast.Attribute(
                        value=node.iter, attr="__aiter__", ctx=ast.Load()
                    ),
                    args=[], keywords=[],
                ),
            ),
            ast.Assign(
                targets=[name(finished, True)], value=ast.Constant(value=False)
            ),
            ast.While(
                test=ast.UnaryOp(op=ast.Not(), operand=name(finished)),
                body=[
                    step,
                    ast.If(
                        test=ast.UnaryOp(op=ast.Not(), operand=name(finished)),
                        body=node.body,
                        orelse=[],
                    ),
                ],
                orelse=[],
            ),
        ]

    def _expand(
        self, item: ast.withitem, body: list[ast.stmt], awaited: bool = False
    ) -> list[ast.stmt]:
        self.count += 1
        tag = self.count
        enter_name = "__aenter__" if awaited else "__enter__"
        exit_name = "__aexit__" if awaited else "__exit__"

        def maybe_await(value: ast.expr) -> ast.expr:
            return ast.Await(value=value) if awaited else value

        manager = f"_py2bin_with{tag}"
        leave = f"_py2bin_exit{tag}"
        raised = f"_py2bin_raised{tag}"
        caught = f"_py2bin_error{tag}"

        def name(text: str, store: bool = False) -> ast.Name:
            return ast.Name(
                id=text, ctx=ast.Store() if store else ast.Load()
            )

        def call_exit(arguments: list[ast.expr]) -> ast.expr:
            return maybe_await(
                ast.Call(
                    func=name(leave), args=[name(manager)] + arguments,
                    keywords=[],
                )
            )

        of_type = ast.Call(
            func=ast.Name(id="type", ctx=ast.Load()), args=[name(manager)],
            keywords=[],
        )
        head: list[ast.stmt] = [
            ast.Assign(targets=[name(manager, True)], value=item.context_expr),
            ast.Assign(
                targets=[name(leave, True)],
                value=ast.Attribute(value=of_type, attr=exit_name, ctx=ast.Load()),
            ),
        ]
        entered = maybe_await(ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="type", ctx=ast.Load()),
                    args=[name(manager)], keywords=[],
                ),
                attr=enter_name, ctx=ast.Load(),
            ),
            args=[name(manager)], keywords=[],
        ))
        if item.optional_vars is not None:
            head.append(ast.Assign(targets=[item.optional_vars], value=entered))
        else:
            head.append(ast.Expr(value=entered))
        head.append(
            ast.Assign(targets=[name(raised, True)], value=ast.Constant(value=False))
        )

        nothing = [ast.Constant(value=None)] * 3
        # A `return` in this machine is signalled by raising StopIteration, so
        # the handler below sees the frame leaving as though it were a
        # failure - and hands __aexit__ a StopIteration where CPython passes
        # None. Told apart here: the machine leaving is a clean exit, and the
        # exception is put back afterwards so the frame still ends.
        leaving: list[ast.stmt] = [
            ast.If(
                test=ast.Call(
                    func=ast.Name(id="isinstance", ctx=ast.Load()),
                    args=[
                        ast.Name(id=caught, ctx=ast.Load()),
                        ast.Name(id="StopIteration", ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
                body=[
                    ast.Expr(value=call_exit(nothing)),
                    ast.Raise(
                        exc=ast.Name(id=caught, ctx=ast.Load()), cause=None
                    ),
                ],
                orelse=[],
            )
        ] if awaited else []
        handler = ast.ExceptHandler(
            type=ast.Name(id="BaseException", ctx=ast.Load()),
            name=caught,
            body=[
                ast.Assign(
                    targets=[name(raised, True)], value=ast.Constant(value=True)
                ),
                *leaving,
                ast.If(
                    test=ast.UnaryOp(
                        op=ast.Not(),
                        operand=call_exit([
                            ast.Call(
                                func=ast.Name(id="type", ctx=ast.Load()),
                                args=[name(caught)], keywords=[],
                            ),
                            name(caught),
                            ast.Attribute(
                                value=name(caught), attr="__traceback__",
                                ctx=ast.Load(),
                            ),
                        ]),
                    ),
                    # Named rather than bare: the handler's body becomes a
                    # block of its own and runs after the `except` has been
                    # left, where a bare `raise` has nothing to re-raise.
                    body=[ast.Raise(exc=name(caught), cause=None)],
                    orelse=[],
                ),
            ],
        )
        cleanup = [
            ast.If(
                test=ast.UnaryOp(op=ast.Not(), operand=name(raised)),
                body=[ast.Expr(value=call_exit(nothing))],
                orelse=[],
            )
        ]
        if awaited:
            # The cleanup has to be awaited, and a `finally` that suspends is
            # exactly what the machine refuses. It does not need one: the
            # handler already catches everything, so the only way past the try
            # is with the exception dealt with, and the ordinary exit can just
            # follow the statement.
            return head + [
                ast.Try(body=body, handlers=[handler], orelse=[], finalbody=[]),
                *cleanup,
            ]
        return head + [
            ast.Try(body=body, handlers=[handler], orelse=[], finalbody=cleanup)
        ]


def _refuse_scope_calls(node: ast.FunctionDef) -> None:
    """`locals()` here would answer with the machine's names, not the body's.

    A generator's locals are cut out of it and kept as attributes of the
    object that runs it, so a `locals()` compiled in place looks at that
    object and finds one name: `self`. It answered `{'self': ...}` and a bare
    `dir()` answered `['self']`, which is a wrong answer rather than a
    missing one - the caller has no way to tell.

    Refused with somewhere to go instead. Naming what you want listed works,
    and so does reading the names directly.
    """

    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call) or inner.args or inner.keywords:
            continue
        if not isinstance(inner.func, ast.Name):
            continue
        if inner.func.id not in ("locals", "vars", "dir"):
            continue
        raise GeneratorRewriteError(
            inner,
            f"`{inner.func.id}()` with nothing passed, inside a generator or "
            "`async def` - the body's names are kept on the object that runs "
            "it, so this would answer with that object's names rather than "
            "the body's. Name what you want listed",
        )


def rewrite(
    node: ast.FunctionDef,
    index: int,
    awaitable: bool = False,
    asynchronous: bool = False,
) -> tuple[ast.ClassDef, ast.FunctionDef]:
    """The class that runs the generator, and the function that makes one.

    `asynchronous` says this is an async generator: driven by `__aiter__` and
    `__anext__` rather than awaited, with its own yields marked so that the
    step object can tell them from the ones an `await` inside it makes.
    """

    # Every name the signature binds, in the order the maker will hand them
    # over. `*rest` and `**more` need nothing special: by the time the maker
    # runs they are an ordinary tuple and dict in ordinary locals, and the
    # machine stores them like any other parameter. Refusing them refused
    # `async def __aexit__(self, *exc)`, which is how that method is written.
    parameters = _every_parameter(node.args)
    _refuse_scope_calls(node)
    if "self" in parameters:
        # A method's first parameter is spelled the same as the machine's own
        # receiver, and every name the function binds is rewritten to an
        # attribute of that receiver - so the machine's own `self.<state>`
        # was rewritten too, into `self.self.<state>`. The state then lived
        # on the instance rather than on the machine, never advanced, and
        # `list(obj.items())` yielded the first value for ever. Renamed here,
        # before anything is built, so the two never meet.
        node = _Renamed("self", _RECEIVER).visit(node)
        ast.fix_missing_locations(node)
        parameters = _every_parameter(node.args)
    if asynchronous:
        # Before the machine is built, and so before the delegation pass turns
        # every `await` into a `yield`: what is marked is what the program
        # wrote, and what is left plain belongs to the event loop.
        # The statements, not the `def`: the pass refuses to descend into a
        # function so that a nested one keeps its own yields, and handing it
        # this one would have it refuse immediately.
        tagger = _TagYields()
        node.body = [tagger.visit(statement) for statement in node.body]
        ast.fix_missing_locations(node)
    node.body = [_WithRewriter().visit(inner) for inner in node.body]
    node.body = [
        statement
        for inner in node.body
        for statement in (inner if isinstance(inner, list) else [inner])
    ]
    ast.fix_missing_locations(node)
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
        for handlers, entries, cleanup_body, leave, pending in reversed(
            machine.guards[number]
        ):
            caught = []
            for handler, entry in zip(handlers, entries):
                # Always named, even where the clause did not name it: the
                # block that runs the clause needs the object to put on
                # record as the one being handled, and this is the only
                # place it is in hand.
                jump: list[ast.stmt] = [
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr=_kept(entry),
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Name(id="_py2bin_caught", ctx=ast.Load()),
                    )
                ]
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
                        name="_py2bin_caught",
                        body=jump,
                    )
                )
            if cleanup_body:
                # A real `finally:` here would fire on the way out of every
                # `yield` too, because a yield returns from __next__. So what
                # was raised is put where the cleanup block can find it, and
                # that block is jumped to - a block may suspend, which is what
                # lets the cleanup hold a `yield` of its own. It puts the
                # exception back when it is done.
                store: list[ast.stmt] = [
                    rename.visit(
                        ast.Assign(
                            targets=[ast.Name(id=pending, ctx=ast.Store())],
                            value=ast.Name(id="_py2bin_caught", ctx=ast.Load()),
                        )
                    )
                ]
                store.extend(machine._goto(leave))
                caught.append(
                    ast.ExceptHandler(
                        type=ast.Name(id="BaseException", ctx=ast.Load()),
                        name="_py2bin_caught",
                        body=store,
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
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=THROWN,
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
                    ast.Name(id=_incoming(parameters.index(name)), ctx=ast.Load())
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
                # The parameters arrive under names of this machine's own
                # choosing rather than the ones the function gave them. A
                # method's first parameter is `self`, and so is the machine's
                # own receiver - two of them in one signature is not a
                # signature, which is what refused every generator method.
                args=_arguments(
                    ["self", *(_incoming(i) for i in range(len(parameters)))]
                ),
                body=setup,
                decorator_list=[],
                type_params=[],
                returns=None,
            ),
            # An async generator is not iterable, and saying it was would let
            # `list(agen)` half-work and hand back the marked tuples. It gets
            # `__aiter__` instead, and an `__anext__` that answers with the
            # step object rather than a value.
            *(
                [
                    ast.FunctionDef(
                        name="__aiter__",
                        args=_arguments(["self"]),
                        body=[
                            ast.Return(value=ast.Name(id="self", ctx=ast.Load()))
                        ],
                        decorator_list=[],
                        type_params=[],
                        returns=None,
                    ),
                    ast.FunctionDef(
                        name="__anext__",
                        args=_arguments(["self"]),
                        body=[
                            ast.Return(
                                value=ast.Call(
                                    func=ast.Name(
                                        id="_py2bin_astep", ctx=ast.Load()
                                    ),
                                    args=[ast.Name(id="self", ctx=ast.Load())],
                                    keywords=[],
                                )
                            )
                        ],
                        decorator_list=[],
                        type_params=[],
                        returns=None,
                    ),
                    # The same step object with something to send, the one
                    # that raises at the suspension point instead, and the
                    # shutdown the language pairs with them. All three are
                    # awaited rather than called, which is why each is an
                    # object with an `__await__` and not a method that runs.
                    *ast.parse(
                        "def asend(self, _py2bin_value):\n"
                        "    return _py2bin_astep(self, _py2bin_value)\n"
                        "def athrow(self, _py2bin_exc):\n"
                        "    return _py2bin_athrow(self, _py2bin_exc)\n"
                        "def aclose(self):\n"
                        "    return _py2bin_aclose(self)\n"
                    ).body,
                ]
                if asynchronous
                else [
                    ast.FunctionDef(
                        name="__iter__",
                        args=_arguments(["self"]),
                        body=[
                            ast.Return(value=ast.Name(id="self", ctx=ast.Load()))
                        ],
                        decorator_list=[],
                        type_params=[],
                        returns=None,
                    )
                ]
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
                body=[
                    *_edges(),
                    # Anything that leaves the machine leaves it finished.
                    # A `yield` is a `return` from here and never comes this
                    # way; an exception the body did not catch does, and
                    # Python does not offer a generator that raised a second
                    # chance to raise the same thing again.
                    ast.Try(
                        body=[
                            ast.While(
                                test=ast.Constant(value=True),
                                body=dispatch,
                                orelse=[],
                            )
                        ],
                        handlers=[
                            ast.ExceptHandler(
                                type=ast.Name(id="BaseException", ctx=ast.Load()),
                                name=None,
                                body=[
                                    ast.Assign(
                                        targets=[
                                            ast.Attribute(
                                                value=ast.Name(
                                                    id="self", ctx=ast.Load()
                                                ),
                                                attr=STATE,
                                                ctx=ast.Store(),
                                            )
                                        ],
                                        value=ast.Constant(value=-1),
                                    ),
                                    ast.Raise(exc=None, cause=None),
                                ],
                            )
                        ],
                        orelse=[],
                        finalbody=[],
                    ),
                ],
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
            # `throw` puts the exception where the resume block will find
            # it; `close` is `throw(GeneratorExit)` and the language's rules
            # about what the generator may do with it. Both are written out
            # as Python and compiled like the rest of the class.
            # A generator that has not run yet has nowhere to put what
            # `send` is given: there is no suspended `yield` waiting for it.
            # Python says so; this accepted the value and dropped it.
            *ast.parse(
                f"def _py2bin_refuse_send(self, _py2bin_value):\n"
                f"    if self.{STATE} == 0:\n"
                f"        if _py2bin_value is not None:\n"
                f"            raise TypeError("
                f"'can\\'t send non-None value to a just-started generator')\n"
            ).body,
            *ast.parse(
                f"def throw(self, _py2bin_exc):\n"
                f"    self.{THROWN} = _py2bin_exc\n"
                f"    self.{SENT} = None\n"
                f"    return self._py2bin_run()\n"
                f"def close(self):\n"
                f"    try:\n"
                f"        self.throw(GeneratorExit())\n"
                f"    except GeneratorExit:\n"
                f"        return None\n"
                f"    except StopIteration:\n"
                f"        return None\n"
                f"    raise RuntimeError('generator ignored GeneratorExit')\n"
            ).body,
            ast.FunctionDef(
                name="send",
                args=_arguments(["self", "value"]),
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="_py2bin_refuse_send",
                                ctx=ast.Load(),
                            ),
                            args=[ast.Name(id="value", ctx=ast.Load())],
                            keywords=[],
                        )
                    ),
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
        # What the `def` said it answers with. The parameters keep their
        # annotations because the signature is reused as it stands; this one
        # is written separately and was being dropped, so a generator or an
        # `async def` reported every annotation but `return`.
        returns=node.returns,
    )
    for tree in (made, maker):
        ast.fix_missing_locations(ast.copy_location(tree, node))
    return made, maker


#: The name of the sentinel that tells an async generator's own yields from
#: the ones an `await` inside it produces. Both come out of the same state
#: machine as a plain `yield`, and the two go to different places: one to
#: whoever is iterating, the other to the event loop.
_AGEN_MARK = "_py2bin_agen_mark"

#: The awaitable `__anext__` answers with. It drives the machine, passes
#: anything untagged out to the loop, and answers with the payload of the
#: first tagged value it sees - which is `await`'s value, delivered the way
#: `__await__` delivers one: by returning it.
_AGEN_HELPER = f"""
{_AGEN_MARK} = object()


class _py2bin_astep:

    def __init__(self, _py2bin_owner, _py2bin_first=None):
        self.owner = _py2bin_owner
        self.first = _py2bin_first

    def __await__(self):
        _py2bin_sent = self.first
        while True:
            try:
                _py2bin_item = self.owner.send(_py2bin_sent)
            except StopIteration:
                raise StopAsyncIteration
            if type(_py2bin_item) is tuple:
                if len(_py2bin_item) == 2:
                    if _py2bin_item[0] is {_AGEN_MARK}:
                        return _py2bin_item[1]
            _py2bin_sent = yield _py2bin_item


class _py2bin_athrow:

    def __init__(self, _py2bin_owner, _py2bin_exc):
        self.owner = _py2bin_owner
        self.exc = _py2bin_exc

    def __await__(self):
        try:
            _py2bin_item = self.owner.throw(self.exc)
        except StopIteration:
            raise StopAsyncIteration
        while True:
            if type(_py2bin_item) is tuple:
                if len(_py2bin_item) == 2:
                    if _py2bin_item[0] is {_AGEN_MARK}:
                        return _py2bin_item[1]
            _py2bin_sent = yield _py2bin_item
            try:
                _py2bin_item = self.owner.send(_py2bin_sent)
            except StopIteration:
                raise StopAsyncIteration


class _py2bin_aclose:

    def __init__(self, _py2bin_owner):
        self.owner = _py2bin_owner

    def __await__(self):
        if False:
            yield None
        self.owner.close()
        return None
"""


class _TagYields(ast.NodeTransformer):
    """Mark this function's own `yield`s, leaving `await` alone.

    Run before the delegation pass turns every `await` into a `yield`, so
    what it marks is exactly what the program wrote.
    """

    def visit_Yield(self, node: ast.Yield) -> ast.AST:
        self.generic_visit(node)
        payload = node.value if node.value is not None else ast.Constant(value=None)
        return ast.copy_location(
            ast.Yield(
                value=ast.Tuple(
                    elts=[ast.Name(id=_AGEN_MARK, ctx=ast.Load()), payload],
                    ctx=ast.Load(),
                )
            ),
            node,
        )

    def visit_FunctionDef(self, node):
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


#: What a generator's own `self` parameter is renamed to, so that it cannot be
#: confused with the machine's receiver of the same spelling.
_RECEIVER = "_py2bin_recv"


class _Renamed(ast.NodeTransformer):
    """Rename one plain name throughout a function, parameters included."""

    def __init__(self, was: str, now: str) -> None:
        self.was = was
        self.now = now

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.was:
            return ast.copy_location(ast.Name(id=self.now, ctx=node.ctx), node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg == self.was:
            return ast.copy_location(ast.arg(arg=self.now), node)
        return node


def _raise_thrown() -> list[ast.stmt]:
    """`if self.<thrown> is not None: raise it` - the top of a resume block."""

    return ast.parse(
        f"if self.{THROWN} is not None:\n"
        f"    _py2bin_e = self.{THROWN}\n"
        f"    self.{THROWN} = None\n"
        f"    raise _py2bin_e\n"
    ).body


def _kept(entry: int) -> str:
    """Where the exception a clause caught is kept while the clause runs."""

    return f"_py2bin_kept{entry}"


def _before(entry: int) -> str:
    """Where what was being handled before a clause began is kept."""

    return f"_py2bin_before{entry}"


def _edges() -> list[ast.stmt]:
    """The two states with no block of their own: not started, and finished.

    The dispatch below is a chain of `if self.<state> == N`, one arm per
    block, and there is no arm for either end. Before this, arriving at
    either fell off the bottom of the chain and went round the `while True`
    again, which is a program that stops answering - `next` on a generator
    already exhausted did it, and that is not an unusual thing to write.

    Both ends are also where `throw` lands when it is given a generator that
    has nothing suspended to raise *at*. Python does not run the body in
    that case: there is no `yield` waiting, so the exception simply comes
    back out of `throw`, and the generator is finished afterwards either
    way. `close` is `throw(GeneratorExit)`, so a generator closed before it
    ever ran gets its GeneratorExit straight back here without its body
    running - which is what Python does, and why closing one twice is quiet
    rather than a complaint that it ignored the exit.
    """

    return ast.parse(
        f"if self.{STATE} == 0 or self.{STATE} == -1:\n"
        f"    if self.{THROWN} is not None:\n"
        f"        _py2bin_e = self.{THROWN}\n"
        f"        self.{THROWN} = None\n"
        f"        self.{STATE} = -1\n"
        f"        raise _py2bin_e\n"
        f"if self.{STATE} == -1:\n"
        f"    raise StopIteration\n"
    ).body


def _every_parameter(arguments: ast.arguments) -> list[str]:
    """Every name the signature binds, in the order the maker hands them over.

    `*rest` and `**more` need nothing special: by the time the maker runs
    they are an ordinary tuple and dict in ordinary locals, and the machine
    stores them like any other parameter. Refusing them refused `async def
    __aexit__(self, *exc)`, which is how that method is written.
    """

    names = [
        argument.arg
        for argument in (
            *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs
        )
    ]
    if arguments.vararg:
        names.append(arguments.vararg.arg)
    if arguments.kwarg:
        names.append(arguments.kwarg.arg)
    return names


def _incoming(position: int) -> str:
    """What the machine calls the argument in that position.

    Its own, not the function's: the two meet in `__init__`, and a method's
    first parameter is spelled the same as the receiver. The value still
    lands on the attribute the function's name, so the body reads what it
    always did.
    """

    return f"_py2bin_in{position}"


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


def _awaits(node: ast.AST) -> bool:
    """Whether this expression awaits, not counting a nested function.

    The node itself counts: `x and await f()` has the `await` *as* the second
    value rather than inside it, and looking only at children missed exactly
    the shape this is asked about.
    """

    if isinstance(node, ast.Await):
        return True
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
        ):
            continue
        if isinstance(child, ast.Await) or _awaits(child):
            return True
    return False


def _is_async_comprehension(node: ast.AST) -> bool:
    """A comprehension the state machine cannot leave as one expression.

    Either it iterates something asynchronously, or it awaits somewhere
    inside - `[await one(i) for i in xs]` is as much a suspension point as
    `[x async for x in xs]`, and the machine cuts at statements, so both have
    to be written out as statements.
    """

    if not isinstance(
        node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    ):
        return False
    return any(clause.is_async for clause in node.generators) or _awaits(node)


class _Unfold(ast.NodeTransformer):
    """Turn `[x async for x in g()]` into the loop it is short for.

    An async comprehension is refused everywhere below this: the rewriter that
    turns an `async def` into a state machine will not hoist an `await` out of
    a comprehension, because a comprehension is one expression and the machine
    cuts at statements. So it is written out as statements first - a container,
    an `async for` filling it, and the name in place of the expression - and
    from there it is `async for`, which is already handled.

    The statements go immediately before the one that held it, so a
    comprehension inside a loop builds a new container each time round.
    """

    def __init__(self) -> None:
        self.made: list[ast.stmt] = []
        self.count = 0

    def visit_Lambda(self, node):
        return node

    def visit_FunctionDef(self, node):
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit(self, node):
        if _is_async_comprehension(node):
            return self._unfold(node)
        return super().visit(node)

    def _unfold(self, node):
        # Inner comprehensions first, so one nested in another is a name by
        # the time this one is written out.
        node = super().generic_visit(node)
        self.count += 1
        name = f"_py2bin_ac{self.count}"
        if isinstance(node, ast.DictComp):
            empty: ast.expr = ast.Dict(keys=[], values=[])
        elif isinstance(node, ast.SetComp):
            empty = ast.Call(
                func=ast.Name(id="set", ctx=ast.Load()), args=[], keywords=[]
            )
        else:
            empty = ast.List(elts=[], ctx=ast.Load())
        if isinstance(node, ast.DictComp):
            step: ast.stmt = ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(id=name, ctx=ast.Load()),
                        slice=node.key,
                        ctx=ast.Store(),
                    )
                ],
                value=node.value,
            )
        else:
            step = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=name, ctx=ast.Load()),
                        attr="add" if isinstance(node, ast.SetComp) else "append",
                        ctx=ast.Load(),
                    ),
                    args=[node.elt],
                    keywords=[],
                )
            )
        body = [step]
        for clause in reversed(node.generators):
            for condition in reversed(clause.ifs):
                body = [ast.If(test=condition, body=body, orelse=[])]
            loop = ast.AsyncFor if clause.is_async else ast.For
            body = [
                loop(
                    target=clause.target,
                    iter=clause.iter,
                    body=body,
                    orelse=[],
                    type_comment=None,
                )
            ]
        self.made.append(
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=empty)
        )
        self.made.extend(body)
        return ast.Name(id=name, ctx=ast.Load())


class _UnfoldChoices(ast.NodeTransformer):
    """`c and await g()` and `await g() if c else 1`, written as statements.

    An `await` in one of these runs only when the program reaches it, and
    lifting it out in front - which is how every other `await` becomes a
    suspension point the machine can cut at - would run it whether or not the
    program said to. They were refused for that, and the refusal was right
    about the lifting and wrong about the construct: an `and` is an `if`, and
    a conditional expression is an `if` with two arms, so both can be written
    as the statements they stand for and the awaits inside become ordinary
    ones.
    """

    def __init__(self) -> None:
        self.made: list[ast.stmt] = []
        self.count = 0

    def _name(self) -> str:
        self.count += 1
        return f"_py2bin_pick{self.count}"

    def visit_Lambda(self, node):
        return node

    def visit_FunctionDef(self, node):
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if not any(_awaits(value) for value in node.values[1:]):
            return node
        held = self._name()
        # `a and b and c` is `t = a; if t: t = b; if t: t = c`, nested so that
        # each one is reached only when the one before it allowed it - which
        # is what the operator means and what the refusal was protecting.
        statements: list[ast.stmt] = [
            ast.Assign(
                targets=[ast.Name(id=held, ctx=ast.Store())], value=node.values[0]
            )
        ]
        innermost = statements
        for value in node.values[1:]:
            test: ast.expr = ast.Name(id=held, ctx=ast.Load())
            if isinstance(node.op, ast.Or):
                test = ast.UnaryOp(op=ast.Not(), operand=test)
            branch = ast.If(
                test=test,
                body=[
                    ast.Assign(
                        targets=[ast.Name(id=held, ctx=ast.Store())], value=value
                    )
                ],
                orelse=[],
            )
            innermost.append(branch)
            innermost = branch.body
        self.made.extend(statements)
        return ast.copy_location(ast.Name(id=held, ctx=ast.Load()), node)

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        if not (_awaits(node.body) or _awaits(node.orelse)):
            return node
        held = self._name()
        self.made.append(
            ast.If(
                test=node.test,
                body=[
                    ast.Assign(
                        targets=[ast.Name(id=held, ctx=ast.Store())],
                        value=node.body,
                    )
                ],
                orelse=[
                    ast.Assign(
                        targets=[ast.Name(id=held, ctx=ast.Store())],
                        value=node.orelse,
                    )
                ],
            )
        )
        return ast.copy_location(ast.Name(id=held, ctx=ast.Load()), node)


def _refuse_capturing_the_target(node: ast.GeneratorExp) -> None:
    """A closure written inside a genexp cannot capture what the genexp binds.

    Python closes over the variable, so every such closure sees the last
    value the genexp produced - `list((lambda: i) for i in range(3))` is
    `[2, 2, 2]`. A genexp here becomes a generator function and its names
    become attributes of the object that runs it, so the closure looked for
    `i` where there is none and the call raised `NameError` naming a variable
    the program had plainly written.

    Refused where the list, set and dict forms are already refused, and for
    the same reason: capturing by value would answer `[0, 1, 2]`, and a
    disagreement is worth saying rather than arriving at.
    """

    bound = {
        name.id
        for clause in node.generators
        for name in ast.walk(clause.target)
        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
    }
    for inner in ast.walk(node):
        if not isinstance(inner, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        read = {
            name.id
            for name in ast.walk(inner)
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)
        }
        caught = sorted(read & bound)
        if caught:
            spelled = ", ".join(caught)
            raise GeneratorRewriteError(
                inner,
                f"this closure captures {spelled} from the generator "
                f"expression around it, which rebinds it on every turn - "
                f"captures are taken by value here, so every closure would "
                f"see a different value where Python gives them all the last "
                f"one. Write it as a default (`lambda {caught[0]}="
                f"{caught[0]}: ...`) to say the by-value thing",
            )


class _Genexps(ast.NodeTransformer):
    """`(elt for x in it)` written as the generator function it is.

    It was gathered into a list and an iterator handed back over that, which
    is right for the values and wrong for when they are computed: the whole
    sequence ran before anything asked for the first item. Side effects
    happened too early, a large sequence was built in full, and an endless one
    - `(x * 2 for x in count())` - never came back at all.

    CPython makes a genexp a function, and so does this. The first iterable
    is evaluated where the expression is written, as it is there, and passed
    in; everything else waits to be asked for.
    """

    def __init__(self) -> None:
        self.made: list[ast.stmt] = []
        self.count = 0

    def visit_Lambda(self, node):
        return node

    def visit_FunctionDef(self, node):
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        # Inner ones first, so a genexp inside another is already a call.
        self.generic_visit(node)
        _refuse_capturing_the_target(node)
        if any(clause.is_async for clause in node.generators):
            # An async one is a different animal and is unfolded elsewhere.
            return node
        self.count += 1
        name = f"_py2bin_ge{self.count}"
        source = "_py2bin_ge_src"
        body: list[ast.stmt] = [ast.Expr(value=ast.Yield(value=node.elt))]
        for index, clause in enumerate(reversed(node.generators)):
            for condition in reversed(clause.ifs):
                body = [ast.If(test=condition, body=body, orelse=[])]
            first = index == len(node.generators) - 1
            body = [
                ast.For(
                    target=clause.target,
                    iter=(
                        ast.Name(id=source, ctx=ast.Load())
                        if first
                        else clause.iter
                    ),
                    body=body,
                    orelse=[],
                    type_comment=None,
                )
            ]
        self.made.append(
            ast.FunctionDef(
                name=name,
                args=_arguments([source]),
                body=body,
                decorator_list=[],
                type_params=[],
                returns=None,
            )
        )
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=name, ctx=ast.Load()),
                args=[node.generators[0].iter],
                keywords=[],
            ),
            node,
        )


def expand_genexps(body: list[ast.stmt]) -> list[ast.stmt]:
    """Write out every generator expression in these statements."""

    rebuilt: list[ast.stmt] = []
    for statement in body:
        for field in _NESTED:
            held = getattr(statement, field, None)
            if isinstance(held, list) and held and isinstance(held[0], ast.stmt):
                setattr(statement, field, expand_genexps(held))
        for handler in getattr(statement, "handlers", ()):
            handler.body = expand_genexps(handler.body)
        maker = _Genexps()
        statement = maker.visit(statement)
        if maker.made:
            for made in maker.made:
                ast.fix_missing_locations(made)
            rebuilt.extend(maker.made)
        rebuilt.append(statement)
    return rebuilt


def unfold_conditional_awaits(body: list[ast.stmt]) -> list[ast.stmt]:
    """Write out every `and`, `or` and `a if c else b` that awaits.

    The statements go in front of the one that held the expression, which is
    where they run - except for a `while`, whose test runs again every turn
    and where putting them in front would run them once. That one is left to
    the refusal it already had.
    """

    rebuilt: list[ast.stmt] = []
    for statement in body:
        for field in _NESTED:
            held = getattr(statement, field, None)
            if isinstance(held, list) and held and isinstance(held[0], ast.stmt):
                setattr(statement, field, unfold_conditional_awaits(held))
        for handler in getattr(statement, "handlers", ()):
            handler.body = unfold_conditional_awaits(handler.body)
        if isinstance(statement, ast.While):
            if _awaits(statement.test):
                # A `while` test runs again every turn, so the statements it
                # unfolds to have to run every turn too - which means inside
                # the loop rather than in front of it. `while c:` becomes
                # `while True:` with the test taken at the top and a `break`
                # when it says no. An `else` runs when the test is what ended
                # it, so it goes in front of that `break` and not on the path
                # a `break` in the body takes - which is what `else` means.
                ending: list[ast.stmt] = list(statement.orelse)
                ending.append(ast.Break())
                statement = ast.copy_location(
                    ast.While(
                        test=ast.Constant(value=True),
                        body=[
                            ast.If(
                                test=ast.UnaryOp(
                                    op=ast.Not(), operand=statement.test
                                ),
                                body=ending,
                                orelse=[],
                            ),
                            *statement.body,
                        ],
                        orelse=[],
                    ),
                    statement,
                )
                ast.fix_missing_locations(statement)
                # Its own body is unfolded now that the test lives in it.
                statement.body = unfold_conditional_awaits(statement.body)
            rebuilt.append(statement)
            continue
        unfolder = _UnfoldChoices()
        statement = unfolder.visit(statement)
        if unfolder.made:
            for made in unfolder.made:
                ast.fix_missing_locations(made)
            rebuilt.extend(unfolder.made)
        rebuilt.append(statement)
    return rebuilt


def unfold_async_comprehensions(body: list[ast.stmt]) -> list[ast.stmt]:
    """Write out every async comprehension in these statements, in place."""

    rebuilt: list[ast.stmt] = []
    for statement in body:
        for field in _NESTED:
            held = getattr(statement, field, None)
            if isinstance(held, list) and held and isinstance(held[0], ast.stmt):
                setattr(statement, field, unfold_async_comprehensions(held))
        for handler in getattr(statement, "handlers", ()):
            handler.body = unfold_async_comprehensions(handler.body)
        if any(_is_async_comprehension(inner) for inner in ast.walk(statement)):
            unfolder = _Unfold()
            statement = unfolder.visit(statement)
            for made in unfolder.made:
                ast.fix_missing_locations(made)
            rebuilt.extend(unfolder.made)
        rebuilt.append(statement)
    return rebuilt


def expand(tree: ast.Module) -> ast.Module:
    """Replace every generator function in `tree` with its class and maker.

    Applied to a whole module before anything is compiled, so the rest of the
    emitter never sees a `yield` - it sees a class, which it already knows how
    to compile, and a function that returns an instance of it.
    """

    tree.body = expand_genexps(tree.body)
    counter = 0
    wants_helper = False

    def walk(
        body: list[ast.stmt], hoist: list[ast.stmt] | None = None
    ) -> list[ast.stmt]:
        """Rewrite these statements; `hoist` takes the classes made on the way.

        A generator *method* becomes a machine class and a maker, and a class
        inside a class body is not something the emitter translates - so
        `def items(self): yield` and every `async def` method were refused,
        which is most of the ways either gets written. The machine has no
        business being in there anyway: it is not a member of the class, only
        something the method makes an instance of. It goes out in front of the
        class instead, where it is bound by the time any method can run.
        """

        nonlocal counter
        rebuilt: list[ast.stmt] = []
        made_here = rebuilt if hoist is None else hoist
        for statement in body:
            if isinstance(statement, ast.AsyncFunctionDef):
                counter += 1
                # Every `async def` becomes the machine, whether or not it
                # awaits: what makes it a coroutine is being awaitable, not
                # being suspended.
                plain = ast.FunctionDef(
                    name=statement.name,
                    args=statement.args,
                    # An async comprehension is written out as the `async for`
                    # it is short for before the machine sees it, because the
                    # machine cuts at statements and a comprehension is one
                    # expression.
                    body=walk(
                        unfold_async_comprehensions(
                            unfold_conditional_awaits(statement.body)
                        )
                    ),
                    decorator_list=statement.decorator_list,
                    type_params=[],
                    # Kept, so the machine's maker can carry it: an `async
                    # def` reported every annotation but `return`.
                    returns=statement.returns,
                )
                ast.copy_location(plain, statement)
                # An `async def` that yields is an async generator, which is
                # a different thing from a coroutine: driven by `__aiter__`
                # and `__anext__` rather than awaited. Its own yields are
                # marked so that the step object can tell them from the ones
                # an `await` inside it makes - the machine turns both into
                # the same `yield`, and they go to different places.
                agen = is_generator(plain)
                if agen:
                    nonlocal wants_helper
                    wants_helper = True
                made, maker = rewrite(
                    plain, counter, awaitable=not agen, asynchronous=agen
                )
                made_here.append(made)
                rebuilt.append(maker)
                continue
            if isinstance(statement, ast.FunctionDef) and is_generator(statement):
                counter += 1
                # Its own body first: a generator written inside a generator
                # has to become its machine before this one is cut into
                # blocks, or the blocks would hold a `def` that still yields.
                statement.body = walk(statement.body)
                made, maker = rewrite(statement, counter)
                made_here.append(made)
                rebuilt.append(maker)
                continue
            if isinstance(statement, ast.ClassDef):
                lifted: list[ast.stmt] = []
                statement.body = walk(statement.body, hoist=lifted)
                # In front of the class, not inside it: the maker names the
                # machine when it *runs*, so the machine only has to exist by
                # then - but a class body is not somewhere one can be written.
                rebuilt.extend(lifted)
                rebuilt.append(statement)
                continue
            if isinstance(statement, ast.FunctionDef):
                statement.body = walk(statement.body)
                rebuilt.append(statement)
                continue
            # Every other body a statement holds. Only `def` and `class` were
            # descended into, so a generator written inside an `if`, a `for`
            # or a `try` was never turned into its machine and reached the
            # emitter still yielding - `if True:` around a `def` was enough.
            for field in _NESTED:
                held = getattr(statement, field, None)
                if (
                    isinstance(held, list)
                    and held
                    and isinstance(held[0], ast.stmt)
                ):
                    setattr(statement, field, walk(held))
            for handler in getattr(statement, "handlers", ()):
                handler.body = walk(handler.body)
            rebuilt.append(statement)
        return rebuilt

    tree.body = walk(tree.body)
    if wants_helper:
        # In front of everything, because an `async def` at the top of the
        # file is rewritten into a class whose `__anext__` names it.
        helper = ast.parse(_AGEN_HELPER).body
        for statement in helper:
            ast.fix_missing_locations(statement)
        tree.body = walk(helper) + tree.body
    return tree
