"""`except*` written out as the `try`/`except` it is short for.

PEP 654 is not a new kind of control flow so much as an algebra on exception
groups wrapped in one. The control flow - run a body, take what it raised,
offer it to each clause in turn, put back what nobody wanted - is ordinary
Python. The algebra is not, and it is where this is easy to get wrong: which
leaves come back, in what shape, and whether a handler that raised something
of its own replaces them or joins them.

So the algebra lives in four small functions written in Python and compiled
along with the program, and the construct becomes statements that call them.
Nothing new is emitted; everything below this is `try`, `except`, `if` and a
list, which the compiler already handles and has tests for.

The four were written against CPython as an oracle rather than from its
source, and checked by running the same shape both ways: five original
exceptions, one to three clauses, five class expressions and four handler
bodies each, which is 42,100 programs. All 42,100 agree.

What a handler does with the exception it caught decides the shape of the
answer, and the two cases are told apart *here* rather than at run time:

    except* ValueError as e:        except* ValueError as e:
        raise                           raise e

The first puts the group back as it was; the second nests it inside a new
one. CPython tells them apart by the traceback entry `raise e` adds and the
bare `raise` does not - and a compiled program has no frames, so it adds no
entry either way and cannot see the difference at run time. It can see it in
the source, which is where the difference actually is: a bare `raise` in the
clause's own body becomes a sentinel that says "put this back", and anything
else is a new exception.
"""

from __future__ import annotations

import ast

#: The sentinel a bare `raise` in a clause body becomes, and the four
#: functions the rewritten statements call. Compiled with the program.
HELPERS = '''
class _py2bin_StarReraise(BaseException):
    pass


def _py2bin_star_leaves(exc, out):
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            _py2bin_star_leaves(inner, out)
    else:
        out.append(exc)
    return out


def _py2bin_star_holds(keep, leaf):
    for held in keep:
        if leaf is held:
            return True
    return False


def _py2bin_star_check(wanted):
    items = wanted
    if type(wanted) is not tuple:
        items = (wanted,)
    for item in items:
        if not isinstance(item, type):
            raise TypeError(
                "catching classes that do not inherit from BaseException is "
                "not allowed"
            )
        if not issubclass(item, BaseException):
            raise TypeError(
                "catching classes that do not inherit from BaseException is "
                "not allowed"
            )
    for item in items:
        if issubclass(item, BaseExceptionGroup):
            raise TypeError(
                "catching " + item.__name__ + " with except* is not allowed. "
                "Use except instead."
            )


def _py2bin_star_split(exc, wanted):
    if isinstance(exc, BaseExceptionGroup):
        return exc.split(wanted)
    if isinstance(exc, wanted):
        return BaseExceptionGroup("", [exc]), None
    return None, exc


def _py2bin_star_prep(orig, reraised, raised, rest):
    keep = []
    for held in reraised:
        _py2bin_star_leaves(held, keep)
    if rest is not None:
        _py2bin_star_leaves(rest, keep)
    projected = None
    if keep:
        if isinstance(orig, BaseExceptionGroup):
            projected = orig.split(lambda leaf: _py2bin_star_holds(keep, leaf))[0]
        elif rest is not None:
            projected = rest
        elif reraised:
            if len(reraised) == 1:
                projected = reraised[0]
            else:
                projected = BaseExceptionGroup("", list(reraised))
    parts = list(raised)
    if projected is not None:
        parts.append(projected)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return BaseExceptionGroup("", parts)
'''

#: One clause: ask for its classes, offer it the remainder, and run its body
#: with what it matched on record as the exception being handled - which is
#: the subgroup, not the whole of what the body raised.
_CLAUSE = '''
_py2bin_w{n}_{i} = None
_py2bin_star_check(_py2bin_w{n}_{i})
_py2bin_m{n}_{i} = None
if _py2bin_rest{n} is not None:
    _py2bin_p{n}_{i} = _py2bin_star_split(_py2bin_rest{n}, _py2bin_w{n}_{i})
    _py2bin_m{n}_{i} = _py2bin_p{n}_{i}[0]
    _py2bin_rest{n} = _py2bin_p{n}_{i}[1]
if _py2bin_m{n}_{i} is not None:
    _py2bin_set_handled(_py2bin_m{n}_{i})
    try:
        pass
    except _py2bin_StarReraise:
        _py2bin_re{n}.append(_py2bin_m{n}_{i})
    except BaseException as _py2bin_c{n}_{i}:
        _py2bin_ra{n}.append(_py2bin_c{n}_{i})
    _py2bin_set_handled(_py2bin_o{n})
'''

_SHELL = '''
try:
    pass
except BaseException as _py2bin_o{n}:
    _py2bin_rest{n} = _py2bin_o{n}
    _py2bin_re{n} = []
    _py2bin_ra{n} = []
    _py2bin_res{n} = _py2bin_star_prep(
        _py2bin_o{n}, _py2bin_re{n}, _py2bin_ra{n}, _py2bin_rest{n}
    )
    if _py2bin_res{n} is not None:
        raise _py2bin_res{n}
'''


class ExceptStarError(Exception):
    """Something an `except*` clause may not contain."""

    def __init__(self, node: ast.AST, message: str) -> None:
        super().__init__(message)
        self.node = node
        self.message = message


def _refuse_jumps(body: list[ast.stmt]) -> None:
    """`return`, `break` and `continue` may not leave an `except*` clause.

    Python refuses these where they would jump out of the block, and
    `ast.parse` does not - it is the compiler that rejects them, so a program
    CPython will not run parses perfectly well and has to be refused here
    instead.

    Where they belong to something *inside* the block they are fine: a
    `break` in a loop written in the clause breaks that loop, and a `return`
    in a function defined there returns from it. So the walk stops at a
    function, and stops looking for `break` at a loop - but it does descend
    into a `match`, whose statements hang off its cases and which an
    ordinary field walk misses.
    """

    def look(node: ast.AST, in_loop: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                continue
            if isinstance(child, ast.Return):
                raise ExceptStarError(
                    child,
                    "'break', 'continue' and 'return' cannot appear in an "
                    "except* block",
                )
            if isinstance(child, (ast.Break, ast.Continue)) and not in_loop:
                raise ExceptStarError(
                    child,
                    "'break', 'continue' and 'return' cannot appear in an "
                    "except* block",
                )
            look(child, in_loop or isinstance(child, (ast.For, ast.While, ast.AsyncFor)))

    for statement in body:
        if isinstance(
            statement,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        if isinstance(statement, ast.Return):
            raise ExceptStarError(
                statement,
                "'break', 'continue' and 'return' cannot appear in an except* "
                "block",
            )
        if isinstance(statement, (ast.Break, ast.Continue)):
            raise ExceptStarError(
                statement,
                "'break', 'continue' and 'return' cannot appear in an except* "
                "block",
            )
        look(statement, isinstance(statement, (ast.For, ast.While, ast.AsyncFor)))


class _BareRaises(ast.NodeTransformer):
    """`raise` with nothing after it, in this clause's own body.

    Not one inside a nested `except` - that re-raises whatever *that* clause
    caught, which is a different exception and a new one as far as the group
    is concerned. Not one inside a nested function either: it runs later and
    somewhere else.
    """

    def visit_Raise(self, node: ast.Raise) -> ast.AST:
        self.generic_visit(node)
        if node.exc is not None:
            return node
        return ast.copy_location(
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="_py2bin_StarReraise", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                ),
                cause=None,
            ),
            node,
        )

    def visit_Try(self, node: ast.Try) -> ast.AST:
        # The body and the `else` are still this clause's; a handler is not.
        node.body = [self.visit(inner) for inner in node.body]
        node.orelse = [self.visit(inner) for inner in node.orelse]
        node.finalbody = [self.visit(inner) for inner in node.finalbody]
        return node

    visit_TryStar = visit_Try

    def visit_FunctionDef(self, node):
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


class _Expander(ast.NodeTransformer):
    def __init__(self) -> None:
        self.count = 0
        self.used = False

    def visit_TryStar(self, node: ast.TryStar):
        self.generic_visit(node)
        self.used = True
        self.count += 1
        number = self.count
        shell = ast.parse(_SHELL.format(n=number)).body
        outer = shell[0]
        outer.body = node.body
        outer.orelse = node.orelse
        outer.finalbody = node.finalbody
        landing = outer.handlers[0]
        # The three set-up statements, then a block per clause, then the two
        # that put back what nobody took. The clauses go between.
        head, tail = landing.body[:3], landing.body[3:]
        middle: list[ast.stmt] = []
        for index, clause in enumerate(node.handlers):
            _refuse_jumps(clause.body)
            block = ast.parse(_CLAUSE.format(n=number, i=index)).body
            # `_py2bin_wN_i = None` carries the clause's class expression.
            block[0].value = clause.type
            guard = block[-1]
            body = guard.body
            # `_py2bin_set_handled(match)`, then the clause's own body with
            # its bare raises marked, then the name it bound let go.
            protected = body[1]
            protected.body = [
                _BareRaises().visit(inner) for inner in clause.body
            ]
            if clause.name is not None:
                body.insert(
                    1,
                    ast.Assign(
                        targets=[ast.Name(id=clause.name, ctx=ast.Store())],
                        value=ast.Name(
                            id=f"_py2bin_m{number}_{index}", ctx=ast.Load()
                        ),
                    ),
                )
                # Python unbinds it when the clause ends, however it ends -
                # and everything the clause could raise is caught above, so
                # this is reached whatever the body did.
                body.append(
                    ast.Delete(
                        targets=[ast.Name(id=clause.name, ctx=ast.Del())]
                    )
                )
            middle.extend(block)
        landing.body = head + middle + tail
        return shell

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


def expand(tree: ast.Module) -> ast.Module:
    """Rewrite every `except*` in `tree`, adding the helpers if any was."""

    expander = _Expander()
    tree = expander.visit(tree)
    if expander.used:
        tree.body = ast.parse(HELPERS).body + tree.body
    ast.fix_missing_locations(tree)
    return tree
