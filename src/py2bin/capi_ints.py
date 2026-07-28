"""Which of a function's locals can be held as machine integers.

The C-API tier writes one C-API call per Python operation, and since 3.11 that
is precisely the code CPython's interpreter has learned to avoid: it rewrites a
hot ``+`` between two ints into a guarded machine add and never reaches
``PyNumber_Add``. Compiling every operation to its generic entry point is
therefore *slower* than not compiling at all, which measurement confirmed.

Closing that gap means holding an integer in a register instead of on the heap.
This module decides where that is allowed. It answers one question about one
function body - which local names may be kept as a machine integer - and the
emitter turns that answer into C.

A name qualifies when three things hold.

*It is used as arithmetic.* Unboxing a name that is only ever passed around
costs a branch on every read and buys nothing, so a name has to appear as an
operand of an arithmetic operator or a comparison before it is worth doing.

*Every binding is one the emitter can narrow.* A plain assignment, an augmented
assignment, or a ``for`` target. A name also bound by an ``import``, a ``def``,
an ``except`` clause or a tuple unpacking is left alone: those paths hand over
an object and there is nothing to gain.

*Nothing else can see the storage.* A name declared ``global`` or ``nonlocal``
lives elsewhere. A name a nested function reads is fetched from a closure cell,
which holds objects. A name inside a comprehension is likewise out of reach.

Note what is *not* required: that the name always holds an integer. It never
has to. The representation carries a flag saying whether the value is currently
a machine integer or an object, so a name that holds an int on one path and a
string on another is still legal - the arithmetic simply takes the slow path
when the flag says the value is an object. The analysis here decides where the
faster representation *pays*, not where it is *safe*; it is safe everywhere.
"""

from __future__ import annotations

import ast

#: Operators whose result on two machine integers is exactly what Python would
#: have produced, given no overflow. The shifts are missing on purpose: Python
#: raises for a negative shift count and grows without bound for a large one,
#: where C has nothing to say about either. True division is missing because
#: its result is a float, not an integer.
ARITHMETIC = (
    ast.Add, ast.Sub, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.FloorDiv, ast.Mod,
)

#: Comparisons that are a single machine instruction on two integers.
COMPARISONS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)

#: Widest integer literal that certainly fits the machine word the fast path
#: computes in. Signed 64-bit, one bit kept back so that negating the smallest
#: value cannot itself overflow.
LIMIT = (1 << 62) - 1


def is_machine_integer(node: ast.expr) -> bool:
    """True for a literal that the fast path can spell as a C constant.

    `True` and `False` are `ast.Constant` holding `bool`, which is a subclass
    of `int` and would answer yes to a careless test. They are excluded: their
    arithmetic is an integer's, but writing one back out has to produce the
    bool, not the int, and the fast path does not track which it holds.
    """

    return (
        isinstance(node, ast.Constant)
        and type(node.value) is int
        and -LIMIT <= node.value <= LIMIT
    )


class _Uses(ast.NodeVisitor):
    """One pass over a body, collecting the four facts the decision needs."""

    def __init__(self) -> None:
        #: Names bound only in ways the emitter can give a machine integer to.
        self.narrowly_bound: set[str] = set()
        #: Names bound some other way, which disqualifies them outright.
        self.otherwise_bound: set[str] = set()
        #: Names used as an operand of arithmetic or of a comparison.
        self.calculated: set[str] = set()
        #: Names something other than this body can reach.
        self.escaped: set[str] = set()

    # --- bindings --------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._bound(target)
        self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and isinstance(node.op, ARITHMETIC):
            self.narrowly_bound.add(node.target.id)
            # `x += 1` reads x as arithmetic as surely as `x = x + 1` does.
            self.calculated.add(node.target.id)
        else:
            self._bound(node.target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._bound(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._bound(node.target)
        self.visit(node.iter)
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)

    visit_AsyncFor = visit_For

    def visit_Delete(self, node: ast.Delete) -> None:
        # A deleted name goes back to unbound, which the representation can
        # express - but nothing here needs it, so it is simply refused.
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.otherwise_bound.add(target.id)

    def visit_Global(self, node: ast.Global) -> None:
        self.otherwise_bound.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.otherwise_bound.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.otherwise_bound.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.otherwise_bound.add(alias.asname or alias.name)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._bound(item.optional_vars, narrowable=False)
        for statement in node.body:
            self.visit(statement)
        for item in node.items:
            self.visit(item.context_expr)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.otherwise_bound.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # The name the `def` binds, and then everything the nested body reads:
        # a name it closes over is fetched from a cell, which holds an object.
        self.otherwise_bound.add(node.name)
        self._escapes(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.otherwise_bound.add(node.name)
        self._escapes(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._escapes(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._escapes(node)

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def _escapes(self, node: ast.AST) -> None:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name):
                self.escaped.add(inner.id)

    def _bound(self, target: ast.expr, narrowable: bool = True) -> None:
        if isinstance(target, ast.Name):
            if narrowable:
                self.narrowly_bound.add(target.id)
            else:
                self.otherwise_bound.add(target.id)
            return
        if isinstance(target, ast.Starred):
            self._bound(target.value, narrowable=False)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            # Unpacking hands each name an object out of the sequence; there is
            # no machine integer at that point to keep.
            for element in target.elts:
                self._bound(element, narrowable=False)
            return
        # An attribute or a subscript binds no local name at all.
        self.visit(target)

    # --- uses ------------------------------------------------------------

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ARITHMETIC):
            for side in (node.left, node.right):
                if isinstance(side, ast.Name):
                    self.calculated.add(side.id)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) == 1 and isinstance(node.ops[0], COMPARISONS):
            for side in (node.left, node.comparators[0]):
                if isinstance(side, ast.Name):
                    self.calculated.add(side.id)
        self.generic_visit(node)


def unboxable_locals(body: list[ast.stmt], parameters: set[str]) -> set[str]:
    """The names in `body` worth holding as a machine integer.

    `parameters` are excluded. They arrive from the caller as objects, so the
    representation would have to be established on entry - a check per call
    against a saving per use, which is only worth it once the analysis can see
    how often the parameter is used. That is a later job than this one.
    """

    uses = _Uses()
    for statement in body:
        uses.visit(statement)
    return (
        (uses.calculated & uses.narrowly_bound)
        - uses.otherwise_bound
        - uses.escaped
        - parameters
    )


def narrow_range(node: ast.expr) -> list[ast.expr] | None:
    """The arguments of a `range(...)` call, if that is what this is.

    A `for` over a range is the loop that pays best: the interpreter builds an
    integer object per iteration and the compiled form needs none. Returns the
    one, two or three argument expressions, or None if the loop has to go
    through the iterator protocol like any other.

    The name `range` is only trusted when it is a bare name. A program is free
    to rebind it, which the emitter guards against at run time by comparing the
    object it resolves to against the builtin - this only decides whether that
    guard is worth emitting.
    """

    if not isinstance(node, ast.Call) or node.keywords:
        return None
    if not isinstance(node.func, ast.Name) or node.func.id != "range":
        return None
    if not 1 <= len(node.args) <= 3:
        return None
    if any(isinstance(argument, ast.Starred) for argument in node.args):
        return None
    return list(node.args)
