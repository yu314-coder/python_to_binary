"""Compute what the program already says, before the program runs.

`t + 1.5 * 2.0 - 0.5` has one interesting term in it. CPython's compiler folds
the rest away and stores `2.5` in the code object; without the same pass here,
the compiled form did the two extra operations three million times, and the
comparison against CPython was measuring an optimisation rather than a
backend.

The rule is narrow on purpose: fold only when every operand is a literal - a
number, or a string or bytes for the two operators that mean something to text
- and only when the answer is one CPython would give. Anything else is left
exactly as written.

*The operation is performed by Python itself.* The fold evaluates the operator
on the literal values rather than reimplementing it, so `//` rounds the way
Python rounds and `**` promotes the way Python promotes. Reimplementing them
is how a fold ends up disagreeing with the language in the corners.

*Anything that raises is left alone.* `1 / 0` folds to nothing: it is a program
that raises `ZeroDivisionError` at that point, and a fold that raised at
compile time would refuse a program Python accepts, while one that folded to
some value would run a program Python does not.

*Nothing that grows without bound is folded.* `2 ** 10000000` and
`"-" * 1000000` are literal expressions whose values are megabytes long.
Folding either would move the cost from run time into the size of the binary,
so a result past a fixed width is left for run time. The length is checked
before the multiplication rather than after, so the compiler never builds the
megabyte it is about to discard.

*`bool` is left alone.* `True + True` is `2`, and folding it would be correct;
but `True` and `1` are written the same way once folded, and the surrounding
analysis reads literals to decide what is an integer and what is a float. The
gain is nil and the room for error is not.
"""

from __future__ import annotations

import ast

#: The binary operators worth folding, and how each is spelled as a function of
#: two values. Taken from `operator` rather than written out, so the semantics
#: are the interpreter's.
_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
    ast.BitAnd: lambda a, b: a & b,
    ast.BitOr: lambda a, b: a | b,
    ast.BitXor: lambda a, b: a ^ b,
    ast.LShift: lambda a, b: a << b,
    ast.RShift: lambda a, b: a >> b,
}

_UNARY = {
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: +a,
    ast.Invert: lambda a: ~a,
}

#: How large a folded integer may be, in bits. A literal expression that
#: answers something wider is left for run time rather than written into the
#: binary.
_WIDEST = 1024

#: How long a folded string or bytes may be, in characters. `"-" * 1000000` is
#: a literal expression whose value is a megabyte; folding it would move the
#: cost from run time to the size of the binary, which is the wrong trade.
_LONGEST = 4096


def _text(node: ast.expr):
    """The value of a str or bytes literal, or None if this is not one."""

    if not isinstance(node, ast.Constant):
        return None
    if type(node.value) is str or type(node.value) is bytes:
        return node.value
    return None


def _numeric(node: ast.expr):
    """The value of a numeric literal, or None if this is not one.

    `bool` is refused here, so `True + 1` is never folded - see the module
    docstring. `complex` is refused because nothing downstream narrows one.
    """

    if not isinstance(node, ast.Constant):
        return None
    if type(node.value) is int or type(node.value) is float:
        return node.value
    return None


def _acceptable(value) -> bool:
    """True when this result is worth writing into the program."""

    if type(value) is int:
        return value.bit_length() <= _WIDEST
    if type(value) is float:
        # An infinity or a NaN cannot be written as a literal that reads back
        # as itself, and the emitter would have to spell it some other way.
        return value == value and value not in (float("inf"), float("-inf"))
    if type(value) is str or type(value) is bytes:
        return len(value) <= _LONGEST
    return False


class _Folder(ast.NodeTransformer):
    """Replace literal arithmetic with the value it computes."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        self.generic_visit(node)
        operation = _BINARY.get(type(node.op))
        if operation is None:
            return node
        folded = self._text_pair(node)
        if folded is not None:
            return folded
        left, right = _numeric(node.left), _numeric(node.right)
        if left is None or right is None:
            return node
        if isinstance(node.op, ast.Pow) and (
            not isinstance(right, int) or right > 256 or right < 0
        ):
            # A large or negative exponent is either enormous or a float; both
            # are cases the guards below would catch, and checking first keeps
            # the compiler from computing a number it is going to discard.
            return node
        try:
            value = operation(left, right)
        except (ArithmeticError, ValueError, TypeError):
            # A program that raises here is a program that raises at run time.
            return node
        if not _acceptable(value):
            return node
        return ast.copy_location(ast.Constant(value=value), node)

    def _text_pair(self, node: ast.BinOp) -> ast.expr | None:
        """`'a' + 'b'` and `'-' * 20`, done once rather than at every run.

        Only the two operators that have a meaning for text, and only when the
        types agree - `b'a' + 'b'` is a TypeError, and folding it would refuse
        a program at compile time that Python refuses at run time, which is a
        different program.
        """

        left, right = _text(node.left), _text(node.right)
        if isinstance(node.op, ast.Add):
            if left is None or right is None or type(left) is not type(right):
                return None
            value = left + right
        elif isinstance(node.op, ast.Mult):
            # Either order: `3 * '-'` is `'---'` as much as `'-' * 3` is.
            text, count = (left, _numeric(node.right))
            if text is None:
                text, count = (right, _numeric(node.left))
            if text is None or type(count) is not int:
                return None
            if count > 0 and count * len(text) > _LONGEST:
                return None  # checked before multiplying, not after
            value = text * count
        else:
            return None
        if not _acceptable(value):
            return None
        return ast.copy_location(ast.Constant(value=value), node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        self.generic_visit(node)
        operation = _UNARY.get(type(node.op))
        if operation is None:
            return node
        value = _numeric(node.operand)
        if value is None:
            return node
        if isinstance(node.op, ast.Invert) and not isinstance(value, int):
            return node
        try:
            folded = operation(value)
        except (ArithmeticError, TypeError):
            return node
        if not _acceptable(folded):
            return node
        return ast.copy_location(ast.Constant(value=folded), node)


def fold(tree: ast.AST) -> ast.AST:
    """Fold every literal arithmetic expression in this tree."""

    folded = _Folder().visit(tree)
    ast.fix_missing_locations(folded)
    return folded
