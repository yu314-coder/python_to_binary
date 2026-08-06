"""Translate Python to C that drives CPython through its C API.

This is the third tier, and it answers a different question from the other two.
``py2bin compile`` produces machine code with no interpreter, for a subset that
has no Python object model at all - an ``int`` is a register. ``py2bin freeze``
carries an interpreter and runs the source unchanged. This one sits between
them: every value is a ``PyObject *``, every operation is a C-API call, and the
result is machine code that needs libpython to run.

That is the strategy Nuitka uses. What is different here is that nothing
outside the standard library is involved on the way: py2bin emits the C and
py2bin's own C compiler turns it into machine code, so there is no clang, no
assembler and no linker in the pipeline.

What it buys over the native tier is Python's own semantics - integers that do
not stop at 64 bits, because ``PyNumber_Add`` on two ``PyLong`` objects is the
same arbitrary-precision addition the interpreter performs.

Reference counting follows one rule, chosen because it is easy to check by
reading: **every expression yields a reference the caller owns**. Reading a
name therefore increments before handing the value back, and every value that
a statement finishes with is released. A rule that holds everywhere is worth
more here than one that saves an increment in places.
"""

from __future__ import annotations

import ast
import copy
import builtins
import contextlib
from pathlib import Path

from .capi_cells import CellError, expand as expand_cells
from .capi_fold import fold as fold_constants
from .capi_except_star import (
    ExceptStarError,
    expand as expand_except_star,
)
from .capi_inline import expand_module as inline_calls, place_keywords
from .capi_exact import exact_dicts, exact_lists, exact_strs
from .capi_generators import GeneratorRewriteError, expand as expand_generators
from .capi_ints import (
    ARITHMETIC as _MACHINE_OPS,
    COMPARISONS as _MACHINE_TESTS,
    LIMIT as _MACHINE_LIMIT,
    is_machine_integer,
    narrow_range,
    unboxable_locals,
)
from .capi_floats import (
    ARITHMETIC as _DOUBLE_OPS,
    COMPARISONS as _DOUBLE_TESTS,
    is_machine_float,
    unboxable_locals as double_locals,
)


#: How each double operator is written in C. Division is here like the rest:
#: what makes it special is the divisor test, not the spelling.
#: How many f-string pieces are concatenated in a chain before the tuple and
#: join become the cheaper shape. Past this the join's single allocation wins
#: over a growing number of intermediates.
_CONCAT_UNTIL = 4

_DOUBLE_SPELLING = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
}

#: How each comparison is written in C, for the machine-double path.
_COMPARISON_SPELLING = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=",
}

#: The `Py_LT`-family opcode each comparison passes to the C API. The order is
#: CPython's and is not alphabetical, so it is written out rather than counted.
_COMPARISON_CODES = {
    ast.Lt: 0, ast.LtE: 1, ast.Eq: 2, ast.NotEq: 3, ast.Gt: 4, ast.GtE: 5,
}


#: Names that can read a local without naming it. If a function calls any of
#: these, every local it has may be looked at, so none of them is unread.
#: What `from m import *` binds, worked out where the strings are easy.
_SPREAD_HELPER = """
def _py2bin_spread(_py2bin_module):
    _py2bin_into = globals()
    _py2bin_names = getattr(_py2bin_module, "__all__", None)
    if _py2bin_names is None:
        _py2bin_names = []
        for _py2bin_n in dir(_py2bin_module):
            if not _py2bin_n.startswith("_"):
                _py2bin_names.append(_py2bin_n)
    for _py2bin_n in _py2bin_names:
        _py2bin_into[_py2bin_n] = getattr(_py2bin_module, _py2bin_n)
"""

#: The one slot every unread local is written to.
_UNREAD = "v__unread"

_READS_EVERY_LOCAL = frozenset({"locals", "vars", "eval", "exec", "globals"})


def reads_every_local(body: list) -> bool:
    """Whether this body can look at its locals without naming them.

    `locals()` and `vars()` answer with every slot, so a name held in a
    machine register rather than an object would be missing from what they
    hand back. A function that calls one keeps all of its names as objects:
    the narrowing is a speed decision, and this is a correctness one.
    """

    for statement in body:
        for inner in ast.walk(statement):
            if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load):
                if inner.id in _READS_EVERY_LOCAL:
                    return True
    return False


def write_only_locals(body: list, parameters: set[str]) -> set[str]:
    """Locals this function binds and never reads back.

    A name nothing ever loads does not need storage of its own. The value
    still has to be computed - `v = f()` runs `f` either way, and an
    expression is not dropped for being unread - but it can go into one slot
    shared by every such name instead of one slot each.

    That is not usually worth much: a name written and never read is a thing
    people rarely write. It matters because the frame is a single fixed
    allocation with a ceiling on it, so a generated file with sixty-seven
    thousand of them could not be compiled at all. The whole function needed
    two slots and asked for 67,001.

    Refused for anything that could read a local without naming it, and for
    the shapes where a binding is observable on its own: `global`/`nonlocal`
    storage belongs elsewhere, `del` has to have something to delete, a
    walrus is written for its value, and `except ... as e` unbinds the name
    when the handler ends.
    """

    loaded: set[str] = set()
    stored: set[str] = set()
    barred: set[str] = set(parameters)
    for statement in body:
        for inner in ast.walk(statement):
            if isinstance(inner, ast.Name):
                if isinstance(inner.ctx, ast.Load):
                    if inner.id in _READS_EVERY_LOCAL:
                        return set()
                    loaded.add(inner.id)
                else:
                    stored.add(inner.id)
            elif isinstance(inner, (ast.Global, ast.Nonlocal)):
                barred.update(inner.names)
            elif isinstance(inner, ast.AugAssign):
                if isinstance(inner.target, ast.Name):
                    # `x += 1` reads x before it writes it.
                    loaded.add(inner.target.id)
            elif isinstance(inner, ast.NamedExpr):
                if isinstance(inner.target, ast.Name):
                    barred.add(inner.target.id)
            elif isinstance(inner, ast.Delete):
                for target in inner.targets:
                    if isinstance(target, ast.Name):
                        barred.add(target.id)
            elif isinstance(inner, ast.ExceptHandler) and inner.name:
                barred.add(inner.name)
    return stored - loaded - barred


class _Renamed(ast.NodeTransformer):
    """Rename a set of plain names throughout one statement.

    Stops at anything that makes a scope of its own: a name a nested
    function binds is that function's, not the class body's.
    """

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self.mapping.get(node.id)
        if replacement is None:
            return node
        return ast.copy_location(
            ast.Name(id=replacement, ctx=node.ctx), node
        )

    def visit_alias(self, node: ast.alias) -> ast.AST:
        # An import binds through here rather than through a Name, so the
        # rename has to reach it too - otherwise the statement binds the
        # ordinary name and the class body never sees what it imported.
        spelled = node.asname or node.name.split(".")[0]
        replacement = self.mapping.get(spelled)
        if replacement is None:
            return node
        return ast.alias(name=node.name, asname=replacement)

    def visit_FunctionDef(self, node):
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


def _dotted_plain_import(statement: ast.stmt) -> str | None:
    """`import a.b` with no `as`, which binds `a` and cannot be renamed.

    `import a.b as x` binds the submodule and `import a.b` binds the package,
    so giving the second one an `as` to rename it changes what it means.
    """

    for node in ast.walk(statement):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is None and "." in alias.name:
                    return alias.name
    return None


def _scope_bindings(body: list) -> set[str]:
    """Every name these statements bind, not looking into nested scopes.

    Python has no block scope, so a name bound anywhere in a function body is
    local to all of it - including before the line that binds it. A nested
    function is a scope of its own and what it binds stays there.
    """

    bound: set[str] = set()
    pending = list(body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        pending.extend(ast.iter_child_nodes(node))
    return bound


#: Larger than any statement position: a name that never settles.
_NEVER = 1 << 30


def _module_scope_bindings(tree: ast.Module) -> dict[str, int]:
    """How many times each name is bound *at module scope*.

    Module scope is not the same as the top level of the file: a name bound
    inside a module-level `if`, `for`, `while`, `with` or `try` is still a
    module global, so this descends into those. It stops at a `def`, `class`
    or `lambda`, whose bindings belong to a scope of their own - and which
    `_Function.shadows` already accounts for separately.

    A module-level function is only worth calling directly when its `def` is
    the *sole* thing that binds the name. `f = lambda a: a + 100` after
    `def f(a): return a + 1` rebinds the global, and Python calls whichever
    one is current; the direct C call answered with the `def` forever. This
    is the same class of bug as the `len`/`str` shortcuts - a shortcut keyed
    on a name must first check the program has not bound that name elsewhere.
    """

    counts: dict[str, int] = {}

    def note(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    def walk(node: ast.AST) -> None:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            # The name it binds is this scope's; its body is a scope of its
            # own, so the walk stops here rather than counting its locals.
            note(node.name)
            return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            note(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                note((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            note(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                note(name)
        for child in ast.iter_child_nodes(node):
            walk(child)

    for statement in tree.body:
        walk(statement)

    # A `global x` inside a function binds the module's `x` from a scope this
    # walk deliberately does not enter. Without it, `def a(...)` whose body
    # says `global a; a = something` still looked like the only binding of the
    # name, kept its direct C call, and went on calling itself after Python
    # would have been calling the replacement. Counted here rather than
    # by descending, because the declaration is the whole signal: a `global`
    # that never assigns is a statement with no effect.
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            for name in node.names:
                note(name)
    return counts


def _shadowed_builtins(tree: ast.Module) -> set[str]:
    """Every name this module binds anywhere, at any depth.

    A counted `for` loop is only correct if `range` means what the interpreter
    means by it. Rather than reason about where a rebinding could reach, this
    asks whether the module rebinds the name at all - anywhere, including
    inside a function - and declines the whole optimisation if it does.
    """

    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arguments):
            # A parameter named `range` shadows the builtin for its whole
            # body, and `ast.walk` reaches every parameter list there is.
            bound.update(
                argument.arg
                for argument in (
                    *node.posonlyargs, *node.args, *node.kwonlyargs,
                    *([node.vararg] if node.vararg else []),
                    *([node.kwarg] if node.kwarg else []),
                )
            )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            # `builtins.range = ...` reaches every module at once, so any
            # assignment to an attribute of something called `builtins` is
            # taken as putting every builtin out of reach.
            if isinstance(node.value, ast.Name) and node.value.id == "builtins":
                bound.add(node.attr)
    return bound


class _Machine:
    """An expression the fast path computed, or gave up on.

    `flag` is the C name of an `int` that is 1 when `value` - a `long long` -
    holds the answer. It is None when narrowness is certain without a test,
    which is the case for a literal. A caller that needs an object has to emit
    both arms: the boxed value when the flag is set, and the ordinary C-API
    translation of the same expression when it is not.

    Recomputing the expression on the slow arm is only sound because the fast
    path is offered for trees whose leaves are literals and unboxed locals -
    nothing there can have a side effect, so evaluating it twice is the same
    as evaluating it once. That restriction is what keeps this simple.
    """

    __slots__ = ("flag", "value")

    def __init__(self, flag: str | None, value: str) -> None:
        self.flag = flag
        self.value = value


#: Written into a program built with --crash-log. An uncaught exception in a
#: windowed application has nowhere to print: there is no console attached, so
#: the traceback that would say what went wrong is discarded and the app simply
#: disappears. This puts it in a file as well, beside the executable when that
#: is writable and in the user's home when it is not - which is the case for an
#: app in /Applications.
_CRASH_REPORT = r"""
static void _py2bin_crash_report(void) {
    PyObject *held = PyErr_GetRaisedException();
    if (held) {
        /* Handed to Python through builtins: the traceback module formats an
           exception far better than anything reasonable to write in C, and it
           is already in the interpreter this program links. */
        PyObject *builtins = PyImport_ImportModule("builtins");
        if (builtins) {
            PyObject_SetAttrString(builtins, "_py2bin_crash", held);
            Py_DecRef(builtins);
            PyRun_SimpleString(
                "import builtins, os, sys, traceback\n"
                "_e = builtins._py2bin_crash\n"
                /* Beside the program, which is what _py2bin_dir holds.
                   sys.executable is the interpreter's own installation on a
                   platform that cannot resolve it to the host binary, so the
                   report landed next to the system Python rather than next to
                   the thing that crashed. */
                "_where = getattr(builtins, '_py2bin_dir', '') or "
                "os.path.dirname(os.path.abspath(sys.executable))\n"
                "for _dir in (_where, os.path.expanduser('~')):\n"
                "    try:\n"
                "        _p = os.path.join(_dir, 'crash.txt')\n"
                "        with open(_p, 'w', encoding='utf-8') as _f:\n"
                "            _f.write('argv: %r\\n\\n' % (sys.argv,))\n"
                "            traceback.print_exception(_e, file=_f)\n"
                "        sys.stderr.write('py2bin: wrote ' + _p + '\\n')\n"
                "        break\n"
                "    except Exception:\n"
                "        continue\n"
            );
        }
        PyErr_SetRaisedException(held);
    }
    PyErr_Print();
}
"""

#: The internal name of the module body's function. Deliberately not spellable
#: as a Python identifier a program would choose, because the renderer tells
#: the entry body from the program's own functions by this name.
_ENTRY_BODY = "_py2bin_entry_body"


class CApiEmitError(SyntaxError):
    """Something in the program has no C-API translation here yet."""


#: The entry points the generated C declares for itself. Nothing includes
#: Python.h: its headers carry function-pointer typedefs and macros that this
#: project's C front end does not parse, and the declarations that are
#: genuinely needed fit in a dozen lines.
_PROTOTYPES = """\
/* Generated by py2bin: Python -> C driving the CPython C API. */
typedef struct _object PyObject;

extern void Py_Initialize(void);
extern void Py_Finalize(void);
extern PyObject *PyLong_FromLongLong(long long value);
extern long long PyLong_AsLongLong(PyObject *value);
extern PyObject *PyUnicode_FromString(const char *text);
extern PyObject *PyNumber_Add(PyObject *left, PyObject *right);
extern PyObject *PyNumber_InPlaceAdd(PyObject *left, PyObject *right);
extern PyObject *PyNumber_InPlaceSubtract(PyObject *left, PyObject *right);
extern PyObject *PyNumber_InPlaceMultiply(PyObject *left, PyObject *right);
extern PyObject *PyNumber_InPlaceTrueDivide(PyObject *left, PyObject *right);
extern PyObject *PyNumber_InPlaceFloorDivide(PyObject *left, PyObject *right);
extern PyObject *PyNumber_InPlaceRemainder(PyObject *left, PyObject *right);
extern PyObject *PyNumber_InPlacePower(PyObject *left, PyObject *right, PyObject *modulus);
extern PyObject *PyNumber_InPlaceLshift(PyObject *left, PyObject *right);
extern PyObject *PyNumber_InPlaceRshift(PyObject *left, PyObject *right);
extern PyObject *PyNumber_InPlaceAnd(PyObject *left, PyObject *right);
extern PyObject *PyNumber_InPlaceOr(PyObject *left, PyObject *right);
extern PyObject *PyNumber_InPlaceXor(PyObject *left, PyObject *right);
extern PyObject *PyNumber_MatrixMultiply(PyObject *left, PyObject *right);
extern PyObject *PyNumber_InPlaceMatrixMultiply(PyObject *left, PyObject *right);
extern PyObject *PyNumber_Subtract(PyObject *left, PyObject *right);
extern PyObject *PyNumber_Multiply(PyObject *left, PyObject *right);
extern PyObject *PyNumber_TrueDivide(PyObject *left, PyObject *right);
extern PyObject *PyObject_RichCompare(PyObject *left, PyObject *right, int op);
extern int PyObject_IsTrue(PyObject *value);
extern int PyObject_IsInstance(PyObject *value, PyObject *kind);
extern PyObject *PyObject_Str(PyObject *value);
extern PyObject *PySys_GetObject(const char *name);
extern int PyFile_WriteObject(PyObject *value, PyObject *stream, int flags);
extern int PyFile_WriteString(const char *text, PyObject *stream);
extern void Py_IncRef(PyObject *object);
extern void Py_DecRef(PyObject *object);
extern int PyRun_SimpleString(const char *command);
extern PyObject *PyImport_ImportModule(const char *name);
extern PyObject *PyImport_AddModule(const char *name);
extern PyObject *PyObject_GetAttrString(PyObject *object, const char *name);
extern int PyObject_SetAttrString(PyObject *object, const char *name, PyObject *value);
extern PyObject *PyObject_GetAttr(PyObject *object, PyObject *name);
extern int PyObject_SetAttr(PyObject *object, PyObject *name, PyObject *value);
extern PyObject *PyUnicode_InternFromString(const char *text);
extern PyObject *PyInstanceMethod_New(PyObject *function);
extern int PyObject_RichCompareBool(PyObject *a, PyObject *b, int op);
extern PyObject *PyUnicode_Join(PyObject *separator, PyObject *pieces);
extern PyObject *PyObject_Repr(PyObject *value);
extern int PySequence_Check(PyObject *object);
extern PyObject *PySequence_GetItem(PyObject *object, long long index);
extern PyObject *PyObject_VectorcallMethod(
    PyObject *name, PyObject **arguments, long long count,
    PyObject *keyword_names);
extern PyObject *PyObject_CallNoArgs(PyObject *callable);
extern PyObject *PyObject_CallOneArg(PyObject *callable, PyObject *argument);
extern long long PyObject_Size(PyObject *object);
extern PyObject *PyObject_Call(PyObject *callable, PyObject *args, PyObject *kwargs);
extern PyObject *PyObject_Vectorcall(
    PyObject *callable, PyObject **arguments, long long count,
    PyObject *keyword_names);
extern PyObject *PyTuple_New(long long length);
extern PyObject *PyTuple_GetItem(PyObject *tuple, long long index);
typedef PyObject *(*PyCFunctionFastWithKeywords)(
    PyObject *self, PyObject **arguments, long long count,
    PyObject *keyword_names);
struct PyMethodDef {
    const char *ml_name;
    PyCFunctionFastWithKeywords ml_meth;
    int ml_flags;
    const char *ml_doc;
};
extern PyObject *PyCFunction_New(void *table, PyObject *self);
extern int PyTuple_SetItem(PyObject *tuple, long long index, PyObject *value);
extern PyObject *PyFloat_FromDouble(double value);
extern PyObject *PyNumber_Remainder(PyObject *left, PyObject *right);
extern PyObject *PyNumber_Or(PyObject *left, PyObject *right);
extern PyObject *PyNumber_And(PyObject *left, PyObject *right);
extern PyObject *PyNumber_Xor(PyObject *left, PyObject *right);
extern PyObject *PyNumber_Lshift(PyObject *left, PyObject *right);
extern PyObject *PyNumber_Rshift(PyObject *left, PyObject *right);
extern int PyObject_DelItem(PyObject *container, PyObject *key);
extern PyObject *PyNumber_FloorDivide(PyObject *left, PyObject *right);
extern PyObject *PyNumber_Power(PyObject *base, PyObject *exp, PyObject *mod);
extern PyObject *PyDict_New(void);
extern int PyDict_SetItem(PyObject *mapping, PyObject *key, PyObject *value);
extern PyObject *PyTuple_Pack(long long length, PyObject *a, PyObject *b);
extern int PySequence_Contains(PyObject *container, PyObject *value);
extern int PyErr_ExceptionMatches(PyObject *exception);
extern void PyErr_SetObject(PyObject *exception, PyObject *value);
extern PyObject *PySlice_New(PyObject *start, PyObject *stop, PyObject *step);
extern void PyErr_Clear(void);
extern PyObject *PyErr_GetRaisedException(void);
extern void PyErr_SetRaisedException(PyObject *exception);
extern PyObject *PyErr_GetHandledException(void);
extern void PyErr_SetHandledException(PyObject *exception);
extern PyObject *PyBytes_FromStringAndSize(const char *text, long long length);
extern PyObject *PyNumber_Negative(PyObject *value);
extern PyObject *PyNumber_Positive(PyObject *value);
extern PyObject *PyNumber_Invert(PyObject *value);
extern int Py_EnterRecursiveCall(const char *where);
extern PyObject *PyLong_FromString(const char *text, void *end, int base);
extern PyObject *PyUnicode_DecodeUTF8(
    const char *text, long long length, void *errors);
extern void Py_LeaveRecursiveCall(void);
extern PyObject *PyObject_GetItem(PyObject *container, PyObject *key);
extern int PyObject_SetItem(PyObject *container, PyObject *key, PyObject *value);
extern PyObject *PyList_New(long long length);
extern int PyList_SetItem(PyObject *list, long long index, PyObject *value);
extern PyObject *PyUnicode_Concat(PyObject *left, PyObject *right);
extern PyObject *PyObject_Format(PyObject *value, PyObject *spec);
extern int PyList_Append(PyObject *list, PyObject *value);
extern PyObject *PyObject_GetIter(PyObject *object);
extern PyObject *PyIter_Next(PyObject *iterator);
extern PyObject *PyErr_Occurred(void);
extern void PyErr_Print(void);
extern int exit(int status);
"""

#: CPython's comparison opcodes, in the order Py_LT..Py_GE.
_COMPARISONS = {
    ast.Lt: 0,
    ast.LtE: 1,
    ast.Eq: 2,
    ast.NotEq: 3,
    ast.Gt: 4,
    ast.GtE: 5,
}

#: Methods Python wraps for you, whatever the class body says. `__new__` is
#: called before there is an instance, and the other two are asked of the
#: class itself.
_IMPLICITLY_WRAPPED = {
    "__new__": "staticmethod",
    "__init_subclass__": "classmethod",
    "__class_getitem__": "classmethod",
}

#: The in-place form of each operator, for `x += y` and its relatives. The
#: object gets to answer for itself: a list extends, a number rebuilds.
_IN_PLACE = {
    ast.Add: "PyNumber_InPlaceAdd",
    ast.Sub: "PyNumber_InPlaceSubtract",
    ast.Mult: "PyNumber_InPlaceMultiply",
    ast.Div: "PyNumber_InPlaceTrueDivide",
    ast.FloorDiv: "PyNumber_InPlaceFloorDivide",
    ast.Mod: "PyNumber_InPlaceRemainder",
    ast.Pow: "PyNumber_InPlacePower",
    ast.LShift: "PyNumber_InPlaceLshift",
    ast.RShift: "PyNumber_InPlaceRshift",
    ast.BitAnd: "PyNumber_InPlaceAnd",
    ast.BitOr: "PyNumber_InPlaceOr",
    ast.BitXor: "PyNumber_InPlaceXor",
    ast.MatMult: "PyNumber_InPlaceMatrixMultiply",
}

_BINARY = {
    ast.BitOr: "PyNumber_Or",
    ast.BitAnd: "PyNumber_And",
    ast.BitXor: "PyNumber_Xor",
    ast.LShift: "PyNumber_Lshift",
    ast.RShift: "PyNumber_Rshift",
    ast.Add: "PyNumber_Add",
    ast.MatMult: "PyNumber_MatrixMultiply",
    ast.Sub: "PyNumber_Subtract",
    ast.Mult: "PyNumber_Multiply",
    ast.Div: "PyNumber_TrueDivide",
    ast.Mod: "PyNumber_Remainder",
    ast.FloorDiv: "PyNumber_FloorDivide",
}


def _text_signature(name: str, arguments: ast.arguments) -> str:
    """The doc string CPython reads `__text_signature__` out of.

    A compiled function is a builtin function object, and a builtin carries no
    signature unless its doc begins with one in this exact shape: the name, the
    parameters in brackets, then a line of two dashes.

    A default that is a literal is written as itself, so `def f(a, b=1)` reads
    back as `(a, b=1)` rather than the `(a, b=None)` every default used to be
    spelled as. Anything else - a name, a call, a list - has no spelling that
    could be parsed back, and stays `None`: the shape of the call is right
    either way, and that is what most callers are asking for.
    """

    def spelled(default: ast.expr | None) -> str:
        if isinstance(default, ast.Constant) and (
            default.value is None
            or isinstance(default.value, (bool, int, float, complex, str, bytes))
        ):
            return repr(default.value)
        return "None"

    parts: list[str] = []
    positional = [*arguments.posonlyargs, *arguments.args]
    required = len(positional) - len(arguments.defaults)
    for offset, argument in enumerate(positional):
        parts.append(
            argument.arg
            if offset < required
            else f"{argument.arg}={spelled(arguments.defaults[offset - required])}"
        )
        if arguments.posonlyargs and offset == len(arguments.posonlyargs) - 1:
            parts.append("/")
    if arguments.vararg:
        parts.append(f"*{arguments.vararg.arg}")
    elif arguments.kwonlyargs:
        parts.append("*")
    for offset, argument in enumerate(arguments.kwonlyargs):
        parts.append(
            argument.arg
            if arguments.kw_defaults[offset] is None
            else f"{argument.arg}={spelled(arguments.kw_defaults[offset])}"
        )
    if arguments.kwarg:
        parts.append(f"**{arguments.kwarg.arg}")
    return f"{name}({', '.join(parts)})\n--\n\n"


def _c_bytes(data: bytes) -> str:
    """A C string literal holding exactly these bytes.

    Everything outside printable ASCII goes in as an octal escape, so the
    literal itself stays ASCII and the runtime still sees the bytes.
    """

    out = ['"']
    for byte in data:
        if byte == 0x22:
            out.append('\\"')
        elif byte == 0x5C:
            out.append("\\\\")
        elif 0x20 <= byte < 0x7F:
            out.append(chr(byte))
        else:
            out.append(f"\\{byte:03o}")
    out.append('"')
    return "".join(out)


def _bound_names(target: ast.expr) -> list[str]:
    """Every name one assignment target binds, in the order they appear.

    Only names being stored. `for d[k] in ...` mentions both `d` and `k`, and
    binds neither: they are read to find where the item goes. Counting them
    would give a comprehension its own `d`, shadowing the one holding the
    dictionary being written to.
    """
    found: list[str] = []
    for node in ast.walk(target):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id not in found:
                found.append(node.id)
    return found


def _c_string(text: str) -> str:
    """``text`` as a C string literal, escaped so the C parser reads it back."""

    out = ['"']
    for character in text:
        if character == "\\":
            out.append("\\\\")
        elif character == '"':
            out.append('\\"')
        elif character == "\n":
            out.append("\\n")
        elif character == "\t":
            out.append("\\t")
        elif " " <= character <= "~":
            out.append(character)
        else:
            # Anything outside printable ASCII goes in as its UTF-8 bytes, so
            # the literal stays ASCII and the runtime still sees the text.
            out.extend(f"\\{byte:03o}" for byte in character.encode("utf-8"))
    out.append('"')
    return "".join(out)


#: Why a protected region is being left. `finally` runs the same clause for
#: every one of them and then does what the reason says.
_RETURNING = 1
_PROPAGATING = 2
_BREAKING = 3
_CONTINUING = 4


class _Protected:
    """One `finally` clause and the storage its region leaves through."""

    def __init__(self, label: str, why: str, answer: str, loop_depth: int) -> None:
        self.label = label
        #: The C int holding why the region is being left.
        self.why = why
        #: Where a `return` inside the region puts its value until the clause
        #: has run.
        self.answer = answer
        #: How many loops were open when the region started. A `break` inside
        #: it belongs to this clause only when no loop opened since - one that
        #: did is a loop the break can leave directly.
        self.loop_depth = loop_depth
        #: Which reasons the region actually uses, so the clause tests for
        #: those and no others. `break;` written where no loop encloses it is
        #: not valid C.
        self.reasons: set[int] = set()


#: Raising an unbound-name error, written into the generated C once. Each site
#: is then a single call rather than a dozen lines of construction, which for a
#: module with a thousand of them is the difference between a third more C and
#: none. Every early return here leaves an exception set - a failed attribute
#: lookup sets AttributeError, a failed allocation sets MemoryError - so the
#: caller never answers NULL with nothing pending.
_UNBOUND_HELPER = """
static void _py2bin_unbound(int local, PyObject *message, PyObject *named) {
    /* The strings arrive already built. A `const char *` parameter could not
       be passed on: this C front end materializes a literal in the image and
       will not take a runtime pointer, whose lifetime it cannot verify. */
    if (!message || !named) { Py_DecRef(message); Py_DecRef(named); return; }
    PyObject *kind = local
        ? PyObject_GetAttrString(_py2bin_builtins, "UnboundLocalError")
        : PyObject_GetAttrString(_py2bin_builtins, "NameError");
    if (!kind) { Py_DecRef(message); Py_DecRef(named); return; }
    PyObject *raised = PyObject_CallOneArg(kind, message);
    Py_DecRef(message);
    if (!raised) { Py_DecRef(kind); Py_DecRef(named); return; }
    /* `name` is where CPython's display looks for "Did you mean". */
    PyObject_SetAttrString(raised, "name", named);
    Py_DecRef(named);
    PyErr_SetObject(kind, raised);
    Py_DecRef(kind);
    Py_DecRef(raised);
}
"""


class _Function:
    """One Python function being written out as a C function."""

    def __init__(
        self,
        name: str,
        parameters: tuple[str, ...],
        defaults: int = 0,
        captures: tuple[str, ...] = (),
        closure: bool = False,
    ) -> None:
        self.name = name
        self.parameters = parameters
        #: Names read from an enclosing function. A closure receives them in
        #: the tuple CPython hands back as `self`, which is how a compiled C
        #: function carries state that a plain C function cannot.
        self.captures = captures
        #: True for a nested `def` or a `lambda`. It is written with the
        #: `(self, args)` shape CPython calls, rather than as a direct call.
        self.closure = closure
        #: True once something in the body takes the failure path, which is
        #: what puts the `_unwind` label at the end of the C function.
        self.unwinds = False
        #: `arity -> C name` for the argument arrays a call passes. One per
        #: arity per function is enough: the arguments are all computed before
        #: any of them is stored, so a nested call has finished with the array
        #: before the outer one begins to fill it.
        self.argument_arrays: dict[int, str] = {}
        #: Names the body's own statements bind, *not* counting the parameter
        #: list. `shadows` is this plus the parameters; kept apart because the
        #: two answer different questions - what the scope binds at all, and
        #: what the body itself rebinds.
        self.body_binds: set[str] = set()
        #: Every name this body binds anywhere in itself. A module-level
        #: function is called as a direct C call, which is only right when the
        #: name at the call site is that function - a scope with its own
        #: binding of the same spelling means it is not, and calling the C
        #: function anyway answered with the wrong one, silently.
        self.shadows: set[str] = set()
        #: Names this body declared `global`. They read and write the module's
        #: own storage rather than a local of the same spelling.
        self.module_names: set[str] = set()
        #: Names an unconditional statement of this body has already bound, so
        #: a later read of one cannot find its slot empty and needs no test.
        self.certain: set[str] = set()
        #: Label names must stay unique for the whole function, so this counter
        #: is never wound back the way the temporaries are.
        self.labels = 0
        #: How many trailing parameters have defaults. A call that leaves one
        #: out passes NULL and the body fills it in.
        self.defaults = defaults
        self.locals: list[str] = []
        #: Where this function's defaults were put when its `def` ran, one C
        #: expression per default, in the order they are declared - positional
        #: first, then keyword-only. Empty for a function whose defaults are
        #: still filled in the callee.
        self.default_cells: list[str] = []
        #: Locals bound here and never read back. They share one slot, because
        #: nothing can tell which of them it holds - see `write_only_locals`.
        self.write_only: set[str] = set()
        self.body: list[str] = []
        self.temporaries = 0
        #: Locals held as a machine integer whenever they contain one. Each
        #: becomes three C variables rather than one - see `narrow_slots`.
        self.unboxed: set[str] = set()
        #: Locals held as a machine double whenever they contain one. Held in
        #: the same three-variable shape as `unboxed`, with a `double` in place
        #: of the `long long`. The two sets never overlap: a name is narrowed
        #: one way or the other, never both.
        self.doubles: set[str] = set()
        #: Locals that always hold an exact `list`, and exact `dict`. Decided
        #: from their bindings alone - see `capi_exact` - so no run-time
        #: guard is ever emitted for them.
        self.exact_lists: set[str] = set()
        self.exact_dicts: set[str] = set()
        self.exact_strs: set[str] = set()
        #: Counter for the `long long` scratch the fast path computes in, wound
        #: back at the end of each statement as the object temporaries are.
        self.machines = 0
        #: The same, for the `double` scratch.
        self.reals = 0


class CApiEmitter:
    """Walk a module and write the C that drives the interpreter."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.functions: list[_Function] = []
        self.known_functions: dict[str, int] = {}
        #: The parameters of each of those, in order, and how many of them are
        #: positional-only. A call that names an argument is placed against
        #: these at compile time rather than at run time, so `f(a, step=1)`
        #: reaches the same direct C call `f(a, 1)` does.
        self.known_parameters: dict[str, tuple[tuple[str, ...], int]] = {}
        #: Static slots holding what each module-level `def`'s defaults were
        #: evaluated to, as (base name, how many).
        self.default_stores: list[tuple[str, int]] = []
        #: True when this module asks for its globals as a dictionary -
        #: `globals()`, or an `eval`/`exec` that would read one. The names
        #: then live in the module's own `__dict__` as well as in their C
        #: slots, and are *read* from there, so a write through the
        #: dictionary is seen by the program and the other way round.
        self.globals_in_dict = False
        #: While a class body is being written, the namespace it is going
        #: into and the keys already in it. A decorator and an attribute's
        #: value are evaluated *in* the class body, so they can see what the
        #: body has bound so far - which is what `@v.setter` is reading.
        #: A method body sees none of it, so this is only set around the
        #: pieces that run as the class is built.
        self.class_scope: list[tuple[str, set[str]]] = []
        #: The class scope a comprehension's first iterable is evaluated in,
        #: which is the one around it rather than the comprehension's own.
        self.first_iterable_scope: list[tuple[str, set[str]]] = []
        # Where each of those `def`s sits in the module body, and how far
        # through that body execution has got. A direct call is only correct
        # once the `def` has run: `print(later(3))` above `def later(...)` is
        # a NameError in Python, and used to answer 21 here.
        self.defined_at: dict[str, int] = {}
        self.reached = -1
        #: Names any walrus in the module assigns. A slot one of these
        #: names cannot be borrowed from: a walrus is the one thing that
        #: writes a slot in the middle of an expression, so a borrowed
        #: earlier read of it could be stale by the time it is used.
        self.walrus_names: set[str] = set()
        #: True when the program reads `sys.argv`, which is what decides
        #: whether the recovery for it is emitted at all.
        self.reads_argv = False
        #: Serial for the unique spelling a counted comprehension gives
        #: its target. Never wound back: two comprehensions in one
        #: function must not share slots.
        self.comp_serial = 0
        #: The unique spellings counted comprehensions made up. These are the
        #: emitter's own names: nothing else binds them, they have no module
        #: storage even at module level, and no scope can shadow them - so
        #: `is_unboxed` says yes to them wherever they appear, where a name
        #: the program wrote is refused at module level because its storage
        #: there is the module's `g_` slot.
        self.synthetic: set[str] = set()
        self.current: _Function | None = None
        #: Names bound at module level. They become file-scope statics so a
        #: function body can see them, which is what a Python global is.
        self.globals: set[str] = set()
        #: Every module-level slot across every linked module, already
        #: prefixed. `globals` holds the *current* module's bare names, which
        #: is what the name lookups ask about; this is what render declares.
        self.declared: set[str] = set()
        #: True while the module body itself is being written.
        self.at_module_level = False
        #: What distinguishes one module's C names from another's when several
        #: are linked into one image. Empty for the entry module, so a
        #: single-module program's C reads exactly as it did.
        self.prefix = ""
        #: `(python name, C key)` for each module compiled alongside the entry,
        #: in the order their bodies must run.
        self.linked: list[tuple[str, str]] = []
        #: The bare module-level names each linked module binds, so its module
        #: object can be given them once its body has run.
        self.module_globals: dict[str, set[str]] = {}
        #: Directories put on `sys.path` before anything runs. The interpreter
        #: a compiled program links is whichever one the build machine had, and
        #: its search path knows nothing of where this program's dependencies
        #: were installed - so a binary that is otherwise complete stops at
        #: `ModuleNotFoundError` for a package that is plainly present.
        self.extra_paths: list[str] = []
        #: The package the module being compiled belongs to, which is what a
        #: relative import counts its dots from.
        self.module_package = ""
        #: Builtin names this module binds for itself. A counted loop is only
        #: written where `range` is certainly the interpreter's own.
        self.shadowed_builtins: set[str] = set()
        #: True when the program should write a crash.txt as well as printing.
        self.crash_log = False
        #: Non-zero while emitting the slow arm of a fast path, where the
        #: fast path must not be offered a second time.
        self.boxing = 0
        #: The label a failing C-API call jumps to. Empty outside a `try`,
        #: where a failure ends the process instead.
        self.handlers: list[str] = []
        #: One entry per nested function or lambda: the C function's name and
        #: the name Python knows it by. They become a file-scope array of
        #: PyMethodDef filled in at startup, because the C front end does not
        #: initialise a file-scope struct.
        self.method_table: list[tuple[str, str]] = []
        #: The body of the scope being written, for the late-binding check.
        self.scope: list[ast.stmt] = []
        #: The exception each enclosing `except` clause is handling, innermost
        #: last. A bare `raise` sets the last one again.
        self.handling: list[str] = []
        #: One entry per enclosing `finally`, innermost last. Every way out of
        #: the region it protects goes through it first, so `return`, `break`
        #: and `continue` record why they are leaving and jump to it rather
        #: than leaving directly.
        self.finallys: list[_Protected] = []
        #: How many loops enclose what is being written.
        self.loop_depth = 0
        #: The flag each enclosing loop uses to record that a `break` left it,
        #: innermost last, or None where the loop has no `else` to guard.
        self.loop_flags: list[str | None] = []
        #: How deeply statements are nested in the body being written. Zero
        #: between the statements of a body, which is where a temporary slot
        #: stops being live.
        self.depth = 0
        #: Name to C slot, innermost last, for scopes that are not a function:
        #: a comprehension has its own, so its target must not be the
        #: enclosing name of the same spelling.
        self.shadowed: list[dict[str, str]] = []
        #: True while writing a body that counts its depth. The wrapper for a
        #: module-level `def` does not: it delegates straight to the real
        #: function, which counts for itself, and counting twice would halve
        #: the depth a program may reach.
        self.guards_recursion = False
        #: The scopes enclosing what is being written, as `(name, is a
        #: function)`. It is what builds the qualified name Python puts in a
        #: TypeError - `outer.<locals>.one`, `C.method` - which is the only
        #: part of those messages a compiled function cannot read off itself.
        self.scope_path: list[tuple[str, bool]] = []
        #: `(class name, first parameter)` for the method being written,
        #: innermost last. It is what a zero-argument `super()` needs and has
        #: no other way to reach.
        self.methods_of: list[tuple[str, str]] = []
        #: Every builtin the emitter itself asks for. There are about fifty
        #: distinct ones and thousands of uses, and each use was a dictionary
        #: lookup on a name that cannot change; they are fetched once at
        #: start-up into file-scope slots instead.
        self.cached_builtins: dict[str, str] = {}
        #: Every attribute name the program mentions, interned once at start-up.
        #: `PyObject_GetAttrString` looks harmless and is not: it builds a str
        #: from the char* and hashes it on *every* call, so an attribute read
        #: in a loop pays an allocation and a hash that the name being constant
        #: makes entirely pointless. Interning moves both to start-up, and an
        #: interned str carries its hash, so the lookup that remains is the
        #: dictionary probe and nothing else.
        self.interned_names: dict[str, str] = {}
        #: One tuple of keyword names per distinct call shape, built at
        #: start-up. A call site's names never change, so building the
        #: tuple per call was building the same object over and over.
        self.keyword_tuples: dict[tuple[str, ...], str] = {}
        #: Every literal the program mentions, built once at start-up. A
        #: constant was rebuilt at each execution, so a string inside a loop
        #: was decoded from UTF-8 and allocated on every iteration and an
        #: integer literal was allocated on every one - for a value that,
        #: being a literal, cannot have changed. Keyed by type as well as
        #: value, so `1`, `1.0` and `True` stay three different objects.
        self.pooled: dict[tuple[str, object], str] = {}
        #: True once a name read needs the unbound-name helper, which is then
        #: written into the C once rather than at each of a thousand sites.
        self.needs_unbound = False
        #: Module names an unconditional top-level statement binds. A function
        #: body runs after the module body in every arrangement this compiles,
        #: so one of these is bound by the time any function can read it.
        self.certain_globals: set[str] = set()
        #: Where in the module body each of those became certain. A global is
        #: only certainly bound *after* the statement that binds it has run,
        #: and `print(y)` above `y = 5` is a NameError in Python - so without
        #: the position this skipped the NULL test and handed the program a
        #: raw NULL, which printed as `<NULL>` and would crash anywhere less
        #: forgiving than `print`.
        self.certain_at: dict[str, int] = {}
        #: ``(python name, method-table index)`` for each module-level `def`,
        #: which also gets a Python callable so the name can be *used* and
        #: not only called - as a sort key, or with `*args` spread into it.
        self.value_functions: list[tuple[str, int]] = []

    # --- helpers ---------------------------------------------------------

    def fail(self, node: ast.AST, message: str) -> CApiEmitError:
        line = getattr(node, "lineno", 1)
        column = getattr(node, "col_offset", 0) + 1
        return CApiEmitError(f"{self.path}:{line}:{column}: {message}")

    def emit(self, line: str, indent: int = 1) -> None:
        assert self.current is not None
        self.current.body.append("    " * indent + line)

    def _report(self) -> str:
        """How this program reports a fatal error."""

        return "_py2bin_crash_report()" if self.crash_log else "PyErr_Print()"

    def failure(self) -> str:
        """What to do where a C-API call has just failed.

        Inside a `try`, jump to its handler. Inside a function with no handler
        in reach, hand the failure back to the caller the way every C-API
        function does - answer NULL with the exception still set - so a `try`
        around the *call* catches what the body raised. Only at module level,
        where there is no caller left, is an uncaught exception printed and the
        process ended, which is what the interpreter does with one.
        """

        assert self.current is not None
        if self.handlers:
            return f"goto {self.handlers[-1]};"
        if self.current.name == _ENTRY_BODY:
            # Py_Finalize before leaving, or everything already printed is
            # lost: sys.stdout is buffered inside the interpreter and exit()
            # does not run its shutdown. A program that printed and then
            # raised showed nothing at all.
            report = "_py2bin_crash_report()" if self.crash_log else "PyErr_Print()"
            return f"{report}; Py_Finalize(); exit(1);"
        self.current.unwinds = True
        return "goto _unwind;"

    def checked(self, target: str, indent: int) -> str:
        """Stop if the C-API call failed.

        A C-API function answers NULL and leaves an exception set. Letting that
        NULL travel is how `1 + "x"` came to print `<NULL>` and exit 0 where
        CPython raises TypeError and exits 1. Printing the exception and
        leaving with status 1 is what the interpreter does with one nothing
        catches, and nothing here can catch one yet.
        """

        self.emit(f"if (!{target}) {{ {self.failure()} }}", indent)
        return target

    def temporary(self) -> str:
        """A slot for one intermediate value.

        The count is wound back at the end of each top-level statement, so the
        same slot serves every statement that needs one - a temporary is dead
        once the statement that made it has finished. A fresh slot per
        subexpression made the entry frame of a 7,000-line module larger than
        py2bin's whole stack allowance.
        """

        assert self.current is not None
        self.current.temporaries += 1
        name = f"_t{self.current.temporaries}"
        if name not in self.current.locals:
            self.current.locals.append(name)
        return name

    def note_global(self, name: str) -> str:
        """Record a module-level name, for this module and for the image."""

        self.globals.add(name)
        self.declared.add(self.prefix + name)
        return f"g_{self.prefix}{name}"

    def declare(self, name: str) -> str:
        """The C name for a Python local, declared on first use.

        A parameter that the body assigns to is the same storage, not a new
        local - Python rebinds the parameter. The body owns its parameters
        (they are incremented on entry), so overwriting one releases what it
        held exactly as overwriting any other name does.
        """

        assert self.current is not None
        for scope in reversed(self.shadowed):
            if name in scope:
                return scope[name]
        if name in self.current.module_names:
            # Declared `global` here, so the assignment lands in the module's
            # storage and every other scope sees it.
            self.note_global(name)
            return f"g_{self.prefix}{name}"
        if name in self.current.parameters:
            return f"p_{name}"
        if name in self.current.write_only:
            # Nothing reads any of these, so nothing can tell that they share
            # a slot. Rebinding still releases what was there, which is what
            # keeps the values from piling up.
            if _UNREAD not in self.current.locals:
                self.current.locals.append(_UNREAD)
            return _UNREAD
        if self.at_module_level:
            # The module body's names are the program's globals, so they live
            # at file scope where a function can reach them.
            self.note_global(name)
            return f"g_{self.prefix}{name}"
        c_name = f"v_{name}"
        if c_name not in self.current.locals:
            self.current.locals.append(c_name)
        return c_name

    # --- machine integers ---------------------------------------------------
    #
    # A local the analysis picked out is three C variables rather than one:
    #
    #     long long n_x;   the value, when it is a machine integer
    #     PyObject  *v_x;  the value, when it is an object
    #     int        s_x;  1 when the first of those holds it
    #
    # with the invariant that at most one of `v_x` and `s_x` is set. Both clear
    # is the third state the name can be in - never assigned - which is what
    # the unbound test looks for.
    #
    # A name only ever *becomes* a machine integer from somewhere the value is
    # certainly an `int`: a `range` counter, an integer literal, or arithmetic
    # on values already narrow. Nothing is ever narrowed by inspecting an
    # object at run time. That is what keeps the representation honest without
    # a type check: `PyLong_AsLongLong` would happily convert `True` to 1, and
    # a `True` that came back out as `1` would be a wrong program, not a fast
    # one.

    def is_unboxed(self, name: str) -> bool:
        """True when this name is held in the three-variable form."""

        if self.current is None or name not in self.current.unboxed:
            return False
        if name in self.synthetic:
            # The emitter's own spelling: no module storage, no shadowing.
            return True
        # A name whose storage is the module's is not this body's to hold in
        # registers, whatever the analysis said: every other reference to it
        # goes to `g_`, and two storages for one name is one too many.
        if self.at_module_level or name in self.current.module_names:
            return False
        # A comprehension puts its own scope in front; a name of that scope is
        # a different name that happens to be spelled the same.
        return not any(name in scope for scope in self.shadowed)

    def narrow_slots(self, name: str) -> tuple[str, str, str]:
        """Declare the three variables, and answer with their C names."""

        assert self.current is not None
        value, obj, flag = f"n_{name}", f"v_{name}", f"s_{name}"
        for spelling in (f"long long {value}", obj, f"int {flag}"):
            if spelling not in self.current.locals:
                self.current.locals.append(spelling)
        return value, obj, flag

    def machine_slot(self) -> str:
        """Scratch for one intermediate machine integer."""

        assert self.current is not None
        self.current.machines += 1
        name = f"_m{self.current.machines}"
        if f"long long {name}" not in self.current.locals:
            self.current.locals.append(f"long long {name}")
        return name

    def store_object(self, name: str, value: str, indent: int) -> None:
        """Bind an unboxed name to an object, consuming the reference."""

        held, obj, flag = self.narrow_slots(name)
        self.emit(f"if ({obj}) Py_DecRef({obj});", indent)
        self.emit(f"{obj} = {value};", indent)
        self.emit(f"{flag} = 0;", indent)

    def store_machine(self, name: str, value: str, indent: int) -> None:
        """Bind an unboxed name to a machine integer.

        Nothing is allocated. This is the whole point of the exercise: an
        accumulator around a loop costs an add and a store, where the same
        line through `PyNumber_Add` costs a call, an allocation and two
        reference-count updates.
        """

        held, obj, flag = self.narrow_slots(name)
        self.emit(f"if ({obj}) {{ Py_DecRef({obj}); {obj} = 0; }}", indent)
        self.emit(f"{held} = {value};", indent)
        self.emit(f"{flag} = 1;", indent)

    def unbound_test(self, name: str, indent: int) -> None:
        """Refuse to read an unboxed name that nothing has written."""

        held, obj, flag = self.narrow_slots(name)
        self.needs_unbound = True
        message = (
            f"cannot access local variable {name!r} where it is not "
            "associated with a value"
        )
        self.emit(
            f"if (!{obj} && !{flag}) {{ _py2bin_unbound(1, "
            f"PyUnicode_FromString({_c_string(message)}), "
            f"PyUnicode_FromString({_c_string(name)})); {self.failure()} }}",
            indent,
        )

    def read_unboxed(self, node: ast.Name, indent: int) -> str:
        """An unboxed name as an owned reference, boxing it if it has to.

        Every use that is not arithmetic comes through here, which is what
        lets the representation be invisible to the rest of the emitter.
        """

        assert self.current is not None
        held, obj, flag = self.narrow_slots(node.id)
        target = self.temporary()
        if node.id not in self.current.certain:
            self.unbound_test(node.id, indent)
        self.emit(f"if ({flag}) {{", indent)
        self.emit(f"{target} = PyLong_FromLongLong({held});", indent + 1)
        self.emit(f"if (!{target}) {{ {self.failure()} }}", indent + 1)
        self.emit(f"}} else {{", indent)
        self.emit(f"Py_IncRef({obj});", indent + 1)
        self.emit(f"{target} = {obj};", indent + 1)
        self.emit("}", indent)
        return target

    # --- machine doubles ------------------------------------------------------
    #
    # The same three variables as an integer, with a `double` in place of the
    # `long long`. Everything said about the integer form holds here, and two
    # things are easier: an overflowing double is an infinity, which is the
    # answer Python gives, so nothing has to be checked; and there is no
    # `bool` masquerading as a `float` the way `True` masquerades as `1`.
    #
    # Only division has anything to refuse, and it refuses zero: Python raises
    # where C answers infinity.

    def is_double(self, name: str) -> bool:
        """True when this name is held as a machine double."""

        if self.current is None or name not in self.current.doubles:
            return False
        if self.at_module_level or name in self.current.module_names:
            return False
        return not any(name in scope for scope in self.shadowed)

    def _exactly(self, node: ast.expr, wanted: set[str]) -> bool:
        """True when this is a Name the analysis put in `wanted`, here."""

        if self.boxing or self.current is None or self.at_module_level:
            # Not at module level: a nested function may rebind a module name
            # through `global`, which the module body's own analysis never
            # sees.
            return False
        if not isinstance(node, ast.Name) or node.id not in wanted:
            return False
        if node.id in self.current.module_names:
            return False
        return not any(node.id in scope for scope in self.shadowed)

    def is_exact_list(self, node: ast.expr) -> bool:
        assert self.current is not None
        return self._exactly(node, self.current.exact_lists)

    def is_exact_dict(self, node: ast.expr) -> bool:
        assert self.current is not None
        return self._exactly(node, self.current.exact_dicts)

    def is_exact_str(self, node: ast.expr) -> bool:
        """Whether this expression is certainly an exact `str`.

        A literal and an f-string are, whatever the surrounding code does;
        beyond those it is a name the analysis has shown holds nothing else.
        """

        assert self.current is not None
        if isinstance(node, ast.JoinedStr):
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            # `a + b + c` is `(a + b) + c`, so without this only the innermost
            # pair was claimed and the rest fell back. Concatenating two exact
            # strings answers an exact string, so the property composes.
            return self.is_exact_str(node.left) and self.is_exact_str(node.right)
        return self._exactly(node, self.current.exact_strs)

    def double_slots(self, name: str) -> tuple[str, str, str]:
        """Declare the three variables, and answer with their C names."""

        assert self.current is not None
        value, obj, flag = f"d_{name}", f"v_{name}", f"s_{name}"
        for spelling in (f"double {value}", obj, f"int {flag}"):
            if spelling not in self.current.locals:
                self.current.locals.append(spelling)
        return value, obj, flag

    def real_slot(self) -> str:
        """Scratch for one intermediate machine double."""

        assert self.current is not None
        self.current.reals += 1
        name = f"_r{self.current.reals}"
        if f"double {name}" not in self.current.locals:
            self.current.locals.append(f"double {name}")
        return name

    def store_double_object(self, name: str, value: str, indent: int) -> None:
        """Bind a double-held name to an object, consuming the reference."""

        _held, obj, flag = self.double_slots(name)
        self.emit(f"if ({obj}) Py_DecRef({obj});", indent)
        self.emit(f"{obj} = {value};", indent)
        self.emit(f"{flag} = 0;", indent)

    def store_real(self, name: str, value: str, indent: int) -> None:
        """Bind a double-held name to a machine double. Nothing is allocated."""

        held, obj, flag = self.double_slots(name)
        self.emit(f"if ({obj}) {{ Py_DecRef({obj}); {obj} = 0; }}", indent)
        self.emit(f"{held} = {value};", indent)
        self.emit(f"{flag} = 1;", indent)

    def double_unbound_test(self, name: str, indent: int) -> None:
        """Refuse to read a double-held name that nothing has written."""

        _held, obj, flag = self.double_slots(name)
        self.needs_unbound = True
        message = (
            f"cannot access local variable {name!r} where it is not "
            "associated with a value"
        )
        self.emit(
            f"if (!{obj} && !{flag}) {{ _py2bin_unbound(1, "
            f"PyUnicode_FromString({_c_string(message)}), "
            f"PyUnicode_FromString({_c_string(name)})); {self.failure()} }}",
            indent,
        )

    def read_double(self, node: ast.Name, indent: int) -> str:
        """A double-held name as an owned reference, boxing it if it has to."""

        assert self.current is not None
        held, obj, flag = self.double_slots(node.id)
        target = self.temporary()
        if node.id not in self.current.certain:
            self.double_unbound_test(node.id, indent)
        self.emit(f"if ({flag}) {{", indent)
        self.emit(f"{target} = PyFloat_FromDouble({held});", indent + 1)
        self.emit(f"if (!{target}) {{ {self.failure()} }}", indent + 1)
        self.emit("} else {", indent)
        self.emit(f"Py_IncRef({obj});", indent + 1)
        self.emit(f"{target} = {obj};", indent + 1)
        self.emit("}", indent)
        return target

    def double_tree(self, node: ast.expr) -> bool:
        """True when every leaf of this expression is already a machine double.

        An integer literal is allowed as an operand but never as the whole
        tree: `2` on its own is an `int` and writing it back as a `double`
        would turn it into `2.0`, which is a different object.
        """

        if self.boxing:
            return False
        return self._double_leaf(node) and not is_machine_integer(node)

    def _double_leaf(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return is_machine_float(node) or is_machine_integer(node)
        if isinstance(node, ast.Name):
            return self.is_double(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            return self._double_leaf(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, _DOUBLE_OPS):
            if not (self._double_leaf(node.left) and self._double_leaf(node.right)):
                return False
            # One side has to be a float, or this is integer arithmetic wearing
            # a float's clothes - `2 * 3` must answer 6, not 6.0.
            return not (
                is_machine_integer(node.left) and is_machine_integer(node.right)
            )
        return False

    def double_expression(self, node: ast.expr, indent: int) -> _Machine:
        """Compute a double tree in registers. Only call it on one."""

        assert self.current is not None
        if isinstance(node, ast.Constant):
            # Spelled so C reads it as a double even when it is written as an
            # integer: `1` and `1.0` are the same value and different types,
            # and integer division of two ints is not what this is computing.
            return _Machine(None, f"(double)({node.value!r})")
        if isinstance(node, ast.Name):
            held, _, state = self.double_slots(node.id)
            if node.id not in self.current.certain:
                self.double_unbound_test(node.id, indent)
            return _Machine(state, held)
        if isinstance(node, ast.UnaryOp):
            inner = self.double_expression(node.operand, indent)
            if isinstance(node.op, ast.UAdd):
                return inner
            slot = self.real_slot()
            self.emit(f"{slot} = -({inner.value});", indent)
            return _Machine(inner.flag, slot)
        assert isinstance(node, ast.BinOp)
        left = self.double_expression(node.left, indent)
        right = self.double_expression(node.right, indent)
        known = [side.flag for side in (left, right) if side.flag is not None]
        flag = self.temporary_flag()
        self.emit(f"{flag} = {' && '.join(known) if known else '1'};", indent)
        slot = self.real_slot()
        if isinstance(node.op, ast.Div):
            # Python raises ZeroDivisionError where C answers an infinity, so
            # the zero goes to the slow arm rather than being computed wrongly.
            self.emit(f"if ({flag} && ({right.value}) == 0.0) {flag} = 0;", indent)
        self.emit(f"if ({flag}) {{", indent)
        self.emit(
            f"{slot} = ({left.value}) {_DOUBLE_SPELLING[type(node.op)]} "
            f"({right.value});",
            indent + 1,
        )
        self.emit("}", indent)
        return _Machine(flag, slot)

    def double_comparison(self, node: ast.expr) -> bool:
        """True for a comparison of two machine doubles."""

        if self.boxing or not isinstance(node, ast.Compare):
            return False
        if len(node.ops) != 1:
            return False
        if not isinstance(node.ops[0], _DOUBLE_TESTS):
            return False
        left, right = node.left, node.comparators[0]
        if not (self._double_leaf(left) and self._double_leaf(right)):
            return False
        # At least one side has to be a double, or this is an integer compare.
        return self.double_tree(left) or self.double_tree(right)

    @contextlib.contextmanager
    def boxed_only(self):
        """Emit the plain C-API translation, however narrow the tree looks.

        The slow arm of a fast path is the plain translation of the very tree
        the fast path gave up on. Without this it would be offered the fast
        path again and emit its own pair of arms, and the arms would nest as
        deep as the expression.
        """

        self.boxing += 1
        try:
            yield
        finally:
            self.boxing -= 1

    def narrow_tree(self, node: ast.expr) -> bool:
        """True when every leaf of this expression is already a machine int."""

        if self.boxing:
            return False
        if isinstance(node, ast.Constant):
            return is_machine_integer(node)
        if isinstance(node, ast.Name):
            return self.is_unboxed(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, _MACHINE_OPS):
            return self.narrow_tree(node.left) and self.narrow_tree(node.right)
        return False

    def machine_expression(self, node: ast.expr, indent: int) -> _Machine:
        """Compute a narrow tree in registers. Only call it on one.

        Overflow is where this has to be careful. Python integers do not stop
        at 64 bits, so every operation that can leave the word carries the
        check that says it did, and clears the flag rather than answering
        wrongly. The caller's slow arm then produces the true value through
        `PyNumber_Add` and friends, which is where the unbounded arithmetic
        lives. Nothing is lost but speed, and only for the values that need it.
        """

        assert self.current is not None
        if isinstance(node, ast.Constant):
            return _Machine(None, str(node.value))
        if isinstance(node, ast.Name):
            held, _, state = self.narrow_slots(node.id)
            if node.id not in self.current.certain:
                self.unbound_test(node.id, indent)
            return _Machine(state, held)
        assert isinstance(node, ast.BinOp)
        left = self.machine_expression(node.left, indent)
        right = self.machine_expression(node.right, indent)
        known = [side.flag for side in (left, right) if side.flag is not None]
        flag = self.temporary_flag()
        self.emit(f"{flag} = {' && '.join(known) if known else '1'};", indent)
        slot = self.machine_slot()
        self.emit(f"if ({flag}) {{", indent)
        self.machine_operation(node.op, slot, left.value, right.value, flag, indent + 1)
        self.emit("}", indent)
        return _Machine(flag, slot)

    def machine_operation(
        self, op: ast.operator, slot: str, left: str, right: str,
        flag: str, indent: int,
    ) -> None:
        """One operation on two machine integers, with its overflow test."""

        if isinstance(op, ast.Add):
            self.emit(f"{slot} = {left} + {right};", indent)
            # Two values with the same sign that produced the other sign left
            # the word. Signed overflow wraps here rather than being undefined:
            # py2bin's C compiler lowers this to the machine's own add, which
            # is where the wrapping comes from.
            self.emit(
                f"if ((({left} ^ {slot}) & ({right} ^ {slot})) < 0) {flag} = 0;",
                indent,
            )
        elif isinstance(op, ast.Sub):
            self.emit(f"{slot} = {left} - {right};", indent)
            self.emit(
                f"if ((({left} ^ {right}) & ({left} ^ {slot})) < 0) {flag} = 0;",
                indent,
            )
        elif isinstance(op, ast.Mult):
            # Checked before rather than after: recovering the operands from a
            # wrapped product needs a division, and dividing the smallest
            # negative value by -1 overflows in its own right. Two operands
            # that each fit 32 bits have a product that certainly fits 64.
            tests = [
                f"{side} > 2147483647 || {side} < -2147483647"
                for side in (left, right)
                # A literal already known to fit needs no test at run time.
                if not (side.lstrip("-").isdigit() and abs(int(side)) <= 2147483647)
            ]
            if tests:
                self.emit(f"if ({' || '.join(tests)}) {flag} = 0;", indent)
                self.emit(f"else {slot} = {left} * {right};", indent)
            else:
                self.emit(f"{slot} = {left} * {right};", indent)
        elif isinstance(op, (ast.FloorDiv, ast.Mod)):
            # Python floors, C truncates: -7 // 2 is -4 in Python and -3 in C,
            # and the remainder takes the sign of the divisor rather than of
            # the dividend. One correction after the fact fixes both, and it
            # is only taken when the division was not exact and the signs
            # disagree - which is the case C got wrong.
            quotient, remainder = self.machine_slot(), self.machine_slot()
            # A zero divisor is a ZeroDivisionError, which the C API raises
            # with the right message; -1 is the one divisor that can overflow,
            # against the most negative value there is.
            self.emit(f"if ({right} == 0 || {right} == -1) {flag} = 0;", indent)
            self.emit(f"if ({flag}) {{", indent)
            self.emit(f"{quotient} = {left} / {right};", indent + 1)
            self.emit(f"{remainder} = {left} - {quotient} * {right};", indent + 1)
            self.emit(
                f"if ({remainder} != 0 && (({remainder} < 0) != ({right} < 0))) {{",
                indent + 1,
            )
            self.emit(f"{quotient} = {quotient} - 1;", indent + 2)
            self.emit(f"{remainder} = {remainder} + {right};", indent + 2)
            self.emit("}", indent + 1)
            self.emit(
                f"{slot} = "
                f"{quotient if isinstance(op, ast.FloorDiv) else remainder};",
                indent + 1,
            )
            self.emit("}", indent)
        else:
            # The bitwise three. Python defines them on integers of unbounded
            # length in two's complement, which for two values that fit the
            # word is exactly what the machine does - and the result cannot be
            # wider than the wider operand, so there is nothing to check.
            spelling = {ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^"}
            self.emit(f"{slot} = {left} {spelling[type(op)]} {right};", indent)

    def box_machine(self, machine: _Machine, slow, indent: int) -> str:
        """The value as an object: the fast arm boxed, or `slow()` emitted."""

        target = self.temporary()
        self.emit(f"if ({machine.flag}) {{", indent)
        self.emit(f"{target} = PyLong_FromLongLong({machine.value});", indent + 1)
        self.emit(f"if (!{target}) {{ {self.failure()} }}", indent + 1)
        self.emit("} else {", indent)
        with self.boxed_only():
            spelled = slow(indent + 1)
        self.emit(f"{target} = {spelled};", indent + 1)
        self.emit("}", indent)
        return target

    def narrow_comparison(self, node: ast.expr) -> bool:
        """True for a comparison of two machine integers."""

        return (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], _MACHINE_TESTS)
            and self.narrow_tree(node.left)
            and self.narrow_tree(node.comparators[0])
            and any(isinstance(part, ast.Name) for part in ast.walk(node))
        )

    def machine_comparison(
        self, node: ast.Compare, indent: int
    ) -> tuple[str, str]:
        """`(flag, answer)`: the C ints saying whether it ran, and what it said.

        Two integers compare with one machine instruction. Doing it through
        `PyObject_RichCompare` costs a call that dispatches on both types and
        then hands back one of two singletons for `PyObject_IsTrue` to take
        apart again.
        """

        spelling = {
            ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
            ast.Eq: "==", ast.NotEq: "!=",
        }[type(node.ops[0])]
        left = self.machine_expression(node.left, indent)
        right = self.machine_expression(node.comparators[0], indent)
        known = [side.flag for side in (left, right) if side.flag is not None]
        flag = self.temporary_flag()
        self.emit(f"{flag} = {' && '.join(known) if known else '1'};", indent)
        answer = self.temporary_flag()
        self.emit(
            f"if ({flag}) {answer} = ({left.value} {spelling} {right.value});",
            indent,
        )
        return flag, answer

    def verdict(self, decision: str, indent: int) -> str:
        """A truth flag that is certainly 0 or 1, never a failure.

        `PyObject_IsTrue` and `PyObject_RichCompareBool` answer -1 with an
        exception set, and **-1 is true in C** - so every condition built on
        one silently took its branch and threw the exception away. A class
        whose `__bool__` raised ran the body of the `if` and exited 0 where
        CPython stops. Checked here, once, so no caller has to remember.
        """

        self.emit(f"if ({decision} < 0) {{ {self.failure()} }}", indent)
        return decision

    def truth(self, node: ast.expr, indent: int) -> str:
        """A C int that is 1 when this expression is true, -1 on failure.

        The point of having it separate from `expression` is that a condition
        never needs the value, only the verdict - and for two integers the
        verdict is a machine comparison with no object built at all. Every
        `if` and every `while` comes through here.
        """

        if isinstance(node, ast.BoolOp):
            # `if a and b` wants a verdict, and the generic path below built a
            # *value*: the whole chain was evaluated into a Python object and
            # then asked what it meant. That cost the machine comparison each
            # side would otherwise have got - a bare `i > 5` ran at 1.22x the
            # interpreter and `i > 5 and i < n` at 0.66x, which is the price
            # of boxing, not of the `and`.
            #
            # Each side goes through `truth` instead, so each keeps whatever
            # fast path it qualifies for, and the short circuit is the C `if`
            # that guards the next one - a side that must not be evaluated is
            # not merely discarded, its code never runs.
            decision = self.temporary_flag()
            first = self.truth(node.values[0], indent)
            self.emit(f"{decision} = {first};", indent)
            opened = 0
            for value in node.values[1:]:
                # `and` goes on while the answer is true, `or` while it is
                # false, and both stop on -1 so a failure is not read as a
                # verdict.
                carry_on = (
                    f"{decision} > 0"
                    if isinstance(node.op, ast.And)
                    else f"{decision} == 0"
                )
                self.emit(f"if ({carry_on}) {{", indent + opened)
                opened += 1
                following = self.truth(value, indent + opened)
                self.emit(f"{decision} = {following};", indent + opened)
            while opened:
                opened -= 1
                self.emit("}", indent + opened)
            return decision
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) > 1
            and all(type(operation) in _COMPARISONS for operation in node.ops)
        ):
            # A chain of operands that cost nothing to read twice - a name or
            # a literal - is written out as the `and` Python says it means.
            # That is worth more than comparing them where they stand: each
            # link is then an ordinary two-sided comparison, and picks up the
            # machine comparison that `0 < i` would have had on its own.
            # Naming the operands is what makes the rewrite safe; anything
            # that could be evaluated twice keeps the slots below, which is
            # what the chain is for.
            if all(
                isinstance(part, ast.Name)
                or (isinstance(part, ast.Constant))
                for part in (node.left, *node.comparators)
            ):
                operands = [node.left, *node.comparators]
                links = [
                    ast.copy_location(
                        ast.Compare(
                            left=operands[index],
                            ops=[operation],
                            comparators=[operands[index + 1]],
                        ),
                        node,
                    )
                    for index, operation in enumerate(node.ops)
                ]
                return self.truth(
                    ast.copy_location(
                        ast.BoolOp(op=ast.And(), values=links), node
                    ),
                    indent,
                )
            return self.chain_verdict(node, indent)
        if isinstance(node, ast.Call):
            settled = self.isinstance_verdict(node, indent)
            if settled is not None:
                return settled
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
        ):
            # `PySequence_Contains` answers the verdict directly, so `if x in
            # xs` needs no object at all. It used to build `True` or `False`
            # by looking the name up on the builtins module and then ask
            # `PyObject_IsTrue` what it had built.
            return self.membership_verdict(node, indent)
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
        ):
            # Same reasoning one level down: `if not x` asked the interpreter
            # to build `True` or `False` and then asked which it was.
            inner = self.truth(node.operand, indent)
            decision = self.temporary_flag()
            self.emit(f"{decision} = ({inner} < 0) ? -1 : !{inner};", indent)
            return decision
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], _MACHINE_TESTS)
            and not self.narrow_comparison(node)
        ):
            # `while i < len(xs)` is the loop condition most Python has; the
            # measurement is hoisted so the compare runs on two machine
            # integers, exactly as the assignment above it does.
            hoisted = self.hoisted_lengths(node, indent)
            if hoisted is not None and self.narrow_comparison(hoisted):
                node = hoisted
        if self.narrow_comparison(node):
            flag, answer = self.machine_comparison(node, indent)
            self.emit(f"if (!{flag}) {{", indent)
            with self.boxed_only():
                spelled = self.expression(node, indent + 1)
            self.emit(f"{answer} = PyObject_IsTrue({spelled});", indent + 1)
            self.emit(f"Py_DecRef({spelled});", indent + 1)
            self.verdict(answer, indent + 1)
            self.emit("}", indent)
            return answer
        if self.double_comparison(node):
            answer = self.temporary_flag()
            left = self.double_expression(node.left, indent)
            right = self.double_expression(node.comparators[0], indent)
            known = [
                side.flag for side in (left, right) if side.flag is not None
            ]
            settled = self.temporary_flag()
            self.emit(
                f"{settled} = {' && '.join(known) if known else '1'};", indent
            )
            self.emit(f"if ({settled}) {{", indent)
            self.emit(
                f"{answer} = ({left.value}) "
                f"{_COMPARISON_SPELLING[type(node.ops[0])]} ({right.value});",
                indent + 1,
            )
            self.emit("} else {", indent)
            with self.boxed_only():
                spelled = self.expression(node, indent + 1)
            self.emit(f"{answer} = PyObject_IsTrue({spelled});", indent + 1)
            self.emit(f"Py_DecRef({spelled});", indent + 1)
            self.verdict(answer, indent + 1)
            self.emit("}", indent)
            return answer
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in _COMPARISON_CODES
        ):
            # The verdict straight out, with no `True` to make and no second
            # call to ask what it meant. Every `if` and every loop test that
            # compares two things comes through here.
            decision = self.temporary_flag()
            left, left_owned = self.operand(node.left, indent)
            right, right_owned = self.operand(node.comparators[0], indent)
            self.emit(
                f"{decision} = PyObject_RichCompareBool({left}, {right}, "
                f"{_COMPARISON_CODES[type(node.ops[0])]});",
                indent,
            )
            self.release(left, left_owned, indent)
            self.release(right, right_owned, indent)
            return self.verdict(decision, indent)
        decision = self.temporary_flag()
        test = self.expression(node, indent)
        self.emit(f"{decision} = PyObject_IsTrue({test});", indent)
        self.emit(f"Py_DecRef({test});", indent)
        return self.verdict(decision, indent)

    def narrow_assign(self, name: str, value: ast.expr, indent: int) -> bool:
        """Bind a name straight from a register, if both ends allow it.

        This is where the saving actually lands. An accumulator around a loop
        becomes an add and a store; the same line through the C API is a call,
        a heap allocation and two reference counts, every iteration.
        """

        if self.is_double(name) and self.double_tree(value):
            return self.narrow_double_assign(name, value, indent)
        if not self.is_unboxed(name):
            return False
        if not self.narrow_tree(value):
            hoisted = self.hoisted_lengths(value, indent)
            if hoisted is None or not self.narrow_tree(hoisted):
                return False
            value = hoisted
        machine = self.machine_expression(value, indent)
        if machine.flag is None:
            self.store_machine(name, machine.value, indent)
            return True
        self.emit(f"if ({machine.flag}) {{", indent)
        self.store_machine(name, machine.value, indent + 1)
        self.emit("} else {", indent)
        with self.boxed_only():
            spelled = self.expression(value, indent + 1)
        self.store_object(name, spelled, indent + 1)
        self.emit("}", indent)
        return True

    def narrow_double_assign(self, name: str, value: ast.expr, indent: int) -> bool:
        """`x = <double tree>` straight into the register holding `x`."""

        real = self.double_expression(value, indent)
        if real.flag is None:
            self.store_real(name, real.value, indent)
            return True
        self.emit(f"if ({real.flag}) {{", indent)
        self.store_real(name, real.value, indent + 1)
        self.emit("} else {", indent)
        with self.boxed_only():
            spelled = self.expression(value, indent + 1)
        self.store_double_object(name, spelled, indent + 1)
        self.emit("}", indent)
        return True

    def box_double(self, real: _Machine, slow, indent: int) -> str:
        """A computed double as an object, with the slow arm for when it isn't.

        The counterpart of `box_machine`. A tree whose flag never came off can
        be boxed with no arm at all - only division can clear it.
        """

        target = self.temporary()
        if real.flag is None:
            self.emit(f"{target} = PyFloat_FromDouble({real.value});", indent)
            return self.checked(target, indent)
        self.emit(f"if ({real.flag}) {{", indent)
        self.emit(f"{target} = PyFloat_FromDouble({real.value});", indent + 1)
        self.emit(f"if (!{target}) {{ {self.failure()} }}", indent + 1)
        self.emit("} else {", indent)
        with self.boxed_only():
            spelled = slow(indent + 1)
        self.emit(f"{target} = {spelled};", indent + 1)
        self.emit("}", indent)
        return target

    # --- expressions -----------------------------------------------------

    def expression(self, node: ast.expr, indent: int) -> str:
        """Emit ``node`` and return a C variable owning the result."""

        if isinstance(node, ast.Constant):
            return self.constant(node, indent)
        if isinstance(node, ast.Name):
            return self.name(node, indent)
        if isinstance(node, ast.BinOp):
            return self.binary(node, indent)
        if isinstance(node, ast.Compare):
            return self.comparison(node, indent)
        if isinstance(node, ast.List):
            return self.list_literal(node, indent)
        if isinstance(node, ast.Dict):
            return self.dict_literal(node, indent)
        if isinstance(node, ast.Set):
            return self.set_literal(node, indent)
        if isinstance(
            node, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)
        ):
            return self.comprehension(node, indent)
        if isinstance(node, ast.Tuple):
            return self.tuple_literal(node, indent)
        if isinstance(node, ast.Subscript):
            return self.subscript(node, indent)
        if isinstance(node, ast.UnaryOp):
            return self.unary(node, indent)
        if isinstance(node, ast.IfExp):
            return self.conditional_expression(node, indent)
        if isinstance(node, ast.JoinedStr):
            return self.joined(node, indent)
        if isinstance(node, ast.BoolOp):
            return self.boolean(node, indent)
        if isinstance(node, ast.Attribute):
            return self.attribute(node, indent)
        if isinstance(node, ast.Call):
            return self.call(node, indent)
        if isinstance(node, ast.Lambda):
            return self.make_closure(node, "lambda", indent)
        if isinstance(node, ast.NamedExpr):
            return self.named(node, indent)
        raise self.fail(
            node, f"{type(node).__name__} has no C-API translation here yet"
        )

    #: What a C `long long` holds. A Python integer has no width, so a literal
    #: outside this has no C type to arrive in - its digits do.
    _SIGNED_64 = range(-(1 << 63), 1 << 63)

    def integer(self, value: int, indent: int) -> str:
        """An integer literal of any size."""

        target = self.temporary()
        if value in self._SIGNED_64:
            if value == -(1 << 63):
                # C has no literal for this one. `-9223372036854775808` is a
                # minus applied to `9223372036854775808`, which is one past
                # what the type holds, so it is written as a subtraction that
                # never leaves the range.
                spelled = "(-9223372036854775807LL - 1LL)"
            else:
                spelled = f"{value}LL"
            self.emit(f"{target} = PyLong_FromLongLong({spelled});", indent)
        else:
            # Read from its decimal text, which is the only shape wide enough
            # for a number Python is happy to write down and C is not.
            self.emit(
                f"{target} = PyLong_FromString({_c_string(str(value))}, 0, 10);",
                indent,
            )
        return self.checked(target, indent)

    def constant(self, node: ast.Constant, indent: int) -> str:
        target = self.temporary()
        if isinstance(node.value, bool) or node.value is None:
            # True, False and None are attributes of the builtins module, so
            # they come from the interpreter like every other builtin rather
            # than being reconstructed here.
            self.current.temporaries -= 1
            self.current.locals.remove(target)
            return self.builtin(repr(node.value), indent)
        if isinstance(node.value, (int, float, str, bytes)):
            self.current.temporaries -= 1
            self.current.locals.remove(target)
            return self.pool(node.value, indent)
        if isinstance(node.value, str):
            # `surrogatepass`, because a Python string may hold a lone
            # surrogate - `'a\udcffb'` is what a name that came off a
            # filesystem in some other encoding looks like - and plain UTF-8
            # refuses to encode one. The compiler stopped with the codec's
            # own complaint and no position, on a string the program is
            # perfectly entitled to write.
            encoded = node.value.encode("utf-8", "surrogatepass")
            if b"\0" in encoded or encoded != node.value.encode(
                "utf-8", "replace"
            ):
                # A zero byte is a character in Python and an end in C, so the
                # text goes through the decoder that is told how long it is.
                self.emit(
                    f"{target} = PyUnicode_DecodeUTF8("
                    f"{_c_bytes(encoded)}, {len(encoded)}LL, "
                    f'"surrogatepass");',
                    indent,
                )
            else:
                self.emit(
                    f"{target} = PyUnicode_FromString({_c_string(node.value)});",
                    indent,
                )
            return self.checked(target, indent)
        if isinstance(node.value, bytes):
            target = self.temporary()
            self.emit(
                f"{target} = PyBytes_FromStringAndSize("
                f"{_c_bytes(node.value)}, {len(node.value)}LL);",
                indent,
            )
            return self.checked(target, indent)
        if isinstance(node.value, complex):
            # `1+2j` is folded to a single constant before it gets here, so
            # there is nothing to add for the arithmetic - only a way to say
            # the value, which is the two halves handed to `complex`.
            maker = self.builtin("complex", indent)
            real = self.temporary()
            imaginary = self.temporary()
            self.emit(
                f"{real} = PyFloat_FromDouble({node.value.real!r});", indent
            )
            self.checked(real, indent)
            self.emit(
                f"{imaginary} = PyFloat_FromDouble({node.value.imag!r});",
                indent,
            )
            self.checked(imaginary, indent)
            held = self.temporary()
            self.emit(f"{held} = PyTuple_New(2LL);", indent)
            self.checked(held, indent)
            self.emit(f"PyTuple_SetItem({held}, 0, {real});", indent)
            self.emit(f"PyTuple_SetItem({held}, 1, {imaginary});", indent)
            target = self.temporary()
            self.emit(f"{target} = PyObject_Call({maker}, {held}, 0);", indent)
            self.emit(f"Py_DecRef({maker});", indent)
            self.emit(f"Py_DecRef({held});", indent)
            return self.checked(target, indent)
        if node.value is Ellipsis:
            # `...` is a singleton like None and True, and it is how a stub
            # body is written - `def f(self): ...` is most of an abstract
            # method or a Protocol, so refusing it refused those whole shapes.
            return self.builtin("Ellipsis", indent)
        raise self.fail(node, f"a {type(node.value).__name__} constant is not translated here yet")

    def reference(self, name: str) -> str | None:
        """The C name a Python name reads from, or None if nothing binds it.

        Local first, then parameter, then captured, then module global - the
        order Python looks in.
        """

        assert self.current is not None
        for scope in reversed(self.shadowed):
            if name in scope:
                return scope[name]
        if name in self.current.module_names:
            # `global x` says this name is the module's, whatever a local of
            # the same spelling would otherwise have been.
            return f"g_{self.prefix}{name}"
        if name in self.current.parameters:
            return f"p_{name}"
        if f"v_{name}" in self.current.locals:
            return f"v_{name}"
        if self.is_double(name):
            return f"v_{name}"
        if self.is_unboxed(name):
            # Declared as three variables; the object half is still `v_`.
            # Asked through `is_unboxed` rather than of the set directly, so
            # that a name the set holds but this scope stores in the module's
            # slot answers with the module's slot, like every other reference
            # to it.
            return f"v_{name}"
        if name in self.current.captures:
            return f"c_{name}"
        if name in self.globals:
            return f"g_{self.prefix}{name}"
        return None

    def default_cell(self, index: int | None) -> str | None:
        """Where this function's `index`-th default was put, if it was."""

        if index is None or self.current is None:
            return None
        cells = self.current.default_cells
        return cells[index] if 0 <= index < len(cells) else None

    def scope_dictionary(self, node: ast.Call, indent: int) -> str:
        """`locals()` and `vars()` with nothing passed - and why not `globals()`.

        Both answered `None` before this, which the caller then tried to
        iterate. The builtin needs the frame of whoever called it, and a
        compiled function has no frame, so it got nothing back.

        It does not need one. Since 3.13 `locals()` in a function answers with
        an *independent snapshot* - writing to what it returns does not change
        the local - and a snapshot of the locals is exactly what this can
        build, because it knows every name the function binds and can look at
        each slot to see whether it holds anything yet. Unbound names are left
        out, as they are there.

        `globals()` is refused instead. The module's names are C variables
        rather than entries in a dictionary, so what could be handed back is a
        copy - and a copy would take writes and quietly drop them, where the
        real thing changes the program's globals. That is worth an error at
        compile time rather than a wrong answer at run time.
        """

        if node.func.id == "dir":
            # `dir()` with nothing passed lists the calling frame's names,
            # and a compiled function has no frame - it answered with a
            # SystemError about one not existing. `dir(x)` is untouched.
            raise self.fail(
                node,
                "dir() with no argument lists the calling frame's names, and "
                "a compiled function has no frame - pass what you want listed",
            )
        if node.func.id == "globals" or self.at_module_level:
            # At module scope Python's `locals()` *is* `globals()`, and this
            # is the module's own dictionary rather than a copy of it, so a
            # write through what comes back changes the program's globals.
            held = self.temporary()
            self.emit("Py_IncRef(_py2bin_globals);", indent)
            self.emit(f"{held} = _py2bin_globals;", indent)
            return held
        assert self.current is not None
        target = self.temporary()
        self.emit(f"{target} = PyDict_New();", indent)
        self.checked(target, indent)
        for name in self.scope_locals():
            slot = self.declare(name)
            key = self.interned(name)
            # Every slot is tested: a name the function binds further down, or
            # only on a branch that was not taken, is not a local yet.
            self.emit(f"if ({slot}) {{", indent)
            self.emit(
                f"if (PyDict_SetItem({target}, {key}, {slot}) < 0) "
                f"{{ {self.failure()} }}",
                indent + 1,
            )
            self.emit("}", indent)
        return target

    def scope_locals(self) -> list[str]:
        """The names a `locals()` here should answer with, in Python's order.

        Parameters first and in order, then what the body binds, then what was
        captured - which is the order a code object lists them in, and so the
        order the dictionary comes out in.
        """

        assert self.current is not None
        ordered: list[str] = []
        for name in (
            *self.current.parameters,
            *sorted(self.current.body_binds - set(self.current.parameters)),
            *self.current.captures,
        ):
            if name not in ordered and name not in self.current.module_names:
                ordered.append(name)
        return ordered

    def borrowing_allowed(self, name: str) -> bool:
        """Whether this name's slot may be read without taking a reference.

        Not a global in dictionary mode: the slot is written but not read
        there, so borrowing from it would hand back a value the dictionary
        may since have replaced.
        """

        return not (
            self.globals_in_dict
            and self.reference(name) == f"g_{self.prefix}{name}"
        )

    def bound_around(self, name: str) -> bool:
        """Whether the scope enclosing a closure binds this name itself.

        What decides whether a closure captures a name. A module global is
        not captured, on purpose: Python reads a global when the closure
        *runs*, so rebinding it afterwards is meant to be visible, and a
        captured copy would freeze it.

        The question is which of the two this is, and asking "does the module
        bind this spelling" answered it the wrong way round - a parameter of
        the enclosing function shadows a module name, and the closure must
        take the parameter. It did not, so

            def d(f):
                def w(): return f() + 1
                return w
            @d
            def f(): return 1

        made a `w` that called the module's `f`, which `@d` had just rebound
        to `w` - every call recursed until the stack ran out. The same hole
        let a parameter named after a module-level function be ignored in
        favour of that function, which answered quietly rather than crashing.

        `reference` already looks in Python's order, so the answer is simply
        whether what it found is the module's slot or something nearer.
        """

        slot = self.reference(name)
        return slot is not None and slot != f"g_{self.prefix}{name}"

    def borrowable(self, node: ast.expr) -> str | None:
        """The C slot this name can be read from without taking a reference.

        An operand that is a plain local, parameter or capture does not need
        one. The slot holds a reference for the whole of the function body, and
        nothing else can write to it: it is a C variable, and the only code
        that assigns it is this function's own, between statements rather than
        during an expression. So the value cannot go away while the operation
        using it runs, and the increment-then-decrement around every read is
        two memory writes to arrive back where it started.

        A *global* is a different matter and is refused. Anything called while
        the expression is being evaluated can rebind a module-level name, and
        the reference the slot held may have been the last one - so a borrowed
        global can be read after it is freed.

        A walrus in the same function is also refused, because that is the one
        way a slot *can* be written in the middle of an expression.
        """

        if self.current is None:
            return None
        if (
            isinstance(node, ast.Constant)
            # `bool` is a subclass of `int`, and `True`/`False`/`None` are not
            # pooled at all - they come from the builtins module. Letting them
            # through here would name a static nothing ever fills.
            and not isinstance(node.value, bool)
            and isinstance(node.value, (int, float, str, bytes))
        ):
            # A pooled literal is the safest borrow there is: the static is
            # written once at start-up and never again, by anything. It was
            # being incremented and decremented around every use - `t + 1`,
            # `xs[0]`, every piece of an f-string - which is two writes to
            # arrive back where it started, on some of the commonest operands
            # a program has. Unlike a local this holds at module level too,
            # and while boxing, because nothing about the slot can change.
            return self.pool_slot(node.value)
        if self.boxing or self.at_module_level:
            return None
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            return None
        name = node.id
        if self.is_unboxed(name) or self.is_double(name):
            return None  # read through the narrow form, which may allocate
        if name in self.current.module_names or name in self.walrus_names:
            return None
        for scope in reversed(self.shadowed):
            # A comprehension target lives in a slot of its own, written by
            # the loop between iterations and read only after it is bound -
            # the same discipline as a local, so it is borrowed like one.
            if name in scope:
                return scope[name]
        slot = self.reference(name)
        if slot is None or slot[:2] not in ("v_", "p_", "c_"):
            return None
        if slot.startswith("v_") and name not in self.current.certain:
            # May never have been written; the test that says so belongs with
            # the read, and the borrowing path does not emit one.
            return None
        return slot

    def operand(self, node: ast.expr, indent: int) -> tuple[str, bool]:
        """A value for one operation, and whether the caller has to release it."""

        slot = self.borrowable(node)
        if slot is not None:
            return slot, False
        return self.expression(node, indent), True

    def release(self, value: str, owned: bool, indent: int) -> None:
        """Release what `operand` handed back, if it handed back a reference."""

        if owned:
            self.emit(f"Py_DecRef({value});", indent)

    def name(self, node: ast.Name, indent: int) -> str:
        assert self.current is not None
        if self.class_scope and node.id in self.class_scope[-1][1]:
            # A name the class body has already bound. `@property` puts `v`
            # in the namespace, and the `@v.setter` under it reads that same
            # `v` - from the class body, where nothing else can see it. It was
            # looked for in the enclosing scope and reported as undefined.
            namespace = self.class_scope[-1][0]
            fetched = self.temporary()
            self.emit(
                f"{fetched} = PyObject_GetItem({namespace}, "
                f"{self.interned(node.id)});",
                indent,
            )
            return self.checked(fetched, indent)
        held = self.reference(node.id)
        if self.globals_in_dict and (
            held is None or held == f"g_{self.prefix}{node.id}"
        ):
            # Read from the module's dictionary rather than from the C slot,
            # so that `globals()['x'] = 1` and `del globals()['x']` are seen -
            # which is the whole of what makes `globals()` real rather than a
            # copy. The slot still holds the value and still owns it; it is
            # simply not what anything reads.
            fetched = self.temporary()
            self.emit(
                f"{fetched} = PyObject_GetItem(_py2bin_globals, "
                f"{self.interned(node.id)});",
                indent,
            )
            # A name the module never binds may still be a builtin, and one
            # it binds at run time - `globals()['y'] = 9` - is in the
            # dictionary and nowhere else, which is why the lookup is by name
            # rather than by what the compiler knew.
            self.emit(f"if (!{fetched}) {{", indent)
            self.emit("PyErr_Clear();", indent + 1)
            fallback = self.program_name(node.id, indent + 1)
            self.emit(f"{fetched} = {fallback};", indent + 1)
            self.emit("}", indent)
            return fetched
        if self.is_unboxed(node.id):
            return self.read_unboxed(node, indent)
        if self.is_double(node.id):
            return self.read_double(node, indent)
        c_name = self.reference(node.id)
        if c_name is None:
            # Local, then global, then builtins - the order Python looks in.
            # `bytes` and `len` are names as much as they are callables, and a
            # program may pass one around rather than call it. Past builtins
            # there is nowhere else to look, so a failure here is a NameError.
            return self.program_name(node.id, indent)
        target = self.temporary()
        settled = node.id in self.current.certain or (
            c_name.startswith("g_")
            and node.id in self.certain_globals
            # Unrecorded means never settled, so the test is emitted.
            and self.certain_at.get(node.id, _NEVER) <= self.reached
        )
        if c_name[:2] in ("v_", "g_") and not settled:
            # A slot the program binds may never have been written to: `d` is
            # a name of this module even when the only `d = ...` sits in an
            # `if` that did not run. Reading it then found NULL, and
            # `Py_IncRef(NULL)` turned into `SystemError: null argument to
            # internal routine` - which says nothing about `d`. Parameters and
            # captures are always bound, so they are not tested.
            self.needs_unbound = True
            if c_name.startswith("g_"):
                kind, message = "NameError", f"name {node.id!r} is not defined"
            else:
                kind, message = (
                    "UnboundLocalError",
                    f"cannot access local variable {node.id!r} where it is not "
                    "associated with a value",
                )
            # One call rather than the whole construction. py2bin's C compiler
            # does not inline, so the helper stays one copy - and a large
            # module has well over a thousand of these.
            self.emit(
                f"if (!{c_name}) {{ _py2bin_unbound("
                f"{1 if kind == 'UnboundLocalError' else 0}, "
                f"PyUnicode_FromString({_c_string(message)}), "
                f"PyUnicode_FromString({_c_string(node.id)})); "
                f"{self.failure()} }}",
                indent,
            )
        # Handed back as an owned reference, like every other expression, so a
        # caller never has to ask where a value came from before releasing it.
        self.emit(f"Py_IncRef({c_name});", indent)
        self.emit(f"{target} = {c_name};", indent)
        return target

    def named(self, node: ast.NamedExpr, indent: int) -> str:
        """`(n := value)` - bind the name, and be the value as well.

        Two references are needed and only one is produced: the binding takes
        one and the surrounding expression takes the other, so the value is
        incremented once before either gets it.
        """

        value = self.expression(node.value, indent)
        self.emit(f"Py_IncRef({value});", indent)
        self.bind_target(node.target, value, indent)
        return value

    def binary(self, node: ast.BinOp, indent: int) -> str:
        if isinstance(node.op, ast.Pow):
            left = self.expression(node.left, indent)
            right = self.expression(node.right, indent)
            modulus = self.builtin("None", indent)
            target = self.temporary()
            self.emit(
                f"{target} = PyNumber_Power({left}, {right}, {modulus});", indent
            )
            for value in (left, right, modulus):
                self.emit(f"Py_DecRef({value});", indent)
            return self.checked(target, indent)
        function = _BINARY.get(type(node.op))
        if function is None:
            raise self.fail(node, f"{type(node.op).__name__} is not translated here yet")
        if (
            isinstance(node.op, ast.Add)
            and self.is_exact_str(node.left)
            and self.is_exact_str(node.right)
        ):
            # `PyNumber_Add` has to look for an `__add__` because a `str`
            # subclass may have one, and finding out is most of what the
            # operation costs. Where both sides are certainly exact - a
            # literal, an f-string, or a name the analysis has shown holds
            # nothing else - there is no `__add__` to find and
            # `PyUnicode_Concat` is what `+` means.
            #
            # Exactness is the whole of the argument: `MyStr("a") + "b"` must
            # still reach the subclass, and a call is not a display, so a name
            # bound from one is never claimed here.
            left, left_owned = self.operand(node.left, indent)
            right, right_owned = self.operand(node.right, indent)
            target = self.temporary()
            self.emit(f"{target} = PyUnicode_Concat({left}, {right});", indent)
            self.release(left, left_owned, indent)
            self.release(right, right_owned, indent)
            return self.checked(target, indent)
        if isinstance(node.op, _DOUBLE_OPS) and self.double_tree(node):
            if any(isinstance(part, ast.Name) for part in ast.walk(node)):
                real = self.double_expression(node, indent)
                return self.box_double(
                    real, lambda inner: self.boxed_binary(node, function, inner), indent
                )
        if isinstance(node.op, _MACHINE_OPS) and self.narrow_tree(node):
            # Worth the two arms only when a local is involved; a tree of
            # literals is folded into one constant elsewhere.
            if any(isinstance(part, ast.Name) for part in ast.walk(node)):
                machine = self.machine_expression(node, indent)
                return self.box_machine(
                    machine, lambda inner: self.boxed_binary(node, function, inner), indent
                )
        return self.boxed_binary(node, function, indent)

    def boxed_binary(self, node: ast.BinOp, function: str, indent: int) -> str:
        left, left_owned = self.operand(node.left, indent)
        right, right_owned = self.operand(node.right, indent)
        target = self.temporary()
        self.emit(f"{target} = {function}({left}, {right});", indent)
        self.release(left, left_owned, indent)
        self.release(right, right_owned, indent)
        return self.checked(target, indent)

    def comparison(self, node: ast.Compare, indent: int) -> str:
        if len(node.ops) != 1:
            return self.chained(node, indent)
        if isinstance(node.ops[0], (ast.In, ast.NotIn)):
            return self.membership(node, indent)
        if isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            return self.identity(node, indent)
        operation = _COMPARISONS.get(type(node.ops[0]))
        if operation is None:
            raise self.fail(node, f"{type(node.ops[0]).__name__} is not translated here yet")
        if self.narrow_comparison(node):
            flag, answer = self.machine_comparison(node, indent)
            target = self.temporary()
            self.emit(f"if ({flag}) {{", indent)
            self.emit(f"if ({answer}) {{", indent + 1)
            true_value = self.builtin("True", indent + 2)
            self.emit(f"{target} = {true_value};", indent + 2)
            self.emit("} else {", indent + 1)
            false_value = self.builtin("False", indent + 2)
            self.emit(f"{target} = {false_value};", indent + 2)
            self.emit("}", indent + 1)
            self.emit("} else {", indent)
            with self.boxed_only():
                spelled = self.boxed_comparison(node, operation, indent + 1)
            self.emit(f"{target} = {spelled};", indent + 1)
            self.emit("}", indent)
            return target
        return self.boxed_comparison(node, operation, indent)

    def boxed_comparison(self, node: ast.Compare, operation: int, indent: int) -> str:
        left = self.expression(node.left, indent)
        right = self.expression(node.comparators[0], indent)
        target = self.temporary()
        self.emit(
            f"{target} = PyObject_RichCompare({left}, {right}, {operation});", indent
        )
        self.emit(f"Py_DecRef({left});", indent)
        self.emit(f"Py_DecRef({right});", indent)
        return self.checked(target, indent)

    def chained(self, node: ast.Compare, indent: int) -> str:
        """`0 <= x < n` - each operand once, and it stops at the first false.

        `x` is the right of one link and the left of the next, and Python
        evaluates it once however many links it appears in; a rewrite into
        `0 <= x and x < n` would evaluate it twice, which a call or an index
        with a side effect would notice. So the operands are computed into
        slots and the links read them from there.
        """

        for operation in node.ops:
            if type(operation) not in _COMPARISONS:
                raise self.fail(
                    node,
                    f"a chain containing {type(operation).__name__} is not "
                    "translated here yet",
                )
        operands = [node.left, *node.comparators]
        # Cleared first: a link that short-circuits leaves the later slots
        # unfilled, and inside a loop they would otherwise still hold the
        # previous turn's values and be released twice.
        slots = [self.temporary() for _ in operands]
        for slot in slots:
            self.emit(f"{slot} = 0;", indent)
        target = self.temporary()
        self.emit(f"{target} = 0;", indent)
        depth = 0
        for index, operation in enumerate(node.ops):
            inner = indent + depth
            for position in (index, index + 1):
                if position == index and index > 0:
                    continue  # already computed as the previous link's right
                value = self.expression(operands[position], inner)
                self.emit(f"{slots[position]} = {value};", inner)
            self.emit(f"if ({target}) Py_DecRef({target});", inner)
            self.emit(
                f"{target} = PyObject_RichCompare({slots[index]}, "
                f"{slots[index + 1]}, {_COMPARISONS[type(operation)]});",
                inner,
            )
            self.checked(target, inner)
            if index + 1 == len(node.ops):
                break
            decision = self.temporary_flag()
            self.emit(f"{decision} = PyObject_IsTrue({target});", inner)
            self.emit(f"if ({decision} < 0) {{ {self.failure()} }}", inner)
            self.emit(f"if ({decision}) {{", inner)
            depth += 1
        while depth:
            depth -= 1
            self.emit("}", indent + depth)
        for slot in slots:
            self.emit(f"if ({slot}) {{ Py_DecRef({slot}); {slot} = 0; }}", indent)
        return target

    def chain_verdict(self, node: ast.Compare, indent: int) -> str:
        """`0 < i < n` as a C int, with each operand evaluated once.

        The same shape as the value form beside it, with the object left out:
        every link built a `True` or a `False` through `PyObject_RichCompare`
        and then asked `PyObject_IsTrue` what it had built, where
        `PyObject_RichCompareBool` answers 1, 0 or -1 directly. A chained
        comparison in a condition measured 0.46x the interpreter.

        The middle operand is still computed once however many links mention
        it - that is what the slots are for - and a link that short-circuits
        still leaves the later operands unevaluated, which a call with a side
        effect would notice.
        """

        operands = [node.left, *node.comparators]
        slots = [self.temporary() for _ in operands]
        for slot in slots:
            self.emit(f"{slot} = 0;", indent)
        decision = self.temporary_flag()
        self.emit(f"{decision} = 1;", indent)
        depth = 0
        for index, operation in enumerate(node.ops):
            inner = indent + depth
            for position in (index, index + 1):
                if position == index and index > 0:
                    continue  # already computed as the previous link's right
                value = self.expression(operands[position], inner)
                self.emit(f"{slots[position]} = {value};", inner)
            self.emit(
                f"{decision} = PyObject_RichCompareBool({slots[index]}, "
                f"{slots[index + 1]}, {_COMPARISONS[type(operation)]});",
                inner,
            )
            self.emit(f"if ({decision} < 0) {{ {self.failure()} }}", inner)
            if index + 1 == len(node.ops):
                break
            self.emit(f"if ({decision}) {{", inner)
            depth += 1
        while depth:
            depth -= 1
            self.emit("}", indent + depth)
        for slot in slots:
            self.emit(f"if ({slot}) {{ Py_DecRef({slot}); {slot} = 0; }}", indent)
        return decision

    def conditional_expression(self, node: ast.IfExp, indent: int) -> str:
        """`a if c else b` - a real branch, so only one arm is evaluated."""

        decision = self.truth(node.test, indent)
        self.emit(f"if ({decision} < 0) {{ {self.failure()} }}", indent)
        target = self.temporary()
        self.emit(f"if ({decision}) {{", indent)
        taken = self.expression(node.body, indent + 1)
        self.emit(f"    {target} = {taken};", indent)
        self.emit("} else {", indent)
        other = self.expression(node.orelse, indent + 1)
        self.emit(f"    {target} = {other};", indent)
        self.emit("}", indent)
        return target

    def joined(self, node: ast.JoinedStr, indent: int) -> str:
        """An f-string: every piece rendered, then joined in one pass.

        A format specifier goes to `format()`, whose mini-language is the
        interpreter's own, so `{x:.2f}` means here exactly what it means in
        Python rather than a re-implementation of it. `!r` and `!a` are
        `repr()` and `ascii()` for the same reason.

        The pieces used to be added together one at a time, starting from an
        empty string. That copies everything accumulated so far at every step,
        so an f-string cost time quadratic in its own length and allocated a
        string per piece that was thrown away by the next one. `PyUnicode_Join`
        measures the total first and fills one result, which is what the
        interpreter does for the same syntax.
        """

        # Each piece is carried with whether it has to be released. A literal
        # piece is a pooled static and a value being formatted is often a
        # local, and both were incremented and decremented around a use that
        # cannot outlive them.
        pieces: list[tuple[str, bool]] = []
        for piece in node.values:
            if isinstance(piece, ast.Constant):
                rendered, rendered_owned = self.operand(piece, indent)
                pieces.append((rendered, rendered_owned))
                continue
            if isinstance(piece, ast.FormattedValue):
                rendered_owned = True
                value, value_owned = self.operand(piece.value, indent)
                # `!r`, `!s`, `!a` - and no conversion, which for a value with
                # no specifier is str() as well.
                shaped = {114: "repr", 115: "str", 97: "ascii"}.get(
                    piece.conversion, ""
                )
                if not shaped and piece.format_spec is None:
                    # `f"{x}"` is `format(x, "")`, which is `__format__` - not
                    # `str`. For most types the two agree, because
                    # `object.__format__` with an empty specifier defers to
                    # `str`; for a type that defines `__format__` they do not,
                    # and this answered `str` for it. CPython's own
                    # FORMAT_SIMPLE takes the same two paths: an exact `str`
                    # is already its own formatting, and anything else is
                    # asked.
                    if self.is_exact_str(piece.value):
                        rendered, rendered_owned = value, value_owned
                        pieces.append((rendered, rendered_owned))
                        continue
                    converted = self.temporary()
                    self.emit(
                        f"{converted} = PyObject_Format({value}, 0);", indent
                    )
                    self.release(value, value_owned, indent)
                    self.checked(converted, indent)
                    pieces.append((converted, True))
                    continue
                if shaped:
                    converted = self.temporary()
                    # `str` and `repr` have their own entry points. Calling the
                    # *type* instead went through `type.__call__`, which
                    # allocates and dispatches to reach the same `tp_str` these
                    # two go straight to - and `{x}` is the commonest piece an
                    # f-string has.
                    direct = {"str": "PyObject_Str", "repr": "PyObject_Repr"}
                    if shaped in direct:
                        self.emit(
                            f"{converted} = {direct[shaped]}({value});", indent
                        )
                    else:
                        caller = self.builtin(shaped, indent)
                        self.emit(
                            f"{converted} = PyObject_CallOneArg({caller}, {value});",
                            indent,
                        )
                        self.emit(f"Py_DecRef({caller});", indent)
                    self.release(value, value_owned, indent)
                    value_owned = True
                    self.checked(converted, indent)
                    value = converted
                if piece.format_spec is None:
                    rendered = value
                    rendered_owned = value_owned
                else:
                    # The specifier is itself an f-string, which is how
                    # `{x:.{places}f}` names the width it wants.
                    specifier = self.joined(piece.format_spec, indent)
                    caller = self.builtin("format", indent)
                    arguments = self.temporary()
                    # Built a slot at a time rather than with PyTuple_Pack:
                    # that one is variadic, and on Apple's arm64 a variadic
                    # argument goes on the stack where this call puts it in a
                    # register. PyTuple_SetItem steals, which is what to do
                    # with the two references this is finished with anyway.
                    self.emit(f"{arguments} = PyTuple_New(2);", indent)
                    self.checked(arguments, indent)
                    if not value_owned:
                        # `PyTuple_SetItem` steals, so a borrowed value needs
                        # one of its own to give away.
                        self.emit(f"Py_IncRef({value});", indent)
                    self.emit(f"PyTuple_SetItem({arguments}, 0, {value});", indent)
                    self.emit(f"PyTuple_SetItem({arguments}, 1, {specifier});", indent)
                    rendered = self.temporary()
                    self.emit(
                        f"{rendered} = PyObject_Call({caller}, {arguments}, 0);",
                        indent,
                    )
                    self.emit(f"Py_DecRef({caller});", indent)
                    self.emit(f"Py_DecRef({arguments});", indent)
                    self.checked(rendered, indent)
                pieces.append((rendered, rendered_owned))
                continue
            raise self.fail(node, "unsupported f-string piece")

        if not pieces:
            return self.pool("", indent)
        if len(pieces) == 1:
            # Every piece is already a str - a literal, or the result of
            # `str`/`repr`/`ascii`/`format`, all of which answer one - so
            # joining a single piece would answer that piece.
            only, only_owned = pieces[0]
            if not only_owned:
                # Handed back as an owned reference, like every expression.
                self.emit(f"Py_IncRef({only});", indent)
            return only
        if len(pieces) <= _CONCAT_UNTIL:
            # Concatenated in a chain rather than gathered and joined. The
            # join allocates a tuple to be handed the pieces in and then walks
            # it twice - once to measure, once to fill - where a few
            # concatenations allocate only the intermediates, and for a
            # handful of pieces that is the cheaper shape. Measured: 0.034
            # against 0.038 for the three-piece f-string most programs write.
            #
            # `PyUnicode_Concat` rather than `PyNumber_Add`, and not only for
            # speed: an f-string *joins* its pieces, and `+` would ask the
            # left piece's type for `__add__` - a str subclass out of a
            # `__repr__` could override it, and CPython's join never asks.
            target, target_owned = pieces[0]
            for following, following_owned in pieces[1:]:
                joined = self.temporary()
                self.emit(
                    f"{joined} = PyUnicode_Concat({target}, {following});", indent
                )
                self.release(target, target_owned, indent)
                self.release(following, following_owned, indent)
                self.checked(joined, indent)
                target, target_owned = joined, True
            return target
        gathered = self.temporary()
        self.emit(f"{gathered} = PyTuple_New({len(pieces)});", indent)
        self.checked(gathered, indent)
        for position, (rendered, rendered_owned) in enumerate(pieces):
            # PyTuple_SetItem steals, which is what to do with a reference
            # this is finished with anyway - and a borrowed piece has to be
            # given one to hand over.
            if not rendered_owned:
                self.emit(f"Py_IncRef({rendered});", indent)
            self.emit(f"PyTuple_SetItem({gathered}, {position}, {rendered});", indent)
        blank = self.pool("", indent)
        target = self.temporary()
        self.emit(f"{target} = PyUnicode_Join({blank}, {gathered});", indent)
        self.emit(f"Py_DecRef({blank});", indent)
        self.emit(f"Py_DecRef({gathered});", indent)
        return self.checked(target, indent)

    def boolean(self, node: ast.BoolOp, indent: int) -> str:
        """`a and b` - the operand that decides, which is a value and not a
        bool: `1 and 2` is 2 in Python, and `0 or 3` is 3."""

        target = self.temporary()
        decision = self.temporary_flag()
        self.emit(f"{target} = 0;", indent)
        first = self.expression(node.values[0], indent)
        self.emit(f"{target} = {first};", indent)
        for operand in node.values[1:]:
            self.emit(f"{decision} = PyObject_IsTrue({target});", indent)
            self.emit(f"if ({decision} < 0) {{ {self.failure()} }}", indent)
            # `and` goes on to the next operand when this one is true;
            # `or` goes on when it is false. Getting this the wrong way round
            # makes `0 and boom()` call boom.
            keeps = "" if isinstance(node.op, ast.And) else "!"
            # Short-circuit: the second operand is only evaluated when the
            # first does not already settle the answer.
            self.emit(f"if ({keeps}{decision}) {{", indent)
            value = self.expression(operand, indent + 1)
            self.emit(f"Py_DecRef({target});", indent + 1)
            self.emit(f"{target} = {value};", indent + 1)
            self.emit("}", indent)
        return target

    def unary(self, node: ast.UnaryOp, indent: int) -> str:
        if (
            isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
            and not isinstance(node.operand.value, bool)
        ):
            # `-9223372036854775808` is one literal in Python and two nodes in
            # the tree. Negating afterwards means the positive half has to
            # exist first, and that one is exactly one past what a signed
            # 64-bit integer holds.
            return self.integer(-node.operand.value, indent)
        """`-x`, `+x`, `~x` and `not x`.

        A negation was `0 - x` for want of an entry point, on the reasoning
        that it is the same operation. It is not: `0 - 0.0` is positive zero
        where `-0.0` is negative zero, so a list of floats came back with the
        sign of one of them quietly changed.
        """

        signs = {
            ast.USub: "PyNumber_Negative",
            ast.UAdd: "PyNumber_Positive",
            ast.Invert: "PyNumber_Invert",
        }
        operation = signs.get(type(node.op))
        if operation is not None:
            value = self.expression(node.operand, indent)
            target = self.temporary()
            self.emit(f"{target} = {operation}({value});", indent)
            self.emit(f"Py_DecRef({value});", indent)
            return self.checked(target, indent)
        if isinstance(node.op, ast.Not):
            value = self.expression(node.operand, indent)
            decision = self.temporary_flag()
            self.emit(f"{decision} = PyObject_IsTrue({value});", indent)
            self.emit(f"Py_DecRef({value});", indent)
            self.emit(f"if ({decision} < 0) {{ {self.failure()} }}", indent)
            # A conditional expression cannot be a C-API argument here, so
            # the two names are fetched on their own branches.
            target = self.temporary()
            self.emit(f"if ({decision}) {{", indent)
            self.emit(
                f'    {target} = PyObject_GetAttrString(_py2bin_builtins, "False");',
                indent,
            )
            self.emit("} else {", indent)
            self.emit(
                f'    {target} = PyObject_GetAttrString(_py2bin_builtins, "True");',
                indent,
            )
            self.emit("}", indent)
            return self.checked(target, indent)
        raise self.fail(node, f"{type(node.op).__name__} is not translated here yet")

    def subscript(self, node: ast.Subscript, indent: int) -> str:
        """`xs[k]` - the object protocol, so a list, a dict and a string all
        work without any of them being known about here."""

        container, owned = self.operand(node.value, indent)
        if not isinstance(node.slice, ast.Slice) and self.narrow_tree(node.slice):
            target = self.indexed(
                container, node.slice, indent,
                known_sequence=self.is_exact_list(node.value),
            )
            self.release(container, owned, indent)
            return self.checked(target, indent)
        # `PyObject_GetItem` borrows the key, and a key is very often a
        # literal or a local - `d["name"]`, `xs[i]` - which was incremented
        # and decremented around the lookup. A slice has to be built, so it
        # is owned either way.
        if isinstance(node.slice, ast.Slice):
            key, key_owned = self.slice_object(node.slice, indent), True
        else:
            key, key_owned = self.operand(node.slice, indent)
        target = self.temporary()
        self.emit(f"{target} = PyObject_GetItem({container}, {key});", indent)
        self.release(container, owned, indent)
        self.release(key, key_owned, indent)
        return self.checked(target, indent)

    def indexed(
        self, container: str, index: ast.expr, indent: int,
        known_sequence: bool = False,
    ) -> str:
        """`xs[i]` where `i` is already a machine integer.

        Two things are avoided. The index does not become a `PyLong` only to be
        turned back into a machine integer inside the lookup; and a sequence is
        asked through the sequence protocol, which goes straight to the item,
        rather than through the mapping protocol, which first has to decide
        the subscript is not a slice.

        `PySequence_Check` is what keeps this honest. A dict answers no - `d[0]`
        is a mapping lookup and stays one - and so does anything else without
        the protocol, which then takes the ordinary path with a real index
        object. Nothing is tried and retried: a failure from `PySequence_GetItem`
        is the failure the program should see, not a signal to try again, which
        would run a `__getitem__` twice.

        A list known exact at compile time gets nothing better than this, and
        that was measured rather than assumed: `PyList_GetItem` answers a
        borrowed reference, and the `Py_IncRef` to own it is an out-of-line
        call through the import table, where the reference
        `PySequence_GetItem` takes on the way out is a plain increment inside
        the interpreter. The "faster" spelling lost two nanoseconds an access.
        """

        machine = self.machine_expression(index, indent)
        target = self.temporary()
        # A name the bindings prove is a list needs no protocol test - it has
        # the protocol. `PySequence_GetItem` is still the way in: asking
        # `PyList_GetItem` instead was measured *slower*, because the borrowed
        # reference it answers needs an increment that is an out-of-line call
        # from here, where the one `PySequence_GetItem` takes on the way out is
        # inside the interpreter already.
        guard = (
            f"{machine.flag or '1'}"
            if known_sequence
            else f"{machine.flag or '1'} && PySequence_Check({container})"
        )
        self.emit(f"if ({guard}) {{", indent)
        self.emit(
            f"{target} = PySequence_GetItem({container}, {machine.value});",
            indent + 1,
        )
        self.emit("} else {", indent)
        with self.boxed_only():
            key = self.expression(index, indent + 1)
        self.emit(f"{target} = PyObject_GetItem({container}, {key});", indent + 1)
        self.emit(f"Py_DecRef({key});", indent + 1)
        self.emit("}", indent)
        return target


    def slice_object(self, node: ast.Slice, indent: int) -> str:
        """`a:b:c` as the slice object the subscript protocol expects."""

        parts = []
        for piece in (node.lower, node.upper, node.step):
            parts.append(
                self.builtin("None", indent)
                if piece is None
                else self.expression(piece, indent)
            )
        target = self.temporary()
        self.emit(
            f"{target} = PySlice_New({parts[0]}, {parts[1]}, {parts[2]});", indent
        )
        for part in parts:
            self.emit(f"Py_DecRef({part});", indent)
        return self.checked(target, indent)

    def list_literal(self, node: ast.List, indent: int) -> str:
        """`[a, b, c]` - an empty list, then each element appended.

        PyList_Append does *not* steal its reference, unlike PyTuple_SetItem,
        so each element is released after it goes in.
        """

        target = self.temporary()
        self.emit(f"{target} = PyList_New(0LL);", indent)
        self.checked(target, indent)
        for element in node.elts:
            if isinstance(element, ast.Starred):
                # `[*xs, 3]` - extend rather than append, which is the same
                # iteration Python does over whatever the object offers.
                value = self.expression(element.value, indent)
                self.call_method(target, "extend", [value], indent)
            else:
                value = self.expression(element, indent)
                self.emit(f"PyList_Append({target}, {value});", indent)
            self.emit(f"Py_DecRef({value});", indent)
        return target

    def set_literal(self, node: ast.Set, indent: int) -> str:
        """`{a, b}` - built as a list, then handed to the `set` builtin."""

        listed = self.list_literal(
            ast.copy_location(ast.List(elts=node.elts, ctx=ast.Load()), node), indent
        )
        maker = self.builtin("set", indent)
        target = self.temporary()
        self.emit(f"{target} = PyObject_CallOneArg({maker}, {listed});", indent)
        self.emit(f"Py_DecRef({maker});", indent)
        self.emit(f"Py_DecRef({listed});", indent)
        return self.checked(target, indent)

    def comprehension(self, node, indent: int) -> str:
        """`[e for x in it if c]` - the loop it stands for, written out.

        A generator expression is built eagerly into a list here. That is not
        lazy, so an infinite source would not terminate and the memory is spent
        up front; every generator in the programs this targets is consumed
        immediately, which is the case the trade is made for.
        """

        first = node.generators[0]
        identity = (
            len(node.generators) == 1
            and not first.is_async
            and not first.ifs
            and isinstance(first.target, ast.Name)
            and not isinstance(node, ast.DictComp)
            and isinstance(node.elt, ast.Name)
            and node.elt.id == first.target.id
        )
        if identity:
            # `[x for x in it]` is `list(it)` to the letter - same iteration,
            # same errors, same result - and `list(it)` runs the whole loop
            # inside the interpreter with a length hint, where the written-out
            # loop pays a `PyIter_Next` and an append per item. The name
            # `list` is not consulted: the comprehension's meaning does not
            # involve it, so the genuine type fetched at start-up is exactly
            # right even for a program that rebinds the builtin.
            source = self.expression(first.iter, indent)
            target = self.temporary()
            if isinstance(node, ast.GeneratorExp):
                # An iterator is what a generator expression is; handing one
                # straight over is nearer the real thing than gathering.
                self.emit(f"{target} = PyObject_GetIter({source});", indent)
            else:
                maker = self.builtin(
                    "set" if isinstance(node, ast.SetComp) else "list", indent
                )
                self.emit(
                    f"{target} = PyObject_CallOneArg({maker}, {source});", indent
                )
                self.emit(f"Py_DecRef({maker});", indent)
            self.emit(f"Py_DecRef({source});", indent)
            return self.checked(target, indent)

        # A comprehension has a scope of its own, and a class body's names
        # are not in it: `class C: size = 3; rows = [size for _ in r]` is a
        # NameError in Python, and answered `[3, 3]` here. The one exception
        # is the *first* iterable, which is evaluated in the scope the
        # comprehension is written in - that is what lets `[v * 2 for v in
        # xs]` read the `xs` beside it - so that one keeps the class scope
        # and everything else is written without it.
        outer_class_scope = self.class_scope
        self.class_scope = []
        self.first_iterable_scope = outer_class_scope
        try:
            return self.comprehension_body(node, indent)
        finally:
            self.class_scope = outer_class_scope
            self.first_iterable_scope = []

    def comprehension_body(self, node, indent: int) -> str:
        target = self.temporary()
        if isinstance(node, ast.DictComp):
            # Built as a dict from the start; there is no list shape a
            # key-and-value comprehension could go through.
            self.emit(f"{target} = PyDict_New();", indent)
            self.checked(target, indent)
            self.comprehension_clause(node, 0, target, indent)
            return target
        self.emit(f"{target} = PyList_New(0LL);", indent)
        self.checked(target, indent)
        if not self.counted_comprehension(node, target, indent):
            self.comprehension_clause(node, 0, target, indent)
        if isinstance(node, ast.GeneratorExp):
            # Gathered eagerly, as the docstring says - but handed back as an
            # *iterator*, because that is what a generator expression is. A
            # list answers `for` and `sum()` the same way and `next()` not at
            # all: `next(p for p in candidates)` stopped with "'list' object is
            # not an iterator", which names nothing the program wrote.
            walking = self.temporary()
            self.emit(f"{walking} = PyObject_GetIter({target});", indent)
            self.emit(f"Py_DecRef({target});", indent)
            return self.checked(walking, indent)
        if isinstance(node, ast.SetComp):
            maker = self.builtin("set", indent)
            gathered = self.temporary()
            self.emit(
                f"{gathered} = PyObject_CallOneArg({maker}, {target});", indent
            )
            self.emit(f"Py_DecRef({maker});", indent)
            self.emit(f"Py_DecRef({target});", indent)
            return self.checked(gathered, indent)
        return target

    def counted_comprehension(self, node, target: str, indent: int) -> bool:
        """`[e for x in range(...)]` counted in a register, like a `for` is.

        The written-out comprehension loop paid `PyIter_Next` and an integer
        object per item even over a `range`, and - worse - its target was an
        object, so arithmetic on it in the element was boxed too. Here the
        target becomes a function-level name of its own, spelled uniquely and
        registered with the integer analysis, so `x * 2` in the element runs
        in machine registers and is boxed once, by the append.

        Answers False when the shape is not claimed, and emits nothing then.
        Anything that introduces a scope inside the element - a lambda, a
        nested comprehension, a walrus - declines: the rewrite renames the
        target inside the element, and a nested scope of the same spelling is
        a different name.
        """

        if isinstance(node, ast.DictComp) or len(node.generators) != 1:
            return False
        clause = node.generators[0]
        if clause.is_async or not isinstance(clause.target, ast.Name):
            return False
        bounds = narrow_range(clause.iter)
        if bounds is None:
            return False
        pieces = [node.elt, *clause.ifs]
        for piece in pieces:
            for inner in ast.walk(piece):
                if isinstance(
                    inner,
                    (
                        ast.Lambda, ast.NamedExpr, ast.ListComp, ast.SetComp,
                        ast.DictComp, ast.GeneratorExp, ast.Await,
                        ast.Yield, ast.YieldFrom,
                    ),
                ):
                    return False
        assert self.current is not None
        name = clause.target.id
        self.comp_serial += 1
        unique = f"_py2bin_c{self.comp_serial}_{name}"

        class _Rename(ast.NodeTransformer):
            def visit_Name(self, found: ast.Name) -> ast.AST:
                if found.id == name:
                    found = ast.copy_location(
                        ast.Name(id=unique, ctx=found.ctx), found
                    )
                return found

        import copy as _copy

        renamed = [
            ast.fix_missing_locations(_Rename().visit(_copy.deepcopy(piece)))
            for piece in pieces
        ]
        element, conditions = renamed[0], renamed[1:]
        self.current.unboxed.add(unique)
        self.synthetic.add(unique)
        # The loop binds it before anything reads it, so no unbound test.
        self.current.certain.add(unique)
        held, obj, state = self.narrow_slots(unique)

        spelled = [self.expression(argument, indent) for argument in bounds]
        start, stop, step = (
            self.machine_slot(), self.machine_slot(), self.machine_slot()
        )
        counting = self.temporary_flag()
        self.emit(f"{counting} = 1;", indent)
        order = (
            [(start, spelled[0]), (stop, spelled[1])]
            if len(spelled) > 1
            else [(stop, spelled[0])]
        )
        if len(spelled) > 2:
            order.append((step, spelled[2]))
        if len(spelled) < 2:
            self.emit(f"{start} = 0;", indent)
        if len(spelled) < 3:
            self.emit(f"{step} = 1;", indent)
        for slot, value in order:
            self.emit(f"{slot} = PyLong_AsLongLong({value});", indent)
            self.emit(
                f"if ({slot} == -1 && PyErr_Occurred()) "
                f"{{ PyErr_Clear(); {counting} = 0; }}",
                indent,
            )
        self.emit(f"if ({step} == 0) {counting} = 0;", indent)
        for slot in (start, stop, step):
            self.emit(
                f"if ({slot} > {_MACHINE_LIMIT} || {slot} < -{_MACHINE_LIMIT}) "
                f"{counting} = 0;",
                indent,
            )
        iterator = self.temporary()
        self.emit(f"{iterator} = 0;", indent)
        self.emit(f"if (!{counting}) {{", indent)
        built = self.call_range(spelled, indent + 1)
        self.emit(f"{iterator} = PyObject_GetIter({built});", indent + 1)
        self.emit(f"Py_DecRef({built});", indent + 1)
        self.emit(f"if (!{iterator}) {{ {self.failure()} }}", indent + 1)
        self.emit("}", indent)
        for value in spelled:
            self.emit(f"Py_DecRef({value});", indent)
        filling = None
        if not conditions:
            # With no filter, the counting arm knows how many items are coming
            # and fills a list made at that length: `PyList_SetItem` steals
            # the reference and never grows the storage, where an append pays
            # for both. The count is what `len(range(...))` answers, and zero
            # for a range that runs the wrong way.
            length = self.machine_slot()
            filling = self.machine_slot()
            self.emit(f"{filling} = -1;", indent)
            self.emit(f"if ({counting}) {{", indent)
            self.emit(f"if ({step} > 0) {{", indent + 1)
            self.emit(
                f"{length} = {stop} > {start} "
                f"? ({stop} - {start} + {step} - 1) / {step} : 0;",
                indent + 2,
            )
            self.emit("} else {", indent + 1)
            self.emit(
                f"{length} = {start} > {stop} "
                f"? ({start} - {stop} - {step} - 1) / (-{step}) : 0;",
                indent + 2,
            )
            self.emit("}", indent + 1)
            self.emit(f"Py_DecRef({target});", indent + 1)
            self.emit(f"{target} = PyList_New({length});", indent + 1)
            self.emit(f"if (!{target}) {{ {self.failure()} }}", indent + 1)
            self.emit(f"{filling} = 0;", indent + 1)
            self.emit("}", indent)
        counter = self.machine_slot()
        self.emit(f"{counter} = {start};", indent)
        item = self.temporary()
        self.emit("while (1) {", indent)
        self.emit(f"if ({counting}) {{", indent + 1)
        self.emit(f"if ({step} > 0) {{", indent + 2)
        self.emit(f"if ({counter} >= {stop}) break;", indent + 3)
        self.emit("} else {", indent + 2)
        self.emit(f"if ({counter} <= {stop}) break;", indent + 3)
        self.emit("}", indent + 2)
        self.emit(f"if ({obj}) {{ Py_DecRef({obj}); {obj} = 0; }}", indent + 2)
        self.emit(f"{held} = {counter};", indent + 2)
        self.emit(f"{state} = 1;", indent + 2)
        self.emit(f"{counter} = {counter} + {step};", indent + 2)
        self.emit("} else {", indent + 1)
        self.emit(f"{item} = PyIter_Next({iterator});", indent + 2)
        self.emit(
            f"if (!{item}) {{ if (PyErr_Occurred()) {{ {self.failure()} }} break; }}",
            indent + 2,
        )
        self.store_object(unique, item, indent + 2)
        self.emit("}", indent + 1)
        inner = indent + 1
        closing = 0
        for condition in conditions:
            decision = self.truth(condition, inner)
            self.emit(f"if ({decision} < 0) {{ {self.failure()} }}", inner)
            self.emit(f"if ({decision}) {{", inner)
            inner += 1
            closing += 1
        value = self.expression(element, inner)
        if filling is not None:
            # Stolen by `PyList_SetItem`, so nothing to give back on that arm.
            self.emit(f"if ({filling} >= 0) {{", inner)
            self.emit(f"PyList_SetItem({target}, {filling}, {value});", inner + 1)
            self.emit(f"{filling} = {filling} + 1;", inner + 1)
            self.emit("} else {", inner)
            self.emit(f"PyList_Append({target}, {value});", inner + 1)
            self.emit(f"Py_DecRef({value});", inner + 1)
            self.emit("}", inner)
        else:
            self.emit(f"PyList_Append({target}, {value});", inner)
            self.emit(f"Py_DecRef({value});", inner)
        for _ in range(closing):
            inner -= 1
            self.emit("}", inner)
        self.emit("}", indent)
        self.emit(f"if ({iterator}) Py_DecRef({iterator});", indent)
        return True

    def comprehension_clause(self, node, position: int, target: str, indent: int) -> None:
        """One `for` clause of a comprehension, then whatever follows it."""

        if position == len(node.generators):
            if isinstance(node, ast.DictComp):
                key = self.expression(node.key, indent)
                value = self.expression(node.value, indent)
                self.emit(f"PyDict_SetItem({target}, {key}, {value});", indent)
                # Neither reference is stolen, so both go back.
                self.emit(f"Py_DecRef({key});", indent)
                self.emit(f"Py_DecRef({value});", indent)
                return
            value = self.expression(node.elt, indent)
            self.emit(f"PyList_Append({target}, {value});", indent)
            self.emit(f"Py_DecRef({value});", indent)
            return
        clause = node.generators[position]
        if clause.is_async:
            raise self.fail(node, "an async comprehension is not translated here")
        # May be empty: `for d[k] in ...` stores through a subscript and binds
        # no name of its own, which needs no scope but is perfectly legal.
        names = _bound_names(clause.target)
        if position == 0:
            # See `comprehension_body`: this one, and only this one, is
            # evaluated in the scope around the comprehension.
            held = self.class_scope
            self.class_scope = self.first_iterable_scope
            try:
                sequence = self.expression(clause.iter, indent)
            finally:
                self.class_scope = held
        else:
            sequence = self.expression(clause.iter, indent)
        iterator = self.temporary()
        self.emit(f"{iterator} = PyObject_GetIter({sequence});", indent)
        self.checked(iterator, indent)
        self.emit(f"Py_DecRef({sequence});", indent)
        item = self.temporary()
        # A comprehension has a scope of its own, so its target is a slot of
        # its own: `zs = [x * 2 for x in xs]` must leave the enclosing `x`
        # exactly as it was. Binding the ordinary name instead left `print(x)`
        # after it answering with the comprehension's last item.
        # One slot per name the clause binds, so `for k, v in pairs` works and
        # neither k nor v is the enclosing scope's. Cleared first: the loop
        # releases what a slot held before taking the next item, and a slot
        # reused from an earlier statement still holds that statement's
        # pointer - released there, so releasing it again drops a reference
        # this code no longer owns.
        scope = {}
        for name in names:
            slot = self.temporary()
            self.emit(f"{slot} = 0;", indent)
            scope[name] = slot
        self.shadowed.append(scope)
        self.emit("while (1) {", indent)
        self.emit(f"{item} = PyIter_Next({iterator});", indent + 1)
        self.emit(
            f"if (!{item}) {{ if (PyErr_Occurred()) {{ {self.failure()} }} break; }}",
            indent + 1,
        )
        # bind_target consumes the item, whatever shape the target has, and
        # unpacks a tuple into the slots just declared.
        self.bind_target(clause.target, item, indent + 1)
        inner = indent + 1
        for condition in clause.ifs:
            decision = self.temporary_flag()
            test = self.expression(condition, inner)
            self.emit(f"{decision} = PyObject_IsTrue({test});", inner)
            self.emit(f"Py_DecRef({test});", inner)
            self.emit(f"if ({decision} < 0) {{ {self.failure()} }}", inner)
            self.emit(f"if ({decision}) {{", inner)
            inner += 1
        self.comprehension_clause(node, position + 1, target, inner)
        for _ in clause.ifs:
            inner -= 1
            self.emit("}", inner)
        self.emit("}", indent)
        self.shadowed.pop()
        for slot in scope.values():
            self.emit(f"if ({slot}) {{ Py_DecRef({slot}); {slot} = 0; }}", indent)
        self.emit(f"Py_DecRef({iterator});", indent)

    def wrap_unhashable(self, key: str, where: str, indent: int) -> None:
        """Say which place an unhashable value was being used as.

        `PyDict_SetItem` reports `unhashable type: 'list'`; a dict display
        reports `cannot use 'list' as a dict key (unhashable type: 'list')`,
        which says where. Only a TypeError is rewritten - anything else the
        store raised is the program's and travels as it is.
        """

        matched = self.temporary_flag()
        kind = self.builtin_raw("TypeError", indent)
        self.emit(f"{matched} = PyErr_ExceptionMatches({kind});", indent)
        self.emit(f"if ({matched}) {{", indent)
        held = self.temporary()
        self.emit("PyErr_Clear();", indent + 1)
        self.emit(
            f'{held} = PyObject_GetAttrString({key}, "__class__");', indent + 1
        )
        self.emit(f"if ({held}) {{", indent + 1)
        named = self.temporary()
        self.emit(
            f'{named} = PyObject_GetAttrString({held}, "__name__");', indent + 2
        )
        self.emit(f"Py_DecRef({held});", indent + 2)
        self.emit(f"if ({named}) {{", indent + 2)
        built = self.temporary()
        piece = self.temporary()
        self.emit(f"{built} = 0;", indent + 3)
        for literal, insert in (
            ("cannot use '", True),
            (f"' as {where} (unhashable type: '", True),
            ("')", False),
        ):
            self.emit(
                f"{piece} = PyUnicode_FromString({_c_string(literal)});",
                indent + 3,
            )
            self.emit(f"if (!{built}) {{", indent + 3)
            self.emit(f"{built} = {piece};", indent + 4)
            self.emit("} else {", indent + 3)
            joined = self.temporary()
            self.emit(
                f"{joined} = PyNumber_Add({built}, {piece});", indent + 4
            )
            self.emit(f"Py_DecRef({built});", indent + 4)
            self.emit(f"Py_DecRef({piece});", indent + 4)
            self.emit(f"{built} = {joined};", indent + 4)
            self.emit("}", indent + 3)
            if insert:
                self.emit(f"if ({built}) {{", indent + 3)
                added = self.temporary()
                self.emit(
                    f"{added} = PyNumber_Add({built}, {named});", indent + 4
                )
                self.emit(f"Py_DecRef({built});", indent + 4)
                self.emit(f"{built} = {added};", indent + 4)
                self.emit("}", indent + 3)
        self.emit(f"Py_DecRef({named});", indent + 3)
        self.emit(f"if ({built}) {{", indent + 3)
        raised = self.temporary()
        self.emit(
            f"{raised} = PyObject_CallOneArg({kind}, {built});", indent + 4
        )
        self.emit(f"Py_DecRef({built});", indent + 4)
        self.emit(f"if ({raised}) {{", indent + 4)
        self.emit(f"PyErr_SetObject({kind}, {raised});", indent + 5)
        self.emit(f"Py_DecRef({raised});", indent + 5)
        self.emit("}", indent + 4)
        self.emit("}", indent + 3)
        self.emit("}", indent + 2)
        self.emit("}", indent + 1)
        self.emit("}", indent)
        self.emit(f"Py_DecRef({kind});", indent)
        self.emit(self.failure(), indent)

    def dict_literal(self, node: ast.Dict, indent: int) -> str:
        """`{k: v}` - an empty dict, then each pair set into it."""

        target = self.temporary()
        self.emit(f"{target} = PyDict_New();", indent)
        self.checked(target, indent)
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                # `{**base, "k": 1}` - update, so a later key wins, as in
                # Python.
                value = self.expression(value_node, indent)
                self.call_method(target, "update", [value], indent)
                self.emit(f"Py_DecRef({value});", indent)
                continue
            key = self.expression(key_node, indent)
            value = self.expression(value_node, indent)
            outcome = self.temporary_flag()
            self.emit(
                f"{outcome} = PyDict_SetItem({target}, {key}, {value});", indent
            )
            # The answer was thrown away, and an unhashable key leaves an
            # exception set: `{[1]: 2}` built a dict, carried on, and the
            # TypeError surfaced later attached to whatever ran next. A dict
            # display says which of the two it was, and so does this.
            self.emit(f"if ({outcome} < 0) {{", indent)
            self.wrap_unhashable(key, "a dict key", indent + 1)
            self.emit("}", indent)
            # PyDict_SetItem does not steal either reference, so both go back.
            self.emit(f"Py_DecRef({key});", indent)
            self.emit(f"Py_DecRef({value});", indent)
        return target

    def tuple_literal(self, node: ast.Tuple, indent: int) -> str:
        """`(a, b)` - built through the list the vetted set can grow, then
        handed to the tuple builtin.

        PyTuple_Pack is variadic and py2bin passes no variadic arguments, so
        its vetted arity is fixed at two and cannot serve a general tuple.
        Going through `tuple(list)` costs an allocation and works for any
        length, which is the better trade here.
        """

        listed = self.list_literal(
            ast.copy_location(ast.List(elts=node.elts, ctx=ast.Load()), node), indent
        )
        maker = self.builtin("tuple", indent)
        target = self.temporary()
        self.emit(f"{target} = PyObject_CallOneArg({maker}, {listed});", indent)
        self.emit(f"Py_DecRef({maker});", indent)
        self.emit(f"Py_DecRef({listed});", indent)
        return self.checked(target, indent)

    def builtin(self, name: str, indent: int) -> str:
        """A name from the builtins module, fetched the way Python fetches it.

        `range`, `sum`, `sorted`, `abs`, `dict` - anything the interpreter
        already has. Nothing here reimplements them; the module is imported
        once at startup and this reads an attribute off it.

        This is for names *this emitter* asks for, which exist. A name the
        program wrote goes through :meth:`program_name`.
        """

        slot = self.cached_builtins.get(name)
        if slot is None:
            slot = f"_py2bin_b{len(self.cached_builtins)}"
            self.cached_builtins[name] = slot
        target = self.temporary()
        # Handed back owned, like every other expression, so the one rule for
        # releasing a value still holds - but an increment on a slot already
        # in hand, rather than a lookup by name in the builtins dictionary.
        self.emit(f"Py_IncRef({slot});", indent)
        self.emit(f"{target} = {slot};", indent)
        return target

    def program_name(self, name: str, indent: int) -> str:
        """A name the program used that is not local, global or one of its own.

        Then it is either a builtin or nothing at all. When it is nothing, the
        lookup leaves an AttributeError naming the builtins module rather than
        the program - and left set, it turned the next thing done into
        `SystemError: ... returned a result with an exception set`, which names
        neither. This raises the NameError Python raises, in Python's wording.

        Only this path pays for that. Putting it in :meth:`builtin` instead
        added the whole construction to every `None` at every function tail,
        for a lookup that cannot fail.
        """

        if hasattr(builtins, name):
            # It is a builtin, so the lookup cannot fail and there is nothing
            # to report. This is the common case by a long way - `Exception`,
            # `open`, `len` - and giving each of them the whole NameError
            # construction added forty thousand lines of C to a large module.
            # The interpreter this asks is the one the artifact links, so the
            # answer holds at run time.
            return self.checked(self.builtin_raw(name, indent), indent)
        self.needs_unbound = True
        target = self.builtin_raw(name, indent)
        self.emit(
            f"if (!{target}) {{ PyErr_Clear(); _py2bin_unbound(0, "
            f"PyUnicode_FromString({_c_string(f'name {name!r} is not defined')}), "
            f"PyUnicode_FromString({_c_string(name)})); "
            f"{self.failure()} }}",
            indent,
        )
        return target

    def builtin_raw(self, name: str, indent: int) -> str:
        """The lookup itself, with nothing said about why it might fail.

        The exception classes a raise needs come through here: turning *their*
        failure into a NameError would need a class to build it from, which is
        the lookup that just failed.

        The lookup stays live rather than being cached into a slot the way the
        emitter's own builtins are. A program may rebind one - `builtins.print
        = ...` is legal and people do it in tests - and a cached slot would go
        on answering with what was there at start-up. The *name* is interned,
        so what is paid each time is a dictionary probe and not the building
        and hashing of the name to probe with.
        """

        target = self.temporary()
        self.emit(
            f"{target} = PyObject_GetAttr(_py2bin_builtins, "
            f"{self.interned(name)});",
            indent,
        )
        return target

    def interned(self, text: str) -> str:
        """The file-scope slot holding ``text`` as an interned str.

        One slot per distinct name, however many times it is mentioned: the
        cost is a pointer per attribute name in the program, paid once.
        """

        slot = self.interned_names.get(text)
        if slot is None:
            slot = f"_py2bin_s{len(self.interned_names)}"
            self.interned_names[text] = slot
        return slot

    def keyword_names(self, names: tuple[str, ...]) -> str:
        """The file-scope slot holding these keyword names as a tuple."""

        slot = self.keyword_tuples.get(names)
        if slot is None:
            slot = f"_py2bin_kw{len(self.keyword_tuples)}"
            self.keyword_tuples[names] = slot
        return slot

    def pool(self, value, indent: int) -> str:
        """A literal as an owned reference, taken from the start-up pool.

        The value is immutable, so one object serves every mention of it. That
        also matches CPython more closely than rebuilding did: equal literals
        in one code object are the same object there too.
        """

        # Keyed by `repr`, not by the value. `-0.0 == 0.0` is true and the two
        # hash alike, so keying by value gave both the same slot and turned
        # every `-0.0` in the program into `0.0` - which is a different float,
        # and the one that decides the sign of the result it is multiplied
        # into. `repr` tells them apart, as it tells `1` from `1.0`.
        slot = self.pool_slot(value)
        target = self.temporary()
        self.emit(f"Py_IncRef({slot});", indent)
        self.emit(f"{target} = {slot};", indent)
        return target

    def pool_slot(self, value) -> str:
        """The static this literal lives in, registered on first mention."""

        key = (type(value).__name__, repr(value))
        entry = self.pooled.get(key)
        if entry is None:
            slot = f"_py2bin_k{len(self.pooled)}"
            self.pooled[key] = (value, slot)
            return slot
        return entry[1]

    def get_attr(self, owner: str, name: str, indent: int) -> str:
        """`owner.name`, through the interned form. Does not release `owner`."""

        target = self.temporary()
        self.emit(
            f"{target} = PyObject_GetAttr({owner}, {self.interned(name)});", indent
        )
        return target

    def set_attr(self, owner: str, name: str, value: str, indent: int) -> str:
        """`owner.name = value`, answering the C variable holding the status."""

        outcome = self.temporary_flag()
        self.emit(
            f"{outcome} = PyObject_SetAttr({owner}, {self.interned(name)}, "
            f"{value});",
            indent,
        )
        return outcome

    def attribute(self, node: ast.Attribute, indent: int) -> str:
        value, owned = self.operand(node.value, indent)
        target = self.get_attr(value, node.attr, indent)
        self.release(value, owned, indent)
        return self.checked(target, indent)

    def identity(self, node: ast.Compare, indent: int) -> str:
        """`a is b` - the same object, which is the same pointer.

        No C-API call is needed or wanted: identity is what the pointers say,
        and asking RichCompare would answer a different question.
        """

        left = self.expression(node.left, indent)
        right = self.expression(node.comparators[0], indent)
        decision = self.temporary_flag()
        operator = "==" if isinstance(node.ops[0], ast.Is) else "!=";
        self.emit(f"{decision} = ({left} {operator} {right});", indent)
        self.emit(f"Py_DecRef({left});", indent)
        self.emit(f"Py_DecRef({right});", indent)
        target = self.temporary()
        self.emit(f"if ({decision}) {{", indent)
        self.emit(
            f'    {target} = PyObject_GetAttrString(_py2bin_builtins, "True");', indent
        )
        self.emit("} else {", indent)
        self.emit(
            f'    {target} = PyObject_GetAttrString(_py2bin_builtins, "False");', indent
        )
        self.emit("}", indent)
        return self.checked(target, indent)

    def isinstance_verdict(self, node: ast.Call, indent: int) -> str | None:
        """`isinstance(x, C)` as a C int, or None if this call is not that.

        `PyObject_IsInstance` is what the builtin does, so going straight to
        it skips finding the callable and dispatching through it - and in a
        condition the 1/0/-1 it answers is already the verdict, with no
        `True` to build and nothing to ask what it meant.

        Only when the program has not bound the name, as with `len` and
        `str`. The class argument is still looked up live: replacing
        `builtins.isinstance` is one thing, and a program naming its own
        class is another.
        """

        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "isinstance"
            or node.keywords
            or len(node.args) != 2
            or any(isinstance(item, ast.Starred) for item in node.args)
            or not self.builtin_untouched("isinstance")
        ):
            return None
        value, value_owned = self.operand(node.args[0], indent)
        classes, classes_owned = self.operand(node.args[1], indent)
        decision = self.temporary_flag()
        self.emit(
            f"{decision} = PyObject_IsInstance({value}, {classes});", indent
        )
        self.release(value, value_owned, indent)
        self.release(classes, classes_owned, indent)
        self.emit(f"if ({decision} < 0) {{ {self.failure()} }}", indent)
        return decision

    def membership_verdict(self, node: ast.Compare, indent: int) -> str:
        """`x in xs` as a C int, which is what `PySequence_Contains` answers.

        The -1 is a failure and not a false, so it is tested for separately;
        treating it as false would turn a raised exception into an answer.
        """

        value, value_owned = self.operand(node.left, indent)
        container, container_owned = self.operand(node.comparators[0], indent)
        decision = self.temporary_flag()
        self.emit(f"{decision} = PySequence_Contains({container}, {value});", indent)
        self.release(value, value_owned, indent)
        self.release(container, container_owned, indent)
        self.emit(f"if ({decision} < 0) {{ {self.failure()} }}", indent)
        if isinstance(node.ops[0], ast.NotIn):
            self.emit(f"{decision} = !{decision};", indent)
        return decision

    def membership(self, node: ast.Compare, indent: int) -> str:
        """`x in xs` as a value - `True` or `False` built from the verdict.

        Only where a *value* was asked for. `if x in xs` goes through `truth`
        instead, which stops here: the verdict is already the answer, and
        building `True` out of it - by name, through an attribute lookup on
        the builtins module - only to ask what it meant was most of what the
        test cost.
        """

        decision = self.membership_verdict(node, indent)
        target = self.temporary()
        self.emit(f"if ({decision}) {{", indent)
        self.emit(
            f'    {target} = PyObject_GetAttrString(_py2bin_builtins, "True");', indent
        )
        self.emit("} else {", indent)
        self.emit(
            f'    {target} = PyObject_GetAttrString(_py2bin_builtins, "False");', indent
        )
        self.emit("}", indent)
        return self.checked(target, indent)

    def call(self, node: ast.Call, indent: int) -> str:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "super"
            and not node.args
            and not node.keywords
            and self.methods_of
            and self.builtin_untouched("super")
        ):
            # `super()` is `super(__class__, self)`, and CPython supplies those
            # two through a cell it creates for any method that mentions the
            # name. A compiled method has no cell, so the arguments are written
            # out here instead - the same two values, named rather than
            # implied. The class is read when the method runs, by which time it
            # exists; at the moment the method is written it does not yet.
            owner, first = self.methods_of[-1]
            node = ast.copy_location(
                ast.Call(
                    func=node.func,
                    args=[
                        ast.copy_location(ast.Name(id=owner, ctx=ast.Load()), node),
                        ast.copy_location(ast.Name(id=first, ctx=ast.Load()), node),
                    ],
                    keywords=[],
                ),
                node,
            )
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in ("locals", "vars", "globals", "dir")
            and not node.args
            and not node.keywords
            and self.builtin_untouched(node.func.id)
        ):
            return self.scope_dictionary(node, indent)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in ("eval", "exec")
            and len(node.args) == 1
            and not node.keywords
            and self.builtin_untouched(node.func.id)
        ):
            # Given one argument these read the caller's globals and locals
            # out of its frame, and a compiled function has no frame. It
            # failed at run time with "must be given globals and locals",
            # which is true and says nothing about what to do. Given them
            # both it works, and that is what to do.
            # Given one argument these read the caller's frame. A compiled
            # function has none, so the module's own globals are passed
            # instead - which is what the frame of a module-level `eval` would
            # have answered with anyway.
            node = ast.copy_location(
                ast.Call(
                    func=node.func,
                    args=[
                        node.args[0],
                        ast.copy_location(
                            ast.Call(
                                func=ast.Name(id="globals", ctx=ast.Load()),
                                args=[],
                                keywords=[],
                            ),
                            node,
                        ),
                    ],
                    keywords=[],
                ),
                node,
            )
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "_py2bin_set_handled"
            and len(node.args) == 1
        ):
            # Written by the `except*` rewrite, and by nothing else. A clause
            # runs with what *it* matched on record as the exception being
            # handled - the subgroup, not the whole of what the body raised -
            # which is what `sys.exc_info()` answers with there. There is no
            # spelling for that in Python, so the rewrite asks for it by name
            # and this is where the name is honoured.
            value = self.expression(node.args[0], indent)
            self.emit(f"PyErr_SetHandledException({value});", indent)
            self.emit(f"Py_DecRef({value});", indent)
            return self.builtin("None", indent)
        if isinstance(node.func, ast.Attribute):
            return self.method_call(node, indent)
        if not isinstance(node.func, ast.Name):
            # `fs[0](10)`, `(lambda x: x)(2)` - the thing being called is an
            # ordinary expression. Now that a closure is a value like any
            # other, this is how one held in a list gets called.
            callable_value = self.expression(node.func, indent)
            target = self.invoke(callable_value, node.args, indent, node.keywords)
            self.emit(f"Py_DecRef({callable_value});", indent)
            return target
        # `len(x)` and `str(x)` have entry points of their own, and going
        # straight to them skips building the callable and dispatching through
        # it. Only when the program has not bound the name itself, though: a
        # module that defines its own `len` was still getting `PyObject_Size`,
        # so it printed the length where CPython printed what the program's
        # own function returned - a wrong answer, silently.
        shortcut = self.builtin_untouched(node.func.id)
        if shortcut and node.func.id in ("len", "str"):
            # `len` and `str` go straight to `PyObject_Size` and
            # `PyObject_Str` whenever the *program* has not bound the name.
            # Unlike `print`, they are not checked against `builtins` at run
            # time: the check is a dictionary probe on every call, and these
            # two appear in the innermost loops a program has - it measured a
            # fifth of the time on a loop that calls both. Replacing
            # `builtins.len` is also a different sort of act from replacing
            # `builtins.print`: harnesses capture output routinely, and
            # nothing replaces `len` without breaking the interpreter's own
            # machinery along with it. The trade is stated in the README
            # beside the row it costs.
            spelled = node.func.id
            if len(node.args) != 1:
                raise self.fail(node, f"{spelled}() takes one argument")
            value, owned = self.operand(node.args[0], indent)
            target = self.temporary()
            if spelled == "len":
                measured = self.machine_slot()
                # `PyObject_Size` answers -1 with the exception set when the
                # object has no length; boxing that unchecked answered -1
                # where `len(5)` raises.
                self.emit(f"{measured} = PyObject_Size({value});", indent)
                self.release(value, owned, indent)
                self.emit(f"if ({measured} < 0) {{ {self.failure()} }}", indent)
                self.emit(
                    f"{target} = PyLong_FromLongLong({measured});", indent
                )
            else:
                self.emit(f"{target} = PyObject_Str({value});", indent)
                self.release(value, owned, indent)
            return self.checked(target, indent)
        # A spread has to go through the callable form: the direct C call
        # passes a fixed number of arguments and cannot express one whose
        # count is not known until it runs.
        spread = any(isinstance(item, ast.Starred) for item in node.args)
        shadowed_here = (
            self.current is not None and node.func.id in self.current.shadows
        )
        # Not yet defined where this call can run. `self.reached` is the
        # module-body statement whose code is being written, which for a
        # function body is that function's own `def` - so a call inside `f`
        # to a `g` defined below `f` is refused, since the module could call
        # `f` in between and Python would raise NameError there.
        unborn = self.defined_at.get(node.func.id, 0) > self.reached
        # Which parameter each named argument is for, worked out here rather
        # than at run time. A keyword used to force the callable form on its
        # own, which cost far more than the name: the argument-binding
        # trampoline ran per call, and - because the call went through a
        # PyObject - the loop around it could not hold anything in a machine
        # register either.
        placed: list[int] | None = None
        if (
            node.keywords
            and not spread
            and not shadowed_here
            and not unborn
            and node.func.id in self.known_parameters
        ):
            placed = self.placed_keywords(node)
        indirect = spread or (bool(node.keywords) and placed is None)
        if (
            node.func.id not in self.known_functions
            or indirect
            or shadowed_here
            or unborn
        ):
            assert self.current is not None
            if self.reference(node.func.id) is not None:
                # A name the program bound - an imported one, or a value
                # holding something callable. It wins over the builtin of the
                # same spelling, exactly as it does in Python.
                #
                # Borrowed where the slot allows it. A closure held in a local
                # was incremented and decremented around every call to it,
                # which is two writes per call for a slot this function owns
                # outright. `borrowable` still refuses a global, because
                # anything the call runs could rebind one.
                callable_value, callable_owned = self.operand(
                    ast.copy_location(
                        ast.Name(id=node.func.id, ctx=ast.Load()), node
                    ),
                    indent,
                )
            elif self.globals_in_dict:
                # The module keeps its globals in the module's dictionary, so
                # a name this compiler never saw bound may still be there -
                # `from m import *` puts them there at run time. Reading it
                # as a name looks in the dictionary first and falls back to
                # the builtins, where reaching for the builtins directly
                # skipped everything a spread had bound and answered with a
                # NameError for `sqrt`.
                callable_value = self.name(
                    ast.copy_location(
                        ast.Name(id=node.func.id, ctx=ast.Load()), node
                    ),
                    indent,
                )
                callable_owned = True
            else:
                # Not one of ours, so ask the interpreter for it.
                callable_value = self.program_name(node.func.id, indent)
                callable_owned = True
            target = self.invoke(callable_value, node.args, indent, node.keywords)
            self.release(callable_value, callable_owned, indent)
            return target
        expected, defaulted = self.known_functions[node.func.id]
        if placed is None:
            if not expected - defaulted <= len(node.args) <= expected:
                raise self.fail(
                    node,
                    f"{node.func.id}() takes {expected - defaulted} to "
                    f"{expected} argument(s), {len(node.args)} given",
                )
            arguments = [self.expression(item, indent) for item in node.args]
            # A parameter the call leaves out is passed as NULL and the callee
            # puts its default in - evaluated there rather than here, so the
            # default expression exists once however many call sites there are.
            arguments.extend(["(PyObject *)0"] * (expected - len(node.args)))
        else:
            # Evaluated in the order written, then put where each one belongs.
            # Python evaluates `f(g(), k=h())` as g() then h() whatever order
            # the parameters are in, and a keyword cannot precede a positional
            # argument in the grammar, so writing the positional ones first is
            # writing them in source order.
            slotted: dict[int, str] = {}
            for index, item in enumerate(node.args):
                slotted[index] = self.expression(item, indent)
            for keyword, slot in zip(node.keywords, placed):
                slotted[slot] = self.expression(keyword.value, indent)
            # A gap is left NULL exactly as a missing tail is: `f(1, c=9)` on
            # `def f(a, b=2, c=3)` passes NULL for b, and the callee fills it.
            arguments = [
                slotted.get(index, "(PyObject *)0") for index in range(expected)
            ]
        target = self.temporary()
        self.emit(
            f"{target} = f_{self.prefix}{node.func.id}({', '.join(arguments)});",
            indent,
        )
        for argument in arguments:
            # An omitted parameter went in as NULL, which is not a reference
            # and must not be released.
            if argument != "(PyObject *)0":
                self.emit(f"Py_DecRef({argument});", indent)
        return self.checked(target, indent)

    def placed_keywords(self, node: ast.Call) -> list[int] | None:
        """Which parameter each named argument fills, or None if it cannot say.

        Answering None is always safe: the call goes through the callable
        instead, which is where it went for every keyword before this existed.
        So every case this cannot settle - and every case Python would raise
        for - is refused here rather than guessed at, and the interpreter
        raises it at run time with the wording it has always used.
        """
        parameters, positional_only = self.known_parameters[node.func.id]
        expected, defaulted = self.known_functions[node.func.id]
        if len(node.args) > expected:
            return None
        filled = [index < len(node.args) for index in range(expected)]
        # A positional-only parameter cannot be reached by name. These
        # functions have no `**kwargs` for the name to fall into, so Python
        # raises TypeError - let it.
        nameable = parameters[positional_only:]
        placed: list[int] = []
        for keyword in node.keywords:
            if keyword.arg is None:
                # `**mapping` - what it holds is not known until it runs.
                return None
            if keyword.arg not in nameable:
                return None
            slot = parameters.index(keyword.arg)
            if filled[slot]:
                # Given twice: once by position and once by name. Python calls
                # that "multiple values for argument", and the callable form
                # is what says so.
                return None
            filled[slot] = True
            placed.append(slot)
        if not all(filled[: expected - defaulted]):
            # A parameter with no default left unfilled. It would arrive NULL,
            # and the callee increments every parameter it was given.
            return None
        return placed

    def invoke_spread(
        self, callable_value: str, args: list, keywords: list, indent: int
    ) -> str:
        """`f(a, *rest, k=1, **more)` - the argument count is not known here.

        The positional part is gathered in a list, because a list is the thing
        that grows: `*rest` extends it with whatever the object yields, which
        is the same iteration Python does. `tuple()` of it is what
        `PyObject_Call` wants. The keywords are a dict for the same reason -
        `**more` updates it, and a later key wins, as it does in Python.
        """

        gathered = self.temporary()
        self.emit(f"{gathered} = PyList_New(0LL);", indent)
        self.checked(gathered, indent)
        for item in args:
            if isinstance(item, ast.Starred):
                value = self.expression(item.value, indent)
                self.call_method(gathered, "extend", [value], indent)
            else:
                value = self.expression(item, indent)
                self.emit(f"PyList_Append({gathered}, {value});", indent)
            self.emit(f"Py_DecRef({value});", indent)
        maker = self.builtin("tuple", indent)
        holder = self.temporary()
        self.emit(f"{holder} = PyObject_CallOneArg({maker}, {gathered});", indent)
        self.emit(f"Py_DecRef({maker});", indent)
        self.emit(f"Py_DecRef({gathered});", indent)
        self.checked(holder, indent)
        mapping = self.temporary()
        self.emit(f"{mapping} = PyDict_New();", indent)
        self.checked(mapping, indent)
        for keyword in keywords:
            value = self.expression(keyword.value, indent)
            if keyword.arg is None:
                self.call_method(mapping, "update", [value], indent)
            else:
                key = self.temporary()
                self.emit(
                    f"{key} = PyUnicode_FromString({_c_string(keyword.arg)});", indent
                )
                self.checked(key, indent)
                self.emit(f"PyDict_SetItem({mapping}, {key}, {value});", indent)
                self.emit(f"Py_DecRef({key});", indent)
            self.emit(f"Py_DecRef({value});", indent)
        target = self.temporary()
        self.emit(
            f"{target} = PyObject_Call({callable_value}, {holder}, {mapping});", indent
        )
        self.emit(f"Py_DecRef({holder});", indent)
        self.emit(f"Py_DecRef({mapping});", indent)
        return self.checked(target, indent)

    def call_method(
        self, owner: str, name: str, arguments: list[str], indent: int
    ) -> None:
        """Call a method of an already-evaluated object for its effect."""

        method = self.temporary()
        self.emit(
            f"{method} = PyObject_GetAttr({owner}, {self.interned(name)});", indent
        )
        self.checked(method, indent)
        outcome = self.temporary()
        if len(arguments) == 1:
            self.emit(
                f"{outcome} = PyObject_CallOneArg({method}, {arguments[0]});", indent
            )
        else:
            self.emit(f"{outcome} = PyObject_CallNoArgs({method});", indent)
        self.emit(f"Py_DecRef({method});", indent)
        self.checked(outcome, indent)
        self.emit(f"Py_DecRef({outcome});", indent)

    def invoke_with_keywords(
        self, callable_value: str, args: list, keywords: list, indent: int
    ) -> str:
        """`f(a, key=b)` - one array, with the names in a tuple beside it.

        This built a tuple for the positional part *and* a dict for the rest,
        and made the keyword's name from a C string on every call - two
        allocations and a string build to pass one argument by name. It
        measured 0.13x the interpreter, the worst shape found anywhere.

        Vectorcall takes the keyword values in the same array as the
        positional ones with their names in a tuple alongside, which is what
        CPython itself does and what every compiled function here already
        accepts - the wrapper's signature ends in `kwnames`. The names at a
        call site never change, so that tuple is built once at start-up.
        """

        for keyword in keywords:
            if keyword.arg is None:
                raise self.fail(
                    keyword.value, "`**kwargs` at a call is not translated here yet"
                )
        held = [self.operand(item, indent) for item in args]
        held += [self.operand(keyword.value, indent) for keyword in keywords]
        holder = self.argument_array(len(held))
        for position, (value, _owned) in enumerate(held):
            self.emit(f"{holder}[{position}] = {value};", indent)
        names = self.keyword_names(tuple(keyword.arg for keyword in keywords))
        target = self.temporary()
        # `nargsf` counts only the positional ones; the rest are read off the
        # end of the array by name. Vectorcall borrows every argument, so
        # each goes back rather than being given away.
        self.emit(
            f"{target} = PyObject_Vectorcall({callable_value}, {holder}, "
            f"{len(args)}LL, {names});",
            indent,
        )
        for value, owned in held:
            self.release(value, owned, indent)
        return self.checked(target, indent)

    def method_call(self, node: ast.Call, indent: int) -> str:
        """`x.f()` and `x.f(a)` - the two arities the vetted set can call.

        More arguments need PyObject_Call and a tuple to put them in, and
        neither is in the vetted set, so they are refused by name rather than
        approximated.
        """

        assert isinstance(node.func, ast.Attribute)
        plain = not node.keywords and not any(
            isinstance(item, ast.Starred) for item in node.args
        )
        if plain:
            if (
                node.func.attr == "append"
                and len(node.args) == 1
                and self.is_exact_list(node.func.value)
            ):
                # The name always holds an exact `list` - decided from its
                # bindings, so unlike the run-time guard this replaces, the
                # knowledge costs nothing per call. The lookup, the bound
                # method and both dispatch layers all go.
                owner, owned = self.operand(node.func.value, indent)
                value, value_owned = self.operand(node.args[0], indent)
                outcome = self.temporary_flag()
                self.emit(f"{outcome} = PyList_Append({owner}, {value});", indent)
                self.release(owner, owned, indent)
                self.release(value, value_owned, indent)
                self.emit(f"if ({outcome} < 0) {{ {self.failure()} }}", indent)
                return self.builtin("None", indent)
            return self.method_vectorcall(node, indent)
        callable_value = self.attribute(node.func, indent)
        target = self.invoke(callable_value, node.args, indent, node.keywords)
        self.emit(f"Py_DecRef({callable_value});", indent)
        return self.checked(target, indent)

    def builtin_untouched(self, name: str) -> bool:
        """True when this name can only mean the builtin, here and now.

        A program that binds the name anywhere this could see - a module
        global, a local, an enclosing scope, a comprehension target - gets its
        own binding, exactly as it would from the interpreter.
        """

        return (
            name not in self.known_functions
            and self.reference(name) is None
            and name not in self.globals
            and not any(name in scope for scope in self.shadowed)
            and not (self.current is not None and name in self.current.shadows)
        )

    def hoisted_lengths(self, value: ast.expr, indent: int) -> ast.expr | None:
        """`len(name)` subexpressions loaded into machine slots, or None.

        `n = n + len(s)` was three heap allocations to add a machine integer
        the C already had: `PyObject_Size` answers a `long long`, which was
        boxed, added to a boxed `n`, and the result stored as an object -
        which then unmade `n`'s register form for the rest of the loop. Each
        `len(name)` becomes a synthetic name of the emitter's own, loaded once
        here, so the surrounding expression stays narrow and the slow arm -
        which re-evaluates the tree - reads the slot rather than measuring
        again. That re-evaluation is why this exists at all: substituting the
        size straight into both arms of the fast path would run a program's
        `__len__` twice whenever the fast arm declined.

        Only an expression that is effect-free apart from the measurements is
        rewritten, because the loads run first: in `f() + len(s)` the measure
        would happen before `f`, and `f` may rebind `s`.
        """

        if self.boxing or self.current is None:
            return None
        measured: list[tuple[str, ast.Name]] = []

        def walk(node: ast.expr) -> bool:
            if isinstance(node, ast.Constant):
                return True
            if isinstance(node, ast.Name):
                return isinstance(node.ctx, ast.Load)
            if isinstance(node, ast.BinOp) and isinstance(node.op, _MACHINE_OPS):
                return walk(node.left) and walk(node.right)
            if (
                isinstance(node, ast.Compare)
                and len(node.ops) == 1
                and isinstance(node.ops[0], _MACHINE_TESTS)
            ):
                return walk(node.left) and walk(node.comparators[0])
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "len"
                and len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Name)
                and isinstance(node.args[0].ctx, ast.Load)
                and self.builtin_untouched("len")
            ):
                return True
            return False

        class _Load(ast.NodeTransformer):
            def visit_Call(inner, node: ast.Call) -> ast.AST:
                self.comp_serial += 1
                unique = f"_py2bin_n{self.comp_serial}"
                measured.append((unique, node.args[0]))
                return ast.copy_location(
                    ast.Name(id=unique, ctx=ast.Load()), node
                )

        if not walk(value):
            return None
        import copy as _copy

        rewritten = ast.fix_missing_locations(
            _Load().visit(_copy.deepcopy(value))
        )
        if not measured:
            return None
        for unique, argument in measured:
            self.current.unboxed.add(unique)
            self.synthetic.add(unique)
            self.current.certain.add(unique)
            held, obj, state = self.narrow_slots(unique)
            slot, owned = self.operand(argument, indent)
            self.emit(f"if ({obj}) {{ Py_DecRef({obj}); {obj} = 0; }}", indent)
            self.emit(f"{held} = PyObject_Size({slot});", indent)
            self.emit(f"if ({held} < 0) {{ {self.failure()} }}", indent)
            self.emit(f"{state} = 1;", indent)
            self.release(slot, owned, indent)
        return rewritten

    def method_vectorcall(self, node: ast.Call, indent: int) -> str:
        """`x.f(a)` without ever building the bound method it looks up.

        Fetching `x.f` and then calling it allocates a bound method whose only
        job is to remember `x` until the call one line later reads it back out.
        `PyObject_VectorcallMethod` is handed the name and an argument array
        whose first element is `x`, and finds the function on the type without
        pairing it with anything - which is what CPython does for a method call
        it has not been asked to store.
        """

        assert isinstance(node.func, ast.Attribute)
        owner, owner_owned = self.operand(node.func.value, indent)
        held = [self.operand(item, indent) for item in node.args]
        array = self.argument_array(len(held) + 1)
        self.emit(f"{array}[0] = {owner};", indent)
        for position, (value, _) in enumerate(held, start=1):
            self.emit(f"{array}[{position}] = {value};", indent)
        target = self.temporary()
        self.emit(
            f"{target} = PyObject_VectorcallMethod("
            f"{self.interned(node.func.attr)}, {array}, {len(held) + 1}LL, 0);",
            indent,
        )
        self.release(owner, owner_owned, indent)
        for value, owned in held:
            self.release(value, owned, indent)
        return self.checked(target, indent)

    def invoke(
        self, callable_value: str, args: list, indent: int, keywords: list = ()
    ) -> str:
        """Call something with any number of arguments.

        Nought and one positional arguments have their own entry points; beyond
        that, and whenever there are keywords, the arguments go into a tuple and
        the keywords into a dict for PyObject_Call. PyTuple_SetItem *steals* the
        reference it is given, which is why nothing is released after it -
        releasing again would be a second drop of a reference this code no
        longer owns. PyDict_SetItem does not steal, so those are released.
        """

        if any(isinstance(item, ast.Starred) for item in args) or any(
            keyword.arg is None for keyword in keywords
        ):
            return self.invoke_spread(callable_value, args, keywords, indent)
        if keywords:
            return self.invoke_with_keywords(callable_value, args, keywords, indent)
        target = self.temporary()
        if not args:
            self.emit(f"{target} = PyObject_CallNoArgs({callable_value});", indent)
            return self.checked(target, indent)
        if len(args) == 1:
            # Borrowed, like the two-or-more path below and for the same
            # reason: `PyObject_CallOneArg` borrows its argument, so a local
            # or parameter already holding a reference needs no second one.
            # This path took one anyway - an increment and a decrement per
            # call, on the commonest call shape there is, arriving back where
            # it started.
            argument, owned = self.operand(args[0], indent)
            self.emit(
                f"{target} = PyObject_CallOneArg({callable_value}, {argument});", indent
            )
            self.release(argument, owned, indent)
            return self.checked(target, indent)
        held = [self.operand(item, indent) for item in args]
        values = [value for value, _ in held]
        # A plain array rather than a tuple. Building a tuple meant an
        # allocation, a fill and a free for every call a program makes, which
        # measured four times slower than the same call under the interpreter -
        # the interpreter has not used the tuple protocol for years.
        # Vectorcall *borrows* its arguments, so each is released here rather
        # than being given away as PyTuple_SetItem would.
        holder = self.argument_array(len(values))
        for position, value in enumerate(values):
            self.emit(f"{holder}[{position}] = {value};", indent)
        self.emit(
            f"{target} = PyObject_Vectorcall({callable_value}, {holder}, "
            f"{len(values)}LL, 0);",
            indent,
        )
        for value, owned in held:
            self.release(value, owned, indent)
        return self.checked(target, indent)

    def argument_array(self, arity: int) -> str:
        """The array a call of this many arguments passes.

        Not offered to the callee as writable. `PY_VECTORCALL_ARGUMENTS_OFFSET`
        says a callee may put a value in `args[-1]`, which is what a bound
        method wants to do with `self`, and passing it was measured and taken
        back out: CPython only allocates for that when the call has more
        arguments than its small stack buffer holds, so almost never, while the
        flagged count is a full 64-bit immediate that this compiler has to
        materialise at every call site. It cost about seven nanoseconds a call
        and saved nothing.
        """

        assert self.current is not None
        existing = self.current.argument_arrays.get(arity)
        if existing is None:
            existing = f"_args{arity}"
            self.current.argument_arrays[arity] = existing
            self.current.locals.append(f"{existing}[{arity}]")
        return existing

    # --- statements ------------------------------------------------------

    def statement(self, node: ast.stmt, indent: int) -> None:
        assert self.current is not None
        self.depth += 1
        mark = self.current.temporaries
        machined = self.current.machines
        try:
            self.write_statement(node, indent)
            if self.depth == 1:
                # Directly in the body, so it ran: anything it bound is
                # settled from here on and a later read needs no test.
                self.current.certain.update(self.settles(node))
        finally:
            self.depth -= 1
            if self.depth == 0:
                # Back at the top of a body: everything this statement needed
                # is finished with, so the slots go back. Anything a construct
                # holds across the statements *inside* it - a `try` keeping the
                # classes it catches, a `finally` keeping what it will return -
                # is at a greater depth and is not wound back under it.
                self.current.temporaries = mark
                self.current.machines = machined

    @contextlib.contextmanager
    def settled_within(self, names):
        """Treat these names as bound for the duration of a nested body.

        A `for` target may leave the loop unbound - the sequence can be empty -
        but *inside* the body it has just been assigned, and so has the name a
        `with ... as` or an `except ... as` introduced. Testing those would put
        an unbound-name check on the most common read in the language.
        """

        assert self.current is not None
        fresh = {name for name in names if name not in self.current.certain}
        self.current.certain.update(fresh)
        try:
            yield
        finally:
            self.current.certain.difference_update(fresh)

    @staticmethod
    def settles(node: ast.stmt) -> set[str]:
        """Names this statement binds whenever it runs at all.

        A `for` target is not among them: the loop may run no times. Nor is
        anything inside an `if`, a `try` or a loop body - those are what the
        depth test outside this already excludes.
        """

        bound: set[str] = set()
        if isinstance(node, ast.AnnAssign):
            if node.value is not None and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for element in (
                    target.elts
                    if isinstance(target, (ast.Tuple, ast.List))
                    else [target]
                ):
                    if isinstance(element, ast.Name):
                        bound.add(element.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        return bound

    def write_statement(self, node: ast.stmt, indent: int) -> None:
        if isinstance(node, ast.Assign):
            self.assignment(node, indent)
        elif isinstance(node, ast.Expr):
            self.expression_statement(node, indent)
        elif isinstance(node, ast.If):
            self.conditional(node, indent)
        elif isinstance(node, ast.While):
            self.loop(node, indent)
        elif isinstance(node, ast.For):
            self.for_loop(node, indent)
        elif isinstance(node, ast.Return):
            self.give_back(node, indent)
        elif isinstance(node, ast.Import):
            self.import_module(node, indent)
        elif isinstance(node, ast.ImportFrom):
            self.import_names(node, indent)
        elif isinstance(node, ast.Delete):
            self.remove(node, indent)
        elif isinstance(node, ast.Raise):
            self.throw(node, indent)
        elif isinstance(node, ast.With):
            self.with_block(node, indent)
        elif isinstance(node, ast.Try):
            self.guarded(node, indent)
        elif isinstance(node, ast.AugAssign):
            self.augmented(node, indent)
        elif isinstance(node, ast.Break):
            if self.finallys and self.finallys[-1].loop_depth == self.loop_depth:
                self.leave_through_finally(_BREAKING, indent)
            else:
                self.mark_broken(indent)
                self.emit("break;", indent)
        elif isinstance(node, ast.Continue):
            if self.finallys and self.finallys[-1].loop_depth == self.loop_depth:
                self.leave_through_finally(_CONTINUING, indent)
            else:
                self.emit("continue;", indent)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            if isinstance(node, ast.Nonlocal):
                raise self.fail(
                    node,
                    "`nonlocal` is not translated here yet; a closure captures "
                    "by value, so there is no cell to rebind",
                )
            assert self.current is not None
            for name in node.names:
                self.current.module_names.add(name)
                self.note_global(name)
        elif isinstance(node, ast.AnnAssign):
            # `x: int = 5` is an assignment with a note attached, and the note
            # is not something the program can observe here. `x: int` on its
            # own binds nothing at all - it only tells a reader, and a type
            # checker, what to expect.
            if node.value is not None:
                self.assignment(
                    ast.copy_location(
                        ast.Assign(targets=[node.target], value=node.value), node
                    ),
                    indent,
                )
        elif isinstance(node, ast.Match):
            self.match_statement(node, indent)
        elif isinstance(node, ast.Assert):
            self.check(node, indent)
        elif isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.ClassDef):
            self.class_definition(node, indent)
        elif isinstance(node, ast.FunctionDef):
            # The slot is declared *before* the closure is built. A nested
            # function that calls itself reads its own name, and a name with
            # no slot yet is not seen as a capture at all - which is why a
            # nested `fact` raised `NameError` on its first recursive call.
            # Declaring first makes the capture visible; `make_closure` fills
            # it in once the callable exists.
            target = self.declare(node.name)
            value = self.make_closure(node, node.name, indent)
            value = self.apply_decorators(node.decorator_list, value, indent)
            self.emit(f"if ({target}) Py_DecRef({target});", indent)
            self.emit(f"{target} = {value};", indent)
            self.publish(node.name, target, indent)
        else:
            raise self.fail(
                node, f"{type(node).__name__} has no C-API translation here yet"
            )

    def import_module(self, node: ast.Import, indent: int) -> None:
        """`import x` - the module object, bound to a name like any value.

        This is where the tier earns its cost. The interpreter is present and
        its import machinery works, so a compiled program can reach anything
        installed beside it - which is the whole reason to pay for libpython.
        """

        for alias in node.names:
            # `import a.b` imports a.b and binds *a*; `import a.b as c` binds
            # the submodule itself. PyImport_ImportModule answers the tail
            # module either way, so the plain form asks for the package again
            # - which costs nothing, the first import having put both in
            # sys.modules.
            bound = alias.asname or alias.name.split(".")[0]
            if alias.asname is None and "." in alias.name:
                loaded = self.temporary()
                self.emit(
                    f"{loaded} = PyImport_ImportModule({_c_string(alias.name)});",
                    indent,
                )
                self.checked(loaded, indent)
                self.emit(f"Py_DecRef({loaded});", indent)
                wanted = alias.name.split(".")[0]
            else:
                wanted = alias.name
            target = self.declare(bound)
            self.emit(f"if ({target}) Py_DecRef({target});", indent)
            self.emit(
                f'{target} = PyImport_ImportModule({_c_string(wanted)});', indent
            )
            self.checked(target, indent)
            self.publish(bound, target, indent)

    def absolute_module(self, node: ast.ImportFrom) -> str:
        """The module `from ... import` names, with any dots resolved away.

        A relative import is relative to where the importing module *is*, and
        that is fixed once it has been compiled - so the dots can be counted
        here rather than carried into the binary. This is the arithmetic the
        import system does at run time from `__package__`: one dot means the
        package the module is in, and each further dot means one package up.

        Resolving it at compile time also means no new entry point. Asking the
        interpreter to do it needs `PyImport_ImportModuleLevelObject`, with a
        globals mapping to read `__package__` out of; spelling the answer out
        needs only the `PyImport_ImportModule` already used by every absolute
        import.
        """

        if not node.level:
            if node.module is None:
                raise self.fail(node, "this import form is not translated here yet")
            return node.module
        if not self.module_package:
            # The message CPython gives, because the situation is the same one.
            raise self.fail(
                node, "attempted relative import with no known parent package"
            )
        parts = self.module_package.split(".")
        if node.level - 1 > len(parts) - 1:
            raise self.fail(
                node, "attempted relative import beyond top-level package"
            )
        base = ".".join(parts[: len(parts) - (node.level - 1)])
        return f"{base}.{node.module}" if node.module else base

    def spread_module(self, module: str, indent: int) -> None:
        """`from m import *` - every public name of `m`, bound here.

        Which names those are is `m`'s business and is not known until it has
        been imported: its `__all__` if it has one, and otherwise everything
        it holds that does not begin with an underscore. So there is no set
        of C slots to put them in, and a module that writes one keeps its
        globals in the module's own dictionary instead - which it has to do
        anyway for `globals()`, and which is what makes this a few lines
        rather than a new mechanism.

        The work is done by a function written in Python and compiled with
        the program. It reaches the dictionary through `globals()`, which
        answers with the very one this module's names live in.
        """

        self.needs_spread = True
        maker = self.temporary()
        self.emit(
            f"{maker} = PyObject_GetItem(_py2bin_globals, "
            f"{self.interned('_py2bin_spread')});",
            indent,
        )
        self.checked(maker, indent)
        held = self.temporary()
        self.emit(f"{held} = PyTuple_New(1LL);", indent)
        self.checked(held, indent)
        self.emit(f"Py_IncRef({module});", indent)
        self.emit(f"PyTuple_SetItem({held}, 0, {module});", indent)
        answer = self.temporary()
        self.emit(f"{answer} = PyObject_Call({maker}, {held}, 0);", indent)
        self.emit(f"Py_DecRef({maker});", indent)
        self.emit(f"Py_DecRef({held});", indent)
        self.checked(answer, indent)
        self.emit(f"Py_DecRef({answer});", indent)

    def import_names(self, node: ast.ImportFrom, indent: int) -> None:
        """`from x import a, b` - import the module, then read the names off it.

        A dotted module works because the import machinery is the
        interpreter's: `from scipy import optimize` imports `scipy` and takes
        the attribute, exactly as the statement means.
        """

        spelled = self.absolute_module(node)
        module = self.temporary()
        self.emit(
            f"{module} = PyImport_ImportModule({_c_string(spelled)});", indent
        )
        self.checked(module, indent)
        for alias in node.names:
            if alias.name == "*":
                if not self.at_module_level:
                    # Python refuses one anywhere else, and says so at compile
                    # time: there is nowhere for the names to go.
                    raise self.fail(
                        node, "import * only allowed at module level"
                    )
                self.spread_module(module, indent)
                continue
            target = self.declare(alias.asname or alias.name)
            self.emit(f"if ({target}) Py_DecRef({target});", indent)
            self.emit(
                f"{target} = PyObject_GetAttrString({module}, "
                f"{_c_string(alias.name)});",
                indent,
            )
            # A name that is not an attribute may be a submodule that has not
            # been imported yet - `from matplotlib import pyplot` is exactly
            # that. Python's import system tries the submodule at this point,
            # so this does too, and the failed lookup is cleared first because
            # an exception left set would surface at the next call.
            dotted = f"{spelled}.{alias.name}"
            self.emit(f"if (!{target}) {{", indent)
            self.emit("    PyErr_Clear();", indent)
            self.emit(
                f"    {target} = PyImport_ImportModule({_c_string(dotted)});", indent
            )
            self.emit("}", indent)
            self.checked(target, indent)
            # Through `publish`, like every other binding. Writing the slot
            # and stopping was invisible until `globals()` began answering
            # with the module's real dictionary: reads go through that
            # dictionary in this mode, so a name bound only in its C slot was
            # not there to be read, and `from dataclasses import dataclass`
            # followed by a `globals()` anywhere in the file raised NameError
            # on the decorator. `import x` already went through here; this is
            # the sibling that did not.
            self.publish(alias.asname or alias.name, target, indent)
        self.emit(f"Py_DecRef({module});", indent)

    def assignment(self, node: ast.Assign, indent: int) -> None:
        """`a = v`, and `a = b = v`.

        The value is computed first and each target bound from it, which is
        the order Python uses: the right-hand side, then the targets left to
        right. One value however many names it is given to.
        """

        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if self.narrow_assign(node.targets[0].id, node.value, indent):
                return
        value = self.expression(node.value, indent)
        for position, target in enumerate(node.targets):
            if position + 1 < len(node.targets):
                # Every target but the last gets its own reference; the last
                # one takes the value this already holds.
                self.emit(f"Py_IncRef({value});", indent)
            self.bind_target(target, value, indent)

    def bind_target(self, target: ast.expr, value: str, indent: int) -> None:
        """Give an already-computed value to one assignment target.

        The reference is consumed here, whichever shape the target has.
        """

        if isinstance(target, ast.Name):
            if self.is_unboxed(target.id):
                self.store_object(target.id, value, indent)
                return
            if self.is_double(target.id):
                self.store_double_object(target.id, value, indent)
                return
            slot = self.declare(target.id)
            # The name may already hold something; that reference is released
            # before it is overwritten, which is what keeps a loop from
            # growing.
            self.emit(f"if ({slot}) Py_DecRef({slot});", indent)
            self.emit(f"{slot} = {value};", indent)
            self.publish(target.id, slot, indent)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            self.unpack_value(target, value, indent)
            self.emit(f"Py_DecRef({value});", indent)
            return
        if isinstance(target, ast.Attribute):
            # Borrowed, like the subscript store below it: `PyObject_SetAttr`
            # borrows what it is given, and `obj.field = v` on a local took a
            # reference to the object and dropped it again around the call.
            owner, owner_owned = self.operand(target.value, indent)
            outcome = self.temporary_flag()
            self.emit(
                f"{outcome} = PyObject_SetAttr({owner}, "
                f"{self.interned(target.attr)}, {value});",
                indent,
            )
            self.release(owner, owner_owned, indent)
            self.emit(f"Py_DecRef({value});", indent)
            self.emit(f"if ({outcome} < 0) {{ {self.failure()} }}", indent)
            return
        if isinstance(target, ast.Subscript):
            container, owned = self.operand(target.value, indent)
            if isinstance(target.slice, ast.Slice):
                # `xs[1:3] = ys` is a store through the mapping protocol like
                # any other, with a slice object for the key - which is built
                # the same way reading one builds it. Refusing it refused
                # `xs[:] = ...` and every splice.
                key, key_owned = self.slice_object(target.slice, indent), True
            else:
                key, key_owned = self.operand(target.slice, indent)
            outcome = self.temporary_flag()
            # `d[k] = v` on a name that always holds an exact dict skips the
            # mapping-protocol dispatch; an unhashable key raises through
            # either spelling.
            store = (
                "PyDict_SetItem"
                if not isinstance(target.slice, ast.Slice)
                and self.is_exact_dict(target.value)
                else "PyObject_SetItem"
            )
            self.emit(
                f"{outcome} = {store}({container}, {key}, {value});", indent
            )
            self.release(container, owned, indent)
            self.release(key, key_owned, indent)
            self.emit(f"Py_DecRef({value});", indent)
            self.emit(f"if ({outcome} < 0) {{ {self.failure()} }}", indent)
            return
        raise self.fail(
            target, "only a name, an attribute or a subscript is assigned to here"
        )

    def show_on_module(self, name: str, slot: str, indent: int) -> None:
        """Put a module-level class or function on the module object too.

        `pickle` does not store a class; it stores the path to one, and gets
        it back by importing the module named in `__module__` and looking up
        `__qualname__` on it - then checks that what it found *is* the class.
        py2bin keeps a module's globals in C variables, so the class was
        nowhere on the module and pickling any instance failed with "when
        serializing P object".

        Only a `def` and a `class`, and only at module level: those run once,
        so the extra store costs nothing measurable, and they are the two
        things `pickle` looks up by name. Every other global stays where it
        was - mirroring all of them would put a store in front of every
        assignment to a global, which is a loop's worth of work for something
        almost no program asks for. A module that does ask - one that mentions
        `globals()` - already keeps everything in the module's dictionary.
        """

        if self.prefix or self.globals_in_dict or not slot.startswith("g_"):
            # A linked module has `publish` for this, and a module whose
            # globals live in the dictionary is already there.
            return
        self.emit(
            f'PyObject_SetAttrString(PyImport_AddModule("__main__"), '
            f"{_c_string(name)}, {slot});",
            indent,
        )

    def publish(self, name: str, slot: str, indent: int) -> None:
        """Show a linked module's global on the module object as well.

        Another module reads it as an attribute, and the attribute has to
        follow the slot or it would answer with whatever the value was when
        the body finished. The entry module needs none of this: nothing
        imports `__main__`.
        """

        if self.globals_in_dict and slot.startswith("g_"):
            # The module's dictionary is what a global is read from in this
            # mode, so every binding has to reach it - this is the one place
            # every global binding passes through.
            self.emit(
                f"if (PyObject_SetItem(_py2bin_globals, {self.interned(name)}, "
                f"{slot}) < 0) {{ {self.failure()} }}",
                indent,
            )
        if not self.prefix or not slot.startswith("g_"):
            return
        key = self.prefix[:-1]
        self.emit(
            f"PyObject_SetAttrString(m_{key}, {_c_string(name)}, {slot});", indent
        )

    def check(self, node: ast.Assert, indent: int) -> None:
        """`assert c` and `assert c, message`.

        Always emitted. CPython skips assertions under `-O`, which is a switch
        given to the interpreter at run time; a compiled program has no such
        moment, so the honest thing is to keep them.
        """

        decision = self.temporary_flag()
        test = self.expression(node.test, indent)
        self.emit(f"{decision} = PyObject_IsTrue({test});", indent)
        self.emit(f"Py_DecRef({test});", indent)
        self.emit(f"if ({decision} < 0) {{ {self.failure()} }}", indent)
        self.emit(f"if (!{decision}) {{", indent)
        kind = self.builtin_raw("AssertionError", indent + 1)
        self.checked(kind, indent + 1)
        raised = self.temporary()
        if node.msg is None:
            self.emit(f"{raised} = PyObject_CallNoArgs({kind});", indent + 1)
        else:
            message = self.expression(node.msg, indent + 1)
            self.emit(
                f"{raised} = PyObject_CallOneArg({kind}, {message});", indent + 1
            )
            self.emit(f"Py_DecRef({message});", indent + 1)
        self.checked(raised, indent + 1)
        self.emit(f"PyErr_SetObject({kind}, {raised});", indent + 1)
        self.emit(f"Py_DecRef({kind});", indent + 1)
        self.emit(f"Py_DecRef({raised});", indent + 1)
        self.emit(self.failure(), indent + 1)
        self.emit("}", indent)

    def remove(self, node: ast.Delete, indent: int) -> None:
        """`del xs[k]` - the only del shape with an object-protocol answer.

        `del name` unbinds, and a C local has nothing that records being
        unbound, so it is refused rather than turned into a store of None.
        """

        for target in node.targets:
            if isinstance(target, ast.Name):
                # Emptying the slot *is* how being unbound is recorded, now
                # that a read tests for it: the next one raises NameError or
                # UnboundLocalError, which is what Python does.
                slot = self.reference(target.id)
                if slot is None or slot.startswith(("p_", "c_")):
                    raise self.fail(
                        target,
                        f"`del {target.id}` names something this scope does "
                        "not bind",
                    )
                self.emit(f"if (!{slot}) {{", indent)
                self.needs_unbound = True
                kind = 0 if slot.startswith("g_") else 1
                message = (
                    f"name {target.id!r} is not defined"
                    if kind == 0
                    else f"cannot access local variable {target.id!r} where it "
                    "is not associated with a value"
                )
                self.emit(
                    f"_py2bin_unbound({kind}, "
                    f"PyUnicode_FromString({_c_string(message)}), "
                    f"PyUnicode_FromString({_c_string(target.id)}));",
                    indent + 1,
                )
                self.emit(self.failure(), indent + 1)
                self.emit("}", indent)
                self.emit(f"Py_DecRef({slot});", indent)
                self.emit(f"{slot} = 0;", indent)
                if self.globals_in_dict and slot.startswith("g_"):
                    # Emptying the slot is not enough when reads come from the
                    # dictionary: the entry would stay, so `globals()` listed
                    # a name the program had let go and reading it answered
                    # with the value it was supposed to have lost.
                    self.emit(
                        f"if (PyObject_DelItem(_py2bin_globals, "
                        f"{self.interned(target.id)}) < 0) PyErr_Clear();",
                        indent,
                    )
                assert self.current is not None
                self.current.certain.discard(target.id)
                self.certain_globals.discard(target.id)
                self.certain_at.pop(target.id, None)
                continue
            if isinstance(target, ast.Attribute):
                # Deleting an attribute is setting it to nothing: that is what
                # PyObject_DelAttrString is a spelling of, and the vetted set
                # already has the setter.
                owner = self.expression(target.value, indent)
                outcome = self.temporary_flag()
                self.emit(
                    f"{outcome} = PyObject_SetAttrString({owner}, "
                    f"{_c_string(target.attr)}, 0);",
                    indent,
                )
                self.emit(f"Py_DecRef({owner});", indent)
                self.emit(f"if ({outcome} < 0) {{ {self.failure()} }}", indent)
                continue
            if not isinstance(target, ast.Subscript):
                raise self.fail(
                    target,
                    "only a name, an attribute or `container[key]` is deleted "
                    "here",
                )
            container = self.expression(target.value, indent)
            key = (
                self.slice_object(target.slice, indent)
                if isinstance(target.slice, ast.Slice)
                else self.expression(target.slice, indent)
            )
            outcome = self.temporary_flag()
            self.emit(f"{outcome} = PyObject_DelItem({container}, {key});", indent)
            self.emit(f"Py_DecRef({container});", indent)
            self.emit(f"Py_DecRef({key});", indent)
            self.emit(f"if ({outcome} < 0) {{ {self.failure()} }}", indent)

    def throw(self, node: ast.Raise, indent: int) -> None:
        """`raise E(...)` - set the exception, then take the failure path.

        Setting it is all the C API does; unwinding is the caller's business,
        and here that is the same `goto` a failing call takes.
        """

        if node.cause is not None and node.exc is None:
            raise self.fail(
                node, "`raise ... from ...` needs something to raise"
            )
        if node.exc is None:
            if not self.handling:
                raise self.fail(
                    node,
                    "a bare `raise` outside an except clause has nothing to "
                    "re-raise",
                )
            # The object the enclosing clause caught. The reference belongs to
            # that clause, so this sets it again without giving it away.
            value, owned = self.handling[-1], False
        else:
            if node.cause is None:
                    # `raise ValueError('x')` and the bare `raise ValueError` are how
                # nearly every raise is written, and for an untouched builtin
                # exception name the class-or-instance question below is settled
                # at compile time: a call on the class answers an instance of it,
                # and the bare name is the class. The name is still looked up
                # live - `builtins.ValueError` may have been replaced - which is
                # `program_name`'s job; only the run-time classification goes.
                spelled = (
                    node.exc.func.id
                    if isinstance(node.exc, ast.Call)
                    and isinstance(node.exc.func, ast.Name)
                    and not node.exc.keywords
                    and not any(
                        isinstance(item, ast.Starred) for item in node.exc.args
                    )
                    else node.exc.id if isinstance(node.exc, ast.Name) else None
                )
                named = getattr(builtins, spelled, None) if spelled else None
                if (
                    spelled is not None
                    and self.builtin_untouched(spelled)
                    and isinstance(named, type)
                    and issubclass(named, BaseException)
                ):
                    if isinstance(node.exc, ast.Call):
                        kind = self.program_name(spelled, indent)
                        raised = self.invoke(kind, node.exc.args, indent, [])
                        self.emit(f"PyErr_SetObject({kind}, {raised});", indent)
                        self.emit(f"Py_DecRef({raised});", indent)
                    else:
                        kind = self.program_name(spelled, indent)
                        self.emit(f"PyErr_SetObject({kind}, NULL);", indent)
                    self.emit(f"Py_DecRef({kind});", indent)
                    if self.handlers:
                        self.emit(f"goto {self.handlers[-1]};", indent)
                    else:
                        self.emit(self.failure(), indent)
                    return
            value, owned = self.expression(node.exc, indent), True
        if node.cause is not None:
            value = self.with_cause(node, value, owned, indent)
            owned = True
        # `raise X` names either a class or an instance, and the two do not
        # want the same arguments. For an instance the class is type(it); for a
        # class there is no instance yet, and asking type() for *its* class
        # answers `type`, the metaclass. That is what produced
        #
        #     SystemError: exception <class 'type'> is not a BaseException
        #
        # for the plain `raise ValueError` any Python program writes. A class
        # is handed over on its own instead, which is the shape PyErr_SetObject
        # expects and normalises when it is caught.
        kind = self.builtin("type", indent)
        is_class = self.temporary_flag()
        self.emit(f"{is_class} = PyObject_IsInstance({value}, {kind});", indent)
        classified = self.temporary()
        self.emit(f"if ({is_class} == 1) {{", indent)
        self.emit(f"PyErr_SetObject({value}, NULL);", indent + 1)
        self.emit("} else {", indent)
        self.emit(
            f"{classified} = PyObject_CallOneArg({kind}, {value});", indent + 1
        )
        self.emit(f"if ({classified} != NULL) {{", indent + 1)
        self.emit(f"PyErr_SetObject({classified}, {value});", indent + 2)
        self.emit(f"Py_DecRef({classified});", indent + 2)
        self.emit("}", indent + 1)
        self.emit("}", indent)
        self.emit(f"Py_DecRef({kind});", indent)
        if owned:
            self.emit(f"Py_DecRef({value});", indent)
        if self.handlers:
            self.emit(f"goto {self.handlers[-1]};", indent)
        else:
            self.emit(self.failure(), indent)

    def match_statement(self, node: ast.Match, indent: int) -> None:
        """`match` - the cases in order, the first one that fits.

        Lowered to a chain of tests rather than to anything clever, because
        that is what it is: each case asks whether the subject fits its
        pattern, binds whatever the pattern captures, and runs its body. A
        `case` that has matched stops the rest from being tried, which the
        flag carries - a `goto` past the chain would do as well but would need
        a label per statement.
        """

        subject = self.expression(node.subject, indent)
        done = self.temporary_flag()
        self.emit(f"{done} = 0;", indent)
        for case in node.cases:
            self.emit(f"if (!{done}) {{", indent)
            fits = self.pattern_test(case.pattern, subject, indent + 1)
            self.emit(f"if ({fits}) {{", indent + 1)
            inner = indent + 2
            if case.guard is not None:
                # The guard runs only once the pattern has bound its names,
                # because a guard is allowed to mention them.
                verdict = self.truth(case.guard, inner)
                self.emit(f"if ({verdict}) {{", inner)
                inner += 1
            self.emit(f"{done} = 1;", inner)
            for statement in case.body:
                self.statement(statement, inner)
            if case.guard is not None:
                self.emit("}", indent + 2)
            self.emit("}", indent + 1)
            self.emit("}", indent)
        self.emit(f"Py_DecRef({subject});", indent)

    def pattern_test(self, pattern, subject: str, indent: int) -> str:
        """Emit a test of `subject` against `pattern`; answer a C int flag.

        Whatever the pattern captures is bound as a side effect, before the
        flag is read - a capture that matched is visible to the guard and to
        the body, which is what `case [x, y] if x < y` needs.
        """

        fits = self.temporary_flag()
        if isinstance(pattern, ast.MatchValue):
            wanted = self.expression(pattern.value, indent)
            outcome = self.temporary()
            self.emit(
                f"{outcome} = PyObject_RichCompare({subject}, {wanted}, 2);", indent
            )
            self.emit(f"Py_DecRef({wanted});", indent)
            self.checked(outcome, indent)
            self.emit(f"{fits} = PyObject_IsTrue({outcome});", indent)
            self.emit(f"Py_DecRef({outcome});", indent)
            self.emit(f"if ({fits} < 0) {{ {self.failure()} }}", indent)
            return fits
        if isinstance(pattern, ast.MatchSingleton):
            # `case None` is an identity test, not an equality one, which is
            # the difference between it and `case 0` for a value that is both
            # falsey and not None.
            wanted = self.builtin(repr(pattern.value), indent)
            self.emit(f"{fits} = ({subject} == {wanted});", indent)
            self.emit(f"Py_DecRef({wanted});", indent)
            return fits
        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is None:
                # `case _` and `case name` both always fit; the second binds.
                self.emit(f"{fits} = 1;", indent)
            else:
                inner = self.pattern_test(pattern.pattern, subject, indent)
                self.emit(f"{fits} = {inner};", indent)
            if pattern.name is not None:
                self.emit(f"if ({fits}) {{", indent)
                self.emit(f"Py_IncRef({subject});", indent + 1)
                self.bind_target(
                    ast.Name(id=pattern.name, ctx=ast.Store()), subject, indent + 1
                )
                self.emit("}", indent)
            return fits
        if isinstance(pattern, ast.MatchOr):
            self.emit(f"{fits} = 0;", indent)
            for alternative in pattern.patterns:
                self.emit(f"if (!{fits}) {{", indent)
                inner = self.pattern_test(alternative, subject, indent + 1)
                self.emit(f"{fits} = {inner};", indent + 1)
                self.emit("}", indent)
            return fits
        if isinstance(pattern, ast.MatchSequence):
            return self.sequence_pattern(pattern, subject, fits, indent)
        if isinstance(pattern, ast.MatchMapping):
            return self.mapping_pattern(pattern, subject, fits, indent)
        if isinstance(pattern, ast.MatchClass):
            return self.class_pattern(pattern, subject, fits, indent)
        raise self.fail(
            pattern,
            f"a {type(pattern).__name__[5:].lower()} pattern is not translated "
            "here yet",
        )

    def sequence_pattern(self, pattern, subject: str, fits: str, indent: int) -> str:
        """`case [a, b]` - a list or a tuple of the right length, taken apart.

        Only a list or a tuple. Python matches any sequence except `str` and
        `bytes`, which is a wider test than this makes, so a pattern that would
        have matched some other sequence type does not fit here rather than
        fitting wrongly.
        """

        stars = [
            index
            for index, part in enumerate(pattern.patterns)
            if isinstance(part, ast.MatchStar)
        ]
        if len(stars) > 1:
            raise self.fail(pattern, "two starred names in one pattern")
        kinds = self.temporary()
        # A slot at a time, not PyTuple_Pack: that one is variadic, and Apple's
        # arm64 ABI puts variadic arguments on the stack where this backend
        # passes registers. Called with a fixed prototype it reads two
        # addresses that were never written, which is a segfault rather than a
        # wrong answer. PyTuple_SetItem steals, so neither builtin is released.
        self.emit(f"{kinds} = PyTuple_New(2LL);", indent)
        self.checked(kinds, indent)
        as_list = self.builtin("list", indent)
        as_tuple = self.builtin("tuple", indent)
        self.emit(f"PyTuple_SetItem({kinds}, 0LL, {as_list});", indent)
        self.emit(f"PyTuple_SetItem({kinds}, 1LL, {as_tuple});", indent)
        self.instance_test(subject, kinds, fits, indent)
        length = self.temporary_flag()
        self.emit(f"if ({fits}) {{", indent)
        self.emit(f"{length} = (int)PyObject_Size({subject});", indent + 1)
        # Without a star the length has to match; with one it has to be at
        # least what the fixed names take, and the star gets the difference.
        fixed = len(pattern.patterns) - len(stars)
        self.emit(
            f"{fits} = ({length} {'>=' if stars else '=='} {fixed});", indent + 1
        )
        self.emit("}", indent)
        star = stars[0] if stars else len(pattern.patterns)
        for position, part in enumerate(pattern.patterns):
            self.emit(f"if ({fits}) {{", indent)
            if position == star:
                # The middle, from where the leading names stopped to where
                # the trailing ones begin - counted from the end, so its size
                # does not have to be known here.
                after = len(pattern.patterns) - star - 1
                start = self.temporary()
                stop = self.temporary()
                self.emit(f"{start} = PyLong_FromLongLong({star}LL);", indent + 1)
                self.checked(start, indent + 1)
                self.emit(
                    f"{stop} = PyLong_FromLongLong((long long)({length} - {after}));",
                    indent + 1,
                )
                self.checked(stop, indent + 1)
                none = self.builtin("None", indent + 1)
                cut = self.temporary()
                self.emit(f"{cut} = PySlice_New({start}, {stop}, {none});", indent + 1)
                for value in (start, stop, none):
                    self.emit(f"Py_DecRef({value});", indent + 1)
                self.checked(cut, indent + 1)
                rest = self.temporary()
                self.emit(f"{rest} = PyObject_GetItem({subject}, {cut});", indent + 1)
                self.emit(f"Py_DecRef({cut});", indent + 1)
                self.checked(rest, indent + 1)
                if part.name is not None:
                    listed = self.builtin("list", indent + 1)
                    made = self.temporary()
                    self.emit(
                        f"{made} = PyObject_CallOneArg({listed}, {rest});", indent + 1
                    )
                    self.emit(f"Py_DecRef({listed});", indent + 1)
                    self.emit(f"Py_DecRef({rest});", indent + 1)
                    self.checked(made, indent + 1)
                    self.bind_target(
                        ast.Name(id=part.name, ctx=ast.Store()), made, indent + 1
                    )
                else:
                    self.emit(f"Py_DecRef({rest});", indent + 1)
                self.emit("}", indent)
                continue
            # Positions after the star are counted from the end.
            where = (
                str(position)
                if position < star
                else f"{length} - {len(pattern.patterns) - position}"
            )
            index = self.temporary()
            self.emit(
                f"{index} = PyLong_FromLongLong((long long)({where}));", indent + 1
            )
            self.checked(index, indent + 1)
            picked = self.temporary()
            self.emit(f"{picked} = PyObject_GetItem({subject}, {index});", indent + 1)
            self.emit(f"Py_DecRef({index});", indent + 1)
            self.checked(picked, indent + 1)
            inner = self.pattern_test(part, picked, indent + 1)
            self.emit(f"Py_DecRef({picked});", indent + 1)
            self.emit(f"{fits} = {inner};", indent + 1)
            self.emit("}", indent)
        return fits

    def instance_test(self, subject: str, kinds: str, fits: str, indent: int) -> None:
        """Set `fits` from `isinstance(subject, kinds)`, consuming `kinds`."""

        checker = self.builtin("isinstance", indent)
        array = self.argument_array(2)
        self.emit(f"{array}[0] = {subject};", indent)
        self.emit(f"{array}[1] = {kinds};", indent)
        answer = self.temporary()
        self.emit(f"{answer} = PyObject_Vectorcall({checker}, {array}, 2, 0);", indent)
        self.emit(f"Py_DecRef({checker});", indent)
        self.emit(f"Py_DecRef({kinds});", indent)
        self.checked(answer, indent)
        self.emit(f"{fits} = PyObject_IsTrue({answer});", indent)
        self.emit(f"Py_DecRef({answer});", indent)
        self.emit(f"if ({fits} < 0) {{ {self.failure()} }}", indent)

    def mapping_pattern(self, pattern, subject: str, fits: str, indent: int) -> str:
        """`case {"k": v}` - the named keys present, their values matching.

        A mapping pattern does not care what else is in the mapping, which is
        the difference from a sequence pattern: `{"a": 1}` fits a dict of ten
        keys as long as `a` is one of them. `**rest` collects what the pattern
        did not name.

        `dict` rather than `collections.abc.Mapping`, so a custom mapping does
        not fit here where Python would let it. That is narrower than the
        language and not wider, which is the direction a compiler may err in.
        """

        self.instance_test(subject, self.builtin("dict", indent), fits, indent)
        seen: list[str] = []
        for key_node, value_pattern in zip(pattern.keys, pattern.patterns):
            self.emit(f"if ({fits}) {{", indent)
            key = self.expression(key_node, indent + 1)
            seen.append(key)
            present = self.temporary_flag()
            self.emit(
                f"{present} = PySequence_Contains({subject}, {key});", indent + 1
            )
            self.emit(f"if ({present} < 0) {{ {self.failure()} }}", indent + 1)
            self.emit(f"{fits} = {present};", indent + 1)
            self.emit(f"if ({fits}) {{", indent + 1)
            picked = self.temporary()
            self.emit(f"{picked} = PyObject_GetItem({subject}, {key});", indent + 2)
            self.checked(picked, indent + 2)
            inner = self.pattern_test(value_pattern, picked, indent + 2)
            self.emit(f"Py_DecRef({picked});", indent + 2)
            self.emit(f"{fits} = {inner};", indent + 2)
            self.emit("}", indent + 1)
            self.emit("}", indent)
        if pattern.rest is not None:
            # What the pattern did not name, as a new dict. Built by copying
            # and deleting rather than by comprehension, because the keys to
            # remove are known here and the copy is one call.
            self.emit(f"if ({fits}) {{", indent)
            copier = self.builtin("dict", indent + 1)
            rest = self.temporary()
            self.emit(
                f"{rest} = PyObject_CallOneArg({copier}, {subject});", indent + 1
            )
            self.emit(f"Py_DecRef({copier});", indent + 1)
            self.checked(rest, indent + 1)
            for key in seen:
                self.emit(f"PyObject_DelItem({rest}, {key});", indent + 1)
            self.bind_target(
                ast.Name(id=pattern.rest, ctx=ast.Store()), rest, indent + 1
            )
            self.emit("}", indent)
        for key in seen:
            self.emit(f"Py_DecRef({key});", indent)
        return fits

    def class_pattern(self, pattern, subject: str, fits: str, indent: int) -> str:
        """`case Point(x=1)` - the right class, then its attributes.

        A positional sub-pattern is matched against the attribute
        `__match_args__` names at that position, which is the protocol Python
        uses and the reason a class can be matched positionally at all.
        """

        self.instance_test(subject, self.expression(pattern.cls, indent), fits, indent)
        if pattern.patterns:
            # `__match_args__` is a tuple of attribute names; position i of the
            # pattern matches the attribute it names at position i.
            self.emit(f"if ({fits}) {{", indent)
            names = self.temporary()
            self.emit(
                f'{names} = PyObject_GetAttrString({subject}, "__match_args__");',
                indent + 1,
            )
            self.emit(f"if (!{names}) {{ PyErr_Clear(); {fits} = 0; }}", indent + 1)
            self.emit(f"if ({fits}) {{", indent + 1)
            self.emit(
                f"{fits} = (PyObject_Size({names}) >= {len(pattern.patterns)});",
                indent + 2,
            )
            self.emit("}", indent + 1)
            for position, sub in enumerate(pattern.patterns):
                self.emit(f"if ({fits}) {{", indent + 1)
                index = self.temporary()
                self.emit(
                    f"{index} = PyLong_FromLongLong({position}LL);", indent + 2
                )
                self.checked(index, indent + 2)
                attribute = self.temporary()
                self.emit(
                    f"{attribute} = PyObject_GetItem({names}, {index});", indent + 2
                )
                self.emit(f"Py_DecRef({index});", indent + 2)
                self.checked(attribute, indent + 2)
                got = self.temporary()
                getter = self.builtin("getattr", indent + 2)
                array = self.argument_array(2)
                self.emit(f"{array}[0] = {subject};", indent + 2)
                self.emit(f"{array}[1] = {attribute};", indent + 2)
                self.emit(
                    f"{got} = PyObject_Vectorcall({getter}, {array}, 2, 0);",
                    indent + 2,
                )
                self.emit(f"Py_DecRef({getter});", indent + 2)
                self.emit(f"Py_DecRef({attribute});", indent + 2)
                self.emit(
                    f"if (!{got}) {{ PyErr_Clear(); {fits} = 0; }}", indent + 2
                )
                self.emit(f"if ({fits}) {{", indent + 2)
                inner = self.pattern_test(sub, got, indent + 3)
                self.emit(f"{fits} = {inner};", indent + 3)
                self.emit(f"Py_DecRef({got});", indent + 3)
                self.emit("}", indent + 2)
                self.emit("}", indent + 1)
            self.emit(f"Py_DecRef({names});", indent + 1)
            self.emit("}", indent)
        for keyword, sub in zip(pattern.kwd_attrs, pattern.kwd_patterns):
            self.emit(f"if ({fits}) {{", indent)
            got = self.temporary()
            self.emit(
                f'{got} = PyObject_GetAttrString({subject}, {_c_string(keyword)});',
                indent + 1,
            )
            self.emit(f"if (!{got}) {{ PyErr_Clear(); {fits} = 0; }}", indent + 1)
            self.emit(f"if ({fits}) {{", indent + 1)
            inner = self.pattern_test(sub, got, indent + 2)
            self.emit(f"{fits} = {inner};", indent + 2)
            self.emit(f"Py_DecRef({got});", indent + 2)
            self.emit("}", indent + 1)
            self.emit("}", indent)
        return fits

    def unpack_with_star(self, target, value: str, indent: int) -> None:
        """`a, *rest, b = xs` - the fixed names, and a list for the rest.

        The star's share is whatever is left once the names on either side
        have taken theirs, so the length has to be known before anything is
        bound - which is why this goes through a list rather than pulling from
        an iterator. It is also the only unpacking whose failure is one-sided:
        there can never be *too many* values, only too few.
        """

        star = next(
            index
            for index, element in enumerate(target.elts)
            if isinstance(element, ast.Starred)
        )
        if any(
            isinstance(element, ast.Starred)
            for element in target.elts[star + 1 :]
        ):
            raise self.fail(target, "two starred names in one unpacking")
        for element in target.elts:
            plain = element.value if isinstance(element, ast.Starred) else element
            if not isinstance(plain, ast.Name):
                raise self.fail(target, "unpacking binds plain names here")
        before, after = star, len(target.elts) - star - 1
        maker = self.builtin("list", indent)
        items = self.temporary()
        self.emit(f"{items} = PyObject_CallOneArg({maker}, {value});", indent)
        self.emit(f"Py_DecRef({maker});", indent)
        self.checked(items, indent)
        length = self.temporary_flag()
        self.emit(f"{length} = (int)PyObject_Size({items});", indent)
        self.emit(f"if ({length} < 0) {{ {self.failure()} }}", indent)
        self.emit(f"if ({length} < {before + after}) {{", indent)
        counted = self.temporary()
        self.emit(
            f"{counted} = PyLong_FromLongLong((long long){length});", indent + 1
        )
        self.checked(counted, indent + 1)
        self.raise_value_error(
            f"not enough values to unpack (expected at least "
            f"{before + after}, got ",
            counted,
            indent + 1,
        )
        self.emit("}", indent)
        for position in range(before):
            self.bind_index(target.elts[position], items, str(position), indent)
        # The star's slice runs from where the leading names stopped to where
        # the trailing ones begin, counted from the end so it does not matter
        # how long the middle is.
        rest = self.temporary()
        start = self.temporary()
        stop = self.temporary()
        self.emit(f"{start} = PyLong_FromLongLong({before}LL);", indent)
        self.checked(start, indent)
        self.emit(
            f"{stop} = PyLong_FromLongLong((long long)({length} - {after}));", indent
        )
        self.checked(stop, indent)
        none = self.builtin("None", indent)
        cut = self.temporary()
        self.emit(f"{cut} = PySlice_New({start}, {stop}, {none});", indent)
        self.emit(f"Py_DecRef({start});", indent)
        self.emit(f"Py_DecRef({stop});", indent)
        self.emit(f"Py_DecRef({none});", indent)
        self.checked(cut, indent)
        self.emit(f"{rest} = PyObject_GetItem({items}, {cut});", indent)
        self.emit(f"Py_DecRef({cut});", indent)
        self.checked(rest, indent)
        self.bind_target(target.elts[star].value, rest, indent)
        for offset in range(after):
            self.bind_index(
                target.elts[star + 1 + offset],
                items,
                f"{length} - {after - offset}",
                indent,
            )
        self.emit(f"Py_DecRef({items});", indent)

    def bind_index(self, element, items: str, position: str, indent: int) -> None:
        """Bind one name to `items[position]`, where position is C, not Python."""

        index = self.temporary()
        self.emit(f"{index} = PyLong_FromLongLong((long long)({position}));", indent)
        self.checked(index, indent)
        picked = self.temporary()
        self.emit(f"{picked} = PyObject_GetItem({items}, {index});", indent)
        self.emit(f"Py_DecRef({index});", indent)
        self.checked(picked, indent)
        self.bind_target(element, picked, indent)

    def with_cause(
        self, node: ast.Raise, value: str, owned: bool, indent: int
    ) -> str:
        """`raise E from C` - the `from` half, which is one attribute.

        `__cause__` has a setter that also sets `__suppress_context__`, so
        assigning it is the whole of what the statement means; there is nothing
        else for a `from` to do.

        The awkward part is that `raise ValueError from C` names a class, and a
        class has nowhere to keep a cause - the instance does. Python
        instantiates before attaching, so this asks whether what it has is a
        class and calls it when it is.
        """

        cause = self.expression(node.cause, indent)
        is_class = self.temporary_flag()
        kind = self.builtin("type", indent)
        checker = self.builtin("isinstance", indent)
        array = self.argument_array(2)
        self.emit(f"{array}[0] = {value};", indent)
        self.emit(f"{array}[1] = {kind};", indent)
        answer = self.temporary()
        self.emit(
            f"{answer} = PyObject_Vectorcall({checker}, {array}, 2, 0);", indent
        )
        self.emit(f"Py_DecRef({checker});", indent)
        self.emit(f"Py_DecRef({kind});", indent)
        self.checked(answer, indent)
        self.emit(f"{is_class} = PyObject_IsTrue({answer});", indent)
        self.emit(f"Py_DecRef({answer});", indent)
        instance = self.temporary()
        self.emit(f"if ({is_class}) {{", indent)
        self.emit(f"{instance} = PyObject_CallNoArgs({value});", indent + 1)
        if owned:
            self.emit(f"Py_DecRef({value});", indent + 1)
        self.emit(f"if (!{instance}) {{ {self.failure()} }}", indent + 1)
        self.emit("} else {", indent)
        if not owned:
            self.emit(f"Py_IncRef({value});", indent + 1)
        self.emit(f"{instance} = {value};", indent + 1)
        self.emit("}", indent)
        outcome = self.temporary_flag()
        self.emit(
            f'{outcome} = PyObject_SetAttrString({instance}, "__cause__", '
            f"{cause});",
            indent,
        )
        self.emit(f"Py_DecRef({cause});", indent)
        self.emit(f"if ({outcome} < 0) {{ {self.failure()} }}", indent)
        return instance

    def with_block(self, node: ast.With, indent: int) -> None:
        """`with a as b:` - __enter__, the body, then __exit__ however it ends.

        __exit__ used to be written after the body, which meant it ran only
        when the body fell off the end. A `break`, a `return` or an exception
        left without it - so the thing the `with` exists to close was not
        closed, silently. It is a `finally` in every respect, so it is written
        as one.

        The three arguments are the exception when there is one, and `None`
        three times when there is not. A truthy answer from __exit__ suppresses
        the exception, which is how `contextlib.suppress` and every `__exit__`
        that swallows works.
        """

        if len(node.items) > 1:
            inner = ast.copy_location(
                ast.With(items=node.items[1:], body=node.body, type_comment=None),
                node,
            )
            ast.fix_missing_locations(inner)
            node = ast.copy_location(
                ast.With(items=node.items[:1], body=[inner], type_comment=None),
                node,
            )
        item = node.items[0]
        manager = self.expression(item.context_expr, indent)
        entered = self.temporary()
        enter = self.temporary()
        self.emit(
            f'{enter} = PyObject_GetAttr({manager}, {self.interned("__enter__")});',
            indent,
        )
        self.checked(enter, indent)
        self.emit(f"{entered} = PyObject_CallNoArgs({enter});", indent)
        self.emit(f"Py_DecRef({enter});", indent)
        self.checked(entered, indent)
        if item.optional_vars is not None:
            if not isinstance(item.optional_vars, ast.Name):
                raise self.fail(node, "`with ... as` binds one name here")
            target = self.declare(item.optional_vars.id)
            self.emit(f"if ({target}) Py_DecRef({target});", indent)
            self.emit(f"Py_IncRef({entered});", indent)
            self.emit(f"{target} = {entered};", indent)
            self.publish(item.optional_vars.id, target, indent)
        introduced = (
            {item.optional_vars.id} if item.optional_vars is not None else set()
        )

        def emit_body() -> None:
            with self.settled_within(introduced):
                for statement in node.body:
                    self.statement(statement, indent)

        def emit_clause(held: str, protection) -> None:
            self.close_manager(manager, held, protection, indent)
            self.emit(f"Py_DecRef({entered});", indent)
            self.emit(f"Py_DecRef({manager});", indent)

        self.protect(emit_body, emit_clause, indent)

    def close_manager(self, manager: str, held: str, protection, indent: int) -> None:
        """Call __exit__ with what happened, and let it suppress."""

        exit_call = self.temporary()
        self.emit(
            f'{exit_call} = PyObject_GetAttrString({manager}, "__exit__");', indent
        )
        self.checked(exit_call, indent)
        arguments = self.temporary()
        self.emit(f"{arguments} = PyTuple_New(3LL);", indent)
        self.checked(arguments, indent)
        # With an exception in hand the three are its class, itself and its
        # traceback, which is what an __exit__ is written to read.
        self.emit(f"if ({held}) {{", indent)
        kind = self.builtin("type", indent + 1)
        classified = self.temporary()
        self.emit(f"{classified} = PyObject_CallOneArg({kind}, {held});", indent + 1)
        self.emit(f"Py_DecRef({kind});", indent + 1)
        self.checked(classified, indent + 1)
        trace = self.temporary()
        self.emit(
            f'{trace} = PyObject_GetAttrString({held}, "__traceback__");', indent + 1
        )
        self.checked(trace, indent + 1)
        self.emit(f"Py_IncRef({held});", indent + 1)
        self.emit(f"PyTuple_SetItem({arguments}, 0LL, {classified});", indent + 1)
        self.emit(f"PyTuple_SetItem({arguments}, 1LL, {held});", indent + 1)
        self.emit(f"PyTuple_SetItem({arguments}, 2LL, {trace});", indent + 1)
        self.emit("} else {", indent)
        nothing = self.builtin("None", indent + 1)
        for position in range(3):
            self.emit(f"Py_IncRef({nothing});", indent + 1)
            self.emit(
                f"PyTuple_SetItem({arguments}, {position}LL, {nothing});", indent + 1
            )
        self.emit(f"Py_DecRef({nothing});", indent + 1)
        self.emit("}", indent)
        # While `__exit__` runs, the exception the body raised is the one
        # being handled - so anything `__exit__` raises takes its
        # `__context__` from it, and a traceback keeps the chain. Without
        # this, an `__exit__` that raised produced an exception with no
        # context at all and the original vanished from the report.
        previous = self.temporary()
        self.emit(f"{previous} = PyErr_GetHandledException();", indent)
        self.emit(f"if ({held}) {{", indent)
        self.emit(f"PyErr_SetHandledException({held});", indent + 1)
        self.emit("}", indent)
        outcome = self.temporary()
        self.emit(
            f"{outcome} = PyObject_Call({exit_call}, {arguments}, (PyObject *)0);",
            indent,
        )
        self.emit(f"PyErr_SetHandledException({previous});", indent)
        self.emit(f"if ({previous}) Py_DecRef({previous});", indent)
        self.emit(f"Py_DecRef({exit_call});", indent)
        self.emit(f"Py_DecRef({arguments});", indent)
        self.checked(outcome, indent)
        # A truthy answer means "I have dealt with it", so the region stops
        # leaving because of an exception and falls out normally instead.
        swallowed = self.temporary_flag()
        self.emit(f"{swallowed} = PyObject_IsTrue({outcome});", indent)
        self.emit(f"Py_DecRef({outcome});", indent)
        self.emit(f"if ({swallowed} < 0) {{ {self.failure()} }}", indent)
        self.emit(
            f"if ({swallowed} && {protection.why} == {_PROPAGATING}) {{", indent
        )
        self.emit(f"Py_DecRef({held});", indent + 1)
        self.emit(f"{held} = 0;", indent + 1)
        self.emit(f"{protection.why} = 0;", indent + 1)
        self.emit("}", indent)

    def guarded(self, node: ast.Try, indent: int) -> None:
        """`try: ... except E: ...` - the body, then a handler it jumps to.

        A C-API call that fails leaves an exception set and answers NULL, and
        inside a `try` that NULL becomes a jump rather than the end of the
        process. `PyErr_ExceptionMatches` asks whether the exception is the
        class this clause catches, exactly as the interpreter asks it.
        """

        if node.finalbody:
            self.protected(node, indent)
            return

        assert self.current is not None
        self.current.labels += 1
        number = self.current.labels
        handler = f"_handler{number}"
        done = f"_after{number}"
        self.handlers.append(handler)
        try:
            for statement in node.body:
                self.statement(statement, indent)
        finally:
            self.handlers.pop()
        # The `else` clause runs when the body raised nothing, and is *not*
        # protected by these handlers - an exception in it belongs outside, as
        # it does in Python. The handler label is already popped by here.
        for statement in node.orelse:
            self.statement(statement, indent)
        self.emit(f"goto {done};", indent)
        self.emit(f"{handler}:", 0)
        # The classes each clause catches are evaluated *here*, where the body
        # has already raised - not before it, where every `try` paid for them
        # whether it needed them or not, and where they were then left held:
        # `except (ValueError, TypeError)` builds a fresh tuple each time it
        # is evaluated, so a `try` in a loop leaked one per turn.
        #
        # Evaluating them with an exception set is what CPython refuses -
        # `except (A, B)` failed with "returned a result with an exception
        # set". So the exception is lifted out first and put back once they
        # are built, which is what `PyErr_ExceptionMatches` below needs
        # anyway.
        pending = self.temporary()
        self.emit(f"{pending} = PyErr_GetRaisedException();", indent)
        caught: list[str | None] = []
        for clause in node.handlers:
            caught.append(
                None if clause.type is None else self.expression(clause.type, indent)
            )
        self.emit(f"PyErr_SetRaisedException({pending});", indent)
        for clause, wanted in zip(node.handlers, caught):
            if clause.type is None:
                # A bare except catches whatever is set. Python requires it to
                # be the last clause, so there is never one after it holding a
                # class that would go unreleased.
                held = self.bind_exception(clause, indent)
                self.run_handler(clause, held, indent)
                self.emit(f"Py_DecRef({held});", indent)
                self.emit(f"goto {done};", indent)
                continue
            # PyErr_ExceptionMatches takes a tuple as readily as a class.
            decision = self.temporary_flag()
            self.emit(f"{decision} = PyErr_ExceptionMatches({wanted});", indent)
            self.emit(f"Py_DecRef({wanted});", indent)
            self.emit(f"if ({decision}) {{", indent)
            # The clauses after this one are never tested, so nothing else
            # will release them. Their classes were built before the body ran
            # like every other clause's.
            position = node.handlers.index(clause)
            for later in caught[position + 1 :]:
                if later is not None:
                    self.emit(f"Py_DecRef({later});", indent + 1)
            held = self.bind_exception(clause, indent + 1)
            self.run_handler(clause, held, indent + 1)
            self.emit(f"Py_DecRef({held});", indent + 1)
            self.emit(f"    goto {done};", indent)
            self.emit("}", indent)
        # Nothing matched, so the exception carries on outward.
        if self.handlers:
            self.emit(f"goto {self.handlers[-1]};", indent)
        else:
            self.emit(self.failure(), indent)
        self.emit(f"{done}:", 0)
        self.emit(";", indent)

    def protected(self, node: ast.Try, indent: int) -> None:
        """`try: ... finally: ...` - every way out goes through the clause.

        There are four ways out of the protected region and the clause runs
        for all of them: falling off the end, an exception nothing caught,
        `return`, and `break`/`continue`. Each records *why* it is leaving in
        an int and jumps to the clause, which runs once and then does what the
        reason says. Writing the clause out at each exit instead would put a
        copy of it in four places.

        The exception is **taken** before the clause runs. CPython refuses to
        build anything Python-side while one is set, so a clause that so much
        as calls a method would fail with "returned a result with an exception
        set". `PyErr_SetRaisedException` puts the same object back afterwards,
        traceback intact.
        """

        def emit_body() -> None:
            if node.handlers or node.orelse:
                inner = ast.copy_location(
                    ast.Try(
                        body=node.body,
                        handlers=node.handlers,
                        orelse=node.orelse,
                        finalbody=[],
                    ),
                    node,
                )
                self.guarded(inner, indent)
            else:
                for statement in node.body:
                    self.statement(statement, indent)

        def emit_clause(_held: str, _protection) -> None:
            for statement in node.finalbody:
                self.statement(statement, indent)

        self.protect(emit_body, emit_clause, indent)

    def run_handler(self, clause, held: str, indent: int) -> None:
        """An `except` clause's body, with the exception on record while it runs.

        Taking the exception is what lets the clause run Python at all, but it
        also took it off the thread's record of what is being *handled* - and
        that record is what `sys.exc_info()` answers with and what CPython
        attaches as `__context__` to anything raised from inside a handler. So
        both came out empty: a traceback lost its "during handling of the
        above exception" chain, and `sys.exc_info()` said `None` where every
        piece of logging code expects a type.

        Put back on the way out, whichever way that is - the clause can
        return, break, or raise - which is what `protect` is for. Restoring
        what was there rather than clearing it is what lets handlers nest.
        """

        previous = self.temporary()
        # `PyErr_SetHandledException` takes a reference of its own rather than
        # stealing one, which is the opposite of its `Raised` counterpart -
        # incrementing before the call leaked the exception once per handler,
        # and 800,000 of them came to 137 MB against the interpreter's 14.
        # `PyErr_GetHandledException` does hand over a reference, so what it
        # returns is released once it has been put back.
        self.emit(f"{previous} = PyErr_GetHandledException();", indent)
        self.emit(f"PyErr_SetHandledException({held});", indent)

        def emit_body() -> None:
            self.handling.append(held)
            with self.settled_within({clause.name} if clause.name else set()):
                for statement in clause.body:
                    self.statement(statement, indent)
            self.handling.pop()

        def emit_clause(_held: str, _protection) -> None:
            self.emit(f"PyErr_SetHandledException({previous});", indent)
            self.emit(f"if ({previous}) Py_DecRef({previous});", indent)
            if clause.name is not None:
                # Python unbinds the name when the handler ends, however it
                # ends - `except E as e` leaves no `e` behind, and reading one
                # afterwards is a NameError. It stayed bound here, which
                # nothing could see until `globals()` began answering with the
                # real dictionary and listed a name the program had let go.
                slot = self.declare(clause.name)
                self.emit(
                    f"if ({slot}) {{ Py_DecRef({slot}); {slot} = 0; }}", indent
                )
                if self.globals_in_dict and slot.startswith("g_"):
                    self.emit(
                        f"if (PyObject_DelItem(_py2bin_globals, "
                        f"{self.interned(clause.name)}) < 0) PyErr_Clear();",
                        indent,
                    )

        self.protect(emit_body, emit_clause, indent)

    def protect(self, emit_body, emit_clause, indent: int) -> None:
        """The machinery a `finally` and a `with` both need.

        ``emit_body`` writes the region being protected; ``emit_clause`` writes
        what must happen however it is left, and is handed the slot holding the
        exception - empty unless the region is leaving because of one.
        """

        assert self.current is not None
        self.current.labels += 1
        number = self.current.labels
        clause = f"_finally{number}"
        landing = f"_finally_raise{number}"
        protection = _Protected(
            clause, self.temporary_flag(), self.temporary(), self.loop_depth
        )
        held = self.temporary()
        self.emit(f"{protection.why} = 0;", indent)
        self.emit(f"{protection.answer} = 0;", indent)
        self.emit(f"{held} = 0;", indent)
        # Anything failing inside the region lands here, which is also where
        # the `except` clauses send what they did not match.
        self.handlers.append(landing)
        self.finallys.append(protection)
        try:
            emit_body()
        finally:
            self.finallys.pop()
            self.handlers.pop()
        self.emit(f"goto {clause};", indent)
        self.emit(f"{landing}:", 0)
        self.emit(f"{held} = PyErr_GetRaisedException();", indent)
        self.emit(f"{protection.why} = {_PROPAGATING};", indent)
        protection.reasons.add(_PROPAGATING)
        # Falls straight into the clause, which is the point: the exception
        # path and the ordinary path run the same code.
        self.emit(f"{clause}:", 0)
        emit_clause(held, protection)
        if _PROPAGATING in protection.reasons:
            self.emit(f"if ({protection.why} == {_PROPAGATING}) {{", indent)
            # It steals the reference, so nothing is released after it.
            self.emit(f"PyErr_SetRaisedException({held});", indent + 1)
            self.emit(self.failure(), indent + 1)
            self.emit("}", indent)
        if _RETURNING in protection.reasons:
            self.emit(f"if ({protection.why} == {_RETURNING}) {{", indent)
            if self.finallys:
                # Another clause encloses this one, and it has to run too.
                outer = self.finallys[-1]
                outer.reasons.add(_RETURNING)
                self.emit(f"{outer.answer} = {protection.answer};", indent + 1)
                self.emit(f"{outer.why} = {_RETURNING};", indent + 1)
                self.emit(f"goto {outer.label};", indent + 1)
            else:
                self.release_locals(indent + 1)
                self.leave(protection.answer, indent + 1)
            self.emit("}", indent)
        if _BREAKING in protection.reasons:
            self.emit(f"if ({protection.why} == {_BREAKING}) {{", indent)
            self.mark_broken(indent + 1)
            self.emit("break;", indent + 1)
            self.emit("}", indent)
        if _CONTINUING in protection.reasons:
            self.emit(f"if ({protection.why} == {_CONTINUING}) continue;", indent)

    def bind_exception(self, clause: ast.ExceptHandler, indent: int) -> str:
        """Take the exception, binding it too when the clause names it.

        Taking it is what clears it, so `except E as name` and a bare `except`
        differ only in whether the object also gets a name.
        """

        caught = self.temporary()
        # Taken rather than cleared even when the clause does not name it: a
        # bare `raise` in the body sets this same object again, and clearing
        # would have thrown away the only thing it could re-raise.
        self.emit(f"{caught} = PyErr_GetRaisedException();", indent)
        if clause.name is not None:
            target = self.declare(clause.name)
            self.emit(f"if ({target}) Py_DecRef({target});", indent)
            self.emit(f"Py_IncRef({caught});", indent)
            self.emit(f"{target} = {caught};", indent)
            self.publish(clause.name, target, indent)
        return caught

    def augmented(self, node: ast.AugAssign, indent: int) -> None:
        """`x += 1` - the operation, then the assignment, as Python does it."""

        if isinstance(node.target, (ast.Subscript, ast.Attribute)):
            self.augmented_place(node, indent)
            return
        if not isinstance(node.target, ast.Name):
            raise self.fail(node, "only augmented assignment to a place is translated")
        rewritten = ast.copy_location(
            ast.BinOp(
                left=ast.copy_location(
                    ast.Name(id=node.target.id, ctx=ast.Load()), node
                ),
                op=node.op,
                right=node.value,
            ),
            node,
        )
        if self.narrow_assign(node.target.id, rewritten, indent):
            return
        value = self.in_place(node, indent)
        if value is None:
            value = self.expression(rewritten, indent)
        if self.is_unboxed(node.target.id):
            self.store_object(node.target.id, value, indent)
            return
        if self.is_double(node.target.id):
            self.store_double_object(node.target.id, value, indent)
            return
        target = self.declare(node.target.id)
        self.emit(f"if ({target}) Py_DecRef({target});", indent)
        self.emit(f"{target} = {value};", indent)
        self.publish(node.target.id, target, indent)

    def in_place(self, node: ast.AugAssign, indent: int) -> str | None:
        """`x += y` through the operator that lets `x` answer for itself.

        `x += y` is not `x = x + y`. A list extends itself and every other
        name for it sees that; rebuilding it leaves them holding the old one,
        which is what happened here - `xs += [2]` gave `xs` the new list and
        left an alias on the first, where Python has both. A class with
        `__iadd__` never had it called at all.

        Answers None for an operator with no in-place form here, and the
        caller falls back to building the value.
        """

        operator = _IN_PLACE.get(type(node.op))
        if operator is None:
            return None
        assert isinstance(node.target, ast.Name)
        name = node.target.id
        # Always the in-place operator, even where the name looks numeric.
        # Branching on the narrowing flag was tried and measured *worse*: the
        # flag is only set while the value is a machine number, and the first
        # `t += x` with an object on the right puts an object in the slot and
        # clears it - so the branch never took the fast side again and every
        # turn paid for the test. What that row costs is written down under
        # "how fast each one is"; it is what `x += y` meaning what Python
        # means is worth.
        held = self.expression(
            ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node), indent
        )
        right = self.expression(node.value, indent)
        target = self.temporary()
        if isinstance(node.op, ast.Pow):
            # `PyNumber_InPlacePower` takes the modulus `pow(a, b, m)` does,
            # and `a **= b` passes None for it.
            blank = self.builtin("None", indent)
            self.emit(
                f"{target} = {operator}({held}, {right}, {blank});", indent
            )
            self.emit(f"Py_DecRef({blank});", indent)
        else:
            self.emit(f"{target} = {operator}({held}, {right});", indent)
        self.emit(f"Py_DecRef({held});", indent)
        self.emit(f"Py_DecRef({right});", indent)
        return self.checked(target, indent)

    def augmented_place(self, node: ast.AugAssign, indent: int) -> None:
        """`xs[k] += v` and `a.b += v` - read, combine, write back.

        The container and the key are computed once and used for both halves.
        Rewriting into `xs[f()] = xs[f()] + v` would call `f` twice, which
        Python does not.
        """

        # The in-place form for the same reason a plain name gets one: what
        # is read out of the container may be a list, and `xs['k'] += [2]`
        # has to extend it rather than put a different list back. The write
        # below still happens - Python does it too, and it is what makes the
        # difference for something that answers with a *new* object.
        # `**` is left to the binary form here: its in-place twin takes a
        # third argument and this path passes two.
        function = None
        if not isinstance(node.op, ast.Pow):
            function = _IN_PLACE.get(type(node.op))
        function = function or _BINARY.get(type(node.op))
        if function is None:
            raise self.fail(
                node, f"{type(node.op).__name__} is not translated here yet"
            )
        owner = self.expression(node.target.value, indent)
        if isinstance(node.target, ast.Subscript):
            key = (
                self.slice_object(node.target.slice, indent)
                if isinstance(node.target.slice, ast.Slice)
                else self.expression(node.target.slice, indent)
            )
            current = self.temporary()
            self.emit(f"{current} = PyObject_GetItem({owner}, {key});", indent)
            self.checked(current, indent)
        else:
            current = self.temporary()
            self.emit(
                f"{current} = PyObject_GetAttr({owner}, "
                f"{self.interned(node.target.attr)});",
                indent,
            )
            self.checked(current, indent)
        addend = self.expression(node.value, indent)
        combined = self.temporary()
        self.emit(f"{combined} = {function}({current}, {addend});", indent)
        self.emit(f"Py_DecRef({current});", indent)
        self.emit(f"Py_DecRef({addend});", indent)
        self.checked(combined, indent)
        outcome = self.temporary_flag()
        if isinstance(node.target, ast.Subscript):
            self.emit(
                f"{outcome} = PyObject_SetItem({owner}, {key}, {combined});", indent
            )
            self.emit(f"Py_DecRef({key});", indent)
        else:
            self.emit(
                f"{outcome} = PyObject_SetAttr({owner}, "
                f"{self.interned(node.target.attr)}, {combined});",
                indent,
            )
        self.emit(f"Py_DecRef({owner});", indent)
        self.emit(f"Py_DecRef({combined});", indent)
        self.emit(f"if ({outcome} < 0) {{ {self.failure()} }}", indent)

    def unpack_value(self, target, value: str, indent: int) -> None:
        """Take ``value`` apart into the names of ``target``."""

        if any(isinstance(element, ast.Starred) for element in target.elts):
            self.unpack_with_star(target, value, indent)
            return
        # Any target shape: bind_target below takes a name, a nested tuple, an
        # attribute or a subscript, so `(a, b), c = pair` and `d[k], e = row`
        # work, and so does unpacking into a `nonlocal` name once it has
        # become a cell.
        # Through a tuple first, for two reasons: unpacking works on any
        # iterable and indexing does not, and the length has to be known to
        # say whether it matches. Indexing the value directly took `a, b` from
        # a three-item tuple without a word, where Python raises ValueError.
        # A sequence can be taken apart where it stands. `tuple()` is what
        # makes this work for *any* iterable - a generator has no length and
        # no index - but for a tuple or a list, which is what almost every
        # unpack actually holds, it allocates a copy per unpack and frees it
        # again. The copy is made only when the value cannot answer for
        # itself.
        #
        # Both questions have to be asked: `PySequence_Check` is true for a
        # class defining only `__getitem__`, and such a class has no length,
        # so the size is tried and a failure sends it back to the general
        # path rather than out of the program.
        items = self.temporary()
        size = self.machine_slot()
        direct = self.temporary_flag()
        held = self.temporary_flag()
        self.emit(f"{items} = {value};", indent)
        self.emit(f"{held} = 0;", indent)
        self.emit(f"{direct} = PySequence_Check({value});", indent)
        self.emit(f"if ({direct}) {{", indent)
        self.emit(f"{size} = PyObject_Size({value});", indent + 1)
        self.emit(
            f"if ({size} < 0) {{ PyErr_Clear(); {direct} = 0; }}", indent + 1
        )
        self.emit("}", indent)
        self.emit(f"if (!{direct}) {{", indent)
        maker = self.builtin("tuple", indent + 1)
        self.emit(f"{items} = PyObject_CallOneArg({maker}, {value});", indent + 1)
        self.emit(f"Py_DecRef({maker});", indent + 1)
        self.emit(f"if (!{items}) {{ {self.failure()} }}", indent + 1)
        self.emit(f"{held} = 1;", indent + 1)
        self.emit(f"{size} = PyObject_Size({items});", indent + 1)
        self.emit(f"if ({size} < 0) {{ {self.failure()} }}", indent + 1)
        self.emit("}", indent)
        wanted = len(target.elts)
        # How long it is, as a machine integer. This used to box the length,
        # box the expected count twice, run two `PyObject_RichCompare`s and
        # ask `PyObject_IsTrue` of each answer - eleven C-API calls and five
        # allocations to decide whether a two-item tuple has two items, which
        # is a single comparison. `a, b = pair` measured 0.18x the
        # interpreter; the check was almost all of it.
        for test, message in (
            (f"{size} > {wanted}", f"too many values to unpack (expected {wanted}, got "),
            (f"{size} < {wanted}", f"not enough values to unpack (expected {wanted}, got "),
        ):
            self.emit(f"if ({test}) {{", indent)
            # The count is only boxed on the way to raising, where the cost is
            # beside the point and the message wants the number.
            counted = self.temporary()
            self.emit(f"{counted} = PyLong_FromLongLong({size});", indent + 1)
            self.checked(counted, indent + 1)
            self.raise_value_error(message, counted, indent + 1)
            self.emit("}", indent)
        for position, element in enumerate(target.elts):
            item = self.temporary()
            # `items` is what `tuple()` just answered, so it is a tuple and
            # the sequence protocol is the right one - no boxed index, and no
            # mapping lookup to rule out first.
            self.emit(
                f"{item} = PySequence_GetItem({items}, {position}LL);", indent
            )
            self.checked(item, indent)
            self.bind_target(element, item, indent)
        self.emit(f"if ({held}) Py_DecRef({items});", indent)

    def expression_statement(self, node: ast.Expr, indent: int) -> None:
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print"
            and self.builtin_untouched("print")
        ):
            # Only when the program has not bound the name itself. Writing
            # straight to `sys.stdout` is right for the builtin and wrong for
            # a `def print` of the program's own, which was being skipped in
            # silence - the output went out, just not through the function the
            # program wrote.
            self.write_out(node.value, indent)
            return
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "append"
            and len(call.args) == 1
            and not call.keywords
            and not isinstance(call.args[0], ast.Starred)
            and self.is_exact_list(call.func.value)
        ):
            # `xs.append(v)` as a statement - which is how append is written -
            # answers None, and the expression path dutifully took a reference
            # to it and gave it straight back. Nothing is made here at all.
            owner, owned = self.operand(call.func.value, indent)
            value, value_owned = self.operand(call.args[0], indent)
            outcome = self.temporary_flag()
            self.emit(f"{outcome} = PyList_Append({owner}, {value});", indent)
            self.release(owner, owned, indent)
            self.release(value, value_owned, indent)
            self.emit(f"if ({outcome} < 0) {{ {self.failure()} }}", indent)
            return
        value = self.expression(node.value, indent)
        self.emit(f"Py_DecRef({value});", indent)

    def still_the_builtin(self, name: str, indent: int) -> str:
        """A C flag: is `builtins.<name>` still the object it was at start-up?

        The shortcuts below reach past the name to what it usually means -
        `print` writes to `sys.stdout`, `len` calls `PyObject_Size`. That is
        only right while the name still means that. A program is free to put
        something else on `builtins`, and test harnesses and logging shims do,
        so the shortcut asks first: one dictionary probe and a pointer compare
        against the object cached at start-up, against a whole call saved when
        the answer is yes - and the ordinary path taken when it is no.
        """

        live = self.builtin_raw(name, indent)
        answer = self.temporary_flag()
        slot = self.cached_builtins.get(name)
        if slot is None:
            slot = f"_py2bin_b{len(self.cached_builtins)}"
            self.cached_builtins[name] = slot
        self.emit(f"{answer} = ({live} == {slot});", indent)
        self.emit(f"if ({live}) Py_DecRef({live});", indent)
        self.emit(f"else PyErr_Clear();", indent)
        return answer

    def write_out(self, node: ast.Call, indent: int) -> None:
        """print(...) - straight to sys.stdout through the file API."""

        if node.keywords or any(
            isinstance(item, ast.Starred) for item in node.args
        ):
            # `end=`, `sep=`, `file=`, `flush=`, `*parts` - the fast path here
            # writes straight to sys.stdout and knows none of them, so this
            # hands the whole call to the interpreter's own print, which knows
            # all of them and is the definition of what they mean.
            printer = self.builtin("print", indent)
            answer = self.invoke(printer, node.args, indent, node.keywords)
            self.emit(f"Py_DecRef({printer});", indent)
            self.emit(f"Py_DecRef({answer});", indent)
            return
        # Every argument first, then the writing. print is a call, and a call
        # evaluates all of its arguments before any of it runs: interleaving
        # them let `print("x:", loud())` write "x: " before loud() spoke, and
        # let `print(7, 1 // 0)` write "7 " before raising.
        values = [self.expression(argument, indent) for argument in node.args]
        # Writing to the stream is only what `print` means while `print` still
        # means the builtin. A program that puts its own callable on
        # `builtins` - which harnesses that capture output do - gets that
        # callable, with the arguments already evaluated above so each is
        # computed exactly once whichever arm runs.
        original = self.still_the_builtin("print", indent)
        self.emit(f"if ({original}) {{", indent)
        stream = self.temporary()
        self.emit(f'{stream} = PySys_GetObject("stdout");', indent + 1)
        for position, value in enumerate(values):
            if position:
                self.emit(f'PyFile_WriteString(" ", {stream});', indent + 1)
            # Py_PRINT_RAW is 1: str() of the object rather than its repr,
            # which is what print writes.
            self.emit(f"PyFile_WriteObject({value}, {stream}, 1);", indent + 1)
            self.emit(f"Py_DecRef({value});", indent + 1)
        self.emit(f'PyFile_WriteString("\\n", {stream});', indent + 1)
        self.emit("} else {", indent)
        replaced = self.program_name("print", indent + 1)
        array = self.argument_array(max(len(values), 1))
        for position, value in enumerate(values):
            self.emit(f"{array}[{position}] = {value};", indent + 1)
        answer = self.temporary()
        self.emit(
            f"{answer} = PyObject_Vectorcall({replaced}, {array}, "
            f"{len(values)}LL, 0);",
            indent + 1,
        )
        self.emit(f"Py_DecRef({replaced});", indent + 1)
        for value in values:
            self.emit(f"Py_DecRef({value});", indent + 1)
        self.emit(f"if (!{answer}) {{ {self.failure()} }}", indent + 1)
        self.emit(f"Py_DecRef({answer});", indent + 1)
        self.emit("}", indent)

    def conditional(self, node: ast.If, indent: int) -> None:
        decision = self.truth(node.test, indent)
        self.emit(f"if ({decision}) {{", indent)
        for statement in node.body:
            self.statement(statement, indent + 1)
        if node.orelse:
            self.emit("} else {", indent)
            for statement in node.orelse:
                self.statement(statement, indent + 1)
        self.emit("}", indent)

    def loop(self, node: ast.While, indent: int) -> None:
        """`while c: ... else: ...` - the else runs unless a `break` left.

        Not "unless the loop ended early": the test failing is the ordinary
        way out and the else runs then. Only a `break` skips it, so only a
        `break` sets the flag - the exhaustion test does not.
        """

        broke = self.begin_loop(node, indent)
        self.emit("while (1) {", indent)
        decision = self.truth(node.test, indent + 1)
        self.emit(f"if (!{decision}) break;", indent + 1)
        self.loop_depth += 1
        try:
            for statement in node.body:
                self.statement(statement, indent + 1)
        finally:
            self.loop_depth -= 1
        self.emit("}", indent)
        self.finish_loop(node, broke, indent)

    def begin_loop(self, node, indent: int) -> str | None:
        """Set up the flag an `else` clause needs, if there is one."""

        if not node.orelse:
            self.loop_flags.append(None)
            return None
        broke = self.temporary_flag()
        self.emit(f"{broke} = 0;", indent)
        self.loop_flags.append(broke)
        return broke

    def finish_loop(self, node, broke: str | None, indent: int) -> None:
        """Run the `else` clause, unless a `break` left the loop."""

        self.loop_flags.pop()
        if broke is None:
            return
        self.emit(f"if (!{broke}) {{", indent)
        for statement in node.orelse:
            self.statement(statement, indent + 1)
        self.emit("}", indent)

    def for_loop(self, node: ast.For, indent: int) -> None:
        """`for x in seq:` - the iterator protocol, exactly as Python runs it.

        An `else` clause runs when the sequence ran out, which is the ordinary
        way for a loop to end; only a `break` skips it.

        `range(...)`, a list, a string, a file, a generator: whatever the
        object offers, because the interpreter is the one being asked. That is
        the difference from the native tier, which has to know the shape of
        every iterable it supports.
        """

        # Any target shape. `for d[k] in ...` is legal Python, and a loop
        # variable a closure captures becomes a subscript of its cell, which
        # is how a lambda made in a loop sees the value Python would show it.
        if isinstance(node.target, ast.Name) and self.is_unboxed(node.target.id):
            bounds = narrow_range(node.iter)
            if bounds is not None and "range" not in self.shadowed_builtins:
                self.counted_loop(node, bounds, indent)
                return
        sequence = self.expression(node.iter, indent)
        iterator = self.temporary()
        self.emit(f"{iterator} = PyObject_GetIter({sequence});", indent)
        self.checked(iterator, indent)
        self.emit(f"Py_DecRef({sequence});", indent)
        item = self.temporary()
        # Through `bind_target`, not `declare`: a name held in the unboxed
        # representation is three C variables, and `declare` answers with the
        # storage a *plain* local or global would use. Binding one and reading
        # the other is how `for arg in cmd:` came to write `g_arg` and read
        # `v_arg` - a NULL that reached PySequence_Contains and segfaulted.
        target = None
        # Before the loop opens, not inside it. A loop whose sequence is empty
        # runs its body no times, so a flag set up in the body is never set up
        # at all - and the slot is a reused one, so the `else` read whatever
        # the last loop that broke had left in it and did not run. `for i in
        # range(0, 0): ... else:` printed nothing after any earlier loop that
        # broke.
        broke = self.begin_loop(node, indent)
        self.emit("while (1) {", indent)
        self.emit(f"{item} = PyIter_Next({iterator});", indent + 1)
        # NULL means the sequence ended, or that producing the next item
        # failed. Asking whether an exception is set is what tells them apart.
        self.emit(
            f"if (!{item}) {{ if (PyErr_Occurred()) {{ {self.failure()} }} break; }}",
            indent + 1,
        )
        if isinstance(node.target, (ast.Tuple, ast.List)):
            # `for a, b in pairs` - the item is taken apart the way an
            # assignment to the same target would take it apart.
            self.unpack_value(node.target, item, indent + 1)
            self.emit(f"Py_DecRef({item});", indent + 1)
        else:
            # One target: a name, or a subscript or attribute to store
            # through. bind_target consumes the item either way.
            self.bind_target(node.target, item, indent + 1)
        self.loop_depth += 1
        try:
            targets = (
                {node.target.id}
                if isinstance(node.target, ast.Name)
                else {
                    element.id
                    for element in getattr(node.target, "elts", [])
                    if isinstance(element, ast.Name)
                }
            )
            with self.settled_within(targets):
                for statement in node.body:
                    self.statement(statement, indent + 1)
        finally:
            self.loop_depth -= 1
        self.emit("}", indent)
        self.emit(f"Py_DecRef({iterator});", indent)
        self.finish_loop(node, broke, indent)

    def counted_loop(
        self, node: ast.For, bounds: list[ast.expr], indent: int
    ) -> None:
        """`for i in range(...)` with the counting done in a register.

        The interpreter builds one integer object per iteration and throws it
        away again; a counted loop builds none. That is the single largest
        saving this tier has, because a `range` loop is how most Python
        arithmetic gets written.

        There is still exactly one copy of the body. The choice between
        counting and the iterator protocol is a branch *inside* the loop, not
        two loops - a duplicated body would grow the binary in proportion to
        how much of the program this helped, which is the wrong trade. The
        branch itself costs nothing worth measuring: it goes the same way
        every iteration, which is the case branch prediction is built for.

        The counting is declined at run time, not refused at compile time,
        whenever the arguments are not ordinary machine integers - `range` over
        a value too wide for a machine word is perfectly legal Python, and so
        is `range` over anything with an `__index__`. Both then go the long way
        round and behave exactly as they did.
        """

        assert self.current is not None
        name = node.target.id
        held, obj, state = self.narrow_slots(name)
        # Evaluated once, here, whatever path the loop then takes: these are
        # arbitrary expressions and Python evaluates each exactly once.
        spelled = [self.expression(argument, indent) for argument in bounds]
        start, stop, step = self.machine_slot(), self.machine_slot(), self.machine_slot()
        counting = self.temporary_flag()
        self.emit(f"{counting} = 1;", indent)
        order = (
            [(start, spelled[0]), (stop, spelled[1])]
            if len(spelled) > 1
            else [(stop, spelled[0])]
        )
        if len(spelled) > 2:
            order.append((step, spelled[2]))
        if len(spelled) < 2:
            self.emit(f"{start} = 0;", indent)
        if len(spelled) < 3:
            self.emit(f"{step} = 1;", indent)
        for slot, value in order:
            self.emit(f"{slot} = PyLong_AsLongLong({value});", indent)
            # A value that does not fit, or that is not an integer at all, is
            # not an error here - the generic path will call `range` with it
            # and let the interpreter say what it thinks.
            self.emit(
                f"if ({slot} == -1 && PyErr_Occurred()) "
                f"{{ PyErr_Clear(); {counting} = 0; }}",
                indent,
            )
        # A zero step is a ValueError, and the bounds are kept clear of the
        # edge of the word so that advancing the counter cannot itself
        # overflow. Both are left for `range` itself to handle.
        self.emit(f"if ({step} == 0) {counting} = 0;", indent)
        for slot in (start, stop, step):
            self.emit(
                f"if ({slot} > {_MACHINE_LIMIT} || {slot} < -{_MACHINE_LIMIT}) "
                f"{counting} = 0;",
                indent,
            )
        iterator = self.temporary()
        self.emit(f"{iterator} = 0;", indent)
        self.emit(f"if (!{counting}) {{", indent)
        built = self.call_range(spelled, indent + 1)
        self.emit(f"{iterator} = PyObject_GetIter({built});", indent + 1)
        self.emit(f"Py_DecRef({built});", indent + 1)
        self.emit(f"if (!{iterator}) {{ {self.failure()} }}", indent + 1)
        self.emit("}", indent)
        for value in spelled:
            self.emit(f"Py_DecRef({value});", indent)
        counter = self.machine_slot()
        self.emit(f"{counter} = {start};", indent)
        item = self.temporary()
        # Before the loop, for the reason given in `for_loop`: an empty range
        # never reaches the body, and the flag lives in a reused slot.
        broke = self.begin_loop(node, indent)
        self.emit("while (1) {", indent)
        self.emit(f"if ({counting}) {{", indent + 1)
        self.emit(f"if ({step} > 0) {{", indent + 2)
        self.emit(f"if ({counter} >= {stop}) break;", indent + 3)
        self.emit("} else {", indent + 2)
        self.emit(f"if ({counter} <= {stop}) break;", indent + 3)
        self.emit("}", indent + 2)
        self.emit(f"if ({obj}) {{ Py_DecRef({obj}); {obj} = 0; }}", indent + 2)
        self.emit(f"{held} = {counter};", indent + 2)
        self.emit(f"{state} = 1;", indent + 2)
        self.emit(f"{counter} = {counter} + {step};", indent + 2)
        self.emit("} else {", indent + 1)
        self.emit(f"{item} = PyIter_Next({iterator});", indent + 2)
        self.emit(
            f"if (!{item}) {{ if (PyErr_Occurred()) {{ {self.failure()} }} break; }}",
            indent + 2,
        )
        self.store_object(name, item, indent + 2)
        self.emit("}", indent + 1)
        self.loop_depth += 1
        try:
            with self.settled_within({name}):
                for statement in node.body:
                    self.statement(statement, indent + 1)
        finally:
            self.loop_depth -= 1
        self.emit("}", indent)
        self.emit(f"if ({iterator}) Py_DecRef({iterator});", indent)
        self.finish_loop(node, broke, indent)

    def call_range(self, arguments: list[str], indent: int) -> str:
        """Call the builtin `range` with values already in hand.

        Only reached when the counting was declined, so this is not on any
        path that matters for speed - it is here so that a program whose
        bounds are wider than a machine word behaves exactly as before.
        """

        callable_ = self.builtin("range", indent)
        array = self.argument_array(len(arguments))
        for offset, value in enumerate(arguments):
            self.emit(f"{array}[{offset}] = {value};", indent)
        built = self.temporary()
        self.emit(
            f"{built} = PyObject_Vectorcall({callable_}, {array}, "
            f"{len(arguments)}, 0);",
            indent,
        )
        self.emit(f"Py_DecRef({callable_});", indent)
        return self.checked(built, indent)

    def temporary_flag(self) -> str:
        assert self.current is not None
        self.current.temporaries += 1
        name = f"_c{self.current.temporaries}"
        # Declared as `int <name>`, so the membership test has to look for that
        # spelling: comparing the bare name never matched, and a reused slot
        # was declared a second time.
        if f"int {name}" not in self.current.locals:
            self.current.locals.append(f"int {name}")
        return name

    def give_back(self, node: ast.Return, indent: int) -> None:
        # A bare return is `return None`, which is what falling off the end
        # gives too.
        value = (
            self.builtin("None", indent)
            if node.value is None
            else self.expression(node.value, indent)
        )
        if self.finallys:
            # Leaving through a `finally` is not leaving yet: the value is put
            # aside, the reason recorded, and the clause runs first.
            self.leave_through_finally(_RETURNING, indent, value)
            return
        self.release_locals(indent)
        self.leave(value, indent)

    def mark_broken(self, indent: int) -> None:
        """Say that this loop is being left by `break`, for its `else`."""

        if self.loop_flags and self.loop_flags[-1] is not None:
            self.emit(f"{self.loop_flags[-1]} = 1;", indent)

    def leave_through_finally(
        self, why: int, indent: int, value: str | None = None
    ) -> None:
        """Record why the region is being left, and go run the clause."""

        protection = self.finallys[-1]
        if value is not None:
            self.emit(f"{protection.answer} = {value};", indent)
        self.emit(f"{protection.why} = {why};", indent)
        protection.reasons.add(why)
        self.emit(f"goto {protection.label};", indent)

    def leave(self, value: str, indent: int) -> None:
        """Return from a compiled function, counting back out of the recursion.

        Every path out goes through here. A level entered and not left is
        never recovered, so the interpreter would come to believe the stack is
        deeper than it is and refuse calls that are perfectly fine.
        """

        if self.guards_recursion:
            self.emit("Py_LeaveRecursiveCall();", indent)
        self.emit(f"return {value};", indent)

    def release_locals(self, indent: int, guarded: bool = False) -> None:
        """Give back what the body still holds, on the way out.

        Every name the body bound owns a reference, and leaving without
        releasing them leaks one per call - which a recursive function turns
        into one per level. The value being returned is a temporary and is not
        in this list, so it survives.
        """

        assert self.current is not None
        for name in self.current.parameters:
            # On the failure path a parameter may not have been filled in yet
            # - a default whose expression raised leaves the ones after it
            # NULL - so that path tests before it releases.
            if guarded:
                self.emit(f"if (p_{name}) Py_DecRef(p_{name});", indent)
            else:
                self.emit(f"Py_DecRef(p_{name});", indent)
        # Captures are borrowed from the tuple the callable holds - see the
        # binding in `write_closure` - so there is nothing of theirs to give
        # back here.
        for name in self.current.locals:
            # Only the names the program bound. A temporary was released where
            # it was consumed, so releasing it again here would be a second
            # drop of a reference this code no longer owns.
            if not name.startswith("v_"):
                continue
            self.emit(f"if ({name}) Py_DecRef({name});", indent)

    # --- closures --------------------------------------------------------

    @classmethod
    def scope_names(cls, node: ast.AST) -> tuple[set[str], set[str]]:
        """The names a nested function binds, and the names it reads.

        Everything under the node counts, including the body of a `for` or a
        `try`, because Python has no block scope: a name bound anywhere in a
        function is local to the whole of it. A function *inside* this one is
        a scope of its own, so what it binds stays there and only the names it
        could not resolve for itself are read from here - which is how a
        capture reaches through two levels.
        """

        arguments = node.args
        bound = {argument.arg for argument in arguments.args}
        bound.update(argument.arg for argument in arguments.posonlyargs)
        bound.update(argument.arg for argument in arguments.posonlyargs)
        bound.update(argument.arg for argument in arguments.kwonlyargs)
        if arguments.vararg:
            bound.add(arguments.vararg.arg)
        if arguments.kwarg:
            bound.add(arguments.kwarg.arg)
        read: set[str] = set()
        body = node.body if isinstance(node.body, list) else [node.body]
        for statement in body:
            cls.gather_names(statement, bound, read)
        return bound, read

    @classmethod
    def gather_names(cls, node: ast.AST, bound: set[str], read: set[str]) -> None:
        """Add what this node binds and reads, not descending into a scope."""

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not isinstance(node, ast.Lambda):
                bound.add(node.name)
            inner_bound, inner_read = cls.scope_names(node)
            read.update(inner_read - inner_bound)
            # A default is evaluated where the `def` is, not where it is
            # called, so its names belong to this scope.
            for default in node.args.defaults:
                cls.gather_names(default, bound, read)
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                read.add(node.id)
            else:
                bound.add(node.id)
            return
        if isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.alias):
            bound.add(node.asname or node.name.split(".")[0])
        for child in ast.iter_child_nodes(node):
            cls.gather_names(child, bound, read)

    @staticmethod
    def bindings_in(statements: list[ast.stmt]) -> list[tuple[str, int]]:
        """Every name these statements bind, with the line that binds it."""

        found: list[tuple[str, int]] = []
        for statement in statements:
            for inner in ast.walk(statement):
                line = getattr(inner, "lineno", 0)
                if isinstance(inner, ast.Name) and not isinstance(
                    inner.ctx, ast.Load
                ):
                    found.append((inner.id, line))
                elif isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found.append((inner.name, line))
                elif isinstance(inner, ast.ExceptHandler) and inner.name:
                    found.append((inner.name, line))
                elif isinstance(inner, ast.alias):
                    found.append((inner.asname or inner.name.split(".")[0], line))
        return found

    def refuse_late_binding(self, node: ast.AST, captures: tuple[str, ...]) -> None:
        """Refuse a capture whose value would still be moving.

        Python closes over the *variable*, so a lambda made in a loop sees the
        loop's last value, not the value at the moment it was made. This
        captures by value at that moment, which is the same answer whenever the
        captured name is settled by then and a different one when it is not.
        Rather than quietly disagree, the cases where it would are refused:
        a name the enclosing scope binds again after this line, and a name
        bound inside a loop that this definition sits in.
        """

        line = getattr(node, "lineno", 0)
        moving: set[str] = set()
        for name, bound_at in self.bindings_in(self.scope):
            if bound_at > line and name in captures:
                moving.add(name)
        for statement in self.scope:
            for inner in ast.walk(statement):
                if not isinstance(inner, (ast.For, ast.While)):
                    continue
                start = getattr(inner, "lineno", 0)
                end = getattr(inner, "end_lineno", start) or start
                if not start <= line <= end:
                    continue
                # The `for` target counts as much as anything the body binds:
                # it is the name that moves on every turn of the loop.
                inside = list(inner.body)
                if isinstance(inner, ast.For):
                    inside.append(inner.target)
                for name, _ in self.bindings_in(inside):
                    if name in captures:
                        moving.add(name)
        if moving:
            spelled = ", ".join(sorted(moving))
            raise self.fail(
                node,
                f"this closure captures {spelled}, which the enclosing scope "
                "binds again afterwards; Python would see the later value and "
                "this captures the one in hand, so it is refused rather than "
                "quietly disagreeing",
            )

    def apply_decorators(self, decorators, value: str, indent: int) -> str:
        """`@a` then `@b` on a `def` is `a(b(f))`.

        The list is in source order and they are applied from the bottom up,
        which is the order Python applies them in.
        """

        for decorator in reversed(decorators):
            applied = self.expression(decorator, indent)
            wrapped = self.temporary()
            self.emit(
                f"{wrapped} = PyObject_CallOneArg({applied}, {value});", indent
            )
            self.emit(f"Py_DecRef({applied});", indent)
            self.emit(f"Py_DecRef({value});", indent)
            self.checked(wrapped, indent)
            value = wrapped
        return value

    def qualified(self, name: str) -> str:
        """The name Python would show for something defined right here.

        A function's own names sit under `<locals>`; a class's do not.
        """

        parts: list[str] = []
        for scope, is_function in self.scope_path:
            parts.append(scope)
            if is_function:
                parts.append("<locals>")
        parts.append(name)
        return ".".join(parts)

    def make_closure(
        self, node: ast.AST, label: str, indent: int, display: str | None = None
    ) -> str:
        """A nested `def` or a `lambda`, as a real Python callable.

        The body becomes a C function with the `(self, args)` shape CPython
        calls, and `PyCFunction_New` wraps it in an object. What it captured
        travels as the `self` that object holds, which is exactly how CPython
        gives a C function state of its own.
        """

        assert self.current is not None
        arguments = node.args
        bound, read = self.scope_names(node)
        captures = tuple(
            sorted(name for name in read - bound if self.bound_around(name))
        )
        self.refuse_late_binding(node, captures)
        for scope in self.shadowed:
            caught = [name for name in captures if name in scope]
            if caught:
                # A comprehension binds its target once and rebinds it each
                # turn, so a closure made inside one shares that binding and
                # every closure sees the last value - `[lambda: i for i in
                # range(3)]` is `[2, 2, 2]`. Captures here are taken by value
                # when the closure is made, which would answer `[0, 1, 2]`.
                # Refused rather than answered differently; `lambda i=i: i`
                # says the other thing and compiles.
                raise self.fail(
                    node,
                    f"this closure captures {', '.join(sorted(caught))} from "
                    f"the comprehension around it, which rebinds it on every "
                    f"turn - captures are taken by value here, so every "
                    f"closure would see a different value where Python gives "
                    f"them all the last one. Write it as a default "
                    f"(`lambda {sorted(caught)[0]}={sorted(caught)[0]}: ...`) "
                    f"to say the by-value thing",
                )
        # A name the body reads that this scope has no slot for *yet*, but
        # binds further down: mutual recursion between two nested functions is
        # the shape that reaches here. Captures are taken by value when the
        # closure is made, so the second name is simply absent and the call
        # failed at run time with a `NameError` naming a function plainly
        # written above it. Refused with an explanation instead - the same
        # choice this module makes wherever capture-by-value would disagree
        # with Python rather than merely be slower.
        pending = sorted(
            name
            for name in (read - bound)
            if name not in self.globals
            and name not in self.known_functions
            and self.reference(name) is None
            and any(
                bound_name == name and bound_at > getattr(node, "lineno", 0)
                for bound_name, bound_at in self.bindings_in(self.scope)
            )
        )
        if pending:
            raise self.fail(
                node,
                f"this closure reads {', '.join(pending)}, which "
                f"{'are' if len(pending) > 1 else 'is'} bound further down the "
                "enclosing scope; captures are taken by value when the closure "
                "is made, so the name would not be there when it ran - write "
                "the definitions the other way round, or move them to module "
                "level where the order does not matter",
            )

        index = len(self.method_table)
        c_name = f"_closure{index}_{label}"
        self.method_table.append((c_name, label, _text_signature(label, node.args)))

        held = self.temporary()
        # Python evaluates a default when the `def` runs, once, and every
        # call afterwards is handed that same object - which is what makes
        # `def f(x=[])` share one list. They are evaluated here, then, and
        # travel with the captures in the tuple the callable holds, so a `def`
        # inside a loop gets a set of its own each time round.
        given = list(node.args.defaults) + list(node.args.kw_defaults)
        self.emit(
            f"{held} = PyTuple_New({len(captures) + len(given)});", indent
        )
        self.checked(held, indent)
        for offset, name in enumerate(captures):
            source = self.reference(name)
            # PyTuple_SetItem steals, and the enclosing scope still needs its
            # own reference, so one is added for the tuple to consume.
            self.emit(f"Py_IncRef({source});", indent)
            self.emit(f"PyTuple_SetItem({held}, {offset}, {source});", indent)
        for offset, default in enumerate(given):
            place = len(captures) + offset
            if default is None:
                # A keyword-only parameter with no default keeps its place so
                # the ones after it do not move. Nothing ever reads it.
                blank = self.builtin("None", indent)
                self.emit(f"PyTuple_SetItem({held}, {place}, {blank});", indent)
                continue
            value = self.expression(default, indent)
            self.emit(f"PyTuple_SetItem({held}, {place}, {value});", indent)
        target = self.temporary()
        self.emit(
            f"{target} = PyCFunction_New(&_py2bin_methods[{index}], {held});",
            indent,
        )
        self.emit(f"Py_DecRef({held});", indent)
        self.checked(target, indent)
        if label in captures:
            # A nested function that calls itself. Its own name is a capture,
            # and at the moment the tuple is filled that name is not bound yet
            # - the `def` being compiled is what binds it - so the slot took a
            # NULL and every recursive call failed with `NameError`, for a
            # shape as ordinary as a nested `fact`.
            #
            # The slot is filled with the callable once it exists.
            # `PyTuple_SetItem` refuses a tuple anything else holds, which is
            # why this comes *after* the reference above is dropped: the
            # callable is then the only owner. It leaves a cycle, callable to
            # tuple to callable, which is collectable and is the same cycle
            # CPython's own closure cells make.
            self.emit(f"Py_IncRef({target});", indent)
            self.emit(
                f"PyTuple_SetItem({held}, {captures.index(label)}, {target});",
                indent,
            )

        self.write_closure(
            node, c_name, captures, display or self.qualified(label), label
        )
        return target

    def class_definition(self, node: ast.ClassDef, indent: int) -> None:
        """`class C: ...` at a place where a name is bound."""

        made = self.build_class(node, indent)
        target = self.declare(node.name)
        self.emit(f"if ({target}) Py_DecRef({target});", indent)
        self.emit(f"{target} = {made};", indent)
        self.publish(node.name, target, indent)
        if self.at_module_level:
            self.show_on_module(node.name, target, indent)

    def build_class(self, node: ast.ClassDef, indent: int) -> str:
        """`class C(Base):` - the namespace built, then handed to `type`.

        A class is what `type(name, bases, namespace)` answers, and that is
        the interpreter's own class machinery rather than a re-implementation
        of it: inheritance, `__init__`, `__repr__`, attribute lookup and the
        method resolution order all behave because CPython is the one doing
        them.

        A method is a closure like any other, wrapped in `instancemethod` on
        the way into the namespace. A raw `PyCFunction` is not a descriptor
        and so would never bind - the instance would simply not arrive. The
        wrapper binds it and passes the instance first, which lands in the
        argument tuple at position zero, exactly where the compiled body
        already reads its first parameter from.

        This was `functools.partialmethod`, which binds correctly and is
        written in Python: every `obj.method` ran interpreted code and built a
        `functools.partial`, and a method call measured sixteen times slower
        than CPython running the same class. `instancemethod` is CPython's own
        C type for this - its `__get__` is `PyMethod_New` and nothing else -
        and it is what a plain Python function does, so the semantics are the
        ones being copied rather than an approximation of them.
        """

        # `class A(metaclass=M)` calls M rather than `type` to make the
        # class; any other keyword in the header is handed to it as well,
        # which is where `__init_subclass__` reads them from. A metaclass a
        # *base* carries needs nothing here: `type(name, bases, ns)` works out
        # the most derived metaclass of the bases itself and hands over to it,
        # which is why inheriting from one already worked.
        chosen = None
        header = []
        for keyword in node.keywords:
            if keyword.arg is None:
                raise self.fail(
                    node,
                    "a class header that spreads a mapping is not translated "
                    "here yet",
                )
            if keyword.arg == "metaclass":
                chosen = keyword.value
            else:
                header.append(keyword)
        # Before the body, which is the order Python does it in - `class
        # C(f())` calls f first - and which the namespace now depends on:
        # a metaclass gets to choose what the body is populated into.
        bases = self.tuple_literal(
            ast.copy_location(ast.Tuple(elts=node.bases, ctx=ast.Load()), node),
            indent,
        )
        title = self.temporary()
        self.emit(f"{title} = PyUnicode_FromString({_c_string(node.name)});", indent)
        self.checked(title, indent)
        if chosen is None:
            maker = self.builtin("type", indent)
        else:
            maker = self.expression(chosen, indent)
        # PEP 560, and only now: `__build_class__` evaluates every argument
        # it is given - the bases *and* the `metaclass=` expression - and runs
        # `update_bases` inside itself afterwards. Resolving before `maker`
        # was built would call `__mro_entries__` first, which a program can
        # see when either expression has a side effect.
        written = bases
        if node.bases:
            bases = self.resolve_mro_entries(bases, indent)
        self.scope_path.append((node.name, False))
        # `__qualname__` is the path, not the bare name: `Outer.Inner`, and
        # `make.<locals>.Local` for one made in a function. It is what `repr`
        # of the class and of every instance shows, so a nested class printed
        # as though it were at module level until now. The scope path already
        # has this class on it - `qualified` wants the *name*, so it is asked
        # for the one below.
        walked = self.qualified("")[:-1] or node.name
        namespace = self.class_namespace(
            node, maker, title, bases, indent,
            seeded=chosen is None, qualname=walked,
        )
        if node.bases:
            # What the header said, kept under the name `typing` reads to
            # recover `Generic[T]` from a class whose bases were replaced.
            # Only when they were: `__build_class__` leaves it off otherwise.
            self.emit(f"if ({written} != {bases}) {{", indent)
            self.emit(
                f"if (PyObject_SetItem({namespace}, "
                f"{self.interned('__orig_bases__')}, {written}) < 0) "
                f"{{ {self.failure()} }}",
                indent + 1,
            )
            self.emit("}", indent)
        binder = None
        # What the body has bound so far. A decorator and an attribute's value
        # run as the class is built and can see it; a method body cannot, and
        # is written with this off.
        known: set[str] = set()
        for statement in node.body:
            if isinstance(statement, ast.Pass):
                continue
            if isinstance(statement, ast.Expr) and isinstance(
                statement.value, ast.Constant
            ):
                continue  # a docstring
            if isinstance(statement, ast.FunctionDef):
                binder = True
                first = (
                    statement.args.posonlyargs or statement.args.args
                )
                self.methods_of.append(
                    (node.name, first[0].arg if first else "")
                )
                try:
                    body = self.make_closure(statement, statement.name, indent)
                finally:
                    self.methods_of.pop()
                if statement.decorator_list:
                    self.class_scope.append((namespace, known))
                    # The decorator is handed the plain callable, not the
                    # bound wrapper. `staticmethod`, `classmethod` and
                    # `property` are descriptors that do their own binding,
                    # and an ordinary wrapping decorator returns a Python
                    # function - which binds by itself and passes the instance
                    # as the first argument, which is where a compiled method
                    # reads it from anyway.
                    try:
                        value = self.apply_decorators(
                            statement.decorator_list, body, indent
                        )
                    finally:
                        self.class_scope.pop()
                elif statement.name in _IMPLICITLY_WRAPPED:
                    # Python wraps three of these itself, whether or not the
                    # class says so: `__new__` is a staticmethod and
                    # `__init_subclass__` and `__class_getitem__` are
                    # classmethods. Bound as ordinary methods they were handed
                    # an instance that does not exist yet, and
                    # `__init_subclass__` reported `cls` missing.
                    wrapper = self.builtin(
                        _IMPLICITLY_WRAPPED[statement.name], indent
                    )
                    value = self.temporary()
                    self.emit(
                        f"{value} = PyObject_CallOneArg({wrapper}, {body});",
                        indent,
                    )
                    self.emit(f"Py_DecRef({wrapper});", indent)
                    self.emit(f"Py_DecRef({body});", indent)
                    self.checked(value, indent)
                else:
                    value = self.temporary()
                    self.emit(f"{value} = PyInstanceMethod_New({body});", indent)
                    self.emit(f"Py_DecRef({body});", indent)
                    self.checked(value, indent)
                key = statement.name
            elif isinstance(statement, ast.Assign) and len(
                statement.targets
            ) == 1 and isinstance(statement.targets[0], ast.Name):
                self.class_scope.append((namespace, known))
                try:
                    value = self.expression(statement.value, indent)
                finally:
                    self.class_scope.pop()
                if isinstance(statement.value, ast.Lambda):
                    # `f = lambda self: ...` in a class body is a method, and
                    # binds like one. A compiled function is a PyCFunction and
                    # does not bind itself, so it arrived unbound and the call
                    # said `self` was missing - the same wrapping a `def` in
                    # the same place already gets.
                    bound = self.temporary()
                    self.emit(
                        f"{bound} = PyInstanceMethod_New({value});", indent
                    )
                    self.emit(f"Py_DecRef({value});", indent)
                    value = self.checked(bound, indent)
                key = statement.targets[0].id
            elif isinstance(statement, ast.ClassDef):
                # A class in a class body is a value the body binds, like an
                # attribute - it is made here and put in the namespace under
                # its own name. Refusing it refused `class Outer: class
                # Inner`, which is how a nested enum or a config block is
                # usually written.
                value = self.build_class(statement, indent)
                key = statement.name
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            ):
                # `n: int = 3` in a class body is a class attribute. `n: int`
                # alone binds nothing - but both are recorded in the class's
                # `__annotations__`, which is not decoration: `dataclasses`
                # reads exactly that to find out what the fields are, and with
                # no `__annotations__` every dataclass came out with no fields
                # and an `__init__` taking nothing.
                self.note_annotation(
                    namespace, statement.target.id, statement.annotation, indent
                )
                if statement.value is None:
                    continue
                self.class_scope.append((namespace, known))
                try:
                    value = self.expression(statement.value, indent)
                finally:
                    self.class_scope.pop()
                key = statement.target.id
            else:
                # Anything else - an `if` guarding a platform, a `for`
                # filling a table, a `try` around an optional import. It runs
                # as ordinary statements and whatever it binds is moved into
                # the namespace afterwards.
                self.class_body_statement(statement, namespace, known, indent)
                continue
            known.add(key)
            named = self.interned(key)
            # Through the mapping protocol: `__prepare__` may have answered
            # with something that is not a dict, and `enum` does exactly that
            # - its namespace is what notices a repeated member name.
            self.emit(
                f"if (PyObject_SetItem({namespace}, {named}, {value}) < 0) "
                f"{{ {self.failure()} }}",
                indent,
            )
            self.emit(f"Py_DecRef({value});", indent)
        del binder  # `instancemethod` needs nothing fetched and so nothing freed
        # All three were built before the body ran and are released together
        # below, so each gets a reference of its own for PyTuple_SetItem to
        # steal. Handing over the only one and releasing it afterwards took
        # the class apart while `type` was still reading it.
        self.emit(f"Py_IncRef({title});", indent)
        self.emit(f"Py_IncRef({bases});", indent)
        self.emit(f"Py_IncRef({namespace});", indent)
        arguments = self.temporary()
        self.emit(f"{arguments} = PyTuple_New(3);", indent)
        self.checked(arguments, indent)
        # PyTuple_SetItem steals all three, which is what to do with the
        # references this is finished with anyway.
        self.emit(f"PyTuple_SetItem({arguments}, 0, {title});", indent)
        self.emit(f"PyTuple_SetItem({arguments}, 1, {bases});", indent)
        self.emit(f"PyTuple_SetItem({arguments}, 2, {namespace});", indent)
        named_arguments = "0"
        if header:
            named_arguments = self.temporary()
            self.emit(f"{named_arguments} = PyDict_New();", indent)
            self.checked(named_arguments, indent)
            for keyword in header:
                value = self.expression(keyword.value, indent)
                key = self.interned(keyword.arg)
                self.emit(
                    f"if (PyDict_SetItem({named_arguments}, {key}, {value}) < 0)"
                    f" {{ {self.failure()} }}",
                    indent,
                )
                self.emit(f"Py_DecRef({value});", indent)
        made = self.temporary()
        self.emit(
            f"{made} = PyObject_Call({maker}, {arguments}, {named_arguments});",
            indent,
        )
        self.emit(f"Py_DecRef({maker});", indent)
        self.emit(f"Py_DecRef({arguments});", indent)
        self.emit(f"Py_DecRef({title});", indent)
        self.emit(f"Py_DecRef({bases});", indent)
        self.emit(f"Py_DecRef({namespace});", indent)
        if named_arguments != "0":
            self.emit(f"Py_DecRef({named_arguments});", indent)
        self.checked(made, indent)
        self.scope_path.pop()
        return self.apply_decorators(node.decorator_list, made, indent)

    def derive_metaclass(
        self, maker: str, bases: str, indent: int, seeded: bool = False
    ) -> None:
        """Replace `maker` with the most derived metaclass among the bases.

        Python's rule: for each base take its metaclass if that is the more
        derived of the two. A pair that are unrelated is a conflict, which
        `type` reports when it gets there - so this leaves `maker` alone and
        lets it.

        Where the header names no metaclass, `__build_class__` starts from
        the type of the *first base* rather than from `type`, and only then
        compares the rest. Starting from `type` is not the same thing when a
        base is not a class at all: for `class Bad(42)` the first base's type
        is `int`, which is not a subclass of `type`, so nothing ever replaced
        `type` and the conflict message came out - where CPython gets as far
        as calling `int` and answers `int expected at most 2 arguments, got
        3`. Two different complaints about the same program, and CPython's is
        the one about what the program actually did.
        """

        if seeded:
            first = self.temporary()
            self.emit(f"{first} = PyTuple_GetItem({bases}, 0);", indent)
            self.emit(f"if (!{first}) {{ {self.failure()} }}", indent)
            self.emit(f"Py_DecRef({maker});", indent)
            self.emit(
                f'{maker} = PyObject_GetAttrString({first}, "__class__");',
                indent,
            )
            self.emit(f"if (!{maker}) {{ {self.failure()} }}", indent)

        counter = self.temporary_flag()
        span = self.temporary_flag()
        base = self.temporary()
        kind = self.temporary()
        verdict = self.temporary_flag()
        checker = self.builtin("issubclass", indent)
        self.emit(f"{span} = (int)PyObject_Size({bases});", indent)
        self.emit(
            f"for ({counter} = 0; {counter} < {span}; "
            f"{counter} = {counter} + 1) {{",
            indent,
        )
        self.emit(f"{base} = PyTuple_GetItem({bases}, {counter});", indent + 1)
        self.emit(f"if (!{base}) {{ {self.failure()} }}", indent + 1)
        self.emit(
            f'{kind} = PyObject_GetAttrString({base}, "__class__");', indent + 1
        )
        self.emit(f"if (!{kind}) {{ {self.failure()} }}", indent + 1)
        held = self.temporary()
        # A slot at a time, not PyTuple_Pack: that one is variadic, and on
        # Apple's arm64 ABI variadic arguments go on the stack where this
        # backend passes registers, so a fixed prototype reads two addresses
        # nothing wrote. It does not crash here - it hands `issubclass` two
        # words of rubbish, which said "arg 1 must be a class".
        self.emit(f"{held} = PyTuple_New(2LL);", indent + 1)
        self.emit(f"if (!{held}) {{ {self.failure()} }}", indent + 1)
        self.emit(f"Py_IncRef({kind});", indent + 1)
        self.emit(f"PyTuple_SetItem({held}, 0, {kind});", indent + 1)
        self.emit(f"Py_IncRef({maker});", indent + 1)
        self.emit(f"PyTuple_SetItem({held}, 1, {maker});", indent + 1)
        answer = self.temporary()
        self.emit(
            f"{answer} = PyObject_Call({checker}, {held}, 0);", indent + 1
        )
        self.emit(f"Py_DecRef({held});", indent + 1)
        self.emit(f"if (!{answer}) {{ {self.failure()} }}", indent + 1)
        self.emit(f"{verdict} = PyObject_IsTrue({answer});", indent + 1)
        self.emit(f"Py_DecRef({answer});", indent + 1)
        self.emit(f"if ({verdict} < 0) {{ {self.failure()} }}", indent + 1)
        # More derived than what is held, so it wins; otherwise this base has
        # nothing to add and `kind` is dropped.
        self.emit(f"if ({verdict}) {{", indent + 1)
        self.emit(f"Py_DecRef({maker});", indent + 2)
        self.emit(f"{maker} = {kind};", indent + 2)
        self.emit("} else {", indent + 1)
        self.emit(f"Py_DecRef({kind});", indent + 2)
        self.emit("}", indent + 1)
        self.emit("}", indent)
        self.emit(f"Py_DecRef({checker});", indent)

    def note_annotation(
        self, namespace: str, name: str, annotation: ast.expr, indent: int
    ) -> None:
        """Record `name: kind` in the class's `__annotations__`.

        Made on first use rather than always, so a class that annotates
        nothing does not carry an empty dictionary it never had in Python.
        """

        held = self.temporary()
        key = self.interned("__annotations__")
        self.emit(f"{held} = PyObject_GetItem({namespace}, {key});", indent)
        self.emit(f"if (!{held}) {{", indent)
        self.emit("PyErr_Clear();", indent + 1)
        self.emit(f"{held} = PyDict_New();", indent + 1)
        self.emit(f"if (!{held}) {{ {self.failure()} }}", indent + 1)
        self.emit(
            f"if (PyObject_SetItem({namespace}, {key}, {held}) < 0) "
            f"{{ {self.failure()} }}",
            indent + 1,
        )
        self.emit("}", indent)
        # The annotation is evaluated, as it is in a class body - `x: int`
        # looks `int` up there and then.
        value = self.expression(annotation, indent)
        self.emit(
            f"if (PyObject_SetItem({held}, {self.interned(name)}, {value}) < 0)"
            f" {{ {self.failure()} }}",
            indent,
        )
        self.emit(f"Py_DecRef({value});", indent)
        self.emit(f"Py_DecRef({held});", indent)

    def class_body_statement(
        self, statement: ast.stmt, namespace: str, known: set[str], indent: int
    ) -> None:
        """A statement in a class body that is not simply binding a name.

        A class body is code, and only its *bindings* are special - they end
        up in the namespace the class is made from rather than in a scope.
        Everything else is ordinary: `if TYPE_CHECKING:` around an
        annotation, a `for` filling a table of constants, a `try` around an
        optional import. All of it was refused.

        The names it binds are renamed to ones nothing else uses, so running
        it cannot disturb a name of the same spelling in the scope around the
        class - `v = 99` outside and `v = 1` inside a class body are two
        different names, and Python keeps them apart. What each one ends up
        holding is moved into the namespace afterwards, and only if it was
        bound at all: an `if` with no `else` binds nothing down the other
        branch.
        """

        # Every name it *may* bind, not only the ones it certainly binds:
        # an `if` with two branches binds a name down either, and which one
        # ran is not known here.
        dotted = _dotted_plain_import(statement)
        if dotted is not None:
            raise self.fail(
                statement,
                f"`import {dotted}` inside a class body's `if`, `for` or "
                f"`try` is not translated here yet - it binds "
                f"{dotted.split('.')[0]!r}, and the rename that keeps a class "
                f"body's names out of the scope around it cannot express "
                f"that. Give the import an alias, so it binds a name of "
                f"its own, and it compiles.",
            )
        bound = sorted(_scope_bindings([statement]))
        renamed = {name: f"_py2bin_cls{id(statement) & 0xFFFF}_{name}" for name in bound}
        if renamed:
            statement = _Renamed(renamed).visit(
                copy.deepcopy(statement)
            )
            ast.fix_missing_locations(statement)
        self.class_scope.append((namespace, known))
        try:
            self.statement(statement, indent)
        finally:
            self.class_scope.pop()
        for name in bound:
            slot = self.declare(renamed[name])
            self.emit(f"if ({slot}) {{", indent)
            self.emit(
                f"if (PyObject_SetItem({namespace}, {self.interned(name)}, "
                f"{slot}) < 0) {{ {self.failure()} }}",
                indent + 1,
            )
            # Let go of it here: it was only ever somewhere to put the value
            # between the statement running and the namespace taking it.
            self.emit(f"Py_DecRef({slot});", indent + 1)
            self.emit(f"{slot} = 0;", indent + 1)
            self.emit("}", indent)
            known.add(name)

    def resolve_mro_entries(self, bases: str, indent: int) -> str:
        """PEP 560: a base that is not a class says what to put in its place.

        `class Box(Generic[T])` names something that is not a class at all,
        and `type` said so - "does not support MRO entry resolution". Python
        asks any such entry for `__mro_entries__`, hands it the whole bases
        tuple as written, and splices the tuple that comes back in.

        A base that is already a class is passed over untouched rather than
        asked: `__mro_entries__` looked up on a class would find one defined
        in a base and call it with the bases tuple as `self`.

        Answers with the same tuple when nothing was replaced, so the caller
        can tell by comparing - which is how it knows whether to record
        `__orig_bases__`.
        """

        gathered = self.temporary()
        changed = self.temporary_flag()
        counter = self.temporary_flag()
        span = self.temporary_flag()
        base = self.temporary()
        verdict = self.temporary_flag()
        entries = self.temporary()
        answer = self.temporary()
        inner = self.temporary_flag()
        reach = self.temporary_flag()
        piece = self.temporary()
        kind = self.builtin("type", indent)
        self.emit(f"{gathered} = PyList_New(0LL);", indent)
        self.checked(gathered, indent)
        self.emit(f"{changed} = 0;", indent)
        self.emit(f"{span} = (int)PyObject_Size({bases});", indent)
        self.emit(
            f"for ({counter} = 0; {counter} < {span}; "
            f"{counter} = {counter} + 1) {{",
            indent,
        )
        self.emit(f"{base} = PyTuple_GetItem({bases}, {counter});", indent + 1)
        self.emit(f"if (!{base}) {{ {self.failure()} }}", indent + 1)
        self.emit(
            f"{verdict} = PyObject_IsInstance({base}, {kind});", indent + 1
        )
        self.emit(f"if ({verdict} < 0) {{ {self.failure()} }}", indent + 1)
        self.emit(f"{entries} = 0;", indent + 1)
        self.emit(f"if (!{verdict}) {{", indent + 1)
        self.emit(
            f'{entries} = PyObject_GetAttrString({base}, "__mro_entries__");',
            indent + 2,
        )
        self.emit(f"if (!{entries}) {{ PyErr_Clear(); }}", indent + 2)
        self.emit("}", indent + 1)
        self.emit(f"if (!{entries}) {{", indent + 1)
        self.emit(f"PyList_Append({gathered}, {base});", indent + 2)
        self.emit("} else {", indent + 1)
        # The tuple as written, which is what the hook is told about.
        self.emit(
            f"{answer} = PyObject_CallOneArg({entries}, {bases});", indent + 2
        )
        self.emit(f"Py_DecRef({entries});", indent + 2)
        self.emit(f"if (!{answer}) {{ {self.failure()} }}", indent + 2)
        self.emit(f"{changed} = 1;", indent + 2)
        self.emit(f"{reach} = (int)PyObject_Size({answer});", indent + 2)
        self.emit(
            f"for ({inner} = 0; {inner} < {reach}; {inner} = {inner} + 1) {{",
            indent + 2,
        )
        self.emit(f"{piece} = PyTuple_GetItem({answer}, {inner});", indent + 3)
        self.emit(f"if (!{piece}) {{ {self.failure()} }}", indent + 3)
        self.emit(f"PyList_Append({gathered}, {piece});", indent + 3)
        self.emit("}", indent + 2)
        self.emit(f"Py_DecRef({answer});", indent + 2)
        self.emit("}", indent + 1)
        self.emit("}", indent)
        self.emit(f"Py_DecRef({kind});", indent)
        settled = self.temporary()
        self.emit(f"if ({changed}) {{", indent)
        maker = self.builtin("tuple", indent + 1)
        self.emit(
            f"{settled} = PyObject_CallOneArg({maker}, {gathered});", indent + 1
        )
        self.emit(f"Py_DecRef({maker});", indent + 1)
        self.emit(f"if (!{settled}) {{ {self.failure()} }}", indent + 1)
        self.emit("} else {", indent)
        self.emit(f"Py_IncRef({bases});", indent + 1)
        self.emit(f"{settled} = {bases};", indent + 1)
        self.emit("}", indent)
        self.emit(f"Py_DecRef({gathered});", indent)
        return settled

    def class_namespace(
        self,
        node: ast.ClassDef,
        maker: str,
        title: str,
        bases: str,
        indent: int,
        seeded: bool = False,
        qualname: str | None = None,
    ) -> str:
        """What the class body is populated into, and what has to be in it first.

        A plain dict is only right when the class is made by `type`. A
        metaclass may answer `__prepare__` with a mapping of its own, and
        `enum` does: its namespace is what notices a member name used twice,
        so populating a dict instead and handing that over made every `Enum`
        subclass fail. Asked for at run time rather than decided here, because
        which metaclass it is need not be known until then.

        `__module__` and `__qualname__` go in before the body, as they do in
        Python. Without them `dataclasses` raised `AttributeError: __module__`
        the moment it looked at the class it was decorating.
        """

        namespace = self.temporary()
        if node.bases:
            # Which metaclass to ask is the most derived of the ones the bases
            # carry, not `type` - `class C(Enum)` is made by `EnumMeta`, whose
            # `__prepare__` answers with a mapping that notices a member name
            # used twice. Asking `type` got a plain dict and every `Enum`
            # subclass failed. The *call* below needs none of this: `type`
            # works the winner out itself and hands over.
            self.derive_metaclass(maker, bases, indent, seeded=seeded)
        prepare = self.temporary()
        self.emit(
            f'{prepare} = PyObject_GetAttrString({maker}, "__prepare__");',
            indent,
        )
        self.emit(f"if ({prepare}) {{", indent)
        packed = self.temporary()
        # See `derive_metaclass` for why this is not PyTuple_Pack.
        self.emit(f"{packed} = PyTuple_New(2LL);", indent + 1)
        self.emit(f"if (!{packed}) {{ {self.failure()} }}", indent + 1)
        self.emit(f"Py_IncRef({title});", indent + 1)
        self.emit(f"PyTuple_SetItem({packed}, 0, {title});", indent + 1)
        self.emit(f"Py_IncRef({bases});", indent + 1)
        self.emit(f"PyTuple_SetItem({packed}, 1, {bases});", indent + 1)
        self.emit(
            f"{namespace} = PyObject_Call({prepare}, {packed}, 0);", indent + 1
        )
        self.emit(f"Py_DecRef({packed});", indent + 1)
        self.emit(f"Py_DecRef({prepare});", indent + 1)
        self.emit(f"if (!{namespace}) {{ {self.failure()} }}", indent + 1)
        self.emit("} else {", indent)
        # Every type has `__prepare__`, so this is the path for a metaclass
        # that is not a type at all - a plain function used as one.
        self.emit("PyErr_Clear();", indent + 1)
        self.emit(f"{namespace} = PyDict_New();", indent + 1)
        self.emit(f"if (!{namespace}) {{ {self.failure()} }}", indent + 1)
        self.emit("}", indent)
        # The program's own `__name__`, read as the name it is. Asking
        # `program_name` for it found `builtins.__name__` first - that
        # function tries the builtins before the program, because for `len`
        # and `Exception` it cannot fail - so every compiled class said it
        # was defined in `builtins`, and `print(SomeClass)` answered
        # `<class 'Plain'>` where CPython says `<class '__main__.Plain'>`.
        for key, value in (
            (
                "__module__",
                self.expression(
                    ast.copy_location(
                        ast.Name(id="__name__", ctx=ast.Load()), node
                    ),
                    indent,
                ),
            ),
            (
                "__qualname__",
                title if qualname is None else self.interned(qualname),
            ),
        ):
            self.emit(
                f"if (PyObject_SetItem({namespace}, {self.interned(key)}, "
                f"{value}) < 0) {{ {self.failure()} }}",
                indent,
            )
            if key == "__module__":
                self.emit(f"Py_DecRef({value});", indent)
        return namespace

    def write_wrapper(self, node: ast.FunctionDef) -> None:
        """A Python callable for a module-level `def`.

        The `def` itself compiles to a plain C function taking its arguments in
        registers, which is the fast shape and the one an ordinary call uses.
        It is not a Python object, though, so `sorted(xs, key=weight)` had
        nothing to pass and `f(*rest)` had no way to say how many arguments it
        was passing. This wrapper is the object: it unpacks the tuple and calls
        the real function, so both spellings reach the same body.

        The arguments are *borrowed* from the tuple and passed on as they are.
        A callee increments what it is given on entry and releases it on the
        way out, so a borrowed reference nets to zero and the tuple keeps
        owning it. An argument the call left out is passed as NULL, which is
        how the callee knows to put its default in.
        """

        parameters = tuple(
            argument.arg
            for argument in (*node.args.posonlyargs, *node.args.args)
        )
        index = len(self.method_table)
        function = _Function(f"_value{index}_{node.name}", (), closure=True)
        self.method_table.append(
            (function.name, node.name, _text_signature(node.name, node.args))
        )
        self.value_functions.append((self.prefix + node.name, index))
        outer = self.current
        self.current = function
        for name in parameters:
            function.locals.append(f"w_{name}")
        # The same binding a closure does - by position, then by name - so a
        # keyword reaches the body whichever spelling the call used. A
        # parameter nothing supplied stays NULL, which is how the callee is
        # asked for its default.
        self.bind_parameters(
            node,
            indent=1,
            keyword_source=None,
            prefix="w_",
            missing_is_null=True,
            display=self.qualified(node.name),
        )
        answer = self.temporary()
        passed = ", ".join(f"w_{name}" for name in parameters)
        self.emit(f"{answer} = f_{self.prefix}{node.name}({passed});", 1)
        for name in parameters:
            self.emit(f"if (w_{name}) Py_DecRef(w_{name});", 1)
        self.leave(answer, 1)
        self.write_unwind(function)
        self.functions.append(function)
        self.current = outer

    def refuse_positional_only_by_name(
        self, names: tuple[str, ...], display: str, indent: int
    ) -> None:
        """`def f(a, /)` called as `f(1, a=2)` is an error, and was not one.

        A positional-only parameter is filled from the tuple and never looked
        for among the keywords, which is right - with a `**kwargs` a keyword
        of the same spelling belongs to it. Without one there is nowhere for
        the name to go, and Python says so. Here the name simply went unread:
        `f(1, a=2)` answered 1, and `f(a=1)` reported the parameter missing
        rather than the keyword that could not fill it.

        CPython names every offending parameter in one message, in the order
        the parameters are declared rather than the order the call named them,
        and reports this ahead of a duplicate, an unknown name or a missing
        argument - so it is emitted before any of those are looked for.
        """

        offenders = self.temporary()
        counter = self.temporary_flag()
        span = self.temporary_flag()
        probe = self.temporary()
        matched = self.temporary_flag()
        self.emit("if (_kwnames) {", indent)
        self.emit(f"{span} = (int)PyObject_Size(_kwnames);", indent + 1)
        self.emit(f"{offenders} = PyList_New(0LL);", indent + 1)
        self.checked(offenders, indent + 1)
        # The parameters outside, the passed names inside, so what comes out
        # is in the order the parameters are declared.
        for name in names:
            key = self.interned(name)
            self.emit(
                f"for ({counter} = 0; {counter} < {span}; "
                f"{counter} = {counter} + 1) {{",
                indent + 1,
            )
            self.emit(
                f"{probe} = PyTuple_GetItem(_kwnames, {counter});", indent + 2
            )
            self.emit(f"if (!{probe}) {{ {self.failure()} }}", indent + 2)
            self.emit(
                f"{matched} = PyObject_RichCompareBool({probe}, {key}, 2);",
                indent + 2,
            )
            self.emit(f"if ({matched} < 0) {{ {self.failure()} }}", indent + 2)
            self.emit(
                f"if ({matched}) {{ PyList_Append({offenders}, {probe}); "
                f"break; }}",
                indent + 2,
            )
            self.emit("}", indent + 1)
        found = self.temporary_flag()
        self.emit(f"{found} = (int)PyObject_Size({offenders});", indent + 1)
        self.emit(f"if ({found} > 0) {{", indent + 1)
        separator = self.temporary()
        joined = self.temporary()
        self.emit(f'{separator} = PyUnicode_FromString(", ");', indent + 2)
        self.checked(separator, indent + 2)
        self.emit(
            f"{joined} = PyUnicode_Join({separator}, {offenders});", indent + 2
        )
        self.emit(f"Py_DecRef({separator});", indent + 2)
        self.checked(joined, indent + 2)
        # `raise_counted` spells its middle piece with `str`, and `str` of a
        # string is that string, so the joined names go straight through.
        self.raise_counted(
            "TypeError",
            f"{display}() got some positional-only arguments passed as "
            f"keyword arguments: '",
            joined,
            "'",
            indent + 2,
        )
        self.emit("}", indent + 1)
        self.emit(f"Py_DecRef({offenders});", indent + 1)
        self.emit("}", indent)

    def bind_parameters(
        self,
        node: ast.AST,
        indent: int,
        keyword_source: str | None,
        prefix: str = "p_",
        missing_is_null: bool = False,
        display: str = "<lambda>",
    ) -> None:
        """Fill each parameter from the argument tuple, then from the keywords.

        Python lets any parameter be passed by name, so reading the tuple alone
        is not enough: `show(1, c=9)` puts nothing at position 2 and `c` in the
        keywords. A function that only looked at the tuple answered with c's
        default and said nothing, which is the worst way to be wrong.

        Where the function has a `**` parameter the keywords are copied first
        and each named parameter is *removed* from the copy as it is taken, so
        what is left is exactly what `**` should see. Without one there is
        nothing to remove them from, and the dict the caller owns is only read.

        """

        arguments = node.args
        # A positional-only parameter is positional and cannot be named, so
        # it is filled from the tuple and never looked for in the keywords -
        # a caller may legitimately pass a keyword of the same spelling, and
        # it belongs to `**kwargs`.
        named_from = len(arguments.posonlyargs)
        positional = tuple(
            argument.arg
            for argument in (*arguments.posonlyargs, *arguments.args)
        )
        keyword_only = tuple(argument.arg for argument in arguments.kwonlyargs)
        defaults = list(arguments.defaults)
        required = len(positional) - len(defaults)

        # The overwhelming majority of calls pass exactly the parameters the
        # function declares, positionally, with no keywords at all. Recognising
        # that in one test skips the whole apparatus below - which builds a
        # dict of the names, looks for each parameter in it, decides whether a
        # default is needed and counts the extras - and leaves a call binding
        # its arguments in two instructions each. Every other call falls into
        # the general path, which is unchanged, so nothing is given up.
        simple = (
            not arguments.vararg
            and not arguments.kwarg
            and not arguments.kwonlyargs
            and not defaults
            and keyword_source is None
            and positional
        )
        closing = 0
        if simple:
            self.emit(f"if (!_kwnames && _nargs == {len(positional)}) {{", indent)
            for offset, name in enumerate(positional):
                slot = f"{prefix}{name}"
                self.emit(f"{slot} = _args[{offset}];", indent + 1)
                self.emit(f"Py_IncRef({slot});", indent + 1)
            self.emit("} else {", indent)
            closing = 1

        wants_names = bool(
            keyword_source or arguments.kwonlyargs or positional[named_from:]
        )
        source = "0"
        # Without a `**` parameter there is nothing to hand leftovers to, so
        # there is no reason to build a dict of them. The names arrive as a
        # tuple; each parameter looks through it. That is O(names x
        # parameters), which for the one or two of each a call actually has
        # beats an allocation, a hash per entry and a probe per parameter -
        # `f(a, key=b)` measured 0.13x the interpreter and the dict was most
        # of what remained.
        scanning = wants_names and not keyword_source
        span = taken = None
        if scanning:
            span = self.temporary_flag()
            taken = self.temporary_flag()
            # Which names found a parameter, one bit each. A count alone says
            # that something went unclaimed but not which, and guessing from
            # "is this a parameter name" named the wrong one: in
            # `g(1, b=2, a=3)` both `a` and `b` are parameters and only `a` is
            # the duplicate.
            claimed = self.machine_slot()
            self.emit(f"{span} = 0;", indent)
            self.emit(f"{taken} = 0;", indent)
            self.emit(f"{claimed} = 0;", indent)
            self.emit("if (_kwnames) {", indent)
            self.emit(f"{span} = (int)PyObject_Size(_kwnames);", indent + 1)
            self.emit("}", indent)
        if wants_names and not scanning:
            # The names arrive as a tuple beside the values, which is the
            # cheapest thing for a caller to hand over. A dict is built from
            # them only when a call actually passed keywords - almost none do,
            # and the ones that do pay for it here rather than every call
            # paying for it at the boundary.
            gathered = (
                f"{prefix}{keyword_source}" if keyword_source else self.temporary()
            )
            counter = self.temporary_flag()
            span = self.temporary_flag()
            key = self.temporary()
            self.emit(f"{gathered} = 0;", indent)
            self.emit("if (_kwnames) {", indent)
            self.emit(f"{gathered} = PyDict_New();", indent + 1)
            self.checked(gathered, indent + 1)
            self.emit(f"{span} = (int)PyObject_Size(_kwnames);", indent + 1)
            self.emit(f"for ({counter} = 0; {counter} < {span}; {counter} = {counter} + 1) {{", indent + 1)
            self.emit(f"{key} = PyTuple_GetItem(_kwnames, {counter});", indent + 2)
            self.emit(f"if (!{key}) {{ {self.failure()} }}", indent + 2)
            self.emit(
                f"PyDict_SetItem({gathered}, {key}, _args[_nargs + {counter}]);",
                indent + 2,
            )
            self.emit("}", indent + 1)
            self.emit("}", indent)
            if keyword_source:
                # `**kwargs` must exist even when nothing was passed by name.
                self.emit(f"if (!{gathered}) {{", indent)
                self.emit(f"{gathered} = PyDict_New();", indent + 1)
                self.checked(gathered, indent + 1)
                self.emit("}", indent)
            source = gathered

        if named_from and not arguments.kwarg:
            self.refuse_positional_only_by_name(
                positional[:named_from], display, indent
            )

        def from_keywords(name: str, slot: str) -> None:
            if scanning:
                # Interned on both sides, so the comparison inside
                # `PyObject_RichCompareBool` is a pointer test before it is
                # anything else.
                key = self.interned(name)
                counter = self.temporary_flag()
                matched = self.temporary_flag()
                probe = self.temporary()
                self.emit(f"if (!{slot} && {span} > 0) {{", indent)
                self.emit(
                    f"for ({counter} = 0; {counter} < {span}; "
                    f"{counter} = {counter} + 1) {{",
                    indent + 1,
                )
                self.emit(f"{probe} = PyTuple_GetItem(_kwnames, {counter});", indent + 2)
                self.emit(f"if (!{probe}) {{ {self.failure()} }}", indent + 2)
                self.emit(
                    f"{matched} = PyObject_RichCompareBool({probe}, {key}, 2);",
                    indent + 2,
                )
                self.emit(f"if ({matched} < 0) {{ {self.failure()} }}", indent + 2)
                self.emit(f"if ({matched}) {{", indent + 2)
                self.emit(f"{slot} = _args[_nargs + {counter}];", indent + 3)
                self.emit(f"Py_IncRef({slot});", indent + 3)
                self.emit(f"{taken} = {taken} + 1;", indent + 3)
                # Past sixty-three the mask has no room; the report falls back
                # to naming the first name that is not a parameter at all,
                # which is what it did before there was a mask.
                self.emit(
                    f"if ({counter} < 63) {{ {claimed} = {claimed} | "
                    f"(1LL << {counter}); }}",
                    indent + 3,
                )
                self.emit("break;", indent + 3)
                self.emit("}", indent + 2)
                self.emit("}", indent + 1)
                self.emit("}", indent)
                return
            if source == "0":
                return
            # The parameter's own name, interned once at start-up. This built
            # it from a C string on every call, for every parameter a keyword
            # could fill - an allocation and a hash per parameter per call, to
            # look up a name that never changes. Interning also makes the
            # dictionary probe a pointer comparison rather than a character
            # one.
            key = self.interned(name)
            self.emit(f"if (!{slot} && {source}) {{", indent)
            self.emit(f"{slot} = PyObject_GetItem({source}, {key});", indent + 1)
            self.emit(f"if (!{slot}) {{", indent + 1)
            self.emit("PyErr_Clear();", indent + 2)
            self.emit("} else {", indent + 1)
            # Removed whether or not there is a `**kwargs` to hand the rest
            # to. Without this nothing could tell a keyword that was taken
            # from one that matched no parameter at all, and the second kind
            # was accepted in silence: `def f(a)` called as `f(1, b=2)` ran
            # and answered where CPython raises TypeError.
            self.emit(f"PyObject_DelItem({source}, {key});", indent + 2)
            self.emit("}", indent + 1)
            self.emit("}", indent)

        for offset, name in enumerate(positional):
            slot = f"{prefix}{name}"
            # Straight out of the array the caller already had. Borrowed, so it
            # is taken over here; one that came from the keywords is owned
            # already, which is why the increment only guards this branch.
            self.emit(f"if ({offset} < _nargs) {{", indent)
            self.emit(f"{slot} = _args[{offset}];", indent + 1)
            self.emit(f"Py_IncRef({slot});", indent + 1)
            self.emit("} else {", indent)
            self.emit(f"{slot} = 0;", indent + 1)
            self.emit("}", indent)
            if offset >= named_from:
                from_keywords(name, slot)
        for offset, name in enumerate(keyword_only):
            # Never read from the tuple: that is what keyword-only means.
            from_keywords(name, f"{prefix}{name}")
        if scanning:
            # Every name should have found a parameter. When one did not, it
            # is either a keyword the function has no parameter for or one the
            # caller had already filled positionally, and CPython words those
            # two differently. Only reached when the counts disagree, so the
            # second scan costs nothing on the path that works.
            eligible = [
                self.interned(name)
                for name in (*positional[named_from:], *keyword_only)
            ]
            self.emit(f"if ({taken} != {span}) {{", indent)
            counter = self.temporary_flag()
            probe = self.temporary()
            known = self.temporary_flag()
            each = self.temporary_flag()
            self.emit(
                f"for ({counter} = 0; {counter} < {span}; "
                f"{counter} = {counter} + 1) {{",
                indent + 1,
            )
            self.emit(f"{probe} = PyTuple_GetItem(_kwnames, {counter});", indent + 2)
            self.emit(f"if (!{probe}) {{ {self.failure()} }}", indent + 2)
            # Anything already claimed is not the problem.
            self.emit(
                f"if ({counter} < 63 && (({claimed} >> {counter}) & 1LL)) "
                f"{{ continue; }}",
                indent + 2,
            )
            self.emit(f"{known} = 0;", indent + 2)
            for key in eligible:
                self.emit(
                    f"{each} = PyObject_RichCompareBool({probe}, {key}, 2);",
                    indent + 2,
                )
                self.emit(f"if ({each} < 0) {{ {self.failure()} }}", indent + 2)
                self.emit(f"if ({each}) {{ {known} = 1; }}", indent + 2)
            for verdict, message in (
                (known, f"{display}() got multiple values for argument '"),
                (f"!{known}", f"{display}() got an unexpected keyword argument '"),
            ):
                self.emit(f"if ({verdict}) {{", indent + 2)
                self.raise_counted("TypeError", message, probe, "'", indent + 3)
                self.emit("}", indent + 2)
            self.emit("}", indent + 1)
            self.emit("}", indent)
        def supply_all() -> None:
            # After the keyword check, not before it. `f(1, c=2)` on
            # `def f(a, b)` said "missing 1 required positional argument: 'b'"
            # where CPython names the keyword it could not place - both are
            # true, and CPython reports the one the caller got wrong.
            for offset, name in enumerate(positional):
                self.supply_missing(
                    node, name, f"{prefix}{name}", indent,
                    None if offset < required else defaults[offset - required],
                    missing_is_null,
                    display,
                    None if offset < required else offset - required,
                )
            if not arguments.vararg:
                # Too many is as wrong as too few, and was accepted in
                # silence: the extras simply sat unread. The wrapper needs
                # this as much as the closure does - a module-level function
                # reached *as a value* has no call site for the build-time
                # check to look at.
                self.refuse_extra_arguments(
                    display, len(positional), len(defaults), required, indent
                )
            for offset, name in enumerate(keyword_only):
                # After the positional ones, one place per keyword-only
                # parameter whether or not it has a default, so the index does
                # not move when one of them has none.
                self.supply_missing(
                    node, name, f"{prefix}{name}", indent,
                    arguments.kw_defaults[offset],
                    missing_is_null,
                    display,
                    len(defaults) + offset,
                )

        if not arguments.kwarg and source != "0":
            # Whatever is left matched no parameter, and there is no `**` to
            # hand it to. CPython names the first one, and so does this - the
            # leftovers keep the order they were passed in, because a dict
            # does.
            leftover = self.temporary_flag()
            self.emit(f"if ({source}) {{", indent)
            self.emit(f"{leftover} = (int)PyObject_Size({source});", indent + 1)
            self.emit(f"if ({leftover} > 0) {{", indent + 1)
            walker = self.temporary()
            self.emit(f"{walker} = PyObject_GetIter({source});", indent + 2)
            self.checked(walker, indent + 2)
            first = self.temporary()
            self.emit(f"{first} = PyIter_Next({walker});", indent + 2)
            self.emit(f"Py_DecRef({walker});", indent + 2)
            self.checked(first, indent + 2)
            self.raise_counted(
                "TypeError",
                f"{display}() got an unexpected keyword argument '",
                first,
                "'",
                indent + 2,
            )
            self.emit("}", indent + 1)
            self.emit("}", indent)
        supply_all()
        if arguments.vararg:
            # Everything past the named parameters, gathered out of the array.
            # There is no tuple to slice any more, so the tuple is built here -
            # which is the one place a `*args` function pays for the shape its
            # own signature asks for.
            slot = f"{prefix}{arguments.vararg.arg}"
            counter = self.temporary_flag()
            extra = self.temporary_flag()
            self.emit(f"{extra} = (int)(_nargs - {len(positional)});", indent)
            self.emit(f"if ({extra} < 0) {{ {extra} = 0; }}", indent)
            self.emit(f"{slot} = PyTuple_New((long long){extra});", indent)
            self.checked(slot, indent)
            self.emit(
                f"for ({counter} = 0; {counter} < {extra}; {counter} = {counter} + 1) {{",
                indent,
            )
            held = self.temporary()
            self.emit(f"{held} = _args[{len(positional)} + {counter}];", indent + 1)
            # PyTuple_SetItem steals, and this reference is borrowed from the
            # caller's array, so one is added for the tuple to consume.
            self.emit(f"Py_IncRef({held});", indent + 1)
            self.emit(
                f"PyTuple_SetItem({slot}, (long long){counter}, {held});", indent + 1
            )
            self.emit("}", indent)
        if closing:
            self.emit("}", indent)

    def refuse_extra_arguments(
        self, display: str, accepted: int, defaulted: int, required: int, indent: int
    ) -> None:
        """Stop when the call passed more positional arguments than there are.

        The extras used to sit unread in the tuple, so a call with the wrong
        shape ran anyway and answered - `super().__init__(1, 2)` against an
        `__init__(self, v)` gave a result where CPython raises. The wording
        follows CPython's, which says "from N to M" when some of the
        parameters have defaults and a single number when none do.
        """

        span = (
            f"from {required} to {accepted}" if defaulted else f"{accepted}"
        )
        plural = "argument" if accepted == 1 else "arguments"
        # A plain C comparison. This used to build two Python integers and
        # ask the interpreter to compare them - six calls into libpython on
        # every call a program makes, to answer a question the C argument
        # count already knows. The objects are built only to *report* the
        # failure, which almost never happens.
        self.emit(f"if (_nargs > {accepted}) {{", indent)
        given = self.temporary()
        self.emit(f"{given} = PyLong_FromLongLong(_nargs);", indent + 1)
        self.checked(given, indent + 1)
        self.raise_counted(
            "TypeError",
            f"{display}() takes {span} positional {plural} but ",
            given,
            " were given",
            indent + 1,
        )
        self.emit("}", indent)

    def supply_missing(
        self,
        node: ast.AST,
        name: str,
        slot: str,
        indent: int,
        default: ast.expr | None,
        missing_is_null: bool,
        display: str = "<lambda>",
        cell: int | None = None,
    ) -> None:
        """What to do when neither the tuple nor the keywords had it."""

        if default is not None:
            if missing_is_null:
                # The callee has this default and NULL is how it is asked for,
                # so it is filled there rather than twice.
                return
            self.emit(f"if (!{slot}) {{", indent)
            stored = self.default_cell(cell)
            if stored is not None:
                # Evaluated once, where the `def` ran. Taking a reference
                # rather than evaluating again is what makes `def f(x=[])`
                # share one list across calls, as Python does.
                self.emit(f"{slot} = {stored};", indent + 1)
                self.emit(f"if (!{slot}) {{ {self.failure()} }}", indent + 1)
                self.emit(f"Py_IncRef({slot});", indent + 1)
            else:
                value = self.expression(default, indent + 1)
                self.emit(f"{slot} = {value};", indent + 1)
            self.emit("}", indent)
            return
        self.emit(f"if (!{slot}) {{", indent)
        self.raise_type_error(
            f"{display}() missing 1 required positional argument: {name!r}",
            indent + 1,
        )
        self.emit("}", indent)

    def raise_value_error(self, message: str, count: str, indent: int) -> None:
        """A ValueError whose message ends with a number only known at runtime."""

        self.raise_counted("ValueError", message, count, ")", indent)

    def raise_counted(
        self, kind: str, before: str, count: str, after: str, indent: int
    ) -> None:
        """`before` + str(count) + `after`, raised as `kind`.

        The count is only known while the program runs - how many values were
        unpacked, how many arguments a call passed - so the message is built
        rather than written out.
        """

        message, closing = before, after
        text = self.temporary()
        self.emit(f"{text} = PyUnicode_FromString({_c_string(message)});", indent)
        self.checked(text, indent)
        spelled = self.temporary()
        self.emit(f"{spelled} = PyObject_Str({count});", indent)
        self.checked(spelled, indent)
        joined = self.temporary()
        self.emit(f"{joined} = PyNumber_Add({text}, {spelled});", indent)
        self.emit(f"Py_DecRef({text});", indent)
        self.emit(f"Py_DecRef({spelled});", indent)
        self.checked(joined, indent)
        tail = self.temporary()
        self.emit(f"{tail} = PyUnicode_FromString({_c_string(closing)});", indent)
        self.checked(tail, indent)
        whole = self.temporary()
        self.emit(f"{whole} = PyNumber_Add({joined}, {tail});", indent)
        self.emit(f"Py_DecRef({joined});", indent)
        self.emit(f"Py_DecRef({tail});", indent)
        self.checked(whole, indent)
        # Fetched raw for the same reason raise_named does: an exception class
        # whose lookup failed cannot be reported as a missing *program* name.
        kind = self.builtin_raw(kind, indent)
        self.checked(kind, indent)
        raised = self.temporary()
        self.emit(f"{raised} = PyObject_CallOneArg({kind}, {whole});", indent)
        self.emit(f"Py_DecRef({whole});", indent)
        self.checked(raised, indent)
        self.emit(f"PyErr_SetObject({kind}, {raised});", indent)
        self.emit(f"Py_DecRef({kind});", indent)
        self.emit(f"Py_DecRef({raised});", indent)
        self.emit(self.failure(), indent)

    def raise_type_error(self, message: str, indent: int) -> None:
        """Set a TypeError with this message, then take the failure path."""

        self.raise_named("TypeError", message, indent)

    def raise_named(
        self, kind_name: str, message: str, indent: int, named: str | None = None
    ) -> None:
        """Set ``kind_name(message)``, then take the failure path.

        The class comes through ``builtin_raw``: fetching it with the ordinary
        lookup would turn *its* failure into a NameError, which needs a class
        to build - the lookup that just failed.

        ``named`` sets the exception's ``name`` attribute, which is where
        CPython's display gets the material for "Did you mean: 'id'?". Without
        it the message is right and the suggestion is missing, which is a
        visible difference from the same program run under CPython.
        """

        kind = self.builtin_raw(kind_name, indent)
        self.checked(kind, indent)
        text = self.temporary()
        self.emit(f"{text} = PyUnicode_FromString({_c_string(message)});", indent)
        self.checked(text, indent)
        raised = self.temporary()
        self.emit(f"{raised} = PyObject_CallOneArg({kind}, {text});", indent)
        self.emit(f"Py_DecRef({text});", indent)
        self.checked(raised, indent)
        if named is not None:
            spelled = self.temporary()
            self.emit(
                f"{spelled} = PyUnicode_FromString({_c_string(named)});", indent
            )
            self.checked(spelled, indent)
            self.emit(
                f'PyObject_SetAttrString({raised}, "name", {spelled});', indent
            )
            self.emit(f"Py_DecRef({spelled});", indent)
        self.emit(f"PyErr_SetObject({kind}, {raised});", indent)
        self.emit(f"Py_DecRef({kind});", indent)
        self.emit(f"Py_DecRef({raised});", indent)
        self.emit(self.failure(), indent)

    def write_closure(
        self,
        node: ast.AST,
        c_name: str,
        captures: tuple[str, ...],
        display: str = "<lambda>",
        simple: str = "<lambda>",
    ) -> None:
        """Write the closure's body out as its own C function."""

        arguments = node.args
        parameters = tuple(
            argument.arg
            for argument in (*arguments.posonlyargs, *arguments.args)
        )
        keyword_only = tuple(argument.arg for argument in arguments.kwonlyargs)
        rest = arguments.vararg.arg if arguments.vararg else None
        named_rest = arguments.kwarg.arg if arguments.kwarg else None
        held = parameters + keyword_only
        if rest:
            held += (rest,)
        if named_rest:
            held += (named_rest,)
        function = _Function(
            c_name, held, len(arguments.defaults), captures, closure=True
        )
        if isinstance(node.body, list):
            given = set(held) | set(captures)
            # Ahead of both narrowings, and removed from them: holding a name
            # in a register pays for reads, and these have none.
            function.write_only = write_only_locals(node.body, given)
            if reads_every_local(node.body):
                # `locals()` has to be able to see every name, and a name in
                # a machine register is not one it could show.
                function.doubles = set()
                function.unboxed = set()
            else:
                function.doubles = (
                    double_locals(node.body, given) - function.write_only
                )
                function.unboxed = (
                    unboxable_locals(node.body, given)
                    - function.doubles
                    - function.write_only
                )
            function.exact_lists = exact_lists(node.body, given)
            function.exact_dicts = exact_dicts(node.body, given)
            function.exact_strs = exact_strs(node.body, given)
            function.body_binds = _scope_bindings(node.body)
            # The captured names belong here as much as the parameters do. A
            # name this closure took from the scope around it is *bound* -
            # whatever the module happens to call the same spelling. Without
            # them, `def d(f): def w(): return f() + 1` reached the module's
            # own `f` rather than the one handed to `d`, and since `@d` had
            # just rebound that name to `w`, the wrapper called itself until
            # the stack ran out. Every decorator whose parameter is named
            # after the function it decorates is that shape.
            function.shadows = (
                function.body_binds | set(held) | set(captures)
            )
        outer, outer_handlers, outer_scope = (
            self.current,
            self.handlers,
            self.scope,
        )
        # The comprehension scopes around the `def` are not this function's.
        # A comprehension gives its target a slot of its own and records that
        # here; a nested function that happens to use the same name has its
        # own parameter, and `reference` looks in these before it looks at
        # parameters - so `[lambda i=i: i for i in xs]` compiled a lambda
        # whose body read the comprehension's slot rather than its own `i`,
        # naming a C variable that does not exist in it. Put back below.
        outer_shadowed = self.shadowed
        self.shadowed = []
        self.current, self.handlers = function, []
        self.scope = node.body if isinstance(node.body, list) else []
        # A closure is not inside the region that encloses its definition. Its
        # `return` is its own, and leaving through the enclosing `finally`
        # would set a flag and jump to a label that belong to another C
        # function - which is how a closure defined inside a try/finally came
        # to emit `_c27 = 1; goto _finally1;` where neither exists. The same
        # for loops: a `break` here is not the outer loop's.
        outer_finallys, self.finallys = self.finallys, []
        outer_loop_depth, self.loop_depth = self.loop_depth, 0
        outer_depth, self.depth = self.depth, 0
        outer_guard, self.guards_recursion = self.guards_recursion, True
        # A closure body is not the module body, however it got here. A method
        # is written while the module's own statements are being emitted, so
        # this flag was still set inside it - which switched off both the
        # register analysis and the borrowing of locals for every method in
        # every class, the one place they matter most.
        outer_module_level, self.at_module_level = self.at_module_level, False
        self.scope_path.append((simple, True))
        # Before anything is acquired, so the failure path is a plain return
        # rather than the unwind label - nothing has been entered to leave.
        self.emit('if (Py_EnterRecursiveCall("")) { return 0; }', 1)
        # The parameters arrive in a tuple rather than as C arguments, so they
        # are locals here and are declared alongside the rest.
        for name in held:
            function.locals.append(f"p_{name}")
        for name in captures:
            function.locals.append(f"c_{name}")
        function.default_cells = [
            f"PyTuple_GetItem(_self, {len(captures) + offset})"
            for offset in range(
                len(arguments.defaults) + len(arguments.kw_defaults)
            )
        ]
        self.bind_parameters(
            node, indent=1, keyword_source=named_rest, display=display
        )
        for offset, name in enumerate(captures):
            # Borrowed from the tuple the callable holds, so it is taken over
            # for the length of the call like every other name here.
            self.emit(f"c_{name} = PyTuple_GetItem(_self, {offset});", 1)
            self.emit(f"if (!c_{name}) {{ {self.failure()} }}", 1)
            # Borrowed for the whole call, not taken: the tuple is immutable,
            # `_self` is the caller's reference to it and the caller keeps the
            # callable alive for as long as the call runs, and nothing below
            # ever rebinds a `c_` slot - the one write is this one.
        try:
            if isinstance(node.body, list):
                for statement in node.body:
                    self.statement(statement, 1)
                tail = self.builtin("None", 1)
            else:
                tail = self.expression(node.body, 1)
            self.release_locals(1)
            self.leave(tail, 1)
            self.write_unwind(function)
            self.functions.append(function)
        finally:
            # Restored even when the body is refused. Leaving the enclosing
            # scope's handler stack replaced turned the refusal into an
            # IndexError from an unrelated `try`, which said nothing about
            # what the program actually did.
            self.shadowed = outer_shadowed
            self.current, self.handlers, self.scope = (
                outer,
                outer_handlers,
                outer_scope,
            )
            self.depth = outer_depth
            self.guards_recursion = outer_guard
            self.finallys = outer_finallys
            self.loop_depth = outer_loop_depth
            self.at_module_level = outer_module_level
            self.scope_path.pop()

    # --- assembly --------------------------------------------------------

    def module(self, tree: ast.Module) -> str:
        """One module, compiled on its own. The shape a single file has."""

        self.write_module(tree, entry=True, origin=str(self.path))
        return self.render()

    def program(self, modules) -> str:
        """Several modules linked into one image.

        ``modules`` is ``(python name, tree, source path)`` for each module the entry
        imports, in the order their bodies must run, and the entry itself
        last. Each becomes a module object registered under its own name
        before its body runs, so an `import` of it finds the compiled one
        rather than reading the source beside the binary.
        """

        *imported, entry_tree = modules
        for name, tree, origin in imported:
            key = name.replace(".", "_")
            self.prefix = f"{key}_"
            self.linked.append((name, key))
            self.write_module(
                tree, entry=False, key=key, name=name, origin=origin
            )
        self.prefix = ""
        self.write_module(
            entry_tree[1], entry=True, name="__main__", origin=entry_tree[2]
        )
        return self.render()

    def write_module(
        self,
        tree: ast.Module,
        entry: bool,
        key: str = "",
        name: str = "__main__",
        origin: str = "",
    ) -> None:
        """Compile one module's functions, then its body."""

        # Per-module state. A name is a global *of its own module*, and a
        # function of one module is not callable by bare name from another.
        # Before anything else looks at the tree: a generator becomes a class
        # and a function that makes one, so nothing below ever sees a `yield`.
        try:
            # Before the generators: a `nonlocal` inside one has to become
            # a cell while it is still an ordinary function body.
            try:
                tree = expand_cells(tree)
            except CellError as error:
                raise self.fail(error.node, error.message) from None
            # Before the generator rewrite, so the value a fold produces is
            # what gets moved into the state machine, and before the narrowing
            # analyses, which read literals to decide what a name holds.
            tree = fold_constants(tree)
            # Before the narrowing analyses read it, which is the whole point:
            # they decide what can live in a register by looking at what each
            # name is assigned, and a value arriving from a call tells them
            # nothing. Folding runs first so an inlined body carries constants
            # already computed.
            #
            # Named arguments are put in their parameters' places before both,
            # because inlining and the narrowing analyses each step over a call
            # that has a keyword on it. Settled here, `f(a, step=1)` is an
            # ordinary call to everything downstream.
            # Does this module ask for its globals as a dictionary? Only
            # then do they live in one - a dictionary read per global costs
            # more than a C slot, and almost no module asks.
            self.globals_in_dict = any(
                (
                    isinstance(inner, ast.Name)
                    and isinstance(inner.ctx, ast.Load)
                    and inner.id in ("globals", "eval", "exec")
                )
                or (
                    # `from m import *` binds names nobody wrote down, so
                    # there is no set of C slots to put them in - they go in
                    # the module's dictionary, which means every global in
                    # this module does.
                    isinstance(inner, ast.ImportFrom)
                    and any(alias.name == "*" for alias in inner.names)
                )
                for inner in ast.walk(tree)
            )
            # Before everything: `except*` becomes ordinary `try`/`except`
            # calling four small functions, so nothing below this ever sees
            # a TryStar.
            if any(
                isinstance(inner, ast.ImportFrom)
                and any(alias.name == "*" for alias in inner.names)
                for inner in ast.walk(tree)
            ):
                # The function that works out what a spread binds, compiled
                # with the program because the question is about strings.
                tree.body = ast.parse(_SPREAD_HELPER).body + tree.body
                ast.fix_missing_locations(tree)
            try:
                tree = expand_except_star(tree)
            except ExceptStarError as error:
                raise self.fail(error.node, error.message) from None
            tree = place_keywords(tree)
            tree = inline_calls(tree)
            # Does the program ever read `sys.argv`? Only then is recovering
            # it worth what recovering it costs. `argv` under any spelling
            # counts - `sys.argv`, `from sys import argv`, a module the
            # program links that reads it - because guessing narrowly here
            # would leave a program with an empty argument list and no sign
            # of why.
            self.reads_argv = self.reads_argv or any(
                (isinstance(node, ast.Attribute) and node.attr == "argv")
                or (isinstance(node, ast.Name) and node.id == "argv")
                or (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "sys"
                    and any(alias.name == "argv" for alias in node.names)
                )
                for node in ast.walk(tree)
            )
            tree = expand_generators(tree)
        except GeneratorRewriteError as error:
            raise self.fail(
                error.node, f"{error.message} is not translated here yet"
            ) from None
        self.globals = set()
        self.known_functions = {}
        self.defined_at = {}
        self.reached = -1
        self.walrus_names = {
            node.target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.NamedExpr)
            and isinstance(node.target, ast.Name)
        }
        self.certain_globals = set()
        self.certain_at = {}
        self.shadowed_builtins = _shadowed_builtins(tree)
        # What `__package__` would hold. A package's `__init__` is in itself;
        # any other module is in the package above it. `__main__` is in none,
        # which is why a relative import in a script is an error in Python too.
        self.module_package = (
            name
            if origin.replace("\\", "/").endswith("__init__.py")
            else name.rpartition(".")[0]
        )

        module_bindings = _module_scope_bindings(tree)
        for position, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef):
                arguments = node.args
                if module_bindings.get(node.name, 0) != 1:
                    # Something else at module scope binds this name too - a
                    # second `def`, a `name = ...`, an import, a `for` target.
                    # Python calls whichever binding is current at the call;
                    # a direct C call would always mean this one.
                    continue
                if node.decorator_list or (
                    arguments.vararg or arguments.kwarg or arguments.kwonlyargs
                ):
                    # A C function takes a fixed number of arguments in
                    # registers and cannot express these, so the whole function
                    # is compiled as a closure instead and every call to it
                    # goes through the callable. Nothing is lost but the direct
                    # call, which needs a count known here.
                    continue
                self.known_functions[node.name] = (
                    len(arguments.posonlyargs) + len(arguments.args),
                    len(arguments.defaults),
                )
                self.known_parameters[node.name] = (
                    tuple(
                        item.arg
                        for item in (*arguments.posonlyargs, *arguments.args)
                    ),
                    len(arguments.posonlyargs),
                )
                self.defined_at[node.name] = position

        for position, node in enumerate(tree.body):
            for settled_name in self.settles(node):
                self.certain_globals.add(settled_name)
                self.certain_at.setdefault(settled_name, position)

        # What the module body binds, gathered before any function is written:
        # a function may read a global defined further down the file, exactly
        # as it may in Python, so the set has to be complete first.
        for node in tree.body:
            self.note_module_bindings(node)

        # Registered before any body is written, not when the module body
        # reaches them. A function - or a class body - written earlier that
        # reads `__name__` would otherwise not find it among the module's
        # globals and would go on to the builtins, where `__name__` also
        # exists and answers "builtins".
        for dunder in ("__name__", "__file__", "__doc__"):
            self.note_global(dunder)
            self.certain_globals.add(dunder)
            self.certain_at.setdefault(dunder, -1)
        for position, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef) and node.name in self.known_functions:
                # A body runs only after its own `def` has, so every earlier
                # `def` is certainly bound by then - and its own name is too,
                # which is what keeps recursion on the direct path.
                self.reached = position
                self.write_function(node)
                self.write_wrapper(node)

        # Not "main": that is a name a Python program may give a function of
        # its own, and the renderer skips the entry body by name. A program
        # with `def main()` had its function silently dropped and every call
        # to it left dangling at the C stage.
        body = _Function(_ENTRY_BODY if entry else f"_module_{key}", ())
        self.current = body
        self.scope = tree.body
        self.at_module_level = True
        indent = 2 if entry else 1
        # Every module has these, and a program notices when they are absent:
        # `if __name__ == "__main__":` is how a script says where it starts,
        # and `os.path.dirname(os.path.abspath(__file__))` is how it finds
        # what sits beside it.
        # The module's own docstring, or None where it has none. Left unset
        # it was looked for among the builtins, where `__doc__` exists and
        # is the builtins module's own - so `print(__doc__)` printed a page
        # about built-in functions.
        written = ast.get_docstring(tree, clean=False)
        self.certain_globals.add("__doc__")
        self.certain_at["__doc__"] = -1
        slot = self.note_global("__doc__")
        if written is None:
            blank = self.builtin("None", indent)
            self.emit(f"{slot} = {blank};", indent)
        else:
            self.emit(
                f"{slot} = PyUnicode_FromString({_c_string(written)});", indent
            )
            self.checked(slot, indent)
        self.publish("__doc__", slot, indent)
        for dunder, text in (("__name__", name), ("__file__", origin)):
            self.certain_globals.add(dunder)
            self.certain_at[dunder] = -1
            slot = self.note_global(dunder)
            if dunder == "__file__":
                # Beside the binary, wherever that is now. Baking the build
                # path meant `os.path.dirname(__file__)` named a directory on
                # the machine that compiled it, so a moved bundle looked for
                # its own files somewhere that did not exist.
                where = self.builtin("_py2bin_dir", indent)
                separator = self.temporary()
                self.emit(
                    f"{separator} = PyUnicode_FromString("
                    f"{_c_string('/' + Path(text).name)});",
                    indent,
                )
                self.checked(separator, indent)
                self.emit(f"{slot} = PyNumber_Add({where}, {separator});", indent)
                self.emit(f"Py_DecRef({where});", indent)
                self.emit(f"Py_DecRef({separator});", indent)
            else:
                self.emit(
                    f"{slot} = PyUnicode_FromString({_c_string(text)});", indent
                )
            self.checked(slot, indent)
            self.publish(dunder, slot, indent)
        for position, node in enumerate(tree.body):
            self.reached = position
            if isinstance(node, ast.FunctionDef) and node.name in self.known_functions:
                # The body was written above; what happens *here*, where the
                # `def` is, is the binding - the name starts unbound and this
                # statement is what gives it a value, exactly as in Python.
                # The defaults go first: Python evaluates them as the `def`
                # runs, once, and hands every later call the same object. They
                # used to be evaluated in the callee, so `def f(x=[])` made a
                # new list per call and the memoisation idiom quietly did
                # nothing.
                given = list(node.args.defaults) + list(node.args.kw_defaults)
                if given:
                    base = f"_py2bin_dflt_{self.prefix}{node.name}"
                    for offset, default in enumerate(given):
                        if default is None:
                            continue
                        value = self.expression(default, indent)
                        self.emit(
                            f"if ({base}_{offset}) "
                            f"Py_DecRef({base}_{offset});",
                            indent,
                        )
                        self.emit(f"{base}_{offset} = {value};", indent)
                slot = self.note_global(node.name)
                held = f"_py2bin_fn_{self.prefix}{node.name}"
                self.emit(f"Py_IncRef({held});", indent)
                self.emit(f"{slot} = {held};", indent)
                self.publish(node.name, slot, indent)
                self.show_on_module(node.name, slot, indent)
                continue
            # A `def` this could not give a fixed C shape is written here, as
            # a closure bound to its name like any other value.
            self.statement(node, indent)
        self.at_module_level = False
        if not entry:
            nothing = self.builtin("None", 1)
            self.leave(nothing, 1)
            self.write_unwind(body)
            self.module_globals[key] = set(self.globals)
        self.functions.append(body)
        self.current = None

    def note_module_bindings(self, node: ast.stmt) -> None:
        """Record every name this module-level statement binds."""

        if isinstance(node, ast.ClassDef):
            self.note_global(node.name)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # The name it binds is a module name; what its body binds is not.
            # Descending would make a function's own locals look like globals,
            # which is how `total` inside one came to be read from a file-scope
            # static that nothing ever wrote.
            self.note_global(node.name)
            return
        if isinstance(node, ast.AnnAssign):
            # `xs: list[float] = [...]` binds `xs` exactly as a plain
            # assignment does. Missing it meant a function written before the
            # module body could not see the name at all, and looked for it in
            # builtins instead.
            if node.value is not None and isinstance(node.target, ast.Name):
                self.note_global(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for element in (
                    target.elts
                    if isinstance(target, (ast.Tuple, ast.List))
                    else [target]
                ):
                    if isinstance(element, ast.Name):
                        self.note_global(element.id)
        elif isinstance(node, (ast.AugAssign, ast.For)) and isinstance(
            node.target, ast.Name
        ):
            self.note_global(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    self.note_global(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.optional_vars, ast.Name):
                    self.note_global(item.optional_vars.id)
        # A name bound inside a module-level if/for/while/try is still global.
        for field in ("body", "orelse", "finalbody"):
            for inner in getattr(node, field, []) or []:
                self.note_module_bindings(inner)
        for handler in getattr(node, "handlers", []) or []:
            for inner in handler.body:
                self.note_module_bindings(inner)

    def write_function(self, node: ast.FunctionDef) -> None:
        parameters = tuple(
            argument.arg
            for argument in (*node.args.posonlyargs, *node.args.args)
        )
        defaults = node.args.defaults
        # The C name carries the module's prefix, because two modules linked
        # into one image may each define a function of the same name.
        function = _Function(self.prefix + node.name, parameters, len(defaults))
        # Both analyses can claim a name, because eligibility is the same
        # question for each: `t = t + 1.5` uses `+`, and `+` is an integer
        # operator too. The float side wins, because it is the side holding
        # evidence - it has checked that every value the name is ever bound to
        # is a float, where the integer side has only checked that nothing
        # stops it trying. Letting the integer side win gave such a name the
        # three-variable form and then never once took the fast path, since no
        # binding was an integer: all of the cost and none of the saving.
        # Ahead of both narrowings, and removed from them: holding a name in
        # a register pays for reads, and these have none.
        function.write_only = write_only_locals(node.body, set(parameters))
        if reads_every_local(node.body):
            # See `reads_every_local`: correctness before speed, and only for
            # the functions that ask.
            function.doubles = set()
            function.unboxed = set()
        else:
            function.doubles = (
                double_locals(node.body, set(parameters)) - function.write_only
            )
            function.unboxed = (
                unboxable_locals(node.body, set(parameters))
                - function.doubles
                - function.write_only
            )
        function.exact_lists = exact_lists(node.body, set(parameters))
        function.exact_dicts = exact_dicts(node.body, set(parameters))
        function.exact_strs = exact_strs(node.body, set(parameters))
        function.body_binds = _scope_bindings(node.body)
        function.shadows = function.body_binds | set(parameters)
        # Where the defaults were put when the `def` ran, which is once - see
        # `default_cells`. A module-level `def` runs once, so they are ordinary
        # statics rather than something carried per callable.
        given = list(node.args.defaults) + list(node.args.kw_defaults)
        if given:
            base = f"_py2bin_dflt_{self.prefix}{node.name}"
            function.default_cells = [
                f"{base}_{offset}" for offset in range(len(given))
            ]
            self.default_stores.append((base, len(given)))
        self.current = function
        self.scope = node.body
        self.scope_path.append((node.name, True))
        self.guards_recursion = True
        self.emit('if (Py_EnterRecursiveCall("")) { return 0; }', 1)
        # A parameter the call left out arrives as NULL and takes the value
        # its default was evaluated to, before the increments below, so there
        # is one rule for what the body owns.
        for offset, default in enumerate(defaults):
            name = parameters[len(parameters) - len(defaults) + offset]
            self.emit(f"if (!p_{name}) {{", 1)
            self.emit(f"    p_{name} = {function.default_cells[offset]};", 1)
            self.emit(f"    if (!p_{name}) {{ {self.failure()} }}", 1)
            self.emit("}", 1)
        # The body owns its parameters, so rebinding one releases what it held
        # rather than dropping a reference the caller still owns.
        for name in parameters:
            self.emit(f"Py_IncRef(p_{name});", 1)
        for statement in node.body:
            self.statement(statement, 1)
        # Falling off the end is `return None` in Python.
        tail = self.builtin("None", 1)
        self.release_locals(1)
        self.leave(tail, 1)
        self.write_unwind(function)
        self.guards_recursion = False
        self.functions.append(function)
        self.scope_path.pop()
        self.current = None
        self.scope = []

    def write_unwind(self, function: _Function) -> None:
        """The tail a raising body leaves by: give everything back, answer NULL.

        The exception stays set, so the caller sees exactly what a failing
        C-API call looks like and its own `try` gets the chance to catch it.
        """

        if not function.unwinds:
            return
        self.current.body.append("_unwind:")
        self.release_locals(1, guarded=True)
        self.leave("0", 1)

    def render(self) -> str:
        def signature(function: _Function) -> str:
            if function.closure:
                # What CPython calls a METH_FASTCALL | METH_KEYWORDS function
                # with: the object the callable holds, the arguments in a
                # plain array, how many of them are positional, and a tuple of
                # the keyword names - or NULL when the call passed none. No
                # tuple and no dict is built to reach this.
                return (
                    "PyObject *_self, PyObject **_args, long long _nargs, "
                    "PyObject *_kwnames"
                )
            return (
                ", ".join(f"PyObject *p_{name}" for name in function.parameters)
                or "void"
            )

        out = [_PROTOTYPES]
        for function in self.functions:
            if function.name == _ENTRY_BODY:
                continue
            out.append(f"static PyObject *f_{function.name}({signature(function)});")
        # The file-scope storage comes before the bodies that read it. A
        # function placed above the declaration was reading a different slot
        # from the one main fills in, so a method that reached for a builtin
        # dereferenced whatever that slot happened to hold.
        out.append("static PyObject *_py2bin_builtins = 0;")
        if self.globals_in_dict:
            out.append("static PyObject *_py2bin_globals = 0;")
        if self.method_table:
            # Declared empty and filled at startup: the C front end does not
            # initialise a file-scope struct, and the address has to be stable
            # for as long as a callable made from it can be called.
            out.append(
                f"static struct PyMethodDef _py2bin_methods[{len(self.method_table)}];"
            )
        for name in sorted(self.declared):
            out.append(f"static PyObject *g_{name} = 0;")
        for name, _index in self.value_functions:
            # Holds the callable from start-up until its `def` binds it to the
            # module name. Only the binding is deferred; the object is not
            # remade each time the `def` is reached.
            out.append(f"static PyObject *_py2bin_fn_{name} = 0;")
        for base, count in self.default_stores:
            # One per default, filled where the `def` is and read by
            # every call that leaves that parameter out.
            for offset in range(count):
                out.append(f"static PyObject *{base}_{offset} = 0;")
        for _name, key in self.linked:
            out.append(f"static PyObject *m_{key} = 0;")
        for name, slot in self.cached_builtins.items():
            out.append(f"static PyObject *{slot} = 0;  /* {name} */")
        for text, slot in self.interned_names.items():
            out.append(f"static PyObject *{slot} = 0;  /* {text!r} */")
        for names, slot in self.keyword_tuples.items():
            out.append(f"static PyObject *{slot} = 0;  /* {names!r:.40} */")
        for value, slot in self.pooled.values():
            out.append(f"static PyObject *{slot} = 0;  /* {value!r:.40} */")
        if self.crash_log:
            out.append(_CRASH_REPORT)
        if self.needs_unbound:
            out.append(_UNBOUND_HELPER)
        out.append("")
        for function in self.functions:
            if function.name == _ENTRY_BODY:
                continue
            out.append(f"static PyObject *f_{function.name}({signature(function)}) {{")
            out.extend(self.declarations(function, 1))
            out.extend(function.body)
            out.append("}")
            out.append("")
        entry = self.functions[-1]
        out.append("int main(void) {")
        out.append("    Py_Initialize();")
        out.append('    _py2bin_builtins = PyImport_ImportModule("builtins");')
        if self.globals_in_dict:
            # The running module's own dictionary, not a new one: what
            # `globals()` answers with has to *be* the program's globals.
            out.append(
                '    _py2bin_globals = PyObject_GetAttrString('
                'PyImport_AddModule("__main__"), "__dict__");'
            )
            out.append(
                "    if (!_py2bin_globals) { PyErr_Print(); exit(1); }"
            )
        out.append(
            "    if (!_py2bin_builtins) { PyErr_Print(); exit(1); }"
        )
        # An embedded interpreter picks its own stdout encoding, and it is not
        # always UTF-8; a program that prints text outside ASCII would stop
        # with a UnicodeEncodeError that has nothing to do with the program.
        # `del` on the end for the same reason the anchor below runs inside a
        # function: this is py2bin's doing, not the program's, and `globals()`
        # would otherwise list a `sys` nobody imported.
        setup = (
            "import sys\n"
            "sys.stdout.reconfigure(encoding='utf-8')\n"
            "del sys\n"
        )
        out.append(f"    PyRun_SimpleString({_c_string(setup)});")
        # Where this binary is, right now, rather than where it was built. An
        # embedded interpreter resolves sys.executable to the host program, so
        # a compiled artifact can find itself and everything shipped beside it
        # - which is what lets the bundle be moved at all.
        anchor = (
            "import sys, os, builtins\n"
            # Where this binary is, asked of the operating system rather
            # than assumed. An embedded interpreter is given no argument
            # vector, so it has nothing to locate itself from and answers
            # `sys.executable` with the installation it was *configured*
            # with - on Linux `/usr/local/bin/python3.14`, wherever the
            # program actually sits. A bundle then looked for the packages it
            # carries next to the system Python and found none of them, and
            # the program stopped on an import of something it was shipped
            # with. `/proc/self/exe` is the exact answer where it exists;
            # macOS resolves `sys.executable` to the host program already.
            "_p = ''\n"
            "try:\n"
            "    _p = os.readlink('/proc/self/exe')\n"
            "except OSError:\n"
            "    pass\n"
            "if not _p:\n"
            "    _a = sys.argv[0] if sys.argv and sys.argv[0] else ''\n"
            "    if _a and not os.path.dirname(_a):\n"
            "        import shutil\n"
            "        _a = shutil.which(_a) or _a\n"
            "    _p = os.path.realpath(_a) if _a else (sys.executable or '')\n"
            "builtins._py2bin_dir = os.path.dirname(_p) if _p else ''\n"
            # A library that reads argv[0] - to name a window, to find its own
            # resources - gets a path rather than an empty string.
            "if not sys.argv or not sys.argv[0]:\n"
            "    sys.argv = [_p or (sys.executable or '')]\n"
        )
        if self.reads_argv:
            # The arguments the process was started with, which the embedded
            # interpreter never saw: it is handed no argument vector, so
            # `sys.argv` holds one entry this compiler put there and a
            # command-line program could not read what it was asked to do.
            #
            # Recovered from the operating system rather than through the C
            # entry point, whose signature this compiler's own C front end
            # fixes at `int main(void)` - and which would still leave Windows
            # out, where the entry is passed nothing at all.
            #
            # Emitted only for a program that mentions `sys.argv`. On Linux
            # the answer costs a file read; elsewhere it costs importing
            # ctypes, and a program that never asks should not pay for it.
            anchor += (
                "def _py2bin_argv():\n"
                "    if sys.platform.startswith('linux'):\n"
                "        with open('/proc/self/cmdline', 'rb') as _f:\n"
                "            _raw = _f.read()\n"
                "        _parts = _raw.split(b'\\0')[:-1]\n"
                "        return [_x.decode('utf-8', 'surrogateescape') "
                "for _x in _parts]\n"
                "    import ctypes\n"
                "    if sys.platform == 'darwin':\n"
                "        _lib = ctypes.CDLL(None)\n"
                "        _get = _lib._NSGetArgv\n"
                "        _get.restype = ctypes.POINTER("
                "ctypes.POINTER(ctypes.c_char_p))\n"
                "        _n = ctypes.c_int.in_dll(_lib, 'NXArgc').value\n"
                "        _v = _get().contents\n"
                "        return [_v[_i].decode('utf-8', 'surrogateescape') "
                "for _i in range(_n)]\n"
                "    _line = ctypes.windll.kernel32.GetCommandLineW\n"
                "    _line.restype = ctypes.c_wchar_p\n"
                "    _count = ctypes.c_int(0)\n"
                "    _split = ctypes.windll.shell32.CommandLineToArgvW\n"
                "    _split.restype = ctypes.POINTER(ctypes.c_wchar_p)\n"
                "    _got = _split(_line(), ctypes.byref(_count))\n"
                "    return [_got[_i] for _i in range(_count.value)]\n"
                "try:\n"
                "    _found = _py2bin_argv()\n"
                "except Exception:\n"
                # Any platform that answers differently, or a stripped /proc:
                # the single entry set above is what was there before, and is
                # better than a traceback out of the program's first line.
                "    _found = None\n"
                "if _found:\n"
                "    sys.argv = _found\n"
            )
        # Run inside a function, so that what it needs to do its job -
        # `sys`, `os`, `builtins`, the couple of working names - does not stay
        # bound in the program's own module. Nothing could see the difference
        # until `globals()` started answering with the real dictionary, and
        # then a program's own `globals()` listed three names it never wrote.
        anchor = (
            "def _py2bin_boot():\n"
            + "".join(
                f"    {line}\n" for line in anchor.splitlines()
            )
            + "_py2bin_boot()\n"
            "del _py2bin_boot\n"
        )
        out.append(f"    PyRun_SimpleString({_c_string(anchor)});")
        # Before the builtins, and before any body runs: an interned name
        # is what those lookups are about to be spelled with.
        for text, slot in self.interned_names.items():
            out.append(
                f"    {slot} = PyUnicode_InternFromString({_c_string(text)});"
            )
        for names, slot in self.keyword_tuples.items():
            out.append(f"    {slot} = PyTuple_New({len(names)}LL);")
            out.append(
                f"    if (!{slot}) {{ {self._report()}; Py_Finalize(); exit(1); }}"
            )
            for position, text in enumerate(names):
                out.append(
                    f"    PyTuple_SetItem({slot}, {position}LL, "
                    f"PyUnicode_InternFromString({_c_string(text)}));"
                )
        for value, slot in self.pooled.values():
            out.append(f"    {slot} = {self._build_constant(value)};")
            out.append(
                f"    if (!{slot}) {{ {self._report()}; Py_Finalize(); exit(1); }}"
            )
        for name, slot in self.cached_builtins.items():
            out.append(
                f"    {slot} = PyObject_GetAttrString(_py2bin_builtins, "
                f"{_c_string(name)});"
            )
            out.append(
                f"    if (!{slot}) {{ {self._report()}; Py_Finalize(); exit(1); }}"
            )
        for directory in self.extra_paths:
            # In front, so a directory named at build time wins over whatever
            # the linked interpreter happens to have. A relative one is taken
            # against the binary, so a bundle carries its own packages.
            added = (
                "import sys, os, builtins\n"
                f"sys.path.insert(0, os.path.normpath(os.path.join("
                f"builtins._py2bin_dir, {directory!r})))\n"
            )
            out.append(f"    PyRun_SimpleString({_c_string(added)});")
        for index, (c_name, label, signature) in enumerate(self.method_table):
            out.append(f"    _py2bin_methods[{index}].ml_name = {_c_string(label)};")
            out.append(f"    _py2bin_methods[{index}].ml_meth = f_{c_name};")
            # METH_FASTCALL | METH_KEYWORDS (0x80 | 0x02): the arguments
            # arrive in the array the caller already had, rather than being
            # packed into a tuple for the crossing and unpacked again. Keywords
            # are taken by every compiled function, because Python lets any
            # parameter be passed by name and one that quietly ignored that
            # would answer `show(1, c=9)` with c's default.
            out.append(f"    _py2bin_methods[{index}].ml_flags = 130;")
            # The signature goes in the doc slot, in the shape CPython reads
            # `__text_signature__` out of. Without it `inspect.signature` says
            # "unsupported callable" for every compiled function, and anything
            # that introspects - pywebview binding a JS API, and much else -
            # refuses to work with them.
            out.append(
                f"    _py2bin_methods[{index}].ml_doc = {_c_string(signature)};"
            )
        for name, index in self.value_functions:
            # Made here, where a failure can still be reported cleanly, but
            # *bound* to the module name at the `def` itself. Binding it here
            # too meant a function existed before its own `def` had run, so
            # `print(later(3))` above `def later(...)` answered rather than
            # raising the NameError Python raises.
            out.append(
                f"    _py2bin_fn_{name} = "
                f"PyCFunction_New(&_py2bin_methods[{index}], 0);"
            )
            out.append(
                f"    if (!_py2bin_fn_{name}) {{ PyErr_Print(); exit(1); }}"
            )
        for name, key in self.linked:
            # Registered under its own name *before* the body runs, so an
            # import of it - even from inside its own body - finds this object
            # rather than going to look for a file beside the binary.
            out.append(
                f"    m_{key} = PyImport_AddModule({_c_string(name)});"
            )
            out.append(
                f"    if (!m_{key}) {{ {self._report()}; Py_Finalize(); exit(1); }}"
            )
            # PyImport_AddModule borrows; sys.modules owns it, and this holds
            # its own reference for as long as the program runs.
            out.append(f"    Py_IncRef(m_{key});")
        for name, key in self.linked:
            out.append(f"    if (!f__module_{key}()) {{")
            out.append("        PyErr_Print(); Py_Finalize(); exit(1);")
            out.append("    }")
            for bound in sorted(self.module_globals.get(key, ())):
                slot = f"g_{key}_{bound}"
                out.append(
                    f"    if ({slot}) PyObject_SetAttrString("
                    f"m_{key}, {_c_string(bound)}, {slot});"
                )
        out.append("    {")
        out.extend(self.declarations(entry, 2))
        out.extend(entry.body)
        out.append("    }")
        out.append("    Py_Finalize();")
        out.append("    return 0;")
        out.append("}")
        return "\n".join(out) + "\n"

    def _build_constant(self, value) -> str:
        """The C expression that makes this literal, run once at start-up."""

        if isinstance(value, bool):  # never pooled, but never guessed at either
            raise AssertionError("bool is fetched from builtins, not pooled")
        if isinstance(value, int):
            if -(1 << 63) < value < (1 << 63):
                return f"PyLong_FromLongLong({value}LL)"
            # Wider than the machine word, so it is read from its decimal text.
            return f"PyLong_FromString({_c_string(str(value))}, 0, 10)"
        if isinstance(value, float):
            return f"PyFloat_FromDouble({value!r})"
        if isinstance(value, bytes):
            return (
                f"PyBytes_FromStringAndSize({_c_bytes(value)}, {len(value)}LL)"
            )
        encoded = value.encode("utf-8", "surrogatepass")
        if b"\0" in encoded or encoded != value.encode("utf-8", "replace"):
            # A zero byte is a character in Python and an end in C, and a lone
            # surrogate is not UTF-8 at all - both go through the decoder,
            # which is told how long the text is and how to read it back.
            return (
                f"PyUnicode_DecodeUTF8({_c_bytes(encoded)}, "
                f'{len(encoded)}LL, "surrogatepass")'
            )
        return f"PyUnicode_FromString({_c_string(value)})"

    @staticmethod
    def declarations(function: _Function, depth: int) -> list[str]:
        lines = []
        pad = "    " * depth
        for name in function.locals:
            if name.startswith(("long long ", "int ", "double ")):
                lines.append(f"{pad}{name} = 0;")
            elif "[" in name:
                lines.append(f"{pad}PyObject *{name};")
            else:
                lines.append(f"{pad}PyObject *{name} = 0;")
        return lines


def local_modules(entry: Path) -> list[tuple[str, Path]]:
    """The program's own modules, in the order their bodies must run.

    A module is "the program's own" when a `.py` of that name sits beside the
    entry. Everything else - the standard library, installed packages - is left
    to the interpreter, which is the whole point of this tier. Depth first, so
    a module is listed after anything it imports.
    """

    root = entry.parent
    ordered: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def visit(path: Path) -> None:
        for name in sorted(imported_names(path)):
            candidate = root / f"{name}.py"
            if name in seen or not candidate.exists() or candidate == entry:
                continue
            seen.add(name)
            # Its own imports first: a module's body may use what it imported.
            visit(candidate)
            ordered.append((name, candidate))

    visit(entry)
    return ordered


def imported_names(path: Path) -> set[str]:
    """The top-level module names this file imports."""

    found: set[str] = set()
    # Named, so a syntax error in the program says which file and not
    # "<unknown>", which is what `ast.parse` calls a source with no filename.
    for node in ast.walk(
        ast.parse(path.read_text(encoding="utf-8"), str(path))
    ):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module.split(".")[0])
    return found


def python_program_to_capi_c(
    entry: Path, extra_paths: tuple[str, ...] = (), crash_log: bool = False
) -> tuple[str, list[str]]:
    """Translate a program - the entry and the modules beside it - into one C.

    Compiling only the entry left its own imports to be read as source beside
    the binary, so most of a multi-file program was not compiled at all.
    """

    modules = local_modules(entry)
    emitter = CApiEmitter(entry)
    emitter.extra_paths = list(extra_paths)
    emitter.crash_log = crash_log
    trees = [
        (name, ast.parse(path.read_text(encoding="utf-8")), str(path))
        for name, path in modules
    ]
    trees.append(
        (entry.stem, ast.parse(entry.read_text(encoding="utf-8")), str(entry))
    )
    return emitter.program(trees), [name for name, _ in modules]


def python_to_capi_c(source: str, path: Path | str = "<string>") -> str:
    """Translate a Python module into C that drives the CPython C API."""

    return CApiEmitter(Path(path)).module(ast.parse(source))
