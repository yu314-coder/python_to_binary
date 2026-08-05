from __future__ import annotations

from dataclasses import dataclass

from dataclasses import replace

from .ir import (
    Exit,
    ExitValue,
    Function,
    IntConstant,
    Jump,
    JumpIfFalse,
    Label,
    Module,
    Write,
)


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    """Statistics from py2bin's dependency-free native IR optimizer."""

    before: int
    after: int
    merged_writes: int
    removed_operations: int
    settled_branches: int = 0


def _settle_branches(operations: list) -> tuple[list, int]:
    """Take out the branches whose condition is already known.

    Lowering folds a constant comparison to a constant - `8 == 0` becomes 0 -
    but the *branch* on it survived, so the machine code loaded a constant,
    compared it and jumped, on every turn of the loop. It arises constantly
    from generated code rather than from anything a person writes: `%` and `/`
    by a literal each guard themselves against a zero and a minus-one divisor
    that a literal already answers. Removing the guards from one such loop
    measured 0.23 s against 0.16 s.

    A false condition always jumps, so the test becomes the jump. A true one
    never jumps, so the test goes entirely. What follows an unconditional jump
    cannot be reached until something labels it, and is dropped - which is
    what takes the dead arm out rather than leaving it to occupy space.
    """

    settled = []
    removed = 0
    unreachable = False
    for operation in operations:
        if isinstance(operation, Label):
            unreachable = False
        elif unreachable:
            removed += 1
            continue
        if isinstance(operation, JumpIfFalse) and isinstance(
            operation.condition, IntConstant
        ):
            removed += 1
            if operation.condition.value == 0:
                settled.append(Jump(operation.target))
                unreachable = True
            continue
        settled.append(operation)
        if isinstance(operation, Jump):
            unreachable = True
    return settled, removed


def optimize(module: Module) -> tuple[Module, OptimizationReport]:
    """Return an equivalent, smaller native IR module.

    Frontend lowering already propagates constants, folds constant expressions,
    and selects only reachable branches. This pass performs target-independent
    operation optimization:

    * remove empty writes;
    * merge adjacent writes into one system call;
    * remove every operation after the first process exit;
    * insert the implicit successful exit when needed.

    The implementation is pure Python and invokes no external optimizer,
    compiler, assembler, Cython, Nuitka, or native toolchain.
    """

    optimized = []
    merged_writes = 0
    settled_branches = 0
    entry, removed_operations = _settle_branches(list(module.operations))
    settled_branches += removed_operations
    terminated = False
    has_control_flow = any(
        isinstance(operation, (Label, Jump, JumpIfFalse))
        for operation in entry
    )

    for operation in entry:
        if terminated:
            removed_operations += 1
            continue
        if isinstance(operation, Write):
            if not operation.data:
                removed_operations += 1
                continue
            if (
                optimized
                and isinstance(optimized[-1], Write)
                and optimized[-1].fd == operation.fd
            ):
                # Only merge writes going to the same file descriptor; stdout
                # and stderr must stay separate streams.
                previous = optimized[-1]
                optimized[-1] = Write(previous.data + operation.data, operation.fd)
                merged_writes += 1
                removed_operations += 1
                continue
        optimized.append(operation)
        if isinstance(operation, (Exit, ExitValue)) and not has_control_flow:
            terminated = True

    if not optimized or not isinstance(optimized[-1], (Exit, ExitValue)):
        optimized.append(Exit(0))

    # Function bodies get the branch pass too. They used to be carried through
    # untouched, and they are where a compiled program spends its time: the
    # entry point of a `compile-capi` build is a few dozen operations and every
    # loop the program runs is inside one of these.
    functions = []
    for function in module.functions:
        body, dropped = _settle_branches(list(function.operations))
        settled_branches += dropped
        functions.append(replace(function, operations=body) if dropped else function)

    report = OptimizationReport(
        before=len(module.operations),
        after=len(optimized),
        merged_writes=merged_writes,
        removed_operations=removed_operations,
        settled_branches=settled_branches,
    )
    # Callable bodies are carried through untouched: they have their own
    # control flow and their own terminator (Return), so none of the rules
    # above -- which all reason about the entry point's single exit -- apply.
    return (
        Module(optimized, module.stack_slots, functions, module.static_bytes),
        report,
    )
