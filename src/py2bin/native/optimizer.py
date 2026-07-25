from __future__ import annotations

from dataclasses import dataclass

from .ir import Exit, ExitValue, Jump, JumpIfFalse, Label, Module, Write


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    """Statistics from py2bin's dependency-free native IR optimizer."""

    before: int
    after: int
    merged_writes: int
    removed_operations: int


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
    removed_operations = 0
    terminated = False
    has_control_flow = any(
        isinstance(operation, (Label, Jump, JumpIfFalse))
        for operation in module.operations
    )

    for operation in module.operations:
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

    report = OptimizationReport(
        before=len(module.operations),
        after=len(optimized),
        merged_writes=merged_writes,
        removed_operations=removed_operations,
    )
    # Callable bodies are carried through untouched: they have their own
    # control flow and their own terminator (Return), so none of the rules
    # above -- which all reason about the entry point's single exit -- apply.
    return (
        Module(optimized, module.stack_slots, module.functions, module.static_bytes),
        report,
    )
