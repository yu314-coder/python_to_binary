from __future__ import annotations

from dataclasses import dataclass, field


# --- signed 64-bit integer expressions ---------------------------------------


@dataclass(frozen=True, slots=True)
class IntConstant:
    value: int


@dataclass(frozen=True, slots=True)
class IntLoad:
    slot: int


@dataclass(frozen=True, slots=True)
class SlotAddress:
    """The runtime address of stack slot ``slot``.

    Stack slots are ordinary 8-byte cells in the frame, so taking their address
    is what lets a C ``&x``, a local array, or a dereferenceable pointer work
    without a heap. The value is a full 64-bit address in the frame that
    ``IntLoad``/``Store`` already address by index.
    """

    slot: int


@dataclass(frozen=True, slots=True)
class GlobalAddress:
    """The runtime address of byte ``offset`` of the module's static storage.

    ``Module.static_bytes`` reserves one contiguous, zero-filled, readable and
    writable block that outlives every stack frame, which is exactly what an
    object with static storage duration -- a C file-scope variable -- needs. It
    is addressed relative to a base register the entry point establishes once,
    so a ``Function`` reaches the same object as the entry point does even
    though the two run on different frames.
    """

    offset: int


@dataclass(frozen=True, slots=True)
class IntUnary:
    operator: str
    operand: "IntExpression"


@dataclass(frozen=True, slots=True)
class IntBinary:
    """A two-operand integer operation on signed 64-bit registers.

    ``add``/``sub``/``mul``/``and``/``or``/``xor``/``lshift`` are
    signedness-agnostic bit operations. The rest are not, and each has an
    explicit signed and unsigned form so a frontend can never pick the wrong
    one by accident: ``rshift`` is arithmetic and ``urshift`` logical;
    ``sdiv``/``smod`` truncate toward zero as C requires, while ``udiv``/
    ``umod`` treat both operands as unsigned 64-bit values.
    """

    operator: str
    left: "IntExpression"
    right: "IntExpression"


@dataclass(frozen=True, slots=True)
class IntCompare:
    """``eq``/``ne`` plus the signed (``lt``/``le``/``gt``/``ge``) and unsigned
    (``ult``/``ule``/``ugt``/``uge``) orderings, producing 1 or 0."""

    operator: str
    left: "IntExpression"
    right: "IntExpression"


# --- IEEE-754 binary64 (double) expressions ----------------------------------
#
# Floating-point values live in their own expression tree so the backends can
# keep them in dedicated SIMD/FP registers (XMM0 on x86-64, D0 on ARM64). The
# only bridges between the two worlds are the explicit conversion nodes below,
# which keeps the existing integer lowering byte-for-byte unchanged.


@dataclass(frozen=True, slots=True)
class FloatConstant:
    value: float


@dataclass(frozen=True, slots=True)
class FloatLoad:
    slot: int


@dataclass(frozen=True, slots=True)
class FloatUnary:
    operator: str
    operand: "FloatExpression"


@dataclass(frozen=True, slots=True)
class FloatBinary:
    operator: str
    left: "FloatExpression"
    right: "FloatExpression"


@dataclass(frozen=True, slots=True)
class IntToFloat:
    """Widen a 64-bit integer expression to a double.

    ``signed`` picks the signed conversion; the unsigned one is a different
    instruction, and using the signed form for a value above 2**63-1 is exactly
    how ``(double)18446744073709551615ULL`` would come out negative.
    """

    value: "IntExpression"
    signed: bool = True


@dataclass(frozen=True, slots=True)
class BitsFloat:
    """Reinterpret an integer's low bits as a floating value, and widen it.

    ``size`` 8 reads the 64 bits as binary64. ``size`` 4 reads the low 32 bits
    as binary32 and widens the result to binary64, which is exact. This is what
    lets a C ``double`` or ``float`` object live in ordinary addressable memory
    and be loaded with the same integer loads every other C object uses.
    """

    value: "IntExpression"
    size: int = 8


# These consume a float but produce an integer, so they belong to the integer
# expression union even though they read the FP register file.


@dataclass(frozen=True, slots=True)
class FloatToInt:
    """Truncate a double toward zero into a 64-bit integer.

    ``signed`` selects the signed conversion. The unsigned one is needed for a
    C conversion to a 64-bit unsigned type, whose range runs past 2**63-1 where
    the signed instruction saturates instead.
    """

    value: "FloatExpression"
    signed: bool = True


@dataclass(frozen=True, slots=True)
class FloatBits:
    """The IEEE-754 bit pattern of a floating value, as an integer.

    ``size`` 8 yields the 64 bits of the binary64 value. ``size`` 4 rounds the
    value to binary32 first and yields those 32 bits, zero-extended -- which is
    both how a C ``float`` object is stored and how a value is rounded to
    ``float`` precision.
    """

    value: "FloatExpression"
    size: int = 8


@dataclass(frozen=True, slots=True)
class FloatCompare:
    operator: str
    left: "FloatExpression"
    right: "FloatExpression"


# --- runtime heap access -----------------------------------------------------
#
# Slice 2 adds a runtime bump-pointer arena (see HeapInit/HeapAlloc below).
# ``HeapLoad`` reads ``size`` bytes (1 or 8) from a runtime address expression
# and produces a signed 64-bit integer, so it joins the integer expression
# union. The address is itself an integer expression, which lets the frontend
# build element/character offsets out of the existing integer IR.


@dataclass(frozen=True, slots=True)
class HeapLoad:
    """Load ``size`` bytes (1, 2, 4 or 8) from a runtime address into an i64.

    ``signed`` selects sign extension over zero extension for the narrow
    widths, which is what makes a C ``signed char`` holding -1 read back as -1
    instead of 255. It is ignored when ``size`` is 8.
    """

    address: "IntExpression"
    size: int = 8
    signed: bool = False


# --- external (adapter-ABI) native calls -------------------------------------
#
# Slice 4 adds the ONLY honest "library" path: declaring and calling a genuine
# external native symbol resolved by the platform dynamic linker. These nodes
# do NOT translate C/C++/CUDA source; they bind to an already-compiled symbol
# (currently a vetted libSystem subset on darwin-arm64) through real dyld
# binding. ``ExternCall`` produces the callee's signed 64-bit return value in
# the integer register file, so it joins the integer expression union.
# ``CStringConstant`` materializes a pointer to a NUL-terminated byte blob so a
# constant string can be handed to a C function (e.g. ``strlen``).


@dataclass(frozen=True, slots=True)
class CStringConstant:
    """A pointer to an embedded NUL-terminated constant byte string."""

    data: bytes


@dataclass(frozen=True, slots=True)
class ExternCall:
    """Call external ``symbol`` with ``arguments`` (i64/pointer values).

    The symbol is resolved through the platform dynamic linker (dyld on Darwin)
    and bound before the entry point runs. Arguments are passed in the
    platform integer argument registers; the result is the callee's i64 return.
    ``symbol`` is the bare C name (no leading underscore); the Mach-O writer
    applies the platform's symbol decoration.
    """

    symbol: str
    arguments: tuple["IntExpression | FloatExpression", ...] = ()
    # Width and signedness of the callee's C result. AAPCS64 leaves bits 32-63
    # of the return register UNSPECIFIED for a 32-bit result, so a C ``int``
    # must be sign-extended (and ``unsigned int`` zero-extended) before it is
    # used as a signed 64-bit value. Without this, CPython's -1 error return
    # reads as 4294967295 and every ``if (rc < 0)`` check silently fails.
    # ``"f64"`` means the callee returns a double in the FP result register, so
    # the node is a float expression rather than an integer one.
    result: str = "i64"


# --- internal (same-image) calls ---------------------------------------------
#
# A ``Call`` names a ``Function`` defined in the same module and is lowered to a
# genuine machine call: the arguments go in the platform's integer argument
# registers, a link-register/return-address branch transfers control, and the
# callee runs on its OWN stack frame. That last part is the whole point -- it is
# what makes recursion expressible, which inlining never can be.


@dataclass(frozen=True, slots=True)
class Call:
    """Call module-level function ``name``; the value is its i64 return.

    ``arguments`` are i64/pointer values passed positionally. A backend that
    cannot honour the call ABI must reject this node rather than approximate
    it; there is no fallback to inlining at this level.
    """

    name: str
    arguments: tuple["IntExpression", ...] = ()


@dataclass(frozen=True, slots=True)
class FunctionAddress:
    """The runtime entry address of module-level function ``name``.

    This is what a C function designator decays to. The value is genuinely
    position-independent: a backend computes it relative to the program counter
    so the image may still be slid by the loader.
    """

    name: str


@dataclass(frozen=True, slots=True)
class IndirectCall:
    """Call whatever ``target`` evaluates to; the value is its i64 return.

    ``target`` is an ordinary integer expression holding a code address --
    normally a ``FunctionAddress`` that has been through memory. The ABI is the
    same one ``Call`` uses, which is what makes the two interchangeable at a
    call site. The target is evaluated BEFORE the arguments and pinned, so a
    target expression with a side effect happens exactly once.
    """

    target: "IntExpression"
    arguments: tuple["IntExpression", ...] = ()


IntExpression = (
    IntConstant
    | IntLoad
    | SlotAddress
    | GlobalAddress
    | FunctionAddress
    | IntUnary
    | IntBinary
    | IntCompare
    | FloatToInt
    | FloatBits
    | FloatCompare
    | HeapLoad
    | CStringConstant
    | ExternCall
    | Call
    | IndirectCall
)
FloatExpression = (
    FloatConstant
    | FloatLoad
    | FloatUnary
    | FloatBinary
    | IntToFloat
    | BitsFloat
    | ExternCall
)

#: The nodes that unconditionally produce a double. ``ExternCall`` is absent
#: because it belongs to whichever union its ``result`` names, which is why the
#: predicate below exists rather than a bare isinstance check.
_FLOAT_NODES = (
    FloatConstant,
    FloatLoad,
    FloatUnary,
    FloatBinary,
    IntToFloat,
    BitsFloat,
)


def is_float_expression(node: object) -> bool:
    """True when ``node`` yields a double rather than an integer word.

    ``ExternCall`` is the one node that can be either, so every place that has
    to choose between the integer and the floating-point path -- argument
    register allocation, ``Store`` versus ``FloatStore``, the frontend's type
    inference -- must ask this rather than test the node's class. Getting it
    wrong reads the wrong register file and produces a plausible wrong number
    with no diagnostic.
    """

    if isinstance(node, ExternCall):
        return node.result == "f64"
    return isinstance(node, _FLOAT_NODES)


# --- operations --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Write:
    """Write constant ``data`` to file descriptor ``fd`` (1 stdout, 2 stderr).

    Runtime diagnostics (for example the ``IndexError`` a failed list bounds
    check reports) use fd 2 so they never corrupt the program's stdout.
    """

    data: bytes
    fd: int = 1


@dataclass(frozen=True, slots=True)
class Store:
    slot: int
    value: IntExpression


@dataclass(frozen=True, slots=True)
class FloatStore:
    slot: int
    value: FloatExpression


@dataclass(frozen=True, slots=True)
class Label:
    name: str


@dataclass(frozen=True, slots=True)
class Jump:
    target: str


@dataclass(frozen=True, slots=True)
class JumpIfFalse:
    condition: IntExpression
    target: str


@dataclass(frozen=True, slots=True)
class Exit:
    status: int


@dataclass(frozen=True, slots=True)
class ExitValue:
    value: IntExpression


# --- runtime heap operations -------------------------------------------------
#
# The arena is a single fixed reservation obtained once at process start with
# an anonymous mmap (POSIX) or VirtualAlloc (Windows). A dedicated stack slot
# holds the bump pointer; there is no per-object reclamation. This is an honest
# arena: it never frees, which is documented, not hidden.


@dataclass(frozen=True, slots=True)
class HeapInit:
    """Reserve ``size`` bytes of RW arena and store its base in ``slot``."""

    slot: int
    size: int


@dataclass(frozen=True, slots=True)
class HeapAlloc:
    """Bump-allocate ``size`` bytes: store the old bump pointer in ``dest_slot``
    and advance the bump pointer in ``bump_slot``. ``size`` must already be a
    multiple of 8 so the arena stays 8-byte aligned."""

    dest_slot: int
    size: IntExpression
    bump_slot: int


@dataclass(frozen=True, slots=True)
class AtomicAdd:
    """Add ``value`` to the 64-bit word at ``address`` and store what was
    there before into ``dest_slot`` -- as one indivisible step.

    The only atomic operation py2bin emits, and the smallest one that lets an
    allocator be shared: a bump pointer moved with this hands two callers two
    different blocks, where a read followed by a write hands them the same
    one. Every other atomic a program might want is refused, and says so,
    rather than being approximated with this.
    """

    dest_slot: int
    address: IntExpression
    value: IntExpression


@dataclass(frozen=True, slots=True)
class HeapStore:
    """Store the low ``size`` bytes (1, 2, 4 or 8) of ``value`` at ``address``."""

    address: IntExpression
    value: IntExpression
    size: int = 8


@dataclass(frozen=True, slots=True)
class WriteRuntime:
    """Write ``length`` runtime bytes at ``address`` to ``fd``.

    ``fd`` is 1 for stdout and 2 for stderr, the same two the constant
    ``Write`` uses.
    """

    address: IntExpression
    length: IntExpression
    fd: int = 1


@dataclass(frozen=True, slots=True)
class EntryArguments:
    """Capture the process's argument count and vector into two slots.

    Only valid as the very first operation of a module: it reads what the
    kernel or the loader left behind at entry, and anything before it may have
    overwritten that.
    """

    count_slot: int
    vector_slot: int


@dataclass(frozen=True, slots=True)
class FileCall:
    """One file syscall, with its result left in ``dest_slot``.

    ``kind`` is "open", "read", "write" or "close". The arguments are in the
    order the kernel wants them, and a negative result is the negated errno,
    which the caller checks - nothing here raises on its own.
    """

    kind: str
    dest_slot: int
    arguments: tuple["IntExpression", ...]


@dataclass(frozen=True, slots=True)
class Return:
    """Return from the enclosing ``Function`` with ``value`` in the result register.

    Only legal inside a ``Function`` body. ``value`` is ``None`` for a function
    whose result is never read, in which case the result register holds an
    unspecified value and the caller must not use it.
    """

    value: IntExpression | None = None


Operation = (
    Write
    | Store
    | FloatStore
    | Label
    | Jump
    | JumpIfFalse
    | Exit
    | ExitValue
    | HeapInit
    | HeapAlloc
    | HeapStore
    | WriteRuntime
    | FileCall
    | EntryArguments
    | Return
)


@dataclass(slots=True)
class Function:
    """A callable body with its own stack frame.

    ``parameters`` incoming i64 arguments are delivered in the platform's
    integer argument registers and stored, in order, into stack slots
    ``0 .. parameters - 1`` of this function's frame before the body runs, so
    the body reads them exactly like any other local. ``stack_slots`` counts
    those parameter slots as well.
    """

    name: str
    parameters: int
    stack_slots: int
    operations: list[Operation]


@dataclass(slots=True)
class Module:
    operations: list[Operation]
    stack_slots: int = 0
    # Callable bodies referenced by ``Call``. The module's own ``operations``
    # remain the entry point; these are laid out after it in the same section.
    functions: list[Function] = field(default_factory=list)
    # Bytes of static storage the module needs. The block is established once,
    # before the first operation runs, and starts out entirely zero; every
    # reference to it is a ``GlobalAddress``. Zero means the module has none.
    static_bytes: int = 0
    # Where a symbol comes from, where the program said so. py2bin knows the
    # DLL behind every function it vets; a program calling into a component
    # somebody else shipped has to name the library itself, the way a build
    # that had a linker would name the import library. Empty for a program
    # that calls nothing but what py2bin knows.
    symbol_libraries: dict[str, str] = field(default_factory=dict)
    # Whether this program is a desktop one rather than a console one. Windows
    # decides by a field in the image: a console program is given a console
    # whether it wants one or not, and a window opening in front of an empty
    # black rectangle is how every program that gets this wrong looks.
    windowed: bool = False


#: The most stack slots one frame may hold. A frame is real OS stack, and the
#: prologue moves SP down by the whole thing at once, so a frame larger than
#: the thread's stack runs off the end of it and the program dies with a signal
#: instead of printing an answer. Nothing at build time can measure the stack
#: the program will get, so the limit is a fixed one and a frame past it is
#: refused. It is deliberately the same on every target -- the smallest stack
#: any of them hands the initial thread is the 1 MiB a PE reserves, and one
#: source file should not compile for macOS and be refused for Windows -- so
#: 512 KiB of slots leaves the rest of that reserve for whatever the body calls.
MAXIMUM_STACK_SLOTS = 65536


def check_stack_slots(stack_slots: int, owner: str) -> int:
    """Reject a frame that would not fit on the thread stack. Returns it."""

    if stack_slots > MAXIMUM_STACK_SLOTS:
        raise ValueError(
            f"{owner} needs {stack_slots} stack slots ({stack_slots * 8} bytes "
            f"of frame), past the {MAXIMUM_STACK_SLOTS}-slot "
            f"({MAXIMUM_STACK_SLOTS * 8}-byte) budget one frame may take from "
            "the thread stack; a larger frame would overrun the stack at run "
            "time rather than answer, so split the code into smaller functions"
        )
    return stack_slots
