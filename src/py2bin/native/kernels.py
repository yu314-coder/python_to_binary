"""Pure-Python static-shape numerical kernels for direct native lowering.

These helpers do not import NumPy or Torch. They describe a deliberately small
integer tensor algebra in terms of py2bin IR expressions. The frontend may use
that algebra only under the explicit experimental-kernel option.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import (
    IntBinary,
    IntCompare,
    IntConstant,
    IntExpression,
    IntUnary,
)


MAX_STATIC_TENSOR_ELEMENTS = 4_096


@dataclass(frozen=True, slots=True)
class StaticI64Tensor:
    """One rank-1, compile-time-shape tensor containing runtime i64 values."""

    elements: tuple[IntExpression, ...]
    origin: str

    def __post_init__(self) -> None:
        if not self.elements:
            raise ValueError("static native tensors cannot be empty")
        if len(self.elements) > MAX_STATIC_TENSOR_ELEMENTS:
            raise ValueError(
                f"static native tensor exceeds {MAX_STATIC_TENSOR_ELEMENTS} elements"
            )

    @property
    def shape(self) -> tuple[int]:
        return (len(self.elements),)


KernelValue = IntExpression | StaticI64Tensor


def _truth_mask(condition: IntExpression) -> IntExpression:
    truth = IntUnary("not", IntUnary("not", condition))
    return IntUnary("neg", truth)


def select(
    condition: IntExpression,
    body: IntExpression,
    alternative: IntExpression,
) -> IntExpression:
    mask = _truth_mask(condition)
    return IntBinary(
        "or",
        IntBinary("and", body, mask),
        IntBinary("and", alternative, IntUnary("invert", mask)),
    )


def _broadcast(
    left: KernelValue,
    right: KernelValue,
) -> tuple[tuple[IntExpression, ...], tuple[IntExpression, ...], str]:
    if isinstance(left, StaticI64Tensor) and isinstance(right, StaticI64Tensor):
        if left.shape != right.shape:
            raise ValueError(
                f"static tensor shape mismatch: {left.shape} versus {right.shape}"
            )
        return left.elements, right.elements, f"{left.origin}+{right.origin}"
    if isinstance(left, StaticI64Tensor):
        return (
            left.elements,
            tuple(right for _ in left.elements),
            left.origin,
        )
    if isinstance(right, StaticI64Tensor):
        return (
            tuple(left for _ in right.elements),
            right.elements,
            right.origin,
        )
    return (left,), (right,), "scalar"


def binary(operator: str, left: KernelValue, right: KernelValue) -> KernelValue:
    left_values, right_values, origin = _broadcast(left, right)
    elements: list[IntExpression] = []
    for left_value, right_value in zip(left_values, right_values):
        if operator in {"add", "sub", "mul", "and", "or", "xor"}:
            elements.append(IntBinary(operator, left_value, right_value))
        elif operator == "maximum":
            elements.append(
                select(
                    IntCompare("ge", left_value, right_value),
                    left_value,
                    right_value,
                )
            )
        elif operator == "minimum":
            elements.append(
                select(
                    IntCompare("le", left_value, right_value),
                    left_value,
                    right_value,
                )
            )
        else:
            raise ValueError(f"unknown static tensor binary kernel {operator!r}")
    if not isinstance(left, StaticI64Tensor) and not isinstance(right, StaticI64Tensor):
        return elements[0]
    return StaticI64Tensor(tuple(elements), origin)


def relu(value: KernelValue) -> KernelValue:
    zero = IntConstant(0)
    if isinstance(value, StaticI64Tensor):
        return StaticI64Tensor(
            tuple(
                select(IntCompare("gt", element, zero), element, zero)
                for element in value.elements
            ),
            value.origin,
        )
    return select(IntCompare("gt", value, zero), value, zero)


def _balanced_reduce(
    operator: str,
    values: tuple[IntExpression, ...],
) -> IntExpression:
    current = list(values)
    while len(current) > 1:
        following: list[IntExpression] = []
        for index in range(0, len(current), 2):
            if index + 1 == len(current):
                following.append(current[index])
                continue
            left = current[index]
            right = current[index + 1]
            if operator in {"add", "mul"}:
                following.append(IntBinary(operator, left, right))
            else:
                combined = binary(operator, left, right)
                assert not isinstance(combined, StaticI64Tensor)
                following.append(combined)
        current = following
    return current[0]


def reduce_tensor(operator: str, tensor: StaticI64Tensor) -> IntExpression:
    if operator == "sum":
        return _balanced_reduce("add", tensor.elements)
    elif operator == "prod":
        return _balanced_reduce("mul", tensor.elements)
    elif operator in {"maximum", "minimum"}:
        return _balanced_reduce(operator, tensor.elements)
    else:
        raise ValueError(f"unknown static tensor reduction {operator!r}")


def dot(left: StaticI64Tensor, right: StaticI64Tensor) -> IntExpression:
    if left.shape != right.shape:
        raise ValueError(
            f"static tensor dot shape mismatch: {left.shape} versus {right.shape}"
        )
    products = tuple(
        IntBinary("mul", left_value, right_value)
        for left_value, right_value in zip(left.elements, right.elements)
    )
    return _balanced_reduce("add", products)
