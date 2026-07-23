"""Native compilation pipeline with no external assembler or linker."""

from .compiler import (
    NativeResult,
    compile_all,
    compile_native,
    resolve_target,
    supported_targets,
)
from .frontend import NativeCompileError

__all__ = [
    "NativeCompileError",
    "NativeResult",
    "compile_all",
    "compile_native",
    "resolve_target",
    "supported_targets",
]
