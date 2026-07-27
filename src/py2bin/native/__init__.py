"""Native compilation pipeline with no external assembler or linker."""

from .application import (
    AOTApplicationPlan,
    AOTAttestation,
    AOTBuildResult,
    AOTPlanError,
    attest_aot_artifact,
    build_aot_application,
    plan_aot_application,
    require_aot_application,
)
from .compiler import (
    host_target,
    NativeResult,
    compile_all,
    compile_native_module,
    compile_native,
    compile_native_source,
    resolve_target,
    supported_targets,
)
from .frontend import NativeCompileError
from .ir_c import (
    IRCanonicalCError,
    emit_ir_c,
    parse_ir_c,
    roundtrip_ir_c,
)
from .library import (
    NativeFunctionAudit,
    NativeLibraryAudit,
    audit_native_library,
    require_native_library,
)

__all__ = [
    "AOTApplicationPlan",
    "AOTAttestation",
    "AOTBuildResult",
    "AOTPlanError",
    "IRCanonicalCError",
    "NativeCompileError",
    "NativeFunctionAudit",
    "NativeLibraryAudit",
    "audit_native_library",
    "attest_aot_artifact",
    "build_aot_application",
    "NativeResult",
    "compile_all",
    "compile_native_module",
    "compile_native",
    "host_target",
    "compile_native_source",
    "emit_ir_c",
    "parse_ir_c",
    "plan_aot_application",
    "resolve_target",
    "require_native_library",
    "require_aot_application",
    "roundtrip_ir_c",
    "supported_targets",
]
