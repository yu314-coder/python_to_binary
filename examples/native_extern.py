"""Adapter-ABI extern calls: the only honest py2bin "library" path.

Each name imported from ``py2bin.cabi`` is a genuine external native symbol.
Compiled for ``darwin-arm64``, these calls are resolved through real dyld
binding to ``/usr/lib/libSystem.B.dylib`` -- no C source is translated. The same
file runs under CPython (``py2bin.cabi`` calls the same libc symbols via
ctypes), so the native exit code is verifiable against ``python3``.

Build and run natively::

    PYTHONPATH=src python3 -c "from pathlib import Path; \
        from py2bin.native.compiler import compile_native; \
        compile_native(Path('examples/native_extern.py'), Path('extern.bin'), \
        'darwin-arm64', clean=True)"
    ./extern.bin; echo "exit=$?"        # -> 24
    PYTHONPATH=src python3 examples/native_extern.py; echo "exit=$?"  # -> 24
"""

from py2bin.cabi import abs, strlen

# abs(-5) = 5 (int argument in x0), strlen("hello native") = 12 (pointer
# argument to an embedded C string), plus a compile-time constant.
result = abs(-5) + strlen("hello native") + 7

raise SystemExit(result)  # 5 + 12 + 7 = 24
