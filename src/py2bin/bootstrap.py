"""Runtime bootstrap copied into generated artifacts; keep standard-library-only."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path


def _archive_root() -> Path:
    source = Path(__file__).resolve()
    if source.parent.name == "runtime" and source.parent.parent.is_dir():
        return source.parent.parent
    archive = Path(sys.argv[0]).resolve()
    stat = archive.stat()
    fingerprint = hashlib.sha256(
        f"{archive}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()[:20]
    cache_base = Path(
        os.environ.get("PY2BIN_CACHE_DIR", Path(tempfile.gettempdir()) / "py2bin")
    )
    extracted = cache_base / fingerprint
    ready = extracted / ".ready"
    if not ready.exists():
        temporary = cache_base / f".{fingerprint}-{os.getpid()}"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(temporary)
            temporary.replace(extracted)
            ready.touch()
        except FileExistsError:
            pass
        finally:
            if temporary.exists():
                import shutil

                shutil.rmtree(temporary)
    return extracted


def main() -> None:
    root = _archive_root()
    manifest = json.loads((root / "py2bin-manifest.json").read_text(encoding="utf-8"))
    app = root / "app"
    dependencies = root / "site-packages"
    sys.path[:0] = [str(app), str(dependencies)]
    entry = app / manifest["entry"]
    sys.argv[0] = str(entry)
    os.environ.setdefault("PY2BIN_BUNDLE_ROOT", str(root))
    _run_as_script(entry)


def _run_as_script(entry: Path) -> None:
    """Run the entry the way the interpreter runs a script named on its command line.

    `runpy.run_path` is the obvious thing and it is not quite that: it sets
    `__package__` to the empty string, where a script started by CPython has
    None. The difference is small and it is the kind that only shows up in
    somebody else's library - `__package__ or __name__` is a common way to
    work out where you are, and the empty string is falsy where None is too,
    but `if __package__ is None` is just as common and reads the other way.

    `__loader__` is set for the same reason: the source really is in the
    bundle, so `inspect.getsource` and anything else that reads it back can
    have the loader that would find it.
    """

    import builtins
    import importlib.machinery
    import types

    module = types.ModuleType("__main__")
    # In `__main__` this is the builtins *module*; anywhere else it is that
    # module's dictionary. `exec` puts the dictionary in when the globals have
    # none, so it has to be set before rather than after.
    module.__builtins__ = builtins
    module.__file__ = str(entry)
    module.__package__ = None
    module.__spec__ = None
    module.__loader__ = importlib.machinery.SourceFileLoader(
        "__main__", str(entry)
    )
    sys.modules["__main__"] = module
    # From bytes: `compile` reads a byte-order mark and a coding line
    # itself, where decoding as UTF-8 first would refuse a Latin-1 file.
    code = compile(entry.read_bytes(), str(entry), "exec")
    exec(code, module.__dict__)


if __name__ == "__main__":
    main()

