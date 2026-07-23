"""Runtime bootstrap copied into generated artifacts; keep standard-library-only."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
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
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()

