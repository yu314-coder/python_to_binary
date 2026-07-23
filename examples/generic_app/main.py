from __future__ import annotations

import json
import sys
from pathlib import Path


message = (Path(__file__).parent / "message.txt").read_text(encoding="utf-8").strip()
argument = sys.argv[1] if len(sys.argv) > 1 else "no-argument"
print(json.dumps({"argument": argument, "message": message, "ok": True}, sort_keys=True))
