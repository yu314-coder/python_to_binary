#!/usr/bin/env python3
"""Build a program with the py2bin sitting next to this file.

    python3 build.py [path/to/your/program]

Nothing is installed and nothing is downloaded to get started: this runs the
`src/py2bin` in the clone it lives in. Answer three questions - which file,
which machine, what shape - and everything else is found or downloaded rather
than typed.

The questions themselves live in `py2bin.interactive`, so `py2bin make` asks
exactly the same ones for anyone who installed with pip. This file is the way
in when there is no install, only a clone.

`get-py2bin.py` is the other half: it fetches py2bin when there is no clone
either.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "src"


def main() -> int:
    if not (SOURCE / "py2bin" / "__init__.py").is_file():
        # Said on stdout: some editors show only that, and an explanation
        # nobody sees is the same as no explanation.
        print(f"No py2bin beside this script - expected {SOURCE / 'py2bin'}.")
        print("Run this from a clone of the repository, or use get-py2bin.py.")
        return 1
    sys.path.insert(0, str(SOURCE))
    from py2bin.interactive import main as ask

    return ask(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    raise SystemExit(main())
