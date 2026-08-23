# py2bin

Turn Python or C into a native executable. No `gcc`, no `clang`, no assembler,
no linker, no SDK — py2bin writes the machine code and the Mach-O, PE and ELF
itself.

```sh
npx py2bin cc main.c util.c -I include -o app
npx py2bin compile-capi app.py --target darwin-universal2 --app --dmg -o App.app
npx py2bin make          # three questions, then a bundle
```

## This is a wrapper

The compiler is a Python program — that is the whole point of it, and it is
what lets a cross-build be arithmetic and file writing rather than a
toolchain. This package finds a Python 3.10 or newer and hands your arguments
to it.

So you need the Python package too:

```sh
pip install python-to-binary
```

If it is missing, this tells you the exact command for the interpreter it
found, rather than failing somewhere further in.

## What it builds

| | |
|---|---|
| **Python** | shipped beside a real interpreter (`freeze`), or compiled to machine code that drives CPython (`compile-capi`) |
| **C** | compiled by py2bin's own C compiler — several `.c` files and their headers together |
| **targets** | Linux, macOS and Windows, each on x86-64 and arm64, plus one universal macOS binary holding both |

C is compiled as a single translation unit, because there is no linker: name
every `.c` file and they are built together.

```sh
npx py2bin cc main.c util.c parser.c -I include -o app
```

py2bin's C compiler implements C itself and ships its own standard headers
(`stdio.h`, `stdlib.h`, `string.h`, `math.h`, `stdint.h` and friends). It has
no system include path, and **no C++**.

## Everything else

Source, documentation and issues:
**https://github.com/yu314-coder/python_to_binary**

MIT.
