"""Runtime IEEE-754 double arithmetic compiled to real machine code.

Every value below lives in a hardware floating-point register (XMM on x86-64,
D on ARM64). No CPython runtime, soft-float library, or external compiler is
involved. Build it for any implemented target, e.g.:

    PYTHONPATH=src python3 -m py2bin compile examples/native_float.py \
        --target darwin-arm64 --output dist/native-float --clean
    ./dist/native-float; echo $?      # -> 47

The program accumulates a double in a loop, mixes an integer in (which is
widened to a double), scales and divides by constants, and finally truncates
to an integer process-exit status.
"""

total = 0.0
for step in range(1, 10):
    total = total + 1.5          # 9 * 1.5 = 13.5

count = 0
for step in range(1, 6):
    count = count + 4            # integer 20

blended = total + count          # promote the integer: 33.5
scaled = blended * 2.0           # 67.0
halved = scaled / 2.0            # 33.5

bonus = 0.0
if halved > 30.0:
    bonus = 14.0                 # comparison drives a native branch

raise SystemExit(int(halved + bonus))   # int(47.5) -> 47
