"""Every Windows x64 call reserves the 32 bytes its callee owns.

The Microsoft x64 ABI gives the callee 32 bytes above the return address to
spill its four register arguments into, and makes the *caller* reserve them.
A call made without them lets the callee write over whatever is there - and
inside a function body what is there is that function's own first four slots,
because a function's frame here is exactly its locals.

`HeapInit` was such a call. It runs inside `malloc`, whose first slot is the
size being asked for, so `VirtualAlloc` overwrote it and the comparison two
lines later answered NULL. The first allocation in every Windows program came
back null; every one after it was fine, because the reservation happens once.
That took a run on real Windows to find, which is why it is pinned here.
"""

import struct
import tempfile
import unittest
from pathlib import Path

from py2bin.c_native import compile_c_native

#: `sub rsp, 32` and `sub rsp, 48` - the two reservations made here. Forty
#: eight where the call takes a fifth argument and writes a count back above
#: the shadow area, rounded up so rsp stays aligned to sixteen.
_SHADOW = (b"\x48\x83\xec\x20", b"\x48\x83\xec\x30")

#: `call [rip + disp32]` - how every import is called.
_INDIRECT = b"\xff\x15"


def _text_of(source: str) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        room = Path(directory)
        entry = room / "program.c"
        entry.write_text(source)
        output = room / "program.exe"
        compile_c_native(entry, output, target="windows-x86_64", clean=True)
        data = output.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    count = struct.unpack_from("<H", data, pe + 6)[0]
    size = struct.unpack_from("<H", data, pe + 20)[0]
    for index in range(count):
        off = pe + 24 + size + index * 40
        if data[off:off + 8].rstrip(b"\0") == b".text":
            _vsz, _va, raw_size, raw = struct.unpack_from("<IIII", data, off + 8)
            return data[raw:raw + raw_size]
    raise AssertionError("no .text section")


class ShadowSpace(unittest.TestCase):
    def calls_are_all_covered(self, source: str) -> None:
        text = _text_of(source)
        at = 0
        seen = 0
        while True:
            at = text.find(_INDIRECT, at)
            if at < 0:
                break
            # Somewhere in the sixteen bytes before the call, the reservation.
            # A window rather than the byte immediately before, because the
            # argument registers are loaded between the two.
            window = text[max(0, at - 24):at]
            self.assertTrue(
                any(one in window for one in _SHADOW),
                f"the call at 0x{at:x} reserves no shadow space: "
                f"{window.hex()}",
            )
            seen += 1
            at += 2
        self.assertGreater(seen, 0, "no indirect calls in the image at all")

    def test_the_call_that_reserves_the_heap_reserves_shadow_space(self):
        # `malloc` is what reaches `__py2bin_arena()`, and the arena is
        # reserved by a call to VirtualAlloc inside a function body.
        self.calls_are_all_covered(
            "#include <stdlib.h>\n"
            "int main(void) {\n"
            "    void *first = malloc(64);\n"
            "    return first == 0 ? 1 : 0;\n"
            "}\n"
        )

    def test_the_first_allocation_is_not_the_one_that_is_lost(self):
        # The shape of the bug: only the first `malloc` was wrong, because
        # the reservation happens once. A program that asks twice and
        # compares told them apart.
        self.calls_are_all_covered(
            "#include <stdlib.h>\n"
            "int main(void) {\n"
            "    void *first = malloc(64);\n"
            "    void *second = malloc(64);\n"
            "    if (first == 0) { return 1; }\n"
            "    if (second == 0) { return 2; }\n"
            "    return 0;\n"
            "}\n"
        )

    def test_writing_and_exiting_reserve_it_too(self):
        self.calls_are_all_covered(
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            "int main(void) {\n"
            "    printf(\"hello\\n\");\n"
            "    exit(3);\n"
            "}\n"
        )


if __name__ == "__main__":
    unittest.main()
