"""The bytes of the one atomic py2bin emits, against an assembler.

These were written by hand, from the manual, and one of them was wrong: the
branch at the end of the ARM64 retry loop read back five instructions rather
than three. Nothing caught it, because the loop only goes round when two
threads collide on the same word - so every test that did not collide passed,
and the encoding would have jumped into the middle of the setup code the
first time two ever did.

So the words are compared against what an assembler produces for the same
instructions. Where there is no assembler for the architecture on this
machine, the test says so and skips rather than passing on nothing.
"""

import shutil
import subprocess
import struct
import tempfile
import unittest
from pathlib import Path

from py2bin.native.arm64 import _atomic_add_words
from py2bin.native.x86_64 import _LOCK_XADD

#: What the ARM64 words are meant to say, in the order they are emitted.
_ARM64_SOURCE = """
.text
loop:
    ldaxr x2, [x0]
    add   x3, x2, x1
    stlxr w4, x3, [x0]
    cbnz  w4, loop
    mov   x0, x2
"""

#: And the x86-64 one. `lock` is the whole of what makes it atomic.
_X86_SOURCE = """
.text
    lock xadd %rax, (%rcx)
"""


def _assembled(source: str, target: str) -> "bytes | None":
    """The bytes an assembler produces, or None where there is not one."""

    if shutil.which("clang") is None:
        return None
    with tempfile.TemporaryDirectory() as directory:
        room = Path(directory)
        (room / "one.s").write_text(source)
        made = subprocess.run(
            [
                "clang", "-c", "-target", target,
                "-o", str(room / "one.o"), str(room / "one.s"),
            ],
            capture_output=True,
        )
        if made.returncode != 0:
            return None
        shown = subprocess.run(
            ["otool", "-t", str(room / "one.o")], capture_output=True, text=True
        )
        if shown.returncode != 0:
            return None
        # `otool` groups a fixed-width architecture into words and a
        # variable-width one into bytes, so a token of eight hex digits is a
        # 32-bit value and needs putting back in the order it is stored in.
        out = bytearray()
        for line in shown.stdout.splitlines()[2:]:
            for token in line.split()[1:]:
                if len(token) == 8:
                    out.extend(struct.pack("<I", int(token, 16)))
                elif len(token) == 2:
                    out.append(int(token, 16))
        return bytes(out)


class AtomicEncoding(unittest.TestCase):
    def test_the_arm64_words_are_the_instructions_they_claim(self):
        wanted = _assembled(_ARM64_SOURCE, "arm64-apple-macos")
        if wanted is None:
            self.skipTest("no assembler for arm64 on this machine")
        got = b"".join(struct.pack("<I", one) for one in _atomic_add_words())
        self.assertEqual(got.hex(), wanted[: len(got)].hex())

    def test_the_arm64_branch_goes_back_to_the_load(self):
        """The retry has to reach the `ldaxr`, and only a collision proves it.

        Read out of the encoding rather than run: the offset is nineteen bits
        of instructions, and it must be exactly minus three.
        """

        words = _atomic_add_words()
        branch = words[3]
        self.assertEqual(branch >> 24, 0x35, "not a CBNZ")
        offset = (branch >> 5) & 0x7FFFF
        if offset >= 1 << 18:
            offset -= 1 << 19
        self.assertEqual(offset, -3)
        self.assertEqual(branch & 0x1F, 4, "the flag CBNZ reads is w4")

    def test_the_arm64_pair_acquires_and_releases(self):
        """Plain `ldxr`/`stxr` order nothing, and the allocator needs them to.

        One thread maps the arena and publishes the end of it; another waits
        for that publication and then reads the base. Without the ordering
        the base it reads may still be the old one.
        """

        words = _atomic_add_words()
        self.assertEqual(words[0] & 0xFFFFFC00, 0xC85FFC00, "ldaxr, not ldxr")
        self.assertEqual(words[2] & 0xFFE0FC00, 0xC800FC00, "stlxr, not stxr")

    def test_the_x86_bytes_are_a_locked_exchange_and_add(self):
        wanted = _assembled(_X86_SOURCE, "x86_64-apple-macos")
        if wanted is None:
            self.skipTest("no assembler for x86-64 on this machine")
        self.assertEqual(_LOCK_XADD.hex(), wanted[: len(_LOCK_XADD)].hex())

    def test_the_x86_bytes_begin_with_the_lock_prefix(self):
        self.assertEqual(_LOCK_XADD[0], 0xF0)


if __name__ == "__main__":
    unittest.main()
