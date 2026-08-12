"""System V wants rsp 16-byte aligned at the call, and where you start matters.

An image the kernel starts - a static Mach-O, or an ELF at its entry point -
begins 16-byte aligned. One entered through LC_MAIN does not: dyld *calls* it,
so its return address is already on the stack and rsp is 8 past alignment. A
frame that is a multiple of 16 preserves that 8 and every call made from the
entry hands the callee a misaligned stack.

That is a fault rather than a slowdown - the first `movaps` to a stack slot in
the callee raises a general-protection fault - and it is invisible on Apple
silicon, because Rosetta does not enforce the alignment. It took a crash report
from a real Intel Mac: SIGSEGV inside `_PyRuntimeState_Init`, with rbp and rsp
both 8 mod 16 where a standard prologue leaves rbp at 0.
"""

from __future__ import annotations

import struct
import unittest

from py2bin.native.ir import Module, Write
from py2bin.native.x86_64 import encode, encode_darwin_extern, encode_linux_extern


def _entry_frame(code: bytes) -> int:
    """How many bytes the first instruction takes off the stack."""

    if code[:3] == b"\x48\x83\xec":       # sub rsp, imm8
        return code[3]
    if code[:3] == b"\x48\x81\xec":       # sub rsp, imm32
        return struct.unpack_from("<I", code, 3)[0]
    if code[:3] == b"\x48\x89\xe5":       # no frame at all
        return 0
    raise AssertionError(f"unrecognised prologue: {code[:8].hex(' ')}")


def _module() -> Module:
    return Module([Write(data=b"x")], stack_slots=8)


class EntryAlignment(unittest.TestCase):
    def test_an_lc_main_image_corrects_for_its_return_address(self) -> None:
        code, _externs, _statics = encode_darwin_extern(_module(), 0x100001000)
        frame = _entry_frame(code)
        self.assertEqual(
            frame % 16,
            8,
            "entered by a call, rsp is 8 past alignment; a frame that is a "
            "multiple of 16 keeps it there and misaligns every call made",
        )

    def test_a_kernel_started_image_does_not(self) -> None:
        for label, code in (
            ("darwin static", encode(_module(), "darwin", 0x100001000)),
            ("linux static", encode(_module(), "linux", 0x401000)),
        ):
            with self.subTest(image=label):
                self.assertEqual(
                    _entry_frame(code) % 16,
                    0,
                    "the kernel starts this one with rsp already aligned",
                )

    def test_the_linux_dynamic_image_is_kernel_started_too(self) -> None:
        # An ELF is entered at its entry point by the kernel whether or not it
        # is dynamically linked, so nothing is on the stack that would not be.
        code, _externs, _statics = encode_linux_extern(_module(), 0x401000)
        self.assertEqual(_entry_frame(code) % 16, 0)


if __name__ == "__main__":
    unittest.main()
