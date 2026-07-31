"""The dynamically linked ELF, for the tier that drives CPython.

An image that binds the interpreter needs what the loader reads: a path to
itself in PT_INTERP, a symbol table to search, a GOT to fill, and relocations
saying which slot holds which symbol. That is the ELF counterpart of the
Mach-O __got and its bind opcodes.
"""

import struct
import tempfile
import unittest
from pathlib import Path

from py2bin.capi_emit import python_to_capi_c
from py2bin.c_native import compile_c_native
from py2bin.native.compiler import _EXTERN_CAPABLE_TARGETS

_SOURCE = "xs = [1, 2, 3]\nprint(sum(xs), len(xs))\n"


def _image() -> bytes:
    generated = python_to_capi_c(_SOURCE, "program.py")
    with tempfile.TemporaryDirectory() as directory:
        room = Path(directory)
        entry = room / "program.c"
        entry.write_text(generated)
        output = room / "program"
        compile_c_native(entry, output, target="linux-arm64", clean=True)
        return output.read_bytes()


class LinuxDynamicTests(unittest.TestCase):
    def test_the_target_may_call_out_at_all(self):
        self.assertIn("linux-arm64", _EXTERN_CAPABLE_TARGETS)

    def test_it_is_an_aarch64_elf_executable(self):
        data = _image()
        self.assertEqual(data[:4], b"\x7fELF")
        self.assertEqual(data[4], 2)  # 64-bit
        self.assertEqual(struct.unpack_from("<H", data, 16)[0], 2)   # ET_EXEC
        self.assertEqual(struct.unpack_from("<H", data, 18)[0], 0xB7)  # AArch64

    def _segments(self, data: bytes):
        phoff = struct.unpack_from("<Q", data, 32)[0]
        phentsize, phnum = struct.unpack_from("<HH", data, 54)
        for index in range(phnum):
            yield struct.unpack_from("<IIQQQQQQ", data, phoff + index * phentsize)

    def test_it_names_an_interpreter_for_the_kernel_to_run(self):
        data = _image()
        found = [s for s in self._segments(data) if s[0] == 3]  # PT_INTERP
        self.assertEqual(len(found), 1, "no PT_INTERP")
        _kind, _flags, offset, _va, _pa, filesz, _memsz, _align = found[0]
        path = data[offset:offset + filesz].rstrip(b"\0").decode()
        self.assertTrue(path.startswith("/lib"), path)
        self.assertIn("ld-linux", path)

    def test_it_carries_a_dynamic_segment(self):
        self.assertTrue(
            any(s[0] == 2 for s in self._segments(_image())), "no PT_DYNAMIC"
        )

    def test_writable_and_executable_are_separate_segments(self):
        # A loader will not map one segment both writable and executable, and
        # the statics have to be writable while the code has to run.
        loads = [s for s in self._segments(_image()) if s[0] == 1]
        self.assertGreaterEqual(len(loads), 2)
        for segment in loads:
            flags = segment[1]
            self.assertNotEqual(flags & 0b011, 0b011, "a segment is W and X")

    def test_it_asks_for_the_interpreter_library(self):
        data = _image()
        dynamic = next(s for s in self._segments(data) if s[0] == 2)
        _k, _f, offset, _va, _pa, filesz, _m, _a = dynamic
        strtab = strsz = None
        needed = []
        for step in range(0, filesz, 16):
            tag, value = struct.unpack_from("<Qq", data, offset + step)
            if tag == 5:
                strtab = value
            elif tag == 10:
                strsz = value
            elif tag == 1:
                needed.append(value)
            elif tag == 0:
                break
        self.assertIsNotNone(strtab, "no DT_STRTAB")
        # The base is subtracted back off: the table is stored by address.
        base = min(s[3] for s in self._segments(data) if s[0] == 1)
        table = data[strtab - base:strtab - base + strsz]
        names = [table[at:table.index(b"\0", at)].decode() for at in needed]
        self.assertTrue(
            any(name.startswith("libpython") for name in names),
            f"the interpreter is not among {names}",
        )


if __name__ == "__main__":
    unittest.main()
