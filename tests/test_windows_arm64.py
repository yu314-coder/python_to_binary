"""The ARM64 Windows image, for the tier that drives CPython.

Nothing here runs the result: this host is not Windows on ARM, and there is
no emulator for it either. What can be checked is that the image says what it
must - the right machine, the imports it needs, and static references that
resolve into the section holding the statics rather than through a register a
callback would find someone else's value in.
"""

import struct
import tempfile
import unittest
from pathlib import Path

from py2bin.capi_emit import python_to_capi_c
from py2bin.c_native import compile_c_native
from py2bin.native.compiler import _EXTERN_CAPABLE_TARGETS


def _image(source: str) -> bytes:
    generated = python_to_capi_c(source, "program.py")
    with tempfile.TemporaryDirectory() as directory:
        room = Path(directory)
        entry = room / "program.c"
        entry.write_text(generated)
        output = room / "program.exe"
        compile_c_native(entry, output, target="windows-arm64", clean=True)
        return output.read_bytes()


class _Pe:
    def __init__(self, data: bytes):
        self.data = data
        self.pe = struct.unpack_from("<I", data, 0x3C)[0]
        self.machine = struct.unpack_from("<H", data, self.pe + 4)[0]
        count = struct.unpack_from("<H", data, self.pe + 6)[0]
        size = struct.unpack_from("<H", data, self.pe + 20)[0]
        self.opt = self.pe + 24
        self.base = struct.unpack_from("<Q", data, self.opt + 24)[0]
        self.sections = {}
        for index in range(count):
            off = self.pe + 24 + size + index * 40
            name = data[off:off + 8].rstrip(b"\0").decode()
            self.sections[name] = struct.unpack_from("<IIII", data, off + 8)

    def offset(self, rva: int) -> int | None:
        for _vsize, vaddr, rsize, raddr in self.sections.values():
            if vaddr <= rva < vaddr + max(_vsize, rsize):
                return raddr + (rva - vaddr)
        return None

    @property
    def dlls(self) -> list[str]:
        rva = struct.unpack_from("<I", self.data, self.opt + 120)[0]
        found, cursor = [], self.offset(rva)
        while cursor is not None:
            fields = struct.unpack_from("<IIIII", self.data, cursor)
            if not any(fields):
                break
            name = self.offset(fields[3])
            found.append(
                self.data[name:self.data.index(b"\0", name)].decode()
            )
            cursor += 20
        return found


class WindowsArm64Tests(unittest.TestCase):
    SOURCE = "xs = [1, 2, 3]\nprint(sum(xs), len(xs))\n"

    def test_the_target_may_call_out_at_all(self):
        self.assertIn("windows-arm64", _EXTERN_CAPABLE_TARGETS)

    def test_the_image_names_the_arm64_machine(self):
        self.assertEqual(_Pe(_image(self.SOURCE)).machine, 0xAA64)

    def test_it_imports_the_interpreter_as_well_as_the_kernel(self):
        dlls = _Pe(_image(self.SOURCE)).dlls
        self.assertIn("KERNEL32.dll", dlls)
        self.assertTrue(
            any(name.startswith("python") for name in dlls),
            f"no interpreter among {dlls}",
        )

    def test_every_static_reference_lands_in_the_data_section(self):
        # The point of the exercise: an adrp/add pair reads the same object
        # whoever's frame called in, where a base kept in X28 does not survive
        # a callback entered from inside CPython.
        image = _image(self.SOURCE)
        pe = _Pe(image)
        tvsize, tvaddr, trsize, traddr = pe.sections[".text"]
        dvsize, dvaddr, _dr, _drs = pe.sections[".data"]
        code = image[traddr:traddr + trsize]
        pairs = landed = 0
        for index in range(0, len(code) - 4, 4):
            word = struct.unpack_from("<I", code, index)[0]
            if word & 0x9F000000 != 0x90000000:  # ADRP
                continue
            following = struct.unpack_from("<I", code, index + 4)[0]
            if following & 0xFFC00000 != 0x91000000:  # ADD immediate
                continue
            pairs += 1
            pages = (((word >> 5) & 0x7FFFF) << 2) | ((word >> 29) & 3)
            if pages & (1 << 20):
                pages -= 1 << 21
            here = pe.base + tvaddr + index
            target = (here & ~0xFFF) + (pages << 12) + ((following >> 10) & 0xFFF)
            if pe.base + dvaddr <= target < pe.base + dvaddr + dvsize:
                landed += 1
        self.assertGreater(pairs, 0, "no static references were emitted")
        self.assertEqual(landed, pairs, "a static reference left the data section")

    def test_calls_go_through_the_import_table(self):
        image = _image(self.SOURCE)
        pe = _Pe(image)
        _vsize, _vaddr, rsize, raddr = pe.sections[".text"]
        code = image[raddr:raddr + rsize]
        blr = sum(
            struct.unpack_from("<I", code, index)[0] == 0xD63F0200
            for index in range(0, len(code) - 4, 4)
        )
        self.assertGreater(blr, 0, "nothing called through the IAT")


if __name__ == "__main__":
    unittest.main()
