"""The import table of a generated PE has to survive the loader reading it.

Nothing here runs a Windows binary - it cannot, on the machines this is
developed on. It reads the image the way the loader does instead, which is
enough to catch the failure that shipped: every RVA inside the import
directory was computed against a data-section address of 0x2000, chosen back
when the code fitted in one page. Once the code outgrew a page the section
moved and those RVAs pointed into the middle of `.text`, so the loader read
machine code as a DLL name and refused the image. The program never started,
printed nothing, and exited with a status that looked like a crash.
"""

from __future__ import annotations

import struct
import unittest

from py2bin.native.formats import pe
from py2bin.native.arm64 import _mov
from py2bin.native.ir import Module, Write


def _sections(image: bytes) -> list[tuple[str, int, int, int]]:
    header = struct.unpack_from("<I", image, 0x3C)[0]
    count = struct.unpack_from("<H", image, header + 6)[0]
    optional_size = struct.unpack_from("<H", image, header + 20)[0]
    table = header + 24 + optional_size
    found = []
    for index in range(count):
        raw = image[table + index * 40: table + index * 40 + 40]
        name = raw[:8].rstrip(b"\0").decode()
        virtual_size, address, raw_size, raw_offset = struct.unpack_from("<IIII", raw, 8)
        found.append((name, address, max(virtual_size, raw_size), raw_offset))
    return found


def _locate(sections, rva: int) -> tuple[str | None, int | None]:
    for name, address, span, offset in sections:
        if address <= rva < address + span:
            return name, offset + (rva - address)
    return None, None


def _walk_imports(image: bytes) -> list[str]:
    """Resolve the import directory, raising if any RVA leaves the data section."""

    header = struct.unpack_from("<I", image, 0x3C)[0]
    optional = header + 24
    sections = _sections(image)
    directory = struct.unpack_from("<I", image, optional + 112 + 8)[0]
    section_name, cursor = _locate(sections, directory)
    if cursor is None:
        raise AssertionError(f"import directory rva {directory:#x} is in no section")
    symbols: list[str] = []
    while True:
        lookup, _, _, name_rva, address_table = struct.unpack_from("<IIIII", image, cursor)
        if not (lookup or name_rva or address_table):
            break
        where, offset = _locate(sections, name_rva)
        if where != section_name:
            raise AssertionError(
                f"DLL-name rva {name_rva:#x} lands in {where}, not {section_name}"
            )
        library = image[offset: image.index(b"\0", offset)].decode("ascii")
        if not library.lower().endswith(".dll"):
            raise AssertionError(f"DLL name reads as {library!r}")
        for label, thunk in (("lookup", lookup), ("address", address_table)):
            where, position = _locate(sections, thunk)
            if where != section_name:
                raise AssertionError(
                    f"{library} {label} table rva {thunk:#x} lands in {where}"
                )
            index = 0
            while True:
                entry = struct.unpack_from("<Q", image, position + index * 8)[0]
                if entry == 0:
                    break
                if not entry >> 63:  # by name rather than by ordinal
                    where, at = _locate(sections, entry & 0xFFFFFFFF)
                    if where != section_name:
                        raise AssertionError(
                            f"{library} {label}[{index}] name rva {entry:#x} "
                            f"lands in {where}"
                        )
                    if label == "lookup":
                        symbols.append(
                            image[at + 2: image.index(b"\0", at + 2)].decode("ascii")
                        )
                index += 1
        cursor += 20
    return symbols


def _module_larger_than_a_page() -> Module:
    """Enough writes that the code cannot fit in the first page after the header."""

    return Module(
        [Write(data=b"x" * 64 + str(index).encode()) for index in range(400)]
    )


class PortableExecutableImports(unittest.TestCase):
    def test_imports_resolve_when_the_code_outgrows_a_page(self) -> None:
        module = _module_larger_than_a_page()
        for label, writer in (
            ("x86-64", pe.write_pe_x86_64),
            ("arm64", pe.write_pe_arm64),
        ):
            with self.subTest(machine=label):
                image = writer(module)
                text = next(s for s in _sections(image) if s[0] == ".text")
                self.assertGreater(
                    text[2],
                    0x1000,
                    "the fixture stopped being big enough to move the data section",
                )
                symbols = _walk_imports(image)
                self.assertIn("WriteFile", symbols)
                self.assertIn("ExitProcess", symbols)

    def test_imports_resolve_for_a_one_page_program(self) -> None:
        module = Module([Write(data=b"hi")])
        for writer in (pe.write_pe_x86_64, pe.write_pe_arm64):
            self.assertIn("WriteFile", _walk_imports(writer(module)))


class ShellLauncherProcessCreation(unittest.TestCase):
    """The frozen launcher has to hand its console down to the child.

    With `bInheritHandles` false the child gets none of the launcher's standard
    handles, so `frozen.exe > out.txt` writes nothing: the program fails on its
    first print and the traceback goes to the same missing handle. From outside
    that is a silent exit 1, which says nothing about the cause.
    """

    def test_child_inherits_handles_and_console(self) -> None:
        image = pe.write_pe_shell_launcher(b"cmd", machine="x86_64", windowed=False)
        at = image.find(b"\x48\xc7\x44\x24\x20")
        self.assertNotEqual(at, -1, "could not find the bInheritHandles store")
        self.assertEqual(struct.unpack_from("<I", image, at + 5)[0], 1)
        self.assertEqual(struct.unpack_from("<I", image, at + 14)[0], 0)

    def test_windowed_build_still_suppresses_the_console(self) -> None:
        image = pe.write_pe_shell_launcher(b"cmd", machine="x86_64", windowed=True)
        at = image.find(b"\x48\xc7\x44\x24\x20")
        self.assertEqual(struct.unpack_from("<I", image, at + 5)[0], 1)
        self.assertEqual(struct.unpack_from("<I", image, at + 14)[0], 0x08000000)

    def test_arm64_launcher_matches(self) -> None:
        for windowed, flags in ((False, 0), (True, 0x08000000)):
            with self.subTest(windowed=windowed):
                image = pe.write_pe_shell_launcher(
                    b"cmd", machine="arm64", windowed=windowed
                )
                inherit = b"".join(struct.pack("<I", word) for word in _mov(4, 1))
                wanted = b"".join(struct.pack("<I", word) for word in _mov(5, flags))
                self.assertIn(inherit, image)
                self.assertIn(wanted, image)


if __name__ == "__main__":
    unittest.main()
