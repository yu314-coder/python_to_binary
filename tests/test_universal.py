"""One macOS binary holding both Darwin slices.

The wrapper is header arithmetic, so most of this can be checked by reading
what was written. The one thing that cannot be read out of the bytes is the
alignment rule, which is why it has a test of its own: a code-signed x86-64
slice on a 4 KB boundary is killed at exec on Apple silicon, and every
inspection of the file says it is fine.
"""

from __future__ import annotations

import struct
import unittest

from py2bin.native.compiler import (
    SELECTABLE_TARGETS,
    TARGETS,
    UNIVERSAL_SLICES,
    UNIVERSAL_TARGET,
)
from py2bin.native.formats.macho import _DARWIN_ARCHITECTURES
from py2bin.native.formats.universal import (
    _SLICE_ALIGNMENT,
    read_universal,
    write_universal,
)


def _slices(count: int = 2) -> "dict[str, bytes]":
    # Distinct lengths, so a slice landing at the wrong offset is visible.
    return {
        "arm64": b"\xcf\xfa\xed\xfe" + b"A" * 5000,
        "x86_64": b"\xcf\xfa\xed\xfe" + b"X" * 9000,
    }


class FatHeader(unittest.TestCase):
    def test_a_slice_comes_back_exactly_as_it_went_in(self) -> None:
        given = _slices()
        self.assertEqual(read_universal(write_universal(given)), given)

    def test_the_header_says_what_the_loader_needs(self) -> None:
        image = write_universal(_slices())
        magic, count = struct.unpack_from(">II", image, 0)
        self.assertEqual(magic, 0xCAFEBABE)
        self.assertEqual(count, 2)
        for index, name in enumerate(UNIVERSAL_SLICES):
            cputype, cpusubtype, offset, size, align = struct.unpack_from(
                ">iiIII", image, 8 + index * 20
            )
            architecture = _DARWIN_ARCHITECTURES[name]
            self.assertEqual(cputype, architecture["cputype"])
            self.assertEqual(cpusubtype, architecture["cpusubtype"])
            self.assertEqual(image[offset: offset + size], _slices()[name])
            self.assertEqual(align, _SLICE_ALIGNMENT[name])

    def test_every_slice_starts_on_a_sixteen_kilobyte_boundary(self) -> None:
        """The rule that cost a SIGKILL to find.

        An x86-64 slice placed on its own 4 KB page - which is what `lipo`
        historically recorded, and what this writer did first - is refused at
        exec on Apple silicon, whose pages are 16 KB. Nothing about the file
        says so: `codesign` calls it valid, and the same bytes copied back out
        to a file of their own run. Apple's own universal2 builds record 2**14
        for both slices.
        """

        image = write_universal(_slices())
        _magic, count = struct.unpack_from(">II", image, 0)
        for index in range(count):
            _cpu, _sub, offset, _size, align = struct.unpack_from(
                ">iiIII", image, 8 + index * 20
            )
            self.assertEqual(align, 14, "a slice would be placed on a 4 KB page")
            self.assertEqual(offset % 0x4000, 0, f"slice {index} is not on a 16 KB page")

    def test_a_thin_image_reads_back_as_no_slices(self) -> None:
        # How a caller tells fat from thin without parsing a header itself.
        self.assertEqual(read_universal(b"\xcf\xfa\xed\xfe" + b"\0" * 64), {})
        self.assertEqual(read_universal(b""), {})

    def test_the_bytes_are_the_same_every_time(self) -> None:
        self.assertEqual(write_universal(_slices()), write_universal(_slices()))

    def test_an_empty_or_foreign_architecture_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            write_universal({})
        with self.assertRaises(ValueError):
            write_universal({"arm64": b""})
        with self.assertRaisesRegex(ValueError, "not a Darwin architecture"):
            write_universal({"riscv64": b"\xcf\xfa\xed\xfe"})


class TargetNaming(unittest.TestCase):
    def test_the_universal_is_selectable_but_is_not_a_backend(self) -> None:
        """`compile-all` iterates the backends, and this is not one of them.

        It is the two Darwin backends run in turn and their images joined, so
        a seventh artifact that is only the fifth and sixth concatenated is not
        another platform covered.
        """

        self.assertIn(UNIVERSAL_TARGET, SELECTABLE_TARGETS)
        self.assertNotIn(UNIVERSAL_TARGET, TARGETS)
        for architecture in UNIVERSAL_SLICES:
            self.assertIn(f"darwin-{architecture}", TARGETS)


if __name__ == "__main__":
    unittest.main()
