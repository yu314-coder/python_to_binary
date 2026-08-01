"""The disk image writer: what it produces must be readable without tools."""

import struct
import tempfile
import unittest
from pathlib import Path

from py2bin.dmg import SECTOR, ImageError, write_image


class DiskImageTests(unittest.TestCase):
    def _tree(self, root: Path) -> None:
        (root / "nested" / "deeper").mkdir(parents=True)
        (root / "README.txt").write_bytes(b"top\n")
        (root / "a-very-long-file-name.json").write_bytes(b"{}\n")
        (root / "nested" / "inner.dat").write_bytes(b"inner\n")
        (root / "nested" / "deeper" / "deep-file-name.txt").write_bytes(b"deep\n")

    def test_writes_a_recognisable_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "src"
            root.mkdir()
            self._tree(root)
            image = Path(directory) / "out.dmg"
            size = write_image(root, image, "PY2BIN TEST")
            data = image.read_bytes()
            self.assertEqual(size, len(data))
            self.assertEqual(len(data) % SECTOR, 0)
            # Both descriptors identify themselves where the format says.
            primary = 16 * SECTOR
            self.assertEqual(data[primary], 1)
            self.assertEqual(data[primary + 1:primary + 6], b"CD001")
            supplementary = 17 * SECTOR
            self.assertEqual(data[supplementary], 2)
            self.assertEqual(data[supplementary + 1:supplementary + 6], b"CD001")
            # The Joliet escape sequence, which is what carries real names.
            self.assertEqual(data[supplementary + 88:supplementary + 91], b"%/E")
            terminator = 18 * SECTOR
            self.assertEqual(data[terminator], 255)

    def test_size_field_matches_the_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "src"
            root.mkdir()
            self._tree(root)
            image = Path(directory) / "out.dmg"
            write_image(root, image)
            data = image.read_bytes()
            declared = struct.unpack_from("<I", data, 16 * SECTOR + 80)[0]
            self.assertEqual(declared * SECTOR, len(data))

    def test_file_contents_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "src"
            root.mkdir()
            self._tree(root)
            image = Path(directory) / "out.dmg"
            write_image(root, image)
            data = image.read_bytes()
            for payload in (b"top\n", b"inner\n", b"deep\n"):
                self.assertIn(payload, data)

    def test_a_symlink_is_refused_rather_than_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "src"
            root.mkdir()
            (root / "real.txt").write_bytes(b"x\n")
            (root / "link.txt").symlink_to("real.txt")
            with self.assertRaises(ImageError):
                write_image(root, Path(directory) / "out.dmg")

    def test_long_names_are_kept_in_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "src"
            root.mkdir()
            name = "an-extremely-long-file-name-that-iso9660-cannot-hold.txt"
            (root / name).write_bytes(b"y\n")
            image = Path(directory) / "out.dmg"
            write_image(root, image)
            self.assertIn(name.encode("utf-16-be"), image.read_bytes())


if __name__ == "__main__":
    unittest.main()


class CompressedImageTests(unittest.TestCase):
    """The same image, deflated - what is handed over rather than what it holds.

    A bundle is mostly native code, which compresses to about two fifths.
    macOS inflates as the volume is read, so the app is unchanged and only the
    file someone downloads is smaller.
    """

    def _tree(self, root: Path) -> None:
        (root / "App").mkdir()
        (root / "App" / "text.txt").write_bytes(b"compress me " * 4000)
        (root / "App" / "other.bin").write_bytes(bytes(range(256)) * 400)

    def test_it_is_smaller_than_the_plain_image(self):
        from py2bin.dmg import write_compressed_image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "src"
            root.mkdir()
            self._tree(root)
            plain = write_image(root, Path(directory) / "a.dmg", "A")
            packed = write_compressed_image(root, Path(directory) / "b.dmg", "B")
            self.assertLess(packed, plain)

    def test_it_ends_with_the_trailer_macos_reads_first(self):
        from py2bin.dmg import write_compressed_image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "src"
            root.mkdir()
            self._tree(root)
            out = Path(directory) / "b.dmg"
            write_compressed_image(root, out, "B")
            data = out.read_bytes()
            # The trailer is the last 512 bytes and names itself.
            self.assertEqual(data[-512:-508], b"koly")

    def test_the_chunk_table_is_present_and_terminated(self):
        from py2bin.dmg import compress_image, _LAST_CHUNK, _blkx

        _packed, table = _blkx(b"\0" * (512 * 5000))
        self.assertEqual(table[:4], b"mish")
        # mish header: magic+version 8, three 8-byte fields 24, two 4-byte 8,
        # reserved 24, checksum 136 - the count sits at 200.
        count = struct.unpack_from(">I", table, 200)[0]
        self.assertGreater(count, 1)
        # The final entry says there is no more.
        last = table[-40:]
        self.assertEqual(struct.unpack_from(">I", last, 0)[0], _LAST_CHUNK)

    def test_an_empty_image_still_produces_a_valid_trailer(self):
        from py2bin.dmg import compress_image

        packed = compress_image(b"")
        self.assertEqual(packed[-512:-508], b"koly")
