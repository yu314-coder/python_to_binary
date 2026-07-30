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
