"""Packing a directory into one file, without packing the file into itself."""

import tempfile
import unittest
from pathlib import Path

from py2bin.onefile import _payload_files, create_onefile


class PayloadTests(unittest.TestCase):
    def test_the_destination_is_not_part_of_the_payload(self):
        # An archive written where it is being read contains itself, and grows
        # as it is read. One such run reached 217 GB before it was stopped.
        with tempfile.TemporaryDirectory() as directory:
            room = Path(directory)
            (room / "a.txt").write_bytes(b"a")
            growing = room / "out.exe"
            growing.write_bytes(b"x")
            names = [p.name for p in _payload_files(room, frozenset({growing}))]
            self.assertEqual(names, ["a.txt"])

    def test_everything_else_is_still_taken(self):
        with tempfile.TemporaryDirectory() as directory:
            room = Path(directory)
            (room / "one.txt").write_bytes(b"1")
            (room / "sub").mkdir()
            (room / "sub" / "two.txt").write_bytes(b"2")
            names = sorted(p.name for p in _payload_files(room))
            self.assertEqual(names, ["one.txt", "two.txt"])

    def test_writing_inside_the_payload_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            room = Path(directory)
            (room / "app").write_bytes(b"\x00")
            with self.assertRaises(ValueError) as caught:
                create_onefile(
                    payload_root=room,
                    output=room / "single.exe",
                    target="windows-x86_64",
                    launcher=room / "app",
                )
            self.assertIn("contains itself", str(caught.exception))

    def test_writing_deeper_inside_the_payload_is_refused_too(self):
        with tempfile.TemporaryDirectory() as directory:
            room = Path(directory)
            (room / "app").write_bytes(b"\x00")
            (room / "deep").mkdir()
            with self.assertRaises(ValueError):
                create_onefile(
                    payload_root=room,
                    output=room / "deep" / "single.exe",
                    target="windows-x86_64",
                    launcher=room / "app",
                )
